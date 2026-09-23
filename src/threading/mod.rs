//! Multi-threaded search: root-level splitting.
//!
//! With `Threads = 1` (the default) the whole search runs inline on the
//! calling thread and is fully deterministic — TT and ordering tables are
//! pure functions of insertion order, and the node counter is a fixed
//! function of the traversal.
//!
//! With more threads the (already ordered) root move list is split
//! round-robin across worker threads. Each worker owns its own ordering
//! tables and search stacks, and shares the lock-free TT, the stop signal and
//! the node counter through [`crate::search::SearchShared`]. Workers run
//! independent iterative deepeners over their subset; the worker reporting
//! the best score provides the result. This yields real parallel speedup on
//! the root (the classic "root splitting" scheme used by Fruit/Cuckoo-style
//! engines) while keeping the code base small and race-free.
//!
//! # Stopping
//!
//! The shared `stop` flag halts every worker: each node marks itself into the
//! shared counter and re-checks the flag (plus the hard deadline and node
//! cap) every 1024 nodes; once set, no further TT stores happen.

use std::sync::atomic::Ordering;
use std::time::Instant;

use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::move_ordering::OrderingTables;
use crate::move_ordering::order_moves_ctx;
use crate::search::iterative::iterative_search;
use crate::search::{
    SearchResult, SearchShared, SearchStats, SearchThread, ThreadResult, TimeLimit,
};
use crate::types::RawMove;

/// The number of threads used when the caller passes `threads == 0`.
pub const MIN_THREADS: usize = 1;

/// Runs a full search. The shared state must belong to exactly one search;
/// `root` and `history` are only read, so sharing them across workers is
/// safe.
pub fn run_search(
    shared: &SearchShared,
    root: &Position,
    history: &[Zobrist64],
    limits: &TimeLimit,
    threads: usize,
    searchmoves: &[RawMove],
) -> SearchResult {
    let start = Instant::now();
    let threads = threads.max(MIN_THREADS);

    // One shared, deterministic ordering of the root moves (used both for the
    // single-thread path and as the baseline order each worker rotates).
    let mut root_moves = root.legal_moves();
    if !searchmoves.is_empty() {
        let allowed: Vec<RawMove> = searchmoves.to_vec();
        let mut filtered = crate::types::MoveList::new();
        for m in root.legal_moves().iter() {
            if allowed.contains(&m) {
                filtered.push(m);
            }
        }
        root_moves = filtered;
    }
    let tt_move = shared
        .tt
        .probe_move(root.hash.into())
        .unwrap_or(RawMove::NULL);
    let tables = OrderingTables::new();
    order_moves_ctx(&mut root_moves, root, &tables, tt_move, 0, None, None);
    let all: Vec<RawMove> = root_moves.iter().collect();
    drop(root_moves);
    drop(tables);

    if threads == 1 {
        let mut thread = SearchThread::new();
        thread.deadline = limits.hard_deadline();
        let res = iterative_search(&mut thread, shared, root, history, limits, &all);
        let stats = res.stats;
        finish(res, stats, shared, start, threads)
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let subset: Vec<RawMove> = all
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(i, _)| i % threads == t)
                        .map(|(_, m)| m)
                        .collect();
                    scope.spawn(move || {
                        let mut thread = SearchThread::new();
                        thread.deadline = limits.hard_deadline();
                        iterative_search(&mut thread, shared, root, history, limits, &subset)
                    })
                })
                .collect();

            let mut results: Vec<ThreadResult> =
                handles.into_iter().map(|h| h.join().unwrap()).collect();
            // The worker with the best score wins; tie-break by depth.
            results.sort_by(|a, b| b.score.cmp(&a.score).then(b.depth.cmp(&a.depth)));
            let best = results.remove(0);
            let mut stats = SearchStats::default();
            for r in results.iter().chain(std::iter::once(&best)) {
                stats.add(&r.stats);
            }
            finish(best, stats, shared, start, threads)
        })
    }
}

/// Wraps a worker result into the public [`SearchResult`], harvesting the
/// shared node count, the aggregated per-thread counters and wall time.
fn finish(
    result: ThreadResult,
    stats: SearchStats,
    shared: &SearchShared,
    start: Instant,
    threads: usize,
) -> SearchResult {
    SearchResult {
        best: result.best,
        score: result.score,
        depth: result.depth,
        nodes: shared.nodes.load(Ordering::Relaxed),
        time_ms: start.elapsed().as_millis(),
        threads,
        pv: result.pv,
        stopped: shared.stop.load(Ordering::Relaxed),
        stats,
        tt_stores: shared.tt.stores(),
    }
}

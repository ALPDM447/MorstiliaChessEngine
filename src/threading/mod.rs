//! Lazy-SMP parallel search: a persistent pool of helper workers that each
//! search the *entire* root move list, coordinated only through the shared
//! transposition table.
//!
//! With `Threads = 1` (the default) the whole search runs inline on the
//! calling thread and is fully deterministic — TT and ordering tables are
//! pure functions of insertion order, and the reported node count is the
//! worker's exact local counter.
//!
//! With more threads the caller's thread becomes the **main search thread**
//! and runs the same iterative deepening; `Threads - 1` persistent helpers
//! run it in parallel. There is no work stealing and no locking in the node
//! loop: helpers simply search the same root set, and whatever they establish
//! (scores, bounds, best moves) lands in the shared lock-free TT, which the
//! main thread's next iteration probes. This is exactly the classic Lazy-SMP
//! scheme: simple, race-free, deterministic at `Threads = 1`.
//!
//! # Worker lifecycle
//!
//! Helpers are spawned once (lazily, at the first multi-threaded search or a
//! `setoption Threads`) and park on a condvar between searches, keeping their
//! per-worker ordering tables warm. `set_helpers` grows/shrinks the pool;
//! dropping the pool (or the owning [`crate::search::Searcher`]) shuts every
//! worker down and joins it.
//!
//! # Stopping
//!
//! The shared `stop` flag halts every worker: each node marks itself into the
//! shared counter and re-checks the flag (plus the hard deadline and node
//! cap) every 1024 nodes; once set, no further TT stores happen. The main
//! thread *never* flips the flag when it finishes — helpers stop on their own
//! (same depth cap, same soft budget, same node cap, or the external stop),
//! so a bounded `go depth N` reports `stopped = false`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::endgame::{Syzygy, TbOutcome};
use crate::move_ordering::OrderingTables;
use crate::move_ordering::order_moves_ctx;
use crate::search::iterative::iterative_search;
use crate::search::{
    SearchResult, SearchShared, SearchStats, SearchThread, ThreadResult, TimeLimit, WorkerDetail,
};
use crate::types::{Depth, RawMove};

/// The number of threads used when the caller passes `threads == 0`.
pub const MIN_THREADS: usize = 1;

/// One search job handed to the helper workers. `SearchShared` and the board
/// are owned (an `Arc` + clones) so workers are `'static` and can park between
/// searches; a fresh job is built per `go`.
pub struct SearchJob {
    /// The shared search state (TT, params, stop, node counter, node cap).
    pub shared: SearchShared,
    /// The root position, owned, so every worker reads its own copy.
    pub root: Position,
    /// Pre-root position hashes for repetition detection.
    pub history: Vec<Zobrist64>,
    pub limits: TimeLimit,
    /// The full, ordered root move list — every worker searches all of it.
    pub moves: Vec<RawMove>,
    /// Completed helper results, drained by the main thread once `pending`
    /// reaches zero.
    pub helpers: Mutex<Vec<(ThreadResult, WorkerDetail)>>,
}

/// Pool-side state protected by the single coordination mutex (only ever
/// touched at search boundaries, never in the node loop).
struct PoolState {
    /// Bumped once per launched search; parked workers re-check it on wake.
    generation: u64,
    shutdown: bool,
    /// The current job. Cleared by the main thread when the search ends so a
    /// stray wake (pool resize) can never make a worker re-run a stale job.
    job: Option<Arc<SearchJob>>,
    /// Helper workers still running the current job.
    pending: usize,
}

struct PoolShared {
    state: Mutex<PoolState>,
    cond: Condvar,
}

/// A persistent pool of Lazy-SMP helper workers.
pub struct LazySmpPool {
    shared: Arc<PoolShared>,
    /// One `(exit_signal, join handle)` per worker. `exit_signal` is flipped
    /// only for the worker being retired, so a resize can never race another
    /// worker into exiting by mistake.
    workers: Vec<(Arc<AtomicBool>, JoinHandle<()>)>,
}

impl LazySmpPool {
    /// A pool with no helpers (the `Threads = 1` default).
    pub fn new() -> LazySmpPool {
        LazySmpPool {
            shared: Arc::new(PoolShared {
                state: Mutex::new(PoolState {
                    generation: 0,
                    shutdown: false,
                    job: None,
                    pending: 0,
                }),
                cond: Condvar::new(),
            }),
            workers: Vec::new(),
        }
    }

    /// Number of live helper workers (the search thread count is this + 1).
    pub fn helper_count(&self) -> usize {
        self.workers.len()
    }

    /// Grows/shrinks the pool to exactly `helpers` workers. Threads are only
    /// created lazily here (never per search); shrinking retires the excess
    /// workers and joins them. Must not be called while the pool is running a
    /// search (the UCI layer joins first).
    pub fn set_helpers(&mut self, helpers: usize) {
        if helpers == self.workers.len() {
            return;
        }
        while self.workers.len() > helpers {
            let (exit, handle) = self.workers.pop().expect("help vec non-empty");
            exit.store(true, Ordering::Relaxed);
            // Wake the parked pool; the generation bump also invalidates any
            // stale wait so the exiting worker cannot miss its flag.
            {
                let mut st = self.shared.state.lock().unwrap();
                st.generation += 1;
            }
            self.shared.cond.notify_all();
            let _ = handle.join();
        }
        while self.workers.len() < helpers {
            let worker_shared = self.shared.clone();
            let exit = Arc::new(AtomicBool::new(false));
            let exit2 = exit.clone();
            let id = self.workers.len() + 1;
            let handle = std::thread::Builder::new()
                .name(format!("morstilia-smp-worker-{id}"))
                .spawn(move || worker_main(worker_shared, exit2))
                .expect("failed to spawn SMP helper worker");
            self.workers.push((exit, handle));
        }
    }

    /// Hands a freshly built job to every worker and returns immediately. The
    /// calling thread runs its own search in parallel (Lazy SMP) and then
    /// calls [`LazySmpPool::await_helpers`].
    pub fn launch(&self, job: Arc<SearchJob>) {
        let mut st = self.shared.state.lock().unwrap();
        st.job = Some(job);
        st.pending = self.workers.len();
        st.generation += 1;
        drop(st);
        self.shared.cond.notify_all();
    }

    /// Blocks until every helper has finished the current job and returns
    /// their results (in no particular order). Never flips the stop flag:
    /// helpers stop on their own via the shared limits, so a naturally
    /// completed bounded search still reports `stopped = false`.
    pub fn await_helpers(&self) -> Vec<(ThreadResult, WorkerDetail)> {
        let job = {
            let mut st = self.shared.state.lock().unwrap();
            while st.pending > 0 {
                st = self.shared.cond.wait(st).unwrap();
            }
            let job = st.job.take();
            // Clear the stale job *inside* the lock: helpers push their
            // results before decrementing pending, so `pending == 0` here
            // means every helper result is already visible.
            job
        };
        let mut out: Vec<(ThreadResult, WorkerDetail)> = Vec::new();
        if let Some(job) = job {
            out.extend(job.helpers.lock().unwrap().iter().cloned());
        }
        out
    }
}

impl Default for LazySmpPool {
    fn default() -> Self {
        LazySmpPool::new()
    }
}

impl Drop for LazySmpPool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.shutdown = true;
            st.generation += 1;
        }
        self.shared.cond.notify_all();
        while let Some((exit, handle)) = self.workers.pop() {
            exit.store(true, Ordering::Relaxed);
            self.shared.cond.notify_all();
            let _ = handle.join();
        }
    }
}

/// Runs a full search. The shared state must belong to exactly one search;
/// `root` and `history` are only read, so sharing them across workers is
/// safe. `pool` shouldn't need to be sized: [`crate::search::Searcher::search`]
/// resizes it before calling back here.
pub fn run_search(
    pool: &LazySmpPool,
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
    // single-thread path and as the baseline order every worker searches).
    let all = order_root_moves(shared, root, searchmoves);

    // `Threads = 1` fast path: the whole search runs inline, with no pool
    // involvement and a byte-for-byte deterministic result.
    if threads == 1 {
        let mut thread = SearchThread::new();
        thread.deadline = limits.hard_deadline();
        let res = iterative_search(&mut thread, shared, root, history, limits, &all);
        let ThreadResult {
            best,
            score,
            depth,
            nodes,
            pv,
            stats,
        } = res;
        let workers = vec![WorkerDetail {
            main: true,
            nodes,
            time_ms: start.elapsed().as_millis(),
            depth,
        }];
        return finish(
            best,
            score,
            depth,
            pv,
            stats,
            nodes,
            workers,
            all.len(),
            shared,
            start,
            threads,
        );
    }

    // Lazy SMP: the calling thread is the main search thread; helpers search
    // the same full root list in parallel. The shared TT is the only
    // coordination channel.
    let job = Arc::new(SearchJob {
        shared: SearchShared {
            tt: shared.tt.clone(),
            params: shared.params.clone(),
            stop: shared.stop.clone(),
            nodes: shared.nodes.clone(),
            node_cap: shared.node_cap,
            tb: shared.tb.clone(),
            nnue: shared.nnue.clone(),
        },
        root: root.clone(),
        history: history.to_vec(),
        limits: *limits,
        moves: all,
        helpers: Mutex::new(Vec::new()),
    });
    pool.launch(job.clone());

    let start_main = Instant::now();
    let mut thread = SearchThread::new();
    thread.deadline = limits.hard_deadline();
    let res = iterative_search(
        &mut thread,
        &job.shared,
        &job.root,
        &job.history,
        &job.limits,
        &job.moves,
    );
    let ThreadResult {
        best,
        score,
        depth,
        nodes,
        pv,
        stats,
    } = res;
    let mut workers = vec![WorkerDetail {
        main: true,
        nodes,
        time_ms: start_main.elapsed().as_millis(),
        depth,
    }];

    let helpers = pool.await_helpers();

    let mut stats = stats;
    let mut nodes = nodes;
    for (helper_res, detail) in &helpers {
        stats.add(&helper_res.stats);
        nodes += helper_res.nodes;
        workers.push(detail.clone());
    }
    let root_moves = job.moves.len();
    finish(
        best,
        score,
        depth,
        pv,
        stats,
        nodes,
        workers,
        root_moves,
        &job.shared,
        start,
        threads,
    )
}

/// Orders the root moves exactly as the single-thread path expects: TT move
/// first, then captures/promotions by MVV-LVA, killers and history.
fn order_root_moves(
    shared: &SearchShared,
    root: &Position,
    searchmoves: &[RawMove],
) -> Vec<RawMove> {
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
    order_moves_ctx(
        &mut root_moves,
        root,
        &tables,
        tt_move,
        0,
        None,
        None,
        &shared.params,
    );
    order_root_by_tb(shared, root, &mut root_moves);
    root_moves.iter().collect()
}

/// Applies Syzygy guidance to the root ordering (runs once per `go`, for
/// every worker at once): the DTZ-optimal move of a *decisive* root is hoisted
/// to the front so the search confirms it first; then winning moves come
/// first (fastest win first), drawn and unranked moves keep their existing
/// order, and losing moves go last (longest defense first). This is ordering
/// only — every legal (filtered) root move stays searchable, and a position
/// the tables do not cover leaves the order untouched. With no tables loaded
/// the function is a no-op, keeping `Threads = 1` byte-for-byte deterministic
/// in bench runs.
fn order_root_by_tb(
    shared: &SearchShared,
    root: &Position,
    root_moves: &mut crate::types::MoveList,
) {
    let tb: &Syzygy = &shared.tb;
    if tb.max_pieces() == 0 {
        return;
    }
    if shakmaty::Position::board(&root.chess).occupied().count() > tb.max_pieces() {
        return;
    }
    // 1) Per-move ranking. `primary` = the outcome class *for us*: 0 = we
    // win, 1 = draw / no tablebase info, 2 = we lose. `secondary` =
    // `-child_dtz` sorts the fastest win first (child dtz negative, closest
    // to 0 = fastest) and the longest defense first among losses (child dtz
    // positive, largest = longest). Unranked moves sit in the draw class and
    // keep their original relative order (stable by index).
    struct Ranked {
        primary: i32,
        secondary: i64,
        idx: usize,
    }
    let mut ranked: Vec<Ranked> = Vec::with_capacity(root_moves.len());
    let mut ordered: Vec<RawMove> = root_moves.iter().collect();
    for (idx, &m) in ordered.iter().enumerate() {
        let child = root.make_child(m);
        let primary = match tb.probe_wdl(&child.chess) {
            Some(TbOutcome::Win | TbOutcome::CursedWin) => 2, // opponent wins
            Some(TbOutcome::Draw) | None => 1,
            Some(TbOutcome::Loss | TbOutcome::BlessedLoss) => 0, // we win
        };
        let secondary = if primary == 1 {
            0
        } else if let Some(d) = tb.probe_dtz(&child.chess) {
            -(d as i64)
        } else {
            0
        };
        ranked.push(Ranked {
            primary,
            secondary,
            idx,
        });
    }
    ranked.sort_by_key(|r| (r.primary, r.secondary, r.idx));
    ordered = ranked.iter().map(|r| ordered[r.idx]).collect();

    // 2) A decisive root leads with the DTZ-optimal move (fastest win, or the
    // longest resistance when the root is a loss). The ranking above already
    // favours it; hoisting makes the preference explicit and deterministic.
    if matches!(
        tb.probe_wdl(&root.chess),
        Some(TbOutcome::Win | TbOutcome::Loss)
    ) {
        if let Some((best, _)) = tb.root_best_move(&root.chess) {
            if let Some(i) = ordered.iter().position(|&m| m == best) {
                let mv = ordered.remove(i);
                ordered.insert(0, mv);
            }
        }
    }

    let mut rebuilt = crate::types::MoveList::new();
    for m in ordered {
        rebuilt.push(m);
    }
    *root_moves = rebuilt;
}

/// The persistent helper worker loop: wait for a job, run one search, publish
/// the result, repeat. Clean shutdown happens through `should_exit` (pool
/// shrink) or the pool's `shutdown` flag (drop).
fn worker_main(pool: Arc<PoolShared>, should_exit: Arc<AtomicBool>) {
    let mut thread = SearchThread::new();
    let mut last_gen = 0u64;

    loop {
        let job = {
            let mut st = pool.state.lock().unwrap();
            while st.generation == last_gen && !st.shutdown {
                if should_exit.load(Ordering::Relaxed) {
                    return;
                }
                st = pool.cond.wait(st).unwrap();
            }
            if should_exit.load(Ordering::Relaxed) || st.shutdown {
                return;
            }
            last_gen = st.generation;
            st.job.clone()
        };
        let Some(job) = job else {
            continue; // a wake that carried no job (pool resize)
        };

        // Run one search. A panic is treated as a dropped helper: release
        // builds abort anyway (`panic = "abort"`), and in tests a panicked
        // worker must not wedge `await_helpers` — the main thread's search is
        // unaffected.
        let start = Instant::now();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            thread.begin(job.limits.hard_deadline());
            iterative_search(
                &mut thread,
                &job.shared,
                &job.root,
                &job.history,
                &job.limits,
                &job.moves,
            )
        }));

        let mut st = pool.state.lock().unwrap();
        st.pending = st.pending.saturating_sub(1);
        if let (Ok(res), Some(job)) = (outcome, st.job.as_ref()) {
            let detail = WorkerDetail {
                main: false,
                nodes: res.nodes,
                time_ms: start.elapsed().as_millis(),
                depth: res.depth,
            };
            job.helpers.lock().unwrap().push((res, detail));
        }
        drop(st);
        pool.cond.notify_all();

        if should_exit.load(Ordering::Relaxed) {
            return;
        }
    }
}

/// Wraps the main worker's result into the public [`SearchResult`] with the
/// exact per-worker node total, aggregated instrumentation and wall time.
#[allow(clippy::too_many_arguments)]
fn finish(
    best: RawMove,
    score: i32,
    depth: Depth,
    pv: Vec<RawMove>,
    stats: SearchStats,
    nodes: u64,
    workers: Vec<WorkerDetail>,
    root_moves: usize,
    shared: &SearchShared,
    start: Instant,
    threads: usize,
) -> SearchResult {
    SearchResult {
        best,
        score,
        depth,
        nodes,
        time_ms: start.elapsed().as_millis(),
        threads,
        pv,
        stopped: shared.stop.load(Ordering::Relaxed),
        stats,
        tt_stores: shared.tt.stores(),
        workers,
        root_moves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64};

    /// Concurrent TT hammering: many threads store and probe into one shared
    /// table. There is no assertion about who wins a slot — the table must
    /// simply stay coherent (no torn reads, no panics) under contention.
    #[test]
    fn concurrent_tt_access_stays_coherent() {
        let tt = Arc::new(crate::tt::TranspositionTable::new(8));
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let tt = tt.clone();
                std::thread::spawn(move || {
                    let mut rng = crate::book::SplitMix64(1000 + t as u64);
                    for i in 0..20_000u64 {
                        let hash = rng.next() ^ (i << 12);
                        let mv = RawMove::new(
                            shakmaty::Square::new((i % 64) as u32),
                            shakmaty::Square::new(((i * 7) % 64) as u32),
                            0,
                        );
                        tt.store(
                            hash,
                            mv,
                            (i % 2000) as i32,
                            (i % 12) as i32 + 1,
                            crate::tt::Bound::Exact,
                        );
                        let _ = tt.probe(hash);
                        let _ = tt.probe_move(hash);
                    }
                })
            })
            .collect();
        for h in threads {
            h.join().unwrap();
        }
        // The shared table must survive the hammering: it still reports the
        // stores every worker issued (each individual store is a valid
        // relaxed-atomic write, so the total is exact).
        assert!(tt.stores() > 0);
    }

    #[test]
    fn pool_shrinks_and_grows_without_leaks() {
        let mut pool = LazySmpPool::new();
        assert_eq!(pool.helper_count(), 0);
        pool.set_helpers(4);
        assert_eq!(pool.helper_count(), 4);
        pool.set_helpers(4); // idempotent
        assert_eq!(pool.helper_count(), 4);
        pool.set_helpers(1);
        assert_eq!(pool.helper_count(), 1, "excess workers must retire");
        // Idempotence, then a full teardown via Drop.
        pool.set_helpers(0);
        assert_eq!(pool.helper_count(), 0);
    }

    #[test]
    fn pool_launch_await_roundtrip() {
        // Launch a trivial search job and ensure the pool actually runs (and
        // awaits) helpers: the pending counter must drain and results come
        // back with helper details marked `main: false`.
        let mut pool = LazySmpPool::new();
        pool.set_helpers(2);
        let root = crate::board::Position::startpos();
        let shared = SearchShared {
            tt: Arc::new(crate::tt::TranspositionTable::new(4)),
            params: Arc::new(crate::evaluation::EvalParams::default()),
            stop: Arc::new(AtomicBool::new(false)),
            nodes: Arc::new(AtomicU64::new(0)),
            node_cap: None,
            tb: Arc::new(crate::endgame::Syzygy::none()),
            nnue: None,
        };
        let limits = TimeLimit {
            depth: Some(3),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let moves: Vec<RawMove> = root.legal_moves().iter().collect();
        let job = Arc::new(SearchJob {
            shared,
            root: root.clone(),
            history: Vec::new(),
            limits,
            moves,
            helpers: Mutex::new(Vec::new()),
        });
        pool.launch(job.clone());
        let results = pool.await_helpers();
        assert_eq!(results.len(), 2, "both helpers must report");
        for (res, detail) in &results {
            assert!(!detail.main);
            assert!(res.best != RawMove::NULL);
            assert!(res.depth >= 1);
        }
    }
}

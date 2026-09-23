//! Search: iterative deepening, PVS alpha-beta, quiescence and the
//! multi-threaded root-split runner.
//!
//! # Ownership
//!
//! The engine keeps no global mutable search state. A search owns:
//!
//! * [`SearchShared`] — the transposition table (a shared reference, lookups
//!   are lock-free), the stop signal and the node counter;
//! * per-worker [`SearchThread`] — the ordering tables and the search stacks
//!   (PV lines, hashes, static evals, move contexts).
//!
//! With `Threads = 1` the whole search runs on one thread with one fresh
//! [`SearchThread`], so it is fully deterministic: TT stores and probing are
//! pure functions of insertion order, and the per-node atomic counter is a
//! fixed function of the traversal.

pub mod alphabeta;
pub mod iterative;
pub mod pruning;
pub mod qsearch;
pub mod reductions;
pub mod stats;
pub mod time;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::move_ordering::OrderingTables;
use crate::move_ordering::history::MoveCtx;
use crate::tt::TranspositionTable;
use crate::types::{Depth, MATE, MAX_PLY, RawMove};

pub use stats::SearchStats;
pub use time::TimeLimit;

/// Scores at least this far from `MATE` are treated as real scores (not mate
/// scores); used to re-base mate scores through the transposition table.
pub const MATE_ZONE: i32 = MATE - MAX_PLY as i32;

/// Re-bases a TT-stored score (relative to its storing ply) to the current
/// node's search-ply, i.e. "mate in N plies from here".
#[inline]
pub fn tt_score_to_node(score: i32, ply: usize) -> i32 {
    let p = ply as i32;
    if score > MATE_ZONE {
        score - p
    } else if score < -MATE_ZONE {
        score + p
    } else {
        score
    }
}

/// Re-bases a score to the form stored in the TT (relative to `ply`).
#[inline]
pub fn node_score_to_tt(score: i32, ply: usize) -> i32 {
    let p = ply as i32;
    if score > MATE_ZONE {
        score + p
    } else if score < -MATE_ZONE {
        score - p
    } else {
        score
    }
}

/// Everything one worker thread (or the single thread of a `Threads = 1`
/// search) needs beyond the shared table/signals.
///
/// Roughly 40 KB of stack-resident arrays — deliberately no heap traffic in
/// the node loop.
pub struct SearchThread {
    /// Killer + history tables, one set per thread so `Threads = 1` is
    /// deterministic and parallel helpers never contend.
    pub tables: OrderingTables,
    /// Nodes visited by this thread (modulo source for the periodic checks).
    pub nodes: u64,
    /// Instrumentation counters for this thread (TT hits, cutoffs, first-move
    /// success, SEE usage; aggregated into the result by the runner).
    pub stats: SearchStats,
    /// Set when the search should abort (external `stop`, node cap or the
    /// hard deadline); TT stores are skipped once set.
    pub stopped: bool,
    /// Absolute hard deadline (from `TimeLimit::hard_ms`).
    pub deadline: Option<Instant>,
    /// Triangular PV table: `pv_table[ply][k]` is the `k`-th move of the best
    /// line found from ply `ply`, `pv_len[ply]` its length.
    pub pv_table: [[RawMove; MAX_PLY]; MAX_PLY],
    pub pv_len: [usize; MAX_PLY],
    /// Zobrist hashes of the current line (`hashes[ply]` — the current
    /// node), used for repetition detection.
    pub hashes: [Zobrist64; MAX_PLY],
    /// Static evaluations per ply, used for the "improving" heuristic.
    pub evals: [i32; MAX_PLY],
    /// Per-ply move contexts (the move that entered ply `k`), used for
    /// countermove and continuation-history lookups.
    pub ctx: [Option<MoveCtx>; MAX_PLY],
}

impl SearchThread {
    pub fn new() -> SearchThread {
        SearchThread {
            tables: OrderingTables::new(),
            nodes: 0,
            stats: SearchStats::default(),
            stopped: false,
            deadline: None,
            pv_table: [[RawMove::NULL; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            hashes: [Zobrist64::default(); MAX_PLY],
            evals: [0; MAX_PLY],
            ctx: [None; MAX_PLY],
        }
    }

    /// Enters a node: counts it and periodically re-checks the stop signal,
    /// the node cap and the hard deadline. Returns `true` when the search
    /// should abort immediately.
    #[inline]
    pub fn mark_node(&mut self, shared: &SearchShared) -> bool {
        self.nodes += 1;
        shared.nodes.fetch_add(1, Ordering::Relaxed);
        if self.nodes & 1023 == 0 {
            self.refresh_stop(shared);
        }
        self.stopped
    }

    /// Loads the external stop signal, the node cap and the deadline into the
    /// local `stopped` flag (which also flips the shared flag so *all* worker
    /// threads halt together).
    #[inline]
    pub fn refresh_stop(&mut self, shared: &SearchShared) {
        if shared.stop.load(Ordering::Relaxed) {
            self.stopped = true;
            return;
        }
        if let Some(cap) = shared.node_cap {
            if shared.nodes.load(Ordering::Relaxed) >= cap {
                self.stopped = true;
                shared.stop.store(true, Ordering::Relaxed);
                return;
            }
        }
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                self.stopped = true;
                shared.stop.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Records `m` as the best move at `ply`, copying the child's PV line
    /// behind it (triangular PV table).
    #[inline]
    pub fn pv_push(&mut self, ply: usize, m: RawMove) {
        let child_len = self.pv_len[ply + 1];
        // RawMove is Copy, so a whole row can be copied out of the borrow
        // checker's way before writing back into a sibling row.
        let child_line = self.pv_table[ply + 1];
        self.pv_table[ply][ply] = m;
        self.pv_table[ply][ply + 1..ply + 1 + child_len]
            .copy_from_slice(&child_line[ply + 1..ply + 1 + child_len]);
        self.pv_len[ply] = child_len + 1;
    }
}

impl Default for SearchThread {
    fn default() -> Self {
        SearchThread::new()
    }
}

/// State created for the lifetime of one `go`: everything the worker threads
/// share. Lifetimes tie this to the owning `Searcher`, so the TT always
/// outlives the search.
pub struct SearchShared<'a> {
    pub tt: &'a TranspositionTable,
    pub stop: &'a AtomicBool,
    pub nodes: AtomicU64,
    pub node_cap: Option<u64>,
}

/// The result of one worker's iterative-deepening run.
#[derive(Debug, Clone)]
pub struct ThreadResult {
    pub best: RawMove,
    pub score: i32,
    pub depth: Depth,
    pub nodes: u64,
    pub pv: Vec<RawMove>,
    pub stats: SearchStats,
}

/// The combined result of a full (possibly multi-threaded) search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best: RawMove,
    pub score: i32,
    pub depth: Depth,
    pub nodes: u64,
    pub time_ms: u128,
    /// Number of worker threads actually used.
    pub threads: usize,
    pub pv: Vec<RawMove>,
    /// True when the search was aborted (stop / deadline / node cap) rather
    /// than finishing naturally.
    pub stopped: bool,
    /// Aggregate instrumentation counters from every worker.
    pub stats: SearchStats,
    /// Total TT entries written by the search (harvested from the shared
    /// table, never counted twice).
    pub tt_stores: u64,
}

impl SearchResult {
    pub fn is_none(&self) -> bool {
        self.best == RawMove::NULL
    }

    pub fn nps(&self) -> u64 {
        if self.time_ms == 0 {
            0
        } else {
            self.nodes.saturating_mul(1000) / self.time_ms as u64
        }
    }

    /// Effective branching factor: the geometric-mean branching per ply
    /// implied by the `(nodes, depth)` pair, `nodes^(1/depth)`. A single
    /// number that summarizes how much the pruning/reduction stack tames the
    /// tree (classical engines sit around 2–4; the raw branching factor of
    /// chess is ~35 at the root).
    pub fn ebf(&self) -> f64 {
        let d = f64::from(self.depth.max(1));
        (self.nodes as f64).powf(1.0 / d)
    }
}

/// Draw detection for a node: the 50-move rule / insufficient material, or a
/// third repetition of the current position (counting the game history since
/// the last zeroing move plus the line searched so far).
///
/// `history` must hold the hashes of all positions *before* the search root
/// (itself excluded) back to the last irreversible move.
#[inline]
pub fn is_draw(pos: &Position, thread: &SearchThread, history: &[Zobrist64], ply: usize) -> bool {
    if pos.is_drawish() {
        return true;
    }
    let h = pos.hash;
    let mut count = 0u32;
    for &ph in history {
        if ph == h {
            count += 1;
        }
    }
    for &lh in &thread.hashes[..ply] {
        if lh == h {
            count += 1;
        }
    }
    count >= 2
}

/// The engine-side search driver: owns the transposition table (persistent
/// across searches, cleared on `ucinewgame`).
pub struct Searcher {
    pub tt: TranspositionTable,
}

impl Default for Searcher {
    fn default() -> Self {
        Searcher::new(crate::config::DEFAULT_HASH_MB)
    }
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Searcher {
        Searcher {
            tt: TranspositionTable::new(hash_mb),
        }
    }

    /// Resizes the TT (from `setoption Hash`). The caller must ensure no
    /// search is currently borrowing the table.
    pub fn resize(&mut self, hash_mb: usize) {
        self.tt.resize(hash_mb);
    }

    /// `ucinewgame`: zeroes the table and restarts the generation counter.
    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    /// Runs a full search of `root` within `limits`.
    ///
    /// `history` carries the hashes of the positions *before* the root since
    /// the last zeroing move (for repetition detection); `stop` lets an
    /// external caller (the UCI `stop` command) abort at any time; `threads`
    /// selects the root-split worker count; `searchmoves`, when non-empty,
    /// restricts the root to those moves.
    pub fn search(
        &mut self,
        root: &Position,
        history: &[Zobrist64],
        limits: &TimeLimit,
        stop: &AtomicBool,
        threads: usize,
        searchmoves: &[RawMove],
    ) -> SearchResult {
        self.tt.new_search();
        let shared = SearchShared {
            tt: &self.tt,
            stop,
            nodes: AtomicU64::new(0),
            node_cap: limits.nodes,
        };
        crate::threading::run_search(&shared, root, history, limits, threads.max(1), searchmoves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use std::sync::atomic::AtomicBool;

    /// Plays `moves` from the start position, building the engine's history
    /// convention: hashes of the positions before the current root, back to
    /// the last zeroing move.
    fn play(moves: &[&str]) -> (Position, Vec<Zobrist64>) {
        let mut pos = Position::startpos();
        let mut history = Vec::new();
        for m in moves {
            let (child, _) = pos.play_uci(m).unwrap();
            if child.halfmoves() == 0 {
                history.clear();
            } else {
                history.push(pos.hash);
            }
            pos = child;
        }
        (pos, history)
    }

    #[test]
    fn threefold_requires_three_occurrences() {
        // A full knight round trip returns to the starting position, which has
        // then occurred twice (the root + one history entry): not yet a draw.
        let (pos, history) = play(&["g1f3", "g8f6", "f3g1", "f6g8"]);
        let thread = SearchThread::new();
        assert!(!is_draw(&pos, &thread, &history, 0));

        // Playing the round trip twice makes it a third occurrence: a draw.
        let (pos2, history2) = play(&[
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ]);
        assert!(is_draw(&pos2, &thread, &history2, 0));
    }

    #[test]
    fn repetition_along_the_search_line_counts() {
        // history holds one startpos hash; a repeated position inside the
        // searched line makes the current node a third occurrence.
        let (pos, history) = play(&["g1f3", "g8f6", "f3g1", "f6g8"]);
        let mut thread = SearchThread::new();
        thread.hashes[0] = pos.hash;
        assert!(is_draw(&pos, &thread, &history, 1));
        // Without the repeated line entry it is only a second occurrence.
        assert!(!is_draw(&pos, &SearchThread::new(), &history, 0));
    }

    #[test]
    fn root_repetition_is_scored_as_a_draw() {
        let (root, history) = play(&[
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ]);
        let mut searcher = Searcher::new(1);
        let stop = AtomicBool::new(false);
        let res = searcher.search(&root, &history, &TimeLimit::unlimited(), &stop, 1, &[]);
        assert_eq!(res.score, 0);
        assert!(!res.is_none());
    }
}

//! Search: iterative deepening, PVS alpha-beta, quiescence and the
//! multi-threaded shared-search runner.
//!
//! # Ownership
//!
//! The engine keeps no global mutable search state. A search owns:
//!
//! * [`SearchShared`] — the shared transposition table, stop signal, node
//!   counter and evaluation parameters. Table lookups are lock-free; the
//!   counter is committed in batches so SMP helpers never contend on it in
//!   the node loop;
//! * per-worker [`SearchThread`] — the ordering tables and the search stacks
//!   (PV lines, hashes, static evals, move contexts).
//!
//! With `Threads = 1` the whole search runs on one thread with one fresh
//! [`SearchThread`], so it is fully deterministic: TT stores and probing are
//! pure functions of insertion order, and the node counter is a fixed
//! function of the traversal.

pub mod alphabeta;
pub mod iterative;
pub mod pruning;
pub mod qsearch;
pub mod reductions;
pub mod stats;
pub mod time;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use shakmaty::Position as _; // `Chess::board()` used by the tablebase piece-count gate.
use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::endgame::{Syzygy, TbOutcome, wdl_score};
use crate::evaluation::{EvalParams, Evaluator};
use crate::move_ordering::OrderingTables;
use crate::move_ordering::history::MoveCtx;
use crate::nnue::accumulator::AccumulatorStack;
use crate::nnue::features::half_ka_v2_hm;
use crate::nnue::network::Network;
use crate::nnue::types::Color as NnueColor;
use crate::nnue::{self, board::Board as NnueBoard};
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
    /// Per-ply NNUE accumulator stack, one slot per search ply.
    ///
    /// Heap-allocated on the *first* NNUE node and `None` in classical mode,
    /// so a classical engine never allocates the ~540 KB, never pays the
    /// per-search reset, and the struct stays small enough to live on a worker
    /// thread's stack. Every worker thread has its own stack: an accumulator
    /// is a function of one position, never shared state.
    accums: Option<Box<AccumulatorStack>>,
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
            accums: None,
        }
    }

    /// Resets a worker's per-search state. [`SearchThread`] instances persist
    /// across searches inside the SMP pool, so every search must start with a
    /// clean local node count, fresh instrumentation and a cleared stop flag.
    #[inline]
    pub fn begin(&mut self, deadline: Option<Instant>) {
        self.nodes = 0;
        self.stopped = false;
        self.stats = SearchStats::default();
        self.deadline = deadline;
        // No-op in classical mode (the stack was never allocated). In NNUE mode
        // every slot goes back to "not computed", so a slot can never leak a
        // stale accumulator from the previous search.
        if let Some(stack) = self.accums.as_mut() {
            stack.reset();
        }
    }

    // --- NNUE accumulator plumbing -----------------------------------------
    //
    // The invariant every caller relies on: **before** `evaluate_at(pos, ..,
    // ply)` runs, slot `ply` holds a computed accumulator for `pos`. The
    // `make_child` / `make_child_null` facades below establish it; the
    // self-heal inside `evaluate_at` exists only so a missed write degrades
    // into a slow node rather than a wrong score or a panic.

    /// The accumulator stack, allocating it on first use.
    #[inline]
    fn accums_mut(&mut self) -> &mut AccumulatorStack {
        self.accums
            .get_or_insert_with(|| Box::new(AccumulatorStack::new()))
    }

    /// The accumulator for `ply`, or `None` before the first allocation.
    #[inline]
    pub fn accumulator(&self, ply: usize) -> Option<&crate::nnue::accumulator::Accumulator> {
        self.accums.as_deref().map(|s| s.get(ply))
    }

    /// Scores `pos` with whichever evaluator this search is configured for.
    ///
    /// In classical mode this is exactly [`Evaluator::evaluate_with`] and the
    /// NNUE stack is never touched. In NNUE mode it reads slot `ply`, which
    /// must already describe `pos`.
    #[inline]
    pub fn evaluate_at(&mut self, pos: &Position, shared: &SearchShared, ply: usize) -> i32 {
        let Some(net) = shared.nnue.as_deref() else {
            return Evaluator.evaluate_with(pos, &shared.params);
        };
        // Self-heal: a slot that was never written for this search.
        let uncomputed = {
            let stack = self.accums_mut();
            let acc = stack.get(ply);
            !acc.computed[0] || !acc.computed[1]
        };
        if uncomputed {
            let board = NnueBoard::from_position(pos);
            let stack = self.accums_mut();
            let acc = stack.get_mut(ply);
            for p in NnueColor::ALL {
                acc.refresh(p, &board, net);
            }
        }
        let acc = self.accums.as_deref().expect("just allocated").get(ply);
        nnue::evaluate(net, pos, acc)
    }

    /// Builds the child position and keeps the accumulator stack in step.
    ///
    /// `ply` is the *child's* search ply. In classical mode this is exactly
    /// [`Position::make_child`]; in NNUE mode slot `ply` is brought up to date
    /// from slot `ply - 1` with a full refresh for whichever perspective moved
    /// its own king, and an incremental update for the other.
    #[inline]
    pub fn make_child(
        &mut self,
        pos: &Position,
        m: RawMove,
        shared: &SearchShared,
        ply: usize,
    ) -> Position {
        let child = pos.make_child(m);
        let Some(net) = shared.nnue.as_deref() else {
            return child;
        };
        debug_assert!(ply >= 1, "the root is not a child");
        let us = NnueColor::from_shakmaty(pos.turn());
        let board = NnueBoard::from_position(pos);
        let d = board.apply_move(m, us);
        let parent_computed = self
            .accums
            .as_deref()
            .is_some_and(|s| s.get(ply - 1).computed[0] && s.get(ply - 1).computed[1]);
        let stack = self.accums_mut();
        let (parent, acc) = stack.parent_and_child_mut(ply - 1, ply);
        for p in NnueColor::ALL {
            if !parent_computed || half_ka_v2_hm::requires_refresh(&d.dirty_piece, p) {
                acc.refresh(p, &d.board, net);
            } else {
                acc.update_incremental(
                    p,
                    d.board.king_square(p),
                    parent,
                    &d.dirty_piece,
                    &d.dirty_threats,
                    &d.dirty_pawn_pairs,
                    net,
                );
            }
        }
        child
    }

    /// The null-move counterpart of [`SearchThread::make_child`].
    ///
    /// Passing changes nothing on the board, so the child slot is a byte-for-byte
    /// copy of the parent's.
    #[inline]
    pub fn make_child_null(
        &mut self,
        pos: &Position,
        shared: &SearchShared,
        ply: usize,
    ) -> Position {
        let child = pos
            .null_move()
            .expect("null_move is Some when this is called");
        let Some(_net) = shared.nnue.as_deref() else {
            return child;
        };
        debug_assert!(ply >= 1, "the root is not a child");
        let stack = self.accums_mut();
        let (parent, acc) = stack.parent_and_child_mut(ply - 1, ply);
        acc.accumulation[0].copy_from_slice(&parent.accumulation[0]);
        acc.accumulation[1].copy_from_slice(&parent.accumulation[1]);
        acc.psqt[0].copy_from_slice(&parent.psqt[0]);
        acc.psqt[1].copy_from_slice(&parent.psqt[1]);
        acc.computed = parent.computed;
        child
    }

    /// Fully refreshes the root accumulator (slot 0).
    ///
    /// Only the root needs this: every other node's slot is written by the
    /// `make_child` that led to it. A no-op when the stack has not been
    /// allocated yet, so a classical search pays nothing.
    pub fn refresh_root(&mut self, pos: &Position, net: &Network) {
        let board = NnueBoard::from_position(pos);
        let stack = self.accums_mut();
        let acc = stack.get_mut(0);
        for p in NnueColor::ALL {
            acc.refresh(p, &board, net);
        }
    }

    /// Enters a node: counts it and periodically re-checks the stop signal,
    /// the node cap and the hard deadline. Returns `true` when the search
    /// should abort immediately.
    ///
    /// The shared node counter is only touched every 1024 nodes (a single
    /// batched add), never per node: with several SMP workers incrementing
    /// one atomic per node, that shared line would serialize the whole node
    /// loop. The exact per-worker total stays in `self.nodes` and is summed
    /// by the runner, so `Threads = 1` reports precisely the same node count
    /// as before.
    #[inline]
    pub fn mark_node(&mut self, shared: &SearchShared) -> bool {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            shared.nodes.fetch_add(1024, Ordering::Relaxed);
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
/// share. All fields are owned `Arc`s so a [`SearchShared`] can be cloned onto
/// persistent SMP helper threads (the table and parameters live on in the
/// owning `Searcher` regardless).
#[derive(Clone)]
pub struct SearchShared {
    /// The shared lock-free transposition table.
    pub tt: Arc<TranspositionTable>,
    /// The evaluation parameters backing *every* evaluation and ordering
    /// decision in this search (static eval, SEE, MVV-LVA, history tiers).
    /// One fixed set per search, so `Threads = 1` stays deterministic.
    pub params: Arc<EvalParams>,
    /// The (per-`go`) stop signal; an external `stop` command flips it and
    /// every worker aborts on its next 1024-node sampling.
    pub stop: Arc<AtomicBool>,
    /// Approximate shared node counter (committed in 1024-node batches; exact
    /// per-worker totals are summed by the runner into the result).
    pub nodes: Arc<AtomicU64>,
    pub node_cap: Option<u64>,
    /// The configured Syzygy tables (one shared instance across every worker;
    /// the inner tablebase is mutex-guarded behind a cheap piece-count gate).
    pub tb: Arc<Syzygy>,
    /// The NNUE net every worker evaluates with, or `None` for the classical
    /// evaluator.
    ///
    /// Shared read-only: the ~115 MB of weights live in one allocation behind
    /// an `Arc`, so N worker threads cost one copy. The per-thread mutable state
    /// (the accumulator stack) stays on [`SearchThread`].
    pub nnue: Option<Arc<Network>>,
}

impl SearchShared {
    /// Consults the configured Syzygy tables at a quiescence leaf, counting
    /// the attempt (and its verdict) in `thread.stats`. Returns the engine's
    /// score band for a definitive tablebase outcome, or `None` when the
    /// tables do not cover the position (out of range, missing material,
    /// castling rights, ..., or no tables configured at all).
    ///
    /// The piece-count gate runs first, so the midgame never pays for the
    /// mutex at all; the probe itself mutates only the locked table's lazy
    /// lookups (once per material) and never touches the TT or the node
    /// counter, so it is invisible to the alpha-beta search other than as a
    /// returned score.
    #[inline]
    pub fn probe_tb(&self, pos: &Position, thread: &mut SearchThread) -> Option<i32> {
        let tb = &self.tb;
        if tb.max_pieces() == 0 {
            return None;
        }
        if pos.chess.board().occupied().count() > tb.max_pieces() {
            return None;
        }
        thread.stats.tb_probes += 1;
        let outcome = tb.probe_wdl(&pos.chess)?;
        thread.stats.tb_hits += 1;
        match outcome {
            TbOutcome::Win => thread.stats.tb_wins += 1,
            TbOutcome::Draw => thread.stats.tb_draws += 1,
            TbOutcome::Loss => thread.stats.tb_losses += 1,
            TbOutcome::CursedWin | TbOutcome::BlessedLoss => thread.stats.tb_cursed += 1,
        }
        Some(wdl_score(outcome))
    }
}

/// The result of one worker's iterative-deepening run.
#[derive(Debug, Clone)]
pub struct ThreadResult {
    pub best: RawMove,
    pub score: i32,
    pub depth: Depth,
    /// This worker's exact local node count (never the shared, batched
    /// counter).
    pub nodes: u64,
    pub pv: Vec<RawMove>,
    pub stats: SearchStats,
}

/// Per-worker parallel-search statistics (instrumentation only — strength is
/// never measured from these).
#[derive(Debug, Clone)]
pub struct WorkerDetail {
    /// Whether this worker is the main search thread (whose result — best
    /// move, score, PV — is authoritative in Lazy SMP).
    pub main: bool,
    /// Exact nodes visited by this worker.
    pub nodes: u64,
    /// Wall time this worker spent searching.
    pub time_ms: u128,
    /// The depth this worker reached.
    pub depth: Depth,
}

/// The combined result of a full (possibly multi-threaded) search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best: RawMove,
    pub score: i32,
    pub depth: Depth,
    pub nodes: u64,
    pub time_ms: u128,
    /// Number of worker threads actually used (helpers + 1 main).
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
    /// Per-worker parallel statistics (one entry per search thread).
    pub workers: Vec<WorkerDetail>,
    /// Number of legal root moves this search considered.
    pub root_moves: usize,
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
/// across searches, cleared on `ucinewgame`), the evaluation parameters every
/// search evaluates with, and the persistent Lazy-SMP helper pool.
pub struct Searcher {
    pub tt: Arc<TranspositionTable>,
    /// The parameter set used by all searches from this searcher. Defaults to
    /// the baseline `EvalParams::default()`; `Searcher::with_params` injects
    /// a tuned set.
    pub params: Arc<EvalParams>,
    /// The configured Syzygy tables, shared into every search (and, through
    /// `SearchShared`, every worker). Defaults to an inert [`Syzygy::none`].
    pub tb: Arc<Syzygy>,
    /// The NNUE net every search evaluates with, or `None` for the classical
    /// evaluator (the default). The weights are immutable once loaded, so one
    /// `Arc` is shared by every worker and every later search.
    pub nnue: Option<Arc<Network>>,
    pool: crate::threading::LazySmpPool,
}

impl Default for Searcher {
    fn default() -> Self {
        Searcher::new(crate::config::DEFAULT_HASH_MB)
    }
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Searcher {
        Searcher::with_params(hash_mb, EvalParams::default())
    }

    /// Builds a searcher that evaluates everything with `params` (a tuned
    /// parameter set from a TOML file, or the baseline defaults).
    pub fn with_params(hash_mb: usize, params: EvalParams) -> Searcher {
        Searcher {
            tt: Arc::new(TranspositionTable::new(hash_mb)),
            params: Arc::new(params),
            tb: Arc::new(Syzygy::none()),
            nnue: None,
            pool: crate::threading::LazySmpPool::new(),
        }
    }

    /// Installs (or, with `None`, removes) the NNUE net.
    ///
    /// Must not be called while a search is running — the UCI layer joins the
    /// search first. A loaded net is shared with every worker through
    /// [`SearchShared::nnue`]; the ~540 KB per-thread accumulator stack is
    /// allocated lazily on the first NNUE node, so switching to `None` leaves
    /// an already-warm worker's stack allocated but unused.
    pub fn set_nnue(&mut self, net: Option<Arc<Network>>) {
        self.nnue = net;
    }

    /// The net this searcher evaluates with, if any (diagnostics / tests).
    pub fn nnue(&self) -> Option<&Arc<Network>> {
        self.nnue.as_ref()
    }

    /// Swaps in a freshly loaded Syzygy tablebase (from `SyzygyPath` /
    /// `--syzygy`). Must not be called while a search is running (the UCI
    /// layer joins first); an inert instance disables probing entirely.
    pub fn set_syzygy(&mut self, tb: Syzygy) {
        self.tb = Arc::new(tb);
    }

    /// The tablebase this searcher currently probes with (diagnostics / tests).
    pub fn syzygy(&self) -> &Arc<Syzygy> {
        &self.tb
    }

    /// Resizes the TT (from `setoption Hash`; the old table is dropped once
    /// nothing references it). The caller must ensure no search is currently
    /// borrowing the table.
    pub fn resize(&mut self, hash_mb: usize) {
        self.tt = Arc::new(TranspositionTable::new(hash_mb));
    }

    /// Resizes the persistent worker pool (from `setoption Threads`). Must
    /// not be called while a search is running (the UCI layer joins first);
    /// spawning is also lazy, so a call with no size change is free.
    pub fn set_threads(&mut self, threads: usize) {
        self.pool.set_helpers(threads.max(1).saturating_sub(1));
    }

    /// Number of persistent helper workers (the thread count of a search is
    /// this + 1). Diagnostic / leak-check accessor.
    pub fn helper_count(&self) -> usize {
        self.pool.helper_count()
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
    /// selects the search worker count (1 = the deterministic fast path);
    /// `searchmoves`, when non-empty, restricts the root to those moves.
    pub fn search(
        &mut self,
        root: &Position,
        history: &[Zobrist64],
        limits: &TimeLimit,
        stop: &Arc<AtomicBool>,
        threads: usize,
        searchmoves: &[RawMove],
    ) -> SearchResult {
        let threads = threads.max(1);
        // Lazy pool sizing: a no-op when the helper count already matches
        // (the UCI `setoption Threads` path pre-sizes). With `Threads = 1`
        // this shrinks the pool to zero so the fast path has no parallel
        // overhead at all.
        self.pool.set_helpers(threads.saturating_sub(1));
        self.tt.new_search();
        let shared = SearchShared {
            tt: self.tt.clone(),
            params: self.params.clone(),
            stop: stop.clone(),
            nodes: Arc::new(AtomicU64::new(0)),
            node_cap: limits.nodes,
            tb: self.tb.clone(),
            nnue: self.nnue.clone(),
        };
        crate::threading::run_search(
            &self.pool,
            &shared,
            root,
            history,
            limits,
            threads,
            searchmoves,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use std::sync::Arc;
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
        let stop = Arc::new(AtomicBool::new(false));
        let res = searcher.search(&root, &history, &TimeLimit::unlimited(), &stop, 1, &[]);
        assert_eq!(res.score, 0);
        assert!(!res.is_none());
    }

    /// Searches `fen` at `depth` with `threads` workers; `stop` is owned by
    /// the caller so it can abort externally.
    fn search_n(fen: &str, depth: i32, threads: usize, stop: Arc<AtomicBool>) -> SearchResult {
        let pos = Position::from_fen(fen).unwrap();
        let mut searcher = Searcher::new(16);
        let limits = TimeLimit {
            depth: Some(depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        searcher.search(&pos, &[], &limits, &stop, threads, &[])
    }

    #[test]
    fn parallel_search_reaches_the_depth_and_returns_a_legal_move() {
        // The Lazy-SMP path must complete bounded searches naturally (no stop
        // involved) with the full requested depth and a legal best move.
        for threads in [2usize, 4] {
            let res = search_n(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                4,
                threads,
                Arc::new(AtomicBool::new(false)),
            );
            assert_eq!(res.depth, 4, "threads={threads} must reach depth 4");
            assert_eq!(res.threads, threads, "worker count must be reported");
            assert_eq!(res.workers.len(), threads, "one WorkerDetail per worker");
            assert!(!res.is_none());
            assert!(!res.stopped, "a bounded depth search finishes naturally");
            let pos = Position::from_fen(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            )
            .unwrap();
            assert!(
                pos.raw_move_legal(res.best),
                "parallel best move must be legal, got {}",
                res.best.to_uci()
            );
        }
    }

    #[test]
    fn parallel_search_stops_mid_search() {
        // An external stop must abort *every* worker and still yield the move
        // found by the completed part of the search. Progress is detected
        // deterministically via the shared TT's store counter (cloned before
        // the searcher moves into its thread): once the table provably holds
        // far more entries than a depth-1 search produces, the main worker
        // has finished at least one full iteration, so its best move is set.
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let mut searcher = Searcher::new(16);
        let tt = searcher.tt.clone();
        let handle = std::thread::spawn(move || {
            let pos = Position::from_fen(
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            )
            .unwrap();
            let limits = TimeLimit {
                depth: Some(12),
                nodes: None,
                movetime_ms: None,
                soft_ms: 0,
                hard_ms: 0,
                infinite: true,
            };
            searcher.search(&pos, &[], &limits, &stop2, 4, &[])
        });
        // Wait until the search has demonstrably done real work (thousands of
        // times more TT stores than a single depth-1 iteration produces), then
        // abort it mid-flight. The store rate in a debug build is a few k/ms,
        // so this settles in a second or two on any machine.
        while tt.stores() < 5_000 {
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
        stop.store(true, Ordering::Relaxed);
        let res = handle.join().unwrap();
        assert!(res.stopped, "an external stop must be reported");
        assert!(
            res.best != RawMove::NULL,
            "the completed part must still yield a move"
        );
        assert!(
            res.nodes > 0 && res.stats.tt_probes > 0,
            "the search must have done real work before the stop"
        );
    }

    #[test]
    fn repeated_parallel_searches_keep_working() {
        // Repeated start/stop cycles on one searcher: the persistent pool must
        // survive, no worker may leak or wedge, and every search must finish.
        let mut searcher = Searcher::new(16);
        let stop = Arc::new(AtomicBool::new(false));
        let pos = Position::from_fen(
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
        )
        .unwrap();
        for i in 0..6 {
            let lim = TimeLimit {
                depth: Some(3 + (i as i32) % 2),
                nodes: None,
                movetime_ms: None,
                soft_ms: 0,
                hard_ms: 0,
                infinite: true,
            };
            let res = searcher.search(&pos, &[], &lim, &stop, 4, &[]);
            assert_eq!(res.depth, 3 + (i as i32) % 2);
            assert!(!res.is_none());
        }
        // No helper left behind after searches (pool is persistent and exact).
        assert_eq!(searcher.helper_count(), 3);
    }

    #[test]
    fn resize_hash_between_searches_is_safe() {
        // `setoption Hash` resizes between searches; the next search must run
        // on the new table with no stale references.
        let mut searcher = Searcher::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let limits = TimeLimit {
            depth: Some(3),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let before = searcher.search(&pos, &[], &limits, &stop, 2, &[]);
        assert!(before.tt_stores > 0, "first search stores into the table");
        let old_entries = searcher.tt.entries();
        searcher.resize(256);
        assert!(searcher.tt.entries() > old_entries);
        let after = searcher.search(&pos, &[], &limits, &stop, 2, &[]);
        assert_eq!(after.depth, 3);
        assert!(!after.is_none());
    }

    #[test]
    fn thread_count_change_resizes_the_pool() {
        // Growing and shrinking the pool across searches must work and leave
        // exactly the requested helper count each time.
        let mut searcher = Searcher::new(16);
        let stop = Arc::new(AtomicBool::new(false));
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let limits = TimeLimit {
            depth: Some(3),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        searcher.set_threads(4);
        assert_eq!(searcher.helper_count(), 3);
        let _ = searcher.search(&pos, &[], &limits, &stop, 4, &[]);
        searcher.set_threads(8);
        assert_eq!(searcher.helper_count(), 7);
        let _ = searcher.search(&pos, &[], &limits, &stop, 8, &[]);
        searcher.set_threads(1);
        assert_eq!(
            searcher.helper_count(),
            0,
            "Threads=1 must leave no helper workers behind (no leaks)"
        );
        let _ = searcher.search(&pos, &[], &limits, &stop, 1, &[]);
        assert_eq!(searcher.helper_count(), 0);
    }

    #[test]
    fn parallel_result_reports_per_worker_statistics() {
        let res = search_n(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            3,
            4,
            Arc::new(AtomicBool::new(false)),
        );
        assert_eq!(res.workers.iter().filter(|w| w.main).count(), 1);
        let sum: u64 = res.workers.iter().map(|w| w.nodes).sum();
        assert_eq!(sum, res.nodes, "nodes = exact sum of worker-local counts");
        assert!(res.root_moves > 0, "root move count must be recorded");
    }

    // --- Syzygy -------------------------------------------------------------

    fn tb_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/syzygy")
    }

    /// Searches `fen` at `depth` with the 3/4-piece test tables loaded.
    fn search_tb(fen: &str, depth: i32, threads: usize) -> SearchResult {
        let pos = Position::from_fen(fen).unwrap();
        let (tb, report) = crate::endgame::Syzygy::load(tb_dir().to_str().unwrap());
        assert_eq!(report.max_pieces, 4, "tables must be present for this test");
        let stop = Arc::new(AtomicBool::new(false));
        let limits = TimeLimit {
            depth: Some(depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let mut searcher = Searcher::new(16);
        searcher.set_syzygy(tb);
        searcher.search(&pos, &[], &limits, &stop, threads, &[])
    }

    #[test]
    fn tablebase_win_is_scored_in_the_win_band_not_as_mate() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let res = search_tb("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 4, 1);
        assert!(!res.is_none());
        assert!(!res.stopped);
        assert_eq!(res.depth, 4);
        assert_eq!(res.score, 19_999, "unconditional KQvK win band");
        assert!(
            !crate::types::is_mate(res.score),
            "a TB win is never a mate score"
        );
        assert!(pos.raw_move_legal(res.best), "best move must be legal");
    }

    #[test]
    fn tablebase_loss_is_reported_as_a_loss_not_a_mate() {
        let res = search_tb("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", 4, 1);
        assert!(res.score <= -19_000, "black must be TB-lost: {}", res.score);
        assert!(
            !crate::types::is_mate(res.score),
            "a TB loss is never a mate score"
        );
    }

    #[test]
    fn tablebase_draw_position_scores_zero() {
        let res = search_tb("4k2r/8/8/8/8/8/8/3RK3 w - - 0 1", 4, 1);
        assert_eq!(res.score, 0, "KRvKR is a tablebase draw");
    }

    #[test]
    fn root_selection_confirms_the_db_optimal_move() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let limits = TimeLimit {
            depth: Some(3),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let mut searcher = Searcher::new(16);
        let (tb, report) = crate::endgame::Syzygy::load(tb_dir().to_str().unwrap());
        assert_eq!(report.max_pieces, 4);
        let (db_best, _) = tb.root_best_move(&pos.chess).expect("KQvK is covered");
        searcher.set_syzygy(tb);
        let res = searcher.search(&pos, &[], &limits, &stop, 1, &[]);
        assert_eq!(res.best, db_best, "the DB-optimal move must be selected");
        assert_eq!(res.score, 19_999);
    }

    #[test]
    fn with_tables_threads_one_stays_byte_for_byte_deterministic() {
        let fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";
        let r1 = search_tb(fen, 4, 1);
        let r2 = search_tb(fen, 4, 1);
        assert_eq!(r1.nodes, r2.nodes);
        assert_eq!(r1.score, r2.score);
        assert_eq!(r1.best, r2.best);
        assert_eq!(r1.stats, r2.stats);
        // Every depth-0 leaf of KQvK sits inside the 3/4-piece tables, so
        // every probe must resolve (hits == probes) deterministically.
        assert!(
            r1.stats.tb_probes > 0,
            "the tablebase must actually be probed"
        );
        assert_eq!(r1.stats.tb_hits, r1.stats.tb_probes);
        assert_eq!(
            r1.stats.tb_wins + r1.stats.tb_draws + r1.stats.tb_losses + r1.stats.tb_cursed,
            r1.stats.tb_hits
        );
    }

    #[test]
    fn with_tables_lazy_smp_stays_stable() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let res = search_tb("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 4, 2);
        assert_eq!(res.depth, 4);
        assert_eq!(res.threads, 2);
        assert!(!res.is_none());
        assert!(!res.stopped);
        assert!(
            res.score >= 19_000,
            "threads=2 must stay in the win band: {}",
            res.score
        );
        assert!(!crate::types::is_mate(res.score));
        assert!(pos.raw_move_legal(res.best), "best move must be legal");
    }
}

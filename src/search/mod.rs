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
pub mod correction;
pub mod iterative;
pub mod params;
pub mod pruning;
pub mod qsearch;
pub mod reductions;
pub mod singular;
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
use crate::move_ordering::is_capture_or_promotion;
use crate::nnue::accumulator::AccumulatorStack;
use crate::nnue::features::half_ka_v2_hm;
use crate::nnue::network::Network;
use crate::nnue::types::Color as NnueColor;
use crate::nnue::{self, board::Board as NnueBoard};
use crate::search::params::SearchParams;
use crate::tt::TranspositionTable;
use crate::types::{Depth, INFINITE, MATE, MAX_PLY, RawMove};

pub use stats::SearchStats;
pub use time::TimeLimit;

/// What the parent knew about the move that entered a given ply.
///
/// Recorded by the parent immediately before it recurses, because the child
/// cannot recover either fact on its own: the child knows whether *it* is in
/// check, not whether its parent was, and it has no board on which to classify
/// the move it arrived by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnteredPly {
    /// The parent's own node was in check — the move was a forced reply and
    /// says little about how good it is.
    pub parent_in_check: bool,
    /// The entering move was a capture or a promotion.
    pub capture: bool,
    /// A real move entered this ply. A null move and a root both do not, which
    /// is what `sf_19`'s `((ss - 1)->currentMove).is_ok()` tests.
    pub real_move: bool,
}

impl EnteredPly {
    /// The all-false, no-move state. Also what the root and a null-move child
    /// get, because neither is the result of a real move being played.
    pub const NONE: EnteredPly = EnteredPly {
        parent_in_check: false,
        capture: false,
        real_move: false,
    };
}

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
    /// The move excluded from the current ply's search by a singular-extension
    /// verification search. `None` means no move is excluded.
    pub excluded_move: [Option<RawMove>; MAX_PLY],
    /// Stockfish's `ss->followPV`: is this node on the principal variation of
    /// the *previous* iteration?
    ///
    /// This is a distinct piece of state from the node's own `pv_node`, and
    /// conflating the two silently changes which branches are reachable (the
    /// quiet-move pruning gate is `!followPV || !PvNode`, a genuine
    /// two-variable condition). Refreshed once per iteration by
    /// [`SearchThread::set_previous_pv`] and once per node by
    /// [`SearchThread::set_follow_pv`].
    pub follow_pv: [bool; MAX_PLY],
    /// The PV the last completed iteration produced.
    ///
    /// Stockfish's `lastIterationIdxPV`; the `followPV` recurrence compares the
    /// move that entered a ply against this.
    pub prev_pv: [RawMove; MAX_PLY],
    /// Length of [`SearchThread::prev_pv`]. `0` on the first iteration, which
    /// makes every non-root `follow_pv` false — the correct answer, since there
    /// is no previous line to follow yet.
    pub prev_pv_len: usize,
    /// The current iteration's root depth. Stockfish's `rootDepth`, needed by the
    /// singular extension margins' `-(ss->ply > rootDepth) * 38 / 43` terms.
    pub root_depth: Depth,
    /// The width of the window the root was last searched with — Stockfish's
    /// `rootDelta`, the divisor in the modern reduction formula.
    ///
    /// Stockfish leaves it uninitialised until its aspiration loop first runs
    /// (`search.cpp:394`); Morstilia seeds it with
    /// [`SearchParams::lmr_root_delta_default`] and refreshes it before every
    /// root search, so the division is always defined.
    pub root_delta: i32,
    /// The `ttMoveHistory` statistic (`sf_19/src/history.h:196`).
    ///
    /// A single damped scalar per worker, **not** a table: how often the node's
    /// TT move has turned out to be the move that actually caused the cutoff.
    /// Positive means the table's move ordering is trustworthy here; the
    /// singular extension margins use it to decide how deep to extend, and the
    /// multi-cut return path penalises it.
    pub tt_move_history: i32,
    /// The reduction the parent applied before entering this ply
    /// (`ss->reduction`), read by the LMR hindsight depth adjustment.
    pub prior_reduction: [Depth; MAX_PLY],
    /// Stockfish's file-static `nmpMinPly`: while a high-depth null-move
    /// verification search is running, null moves are suppressed at plies
    /// below this. Zero outside such a search (the common case), so the
    /// null-move gate `ss->ply >= nmpMinPly` is a no-op everywhere else.
    pub nmp_min_ply: i32,
    /// The score the previous iteration of the root search produced, i.e.
    /// `rootMoves[pvIdx].score` at the start of the current iteration. Drives
    /// `seekMate = rootDepth >= 16 && |rootScore| >= 2000`, which gates the
    /// child reverse-futility depth bound. `-INFINITE` before the first
    /// iteration, which is the correct "we have no mate in sight" answer.
    pub root_score: i32,
    /// How many beta cutoffs this node's first child already produced
    /// (`(ss + 1)->cutoffCnt`), read by the modern reduction formula.
    pub cutoff_count: [i32; MAX_PLY],
    /// Facts about the move that *entered* each ply, recorded by the parent
    /// before it recurses. Both are read by the Stockfish 19 evaluation-difference
    /// history bonus (`search.cpp:978-986`), which needs to know whether the
    /// parent's own node was in check and whether the move into it was a
    /// capture — neither of which is recoverable at the child.
    pub entered: [EnteredPly; MAX_PLY],
    /// Learned static-evaluation corrections (`src/search/correction.rs`).
    ///
    /// Per thread, like every other learned table: one instance per worker
    /// keeps `Threads = 1` fully deterministic and parallel helpers free of
    /// contention. Cleared by [`SearchThread::begin`] so no evidence leaks
    /// between searches.
    pub corrections: crate::search::correction::CorrectionHistory,
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
            excluded_move: [None; MAX_PLY],
            follow_pv: [false; MAX_PLY],
            prev_pv: [RawMove::NULL; MAX_PLY],
            prev_pv_len: 0,
            root_depth: 0,
            root_delta: crate::search::params::SearchParams::default().lmr_root_delta_default,
            tt_move_history: 0,
            nmp_min_ply: 0,
            root_score: -INFINITE,
            prior_reduction: [0; MAX_PLY],
            cutoff_count: [0; MAX_PLY],
            entered: [EnteredPly::NONE; MAX_PLY],
            corrections: crate::search::correction::CorrectionHistory::new(),
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
        // Correction history is per-search evidence, not per-engine knowledge:
        // carrying it across searches would make a later search depend on the
        // order in which earlier positions happened to be visited, and would
        // break the determinism a `Threads = 1` run must have.
        self.corrections.clear();
        // Excluded move is per-search state; clear it to avoid stale exclusions.
        self.excluded_move.fill(None);
        // `followPV` is defined against the *previous* iteration's line, so a
        // new search has none: every non-root node starts off the PV.
        self.follow_pv.fill(false);
        self.prev_pv.fill(RawMove::NULL);
        self.prev_pv_len = 0;
        self.root_depth = 0;
        // `ttMoveHistory` is per-search evidence, exactly like the correction
        // history: it summarises what happened *in this search* and carrying it
        // over would make a later search depend on the order in which earlier
        // positions happened to be visited.
        self.tt_move_history = 0;
        self.prior_reduction.fill(0);
        self.cutoff_count.fill(0);
        self.entered.fill(EnteredPly::NONE);
        // `nmpMinPly` is zero outside a high-depth null-move verification
        // search, and it must start at zero for every search so a
        // verification running in one search cannot leak its floor into the
        // next.
        self.nmp_min_ply = 0;
        // `rootScore` is `-INFINITE` before the first iteration, which is the
        // correct "no mate in sight" answer for the `seekMate` gate.
        self.root_score = -INFINITE;
    }

    /// Records the principal variation the previous iteration produced, which is
    /// what the `followPV` recurrence compares against.
    ///
    /// Called by the iterative-deepening driver once per completed iteration,
    /// before the next iteration's root search. Truncation to [`MAX_PLY`] is
    /// deliberate: a longer line cannot be reached at a legal ply anyway.
    pub fn set_previous_pv(&mut self, pv: &[RawMove]) {
        let n = pv.len().min(MAX_PLY);
        self.prev_pv[..n].copy_from_slice(&pv[..n]);
        self.prev_pv[n..].fill(RawMove::NULL);
        self.prev_pv_len = n;
    }

    /// Refreshes `follow_pv[ply]` and returns it.
    ///
    /// Stockfish's recurrence (`search.cpp:772-776`):
    ///
    /// ```text
    /// ss->followPV = rootNode
    ///             || ((ss - 1)->followPV
    ///                 && ss->ply - 1 < lastIterationIdxPV.size()
    ///                 && (ss - 1)->currentMove == lastIterationIdxPV[ss->ply - 1]);
    /// ```
    ///
    /// Morstilia keeps the move that entered a ply in `ctx[ply - 1]`, which is
    /// the same quantity as `(ss - 1)->currentMove`. The comparison is on
    /// `from`/`to` only, because a [`MoveCtx`] does not store the promotion
    /// kind: a previous-iteration knight promotion and a queen promotion of the
    /// same piece to the same square compare equal. That is a strictly narrow
    /// over-approximation — it can make `followPV` true in one case where
    /// Stockfish would say false — and it is the conservative direction, since
    /// the only consumer that treats `followPV` as "trust this line" is IIR,
    /// where a false positive merely forgoes a reduction.
    #[inline]
    pub fn set_follow_pv(&mut self, ply: usize, root_node: bool) -> bool {
        let on = root_node
            || (ply > 0
                && self.follow_pv[ply - 1]
                && ply <= self.prev_pv_len
                && match self.ctx[ply - 1] {
                    Some(c) => {
                        let want = self.prev_pv[ply - 1];
                        c.from == want.from() && c.to == want.to()
                    }
                    None => false,
                });
        self.follow_pv[ply] = on;
        on
    }

    /// Applies a damped, saturating shift to the `ttMoveHistory` statistic.
    ///
    /// This is Stockfish's `StatsEntry::operator<<`
    /// (`sf_19/src/history.h:69-78`) on a single cell:
    /// `val = val + bonus - val * |bonus| / limit`, with `bonus` clamped to
    /// `±limit`. The damping term is what makes the statistic converge to
    /// `±limit` rather than growing without bound, and it guarantees the value
    /// can never leave the open interval.
    #[inline]
    pub fn shift_tt_move_history(&mut self, bonus: i32, limit: i32) {
        self.tt_move_history =
            crate::search::singular::tt_move_history_shift(self.tt_move_history, bonus, limit);
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

    /// Scores `pos` with whichever evaluator this search is configured for, and
    /// adds the learned correction — **unless the side to move is in check**.
    ///
    /// In classical mode this is exactly [`Evaluator::evaluate_with`] and the
    /// NNUE stack is never touched. In NNUE mode it reads slot `ply`, which
    /// must already describe `pos`.
    ///
    /// The check test is not an optimisation. Stockfish calls
    /// `to_corrected_static_eval` from exactly one place — the "static
    /// evaluation of the position" step — and a check node never reaches it:
    /// `if (ss->inCheck) ss->staticEval = eval = (ss - 2)->staticEval;`. A check
    /// node has no evaluation of its own to correct, and the value it inherits
    /// is deliberately uncorrected, so reading a cell here would learn from a
    /// number the feature is not allowed to touch.
    #[inline]
    pub fn evaluate_at(
        &mut self,
        pos: &Position,
        shared: &SearchShared,
        ply: usize,
        in_check: bool,
    ) -> i32 {
        let raw = self.raw_evaluate_at(pos, shared, ply);
        if in_check {
            // Inherited static evaluations still go through the clamp: that is
            // an engine-wide invariant on every static evaluation, not part of
            // the correction.
            return correction::clamp_non_mate(raw);
        }
        correction::correct(raw, self.correction_at(pos, ply))
    }

    /// The **uncorrected** static evaluation: exactly what the configured
    /// evaluator says, with no learned correction applied.
    ///
    /// Callers that need the evaluator's own opinion rather than the search's
    /// use this. The transposition table stores searched scores, never a static
    /// evaluation, so there is no field in it that a learned correction could be
    /// baked into — but the search's own pruning decisions are the other place a
    /// corrected value would persist, which is exactly why only the number used
    /// for pruning is corrected and the number derived from it is not.
    #[inline]
    pub fn raw_evaluate_at(&mut self, pos: &Position, shared: &SearchShared, ply: usize) -> i32 {
        let Some(net) = shared.nnue.as_deref() else {
            return Evaluator.evaluate_with(pos, &shared.params);
        };
        // Self-heal: a slot that was never written for this search. The check
        // and the repair share one borrow of the stack — the old form walked
        // `self.accums` three times (once to test, once to repair, once to
        // read) on a line that runs for *every* node of the search.
        let stack = self.accums_mut();
        let acc = stack.get_mut(ply);
        if !acc.computed[0] || !acc.computed[1] {
            let board = NnueBoard::from_position(pos);
            for p in NnueColor::ALL {
                acc.refresh(p, &board, net);
            }
        }
        // Reborrow immutably: the mutable borrow above is finished with.
        let acc = stack.get(ply);
        nnue::evaluate(net, pos, acc)
    }

    /// The learned correction for the node at `ply`, in its own raw (scaled)
    /// form.
    ///
    /// The three contexts come off the thread's move stack, so the continuation
    /// families are addressed the way Stockfish addresses them. Near the root the
    /// slots are `None` by construction, which makes the correction fall back to
    /// the material families — and, on the node with no incoming move at all, to
    /// Stockfish's [`correction::NO_INCOMING`] constant.
    #[inline]
    pub fn correction_at(&self, pos: &Position, ply: usize) -> i32 {
        self.corrections
            .correction(pos, correction::CorrectionCtx::from_stack(&self.ctx, ply))
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
        // One borrow of the stack, not two: the parent's `computed` flags and
        // the child's slot are read through the same `parent_and_child_mut`
        // split, so a second `self.accums` walk (an `Option` test plus two
        // slice lookups) is pure overhead on a path that runs once per move
        // made — the single hottest line in the search.
        let stack = self.accums_mut();
        let (parent, acc) = stack.parent_and_child_mut(ply - 1, ply);
        let parent_computed = parent.computed[0] && parent.computed[1];
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

    /// The null-move counterpart of [`SearchThread::make_child`], taking the
    /// already-constructed child position.
    ///
    /// [`Position::null_move`] is not a cheap board edit: it converts the whole
    /// position to a `Setup`, rebuilds a `Chess` from it and recomputes the
    /// Zobrist hash from scratch. The caller has to ask whether the pass is
    /// legal *anyway* (it is part of the null-move guard chain), so the answer
    /// is a value, not a predicate — building it twice, once as a guard and
    /// once here, doubled the most expensive step of the entire null-move path.
    /// The position is therefore built once and handed in.
    #[inline]
    pub fn make_child_null(
        &mut self,
        child: Position,
        shared: &SearchShared,
        ply: usize,
    ) -> Position {
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
    /// as before. The stop signals are checked every 256 nodes: with NNUE a
    /// 1024-node gap is over 15 ms, too coarse for bullet clocks.
    #[inline]
    pub fn mark_node(&mut self, shared: &SearchShared) -> bool {
        self.nodes += 1;
        if self.nodes & 255 == 0 {
            if self.nodes & 1023 == 0 {
                shared.nodes.fetch_add(1024, Ordering::Relaxed);
            }
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
        if let Some(cap) = shared.node_cap
            && shared.nodes.load(Ordering::Relaxed) >= cap
        {
            self.stopped = true;
            shared.stop.store(true, Ordering::Relaxed);
            return;
        }
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            self.stopped = true;
            shared.stop.store(true, Ordering::Relaxed);
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

    /// Tests whether `m` is a "shuffling" move — a move that merely reverses
    /// the move played two plies ago, which in turn reversed the move four
    /// plies ago. This mirrors Stockfish's `is_shuffling` and is used to
    /// suppress singular extensions in positions where pieces are simply
    /// oscillating (a common cause of false singular candidates).
    ///
    /// The check requires:
    /// * `m` is not a capture,
    /// * the 50-move counter is at least 10,
    /// * we are at least 20 plies deep (a proxy for SF's `pliesFromNull >= 6`,
    ///   since Morstilia does not track plies since the last null move),
    /// * the move two plies ago went from `m.to` to `m.from`,
    /// * the move four plies ago went from `m.from` to `m.to`.
    #[inline]
    pub fn is_shuffling(&self, pos: &Position, m: RawMove, ply: usize) -> bool {
        // Captures are never shuffling.
        if is_capture_or_promotion(pos.board(), m) {
            return false;
        }
        // Too early in the game / 50-move clock.
        if pos.halfmoves() < 10 {
            return false;
        }
        // Not deep enough in the tree (SF also checks pliesFromNull >= 6).
        if ply < 20 {
            return false;
        }
        // Need the move two and four plies ago.
        let two_ago = match self.ctx.get(ply.saturating_sub(1)) {
            Some(Some(ctx)) => ctx,
            _ => return false,
        };
        let four_ago = match self.ctx.get(ply.saturating_sub(3)) {
            Some(Some(ctx)) => ctx,
            _ => return false,
        };
        // Check the A->B, B->A pattern: move two ago went to -> from,
        // move four ago went from -> to.
        two_ago.to == m.to()
            && two_ago.from == m.from()
            && four_ago.to == m.from()
            && four_ago.from == m.to()
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
    /// The search parameters: singular extensions, multi-cut, negative
    /// extensions, internal iterative reductions, the modern late-move
    /// reduction formula and the pruning thresholds.
    ///
    /// One fixed set per search for the same reason as [`SearchShared::params`]:
    /// every worker must read the *same* numbers, or a Lazy SMP search stops
    /// being reproducible. Immutable while the search runs — changing a
    /// parameter means starting a new search — so it is shared behind an `Arc`
    /// with no locking on the read path.
    pub sp: Arc<SearchParams>,
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
    /// The search parameters used by all searches from this searcher: singular
    /// extensions, multi-cut, negative extensions, internal iterative
    /// reductions, the modern late-move-reduction formula and the pruning
    /// thresholds.
    ///
    /// Kept separate from [`Searcher::params`] on purpose: the evaluation
    /// parameters are *frozen* (the classical baseline is a correctness
    /// contract for the ground-truth tests), while these are the knobs the
    /// tuner sweeps. Splitting them means a tuning run can never perturb the
    /// evaluator, and a match report can fingerprint the two independently.
    pub sp: Arc<SearchParams>,
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
            sp: Arc::new(SearchParams::default()),
            tb: Arc::new(Syzygy::none()),
            nnue: None,
            pool: crate::threading::LazySmpPool::new(),
        }
    }

    /// Replaces the search parameters for every subsequent search.
    ///
    /// Takes effect on the next `search` call: a running search keeps the `Arc`
    /// it started with, so a parameter can never change under a worker
    /// mid-search. This is the seam the UCI `setoption name SearchParamsPath`
    /// and the tuning binaries use.
    pub fn set_search_params(&mut self, sp: SearchParams) {
        self.sp = Arc::new(sp);
    }

    /// A fingerprint of the active search parameters, for match reports and
    /// configuration dumps. Two engines may only be compared when these match.
    pub fn search_params_fingerprint(&self) -> String {
        self.sp.fingerprint()
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
            sp: self.sp.clone(),
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
        for i in 0..6i32 {
            let lim = TimeLimit {
                depth: Some(3 + i % 2),
                nodes: None,
                movetime_ms: None,
                soft_ms: 0,
                hard_ms: 0,
                infinite: true,
            };
            let res = searcher.search(&pos, &[], &lim, &stop, 4, &[]);
            assert_eq!(res.depth, 3 + i % 2);
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

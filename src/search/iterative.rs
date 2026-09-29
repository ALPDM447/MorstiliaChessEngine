//! Iterative deepening with aspiration windows — the driver that turns a raw
//! node search into a best move.
//!
//! Each worker thread runs [`iterative_search`] over its assigned root moves:
//!
//! 1. order the moves (previous iteration's best first),
//! 2. for each depth 1..=limit: open an aspiration window around the
//!    previous iteration's score (widening on failures), run the root PVS
//!    loop, and record the depth/score/PV,
//! 3. stop between iterations once the soft time budget (or the node cap)
//!    is reached, so an in-flight iteration always finishes cleanly.

use std::time::Instant;

use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::move_ordering::history::MoveCtx;
use crate::search::{SearchShared, SearchThread};
use crate::search::{ThreadResult, TimeLimit, alphabeta, is_draw};
use crate::types::{Depth, INFINITE, MATE, MAX_DEPTH, MAX_PLY, RawMove};

/// Runs iterative deepening over `moves` (this worker's root subset).
///
/// `moves` may be empty (a helper thread with no assigned root moves); the
/// result then reports no best move.
pub fn iterative_search(
    thread: &mut SearchThread,
    shared: &SearchShared,
    root: &Position,
    history: &[Zobrist64],
    limits: &TimeLimit,
    moves: &[RawMove],
) -> ThreadResult {
    let max_depth = limits.depth.unwrap_or(MAX_DEPTH).clamp(0, MAX_DEPTH);

    thread.hashes[0] = root.hash;
    thread.ctx[0] = None;
    // The root is the one node no `make_child` leads to, so its accumulator has
    // to be built from scratch before anything reads it. No-op in classical mode.
    if let Some(net) = shared.nnue.as_deref() {
        thread.refresh_root(root, net);
    }
    thread.evals[0] = thread.evaluate_at(root, shared, 0, root.is_check());
    thread.pv_len[0] = 0;

    let soft = limits.soft_deadline();

    let mut best_move = RawMove::NULL;
    let mut best_score = -INFINITE;
    let mut prev_score = 0i32;
    let mut has_prev = false;
    let mut pv: Vec<RawMove> = Vec::new();
    let mut depth_reached = 0;

    // An empty move list means a `searchmoves` filter excluded everything, or
    // this is a helper that found no work. Exact local node count is reported
    // (Lazy SMP workers never read the shared, batched counter).
    if moves.is_empty() {
        return ThreadResult {
            best: RawMove::NULL,
            score: best_score,
            depth: 0,
            nodes: thread.nodes,
            pv,
            stats: thread.stats,
        };
    }

    // The root itself may already be drawn — a threefold repetition from the
    // game history, the 50-move rule, or insufficient material. Report a draw
    // score instead of searching for a win that cannot be claimed.
    if is_draw(root, thread, history, 0) {
        pv.push(moves[0]);
        return ThreadResult {
            best: moves[0],
            score: 0,
            depth: 0,
            nodes: thread.nodes,
            pv,
            stats: thread.stats,
        };
    }

    for depth in 1..=max_depth {
        if thread.stopped {
            break;
        }
        if depth > MAX_PLY as i32 {
            break;
        }

        // The current iteration's root depth. Read by the singular extension
        // margins as `-(ss->ply > rootDepth) * 38 / 43`: a node deeper than the
        // whole search tree cannot be on the path to a real line, and extending
        // it buys nothing.
        thread.root_depth = depth;

        // `rootScore`: the previous iteration's root score, i.e.
        // `rootMoves[pvIdx].score` at the start of the current iteration in
        // `sf_19`. Drives `seekMate = rootDepth >= 16 && |rootScore| >= 2000`,
        // which gates the child reverse-futility depth bound and the singular
        // extension gate. `-INFINITE` before the first iteration is the correct
        // "we have no mate in sight" answer.
        thread.root_score = if has_prev { prev_score } else { -INFINITE };

        // Put the previous iteration's best move first; keep the caller's
        // ordering (TT-move / captures / history) for the rest.
        let mut ordered: Vec<RawMove> = Vec::with_capacity(moves.len().saturating_add(1));
        if best_move != RawMove::NULL {
            ordered.push(best_move);
        }
        for &m in moves {
            if m != best_move {
                ordered.push(m);
            }
        }

        let (mut alpha, mut beta) = if depth >= 4 && has_prev && !limits.infinite {
            let delta = 20 + depth * 6;
            (prev_score - delta, prev_score + delta)
        } else {
            (-INFINITE, INFINITE)
        };

        let (score, mv) = {
            let mut attempts = 0u32;
            loop {
                // `rootDelta`: the width of the window the root is being
                // searched with right now. The modern reduction formula divides
                // by it (`r -= delta * 577 / rootDelta`), so it must be current
                // for *every* attempt, not just the first — a re-search under a
                // full window is a very different node from one under a 20
                // centipawn aspiration. Stockfish assigns this inside its
                // aspiration loop (`search.cpp:394`); Morstilia seeds it in
                // [`SearchThread::new`] as well so the division is defined even
                // for a search that never opens an aspiration window.
                thread.root_delta = (beta - alpha).max(1);
                let (s, m, fail_low, fail_high) =
                    root_search_depth(thread, shared, root, history, depth, &ordered, alpha, beta);
                if thread.stopped {
                    break (s, m);
                }
                if !fail_low && !fail_high {
                    break (s, m);
                }
                // Failed low or high: the result is a bound, not an exact score.
                // Before widening, check whether we have already exhausted retries
                // or the window already spans the full range.
                if attempts >= 3 && !(alpha <= -MATE && beta >= MATE) {
                    // Aspiration retries exhausted: fall back to a full-window
                    // re-search to obtain an exact result. A bound must never be
                    // committed as an exact iterative score.
                    alpha = -INFINITE;
                    beta = INFINITE;
                    attempts += 1;
                    continue;
                }
                if alpha <= -MATE && beta >= MATE {
                    // Window already full and still failing: accept the bound
                    // (no further widening is possible).
                    break (s, m);
                }
                if fail_low {
                    beta = (alpha + beta) / 2;
                    alpha = -MATE;
                } else {
                    alpha = (alpha + beta) / 2;
                    beta = MATE;
                }
                attempts += 1;
            }
        };
        if thread.stopped {
            break;
        }

        let ply_len = thread.pv_len[0];
        pv = thread.pv_table[0][..ply_len].to_vec();

        // Hand this iteration's line to the next one as its `followPV` reference
        // (`sf_19`'s `lastIterationIdxPV`). Recorded only for iterations that
        // completed and produced a line: a line from a previous *depth* is
        // still the best guess available, and a line from a partial iteration
        // is not.
        thread.set_previous_pv(&pv);

        best_move = mv;
        best_score = score;
        prev_score = score;
        has_prev = true;
        depth_reached = depth;

        // Finish cleanly between iterations once the budget is spent.
        if !limits.infinite
            && let Some(soft) = soft
            && Instant::now() >= soft
        {
            break;
        }
    }

    ThreadResult {
        best: best_move,
        score: best_score,
        depth: depth_reached,
        nodes: thread.nodes,
        pv,
        stats: thread.stats,
    }
}

/// One root iteration: PVS over all assigned root moves at a fixed depth.
///
/// Returns `(best_score, best_move, failed_low, failed_high)`. The failed
/// flags drive the aspiration-window widening in [`iterative_search`].
#[allow(clippy::too_many_arguments)]
fn root_search_depth(
    thread: &mut SearchThread,
    shared: &SearchShared,
    root: &Position,
    history: &[Zobrist64],
    depth: Depth,
    moves: &[RawMove],
    alpha0: i32,
    beta0: i32,
) -> (i32, RawMove, bool, bool) {
    thread.pv_len[0] = 0;
    let mut alpha = alpha0;
    let mut best = -INFINITE;
    let mut best_move = RawMove::NULL;
    thread.ctx[0] = None;

    for (i, &m) in moves.iter().enumerate() {
        if thread.stopped {
            break;
        }
        let child = thread.make_child(root, m, shared, 1);
        thread.ctx[1] = MoveCtx::of(root, m);

        let score = if i == 0 {
            // The first root move is searched with the full window: it is the
            // move the PV will report, and its value must be exact.
            -alphabeta::alphabeta(
                &child,
                -beta0,
                -alpha,
                depth - 1,
                1,
                shared,
                thread,
                history,
                true,
                false,
            )
        } else {
            // The zero-window probe is a fail-high expectation, so its child is
            // a cut node (`sf_19/search.cpp:1417`, `!cutNode` at the root).
            let s = -alphabeta::alphabeta(
                &child,
                -alpha - 1,
                -alpha,
                depth - 1,
                1,
                shared,
                thread,
                history,
                true,
                true,
            );
            if s > alpha && s < beta0 {
                -alphabeta::alphabeta(
                    &child,
                    -beta0,
                    -alpha,
                    depth - 1,
                    1,
                    shared,
                    thread,
                    history,
                    true,
                    false,
                )
            } else {
                s
            }
        };
        if thread.stopped {
            break;
        }

        if score > best {
            best = score;
            best_move = m;
            thread.pv_push(0, m);
            if score > alpha {
                alpha = score;
            }
        }
    }

    let fail_low = best <= alpha0;
    let fail_high = best >= beta0;
    (best, best_move, fail_low, fail_high)
}

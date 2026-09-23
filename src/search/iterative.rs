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
use crate::evaluation::Evaluator;
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
    let max_depth = limits
        .depth
        .unwrap_or(MAX_DEPTH as i32)
        .min(MAX_DEPTH as i32)
        .max(0);

    thread.hashes[0] = root.hash;
    thread.ctx[0] = None;
    thread.evals[0] = Evaluator.evaluate(root);
    thread.pv_len[0] = 0;

    let soft = limits.soft_deadline();

    let mut best_move = RawMove::NULL;
    let mut best_score = -INFINITE;
    let mut prev_score = 0i32;
    let mut has_prev = false;
    let mut pv: Vec<RawMove> = Vec::new();
    let mut depth_reached = 0;

    // A helper thread with no assigned root moves has nothing to do.
    if moves.is_empty() {
        return ThreadResult {
            best: RawMove::NULL,
            score: best_score,
            depth: 0,
            nodes: shared.nodes.load(std::sync::atomic::Ordering::Relaxed),
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
            nodes: shared.nodes.load(std::sync::atomic::Ordering::Relaxed),
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
                let (s, m, fail_low, fail_high) =
                    root_search_depth(thread, shared, root, history, depth, &ordered, alpha, beta);
                if thread.stopped {
                    break (s, m);
                }
                if !fail_low && !fail_high {
                    break (s, m);
                }
                if attempts >= 3 || (alpha <= -MATE && beta >= MATE) {
                    // Give up widening and accept the current bound.
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

        best_move = mv;
        best_score = score;
        prev_score = score;
        has_prev = true;
        depth_reached = depth;

        // Finish cleanly between iterations once the budget is spent.
        if !limits.infinite {
            if let Some(soft) = soft {
                if Instant::now() >= soft {
                    break;
                }
            }
        }
    }

    ThreadResult {
        best: best_move,
        score: best_score,
        depth: depth_reached,
        nodes: shared.nodes.load(std::sync::atomic::Ordering::Relaxed),
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
        let child = root.make_child(m);
        thread.ctx[1] = MoveCtx::of(root, m);

        let score = if i == 0 {
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
            )
        } else {
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

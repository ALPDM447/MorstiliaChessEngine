//! Quiescence search.
//!
//! After the full-width search reaches depth 0, only tactically relevant
//! moves are explored: captures, promotions and — when the side to move is in
//! check — every legal evasion. The node "stands pat" at its static
//! evaluation, and only captures that can actually improve the standing are
//! searched:
//!
//! * **delta pruning** skips captures whose touched material (+
//!   [`pruning::DELTA_MARGIN`] optimism) cannot raise the stand-pat score to
//!   `alpha`;
//! * **SEE pruning** skips captures that lose material outright (negative
//!   static exchange) when not in check.
//!
//! Quiescence never writes the transposition table.

use shakmaty::Role;
use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::evaluation::Evaluator;
use crate::evaluation::material::PIECE_VALUES;
use crate::move_ordering::history::MoveCtx;
use crate::move_ordering::{order_moves_ctx, tt_move::victim_value};
use crate::search::pruning::DELTA_MARGIN;
use crate::search::{SearchShared, SearchThread, is_draw};
use crate::types::{MAX_PLY, RawMove, mated_in};

/// The value a capture "touches" for delta pruning: the captured piece (a
/// pawn for en passant) plus any promotion surplus.
#[inline]
fn touched_value(pos: &Position, m: RawMove) -> i32 {
    let victim = victim_value(pos.board(), m);
    match m.promotion() {
        Some(role) => victim + PIECE_VALUES[role as usize] - PIECE_VALUES[Role::Pawn as usize],
        None => victim,
    }
}

/// Orders a set of quiescence moves (captures only unless in check).
pub fn qsearch(
    pos: &Position,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    shared: &SearchShared,
    thread: &mut SearchThread,
    history: &[Zobrist64],
) -> i32 {
    if thread.mark_node(shared) {
        return 0;
    }
    thread.stats.qsearch_nodes += 1;
    if ply >= MAX_PLY - 1 {
        return Evaluator.evaluate(pos);
    }

    let in_check = pos.is_check();
    let static_eval = Evaluator.evaluate(pos);
    let mut moves = if in_check {
        pos.legal_moves()
    } else {
        pos.capture_moves()
    };
    if moves.is_empty() {
        // Checkmate, or simply nothing worth capturing: stand pat at the
        // static evaluation instead of pretending the position is drawn.
        return if in_check {
            mated_in(ply as i32)
        } else {
            static_eval
        };
    }

    if is_draw(pos, thread, history, ply) {
        return 0;
    }

    // Stand pat: the position's static value alone caps the downside.
    if static_eval >= beta {
        return static_eval;
    }
    if static_eval > alpha {
        alpha = static_eval;
    }
    let mut best = static_eval;

    let prev = thread.ctx[ply];
    let ant = thread.ctx[ply.saturating_sub(1)];
    // The stored TT move (if any) is the strongest cue even here — it only
    // ever reorders the generated plays, which are captures only when not in
    // check, so a quiet TT move is simply absent from the list.
    let tt_move = shared
        .tt
        .probe_move(pos.hash.into())
        .unwrap_or(RawMove::NULL);
    order_moves_ctx(&mut moves, pos, &thread.tables, tt_move, ply, prev, ant);

    for i in 0..moves.len() {
        if thread.stopped {
            return best;
        }
        let m = moves.get(i);

        // Delta pruning: the capture cannot possibly reach near alpha.
        if !in_check && best + touched_value(pos, m) + DELTA_MARGIN <= alpha {
            thread.stats.delta_pruned += 1;
            continue;
        }
        // SEE pruning: a capture that loses material is a horizon blunder,
        // not a way out of the quiet-move tail.
        if !in_check {
            thread.stats.see_calls += 1;
            if crate::move_ordering::see::see(pos.board(), m) < 0 {
                thread.stats.see_pruned += 1;
                continue;
            }
        }

        let child = pos.make_child(m);
        thread.ctx[ply + 1] = MoveCtx::of(pos, m);
        let score = -qsearch(&child, -beta, -alpha, ply + 1, shared, thread, history);
        if score > best {
            best = score;
            if score >= beta {
                break;
            }
            if score > alpha {
                alpha = score;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::search::{SearchShared, SearchThread};
    use std::sync::atomic::AtomicBool;

    fn q(fen: &str) -> i32 {
        let pos = Position::from_fen(fen).unwrap();
        let stop = AtomicBool::new(false);
        let empty_tt = crate::tt::TranspositionTable::new(1);
        let shared = SearchShared {
            tt: &empty_tt,
            stop: &stop,
            nodes: std::sync::atomic::AtomicU64::new(0),
            node_cap: None,
        };
        let mut thread = SearchThread::new();
        qsearch(&pos, -32_001, 32_001, 0, &shared, &mut thread, &[])
    }

    /// Runs qsearch with a custom (narrow) window, returning the score and the
    /// delta-prune counter so delta boundaries can be tested deterministically.
    fn q_window(fen: &str, alpha: i32, beta: i32) -> (i32, u64) {
        let pos = Position::from_fen(fen).unwrap();
        let stop = AtomicBool::new(false);
        let empty_tt = crate::tt::TranspositionTable::new(1);
        let shared = SearchShared {
            tt: &empty_tt,
            stop: &stop,
            nodes: std::sync::atomic::AtomicU64::new(0),
            node_cap: None,
        };
        let mut thread = SearchThread::new();
        let v = qsearch(&pos, alpha, beta, 0, &shared, &mut thread, &[]);
        (v, thread.stats.delta_pruned)
    }

    #[test]
    fn finds_free_queen() {
        // White to move: Rxc7 wins the undefended queen; quiescence must
        // leave white clearly ahead (a full rook/queen swing from -343).
        let v = q("6k1/2q5/8/8/8/8/8/2R3K1 w - - 0 1");
        assert!(v >= 500, "hanging queen must be found: got {v}");
    }

    #[test]
    fn avoids_losing_capture() {
        // Bxe5 (d4xe5) is a bishop-for-knight exchange defended by the f6 pawn
        // (SEE < 0); quiescence must keep the stand-pat instead of capturing.
        let fen = "6k1/8/5p2/4n3/3B4/8/8/4K3 w - - 0 1";
        let v = q(fen);
        let stand_pat = Evaluator.evaluate(&Position::from_fen(fen).unwrap());
        assert_eq!(v, stand_pat, "only capture is losing, must stand pat");
    }

    #[test]
    fn check_evenings_are_searched() {
        // Black to play, in check from the white rook on h8 along the 8th
        // rank; the king must step out (it cannot capture h8) because a
        // stand-pat would leave black down a rook.
        let v = q("4k2R/8/8/8/8/8/8/4K3 b - - 0 1");
        assert!(v < 0, "white is up a rook: got {v}");
    }

    #[test]
    fn pinned_pieces_cannot_capture() {
        // Black's knight on c6 is pinned to its king on c8 by the white rook
        // on c2: Nxd4 would open the c-file and expose the king, so it is
        // illegal. That is black's *only* capture candidate, so quiescence
        // must stand pat instead of grabbing the "free" d4 pawn.
        let fen = "2k5/8/2n5/8/3P4/8/2R5/4K3 b - - 0 1";
        let v = q(fen);
        let stand_pat = Evaluator.evaluate(&Position::from_fen(fen).unwrap());
        assert_eq!(v, stand_pat, "the pinned capture is illegal: got {v}");
    }

    #[test]
    fn en_passant_capture_is_searched() {
        // White e5 pawn can take the black d5 pawn en passant (black just
        // played d7-d5, EP square d6): quiescence must pick up the free pawn.
        let fen = "6k1/8/8/3pP3/8/8/8/6K1 w - d6 0 1";
        let v = q(fen);
        let stand_pat = Evaluator.evaluate(&Position::from_fen(fen).unwrap());
        assert!(
            v >= stand_pat + 50,
            "e.p. capture must win the pawn: {v} vs stand-pat {stand_pat}"
        );
    }

    #[test]
    fn delta_pruning_skips_captures_far_below_alpha() {
        // Black is up a bishop (≈ +330 vs white's d4 pawn) and its only
        // capture is Bxd4 (a pawn). With a window demanding ~+550, the pawn
        // capture cannot possibly bridge `best(+330) + 100 + 200` to alpha,
        // so delta pruning must skip it and stand pat. The counter proves the
        // prune happened; the returned score must equal the stand-pat.
        let (v, delta_pruned) = q_window("4k3/8/8/8/3P4/8/8/4K1b1 b - - 0 1", 550, 900);
        let stand_pat =
            Evaluator.evaluate(&Position::from_fen("4k3/8/8/8/3P4/8/8/4K1b1 b - - 0 1").unwrap());
        assert_eq!(delta_pruned, 1, "the only capture must be delta-pruned");
        assert_eq!(v, stand_pat, "stand pat: the capture was skipped");
    }
}

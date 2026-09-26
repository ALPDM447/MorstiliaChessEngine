//! Static Exchange Evaluation (SEE).
//!
//! Computes the material balance of the exchange sequence starting with the
//! move `m` on square `to`. Positive means the initiating side comes out
//! ahead; used for capture pruning in quiescence, futility decisions and move
//! ordering sanity checks.
//!
//! The implementation follows the classic "swap list" algorithm: attackers of
//! both colors are maintained and iteratively resolved with the least
//! valuable attacker, re-adding x-ray sliders as pieces vacate the line.
//! The moving piece is removed from the occupancy before the scan so it is
//! not double-counted.

use shakmaty::{Bitboard, Board, Color, Role, Square};

use crate::evaluation::params::EvalParams;
use crate::types::RawMove;

/// Returns the SEE of a capture/promotion move, or 0 for quiet moves.
///
/// Implements the classic swap-list algorithm:
///
/// * `swap[0]` is the net material won by the initial capture (victim value,
///   plus promotion surplus for promotions).
/// * Each responder in turn takes the least valuable attacker on the target
///   square; `swap[d] = value(captured at depth d) − swap[d-1]`, where the
///   "piece captured at depth d" is the piece the previous mover just placed
///   on the square (the mover's own piece for the first response).
/// * The first responder is the *opponent* (`stm = !us`); `stm` flips after
///   every simulated capture.
/// * A retrograde minimax walks the list backwards — a side only continues
///   the exchange if it is profitable — and the last speculative entry is
///   discarded; `swap[0]` is the result, positive for the mover.
///
/// Material values are read from the tunable `p` so SEE stays consistent with
/// the evaluation ('p' is the searcher's shared parameter set).
pub fn see(board: &Board, m: RawMove, p: &EvalParams) -> i32 {
    let from = m.from();
    let to = m.to();

    let Some(piece) = board.piece_at(from) else {
        return 0;
    };
    let moving = piece.role;
    if moving == Role::King {
        return 0; // king captures are quiet by convention here
    }
    let us = piece.color;

    // Occupancy used for the x-ray scans: the mover leaves its origin and the
    // captured victim leaves its square, so sliders see through afterwards.
    let mut occ = board.occupied();
    occ ^= Bitboard::from_square(from);
    if m.is_en_passant() {
        let ep = if us == Color::White {
            to.offset(-8)
        } else {
            to.offset(8)
        };
        occ ^= Bitboard::from_square(ep.unwrap_or(to));
    } else if board.role_at(to).is_some() {
        occ ^= Bitboard::from_square(to);
    }

    let promo = m.promotion();
    let pawn_value = p.piece_value(Role::Pawn);
    let victim = if m.is_en_passant() {
        pawn_value
    } else {
        match board.role_at(to) {
            Some(role) => p.piece_value(role),
            None => 0,
        }
    };

    // swap[0]: net material gained by the initial capture.
    let mut swap = [0i32; 32];
    swap[0] = victim + promo.map_or(0, |r| p.piece_value(r) - pawn_value);

    // The value the first responder would win by recapturing the piece the
    // mover just placed on `to` (its promoted form, if any).
    let mut occupant = promo.map_or(p.piece_value(moving), |r| p.piece_value(r));

    let mut attackers = attacks_to(board, to, occ);
    // The mover itself cannot act as a recapturer.
    attackers &= !Bitboard::from_square(from);

    let mut idx = 1usize;
    let mut stm = !us; // the opponent moves first after the initial capture

    while idx < swap.len() {
        let Some((sq, role)) = least_valuable_attacker(board, attackers, stm) else {
            break;
        };
        if role == Role::King {
            // The opponent only has their king left; it is never actually
            // captured, so the exchange stops here.
            break;
        }
        occ ^= Bitboard::from_square(sq);
        // X-ray sliders may appear behind the vacated square.
        attackers = (attackers & !Bitboard::from_square(sq)) | attacks_to(board, to, occ);

        swap[idx] = occupant - swap[idx - 1];
        occupant = p.piece_value(role);
        idx += 1;
        stm = !stm;
    }

    // Retrograde minimax over the swap list; the last speculative entry (the
    // unanswerable capture) is never consulted.
    while idx > 1 {
        idx -= 1;
        swap[idx - 1] = -(-swap[idx - 1]).max(swap[idx]);
    }
    swap[0]
}

/// Attackers of both colors on `to` given occupancy `occ` (includes x-rays).
fn attacks_to(board: &Board, to: Square, occ: Bitboard) -> Bitboard {
    board.attacks_to(to, Color::White, occ) | board.attacks_to(to, Color::Black, occ)
}

/// The least valuable attacker of `stm` among `attackers`. Returns the square
/// and the piece role.
fn least_valuable_attacker(
    board: &Board,
    attackers: Bitboard,
    stm: Color,
) -> Option<(Square, Role)> {
    for role in [
        Role::Pawn,
        Role::Knight,
        Role::Bishop,
        Role::Rook,
        Role::Queen,
        Role::King,
    ] {
        let candidates = attackers & board.by_piece(role.of(stm));
        if let Some(sq) = candidates.first() {
            // Pawns: only forward diagonals can actually capture `to`; the
            // attacker set already encodes that via attacks_to, so nothing
            // extra to filter here.
            return Some((sq, role));
        }
    }
    None
}

/// Convenience wrapper: is this capture winning enough to be worth searching
/// in quiescence (`see >= threshold` relative to position value)?
#[inline]
pub fn see_ge(board: &Board, m: RawMove, threshold: i32, p: &EvalParams) -> bool {
    see(board, m, p) >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;

    fn params() -> EvalParams {
        EvalParams::default()
    }

    fn sees(fen: &str, uci: &str) -> i32 {
        let pos = Position::from_fen(fen).unwrap();
        let m = pos.raw_move_from_uci(uci).unwrap();
        see(pos.board(), m, &params())
    }

    #[test]
    fn winning_exchange_is_positive() {
        // White rook b3 takes the undefended bishop on b6: net +bishop.
        let v = sees("6k1/8/1b6/8/8/1R6/8/4K3 w - - 0 1", "b3b6");
        assert!(v > 0, "rook takes free bishop: got {v}");
    }

    #[test]
    fn losing_exchange_is_negative() {
        // Bxe5, then f6xe5 — bishop (330) for knight (320) with the pawn
        // recapturing: net ~ -10.
        let v = sees("6k1/8/5p2/4n3/3B4/8/8/4K3 w - - 0 1", "d4e5");
        assert!(v < 0, "bishop for knight defended by pawn: got {v}");
    }

    #[test]
    fn queen_takes_pawn_is_positive() {
        let v = sees("6k1/8/8/3p4/3Q4/8/8/4K3 w - - 0 1", "d4d5");
        assert!(v > 0);
    }

    #[test]
    fn quiet_move_returns_zero() {
        let v = sees("6k1/8/8/8/8/8/8/4K3 w - - 0 1", "e1d2");
        assert_eq!(v, 0);
    }

    #[test]
    fn promotion_capture_counts_promoted_value() {
        // e7xd8=Q captures the rook and promotes on an undefended square: the
        // promoted queen's value dominates.
        let v = sees("3r2k1/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8q");
        assert!(v > 0, "promotion to queen on the rook square: got {v}");
    }
}

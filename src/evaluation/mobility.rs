//! Mobility evaluation.
//!
//! Counts the number of squares each knight/bishop/rook/queen attacks (minus
//! squares occupied by own pieces and, for minors, squares attacked by enemy
//! pawns, which are frequently poison), then multiplies by small per-piece
//! weights. Weight constants are deliberately conservative so that the term
//! composes well with the other components.

use shakmaty::{Bitboard, Board, Color, Role, Square, attacks};

use crate::evaluation::params::EvalParams;
use crate::evaluation::{Score, pawns::PawnInfo};

/// (midgame, endgame) centipawns per extra mobility square, read from `p`
/// (roles 1..=5; the king's endgame-only activity lives in the same table).
fn mobility_weight(p: &EvalParams, role: Role) -> (i32, i32) {
    p.mobility_weight(role)
}

pub fn evaluate_mobility(board: &Board, info: &PawnInfo, p: &EvalParams) -> Score {
    let occupied = board.occupied();
    let mut score = Score::zero();

    for color in [Color::White, Color::Black] {
        let own = board.by_color(color);
        let sign = if color == Color::White { 1 } else { -1 };
        let enemy_pawn_attacks = info.pawn_attacks_of(!color);

        let mut color_score = Score::zero();
        for role in [
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
            Role::King,
        ] {
            let (w_mg, w_eg) = mobility_weight(p, role);
            board.by_piece(role.of(color)).for_each(|sq| {
                let attack_bb = attacks_for(role, sq, occupied);
                // Exclude own pieces. Minors also avoid enemy-pawn-attacked
                // squares.
                let usable = if role == Role::Rook || role == Role::Queen {
                    attack_bb & !own
                } else {
                    attack_bb & !own & !enemy_pawn_attacks
                };
                let count = usable.count() as i32;
                color_score.mg += count * w_mg;
                color_score.eg += count * w_eg;
            });
        }

        score.mg += sign * color_score.mg;
        score.eg += sign * color_score.eg;
    }

    score
}

#[inline]
fn attacks_for(role: Role, sq: Square, occupied: Bitboard) -> Bitboard {
    match role {
        Role::Knight => attacks::knight_attacks(sq),
        Role::Bishop => attacks::bishop_attacks(sq, occupied),
        Role::Rook => attacks::rook_attacks(sq, occupied),
        Role::Queen => attacks::queen_attacks(sq, occupied),
        Role::King => attacks::king_attacks(sq),
        _ => Bitboard(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    fn mob(fen: &str) -> Score {
        let board = crate::testutil::chess(fen).board().clone();
        evaluate_mobility(&board, &PawnInfo::scan(&board), &EvalParams::default())
    }

    #[test]
    fn central_knight_is_more_mobile() {
        let s_corner = mob("6k1/8/8/8/8/8/8/N3K3 w - - 0 1");
        let s_center = mob("6k1/8/8/8/3N4/8/8/4K3 w - - 0 1");
        assert!(s_center.mg > s_corner.mg);
    }

    #[test]
    fn blocked_knight_loses_mobility() {
        let s_open = mob("6k1/8/8/8/8/8/8/4K1N1 w - - 0 1");
        let s_blocked = mob("6k1/8/8/8/8/8/PPPPPPPP/4K1N1 w - - 0 1");
        assert!(s_open.mg > s_blocked.mg);
    }

    #[test]
    fn endgame_king_mobility_is_rewarded() {
        // Same material (K vs K), the only difference is the white king's
        // position. A king in the centre has more mobility than one in the
        // corner — the endgame term must see it.
        let s_corner = mob("7k/8/8/8/8/8/8/K7 w - - 0 1");
        let s_central = mob("7k/8/8/8/8/8/8/4K3 w - - 0 1");
        assert!(
            s_central.eg > s_corner.eg,
            "centralized king must have more endgame mobility: {s_central:?} vs {s_corner:?}"
        );
    }
}

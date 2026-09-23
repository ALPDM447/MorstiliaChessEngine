//! Material evaluation: piece values and the bishop pair.

use shakmaty::{Board, Color, Role};

use crate::evaluation::Score;

/// Centipawn value of each piece, indexed directly by `Role as usize`
/// (`Role::Pawn == 1` … `Role::King == 6`). The king has value 0 here
/// (it is never traded); SEE includes a synthetic large king value.
pub const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];

/// Bonus for having both bishops (tapered).
const BISHOP_PAIR: Score = Score::new(40, 60);

/// Material score (white minus black).
pub fn evaluate_material(board: &Board) -> Score {
    let mut score = Score::zero();
    let mut white_bishops = 0i32;
    let mut black_bishops = 0i32;

    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        for role in [
            Role::Pawn,
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
        ] {
            let count = board.by_piece(role.of(color)).count() as i32;
            score.mg += sign * count * PIECE_VALUES[role as usize];
            score.eg += sign * count * PIECE_VALUES[role as usize];
        }
        if board.by_piece(Role::Bishop.of(color)).count() >= 2 {
            if color == Color::White {
                white_bishops = 1;
            } else {
                black_bishops = 1;
            }
        }
    }

    if white_bishops != black_bishops {
        score += if white_bishops > black_bishops {
            BISHOP_PAIR
        } else {
            -BISHOP_PAIR
        };
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    fn eval_of(fen: &str) -> Score {
        let chess = crate::testutil::chess(fen);
        evaluate_material(chess.board())
    }

    #[test]
    fn startpos_material_is_zero() {
        assert_eq!(eval_of(&crate::testutil::startpos_fen()), Score::zero());
    }

    #[test]
    fn extra_knight_is_320() {
        let s = eval_of("6k1/8/8/8/8/8/8/4K1N1 w - - 0 1");
        assert_eq!(s.mg, 320);
        assert_eq!(s.eg, 320);
    }

    #[test]
    fn bishop_pair_awarded() {
        let both = eval_of("6k1/8/8/8/8/8/8/2B1KB2 w - - 0 1");
        let single = eval_of("6k1/8/8/8/8/8/8/2B1K3 w - - 0 1");
        // White bishop pair: 2*330 + 40 (mg) vs 330 → difference = 330 + 40
        assert_eq!(both.mg - single.mg, 330 + 40);
    }
}

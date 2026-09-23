//! Piece-square tables (tapered).
//!
//! Tables are given for White on square indices `A1 == 0 … H8 == 63` and
//! mirrored (vertically) for Black. Positive values favour White.

use shakmaty::{Board, Color, Role, Square};

use crate::evaluation::Score;

type Table = [i16; 64];

/// Pawn PSTs. Values are in centipawns; index 0 = rank 1 file a … 63 = rank 8.
const PAWN_MG: Table = [
    0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5,
    10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20,
    -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
];

const PAWN_EG: Table = [
    0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 50, 50, 50, 50, 50, 50, 50, 50, 30, 30,
    30, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_MG: Table = [
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10,
    0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10,
    5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
];

const BISHOP_MG: Table = [
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10,
    -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
];

const ROOK_MG: Table = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
    0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0,
    0, 5, 5, 0, 0, 0,
];

const QUEEN_MG: Table = [
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0,
    0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
];

const KING_MG: Table = [
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40,
    -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30,
    -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0,
    10, 30, 20,
];

const KING_EG: Table = [
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20, 30,
    30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30,
    -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30,
    -50, -50,
];

/// Minors/rooks/queens use the same table for midgame and endgame (the
/// simplified-evaluation convention).
fn king_eg_table() -> Table {
    KING_EG
}

/// Returns the PST index for `sq` seen from `color`'s point of view.
#[inline]
fn table_index(sq: Square, color: Color) -> usize {
    if color == Color::White {
        sq.to_usize()
    } else {
        sq.flip_vertical().to_usize()
    }
}

/// Piece-square score for the whole board (white minus black).
pub fn evaluate_pst(board: &Board) -> Score {
    let mut score = Score::zero();
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        for role in [
            Role::Pawn,
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
            Role::King,
        ] {
            let (mg, eg): (&Table, Table) = match role {
                Role::Pawn => (&PAWN_MG, PAWN_EG),
                Role::Knight => (&KNIGHT_MG, KNIGHT_MG),
                Role::Bishop => (&BISHOP_MG, BISHOP_MG),
                Role::Rook => (&ROOK_MG, ROOK_MG),
                Role::Queen => (&QUEEN_MG, QUEEN_MG),
                Role::King => (&KING_MG, king_eg_table()),
            };
            board.by_piece(role.of(color)).for_each(|sq| {
                let idx = table_index(sq, color);
                score.mg += sign * i32::from(mg[idx]);
                score.eg += sign * i32::from(eg[idx]);
            });
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    #[test]
    fn centered_knight_beats_corner_knight() {
        let a = crate::testutil::chess("6k1/8/8/8/8/8/8/4K3 w - - 0 1")
            .board()
            .clone();
        let b = crate::testutil::chess("6k1/8/8/8/3N4/8/8/4K3 w - - 0 1")
            .board()
            .clone();
        let corner = crate::testutil::chess("6k1/8/8/8/8/8/8/N3K3 w - - 0 1")
            .board()
            .clone();
        let s_center = evaluate_pst(&b);
        let s_corner = evaluate_pst(&corner);
        let s_none = evaluate_pst(&a);
        assert!(s_center.mg > s_corner.mg);
        assert!(s_center.mg > s_none.mg);
    }

    #[test]
    fn pst_black_mirror_negates() {
        // A position and its true color mirror (colors swapped + board
        // flipped vertically) must produce opposite PST values.
        let w = crate::testutil::chess("6k1/8/8/8/4P3/8/4N3/4K3 w - - 0 1")
            .board()
            .clone();
        let m = crate::testutil::chess("4k3/4n3/8/4p3/8/8/8/6K1 b - - 0 1")
            .board()
            .clone();
        let s_w = evaluate_pst(&w);
        let s_m = evaluate_pst(&m);
        assert_eq!(s_w.mg, -s_m.mg);
        assert_eq!(s_w.eg, -s_m.eg);
    }
}

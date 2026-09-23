//! Game phase and tapered-score blending.
//!
//! Phase runs from 24 (opening, all pieces present) down to 0 (endgame).
//! Each queen contributes 4, each rook 2, each bishop or knight 1. The search
//! and evaluation both use [`crate::evaluation::tapered`] to blend midgame
//! and endgame terms.

use shakmaty::{Board, Color};

use crate::evaluation::Score;

/// Phase when every piece is still on the board.
pub const PHASE_MAX: i32 = 24;

/// Computes the game phase of `board`.
pub fn game_phase(board: &Board) -> i32 {
    let mut phase = 0i32;
    for color in [Color::White, Color::Black] {
        phase += 4 * (board.by_piece(shakmaty::Role::Queen.of(color)).count() as i32);
        phase += 2 * (board.by_piece(shakmaty::Role::Rook.of(color)).count() as i32);
        phase += board.by_piece(shakmaty::Role::Bishop.of(color)).count() as i32;
        phase += board.by_piece(shakmaty::Role::Knight.of(color)).count() as i32;
    }
    phase.clamp(0, PHASE_MAX)
}

/// Scales `score` according to `phase` and returns the blended value.
pub fn blend(score: Score, phase: i32) -> i32 {
    crate::evaluation::tapered(score, phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    #[test]
    fn opening_is_phase_max() {
        let chess = shakmaty::Chess::default();
        let board = chess.board();
        assert_eq!(game_phase(board), PHASE_MAX);
    }

    #[test]
    fn pawn_endgame_is_phase_zero() {
        let board = crate::testutil::chess("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
            .board()
            .clone();
        assert_eq!(game_phase(&board), 0);
    }

    #[test]
    fn queen_endgame_phase() {
        let board = crate::testutil::chess("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1")
            .board()
            .clone();
        assert_eq!(game_phase(&board), 4);
    }
}

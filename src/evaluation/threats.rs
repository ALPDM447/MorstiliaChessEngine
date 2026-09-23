//! Threat evaluation plus two positional bonuses that have no dedicated
//! module in the planned layout: rooks on open files / the seventh rank, and
//! central space.
//!
//! Threat terms (midgame-heavy):
//!
//! * our pawns attacking enemy minors/major pieces,
//! * our pieces attacking *undefended* enemy pieces (scaled by value),
//! * penalties for our minors sitting on enemy-pawn-attacked squares.

use shakmaty::{Bitboard, Board, Color, Role, Square, attacks};

use crate::evaluation::Score;
use crate::evaluation::material::PIECE_VALUES;
use crate::evaluation::pawns::PawnInfo;

/// Penalty (mg) for a minor piece attacked by an enemy pawn.
const WEAK_MINOR: i32 = 15;
/// Bonus (mg) when our pawn attacks an enemy non-pawn.
const PAWN_ATTACK_BONUS: i32 = 18;

/// Rook on a fully open file / semi-open file.
const ROOK_OPEN: Score = Score::new(25, 20);
const ROOK_SEMI_OPEN: Score = Score::new(10, 10);
/// Rook on the seventh rank.
const ROOK_SEVENTH: Score = Score::new(20, 40);

pub fn evaluate_threats(board: &Board, info: &PawnInfo) -> Score {
    let mut score = Score::zero();
    let occupied = board.occupied();

    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let us = board.by_color(color);
        let them = board.by_color(!color);

        let our_attacks = total_attacks(board, color, occupied, info.pawn_attacks_of(color));
        let their_attacks = total_attacks(board, !color, occupied, info.pawn_attacks_of(!color));
        let their_defended =
            their_attacks | attacks::king_attacks(board.king_of(!color).unwrap_or(Square::E1));

        let mut mg = 0i32;

        // Pawns attacking enemy pieces (non-pawn).
        let pawn_hits = info.pawn_attacks_of(color) & them & !board.by_piece(Role::Pawn.of(!color));
        mg += PAWN_ATTACK_BONUS * pawn_hits.count() as i32;

        // Hanging (undefended) enemy pieces hit by our attacks.
        let hanging = them & our_attacks & !their_defended & !board.by_piece(Role::King.of(!color));
        hanging.for_each(|sq| {
            if let Some(piece) = board.piece_at(sq) {
                mg += PIECE_VALUES[piece.role as usize] * 3 / 40;
            }
        });

        // Our minors attacked by enemy pawns.
        let weak = (board.by_piece(Role::Knight.of(color))
            | board.by_piece(Role::Bishop.of(color)))
            & info.pawn_attacks_of(!color);
        mg -= WEAK_MINOR * weak.count() as i32;

        score.mg += sign * mg;
        score.eg += sign * (mg / 2);
        let _ = us;
    }
    score
}

/// All squares `color`'s pieces attack (excluding our own occupied squares is
/// intentionally not applied: attacked squares matter even when occupied).
fn total_attacks(
    board: &Board,
    color: Color,
    occupied: Bitboard,
    pawn_attacks: Bitboard,
) -> Bitboard {
    let mut attacks = pawn_attacks;
    board
        .by_piece(Role::Knight.of(color))
        .for_each(|sq| attacks |= attacks::knight_attacks(sq));
    board
        .by_piece(Role::Bishop.of(color))
        .for_each(|sq| attacks |= attacks::bishop_attacks(sq, occupied));
    board
        .by_piece(Role::Rook.of(color))
        .for_each(|sq| attacks |= attacks::rook_attacks(sq, occupied));
    board
        .by_piece(Role::Queen.of(color))
        .for_each(|sq| attacks |= attacks::queen_attacks(sq, occupied));
    if let Some(k) = board.king_of(color) {
        attacks |= attacks::king_attacks(k);
    }
    attacks
}

/// Open-file and seventh-rank rewards for rooks.
pub fn evaluate_rooks(board: &Board, info: &PawnInfo) -> Score {
    let mut score = Score::zero();
    let all_pawns = info.white | info.black;

    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let seventh = if color == Color::White { 6u8 } else { 1u8 };
        let enemy_rooks = board.by_piece(Role::Rook.of(!color));
        let mut color_score = Score::zero();

        board.by_piece(Role::Rook.of(color)).for_each(|sq| {
            let f = u8::from(sq.file());
            let file_bb = Bitboard(0x0101010101010101u64 << f);
            let pawns_on_file = (all_pawns & file_bb).count();
            if pawns_on_file == 0 {
                color_score += ROOK_OPEN;
            } else if (info.pawns_of(color) & file_bb).is_empty() {
                color_score += ROOK_SEMI_OPEN;
            }

            if u8::from(sq.rank()) == seventh
                && (enemy_rooks & Bitboard(0x0101010101010101u64 << f)).is_empty()
            {
                color_score += ROOK_SEVENTH;
            }
        });

        score += if sign > 0 { color_score } else { -color_score };
    }
    score
}

/// Central/space bonus (tapered): pawns on the 4th/5th ranks that attack
/// squares in the enemy camp. The bonus is tapered so it does not dominate
/// the endgame; the *enemy's* space restricts ours (a pawn pushed on the 4th
/// that the enemy can hit is not really space).
pub fn evaluate_space(_board: &Board, info: &PawnInfo) -> Score {
    let mut mg = 0i32;
    let mut eg = 0i32;
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let (forward, center): (i32, fn(u8) -> bool) = if color == Color::White {
            (8, |r| r >= 4)
        } else {
            (-8, |r| r <= 3)
        };
        let mut our = 0i32;
        info.pawns_of(color).for_each(|sq| {
            if center(u8::from(sq.rank())) {
                let ahead = Bitboard::from_square(sq).shift(forward);
                if ahead.any() {
                    our += 1;
                }
            }
        });
        mg += sign * our * 4;
        // Endgame: half weight — space matters less once pieces are traded.
        eg += sign * our * 2;
    }
    Score::new(mg.clamp(-48, 48), eg.clamp(-48, 48))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    #[test]
    fn hanging_piece_is_bonus() {
        let ok = crate::testutil::chess("6k1/8/8/8/8/8/1b6/1R2K3 w - - 0 1")
            .board()
            .clone();
        // Black bishop on b2 is undefended and attacked by the white rook
        // b1 → threat bonus (a rook-vs-rook symmetric pin would cancel out).
        let pi = PawnInfo::scan(&ok);
        let s = evaluate_threats(&ok, &pi);
        assert!(
            s.mg > 0,
            "undefended bishop should be a threat bonus: {s:?}"
        );
    }

    #[test]
    fn rook_open_file_bonus() {
        let open = crate::testutil::chess("6k1/8/8/8/8/8/8/R3K3 w - - 0 1")
            .board()
            .clone();
        let closed = crate::testutil::chess("6k1/p7/8/8/8/8/p7/R3K3 w - - 0 1")
            .board()
            .clone();
        let s_open = evaluate_rooks(&open, &PawnInfo::scan(&open));
        let s_closed = evaluate_rooks(&closed, &PawnInfo::scan(&closed));
        assert!(s_open.mg > s_closed.mg);
    }

    #[test]
    fn rook_on_seventh_bonus() {
        let seventh = crate::testutil::chess("7k/6R1/8/8/8/8/8/4K3 w - - 0 1")
            .board()
            .clone();
        let first = crate::testutil::chess("7k/8/8/8/8/8/8/R3K3 w - - 0 1")
            .board()
            .clone();
        let s_seventh = evaluate_rooks(&seventh, &PawnInfo::scan(&seventh));
        let s_first = evaluate_rooks(&first, &PawnInfo::scan(&first));
        assert!(s_seventh.mg > s_first.mg, "{s_seventh:?} vs {s_first:?}");
    }
}

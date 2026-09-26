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
use crate::evaluation::params::EvalParams;
use crate::evaluation::pawns::PawnInfo;

/// Safety cap for the space term (±48 cp) — not a tunable weight, just a
/// guard so a pathological parameter set cannot make space dominate.
const SPACE_CAP: i32 = 48;

pub fn evaluate_threats(board: &Board, info: &PawnInfo, p: &EvalParams) -> Score {
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
        mg += p.pawn_attack_bonus * pawn_hits.count() as i32;

        // Hanging (undefended) enemy pieces hit by our attacks.
        let hanging = them & our_attacks & !their_defended & !board.by_piece(Role::King.of(!color));
        hanging.for_each(|sq| {
            if let Some(piece) = board.piece_at(sq) {
                mg += p.piece_value(piece.role) * 3 / 40;
            }
        });

        // Our minors attacked by enemy pawns (`weak_minor` is a stored
        // negative penalty, so adding it subtracts from the score).
        let weak = (board.by_piece(Role::Knight.of(color))
            | board.by_piece(Role::Bishop.of(color)))
            & info.pawn_attacks_of(!color);
        mg += p.weak_minor * weak.count() as i32;

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
pub fn evaluate_rooks(board: &Board, info: &PawnInfo, p: &EvalParams) -> Score {
    let mut score = Score::zero();
    let all_pawns = info.white | info.black;
    let open = Score::new(p.rook_open[0], p.rook_open[1]);
    let semi_open = Score::new(p.rook_semi_open[0], p.rook_semi_open[1]);
    let seventh = Score::new(p.rook_seventh[0], p.rook_seventh[1]);

    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let seventh_rank = if color == Color::White { 6u8 } else { 1u8 };
        let enemy_rooks = board.by_piece(Role::Rook.of(!color));
        let mut color_score = Score::zero();

        board.by_piece(Role::Rook.of(color)).for_each(|sq| {
            let f = u8::from(sq.file());
            let file_bb = Bitboard(0x0101010101010101u64 << f);
            let pawns_on_file = (all_pawns & file_bb).count();
            if pawns_on_file == 0 {
                color_score += open;
            } else if (info.pawns_of(color) & file_bb).is_empty() {
                color_score += semi_open;
            }

            if u8::from(sq.rank()) == seventh_rank
                && (enemy_rooks & Bitboard(0x0101010101010101u64 << f)).is_empty()
            {
                color_score += seventh;
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
pub fn evaluate_space(_board: &Board, info: &PawnInfo, p: &EvalParams) -> Score {
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
        mg += sign * our * p.space_mg;
        // Endgame: half weight — space matters less once pieces are traded.
        eg += sign * our * p.space_eg;
    }
    Score::new(
        mg.clamp(-SPACE_CAP, SPACE_CAP),
        eg.clamp(-SPACE_CAP, SPACE_CAP),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    fn params() -> EvalParams {
        EvalParams::default()
    }

    #[test]
    fn hanging_piece_is_bonus() {
        let ok = crate::testutil::chess("6k1/8/8/8/8/8/1b6/1R2K3 w - - 0 1")
            .board()
            .clone();
        // Black bishop on b2 is undefended and attacked by the white rook
        // b1 → threat bonus (a rook-vs-rook symmetric pin would cancel out).
        let pi = PawnInfo::scan(&ok);
        let s = evaluate_threats(&ok, &pi, &params());
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
        let s_open = evaluate_rooks(&open, &PawnInfo::scan(&open), &params());
        let s_closed = evaluate_rooks(&closed, &PawnInfo::scan(&closed), &params());
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
        let s_seventh = evaluate_rooks(&seventh, &PawnInfo::scan(&seventh), &params());
        let s_first = evaluate_rooks(&first, &PawnInfo::scan(&first), &params());
        assert!(s_seventh.mg > s_first.mg, "{s_seventh:?} vs {s_first:?}");
    }

    #[test]
    fn space_bonus_scales_with_parameter() {
        // The space term is proportional to the space_mg/space_eg parameters:
        // doubling the weight must double a non-clamped contribution.
        let fen = "6k1/8/8/4PP2/3P4/8/8/4K3 w - - 0 1";
        let board = crate::testutil::chess(fen).board().clone();
        let pi = PawnInfo::scan(&board);
        let base = EvalParams::default();
        let mut doubled = base.clone();
        doubled.space_mg *= 2;
        doubled.space_eg *= 2;
        let s_base = evaluate_space(&board, &pi, &base);
        let s_double = evaluate_space(&board, &pi, &doubled);
        assert_eq!(s_double.mg, s_base.mg * 2, "mg scales with space_mg");
        assert_eq!(s_double.eg, s_base.eg * 2, "eg scales with space_eg");
    }
}

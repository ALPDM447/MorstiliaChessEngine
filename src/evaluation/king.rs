//! King safety (midgame-oriented).
//!
//! Four complementary terms:
//!
//! * **Pawn shield**: own pawns in the three files in front of the king are
//!   worth more the closer and the more central they are.
//! * **Attack units**: enemy pieces attacking the squares around the king
//!   (the "king ring") accumulate a midgame penalty that grows quadratically
//!   with the number of attackers.
//! * **Open files near the king**: a file among the king's three that carries
//!   no own pawn gives the enemy's rooks/queen a highway to the king — a
//!   small standing penalty (fully open files cost a little extra).
//! * **Pawn holes (weak squares)**: ring squares that an enemy pawn attacks
//!   but no own pawn can ever defend are holes — an invasion point right by
//!   the king.

use shakmaty::{Bitboard, Board, Color, Role, Square, attacks};

use crate::evaluation::Score;
use crate::evaluation::passed_pawns::is_passed;
use crate::evaluation::pawns::PawnInfo;

/// Max shield bonus per king (verhindert absurd stacks).
const SHIELD_CAP: i32 = 40;

/// Attack weight per enemy piece type aimed at the king ring.
const ATTACK_WEIGHT: [i32; 7] = [0, 2, 3, 3, 4, 5, 0]; // P,N,B,R,Q,K

/// Penalty per king file with no own pawn (mg).
const KING_OPEN_FILE: i32 = 8;
/// Extra penalty when the file has no pawns at all (fully open).
const KING_FULLY_OPEN_FILE: i32 = 5;
/// Cap so a king on a filed edge cannot be drained by this term alone.
const KING_OPEN_FILES_CAP: i32 = 30;

/// Penalty per ring square an enemy pawn attacks that our pawns can never
/// defend (mg).
const KING_HOLE: i32 = 6;
/// Cap for the holes term.
const KING_HOLES_CAP: i32 = 30;

/// Endgame: bonus per step the king moves toward the center (Chebyshev
/// distance from the closest edge). Centralized kings control squares and
/// support passed pawns; the term is endgame-only so it does not fight the
/// midgame king-safety penalties.
const KING_CENTER_BONUS: i32 = 3;
/// Endgame: penalty per step the king is from the promotion files (d/e) when
/// the enemy has a passed pawn — the king must race to the promotion square.
const KING_PASSER_DIST_PENALTY: i32 = 4;

/// Evaluates king safety for both colors (white minus black).
///
/// Midgame terms (shield, danger, open files, holes) are mg-only; the
/// endgame adds king centralization (the king is a fighting piece once the
/// queens are off) and a passed-pawn race adjustment.
pub fn evaluate_king_safety(board: &Board, info: &PawnInfo) -> Score {
    let mut mg = 0i32;
    let mut eg = 0i32;
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };

        let king = match board.king_of(color) {
            Some(k) => k,
            None => continue, // defensive: never happens in legal chess
        };

        let shield = shield_bonus(info, color, king);
        let danger = king_danger(board, info, color, king);
        let open_files = open_files_near_king(info, color, king);
        let holes = king_holes(info, color, king);

        mg += sign * (shield - danger - open_files - holes);
        eg += sign * king_endgame_bonus(board, info, color, king);
    }
    Score::new(mg, eg)
}

/// Sum of pawn-shield bonuses for `color`'s king on `king`.
fn shield_bonus(info: &PawnInfo, color: Color, king: Square) -> i32 {
    let up = if color == Color::White { 8i32 } else { -8i32 };
    let king_file = u8::from(king.file());

    let mut bonus = 0i32;
    let mut dist = 1i32;
    let mut ahead = Bitboard::from_square(king).shift(up);
    while ahead.any() && dist <= 3 {
        // Three files around the king's own file.
        let mask = if king_file > 0 {
            file_of(king_file - 1)
        } else {
            0
        } | file_of(king_file)
            | if king_file < 7 {
                file_of(king_file + 1)
            } else {
                0
            };
        let shielded = (info.pawns_of(color) & ahead & Bitboard(mask)).count() as i32;
        bonus += shielded * (5 - dist); // dist 1 → 4, 2 → 3, 3 → 2 per pawn
        ahead = ahead.shift(up);
        dist += 1;
    }
    bonus.min(SHIELD_CAP)
}

/// Open-file penalties for `color`'s king on `king`: among the king's own
/// file and its two neighbours, an absent own pawn leaves the file open for
/// enemy majors. Fully open files (no pawns at all) cost a bit more.
fn open_files_near_king(info: &PawnInfo, color: Color, king: Square) -> i32 {
    let kf = u8::from(king.file());
    let own = info.files_of(color);
    let all = info.files_of(Color::White) | info.files_of(Color::Black);
    let mut penalty = 0i32;
    for f in kf.saturating_sub(1)..=kf.min(6) + 1 {
        if own & (1 << f) == 0 {
            penalty += KING_OPEN_FILE;
            if all & (1 << f) == 0 {
                penalty += KING_FULLY_OPEN_FILE;
            }
        }
    }
    penalty.min(KING_OPEN_FILES_CAP)
}

/// Weak-square (hole) penalty: ring squares attacked by an enemy pawn that
/// no own pawn attacks — our pawn phalanx cannot recover those squares, so
/// they are permanent invasion points.
fn king_holes(info: &PawnInfo, color: Color, king: Square) -> i32 {
    let ring = ring_of(king, color);
    let enemy_pawn_hits = info.pawn_attacks_of(!color) & ring;
    let undefendable = enemy_pawn_hits & !info.pawn_attacks_of(color);
    (KING_HOLE * undefendable.count() as i32).min(KING_HOLES_CAP)
}

/// Endgame king activity: centralization plus a passed-pawn race adjustment.
/// The king is a fighting piece once the queens are off the board, so it
/// should head for the centre and, when an enemy passer exists, toward the
/// promotion square.
fn king_endgame_bonus(_board: &Board, info: &PawnInfo, color: Color, king: Square) -> i32 {
    let mut bonus = 0i32;

    // Centralization: reward proximity to the centre (d4-e4-d5-e5).
    let kf = i32::from(king.file());
    let kr = i32::from(king.rank());
    let center_dist = (kf - 3).abs().max((kf - 4).abs()) + (kr - 3).abs().max((kr - 4).abs());
    bonus += (4 - center_dist) * KING_CENTER_BONUS;

    // Passed-pawn race: if the enemy has a passer, the king wants to be near
    // the promotion file; the closer the king, the smaller the penalty.
    let enemy_pawns = info.pawns_of(!color);
    let mut race_penalty = 0i32;
    enemy_pawns.for_each(|esq| {
        if is_passed(esq, !color, info.pawns_of(color)) {
            let ef = i32::from(esq.file());
            let promo_dist = (ef - kf).abs();
            race_penalty += promo_dist * KING_PASSER_DIST_PENALTY;
        }
    });
    bonus -= race_penalty;

    bonus
}

/// Midgame penalty for enemy attack pressure on `color`'s king.
fn king_danger(board: &Board, info: &PawnInfo, color: Color, king: Square) -> i32 {
    let occupied = board.occupied();
    let ring = ring_of(king, color);

    let mut units = 0i32;
    let mut attackers_bb = Bitboard(0);
    for role in [
        Role::Pawn,
        Role::Knight,
        Role::Bishop,
        Role::Rook,
        Role::Queen,
    ] {
        let bb = board.by_piece(role.of(!color));
        if bb.is_empty() {
            continue;
        }
        let attack_bb = match role {
            Role::Pawn => info.pawn_attacks_of(!color),
            Role::Knight => {
                let mut a = Bitboard(0);
                bb.for_each(|sq| a |= attacks::knight_attacks(sq));
                // Restrict to the ring region to bound the cost.
                a & (ring | attacks::king_attacks(ring.first().unwrap_or(king)))
            }
            Role::Bishop => {
                let mut a = Bitboard(0);
                bb.for_each(|sq| a |= attacks::bishop_attacks(sq, occupied));
                a
            }
            Role::Rook => {
                let mut a = Bitboard(0);
                bb.for_each(|sq| a |= attacks::rook_attacks(sq, occupied));
                a
            }
            Role::Queen => {
                let mut a = Bitboard(0);
                bb.for_each(|sq| a |= attacks::queen_attacks(sq, occupied));
                a
            }
            _ => Bitboard(0),
        };
        let hitting = attack_bb & ring;
        if hitting.any() {
            let toward = if color == Color::White { 8i32 } else { -8i32 };
            attackers_bb |= bb & attacks::king_attacks(king).shift(toward);
            attackers_bb |= bb & attacks::king_attacks(king);
            units += ATTACK_WEIGHT[role as usize] * hitting.count() as i32;
        }
    }

    // Ignore tiny pressure.
    if units <= 0 {
        return 0;
    }
    // Quadratic-ish growth: two attackers hurt far more than one.
    let attacker_count = attackers_bb.count() as i32;
    let mut danger = units;
    if attacker_count >= 2 {
        danger += units * (attacker_count / 2);
    }
    danger
}

#[inline]
fn file_of(f: u8) -> u64 {
    0x0101010101010101u64 << f
}

/// The king ring: squares the king attacks plus the rank in front of it.
fn ring_of(king: Square, color: Color) -> Bitboard {
    let toward = if color == Color::White { 8i32 } else { -8i32 };
    attacks::king_attacks(king) | attacks::king_attacks(king).shift(toward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    #[test]
    fn shielded_king_scores_higher() {
        // Same bare position; the king on g1 keeps its f/g/h pawn shield in
        // one variant and none in the other.
        let shielded = crate::testutil::chess("6k1/8/8/8/8/8/5PPP/6K1 w - - 0 1")
            .board()
            .clone();
        let exposed = crate::testutil::chess("6k1/8/8/8/8/8/8/6K1 w - - 0 1")
            .board()
            .clone();
        let s1 = evaluate_king_safety(&shielded, &PawnInfo::scan(&shielded));
        let s2 = evaluate_king_safety(&exposed, &PawnInfo::scan(&exposed));
        assert!(s1.mg > s2.mg, "shielded should be safer than unshielded");
    }

    #[test]
    fn surrounded_king_is_penalized() {
        // Black pieces swarming the white king ring.
        let fen = "6k1/8/8/3nn3/3nr3/8/8/4K3 w - - 0 1";
        let chess = crate::testutil::chess(fen);
        let board = chess.board().clone();
        let s = evaluate_king_safety(&board, &PawnInfo::scan(&board));
        assert!(s.mg < 0, "white king under heavy attack: {s:?}");
    }

    #[test]
    fn endgame_king_centralization_is_rewarded() {
        // Same material (K vs K), the only difference is the white king's
        // position. A centralized king must score higher in the endgame term.
        let corner = crate::testutil::chess("7k/8/8/8/8/8/8/K7 w - - 0 1")
            .board()
            .clone();
        let central = crate::testutil::chess("7k/8/8/8/8/8/8/4K3 w - - 0 1")
            .board()
            .clone();
        let s_corner = evaluate_king_safety(&corner, &PawnInfo::scan(&corner));
        let s_central = evaluate_king_safety(&central, &PawnInfo::scan(&central));
        assert!(
            s_central.eg > s_corner.eg,
            "centralized king must get an endgame bonus: {s_central:?} vs {s_corner:?}"
        );
    }

    #[test]
    fn endgame_king_races_toward_passer() {
        // White king far from the black passed e-pawn vs close to it. The
        // far king must be penalized more in the endgame term.
        let far = crate::testutil::chess("7k/8/8/8/4p3/8/8/K7 w - - 0 1")
            .board()
            .clone();
        let near = crate::testutil::chess("7k/8/8/8/4p3/8/8/4K3 w - - 0 1")
            .board()
            .clone();
        let s_far = evaluate_king_safety(&far, &PawnInfo::scan(&far));
        let s_near = evaluate_king_safety(&near, &PawnInfo::scan(&near));
        assert!(
            s_near.eg > s_far.eg,
            "king near the enemy passer must score higher: {s_near:?} vs {s_far:?}"
        );
    }
}

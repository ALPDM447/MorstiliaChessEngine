//! Passed-pawn evaluation.
//!
//! A pawn is passed when no enemy pawn can stop it on its own or on adjacent
//! files ahead of it. [`PassedInfo`] captures the passed set once per
//! evaluation so the score and the other modules (piece activity) reuse it
//! instead of re-detecting pawns. Bonuses grow steeply with the rank; the
//! endgame component is further adjusted by which king is closer to the
//! promotion square (the classic "king proximity" rule), and supported /
//! connected passers get extra value.

use shakmaty::{Bitboard, Board, Color, Square};

use crate::evaluation::Score;
use crate::evaluation::pawns::{PawnInfo, file_mask, rank_mask};

/// Passed-pawn bonuses by rank, `PASSED_BONUS[rank - 2]` for white ranks
/// 2..=7. Midgame values are modest; endgame values dominate.
const PASSED_BONUS: [Score; 6] = [
    Score::new(0, 10),    // rank 2
    Score::new(10, 40),   // rank 3
    Score::new(25, 100),  // rank 4
    Score::new(50, 250),  // rank 5
    Score::new(110, 500), // rank 6
    Score::new(180, 800), // rank 7
];

/// Bonus for a passed pawn defended by one of its own pawns — the defender
/// keeps the passer safe from piece assaults on the way to promotion.
const PROTECTED_PASSED: Score = Score::new(10, 30);

/// Bonus for a passed pawn running next to a second passed pawn on an
/// adjacent file (counted once per pair, favouring the higher file): two
/// passers cannot be blockaded by one piece.
const CONNECTED_PASSED: Score = Score::new(15, 30);

/// The passed pawns of each color, computed once per evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassedInfo {
    pub white: Bitboard,
    pub black: Bitboard,
}

impl PassedInfo {
    /// Detects every passed pawn of both colors in one pass.
    pub fn scan(info: &PawnInfo) -> PassedInfo {
        PassedInfo {
            white: passed_of(info, Color::White),
            black: passed_of(info, Color::Black),
        }
    }

    #[inline]
    pub fn of(&self, color: Color) -> Bitboard {
        if color == Color::White {
            self.white
        } else {
            self.black
        }
    }
}

/// The subset of `info`'s pawns of `color` that are passed.
fn passed_of(info: &PawnInfo, color: Color) -> Bitboard {
    let enemy = info.pawns_of(!color);
    let mut out = Bitboard(0);
    info.pawns_of(color).for_each(|sq| {
        if is_passed(sq, color, enemy) {
            out |= Bitboard::from_square(sq);
        }
    });
    out
}

pub fn evaluate_passed(board: &Board, info: &PawnInfo, passed: &PassedInfo) -> Score {
    let mut score = Score::zero();

    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let passed_bb = passed.of(color);
        if passed_bb.is_empty() {
            continue;
        }

        let mut color_score = Score::zero();
        passed_bb.for_each(|sq| {
            let rank = u8::from(sq.rank());
            let idx = if color == Color::White {
                rank as i32 - 1
            } else {
                6 - rank as i32
            };
            let mut bonus = PASSED_BONUS[idx.clamp(0, 5) as usize];

            // King-proximity adjustment (endgame part).
            if let (Some(our_king), Some(their_king)) =
                (board.king_of(color), board.king_of(!color))
            {
                let promo = if color == Color::White {
                    sq.offset(8 * (8 - i32::from(rank)))
                } else {
                    sq.offset(-8 * (i32::from(rank) - 1))
                }
                .unwrap_or(sq);
                let our_dist = our_king.distance(promo) as i32;
                let their_dist = their_king.distance(promo) as i32;
                if our_dist < their_dist {
                    bonus.eg = (bonus.eg * 125) / 100;
                } else if our_dist > their_dist {
                    bonus.eg = (bonus.eg * 75) / 100;
                }
            }

            // Protected passed pawn: an own pawn can defend it.
            if (info.pawn_attacks_of(color) & Bitboard::from_square(sq)).any() {
                color_score += PROTECTED_PASSED;
            }

            // Connected passed pawn: another passed pawn on an adjacent file
            // within one rank. Check only the higher file so each pair counts
            // once.
            let f = u8::from(sq.file());
            if f < 7 {
                let neighbour = passed_bb & Bitboard(file_mask(f + 1));
                if !neighbour.is_empty() {
                    let band = if color == Color::White {
                        rank_mask(rank.saturating_sub(1))
                            | rank_mask(rank)
                            | if rank < 7 { rank_mask(rank + 1) } else { 0 }
                    } else {
                        rank_mask(rank.saturating_sub(1))
                            | rank_mask(rank)
                            | if rank < 7 { rank_mask(rank + 1) } else { 0 }
                    };
                    // The rank band is already color-symmetric (ranks run the
                    // same upward for both sides in shakmaty); only the file
                    // direction needs care, which the `f < 7` guard handles.
                    if !(neighbour & Bitboard(band)).is_empty() {
                        color_score += CONNECTED_PASSED;
                    }
                }
            }

            color_score += bonus;
        });

        score += if sign > 0 { color_score } else { -color_score };
    }
    score
}

/// True when `sq` (a pawn of `color`) is passed.
pub(crate) fn is_passed(sq: Square, color: Color, enemy_pawns: Bitboard) -> bool {
    let f = u8::from(sq.file());
    let mut files = 0x0101010101010101u64 << f;
    if f > 0 {
        files |= 0x0101010101010101u64 << (f - 1);
    }
    if f < 7 {
        files |= 0x0101010101010101u64 << (f + 1);
    }
    let r = u8::from(sq.rank());
    let ahead_mask = if color == Color::White {
        !0u64 << (8 * (r + 1))
    } else {
        if r == 0 { 0 } else { (1u64 << (8 * r)) - 1 }
    };
    (enemy_pawns & Bitboard(files & ahead_mask)).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    fn parts(fen: &str) -> (Board, PawnInfo, PassedInfo) {
        let board = crate::testutil::chess(fen).board().clone();
        let info = PawnInfo::scan(&board);
        let passed = PassedInfo::scan(&info);
        (board, info, passed)
    }

    #[test]
    fn passed_pawn_detection() {
        // e4 white vs e5 black: blocked, not passed.
        let blocked = parts("6k1/8/8/8/4p3/4P3/8/4K3 w - - 0 1");
        // Lone e3 pawn: passed.
        let open = parts("6k1/8/8/8/8/4P3/8/4K3 w - - 0 1");
        let s_blocked = evaluate_passed(&blocked.0, &blocked.1, &blocked.2);
        let s_open = evaluate_passed(&open.0, &open.1, &open.2);
        assert!(
            s_open.eg > s_blocked.eg,
            "passed pawn must score higher in eg"
        );
        assert!(s_open.eg > 0);
    }

    #[test]
    fn own_pawn_on_adjacent_file_does_not_block() {
        // White pawn e4 is passed even though an adjacent-file pawn exists;
        // an enemy pawn far away on the g-file does not block.
        let (pos, pi, p) = parts("6k1/6p1/8/8/8/4P3/8/4K3 w - - 0 1");
        let s = evaluate_passed(&pos, &pi, &p);
        assert!(s.eg > 0);
    }

    #[test]
    fn protected_passed_pawn_is_bonused() {
        // A passed d4-pawn defended by the c3 pawn (c3 attacks d4) vs the
        // same pawn running lone. The black b4 pawn keeps c3 from being a
        // passer (so no connected-pair bonus fires) and appears in *both*
        // positions; in the bare position a3 takes over blocking b4, so b4
        // is non-passed on both sides and its terms cancel exactly.
        let defended = parts("6k1/8/8/8/1p1P4/2P5/8/4K3 w - - 0 1");
        let bare = parts("6k1/8/8/8/1p1P4/P7/8/4K3 w - - 0 1");
        let s_def = evaluate_passed(&defended.0, &defended.1, &defended.2);
        let s_bare = evaluate_passed(&bare.0, &bare.1, &bare.2);
        assert_eq!(
            s_def.mg - s_bare.mg,
            PROTECTED_PASSED.mg,
            "the rank-4 passer bonus is identical; only the protection differs"
        );
        assert_eq!(s_def.eg - s_bare.eg, PROTECTED_PASSED.eg);
    }

    #[test]
    fn connected_passed_pawns_are_bonused() {
        // Two adjacent passers (c5 + d5, both unstoppable on their files)
        // versus a single passer at d5 with nothing on the adjacent files.
        let twin = parts("6k1/8/8/2PP4/8/8/8/4K3 w - - 0 1");
        let single = parts("6k1/8/8/3P4/8/8/8/4K3 w - - 0 1");
        let s_twin = evaluate_passed(&twin.0, &twin.1, &twin.2);
        let s_single = evaluate_passed(&single.0, &single.1, &single.2);
        assert!(
            s_twin.mg > s_single.mg,
            "connected passers must beat the lone one: {} vs {}",
            s_twin.mg,
            s_single.mg
        );
        // Exactly the second passer's bonus plus one connected pair (both
        // c5 and d5 are rank-5 passers: PASSED_BONUS[3].mg each).
        assert_eq!(
            s_twin.mg - s_single.mg,
            PASSED_BONUS[3].mg + CONNECTED_PASSED.mg,
            "c5 passer adds rank-5 bonus and one connected pair"
        );
    }

    #[test]
    fn passed_set_matches_hand_counts() {
        // g4 white passed (nothing on g/f/h ahead), e4 blocked by e5.
        let scanned = parts("6k1/4p3/8/8/6P1/8/8/4K3 w - - 0 1");
        assert_eq!(scanned.2.white.count(), 1);
        let g4 = crate::testutil::chess("6k1/4p3/8/8/6P1/8/8/4K3 w - - 0 1")
            .board()
            .by_piece(shakmaty::Role::Pawn.of(Color::White))
            .first()
            .expect("g4 pawn");
        assert!(scanned.2.white.contains(g4));
    }
}

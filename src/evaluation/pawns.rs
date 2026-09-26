//! Pawn structure evaluation: doubled, isolated, backward and connected pawns
//! and pawn islands, plus the pawn-attack bitboards every other module needs
//! (and the file/rank masks several modules share).

use shakmaty::{Bitboard, Board, Color, attacks};

use crate::evaluation::Score;
use crate::evaluation::params::EvalParams;

const FILE_A: u64 = 0x0101010101010101;

#[inline]
pub(crate) fn file_mask(file: u8) -> u64 {
    FILE_A << file
}

#[inline]
pub(crate) fn rank_mask(rank: u8) -> u64 {
    0xffu64 << (8 * rank)
}

/// A mask of the files adjacent to `f` (for isolated/connected checks).
#[inline]
pub(crate) fn adjacent_files(f: u8) -> u64 {
    let mut m = 0u64;
    if f > 0 {
        m |= file_mask(f - 1);
    }
    if f < 7 {
        m |= file_mask(f + 1);
    }
    m
}

/// The number of pawn islands implied by a file-occupancy mask (bit `f` set
/// when the side has at least one pawn on file `f`): contiguous runs of
/// occupied files. `0b111` → 1, `0b101` → 2, `0b10000000` → 1.
#[inline]
fn islands(files: u8) -> u32 {
    // A new run starts at every occupied file whose left neighbour is empty.
    (files & !(files >> 1)).count_ones()
}

/// Everything derived from the pawn configuration in one scan, so threat /
/// king / mobility / passed-pawn modules do not recompute it.
#[derive(Clone, Copy, Debug)]
pub struct PawnInfo {
    pub white: Bitboard,
    pub black: Bitboard,
    pub white_attacks: Bitboard,
    pub black_attacks: Bitboard,
    /// Occupied files as bitmasks (bit `f` set when color has a pawn on file
    /// `f`).
    white_files: u8,
    black_files: u8,
}

impl PawnInfo {
    pub fn scan(board: &Board) -> PawnInfo {
        let white = board.by_piece(shakmaty::Role::Pawn.of(Color::White));
        let black = board.by_piece(shakmaty::Role::Pawn.of(Color::Black));
        let mut white_attacks = Bitboard(0);
        let mut black_attacks = Bitboard(0);
        white.for_each(|sq| white_attacks |= attacks::pawn_attacks(Color::White, sq));
        black.for_each(|sq| black_attacks |= attacks::pawn_attacks(Color::Black, sq));
        let mut white_files = 0u8;
        let mut black_files = 0u8;
        for f in 0..8u8 {
            if (white & Bitboard(file_mask(f))).any() {
                white_files |= 1 << f;
            }
            if (black & Bitboard(file_mask(f))).any() {
                black_files |= 1 << f;
            }
        }
        PawnInfo {
            white,
            black,
            white_attacks,
            black_attacks,
            white_files,
            black_files,
        }
    }

    #[inline]
    pub fn pawns_of(&self, color: Color) -> Bitboard {
        if color == Color::White {
            self.white
        } else {
            self.black
        }
    }

    #[inline]
    pub fn pawn_attacks_of(&self, color: Color) -> Bitboard {
        if color == Color::White {
            self.white_attacks
        } else {
            self.black_attacks
        }
    }

    #[inline]
    pub fn files_of(&self, color: Color) -> u8 {
        if color == Color::White {
            self.white_files
        } else {
            self.black_files
        }
    }
}

/// Evaluates pawn structure for both colors (white minus black), reading the
/// per-term weights from `p`.
pub fn evaluate_pawns(info: &PawnInfo, p: &EvalParams) -> Score {
    let doubled = p.doubled;
    let isolated = p.isolated;
    let backward = p.backward;
    let connected = p.connected;
    let island = p.island;
    let protected_pawn = p.protected_pawn;
    let mut score = Score::zero();
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let pawns = info.pawns_of(color);
        let mut color_score = Score::zero();

        // Doubled / isolated / backward / connected.
        pawns.for_each(|sq| {
            let f = u8::from(sq.file());
            let mask = file_mask(f);

            if (pawns & Bitboard(mask)).count() > 1 {
                color_score.mg += doubled[0];
                color_score.eg += doubled[1];
            }

            let adj = adjacent_files(f);
            if (pawns & Bitboard(adj)).is_empty() {
                color_score.mg += isolated[0];
                color_score.eg += isolated[1];
            }

            // Connected: own pawn on an adjacent file and rank ± 1.
            let own_adjacents = pawns & Bitboard(adj);
            if !own_adjacents.is_empty() {
                let r = u8::from(sq.rank());
                let same_or_behind = if color == Color::White {
                    rank_mask(r) | if r > 0 { rank_mask(r - 1) } else { 0 }
                } else {
                    rank_mask(r) | if r < 7 { rank_mask(r + 1) } else { 0 }
                };
                if !(own_adjacents & Bitboard(same_or_behind)).is_empty() {
                    color_score.mg += connected[0];
                    color_score.eg += connected[1];
                }
            }

            // Backward: not defended by own pawn and the square in front is
            // attacked by an enemy pawn.
            let defended = (info.pawn_attacks_of(color) & Bitboard::from_square(sq)).any();
            if defended {
                color_score.mg += protected_pawn[0];
                color_score.eg += protected_pawn[1];
            }
            let front = if color == Color::White {
                sq.offset(8)
            } else {
                sq.offset(-8)
            };
            if !defended
                && front.is_some()
                && (info.pawn_attacks_of(!color) & Bitboard::from_square(front.unwrap())).any()
            {
                color_score.mg += backward[0];
                color_score.eg += backward[1];
            }
        });

        // Pawn islands: too many disconnected groups cost structure value.
        let n = islands(info.files_of(color));
        for _ in 1..n {
            color_score.mg += island[0];
            color_score.eg += island[1];
        }

        score.mg += sign * color_score.mg;
        score.eg += sign * color_score.eg;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{Position as _, Square};

    fn scans(fen: &str) -> PawnInfo {
        let chess = crate::testutil::chess(fen);
        PawnInfo::scan(chess.board())
    }

    #[test]
    fn doubled_pawns_are_penalized() {
        // Doubled = two pawns on the same file (a2 + a3); the reference has
        // just the a2 pawn.
        let doubled = scans("6k1/8/8/8/8/P7/P7/4K3 w - - 0 1");
        let single = scans("6k1/8/8/8/8/8/P7/4K3 w - - 0 1");
        let s1 = evaluate_pawns(&single, &EvalParams::default());
        let s2 = evaluate_pawns(&doubled, &EvalParams::default());
        assert!(s1.mg - s2.mg >= 20, "doubled penalty expected");
    }

    #[test]
    fn isolated_pawns_are_penalized() {
        // a2 pawn isolated (no b/c file pawns), but c2/d2 exist for 'single'
        let iso = scans("6k1/8/8/8/8/8/P1P5/4K3 w - - 0 1");
        let connected = scans("6k1/8/8/8/8/8/PPP5/4K3 w - - 0 1");
        let s_iso = evaluate_pawns(&iso, &EvalParams::default());
        let s_con = evaluate_pawns(&connected, &EvalParams::default());
        assert!(s_con.mg > s_iso.mg, "connected > isolated");
    }

    #[test]
    fn attacks_are_computed() {
        let info = scans("6k1/8/8/8/3P4/8/8/4K3 w - - 0 1");
        // White pawn d4 attacks c5 and e5.
        assert!(info.white_attacks.contains(Square::C5));
        assert!(info.white_attacks.contains(Square::E5));
        assert!(info.white_attacks.count() == 2);
    }

    #[test]
    fn island_counting() {
        assert_eq!(islands(0b0000_0000), 0);
        assert_eq!(islands(0b0000_0001), 1); // a-file only
        assert_eq!(islands(0b0000_0111), 1); // a+b+c contiguous
        assert_eq!(islands(0b0000_0101), 2); // a and c, b missing
        assert_eq!(islands(0b0110_0001), 2); // a + e,f
        assert_eq!(islands(0b1111_1111), 1); // full pawn wall
    }

    #[test]
    fn split_pawns_are_penalized_as_islands() {
        // A solid a-b-c chain (one island) vs the same material split across
        // the a, c and f files (three islands).
        let chain = scans("6k1/8/8/8/8/PPP5/8/4K3 w - - 0 1");
        let split = scans("6k1/8/8/8/8/P1P3P1/8/4K3 w - - 0 1");
        let s_chain = evaluate_pawns(&chain, &EvalParams::default());
        let s_split = evaluate_pawns(&split, &EvalParams::default());
        assert!(
            s_chain.mg > s_split.mg,
            "one island must beat three: {} vs {}",
            s_chain.mg,
            s_split.mg
        );
    }

    #[test]
    fn startpos_has_a_single_island_per_side() {
        let info = scans("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(islands(info.white_files), 1);
        assert_eq!(islands(info.black_files), 1);
    }

    #[test]
    fn protected_pawns_are_bonused() {
        // A pawn defended by a friendly pawn on an adjacent file one rank
        // behind vs the same pawn running lone.
        let defended = scans("6k1/8/8/8/8/PPP5/8/4K3 w - - 0 1"); // b2-c2-d2 chain
        let lone = scans("6k1/8/8/8/8/P1P1P1P1/8/4K3 w - - 0 1"); // isolated
        let s_def = evaluate_pawns(&defended, &EvalParams::default());
        let s_lone = evaluate_pawns(&lone, &EvalParams::default());
        assert!(
            s_def.mg > s_lone.mg,
            "protected chain must beat isolated pawns"
        );
    }
}

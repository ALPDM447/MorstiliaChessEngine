//! Precomputed attack tables and the sliding-attack / line helpers the NNUE
//! feature sets and the threat emission need.
//!
//! Every table is a `const`, so it is materialized by the compiler rather
//! than built at startup — there is no lazy initialization, no `OnceLock` and
//! therefore no synchronization in the hot path. The construction mirrors
//! Stockfish's own `init()` in `attacks.cpp` so the resulting bitboards are
//! bit-for-bit the same.
//!
//! Square numbering is Stockfish's: `a1 = 0`, `h1 = 7`, `a8 = 56`, `h8 = 63.

use crate::nnue::types::{BISHOP, Bitboard, Color, KING, KNIGHT, QUEEN, ROOK, SQ_NONE};

/// Mask of the a-file.
pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
/// Mask of the b-file.
pub const FILE_B: Bitboard = 0x0202_0202_0202_0202;
/// Mask of the c-file.
pub const FILE_C: Bitboard = 0x0404_0404_0404_0404;
/// Mask of the d-file.
pub const FILE_D: Bitboard = 0x0808_0808_0808_0808;
/// Mask of the e-file.
pub const FILE_E: Bitboard = 0x1010_1010_1010_1010;
/// Mask of the f-file.
pub const FILE_F: Bitboard = 0x2020_2020_2020_2020;
/// Mask of the g-file.
pub const FILE_G: Bitboard = 0x4040_4040_4040_4040;
/// Mask of the h-file.
pub const FILE_H: Bitboard = 0x8080_8080_8080_8080;
/// Mask of rank 1.
pub const RANK_1: Bitboard = 0xff;
/// Mask of rank 8.
pub const RANK_8: Bitboard = 0xffu64 << 56;

#[inline]
pub const fn square_bb(s: usize) -> Bitboard {
    1u64 << s
}

/// Pops the least-significant set bit and returns it, clearing it in place.
#[inline]
pub fn pop_lsb(bb: &mut Bitboard) -> usize {
    let s = bb.trailing_zeros() as usize;
    *bb &= *bb - 1;
    s
}

#[inline]
pub const fn file_of(s: usize) -> usize {
    s & 7
}

#[inline]
pub const fn rank_of(s: usize) -> usize {
    s >> 3
}

// --- table construction (all `const`) ---------------------------------------

const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const KING_DELTAS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Single-step ("leaper") attacks: the set of squares reachable in one move on
/// an empty board.
const fn leap_table(deltas: &[(i32, i32)]) -> [Bitboard; 64] {
    let mut out = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let f = (s & 7) as i32;
        let r = (s >> 3) as i32;
        let mut bb = 0u64;
        let mut d = 0usize;
        while d < deltas.len() {
            let nf = f + deltas[d].0;
            let nr = r + deltas[d].1;
            if (nf >= 0 && nf < 8) && (nr >= 0 && nr < 8) {
                bb |= 1u64 << (nr * 8 + nf) as usize;
            }
            d += 1;
        }
        out[s] = bb;
        s += 1;
    }
    out
}

/// Pseudo-attacks of a sliding piece (the full ray set on an empty board).
const fn sliding_table(dirs: &[(i32, i32); 4]) -> [Bitboard; 64] {
    let mut out = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let f = (s & 7) as i32;
        let r = (s >> 3) as i32;
        let mut bb = 0u64;
        let mut d = 0usize;
        while d < 4 {
            let mut nf = f + dirs[d].0;
            let mut nr = r + dirs[d].1;
            while (nf >= 0 && nf < 8) && (nr >= 0 && nr < 8) {
                bb |= 1u64 << (nr * 8 + nf) as usize;
                nf += dirs[d].0;
                nr += dirs[d].1;
            }
            d += 1;
        }
        out[s] = bb;
        s += 1;
    }
    out
}

/// Pawn attacks from `s`. `up` is `+1` for White and `-1` for Black.
const fn pawn_attack_table(up: i32) -> [Bitboard; 64] {
    let mut out = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let f = (s & 7) as i32;
        let r = (s >> 3) as i32 + up;
        let mut bb = 0u64;
        if r >= 0 && r < 8 {
            if f > 0 {
                bb |= 1u64 << (r * 8 + f - 1) as usize;
            }
            if f < 7 {
                bb |= 1u64 << (r * 8 + f + 1) as usize;
            }
        }
        out[s] = bb;
        s += 1;
    }
    out
}

/// `RayPassBB[s1][s2]`: the squares strictly beyond `s1` seen from `s2` along
/// the line `s1`–`s2`, i.e. the line a slider standing on `s1` would need to
/// *pass through* to attack `s2`. Zero for unaligned pairs.
///
/// Built exactly like Stockfish's `init()`:
/// `attacks_bb(pt, s1, 0) & (attacks_bb(pt, s2, square_bb(s1)) | s2)`.
const fn ray_pass_table() -> [[Bitboard; 64]; 64] {
    let mut out = [[0u64; 64]; 64];
    let mut s1 = 0usize;
    while s1 < 64 {
        let mut s2 = 0usize;
        while s2 < 64 {
            let b2 = 1u64 << s2;
            if sliding_attacks(BISHOP, s1, 0) & b2 != 0 {
                out[s1][s2] =
                    sliding_attacks(BISHOP, s1, 0) & (sliding_attacks(BISHOP, s2, 1u64 << s1) | b2);
            }
            if sliding_attacks(ROOK, s1, 0) & b2 != 0 {
                out[s1][s2] =
                    sliding_attacks(ROOK, s1, 0) & (sliding_attacks(ROOK, s2, 1u64 << s1) | b2);
            }
            s2 += 1;
        }
        s1 += 1;
    }
    out
}

/// `PawnPairBB[s]`: the squares that can host a pawn forming a "pawn pair"
/// with a pawn on `s` — the own file plus its two neighbours, restricted to
/// ranks 2..7 and excluding `s` itself. Color independent, like Stockfish's.
const fn pawn_pair_table() -> [Bitboard; 64] {
    let mut out = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        // `file_bb(s)` is a *square* bitboard (the whole file), and the east /
        // west shifts are masked so they do not wrap around the board.
        let file = FILE_A << (s & 7);
        let files = file | ((file & !FILE_H) << 1) | ((file & !FILE_A) >> 1);
        out[s] = files & !(RANK_1 | RANK_8) & !(1u64 << s);
        s += 1;
    }
    out
}

pub const KNIGHT_ATTACKS: [Bitboard; 64] = leap_table(&KNIGHT_DELTAS);
pub const KING_ATTACKS: [Bitboard; 64] = leap_table(&KING_DELTAS);
pub const BISHOP_ATTACKS: [Bitboard; 64] = sliding_table(&BISHOP_DIRS);
pub const ROOK_ATTACKS: [Bitboard; 64] = sliding_table(&ROOK_DIRS);
/// `[color][square]`
pub const PAWN_ATTACKS: [[Bitboard; 64]; 2] = [pawn_attack_table(1), pawn_attack_table(-1)];
pub const RAY_PASS_BB: [[Bitboard; 64]; 64] = ray_pass_table();
pub const PAWN_PAIR_BB: [Bitboard; 64] = pawn_pair_table();

// --- queries ----------------------------------------------------------------

/// The attacks of a sliding piece, stopping *on* the first blocker.
#[inline]
pub const fn sliding_attacks(pt: u8, s: usize, occupied: Bitboard) -> Bitboard {
    let dirs: &[(i32, i32); 4] = if pt == BISHOP {
        &BISHOP_DIRS
    } else {
        &ROOK_DIRS
    };
    let f = (s & 7) as i32;
    let r = (s >> 3) as i32;
    let mut bb = 0u64;
    let mut d = 0usize;
    while d < 4 {
        let mut nf = f + dirs[d].0;
        let mut nr = r + dirs[d].1;
        while (nf >= 0 && nf < 8) && (nr >= 0 && nr < 8) {
            let sq = (nr * 8 + nf) as usize;
            bb |= 1u64 << sq;
            if occupied & (1u64 << sq) != 0 {
                break;
            }
            nf += dirs[d].0;
            nr += dirs[d].1;
        }
        d += 1;
    }
    bb
}

/// The attacks of any piece type (queens being the union of the two sliders).
#[inline]
pub const fn attacks_bb(pt: u8, s: usize, occupied: Bitboard) -> Bitboard {
    match pt {
        BISHOP => sliding_attacks(BISHOP, s, occupied),
        ROOK => sliding_attacks(ROOK, s, occupied),
        QUEEN => sliding_attacks(BISHOP, s, occupied) | sliding_attacks(ROOK, s, occupied),
        KNIGHT => KNIGHT_ATTACKS[s],
        KING => KING_ATTACKS[s],
        _ => 0,
    }
}

/// The attacks of a pawn standing on `s` (i.e. the squares it could capture
/// on, plus nothing else).
#[inline]
pub const fn pawn_attacks(color: Color, s: usize) -> Bitboard {
    PAWN_ATTACKS[color.idx()][s]
}

/// Pseudo-attacks of a *non-pawn* piece type (`Attacks::PseudoAttacks`).
#[inline]
pub const fn pseudo_attacks(pt: u8, s: usize) -> Bitboard {
    match pt {
        KNIGHT => KNIGHT_ATTACKS[s],
        BISHOP => BISHOP_ATTACKS[s],
        ROOK => ROOK_ATTACKS[s],
        QUEEN => BISHOP_ATTACKS[s] | ROOK_ATTACKS[s],
        KING => KING_ATTACKS[s],
        _ => 0,
    }
}

#[inline]
pub const fn knight_attacks(s: usize) -> Bitboard {
    KNIGHT_ATTACKS[s]
}

/// `both_attacks_bb(s, occ)` — the bishop and rook ray sets from `s` in one
/// pass, the quantity `update_piece_threats` needs.
#[inline]
pub fn both_attacks_bb(s: usize, occupied: Bitboard) -> (Bitboard, Bitboard) {
    (
        sliding_attacks(BISHOP, s, occupied),
        sliding_attacks(ROOK, s, occupied),
    )
}

/// The squares a slider on `s1` must traverse to reach `s2`, or zero when the
/// two are not aligned for a rook or bishop.
#[inline]
pub const fn ray_pass_bb(s1: usize, s2: usize) -> Bitboard {
    RAY_PASS_BB[s1][s2]
}

#[inline]
pub const fn pawn_pair_bb(s: usize) -> Bitboard {
    PAWN_PAIR_BB[s]
}

// Directional shifts used by the active-threat scan. The file masks stop the
// shift from wrapping around the board edge.
#[inline]
pub const fn shift_north_east(bb: Bitboard) -> Bitboard {
    (bb & !FILE_H) << 9
}
#[inline]
pub const fn shift_north_west(bb: Bitboard) -> Bitboard {
    (bb & !FILE_A) << 7
}
#[inline]
pub const fn shift_south_east(bb: Bitboard) -> Bitboard {
    (bb & !FILE_H) >> 7
}
#[inline]
pub const fn shift_south_west(bb: Bitboard) -> Bitboard {
    (bb & !FILE_A) >> 9
}

/// `SQ_NONE` re-exported for the feature modules, which treat it as the
/// "square absent" marker in the dirty records.
pub const NO_SQ: usize = SQ_NONE;

#[cfg(test)]
mod tests {
    use super::*;

    /// The tables below are pinned against `tests/data/attacks_reference.txt`,
    /// which is printed straight out of Stockfish 19 by linking against the
    /// engine's own objects. The whole-table check in
    /// `tests/nnue_tables.rs` compares every entry; these tests keep the
    /// hand-checked landmarks readable and name the geometry they describe.

    /// `d4` is index 27, `a1` is 0.
    const D4: usize = 27;
    const A1: usize = 0;

    #[test]
    fn leaper_tables_match_stockfish() {
        // Knight on d4 attacks b3, b5, c2, c6, e2, e6, f3, f5.
        assert_eq!(
            KNIGHT_ATTACKS[D4],
            (1 << 17)
                | (1 << 33)
                | (1 << 10)
                | (1 << 42)
                | (1 << 12)
                | (1 << 44)
                | (1 << 21)
                | (1 << 37),
            "Nd4",
        );
        assert_eq!(KNIGHT_ATTACKS[D4], 0x0000_1422_0022_1400, "Nd4, verbatim");

        // King on a1 attacks a2, b1, b2.
        assert_eq!(KING_ATTACKS[A1], 0x302);
        assert_eq!(KING_ATTACKS[A1].count_ones(), 3);
        assert_eq!(KING_ATTACKS[A1] & (1 << A1), 0, "no self attack");

        // Pawn on e4 (28) attacks d5 (35) and f5 (37) as White, d3 (19) and
        // f3 (21) as Black.
        assert_eq!(pawn_attacks(Color::White, 28), (1 << 35) | (1 << 37));
        assert_eq!(pawn_attacks(Color::Black, 28), (1 << 19) | (1 << 21));
        // A pawn on the first rank still has pseudo-attacks: Stockfish's
        // `PawnAttacks` is a plain diagonal shift, so h1 sees g3 even though no
        // pawn can stand there. The rank-2 pawns see the rank-1 squares instead.
        assert_eq!(pawn_attacks(Color::White, A1), 1 << 9, "a1 attacks only b2");
        assert_eq!(pawn_attacks(Color::Black, 15), 1 << 6, "h2 attacks only g1");
        assert_eq!(pawn_attacks(Color::White, 7), 1 << 14, "h1 shifts up to g3");
        assert_eq!(pawn_attacks(Color::Black, 8), 1 << 1, "a2 attacks only b1");
    }

    #[test]
    fn sliding_attacks_stop_on_the_blocker() {
        // Rook on a1, nothing else: the a-file and the first rank, *not* a1
        // itself (Stockfish's `sliding_attack` starts one square out).
        assert_eq!(sliding_attacks(ROOK, A1, 0), 0x0101_0101_0101_01fe);
        assert_eq!(
            sliding_attacks(ROOK, A1, 0),
            ROOK_ATTACKS[A1],
            "the table is unblocked"
        );
        // From d4: the whole d-file (7 squares) plus a4..c4 and e4..h4 — 14
        // squares, the origin excluded.
        assert_eq!(
            BISHOP_ATTACKS[D4], 0x8041_2214_0014_2241,
            "Bd4 on an empty board"
        );
        assert_eq!(
            ROOK_ATTACKS[D4], 0x0808_0808_f708_0808,
            "Rd4 on an empty board"
        );
        assert_eq!(BISHOP_ATTACKS[D4].count_ones(), 13, "Bd4's two diagonals");
        assert_eq!(ROOK_ATTACKS[D4].count_ones(), 14, "Rd4's file and rank");
        assert_eq!(sliding_attacks(ROOK, D4, 0), ROOK_ATTACKS[D4]);
        assert_eq!(sliding_attacks(BISHOP, D4, 0), BISHOP_ATTACKS[D4]);
        assert_eq!(BISHOP_ATTACKS[D4] & (1 << D4), 0, "no self attack");
        assert_eq!(ROOK_ATTACKS[D4] & (1 << D4), 0, "no self attack");
        // Spot-check the decoded squares rather than trusting the hex.
        assert_eq!(ROOK_ATTACKS[D4] >> 3 & 1, 1, "d1");
        assert_eq!(ROOK_ATTACKS[D4] >> 11 & 1, 1, "d2");
        assert_eq!(ROOK_ATTACKS[D4] >> 19 & 1, 1, "d3");
        assert_eq!(ROOK_ATTACKS[D4] >> 24 & 1, 1, "a4");
        assert_eq!(ROOK_ATTACKS[D4] >> 31 & 1, 1, "h4");
        assert_eq!(ROOK_ATTACKS[D4] >> 35 & 1, 1, "d5");
        assert_eq!(ROOK_ATTACKS[D4] >> 59 & 1, 1, "d8");
        // d4's two diagonals: a1 b2 c3 / e3 f2 g1 / c5 e5 / b6 f6 / a7 h7 / h8.
        for s in [0, 9, 18, 20, 13, 6, 34, 36, 41, 45, 48, 54, 63] {
            assert_eq!(BISHOP_ATTACKS[D4] >> s & 1, 1, "Bd4 sees {s}");
        }
        assert_eq!(BISHOP_ATTACKS[D4].count_ones(), 13, "and nothing else");

        // A blocker on a2 stops the rook's file ray after a2: a1 attacks a2 but
        // nothing beyond it.
        let a = sliding_attacks(ROOK, A1, 1 << 8);
        assert_eq!(a & FILE_A, 1 << 8, "the blocker square is included");
        assert_eq!(a & FILE_A, 0x100, "a2 only");

        // The blocker square itself is always attacked.
        let occ = 1u64 << 24;
        let a = sliding_attacks(ROOK, A1, occ);
        assert!(a & (1 << 24) != 0, "the blocker square is included");
        assert!(a & (1 << 32) == 0, "nothing beyond the blocker");
        // a1 is not in the set; a2, a3 and the blocker a4 are.
        assert_eq!(a & FILE_A, (1 << 8) | (1 << 16) | (1 << 24));
        assert_eq!(a & FILE_A, 0x0101_0100);

        // Stockfish's loop starts at the first *destination*
        // (`for (Square s = sq; Bitboard dest = safe_destination(s, d); s += d)`
        // `attacks |= dest; if (occupied & dest) break;`), so it only ever tests
        // the destination square. Marking the origin occupied therefore blocks
        // nothing: a rook on a1 still walks the whole a-file. The callsite in
        // `evaluate_nnue` relies on this, since Stockfish passes a bitboard that
        // includes the sliding piece itself.
        assert_eq!(
            sliding_attacks(ROOK, A1, 1 << A1) & FILE_A,
            FILE_A & !(1 << A1)
        );
        assert_eq!(sliding_attacks(BISHOP, A1, 1 << A1), BISHOP_ATTACKS[A1]);
        // A blocker *on* the ray stops it. With a6 blocked the a-file ray is
        // a2..a6 inclusive; the origin is never in the set.
        assert_eq!(
            sliding_attacks(ROOK, A1, 1 << A1 | 1 << 40) & FILE_A,
            FILE_A & !(1 << A1) & !((0xffu64 << 48) | RANK_8)
        );

        // Queen = rook + bishop, and a blocked queen is the union of the two
        // blocked sets.
        assert_eq!(
            attacks_bb(QUEEN, D4, 0),
            sliding_attacks(BISHOP, D4, 0) | sliding_attacks(ROOK, D4, 0)
        );
        let occ = (1 << 20) | (1 << 36);
        assert_eq!(
            attacks_bb(QUEEN, D4, occ),
            sliding_attacks(BISHOP, D4, occ) | sliding_attacks(ROOK, D4, occ)
        );
    }

    #[test]
    fn ray_pass_bb_matches_the_stockfish_formula() {
        // Rook line a1-a4. Stockfish's formula
        // `attacks_bb(pt, s1, 0) & (attacks_bb(pt, s2, square_bb(s1)) | s2)` does
        // *not* produce "the squares in between": because the origin counts as
        // a blocker, the second term is the a-file from a5 upwards, so the
        // result is the a-file minus the first rank.
        assert_eq!(ray_pass_bb(A1, 24), 0x0101_0101_0101_0100);
        // The mirror direction is a1 and the two squares between: a1 counts as
        // a blocker, so the ray stops there and a2/a3 are not included.
        assert_eq!(ray_pass_bb(24, A1), 0x0001_0101);
        // Diagonal: a1's attack set meets the whole a1-h8 diagonal.
        assert_eq!(ray_pass_bb(A1, 63), 0x8040_2010_0804_0200);
        assert_eq!(
            ray_pass_bb(A1, 9),
            0x8040_2010_0804_0200,
            "a1-b2 shares the diagonal"
        );
        // d1 and h6 are not aligned.
        assert_eq!(ray_pass_bb(3, 60), 0);
        // A knight's move away is never aligned.
        assert_eq!(ray_pass_bb(A1, 17), 0);
    }

    #[test]
    fn pawn_pair_bb_is_file_neighbours_ranks_2_to_7() {
        // d4: the d-file and its two neighbours, ranks 2..7, minus d4 itself.
        assert_eq!(pawn_pair_bb(D4), 0x001c_1c1c_141c_1c00);
        let b = pawn_pair_bb(D4);
        assert_eq!(b & (1 << D4), 0, "own square excluded");
        assert_eq!(b & RANK_1, 0, "rank 1 excluded");
        assert_eq!(b & RANK_8, 0, "rank 8 excluded");
        assert_eq!(b.count_ones(), 17, "3 files x 6 ranks - own square");
        // Every square of files c..e on ranks 2..7 except d4.
        for f in 2..=4usize {
            for r in 1..=6usize {
                let s = r * 8 + f;
                assert_eq!(b & (1 << s) != 0, s != D4, "c2..e7 coverage at {s}");
            }
        }
        // a-file: the a- and b-files only.
        assert_eq!(pawn_pair_bb(8), 0x0003_0303_0303_0200);
        assert_eq!(pawn_pair_bb(8) & FILE_C, 0, "a-file pawns do not reach c");
        assert_eq!(
            pawn_pair_bb(8) & FILE_B,
            FILE_B & !(RANK_1 | RANK_8),
            "b2..b7"
        );
        // h-file: the g- and h-files only.
        assert_eq!(pawn_pair_bb(15), 0x00c0_c0c0_c0c0_4000);
        assert_eq!(pawn_pair_bb(15) & FILE_F, 0, "h-file pawns do not reach f");
        assert_eq!(
            pawn_pair_bb(15) & FILE_G,
            FILE_G & !(RANK_1 | RANK_8),
            "g2..g7"
        );
        // The band is colour-independent, so it is the same on rank 1 and rank 8
        // apart from the origin square itself.
        assert_eq!(
            pawn_pair_bb(0) & FILE_B,
            FILE_B & !(RANK_1 | RANK_8) & !(1 << 1)
        );
        assert_eq!(
            pawn_pair_bb(63) & FILE_G,
            FILE_G & !(RANK_1 | RANK_8) & !(1 << 62)
        );
    }

    #[test]
    fn ray_pass_is_symmetric_in_the_diagonal_case() {
        // Not a property of the data, but a cheap guard: for a rook pair the
        // two directions are computed by the same expression, so a1->a4 and
        // a4->a1 must both be non-zero whenever the squares share a ray.
        for s1 in [A1, 9, 27, 36, 63] {
            for s2 in [A1, 9, 27, 36, 63] {
                if s1 != s2 && ray_pass_bb(s1, s2) != 0 {
                    assert_ne!(ray_pass_bb(s2, s1), 0, "{s1}->{s2}");
                }
            }
        }
    }

    #[test]
    fn directional_shifts_do_not_wrap_files() {
        // h4 (31) shifted north-east would wrap onto a5 without the mask.
        assert_eq!(shift_north_east(1u64 << 31), 0);
        // a4 (24) shifted north-west would wrap onto h5 without the mask.
        assert_eq!(shift_north_west(1u64 << 24), 0);
        // A normal file-internal shift.
        assert_eq!(shift_north_east(1u64 << 28), 1u64 << 37, "e4 -> f5");
        assert_eq!(shift_south_west(1u64 << 37), 1u64 << 28, "f5 -> e4");
        // The h-file masked off entirely for north-east, the a-file for
        // north-west.
        assert_eq!(shift_north_east(FILE_H), 0);
        assert_eq!(shift_north_west(FILE_A), 0);
        assert_eq!(shift_south_east(FILE_H), 0);
        assert_eq!(shift_south_west(FILE_A), 0);
    }
}

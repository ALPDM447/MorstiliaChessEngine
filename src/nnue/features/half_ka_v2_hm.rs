//! `HalfKAv2_hm`: own-king-square + piece-square.
//!
//! The index is `(s ^ orient ^ flip) + piece_square[perspective][pc] +
//! king_bucket[ksq ^ flip]`, where `orient` puts the king on the e..h half of
//! the board (files a..d are mirrored) so that only 32 king buckets are needed
//! instead of 64. `flip` is the 180° rotation for the black perspective, and the
//! W/B piece blocks in the table are swapped for that perspective so that the
//! whole feature is symmetric under a board flip.

use crate::nnue::board::Board;
use crate::nnue::types::{Color, DirtyPiece, KING, NO_PIECE, SQ_NONE, ValueList, make_piece};

/// Hash the trainer embeds for this feature set.
pub const HASH_VALUE: u32 = 0x7f23_4cb8;

/// `SQUARE_NB * PS_NB / 2` with `PS_NB = 11 * 64`.
pub const DIMENSIONS: usize = 64 * 704 / 2;

/// At most 32 pieces on the board ⇒ at most 32 active features.
pub const MAX_ACTIVE_DIMENSIONS: usize = 32;

pub type IndexList = ValueList<u16, MAX_ACTIVE_DIMENSIONS>;

/// The piece-square offsets, `PS_NONE = PS_W_PAWN = 0`. Colliding "empty" and
/// "white pawn" is harmless: an empty square is never indexed.
#[rustfmt::skip]
const PS: [[u16; 16]; 2] = [
    // Perspective White. Convention: W - us, B - them.
    [0,     0,   128, 256, 384, 512, 640, 0,
          0,    64,  192, 320, 448, 576, 640, 0],
    // Perspective Black: viewed from the other side, W and B are reversed.
    [0,    64,  192, 320, 448, 576, 640, 0,
          0,     0,  128, 256, 384, 512, 640, 0],
];

/// `B(v) = v * PS_NB` with `PS_NB = 704`.
///
/// Each entry is `704 * (28 - 4*rank + min(file, 7-file))`, so the 64 squares
/// collapse onto 32 distinct buckets — `a4 == h4`, `b4 == g4`, `c4 == f4` and
/// `d4 == e4`, which is exactly the mirror [`ORIENT_TBL`] applies to the pieces
/// when the king stands on the a..d half of the board.
const fn king_buckets() -> [u16; 64] {
    let mut out = [0u16; 64];
    let mut s = 0usize;
    while s < 64 {
        let rank = (s >> 3) as i32;
        let file = (s & 7) as i32;
        let min_file = if file < 7 - file { file } else { 7 - file };
        out[s] = ((28 - 4 * rank + min_file) * 704) as u16;
        s += 1;
    }
    out
}

const KING_BUCKETS: [u16; 64] = king_buckets();

/// Mirroring table: `SQ_H1` (= 7) for files a..d, `SQ_A1` (= 0) for e..h.
const fn orient_tbl() -> [usize; 64] {
    let mut out = [0usize; 64];
    let mut s = 0usize;
    while s < 64 {
        out[s] = if (s & 7) < 4 { 7 } else { 0 };
        s += 1;
    }
    out
}

/// Mirroring table: `SQ_H1` (= 7) for files a..d, `SQ_A1` (= 0) for e..h. Public
/// because the sibling `FullThreats` set mirrors the other way and a test pins
/// the two against each other.
pub const ORIENT_TBL: [usize; 64] = orient_tbl();

/// The feature index of `pc` on `s`, seen from `perspective` with own king on
/// `ksq`.
#[inline]
pub fn make_index(perspective: Color, s: usize, pc: u8, ksq: usize) -> u16 {
    debug_assert!(s < 64 && ksq < 64 && pc != NO_PIECE);
    let flip = 56 * perspective.idx();
    (s ^ ORIENT_TBL[ksq] ^ flip) as u16
        + PS[perspective.idx()][pc as usize]
        + KING_BUCKETS[ksq ^ flip]
}

/// Whether the move recorded in `diff` invalidates the whole accumulator for
/// `perspective` because that side's own king moved.
#[inline]
pub fn requires_refresh(diff: &DirtyPiece, perspective: Color) -> bool {
    diff.pc == make_piece(perspective.idx(), KING)
}

/// Appends the indices of the (at most four) features a move removed/added.
#[inline]
pub fn append_changed_indices(
    perspective: Color,
    ksq: usize,
    diff: &DirtyPiece,
    removed: &mut IndexList,
    added: &mut IndexList,
) {
    removed.push(make_index(perspective, diff.from, diff.pc, ksq));
    if diff.to != SQ_NONE {
        added.push(make_index(perspective, diff.to, diff.pc, ksq));
    }
    if diff.remove_sq != SQ_NONE {
        removed.push(make_index(perspective, diff.remove_sq, diff.remove_pc, ksq));
    }
    if diff.add_sq != SQ_NONE {
        added.push(make_index(perspective, diff.add_sq, diff.add_pc, ksq));
    }
}

/// Every active feature of the position, for one perspective.
///
/// Both kings are included: the piece-square table has a `PS_KING` block, so the
/// king is an ordinary feature here rather than a separate input.
pub fn append_active_indices(perspective: Color, board: &Board, active: &mut IndexList) {
    let ksq = board.king_square(perspective);
    let mut bb = board.pieces();
    while bb != 0 {
        let s = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        active.push(make_index(perspective, s, board.piece_on(s), ksq));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::nnue::types::{
        B_BISHOP, B_KING, B_KNIGHT, B_PAWN, B_QUEEN, B_ROOK, ROOK, W_BISHOP, W_KING, W_KNIGHT,
        W_PAWN, W_QUEEN, W_ROOK,
    };

    // FEN square helpers, spelled out for readability in the assertions.
    const A1: usize = 0;
    const D1: usize = 3;
    const E1: usize = 4;
    const H1: usize = 7;
    const A4: usize = 24;
    const H4: usize = 31;
    const E4: usize = 28;
    const A5: usize = 32;
    const H5: usize = 39;
    const D5: usize = 35;
    const A8: usize = 56;
    const H8: usize = 63;
    const D4: usize = 27;

    #[test]
    fn king_buckets_mirror_across_the_board() {
        // `min(file, 7 - file)` pairs a with h, b with g, c with f and d with e
        // — the mirror images that `OrientTBL` also applies to the pieces, so
        // the two half-tables cover the same 32 buckets each.
        for s in 0..64 {
            let mirrored = (7 - (s & 7)) + 8 * (s >> 3);
            assert_eq!(KING_BUCKETS[s], KING_BUCKETS[mirrored], "{s}");
            let bucket = KING_BUCKETS[s];
            assert!(
                (0..32).any(|k| k * 704 == bucket),
                "{s} -> {bucket} is not one of the 32 buckets"
            );
        }
        // Spot values straight from Stockfish's table.
        assert_eq!(KING_BUCKETS[A1], 28 * 704);
        assert_eq!(KING_BUCKETS[D1], 31 * 704);
        assert_eq!(KING_BUCKETS[E1], 31 * 704);
        assert_eq!(KING_BUCKETS[H1], 28 * 704);
        assert_eq!(KING_BUCKETS[A4], 16 * 704);
        assert_eq!(KING_BUCKETS[E4], 19 * 704);
        assert_eq!(KING_BUCKETS[A8], 0);
        assert_eq!(KING_BUCKETS[H8], 0);
    }

    #[test]
    fn orient_table_splits_the_board_down_the_centre() {
        for s in 0..64 {
            assert_eq!(ORIENT_TBL[s], if (s & 7) < 4 { 7 } else { 0 });
        }
    }

    #[test]
    fn indices_stay_in_range_for_every_square_and_piece() {
        for ksq in 0..64 {
            for s in 0..64 {
                for pc in [W_KING, B_QUEEN, B_PAWN] {
                    for p in Color::ALL {
                        let idx = make_index(p, s, pc, ksq) as usize;
                        assert!(
                            idx < DIMENSIONS,
                            "{p:?} {pc} on {s} with ksq {ksq} -> {idx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_piece_feature_is_king_bucket_relative() {
        // The whole feature is measured from the king, and the king on the a..d
        // half of the board mirrors the pieces so that both halves share one
        // bucket table. Mirroring therefore applies to the *piece* as well: a
        // knight on a5 with the king on a4 is the same feature as a knight on
        // h5 with the king on h4.
        let base = make_index(Color::White, A5, W_KNIGHT, A4) as i32;
        assert_eq!(base, make_index(Color::White, H5, W_KNIGHT, H4) as i32);
        // Crossing the centre line without mirroring the piece is a different
        // feature, which is the whole point of the compression.
        assert_ne!(base, make_index(Color::White, A5, W_KNIGHT, H4) as i32);
        // Moving the king along the mirrored half changes only the bucket.
        assert_ne!(base, make_index(Color::White, A5, W_KNIGHT, A5) as i32);
        assert_ne!(base, make_index(Color::White, A5, W_KNIGHT, H5) as i32);
        // The piece block dominates: two pieces on the same square differ by
        // exactly their `PieceSquareIndex` distance.
        assert_eq!(
            (make_index(Color::White, D4, B_QUEEN, A1) as i32)
                - (make_index(Color::White, D4, W_KNIGHT, A1) as i32),
            (PS[0][B_QUEEN as usize] - PS[0][W_KNIGHT as usize]) as i32
        );
    }

    #[test]
    fn the_black_perspective_mirrors_the_board() {
        // Black's whole feature set is White's under a 180° rotation plus the
        // W/B piece-block swap, so the two indices are literally the same
        // number. This is what lets one weight row serve both perspectives.
        for &s in &[A1, D1, D4, A4, E4, A5, H5, A8, H8, 34, 41] {
            for &ksq in &[A1, D1, D4, A4, E4, A5, H5, A8, H8] {
                for &pc in &[
                    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, B_PAWN, B_KNIGHT,
                    B_BISHOP, B_ROOK, B_QUEEN, B_KING,
                ] {
                    assert_eq!(
                        make_index(Color::Black, s, pc, ksq),
                        make_index(Color::White, s ^ 56, pc ^ 8, ksq ^ 56),
                        "black {pc} on {s} with ksq {ksq}"
                    );
                }
            }
        }
        // So the king bucket really is taken through the rotation.
        assert_eq!(KING_BUCKETS[A1 ^ 56], KING_BUCKETS[H8]);
        assert_eq!(KING_BUCKETS[D4 ^ 56], KING_BUCKETS[D5]);
        assert_ne!(KING_BUCKETS[D4 ^ 56], KING_BUCKETS[D4]);
        // But the two perspectives are *not* the same index for the same
        // board — the rotation of the pieces is what separates them.
        assert_ne!(
            make_index(Color::Black, D4, B_QUEEN, A1) as i32,
            make_index(Color::White, D4, W_QUEEN, A1) as i32
        );
    }

    #[test]
    fn active_indices_cover_every_piece_once_per_perspective() {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let board = crate::nnue::board::Board::from_position(&pos);
        for p in Color::ALL {
            let mut list = IndexList::new();
            append_active_indices(p, &board, &mut list);
            assert_eq!(
                list.size() as u32,
                board.piece_count(),
                "{p:?} should see every piece"
            );
        }
    }

    #[test]
    fn changed_indices_emit_three_features_for_a_capture() {
        let dp = DirtyPiece {
            pc: W_KNIGHT,
            from: D4,
            to: 18,
            remove_sq: 18,
            add_sq: SQ_NONE,
            remove_pc: B_PAWN,
            add_pc: NO_PIECE,
        };
        let mut removed = IndexList::new();
        let mut added = IndexList::new();
        append_changed_indices(Color::White, A1, &dp, &mut removed, &mut added);
        assert_eq!(removed.size(), 2, "origin plus the captured piece");
        assert_eq!(added.size(), 1, "destination");
    }

    #[test]
    fn a_promotion_reports_two_features() {
        // `Position::do_move` clears `to` for a promotion and reports the new
        // piece through `add_sq`/`add_pc` instead, so the pawn's destination
        // feature is never added: only the promoted piece's is.
        let mut dp = DirtyPiece::new(W_KNIGHT, 18, SQ_NONE);
        dp.add_sq = 56;
        dp.add_pc = W_QUEEN;
        let mut removed = IndexList::new();
        let mut added = IndexList::new();
        append_changed_indices(Color::White, A1, &dp, &mut removed, &mut added);
        assert_eq!(removed.size(), 1, "the pawn leaves 18");
        assert_eq!(added.size(), 1, "the promoted piece arrives on 56");
        assert_eq!(removed[0], make_index(Color::White, 18, W_KNIGHT, A1));
        assert_eq!(added[0], make_index(Color::White, 56, W_QUEEN, A1));

        // A capturing promotion removes the victim and the pawn, and adds only
        // the promoted piece: two out, one in.
        let mut dp = DirtyPiece::new(W_KNIGHT, 18, SQ_NONE);
        dp.remove_sq = 56;
        dp.remove_pc = B_ROOK;
        dp.add_sq = 56;
        dp.add_pc = W_QUEEN;
        let mut removed = IndexList::new();
        let mut added = IndexList::new();
        append_changed_indices(Color::White, A1, &dp, &mut removed, &mut added);
        assert_eq!(removed.size(), 2);
        assert_eq!(added.size(), 1);
        assert_eq!(removed[1], make_index(Color::White, 56, B_ROOK, A1));
    }

    #[test]
    fn a_castling_move_reports_three_features() {
        // `do_castling` reports the king's destination as `to` and the rook as
        // remove_sq/add_sq, so HalfKAv2_hm sees the king move and the rook
        // move. The king's move is what forces a full refresh.
        let rook = make_piece(0, ROOK);
        let dp = DirtyPiece {
            pc: make_piece(0, KING),
            from: 4,
            to: 6,
            remove_sq: 7,
            remove_pc: rook,
            add_sq: 5,
            add_pc: rook,
        };
        let mut removed = IndexList::new();
        let mut added = IndexList::new();
        append_changed_indices(Color::White, A1, &dp, &mut removed, &mut added);
        assert_eq!(removed.size(), 2, "e1 king and h1 rook");
        assert_eq!(added.size(), 2, "g1 king and f1 rook");
        assert_eq!(
            removed[0],
            make_index(Color::White, 4, make_piece(0, KING), A1)
        );
        assert_eq!(
            added[0],
            make_index(Color::White, 6, make_piece(0, KING), A1)
        );
        assert_eq!(removed[1], make_index(Color::White, 7, rook, A1));
        assert_eq!(added[1], make_index(Color::White, 5, rook, A1));
        assert!(requires_refresh(&dp, Color::White));
    }

    #[test]
    fn only_a_king_move_forces_a_refresh() {
        let dp = DirtyPiece::new(make_piece(0, KING), A1, H1);
        assert!(requires_refresh(&dp, Color::White));
        assert!(!requires_refresh(&dp, Color::Black));
        let dp = DirtyPiece::new(W_KNIGHT, D4, 18);
        assert!(!requires_refresh(&dp, Color::White));
    }
}

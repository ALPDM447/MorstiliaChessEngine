//! `FullThreats`: every `attacker → target` relationship on the board.
//!
//! The index space is a flat enumeration of all geometrically possible
//! (attacker, target) pairs, with three reductions:
//!
//! * pawns may only threaten knights and rooks (pawn-on-pawn is the
//!   `PP_3Wide` set's job) — hence 4 valid targets instead of 10;
//! * a ray threat between two pieces of the same type is "semi-excluded", so a
//!   queen only ever threatens a non-queen target through a ray;
//! * a combination that cannot occur at all collapses to index `Dimensions`,
//!   which every emitter filters out.
//!
//! All of this is precomputed into constant tables by `const fn`s driven off the
//! same attack geometry the rest of the module uses, so the tables cannot drift
//! from the rules they encode.

use crate::nnue::attacks::{
    attacks_bb, pawn_attacks, pseudo_attacks, shift_north_east, shift_north_west, shift_south_east,
    shift_south_west,
};
use crate::nnue::board::Board;
use crate::nnue::types::{
    ALL_PIECES, BISHOP, Bitboard, Color, DirtyThreats, KNIGHT, PAWN, PIECE_NB, QUEEN, ROOK,
    ValueList, color_of, make_piece, type_of,
};

/// Hash the trainer embeds for this feature set.
pub const HASH_VALUE: u32 = 0x2e6b_9d04;

/// Number of (attacker, target) features the net has weights for.
pub const DIMENSIONS: usize = 59808;

/// How many target slots each attacker type keeps, indexed by piece code.
/// Pawns get 4 (knight, rook), knights and queens 10 (everything), bishops and
/// rooks 8 (no queen), kings 0.
const NUM_VALID_TARGETS: [u32; PIECE_NB] = [
    0, 4, 10, 8, 8, 10, 0, 0, //
    0, 4, 10, 8, 8, 10, 0, 0,
];

/// The per-attacker-type target filter; `-1` means "excluded entirely".
#[rustfmt::skip]
const MAP: [[i8; 6]; 6] = [
    [-1,  0, -1,  1, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [-1, -1, -1, -1, -1, -1],
];

/// Mirroring table for this feature set: `SQ_A1` (= 0) for files a..d,
/// `SQ_H1` (= 7) for e..h — the *opposite* of HalfKAv2_hm. Public because
/// `PP_3Wide` reuses it.
const fn orient_tbl() -> [usize; 64] {
    let mut out = [0usize; 64];
    let mut s = 0usize;
    while s < 64 {
        out[s] = if (s & 7) < 4 { 0 } else { 7 };
        s += 1;
    }
    out
}

/// Mirroring table for this feature set: `SQ_A1` (= 0) for files a..d,
/// `SQ_H1` (= 7) for e..h — the *opposite* of HalfKAv2_hm. Public because
/// `PP_3Wide` reuses it.
pub const ORIENT_TBL: [usize; 64] = orient_tbl();

/// Shared index list: 256 covers the ≤80 threat features of a normal move plus
/// the ≤54 pawn-pair features of a pawn move.
pub type IndexList = ValueList<u16, 256>;

// --- generated tables ---------------------------------------------------------

/// `(cumulative_piece_offset, cumulative_offset)` per piece code. The first is
/// the number of geometric attacker-square combinations for that piece; the
/// second is the flat base offset of the piece's block in the index space.
const fn init_threat_offsets() -> ([[u32; 2]; PIECE_NB], [[u32; 64]; PIECE_NB]) {
    let mut helper = [[0u32; 2]; PIECE_NB];
    let mut offsets = [[0u32; 64]; PIECE_NB];

    let mut cumulative_offset = 0u32;
    let mut i = 0usize;
    while i < ALL_PIECES.len() {
        let piece = ALL_PIECES[i];
        let piece_idx = piece as usize;
        let mut cumulative_piece_offset = 0u32;

        let mut from = 0usize;
        while from < 64 {
            offsets[piece_idx][from] = cumulative_piece_offset;
            if type_of(piece) != PAWN {
                cumulative_piece_offset += pseudo_attacks(type_of(piece), from).count_ones();
            } else if from >= 8 && from <= 55 {
                // Pawn index space is only allocated for ranks 2..7, even
                // though `PawnAttacks[WHITE][SQ_A1]` is not empty: a pawn can
                // never stand on the first or last rank, so those squares would
                // only alias onto the rank-2 block.
                let c = if piece_idx < 8 {
                    Color::White
                } else {
                    Color::Black
                };
                cumulative_piece_offset += pawn_attacks(c, from).count_ones();
            }
            from += 1;
        }

        helper[piece_idx] = [cumulative_piece_offset, cumulative_offset];
        cumulative_offset += NUM_VALID_TARGETS[piece_idx] * cumulative_piece_offset;
        i += 1;
    }

    (helper, offsets)
}

const HELPER_AND_OFFSETS: ([[u32; 2]; PIECE_NB], [[u32; 64]; PIECE_NB]) = init_threat_offsets();
const HELPER_OFFSETS: [[u32; 2]; PIECE_NB] = HELPER_AND_OFFSETS.0;
const OFFSETS: [[u32; 64]; PIECE_NB] = HELPER_AND_OFFSETS.1;

/// `index_lut1[attacker][attacked][from_oriented < to_oriented]`: the flat base
/// of the feature, or `Dimensions` for a combination the set excludes.
const fn init_index_lut1() -> [[[u32; 2]; PIECE_NB]; PIECE_NB] {
    let mut luts = [[[0u32; 2]; PIECE_NB]; PIECE_NB];
    let mut i = 0usize;
    while i < ALL_PIECES.len() {
        let attacker = ALL_PIECES[i];
        let mut j = 0usize;
        while j < ALL_PIECES.len() {
            let attacked = ALL_PIECES[j];
            let enemy = (attacker ^ attacked) == 8;
            let attacker_type = type_of(attacker);
            let attacked_type = type_of(attacked);

            let map = MAP[(attacker_type - 1) as usize][(attacked_type - 1) as usize];
            let semi_excluded = attacker_type == attacked_type && (enemy || attacker_type != PAWN);
            let slot = color_of(attacked) as u32 * (NUM_VALID_TARGETS[attacker as usize] / 2)
                + if map < 0 { 0 } else { map as u32 };
            let feature =
                HELPER_OFFSETS[attacker as usize][1] + slot * HELPER_OFFSETS[attacker as usize][0];

            let excluded = map < 0;
            luts[attacker as usize][attacked as usize][0] =
                if excluded { DIMENSIONS as u32 } else { feature };
            luts[attacker as usize][attacked as usize][1] = if excluded || semi_excluded {
                DIMENSIONS as u32
            } else {
                feature
            };
            j += 1;
        }
        i += 1;
    }
    luts
}

const INDEX_LUT1: [[[u32; 2]; PIECE_NB]; PIECE_NB] = init_index_lut1();

/// `index_lut2[attacker][from][to]`: how many of the attacker's pseudo-attacks
/// sit strictly below `to`, i.e. the rank of `to` within its attack set.
const fn init_index_lut2() -> [[[u8; 64]; 64]; PIECE_NB] {
    let mut luts = [[[0u8; 64]; 64]; PIECE_NB];
    let mut i = 0usize;
    while i < ALL_PIECES.len() {
        let piece = ALL_PIECES[i];
        let pt = type_of(piece);
        let mut from = 0usize;
        while from < 64 {
            let attacks = if pt == PAWN {
                let c = if piece < 8 {
                    Color::White
                } else {
                    Color::Black
                };
                pawn_attacks(c, from)
            } else {
                pseudo_attacks(pt, from)
            };
            let mut to = 0usize;
            while to < 64 {
                luts[piece as usize][from][to] = (((1u64 << to) - 1) & attacks).count_ones() as u8;
                to += 1;
            }
            from += 1;
        }
        i += 1;
    }
    luts
}

const INDEX_LUT2: [[[u8; 64]; 64]; PIECE_NB] = init_index_lut2();

// --- index --------------------------------------------------------------------

/// The feature index of `attacker` on `from` threatening `attacked` on `to`, as
/// seen from `perspective` with own king on `ksq`.
///
/// A return value of `Dimensions` or more means "not a feature of this set" and
/// the caller drops it. The `u16` truncation of the final sum is deliberate and
/// load-bearing: the excluded combinations start from `Dimensions` rather than
/// from a sentinel, so their sum can wrap — Stockfish truncates the same way and
/// the `push_if_lt` filter is what discards the result, so this must match
/// bit-for-bit.
#[inline]
pub fn make_index(
    perspective: Color,
    attacker: u8,
    from: usize,
    to: usize,
    attacked: u8,
    ksq: usize,
) -> u16 {
    let orientation = ORIENT_TBL[ksq] ^ (56 * perspective.idx());
    let from_oriented = from ^ orientation;
    let to_oriented = to ^ orientation;

    let swap = 8 * perspective.idx() as u8;
    let attacker_oriented = (attacker ^ swap) as usize;
    let attacked_oriented = (attacked ^ swap) as usize;
    let from_before_to = usize::from(from_oriented < to_oriented);

    (INDEX_LUT1[attacker_oriented][attacked_oriented][from_before_to]
        + OFFSETS[attacker_oriented][from_oriented]
        + u32::from(INDEX_LUT2[attacker_oriented][from_oriented][to_oriented])) as u16
}

// --- active indices -----------------------------------------------------------

/// The four pawn capture directions, as (colour, shift function, signed
/// direction).
///
/// `direction` is Stockfish's `Direction`: `from = to - direction`, with `+9` for
/// north-east, `+7` for north-west, `−9` for south-west and `−7` for
/// south-east. Getting the sign wrong silently produces indices for squares the
/// pawn is not on, so it is passed through as the same value the shift uses.
#[allow(clippy::too_many_arguments)]
fn process_pawn_attacks(
    perspective: Color,
    board: &Board,
    ksq: usize,
    c: Color,
    shift: fn(Bitboard) -> Bitboard,
    direction: i32,
    active: &mut IndexList,
) {
    let mut attacks = shift(board.pawns(c)) & board.pieces_of_many(KNIGHT, ROOK);
    while attacks != 0 {
        let to = attacks.trailing_zeros() as usize;
        attacks &= attacks - 1;
        let from = (to as i32 - direction) as usize;
        debug_assert!(from < 64, "pawn capture from {to} by {direction}");
        let attacked = board.piece_on(to);
        let attacker = make_piece(c.idx(), PAWN);
        active.push_if_lt(
            make_index(perspective, attacker, from, to, attacked, ksq),
            DIMENSIONS,
        );
    }
}

/// Every threat the position contains, from `perspective`'s point of view.
///
/// Emitted in Stockfish's order (White north-east, White north-west, Black
/// south-west, Black south-east, then the sliders), which keeps the dirty-list
/// occupancy identical even though the accumulator is a plain sum.
pub fn append_active_indices(perspective: Color, board: &Board, active: &mut IndexList) {
    let ksq = board.king_square(perspective);
    let occupied = board.pieces();
    let minor_slider_targets = board.pieces_of_four(PAWN, KNIGHT, BISHOP, ROOK);
    let queen_targets = board.pieces_of_five(PAWN, KNIGHT, BISHOP, ROOK, QUEEN);

    process_pawn_attacks(
        perspective,
        board,
        ksq,
        Color::White,
        shift_north_east,
        9,
        active,
    );
    process_pawn_attacks(
        perspective,
        board,
        ksq,
        Color::White,
        shift_north_west,
        7,
        active,
    );
    process_pawn_attacks(
        perspective,
        board,
        ksq,
        Color::Black,
        shift_south_west,
        -9,
        active,
    );
    process_pawn_attacks(
        perspective,
        board,
        ksq,
        Color::Black,
        shift_south_east,
        -7,
        active,
    );

    for c in Color::ALL {
        for pt in [KNIGHT, BISHOP, ROOK, QUEEN] {
            let attacker = make_piece(c.idx(), pt);
            let mut bb = board.pieces_of(pt) & board.pieces_of_color(c);
            let targets = if pt == KNIGHT || pt == QUEEN {
                queen_targets
            } else {
                minor_slider_targets
            };
            while bb != 0 {
                let from = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                let mut attacks = attacks_bb(pt, from, occupied) & targets;
                while attacks != 0 {
                    let to = attacks.trailing_zeros() as usize;
                    attacks &= attacks - 1;
                    let attacked = board.piece_on(to);
                    active.push_if_lt(
                        make_index(perspective, attacker, from, to, attacked, ksq),
                        DIMENSIONS,
                    );
                }
            }
        }
    }
}

// --- changed indices ----------------------------------------------------------

/// Appends the feature indices the threats in `diff` added and removed.
#[inline]
pub fn append_changed_indices(
    perspective: Color,
    ksq: usize,
    diff: &DirtyThreats,
    removed: &mut IndexList,
    added: &mut IndexList,
) {
    for dirty in diff.list.as_slice() {
        let index = make_index(
            perspective,
            dirty.pc(),
            dirty.pc_sq(),
            dirty.threatened_sq(),
            dirty.threatened_pc(),
            ksq,
        );
        let insert = if dirty.add() {
            &mut *added
        } else {
            &mut *removed
        };
        insert.push_if_lt(index, DIMENSIONS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::nnue::features::half_ka_v2_hm;
    use crate::nnue::types::{B_KNIGHT, B_PAWN, B_ROOK, DirtyThreat, W_KING, W_PAWN, W_ROOK};
    use std::collections::HashSet;

    // FEN square helpers (Stockfish numbering), spelled out for readability.
    const A1: usize = 0;
    const E1: usize = 4;
    const E2: usize = 12;
    const D3: usize = 19;
    const F3: usize = 21;
    const D4: usize = 27;
    const H8: usize = 63;

    #[test]
    fn orient_table_is_the_mirror_image_of_half_ka() {
        for s in 0..64 {
            assert_ne!(
                ORIENT_TBL[s],
                half_ka_v2_hm::ORIENT_TBL[s],
                "square {s} should mirror the other way"
            );
        }
    }

    #[test]
    fn offset_blocks_tile_the_whole_index_space() {
        // Every attacker block must start where the previous one ended, and the
        // last block must end exactly at DIMENSIONS.
        let mut running = 0u32;
        for piece in ALL_PIECES {
            let helper = HELPER_OFFSETS[piece as usize];
            assert_eq!(helper[1], running, "block base of {piece}");
            running += NUM_VALID_TARGETS[piece as usize] * helper[0];
        }
        assert_eq!(running as usize, DIMENSIONS);
    }

    #[test]
    fn index_lut1_excludes_impossible_pairs() {
        // Pawn -> pawn is excluded outright.
        assert_eq!(
            INDEX_LUT1[W_PAWN as usize][W_PAWN as usize][0],
            DIMENSIONS as u32
        );
        // A king never threatens anything in this set.
        assert_eq!(
            INDEX_LUT1[W_KING as usize][B_PAWN as usize][0],
            DIMENSIONS as u32
        );
        // Rook -> pawn is fine (a file or a rank away).
        assert!(INDEX_LUT1[W_ROOK as usize][B_PAWN as usize][0] < DIMENSIONS as u32);
        // A white rook threatening a white rook with from < to is semi-excluded.
        assert_eq!(
            INDEX_LUT1[W_ROOK as usize][W_ROOK as usize][1],
            DIMENSIONS as u32
        );
        assert!(INDEX_LUT1[W_ROOK as usize][W_ROOK as usize][0] < DIMENSIONS as u32);
    }

    #[test]
    fn distinct_threats_get_distinct_indices() {
        // The whole point of the three-table construction: two different
        // (attacker, attacked, from, to) tuples that both survive the filter
        // must land on different features. A collision would silently corrupt
        // the accumulator, because the weights are addressed by index alone.
        //
        // Pawns only get index space on ranks 2..7 — `init_threat_offsets`
        // allocates nothing for the first and last rank, so a rank-1 "pawn"
        // would alias onto the rank-2 block. Skip those, as Stockfish does.
        for ksq in [A1, E1, D4, H8] {
            for perspective in Color::ALL {
                let mut seen: HashSet<u32> = HashSet::new();
                for attacker in ALL_PIECES {
                    let pt = type_of(attacker);
                    for from in 0..64 {
                        if pt == PAWN && !(8..=55).contains(&from) {
                            continue;
                        }
                        let pseudo = if pt == PAWN {
                            let c = if attacker < 8 {
                                Color::White
                            } else {
                                Color::Black
                            };
                            pawn_attacks(c, from)
                        } else {
                            pseudo_attacks(pt, from)
                        };
                        let mut tos = pseudo;
                        while tos != 0 {
                            let to = tos.trailing_zeros() as usize;
                            tos &= tos - 1;
                            for attacked in ALL_PIECES {
                                let idx =
                                    make_index(perspective, attacker, from, to, attacked, ksq);
                                if (idx as usize) < DIMENSIONS {
                                    assert!(
                                        seen.insert(u32::from(idx)),
                                        "collision at {idx}: {attacker} on {from} \
                                         attacks {attacked} on {to}, {perspective:?} ksq {ksq}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// `index_lut1[..][from < to]` is set to `Dimensions` whenever the
    /// attacker and the attacked piece share a piece type and are either
    /// enemies or are not pawns. Of the two orders of such a mutual threat
    /// exactly one survives and the pair folds onto a single feature. That is
    /// the only deduplication the set performs, and it is what stops a pair of
    /// knights — friendly or not — from claiming two features for one
    /// relationship. Relations the `map` drops outright are dead in both
    /// orders.
    #[test]
    fn semi_excluded_threats_fold_to_one_feature() {
        fn map_of(attacker: u8, attacked: u8) -> i8 {
            MAP[(type_of(attacker) - 1) as usize][(type_of(attacked) - 1) as usize]
        }
        for ksq in [A1, E1, D4, H8] {
            for perspective in Color::ALL {
                for attacker in ALL_PIECES {
                    for attacked in ALL_PIECES {
                        let enemy = (attacker ^ attacked) == 8;
                        let semi = type_of(attacker) == type_of(attacked)
                            && (enemy || type_of(attacker) != PAWN);
                        for from in 0..64 {
                            for to in 0..64 {
                                if from == to {
                                    continue;
                                }
                                // Pawn index space exists only on ranks 2..7;
                                // outside it the offsets are all zero and the
                                // "threat" aliases onto the rank-2 block.
                                if type_of(attacker) == PAWN
                                    && !((8..=55).contains(&from) && (8..=55).contains(&to))
                                {
                                    continue;
                                }
                                let fwd =
                                    make_index(perspective, attacker, from, to, attacked, ksq);
                                let rev =
                                    make_index(perspective, attacker, to, from, attacked, ksq);
                                let fwd_live = (fwd as usize) < DIMENSIONS;
                                let rev_live = (rev as usize) < DIMENSIONS;
                                if map_of(attacker, attacked) < 0 {
                                    assert!(
                                        !fwd_live && !rev_live,
                                        "{attacker} -> {attacked} is excluded by the map \
                                         in both orders"
                                    );
                                } else if semi {
                                    // Exactly one of the two orders survives:
                                    // `from_oriented < to_oriented` picks the
                                    // semi-excluded slot.
                                    assert_ne!(
                                        fwd_live, rev_live,
                                        "{attacker} on {from} vs {attacked} on {to}: \
                                         exactly one order must survive"
                                    );
                                } else {
                                    // A different type or colour is a genuinely
                                    // different relationship in each direction,
                                    // and both directions are live.
                                    assert!(
                                        fwd_live && rev_live,
                                        "{attacker} on {from} vs {attacked} on {to}: \
                                         both orders must survive"
                                    );
                                    assert_ne!(
                                        fwd, rev,
                                        "different relations must be different features"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_last_valid_index_is_dimensions_minus_one() {
        let mut best = 0u32;
        for ksq in [A1, E1, H8] {
            for attacker in ALL_PIECES {
                for from in 0..64 {
                    for to in 0..64 {
                        for attacked in ALL_PIECES {
                            let idx = make_index(Color::White, attacker, from, to, attacked, ksq);
                            if (idx as usize) < DIMENSIONS {
                                best = best.max(u32::from(idx));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            best as usize,
            DIMENSIONS - 1,
            "the space must be used exactly"
        );
    }

    #[test]
    fn pawn_threats_only_target_knights_and_rooks() {
        // White pawn e2 sees d3 and f3. A black pawn on d3 must be ignored
        // (`MAP[PAWN][PAWN] == -1`) while a black knight on f3 counts.
        let pos = Position::from_fen("7k/8/8/8/8/3p1n2/4P3/K7 w - - 0 1").unwrap();
        let board = crate::nnue::board::Board::from_position(&pos);
        let mut active = IndexList::new();
        append_active_indices(Color::White, &board, &mut active);

        let ksq = board.king_square(Color::White);
        let to_knight = make_index(Color::White, W_PAWN, E2, F3, B_KNIGHT, ksq);
        let to_pawn = make_index(Color::White, W_PAWN, E2, D3, B_PAWN, ksq);

        assert!(
            (to_knight as usize) < DIMENSIONS,
            "pawn -> knight is a live feature"
        );
        assert!(
            (to_pawn as usize) >= DIMENSIONS,
            "pawn -> pawn is excluded by the map"
        );
        assert!(active.as_slice().contains(&to_knight), "pawn -> knight");
        assert!(!active.as_slice().contains(&to_pawn), "pawn -> pawn");
        assert_eq!(active.size(), 1, "d3 holds a pawn and must be skipped");

        // A rook target counts just as a knight does: black rook on d3.
        let pos_r = Position::from_fen("7k/8/8/8/8/3r4/4P3/K7 w - - 0 1").unwrap();
        let board_r = crate::nnue::board::Board::from_position(&pos_r);
        let mut active_r = IndexList::new();
        append_active_indices(Color::White, &board_r, &mut active_r);
        let ksq = board_r.king_square(Color::White);
        let to_rook = make_index(Color::White, W_PAWN, E2, D3, B_ROOK, ksq);
        assert!((to_rook as usize) < DIMENSIONS, "pawn -> rook is live");
        assert!(active_r.as_slice().contains(&to_rook), "pawn -> rook");
        assert_eq!(active_r.size(), 1);
    }

    #[test]
    fn active_indices_are_in_range_and_non_trivial() {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let board = crate::nnue::board::Board::from_position(&pos);
        for p in Color::ALL {
            let mut active = IndexList::new();
            append_active_indices(p, &board, &mut active);
            assert!(active.size() > 10, "{p:?} should see many threats");
            for i in active.as_slice() {
                assert!((*i as usize) < DIMENSIONS, "index {i} out of range");
            }
        }
    }

    #[test]
    fn changed_indices_split_by_the_add_flag() {
        let ksq = A1;
        let rook = W_ROOK;
        // A white rook on b1 threatening a black pawn on b4: the threat exists
        // before the pawn arrives and is gone afterwards, so the same dirty
        // record with the flag flipped lands in the other list.
        let threat = DirtyThreat::new(rook, B_PAWN, 1, 27, true);
        let mut added = IndexList::new();
        let mut removed = IndexList::new();
        let mut diff = DirtyThreats::new();
        diff.push(threat);
        append_changed_indices(Color::White, ksq, &diff, &mut removed, &mut added);
        assert_eq!(added.size(), 1, "an added threat goes to `added`");
        assert_eq!(removed.size(), 0);

        let mut diff2 = DirtyThreats::new();
        diff2.push(DirtyThreat::new(rook, B_PAWN, 1, 27, false));
        let mut added2 = IndexList::new();
        let mut removed2 = IndexList::new();
        append_changed_indices(Color::White, ksq, &diff2, &mut removed2, &mut added2);
        assert_eq!(added2.size(), 0);
        assert_eq!(removed2.size(), 1, "a removed threat goes to `removed`");
        assert_eq!(added[0], removed2[0], "the flag must not change the index");

        // An excluded relation produces nothing at all, whichever list it would
        // have gone to: `push_if_lt` drops it.
        let mut diff3 = DirtyThreats::new();
        diff3.push(DirtyThreat::new(W_PAWN, B_PAWN, E2, D3, true));
        let mut added3 = IndexList::new();
        let mut removed3 = IndexList::new();
        append_changed_indices(Color::White, ksq, &diff3, &mut removed3, &mut added3);
        assert!(added3.is_empty() && removed3.is_empty(), "pawn -> pawn");
    }
}

//! `PP_3Wide`: which pawn stands next to which.
//!
//! For every pair of pawns that could form a "pawn pair" (own file or a
//! neighbouring file, ranks 2..7) there is one feature, regardless of which
//! pawn is which — the pair is sorted by a 96-entry pawn id and placed in the
//! upper triangle of a 96×96 matrix. Because a pair feature is unordered, the
//! index does not depend on the order the two pawns are visited in, which is
//! what lets a single dirty list drive both halves of an incremental update.
//!
//! The feature indices live in `[IndexBase, IndexBase + Dimensions)`, i.e. right
//! after the `FullThreats` block, so one index addresses either set.

use crate::nnue::attacks::pawn_pair_bb;
use crate::nnue::board::Board;
use crate::nnue::features::full_threats;
use crate::nnue::types::{Color, DirtyPawnPairs, ValueList};

/// Hash the trainer embeds for this feature set.
pub const HASH_VALUE: u32 = 0x86f2_b1dd;

/// `COLOR_NB * 48` — 48 usable squares per colour, ranks 2..7.
pub const PAWN_IDS: usize = 2 * 48;

/// Number of unordered pawn pairs.
pub const DIMENSIONS: usize = PAWN_IDS * (PAWN_IDS - 1) / 2;

/// Where this feature set starts in the shared threat/pawn-pair index space.
pub const INDEX_BASE: usize = 59808;

pub type IndexList = ValueList<u16, 256>;

/// `SF_A2` (= 8) — the first rank a pawn can occupy.
const SQ_A2: usize = 8;
/// `SF_H7` (= 55) — the last.
const SQ_H7: usize = 55;

/// The id of a pawn: 48 slots per colour.
#[inline]
const fn make_pawn_id(color: usize, sq: usize) -> u32 {
    assert!(sq >= SQ_A2 && sq <= SQ_H7, "a pawn pair needs ranks 2..7");
    48 * color as u32 + (sq - SQ_A2) as u32
}

/// The feature index of the pair (`color`, `from`) × (`paired_color`, `to`).
///
/// The largest possible value is `95 * 94 / 2 + 94 + IndexBase = 64 367`, inside
/// the `u16` index space, so no truncation happens here.
#[inline]
pub fn make_index(
    perspective: Color,
    color: Color,
    from: usize,
    to: usize,
    paired_color: Color,
    ksq: usize,
) -> u16 {
    // The two pawns of a pair are always on different squares and different
    // files, so `id_a != id_b` and the address below stays inside the block.
    // Pairing a pawn with itself would run one past the end of the block.
    debug_assert!(from != to, "a pawn cannot be its own neighbour");
    let orientation = full_threats::ORIENT_TBL[ksq] ^ (56 * perspective.idx());
    let from_oriented = from ^ orientation;
    let to_oriented = to ^ orientation;

    let id_a = make_pawn_id(color.idx() ^ perspective.idx(), from_oriented);
    let id_b = make_pawn_id(paired_color.idx() ^ perspective.idx(), to_oriented);
    let hi = if id_a > id_b { id_a } else { id_b };
    let lo = if id_a > id_b { id_b } else { id_a };

    (hi * (hi - 1) / 2 + lo + INDEX_BASE as u32) as u16
}

/// Every active pawn pair of the position, for one perspective.
pub fn append_active_indices(perspective: Color, board: &Board, active: &mut IndexList) {
    let ksq = board.king_square(perspective);
    let white = board.pawns(Color::White);
    let black = board.pawns(Color::Black);

    let mut bb = white;
    while bb != 0 {
        let from = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let band = pawn_pair_bb(from);
        let mut ww = band & bb;
        while ww != 0 {
            active.push(make_index(
                perspective,
                Color::White,
                from,
                ww.trailing_zeros() as usize,
                Color::White,
                ksq,
            ));
            ww &= ww - 1;
        }
        let mut wb = band & black;
        while wb != 0 {
            active.push(make_index(
                perspective,
                Color::White,
                from,
                wb.trailing_zeros() as usize,
                Color::Black,
                ksq,
            ));
            wb &= wb - 1;
        }
    }

    let mut bb = black;
    while bb != 0 {
        let from = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let band = pawn_pair_bb(from);
        let mut bk = band & bb;
        while bk != 0 {
            active.push(make_index(
                perspective,
                Color::Black,
                from,
                bk.trailing_zeros() as usize,
                Color::Black,
                ksq,
            ));
            bk &= bk - 1;
        }
    }
}

/// Appends the pair features a pawn move created (`added`) and destroyed
/// (`removed`).
pub fn append_changed_indices(
    perspective: Color,
    ksq: usize,
    diff: &DirtyPawnPairs,
    removed: &mut IndexList,
    added: &mut IndexList,
) {
    let (white_before, black_before) = (diff.before[0], diff.before[1]);
    let (white_after, black_after) = (diff.after[0], diff.after[1]);

    if white_before == white_after && black_before == black_after {
        return;
    }

    let generate =
        |updated_w: u64, updated_b: u64, pawns_w: u64, pawns_b: u64, out: &mut IndexList| {
            let unchanged = (pawns_w | pawns_b) & !(updated_w | updated_b);
            let mut u = updated_w | updated_b;
            while u != 0 {
                let a = u.trailing_zeros() as usize;
                u &= u - 1;
                let mask = pawn_pair_bb(a) & (unchanged | u);
                let a_col = if pawns_b & (1u64 << a) != 0 {
                    Color::Black
                } else {
                    Color::White
                };
                let mut pb = pawns_b & mask;
                while pb != 0 {
                    out.push(make_index(
                        perspective,
                        a_col,
                        a,
                        pb.trailing_zeros() as usize,
                        Color::Black,
                        ksq,
                    ));
                    pb &= pb - 1;
                }
                let mut pw = pawns_w & mask;
                while pw != 0 {
                    out.push(make_index(
                        perspective,
                        a_col,
                        a,
                        pw.trailing_zeros() as usize,
                        Color::White,
                        ksq,
                    ));
                    pw &= pw - 1;
                }
            }
        };

    generate(
        white_after & !white_before,
        black_after & !black_before,
        white_after,
        black_after,
        added,
    );
    generate(
        white_before & !white_after,
        black_before & !black_after,
        white_before,
        black_before,
        removed,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::nnue::features::full_threats::DIMENSIONS as THREAT_DIMENSIONS;
    use std::collections::HashSet;

    const A1: usize = 0;
    const A2: usize = 8;
    const A4: usize = 24;
    const B4: usize = 25;
    const C4: usize = 26;
    const D4: usize = 27;
    const A7: usize = 48;
    const H7: usize = 55;

    #[test]
    fn index_base_continues_the_threat_block() {
        assert_eq!(INDEX_BASE, THREAT_DIMENSIONS);
        assert_eq!(INDEX_BASE + DIMENSIONS, 59808 + 4560);
    }

    #[test]
    fn a_pair_is_unordered() {
        let ksq = A1;
        let a = make_index(Color::White, Color::White, A4, B4, Color::White, ksq);
        let b = make_index(Color::White, Color::White, B4, A4, Color::White, ksq);
        assert_eq!(a, b, "swapping the two pawns must not change the index");
    }

    #[test]
    fn indices_stay_inside_the_pawn_pair_block() {
        // `from != to` is a precondition: a pawn never pairs with itself, and
        // with a single id the formula walks straight off the end of the block.
        for ksq in 0..64 {
            for from in A2..=H7 {
                for to in A2..=H7 {
                    if from == to {
                        continue;
                    }
                    for c in Color::ALL {
                        for pc in Color::ALL {
                            let idx = make_index(Color::White, c, from, to, pc, ksq) as usize;
                            assert!(
                                idx >= INDEX_BASE && idx < INDEX_BASE + DIMENSIONS,
                                "{c:?}/{pc:?} {from}->{to} ksq {ksq} -> {idx}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn distinct_pairs_get_distinct_indices() {
        // The feature is the *unordered* pair of pawn ids, and the id of a pawn
        // is `48 * oriented_colour + oriented_square - 8`. So the index must be
        // exactly the canonical upper-triangular address of that pair — which
        // proves both that nothing collides and that all 4560 are reachable.
        let mut canonical: HashSet<u32> = HashSet::new();
        for hi in 1..PAWN_IDS {
            for lo in 0..hi {
                canonical.insert((hi * (hi - 1) / 2 + lo + INDEX_BASE) as u32);
            }
        }
        assert_eq!(canonical.len(), DIMENSIONS, "4560 distinct pair features");

        // A king square on the a..d half gives orientation 0, which already
        // reaches all 96 pawn ids.
        for from in A2..=H7 {
            for to in A2..=H7 {
                if from == to {
                    continue;
                }
                for c in Color::ALL {
                    for pc in Color::ALL {
                        let id_a = make_pawn_id(c.idx(), from) as usize;
                        let id_b = make_pawn_id(pc.idx(), to) as usize;
                        let (hi, lo) = if id_a > id_b {
                            (id_a, id_b)
                        } else {
                            (id_b, id_a)
                        };
                        let want = (hi * (hi - 1) / 2 + lo + INDEX_BASE) as u32;
                        let got = make_index(Color::White, c, from, to, pc, A1);
                        assert_eq!(u32::from(got), want, "{c:?} {from} with {pc:?} {to}");
                        assert!(canonical.contains(&u32::from(got)));
                    }
                }
            }
        }

        // The cross-colour same-square pairs are part of the space even though
        // two pawns can never share a square on a real board; they are what
        // makes the count 4560 rather than 4512.
        let same = make_index(Color::White, Color::White, A2, A2, Color::Black, A1);
        assert!(canonical.contains(&u32::from(same)));
    }

    #[test]
    fn the_block_is_filled_exactly() {
        let mut best = 0usize;
        let mut worst = usize::MAX;
        for ksq in 0..64 {
            for c in Color::ALL {
                for pc in Color::ALL {
                    for from in A2..=H7 {
                        for to in A2..=H7 {
                            if from == to {
                                continue;
                            }
                            let idx = make_index(Color::White, c, from, to, pc, ksq) as usize;
                            best = best.max(idx);
                            worst = worst.min(idx);
                        }
                    }
                }
            }
        }
        assert_eq!(best, INDEX_BASE + DIMENSIONS - 1, "the top of the block");
        assert_eq!(worst, INDEX_BASE, "the bottom of the block");
    }

    #[test]
    fn active_indices_cover_every_adjacent_pawn_pair() {
        // The band is the pawn's own file plus its two neighbouring files, and
        // each partner is only counted once — the scan pops the attacker off the
        // board before looking for partners. Four white pawns on a4 b4 c4 d4
        // therefore give the three chain links (a4,b4) (b4,c4) (c4,d4) and not
        // the three non-adjacent combinations.
        let pos = Position::from_fen("4k3/8/8/8/PPPP4/8/8/4K3 w - - 0 1").unwrap();
        let board = crate::nnue::board::Board::from_position(&pos);
        let mut active = IndexList::new();
        append_active_indices(Color::White, &board, &mut active);
        assert_eq!(active.size(), 3, "only neighbouring files pair up");
        let ksq = board.king_square(Color::White);
        for i in active.as_slice() {
            assert!(
                active.as_slice().iter().filter(|j| *j == i).count() == 1,
                "index {i} appears twice"
            );
        }
        let expect = [
            make_index(Color::White, Color::White, A4, B4, Color::White, ksq),
            make_index(Color::White, Color::White, B4, C4, Color::White, ksq),
            make_index(Color::White, Color::White, C4, D4, Color::White, ksq),
        ];
        for e in expect {
            assert!(active.as_slice().contains(&e), "missing {e}");
        }
        // a4 and c4 are two files apart, so they must not pair.
        assert!(!active.as_slice().contains(&make_index(
            Color::White,
            Color::White,
            A4,
            C4,
            Color::White,
            ksq
        )));

        // Black-to-move perspective sees exactly the same three pairs: the
        // king square only rotates the board.
        let pos_b = Position::from_fen("4k3/8/8/8/PPPP4/8/8/4K3 b - - 0 1").unwrap();
        let board_b = crate::nnue::board::Board::from_position(&pos_b);
        let mut active_b = IndexList::new();
        append_active_indices(Color::Black, &board_b, &mut active_b);
        assert_eq!(active_b.size(), 3);

        // A black pawn inside the white band pairs with it, and only with the
        // pawns whose own band reaches it: a4 is inside b4's band (files a..c)
        // but outside c4's and d4's, so `pPPP4` replaces no pair and adds just
        // the one mixed feature (b4, a4) — three in total, not four.
        let pos_m = Position::from_fen("4k3/8/8/8/pPPP4/8/8/4K3 w - - 0 1").unwrap();
        let board_m = crate::nnue::board::Board::from_position(&pos_m);
        let mut active_m = IndexList::new();
        append_active_indices(Color::White, &board_m, &mut active_m);
        assert_eq!(active_m.size(), 3);
        let ksq = board_m.king_square(Color::White);
        assert!(active_m.as_slice().contains(&make_index(
            Color::White,
            Color::White,
            B4,
            A4,
            Color::Black,
            ksq
        )));
        // A mixed pair is always attributed to the *white* attacker, so a lone
        // black pawn beside a white one pairs, and a black pawn two files away
        // does not: `pP6` is a4=p b4=P, `pP1P4` is a4=p b4=P d4=P.
        let pos_n = Position::from_fen("4k3/8/8/8/pP6/8/8/4K3 w - - 0 1").unwrap();
        let board_n = crate::nnue::board::Board::from_position(&pos_n);
        let mut active_n = IndexList::new();
        append_active_indices(Color::White, &board_n, &mut active_n);
        assert_eq!(active_n.size(), 1);
        let ksq = board_n.king_square(Color::White);
        assert!(active_n.as_slice().contains(&make_index(
            Color::White,
            Color::White,
            B4,
            A4,
            Color::Black,
            ksq
        )));

        let pos_n2 = Position::from_fen("4k3/8/8/8/pP1P4/8/8/4K3 w - - 0 1").unwrap();
        let board_n2 = crate::nnue::board::Board::from_position(&pos_n2);
        let mut active_n2 = IndexList::new();
        append_active_indices(Color::White, &board_n2, &mut active_n2);
        assert_eq!(active_n2.size(), 1, "a4 and d4 are two files apart");

        // Same-colour black pairs come from the second scan: black a4 and b4
        // plus a white c4 give the black pair (a4,b4) and the mixed (c4,b4).
        let pos_p = Position::from_fen("4k3/8/8/8/ppP5/8/8/4K3 w - - 0 1").unwrap();
        let board_p = crate::nnue::board::Board::from_position(&pos_p);
        let mut active_p = IndexList::new();
        append_active_indices(Color::White, &board_p, &mut active_p);
        assert_eq!(active_p.size(), 2);
        let ksq = board_p.king_square(Color::White);
        assert!(active_p.as_slice().contains(&make_index(
            Color::White,
            Color::Black,
            A4,
            B4,
            Color::Black,
            ksq
        )));
        assert!(active_p.as_slice().contains(&make_index(
            Color::White,
            Color::White,
            C4,
            B4,
            Color::Black,
            ksq
        )));
    }

    #[test]
    fn a_pawn_push_only_touches_its_own_pairs() {
        let pos = Position::from_fen("4k3/8/8/8/PPPP4/8/8/4K3 w - - 0 1").unwrap();
        let board = crate::nnue::board::Board::from_position(&pos);
        let before = {
            let mut l = IndexList::new();
            append_active_indices(Color::White, &board, &mut l);
            l
        };

        // Push a4 -> a5: (a4,b4) is destroyed and (a5,b4) is created, because
        // a5 is still inside b4's band. b4's other partners are untouched.
        let child = pos.make_child(pos.raw_move_from_uci("a4a5").unwrap());
        let after_board = crate::nnue::board::Board::from_position(&child);
        let diff = DirtyPawnPairs {
            before: [board.pawns(Color::White), board.pawns(Color::Black)],
            after: [
                after_board.pawns(Color::White),
                after_board.pawns(Color::Black),
            ],
        };
        let ksq = board.king_square(Color::White);
        let mut removed = IndexList::new();
        let mut added = IndexList::new();
        append_changed_indices(Color::White, ksq, &diff, &mut removed, &mut added);
        assert_eq!(removed.size(), 1);
        assert_eq!(added.size(), 1);
        let a4_b4 = make_index(Color::White, Color::White, A4, B4, Color::White, ksq);
        let a5_b4 = make_index(Color::White, Color::White, 32, B4, Color::White, ksq);
        assert_eq!(removed[0], a4_b4, "(a4,b4) is destroyed");
        assert_eq!(added[0], a5_b4, "(a5,b4) is created");
        assert!(before.as_slice().contains(&a4_b4));
    }

    /// The strongest statement available: for *every* legal move the emitted
    /// `removed`/`added` lists must be exactly the symmetric difference of the
    /// active index sets before and after. A missing or spurious index corrupts
    /// the accumulator, so this walks a wide slice of the move space.
    ///
    /// Moves that move the *perspective's own* king are skipped: the king
    /// square enters the index formula, so such a move legitimately rewrites
    /// every index. That case is covered by the accumulator's
    /// "incremental update equals full refresh" test, which refreshes instead.
    #[test]
    fn changed_indices_are_exactly_the_symmetric_difference() {
        const FENS: [&str; 8] = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "4k3/8/8/8/PPPP4/8/8/4K3 w - - 0 1",
            "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
            "8/8/8/8/p1p1p1p1/P1P1P1P1/8/4K2k w - - 0 1",
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        ];
        let mut checked = 0usize;
        for fen in FENS {
            let pos = Position::from_fen(fen).unwrap();
            let board = crate::nnue::board::Board::from_position(&pos);
            for perspective in Color::ALL {
                let ksq_before = board.king_square(perspective);
                let mut before = IndexList::new();
                append_active_indices(perspective, &board, &mut before);
                let before_sorted: Vec<u16> = {
                    let mut v = before.as_slice().to_vec();
                    v.sort_unstable();
                    v
                };

                for m in pos.legal_moves().iter() {
                    let child = pos.make_child(m);
                    let cb = crate::nnue::board::Board::from_position(&child);
                    if cb.king_square(perspective) != ksq_before {
                        continue;
                    }
                    let diff = DirtyPawnPairs {
                        before: [board.pawns(Color::White), board.pawns(Color::Black)],
                        after: [cb.pawns(Color::White), cb.pawns(Color::Black)],
                    };
                    let mut removed = IndexList::new();
                    let mut added = IndexList::new();
                    append_changed_indices(
                        perspective,
                        ksq_before,
                        &diff,
                        &mut removed,
                        &mut added,
                    );
                    checked += 1;

                    let mut after = IndexList::new();
                    append_active_indices(perspective, &cb, &mut after);
                    let after_sorted: Vec<u16> = {
                        let mut v = after.as_slice().to_vec();
                        v.sort_unstable();
                        v
                    };

                    // Every removed index was active before and is gone after;
                    // every added index is new and is active afterwards.
                    let mut expect_removed: Vec<u16> = before_sorted
                        .iter()
                        .copied()
                        .filter(|i| !after_sorted.contains(i))
                        .collect();
                    let mut expect_added: Vec<u16> = after_sorted
                        .iter()
                        .copied()
                        .filter(|i| !before_sorted.contains(i))
                        .collect();
                    let mut got_removed = removed.as_slice().to_vec();
                    let mut got_added = added.as_slice().to_vec();
                    expect_removed.sort_unstable();
                    expect_added.sort_unstable();
                    got_removed.sort_unstable();
                    got_added.sort_unstable();
                    assert_eq!(
                        got_removed, expect_removed,
                        "{fen} {m:?} {perspective:?}: removed set differs"
                    );
                    assert_eq!(
                        got_added, expect_added,
                        "{fen} {m:?} {perspective:?}: added set differs"
                    );
                }
            }
        }
        assert!(checked > 250, "only checked {checked} moves");
    }

    #[test]
    fn an_unchanged_pawn_set_produces_nothing() {
        let pos = Position::from_fen("4k3/8/8/8/PPPP4/8/8/4K3 w - - 0 1").unwrap();
        let board = crate::nnue::board::Board::from_position(&pos);
        let diff = DirtyPawnPairs {
            before: [board.pawns(Color::White), board.pawns(Color::Black)],
            after: [board.pawns(Color::White), board.pawns(Color::Black)],
        };
        let mut removed = IndexList::new();
        let mut added = IndexList::new();
        append_changed_indices(Color::White, A1, &diff, &mut removed, &mut added);
        assert!(removed.is_empty() && added.is_empty());
    }

    #[test]
    fn pawn_ids_stay_inside_the_ranks_they_occupy() {
        // A white pawn on a2 is id 0, on a7 id 40 (six rank-steps above a2).
        assert_eq!(make_pawn_id(0, A2), 0);
        assert_eq!(make_pawn_id(0, A2 + 1), 1);
        assert_eq!(make_pawn_id(0, A7), 40);
        assert_eq!(make_pawn_id(0, H7), 47);
        assert_eq!(make_pawn_id(1, A2), 48);
    }
}

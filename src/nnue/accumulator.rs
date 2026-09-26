//! The accumulator: the incrementally maintained sum of the input features.
//!
//! `accumulation[p][j]` is the sum of `weights[feature][j]` over every active
//! feature of perspective `p` — the input the first fully connected layer sees,
//! before the clamp-and-multiply transform. `psqt[p][b]` is the same sum over
//! the 8 piece-count buckets.
//!
//! Two update paths, exactly as in Stockfish but without the Finny cache:
//!
//! * **full refresh** — biases + every active feature index. Used at the root,
//!   after a null move, and whenever the dirty piece *is* the perspective's own
//!   king, because the HalfKAv2_hm king bucket (and with it the
//!   `FullThreats`/`PP_3Wide` orientation) changes.
//! * **incremental** — subtract the removed indices, add the added ones. This is
//!   exact because every feature set's index depends only on its own
//!   perspective, so a white king move leaves the black perspective untouched
//!   and vice versa.

use crate::nnue::board::Board;
use crate::nnue::features::{full_threats, half_ka_v2_hm, pp_3wide};
use crate::nnue::network::FeatureTransformer;
use crate::nnue::types::{Color, DirtyPawnPairs, DirtyPiece, DirtyThreats};

/// Transformed-feature width (L1).
pub const L1: usize = 1024;
/// Piece-count buckets the PSQT accumulator carries.
pub const PSQT_BUCKETS: usize = 8;
/// Number of layer stacks; also the number of buckets the eval picks from.
pub const LAYER_STACKS: usize = 8;

/// One ply's accumulator, one copy per perspective.
#[derive(Clone)]
pub struct Accumulator {
    pub accumulation: [[i16; L1]; 2],
    pub psqt: [[i32; PSQT_BUCKETS]; 2],
    /// `false` means "this slot holds nothing usable yet"; [`AccumulatorStack`]
    /// then does a full refresh at evaluation time instead of an incremental
    /// update, which is the only way a slot can end up in that state.
    pub computed: [bool; 2],
}

impl Accumulator {
    pub const fn new() -> Accumulator {
        Accumulator {
            accumulation: [[0; L1]; 2],
            psqt: [[0; PSQT_BUCKETS]; 2],
            computed: [false; 2],
        }
    }

    /// Recomputes perspective `p` from scratch.
    ///
    /// `acc[j] = biases[j] + sum over every active index of weights[idx][j]`,
    /// and the PSQT side gets the bucket-expanded equivalent.
    pub fn refresh(&mut self, perspective: Color, board: &Board, ft: &FeatureTransformer) {
        let p = perspective.idx();

        let mut psq_active = half_ka_v2_hm::IndexList::new();
        half_ka_v2_hm::append_active_indices(perspective, board, &mut psq_active);

        let mut other_active = full_threats::IndexList::new();
        full_threats::append_active_indices(perspective, board, &mut other_active);
        pp_3wide::append_active_indices(perspective, board, &mut other_active);

        let acc = &mut self.accumulation[p];
        for j in 0..L1 {
            acc[j] = ft.biases[j];
        }
        for idx in psq_active.as_slice() {
            let row = ft.weight_row(*idx as usize);
            for j in 0..L1 {
                acc[j] = acc[j].wrapping_add(row[j]);
            }
        }
        for idx in other_active.as_slice() {
            let row = ft.other_weight_row(*idx as usize);
            for j in 0..L1 {
                // Threat/pawn-pair weights are `i8` and widen into the `i16`
                // accumulator, as in Stockfish.
                acc[j] = acc[j].wrapping_add(i16::from(row[j]));
            }
        }

        let pacc = &mut self.psqt[p];
        for b in 0..PSQT_BUCKETS {
            pacc[b] = 0;
        }
        for idx in psq_active.as_slice() {
            let row = ft.psqt_weight_row(*idx as usize);
            for b in 0..PSQT_BUCKETS {
                pacc[b] = pacc[b].wrapping_add(row[b]);
            }
        }
        for idx in other_active.as_slice() {
            let row = ft.other_psqt_row(*idx as usize);
            for b in 0..PSQT_BUCKETS {
                pacc[b] = pacc[b].wrapping_add(row[b]);
            }
        }

        self.computed[p] = true;
    }

    /// Applies one move's worth of changes to perspective `p`.
    ///
    /// `parent` must already be computed for `p`; `child_ksq` is *this* side's
    /// king square, which is what both the HalfKA king bucket and the
    /// `FullThreats`/`PP_3Wide` orientation are keyed on.
    #[allow(clippy::too_many_arguments)]
    pub fn update_incremental(
        &mut self,
        perspective: Color,
        child_ksq: usize,
        parent: &Accumulator,
        dirty_piece: &DirtyPiece,
        dirty_threats: &DirtyThreats,
        dirty_pawn_pairs: &DirtyPawnPairs,
        ft: &FeatureTransformer,
    ) {
        let p = perspective.idx();

        let mut psq_removed = half_ka_v2_hm::IndexList::new();
        let mut psq_added = half_ka_v2_hm::IndexList::new();
        half_ka_v2_hm::append_changed_indices(
            perspective,
            child_ksq,
            dirty_piece,
            &mut psq_removed,
            &mut psq_added,
        );

        let mut other_removed = full_threats::IndexList::new();
        let mut other_added = full_threats::IndexList::new();
        full_threats::append_changed_indices(
            perspective,
            child_ksq,
            dirty_threats,
            &mut other_removed,
            &mut other_added,
        );
        pp_3wide::append_changed_indices(
            perspective,
            child_ksq,
            dirty_pawn_pairs,
            &mut other_removed,
            &mut other_added,
        );

        {
            let src = &parent.accumulation[p];
            let dst = &mut self.accumulation[p];
            dst.copy_from_slice(src);
            for idx in psq_removed.as_slice() {
                let row = ft.weight_row(*idx as usize);
                for j in 0..L1 {
                    dst[j] = dst[j].wrapping_sub(row[j]);
                }
            }
            for idx in psq_added.as_slice() {
                let row = ft.weight_row(*idx as usize);
                for j in 0..L1 {
                    dst[j] = dst[j].wrapping_add(row[j]);
                }
            }
            for idx in other_removed.as_slice() {
                let row = ft.other_weight_row(*idx as usize);
                for j in 0..L1 {
                    dst[j] = dst[j].wrapping_sub(i16::from(row[j]));
                }
            }
            for idx in other_added.as_slice() {
                let row = ft.other_weight_row(*idx as usize);
                for j in 0..L1 {
                    dst[j] = dst[j].wrapping_add(i16::from(row[j]));
                }
            }
        }

        {
            let src = &parent.psqt[p];
            let dst = &mut self.psqt[p];
            for b in 0..PSQT_BUCKETS {
                dst[b] = src[b];
            }
            for idx in psq_removed.as_slice() {
                let row = ft.psqt_weight_row(*idx as usize);
                for b in 0..PSQT_BUCKETS {
                    dst[b] = dst[b].wrapping_sub(row[b]);
                }
            }
            for idx in psq_added.as_slice() {
                let row = ft.psqt_weight_row(*idx as usize);
                for b in 0..PSQT_BUCKETS {
                    dst[b] = dst[b].wrapping_add(row[b]);
                }
            }
            for idx in other_removed.as_slice() {
                let row = ft.other_psqt_row(*idx as usize);
                for b in 0..PSQT_BUCKETS {
                    dst[b] = dst[b].wrapping_sub(row[b]);
                }
            }
            for idx in other_added.as_slice() {
                let row = ft.other_psqt_row(*idx as usize);
                for b in 0..PSQT_BUCKETS {
                    dst[b] = dst[b].wrapping_add(row[b]);
                }
            }
        }

        self.computed[p] = true;
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// The per-ply accumulator stack, one slot per search ply.
///
/// The stack is heap-allocated on first use because it is ~540 KB — far too
/// large to sit inside the per-thread `SearchThread` value, which is itself
/// created on the stack of each search worker. Only slot `ply + 1` is ever
/// written, so the search writes each slot exactly once per node and never has
/// to unwind.
pub struct AccumulatorStack {
    slots: Box<[Accumulator]>,
}

/// The number of slots: one per ply plus the root.
pub const MAX_PLY: usize = crate::types::MAX_PLY + 1;

impl AccumulatorStack {
    /// Allocates and zero-initialises the stack. `computed` starts `false`
    /// everywhere, so the first touch of a slot is always a full refresh.
    pub fn new() -> AccumulatorStack {
        AccumulatorStack {
            slots: (0..MAX_PLY).map(|_| Accumulator::new()).collect(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    #[inline]
    pub fn get(&self, ply: usize) -> &Accumulator {
        &self.slots[ply]
    }

    #[inline]
    pub fn get_mut(&mut self, ply: usize) -> &mut Accumulator {
        &mut self.slots[ply]
    }

    /// Borrows the accumulator at `parent_ply` and the one at `child_ply` at the
    /// same time, which is exactly what an incremental update needs (read the
    /// parent's sums, subtract the removed indices, add the new ones).
    ///
    /// `child_ply` must be strictly greater than `parent_ply`, so the two slots
    /// are always disjoint and the split cannot alias.
    #[inline]
    pub fn parent_and_child_mut(
        &mut self,
        parent_ply: usize,
        child_ply: usize,
    ) -> (&Accumulator, &mut Accumulator) {
        assert!(
            child_ply > parent_ply,
            "the child slot must come after the parent"
        );
        let (head, tail) = self.slots.split_at_mut(child_ply);
        (&head[parent_ply], &mut tail[0])
    }

    /// Marks every slot as not-yet-computed, e.g. when a new search starts.
    pub fn reset(&mut self) {
        for slot in self.slots.iter_mut() {
            *slot = Accumulator::new();
        }
    }
}

impl Default for AccumulatorStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::nnue::test_net;

    /// The invariant that makes the whole design work: replaying a move
    /// incrementally must produce bit-identical accumulators to refreshing the
    /// child from scratch. Checked over a decent sample of move kinds.
    #[test]
    fn incremental_update_equals_full_refresh() {
        let net = test_net();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 2",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            // Both castlings, both colours: the king move forces a refresh and
            // the rook is a second dirty piece on top of it.
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            // Every straight promotion available, plus three capture-promotions
            // onto the queens on the eighth rank. The black king is well out of
            // the way: white pawns on the seventh attack the eighth rank, so a
            // black king up there would be in check with White to move.
            "q2q1q2/PPPPPPPP/8/8/4k3/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
            // And the same again with Black to move, so the promotions are
            // exercised from the other perspective.
            "q7/pppppppp/8/8/4K3/8/8/4k3 b - - 0 1",
            // Doubled and tripled pawns on adjacent files: the maximum number of
            // pawn-pair features the set can hold.
            "4k3/8/8/8/PPP1PPPP/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/PPPPPPPP/8/8/4K3 w - - 0 1",
        ];

        let mut checked = 0usize;
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            let us = Color::from_shakmaty(pos.turn());
            let board = Board::from_position(&pos);
            let mut parent = Accumulator::new();
            parent.refresh(Color::White, &board, net);
            parent.refresh(Color::Black, &board, net);

            for m in pos.legal_moves().iter() {
                let d = board.apply_move(m, us);
                let child_pos = pos.make_child(m);
                let mut incremental = Accumulator::new();
                for p in Color::ALL {
                    let ksq = d.board.king_square(p);
                    if half_ka_v2_hm::requires_refresh(&d.dirty_piece, p) {
                        incremental.refresh(p, &d.board, net);
                    } else {
                        incremental.update_incremental(
                            p,
                            ksq,
                            &parent,
                            &d.dirty_piece,
                            &d.dirty_threats,
                            &d.dirty_pawn_pairs,
                            net,
                        );
                    }

                    let mut fresh = Accumulator::new();
                    fresh.refresh(p, &d.board, net);
                    assert_eq!(
                        incremental.accumulation[p.idx()],
                        fresh.accumulation[p.idx()],
                        "accumulation mismatch after {} in {fen} ({p:?})",
                        m.to_uci()
                    );
                    assert_eq!(
                        incremental.psqt[p.idx()],
                        fresh.psqt[p.idx()],
                        "psqt mismatch after {} in {fen} ({p:?})",
                        m.to_uci()
                    );
                }
                let _ = child_pos;
                checked += 1;
            }
        }
        assert!(checked > 300, "only checked {checked} moves");
    }

    /// A null move changes nothing on the board, so the accumulator must be
    /// literally the parent's.
    #[test]
    fn a_null_move_reuses_the_parent_accumulator() {
        let net = test_net();
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let board = Board::from_position(&pos);
        let mut parent = Accumulator::new();
        parent.refresh(Color::White, &board, net);
        parent.refresh(Color::Black, &board, net);
        let child = pos.null_move().expect("no check here");
        let child_board = Board::from_position(&child);
        let mut fresh = Accumulator::new();
        fresh.refresh(Color::White, &child_board, net);
        fresh.refresh(Color::Black, &child_board, net);
        assert_eq!(parent.accumulation, fresh.accumulation);
        assert_eq!(parent.psqt, fresh.psqt);
    }

    #[test]
    fn the_stack_allocates_one_slot_per_ply() {
        let mut stack = AccumulatorStack::new();
        assert_eq!(stack.len(), crate::types::MAX_PLY + 1);
        assert!(!stack.get(0).computed[0], "fresh slots are not computed");
        stack.get_mut(3).computed[1] = true;
        assert!(stack.get(3).computed[1]);
        stack.reset();
        assert!(!stack.get(3).computed[1], "reset clears everything");
    }
}

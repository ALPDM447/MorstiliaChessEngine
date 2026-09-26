//! A Stockfish-shaped board view plus the threat bookkeeping the incremental
//! accumulator needs.
//!
//! The search in this engine has no make/unmake: every node gets a fresh
//! `Position` clone. So instead of maintaining one long-lived NNUE board across
//! the tree, [`Board::from_position`] rebuilds a compact board from the
//! position (~230 bytes, one pass over the occupied squares) and
//! [`Board::apply_move`] produces the child's board *and* the dirty records in
//! the same pass. That costs a little more per node than Stockfish's
//! make/unmake, but it removes an entire class of "the cached board drifted out
//! of sync with the position" bugs and keeps the whole NNUE layer a pure
//! function of `(Position, move)`.
//!
//! Every board mutation emits the threats it creates or destroys into a
//! [`DirtyThreats`] list, following `Position::update_piece_threats` from
//! Stockfish exactly — same call order, same `noRaysContaining` filtering, same
//! king/queen special cases.

use crate::board::Position;
use crate::nnue::attacks::{
    both_attacks_bb, knight_attacks, pawn_attacks, pseudo_attacks, ray_pass_bb, square_bb,
};
use crate::nnue::types::{
    BISHOP, Bitboard, Color, DirtyPawnPairs, DirtyPiece, DirtyThreat, DirtyThreats, KING, KNIGHT,
    NO_PIECE, PAWN, PIECE_NB, QUEEN, ROOK, SQ_NONE, color_of, make_piece, type_of,
};
use crate::types::RawMove;

/// Stockfish's `can_slider_threat`: a queen is only a "real" slider threat for
/// a non-queen target, because queen-vs-queon is handled by the same-piece
/// exclusion in the feature set.
#[inline]
const fn can_slider_threat(pc: u8, slider: u8) -> bool {
    type_of(pc) != QUEEN || type_of(slider) == QUEEN
}

/// A compact board, structured exactly like Stockfish's `Position` internals.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Board {
    /// Piece code per square, `NO_PIECE` when empty.
    pub squares: [u8; 64],
    /// `by_type[0]` is the occupancy, `by_type[1..=6]` the per-type sets.
    by_type: [Bitboard; 7],
    by_color: [Bitboard; 2],
    /// Per-piece-code counts; `piece_count[0]` and `piece_count[8]` hold the
    /// per-color totals, mirroring `make_piece(c, ALL_PIECES)`.
    piece_count: [u32; PIECE_NB],
}

/// Everything one move changes, ready to be replayed into an accumulator.
#[derive(Clone, Copy)]
pub struct MoveDirties {
    /// The board after the move.
    pub board: Board,
    pub dirty_piece: DirtyPiece,
    pub dirty_threats: DirtyThreats,
    pub dirty_pawn_pairs: DirtyPawnPairs,
}

impl Board {
    /// Builds the NNUE board view of `pos`.
    pub fn from_position(pos: &Position) -> Board {
        let mut b = Board {
            squares: [NO_PIECE; 64],
            by_type: [0; 7],
            by_color: [0; 2],
            piece_count: [0; PIECE_NB],
        };
        let board = pos.board();
        let mut occ: u64 = board.occupied().0;
        while occ != 0 {
            let s = occ.trailing_zeros() as usize;
            occ &= occ - 1;
            let piece = board
                .piece_at(shakmaty::Square::new(s as u32))
                .expect("occupied square holds a piece");
            // Stockfish numbers White as 0, so the colour bit is set for *black*.
            let pc = ((piece.color == shakmaty::Color::Black) as u8) << 3 | u8::from(piece.role);
            b.put_piece_quiet(pc, s);
        }
        b
    }

    #[inline]
    pub fn piece_on(&self, s: usize) -> u8 {
        self.squares[s]
    }

    #[inline]
    pub fn pieces(&self) -> Bitboard {
        self.by_type[0]
    }

    /// Occupancy of a piece type.
    #[inline]
    pub fn pieces_of(&self, pt: u8) -> Bitboard {
        self.by_type[pt as usize]
    }

    /// Union of two piece types.
    #[inline]
    fn pieces_of_two(&self, a: u8, b: u8) -> Bitboard {
        self.by_type[a as usize] | self.by_type[b as usize]
    }

    /// Union of the given piece types.
    #[inline]
    pub fn pieces_of_many(&self, a: u8, b: u8) -> Bitboard {
        self.by_type[a as usize] | self.by_type[b as usize]
    }

    /// Union of three piece types.
    #[inline]
    pub fn pieces_of_three(&self, a: u8, b: u8, c: u8) -> Bitboard {
        self.by_type[a as usize] | self.by_type[b as usize] | self.by_type[c as usize]
    }

    /// Union of four piece types.
    #[inline]
    pub fn pieces_of_four(&self, a: u8, b: u8, c: u8, d: u8) -> Bitboard {
        self.by_type[a as usize]
            | self.by_type[b as usize]
            | self.by_type[c as usize]
            | self.by_type[d as usize]
    }

    /// Union of five piece types.
    #[inline]
    pub fn pieces_of_five(&self, a: u8, b: u8, c: u8, d: u8, e: u8) -> Bitboard {
        self.by_type[a as usize]
            | self.by_type[b as usize]
            | self.by_type[c as usize]
            | self.by_type[d as usize]
            | self.by_type[e as usize]
    }

    #[inline]
    pub fn pieces_of_color(&self, c: Color) -> Bitboard {
        self.by_color[c.idx()]
    }

    /// Pawns of one colour.
    #[inline]
    pub fn pawns(&self, c: Color) -> Bitboard {
        self.by_type[PAWN as usize] & self.by_color[c.idx()]
    }

    /// Every pawn on the board, both colours — the pawn term of
    /// `EvalMeta::pawns`, which Stockfish's bare `pos.count<PAWN>()` also
    /// covers.
    #[inline]
    pub fn all_pawns(&self) -> Bitboard {
        self.by_type[PAWN as usize]
    }

    #[inline]
    pub fn king_square(&self, c: Color) -> usize {
        (self.pieces_of(KING) & self.pieces_of_color(c)).trailing_zeros() as usize
    }

    #[inline]
    pub fn piece_count_of(&self, pc: u8) -> u32 {
        self.piece_count[pc as usize]
    }

    /// Total number of pieces on the board (the evaluation bucket input).
    #[inline]
    pub fn piece_count(&self) -> u32 {
        self.piece_count[0] + self.piece_count[8]
    }

    /// Non-pawn material of both colours, in Stockfish's centipawn-ish units.
    /// Kings are worth 0, pawns are excluded (the caller adds them separately).
    pub fn non_pawn_material(&self) -> i32 {
        let mut total = 0i32;
        for c in Color::ALL {
            for pt in [KNIGHT, BISHOP, ROOK, QUEEN] {
                total += crate::nnue::types::PIECE_VALUE[pt as usize]
                    * self.piece_count[make_piece(c.idx(), pt) as usize] as i32;
            }
        }
        total
    }

    // --- mutators ------------------------------------------------------------

    /// Places a piece without emitting any threats (position setup only).
    #[inline]
    fn put_piece_quiet(&mut self, pc: u8, s: usize) {
        self.squares[s] = pc;
        self.by_type[0] |= square_bb(s);
        self.by_type[type_of(pc) as usize] |= square_bb(s);
        self.by_color[color_of(pc)] |= square_bb(s);
        self.piece_count[pc as usize] += 1;
        self.piece_count[make_piece(color_of(pc), 0) as usize] += 1;
    }

    /// `Position::put_piece`.
    #[inline]
    fn put_piece(&mut self, pc: u8, s: usize, dts: &mut DirtyThreats) {
        // The threats are computed against the *pre-move* occupancy, exactly as
        // in Stockfish.
        self.update_piece_threats(pc, true, s, dts, true, !0);
        self.put_piece_quiet(pc, s);
    }

    /// `Position::remove_piece`.
    #[inline]
    fn remove_piece(&mut self, s: usize, dts: &mut DirtyThreats) {
        let pc = self.squares[s];
        self.update_piece_threats(pc, false, s, dts, true, !0);
        self.by_type[0] ^= square_bb(s);
        self.by_type[type_of(pc) as usize] ^= square_bb(s);
        self.by_color[color_of(pc)] ^= square_bb(s);
        self.squares[s] = NO_PIECE;
        self.piece_count[pc as usize] -= 1;
        self.piece_count[make_piece(color_of(pc), 0) as usize] -= 1;
    }

    /// `Position::move_piece`.
    #[inline]
    fn move_piece(&mut self, from: usize, to: usize, dts: &mut DirtyThreats) {
        let pc = self.squares[from];
        let from_to = square_bb(from) | square_bb(to);
        self.update_piece_threats(pc, false, from, dts, true, from_to);
        self.by_type[0] ^= from_to;
        self.by_type[type_of(pc) as usize] ^= from_to;
        self.by_color[color_of(pc)] ^= from_to;
        self.squares[from] = NO_PIECE;
        self.squares[to] = pc;
        self.update_piece_threats(pc, true, to, dts, true, from_to);
    }

    /// `Position::swap_piece`. The two `update_piece_threats` calls here run with
    /// `ComputeRay = false`, which is what makes a capture's discovered
    /// threats come out right: the occupancy never really changes at `s` from
    /// the sliding pieces' point of view except for the occupant.
    #[inline]
    fn swap_piece(&mut self, s: usize, pc: u8, dts: &mut DirtyThreats) {
        let old = self.squares[s];
        // `remove_piece` without the threat emission, then emit the old piece's
        // departure with rays suppressed.
        self.by_type[0] ^= square_bb(s);
        self.by_type[type_of(old) as usize] ^= square_bb(s);
        self.by_color[color_of(old)] ^= square_bb(s);
        self.squares[s] = NO_PIECE;
        self.piece_count[old as usize] -= 1;
        self.piece_count[make_piece(color_of(old), 0) as usize] -= 1;

        self.update_piece_threats(old, false, s, dts, false, !0);
        self.put_piece_quiet(pc, s);
        self.update_piece_threats(pc, true, s, dts, false, !0);
    }

    // --- threat emission -----------------------------------------------------

    /// `Position::update_piece_threats`.
    ///
    /// `compute_ray == false` is Stockfish's `ComputeRay = false` template
    /// argument: the discovered-threat pass is skipped and queen sliding
    /// attackers are folded into the incoming list instead.
    #[allow(clippy::too_many_arguments)]
    fn update_piece_threats(
        &self,
        pc: u8,
        put_piece: bool,
        s: usize,
        dts: &mut DirtyThreats,
        compute_ray: bool,
        no_rays_containing: Bitboard,
    ) {
        let occupied = self.pieces();
        let (b_attacks, r_attacks) = both_attacks_bb(s, occupied);
        let slider_attacks = b_attacks | r_attacks;
        let occupied_no_k = occupied ^ self.pieces_of(KING);
        let pt = type_of(pc);
        let sliders = (self.pieces_of_two(BISHOP, QUEEN) & b_attacks)
            | (self.pieces_of_two(ROOK, QUEEN) & r_attacks);

        // Discovered attacks exposed (or blocked) at `s`, plus the slider's own
        // new/old direct threat against `pc`.
        let process_sliders = |dts: &mut DirtyThreats, add_direct_attacks: bool| {
            let mut b = sliders;
            while b != 0 {
                let slider_sq = b.trailing_zeros() as usize;
                b &= b - 1;
                let slider = self.squares[slider_sq];

                let ray = ray_pass_bb(slider_sq, s);
                let discovered = ray & slider_attacks & occupied_no_k;
                debug_assert!(
                    discovered.count_ones() <= 1,
                    "a slider can only expose one discovered target through s"
                );
                if discovered != 0 && (ray & no_rays_containing) != no_rays_containing {
                    let threatened_sq = discovered.trailing_zeros() as usize;
                    let threatened_pc = self.squares[threatened_sq];
                    if can_slider_threat(threatened_pc, slider) {
                        dts.push(DirtyThreat::new(
                            slider,
                            threatened_pc,
                            slider_sq,
                            threatened_sq,
                            !put_piece,
                        ));
                    }
                }

                if add_direct_attacks && can_slider_threat(pc, slider) {
                    dts.push(DirtyThreat::new(slider, pc, slider_sq, s, put_piece));
                }
            }
        };

        // Kings emit no direct threats; only discovered ones.
        if pt == KING {
            if compute_ray {
                process_sliders(dts, false);
            }
            return;
        }

        let threat_targets = if pt == PAWN {
            self.pieces_of_two(KNIGHT, ROOK)
        } else if pt == BISHOP || pt == ROOK {
            self.pieces_of_four(PAWN, KNIGHT, BISHOP, ROOK)
        } else {
            occupied_no_k
        };

        let own_attacks = match pt {
            BISHOP => b_attacks,
            ROOK => r_attacks,
            QUEEN => slider_attacks,
            PAWN => pseudo_attacks_by_color(color_of(pc), s),
            _ => pseudo_attacks(pt, s),
        };

        let mut threatened = own_attacks & threat_targets;
        let mut incoming_threats = knight_attacks(s) & self.pieces_of(KNIGHT);

        if pt == KNIGHT || pt == ROOK {
            incoming_threats |= (pawn_attacks(Color::White, s) & self.pawns(Color::Black))
                | (pawn_attacks(Color::Black, s) & self.pawns(Color::White));
        }

        while threatened != 0 {
            let to = threatened.trailing_zeros() as usize;
            threatened &= threatened - 1;
            let attacked = self.squares[to];
            dts.push(DirtyThreat::new(pc, attacked, s, to, put_piece));
        }

        if compute_ray {
            process_sliders(dts, true);
        } else if pt == QUEEN {
            incoming_threats |= sliders & self.pieces_of(QUEEN);
        } else {
            incoming_threats |= sliders;
        }

        while incoming_threats != 0 {
            let from = incoming_threats.trailing_zeros() as usize;
            incoming_threats &= incoming_threats - 1;
            let src = self.squares[from];
            dts.push(DirtyThreat::new(src, pc, from, s, put_piece));
        }
    }

    // --- moves ---------------------------------------------------------------

    /// Applies `m` (assumed legal) for the side to move `us`, returning the
    /// child board together with every dirty record the accumulator needs.
    ///
    /// The emission order and every `DirtyPiece` field follow
    /// `Position::do_move`, including the promotion rule (`to = SQ_NONE`, the
    /// added index coming from `add_sq`) and the castling rule (both pieces
    /// removed before either is placed, so Chess960 overlaps are handled).
    pub fn apply_move(&self, m: RawMove, us: Color) -> MoveDirties {
        let them = us.other();
        let from = m.from().to_usize();
        let to_raw = m.to().to_usize();
        let pc = self.squares[from];
        let kind = m.kind();

        let mut board = *self;
        let mut dts = DirtyThreats::new();
        let mut dp = DirtyPiece::new(pc, from, to_raw);

        if kind == RawMove::CASTLE {
            // "King captures own rook": `to` holds the rook square.
            let king_side = to_raw > from;
            let rfrom = to_raw;
            let (kto, rto) = if king_side {
                (6usize, 5usize)
            } else {
                (2usize, 3usize)
            };
            let (kto, rto) = if us == Color::White {
                (kto, rto)
            } else {
                (kto ^ 56, rto ^ 56)
            };
            let rook = make_piece(us.idx(), ROOK);
            dp.to = kto;
            dp.remove_pc = rook;
            dp.add_pc = rook;
            dp.remove_sq = rfrom;
            dp.add_sq = rto;
            // Both removals first: the squares can overlap in Chess960.
            board.remove_piece(from, &mut dts);
            board.remove_piece(rfrom, &mut dts);
            board.put_piece(make_piece(us.idx(), KING), kto, &mut dts);
            board.put_piece(rook, rto, &mut dts);
        } else {
            let captured = if kind == RawMove::EN_PASSANT {
                make_piece(them.idx(), PAWN)
            } else {
                self.squares[to_raw]
            };

            if captured != NO_PIECE {
                // En passant removes the victim before anything else, so the
                // later `move_piece` sees an empty destination.
                let capsq = if kind == RawMove::EN_PASSANT {
                    if us == Color::White {
                        to_raw - 8
                    } else {
                        to_raw + 8
                    }
                } else {
                    to_raw
                };
                dp.remove_pc = captured;
                dp.remove_sq = capsq;
                if kind == RawMove::EN_PASSANT {
                    board.remove_piece(capsq, &mut dts);
                }
            }

            let to_pc = match m.promotion() {
                Some(role) => {
                    dp.add_pc = make_piece(us.idx(), u8::from(role));
                    dp.add_sq = to_raw;
                    dp.to = SQ_NONE;
                    dp.add_pc
                }
                None => pc,
            };

            if captured != NO_PIECE && kind != RawMove::EN_PASSANT {
                board.remove_piece(from, &mut dts);
                board.swap_piece(to_raw, to_pc, &mut dts);
            } else if pc == to_pc {
                board.move_piece(from, to_raw, &mut dts);
            } else {
                board.remove_piece(from, &mut dts);
                board.put_piece(to_pc, to_raw, &mut dts);
            }
        }

        MoveDirties {
            board,
            dirty_piece: dp,
            dirty_threats: dts,
            dirty_pawn_pairs: DirtyPawnPairs {
                before: [self.pawns(Color::White), self.pawns(Color::Black)],
                after: [board.pawns(Color::White), board.pawns(Color::Black)],
            },
        }
    }
}

#[inline]
fn pseudo_attacks_by_color(color: usize, s: usize) -> Bitboard {
    pawn_attacks(Color::from_index(color), s)
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let s = rank * 8 + file;
                let ch = match self.squares[s] {
                    NO_PIECE => '.',
                    pc => match (color_of(pc), type_of(pc)) {
                        (0, PAWN) => 'P',
                        (0, KNIGHT) => 'N',
                        (0, BISHOP) => 'B',
                        (0, ROOK) => 'R',
                        (0, QUEEN) => 'Q',
                        (0, KING) => 'K',
                        (1, PAWN) => 'p',
                        (1, KNIGHT) => 'n',
                        (1, BISHOP) => 'b',
                        (1, ROOK) => 'r',
                        (1, QUEEN) => 'q',
                        _ => 'k',
                    },
                };
                f.write_str(&ch.to_string())?;
            }
            f.write_str(if rank == 0 { "\n" } else { " / " })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // FEN square helpers (Stockfish numbering), spelled out for readability.
    const A1: usize = 0;
    const C1: usize = 2;
    const D1: usize = 3;
    const D5: usize = 35;
    const E1: usize = 4;
    const F1: usize = 5;
    const G1: usize = 6;
    const H1: usize = 7;
    const A8: usize = 56;
    const C8: usize = 58;
    const D8: usize = 59;
    const H8: usize = 63;
    const F8: usize = 61;
    const G8: usize = 62;

    /// Walks the whole legal move tree, applying each move through
    /// `apply_move` and checking the resulting board against a freshly built
    /// one from the shakmaty position. This is the invariant the whole NNUE
    /// layer rests on.
    #[test]
    fn apply_move_reproduces_the_child_board() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        ];
        for fen in fens {
            let pos = crate::board::Position::from_fen(fen).unwrap();
            verify(pos, 3);
        }
    }

    fn verify(pos: crate::board::Position, depth: u32) {
        let board = Board::from_position(&pos);
        // Spot-check the rebuild against shakmaty itself.
        for s in 0..64 {
            let sq = shakmaty::Square::new(s as u32);
            match pos.board().piece_at(sq) {
                Some(p) => {
                    // Stockfish numbers White as 0, so the colour bit is set for
                    // Black: `make_piece(WHITE, role) == role`.
                    let expected =
                        ((p.color == shakmaty::Color::Black) as u8) << 3 | u8::from(p.role);
                    assert_eq!(board.piece_on(s), expected, "{s} in {}", pos.fen());
                }
                None => assert_eq!(board.piece_on(s), NO_PIECE),
            }
        }
        let us = Color::from_shakmaty(pos.turn());
        if depth == 0 {
            return;
        }
        for m in pos.legal_moves().iter() {
            let child = pos.make_child(m);
            let d = board.apply_move(m, us);
            assert_eq!(d.board, Board::from_position(&child), "move {}", m.to_uci());
            assert_eq!(d.dirty_piece.pc, board.piece_on(m.from().to_usize()));
            // The pawn-pair record brackets the move: it only changes when a
            // pawn leaves or arrives — as the mover or as the victim.
            let moved = d.dirty_piece.pc;
            let removed = d.dirty_piece.remove_pc;
            let touches_pawn = type_of(moved) == crate::nnue::types::PAWN
                || (removed != NO_PIECE && type_of(removed) == crate::nnue::types::PAWN)
                || (d.dirty_piece.add_pc != NO_PIECE
                    && type_of(d.dirty_piece.add_pc) == crate::nnue::types::PAWN);
            assert_eq!(
                d.dirty_pawn_pairs.before == d.dirty_pawn_pairs.after,
                !touches_pawn,
                "move {} changed the pawn set unexpectedly",
                m.to_uci()
            );
            verify(child, depth - 1);
        }
    }

    #[test]
    fn capture_uses_swap_and_promotion_uses_add_sq() {
        let pos = crate::board::Position::from_fen(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        )
        .unwrap();
        let board = Board::from_position(&pos);
        let us = Color::White;

        // e5xd6 e.p.
        let ep = pos.raw_move_from_uci("e5d6").unwrap();
        let d = board.apply_move(ep, us);
        assert_eq!(d.dirty_piece.pc, crate::nnue::types::W_PAWN);
        assert_eq!(d.dirty_piece.remove_sq, D5, "the captured pawn leaves d5");
        assert_eq!(d.dirty_piece.add_sq, crate::nnue::types::SQ_NONE);

        // a7a8=N under-promotion
        let pos2 = crate::board::Position::from_fen("8/P7/8/8/8/8/8/4k1K1 w - - 0 1").unwrap();
        let b2 = Board::from_position(&pos2);
        let promo = pos2.raw_move_from_uci("a7a8n").unwrap();
        let d2 = b2.apply_move(promo, Color::White);
        assert_eq!(
            d2.dirty_piece.to,
            crate::nnue::types::SQ_NONE,
            "to is cleared"
        );
        assert_eq!(d2.dirty_piece.add_sq, A8);
        assert_eq!(
            d2.dirty_piece.add_pc,
            crate::nnue::types::W_KNIGHT,
            "knight, not queen"
        );
    }

    #[test]
    fn castling_reports_the_king_destination() {
        let pos = crate::board::Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let board = Board::from_position(&pos);

        let oo = pos.raw_move_from_uci("e1g1").unwrap();
        let d = board.apply_move(oo, Color::White);
        assert_eq!(d.dirty_piece.pc, crate::nnue::types::W_KING);
        assert_eq!(d.dirty_piece.to, G1, "king destination, not the rook");
        assert_eq!(d.dirty_piece.remove_sq, H1);
        assert_eq!(d.dirty_piece.add_sq, F1);
        assert_eq!(d.dirty_piece.remove_pc, crate::nnue::types::W_ROOK);
        assert_eq!(d.dirty_piece.add_pc, crate::nnue::types::W_ROOK);
        assert_eq!(d.board.piece_on(G1), crate::nnue::types::W_KING);
        assert_eq!(d.board.piece_on(F1), crate::nnue::types::W_ROOK);
        assert_eq!(d.board.piece_on(H1), NO_PIECE);
        assert_eq!(d.board.piece_on(E1), NO_PIECE);

        let oo_o = pos.raw_move_from_uci("e1c1").unwrap();
        let d2 = board.apply_move(oo_o, Color::White);
        assert_eq!(d2.dirty_piece.to, C1);
        assert_eq!(d2.dirty_piece.remove_sq, A1);
        assert_eq!(d2.dirty_piece.add_sq, D1);

        // Black kingside: e8g8 -> king g8, rook f8. Castling rights are the
        // mover's own, so this needs a position with Black to move.
        let bpos =
            crate::board::Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1").unwrap();
        let bboard = Board::from_position(&bpos);
        let oo_b = bpos.raw_move_from_uci("e8g8").unwrap();
        let d3 = bboard.apply_move(oo_b, Color::Black);
        assert_eq!(d3.dirty_piece.pc, crate::nnue::types::B_KING);
        assert_eq!(d3.dirty_piece.to, G8);
        assert_eq!(d3.dirty_piece.add_sq, F8);
        assert_eq!(d3.dirty_piece.remove_sq, H8);
        assert_eq!(d3.board.piece_on(G8), crate::nnue::types::B_KING);
        assert_eq!(d3.board.piece_on(F8), crate::nnue::types::B_ROOK);
        assert_eq!(d3.board.piece_on(H8), NO_PIECE);

        // Black queenside: e8c8 -> king c8, rook d8.
        let oo_b_o = bpos.raw_move_from_uci("e8c8").unwrap();
        let d4 = bboard.apply_move(oo_b_o, Color::Black);
        assert_eq!(d4.dirty_piece.to, C8);
        assert_eq!(d4.dirty_piece.add_sq, D8);
        assert_eq!(d4.dirty_piece.remove_sq, A8);
    }

    #[test]
    fn dirty_threat_list_stays_within_the_documented_capacity() {
        // The densest legal positions we can cheaply build, then every legal
        // move in them. A normal move touches at most 80 features and castling
        // at most 36, so the 96-slot list must never overflow.
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "QQQQQQQQ/QQQQQQQQ/8/8/8/8/qqqqqqqq/qqqqqqqq/K6k w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        let mut worst = 0usize;
        for fen in fens {
            let Ok(pos) = crate::board::Position::from_fen(fen) else {
                continue;
            };
            let board = Board::from_position(&pos);
            let us = Color::from_shakmaty(pos.turn());
            for m in pos.legal_moves().iter() {
                let d = board.apply_move(m, us);
                worst = worst.max(d.dirty_threats.list.size());
                assert!(
                    d.dirty_threats.list.size() <= 96,
                    "{} produced {} threats",
                    m.to_uci(),
                    d.dirty_threats.list.size()
                );
            }
        }
        assert!(worst > 0);
    }

    #[test]
    fn non_pawn_material_matches_a_hand_count() {
        let pos = crate::board::Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        .unwrap();
        let board = Board::from_position(&pos);
        // 2N + 2B + 2R + 1Q per side, both sides counted.
        let expect = 2 * (2 * 781 + 2 * 825 + 2 * 1276 + 2538);
        assert_eq!(expect, 16_604);
        assert_eq!(board.non_pawn_material(), expect);
        assert_eq!(board.piece_count(), 32);

        let pos = crate::board::Position::from_fen("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1").unwrap();
        let board = Board::from_position(&pos);
        assert_eq!(board.non_pawn_material(), 0, "pawns and kings are worth 0");
        assert_eq!(board.piece_count(), 3);
    }
}

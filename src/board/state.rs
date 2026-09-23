//! [`Position`]: a `shakmaty::Chess` plus an incrementally maintained
//! Zobrist hash.
//!
//! `shakmaty` has no undo (`unmake`), so the search never mutates a position
//! in place: [`Position::make_child`] clones the current position, plays a
//! move on the clone and returns it. Clones are ~80 bytes, i.e. a cheap
//! `memcpy` on the stack, and the parent stays valid for sibling moves.

use anyhow::{Context, Result};
use shakmaty::fen::Fen;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingMode, CastlingSide, Chess, Color, EnPassantMode};
use shakmaty::{FromSetup as _, Position as _};

use crate::book::polyglot_key;
use crate::types::{MoveList, RawMove};

/// A position ready for search.
#[derive(Clone)]
pub struct Position {
    pub chess: Chess,
    /// Zobrist hash of `chess`, maintained incrementally when possible.
    /// Halfmove/fullmove counters are deliberately excluded (matches
    /// `zobrist_hash(EnPassantMode::Legal)`), so it is usable as the TT key
    /// and for repetition detection.
    pub hash: Zobrist64,
}

impl Position {
    /// The standard chess starting position.
    pub fn startpos() -> Position {
        Position {
            chess: Chess::default(),
            hash: Chess::default().zobrist_hash::<Zobrist64>(EnPassantMode::Legal),
        }
    }

    /// Parses a FEN string (`EnPassantMode::Legal` for hashing).
    pub fn from_fen(fen: &str) -> Result<Position> {
        let input = fen.to_string();
        let fen: Fen = input
            .parse()
            .with_context(|| format!("invalid FEN: {input}"))?;
        let chess: Chess = fen
            .into_position(CastlingMode::Standard)
            .with_context(|| format!("illegal FEN position: {input}"))?;
        let hash = chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        Ok(Position { chess, hash })
    }

    /// Serializes back to FEN.
    pub fn fen(&self) -> String {
        Fen::from_position(&self.chess, EnPassantMode::Legal).to_string()
    }

    /// Plays a compact `RawMove` (assumed legal) on a clone of this position
    /// and returns the resulting position — a stack-friendly `memcpy` +
    /// `play_unchecked`. The parent is unchanged.
    #[inline]
    pub fn make_child(&self, m: RawMove) -> Position {
        let smove = m.to_shakmaty(self.chess.board());
        let mut chess = self.chess.clone();
        chess.play_unchecked(smove);
        let hash = self
            .chess
            .update_zobrist_hash::<Zobrist64>(self.hash, smove, EnPassantMode::Legal)
            .unwrap_or_else(|| chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal));
        debug_assert_eq!(
            hash,
            chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal),
            "incremental hash mismatch after {:?}",
            smove
        );
        Position { chess, hash }
    }

    /// Plays a UCI move string (`e2e4`, `e7e8q`, `e1g1`, …). The move must be
    /// legal; errors on invalid/illegal moves while leaving `self` untouched.
    pub fn play_uci(&self, uci: &str) -> Result<(Position, RawMove)> {
        let umove: shakmaty::uci::UciMove = uci
            .parse()
            .with_context(|| format!("invalid UCI move: {uci}"))?;
        let m = umove
            .to_move(&self.chess)
            .with_context(|| format!("illegal UCI move: {uci}"))?;
        let raw = RawMove::from_shakmaty(m);
        let child = Position {
            chess: {
                let mut c = self.chess.clone();
                c.play_unchecked(m);
                c
            },
            hash: self
                .chess
                .update_zobrist_hash::<Zobrist64>(self.hash, m, EnPassantMode::Legal)
                .unwrap_or_else(|| {
                    let mut c = self.chess.clone();
                    c.play_unchecked(m);
                    c.zobrist_hash::<Zobrist64>(EnPassantMode::Legal)
                }),
        };
        debug_assert_eq!(
            child.hash,
            child.chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal)
        );
        Ok((child, raw))
    }

    /// The null move: swaps the side to move and clears en passant, if the
    /// side to move is not in check. Returns `None` when the null move would
    /// be illegal (king in check).
    pub fn null_move(&self) -> Option<Position> {
        if self.chess.is_check() {
            return None;
        }
        let mode = self.chess.castles().mode();
        let mut setup = self.chess.to_setup(EnPassantMode::Always);
        setup.swap_turn();
        let chess = Chess::from_setup(setup, mode).ok()?;
        let hash = chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        Some(Position { chess, hash })
    }

    /// All legal moves as compact `RawMove`s.
    #[inline]
    pub fn legal_moves(&self) -> MoveList {
        let mut out = MoveList::new();
        for m in self.chess.legal_moves() {
            out.push(RawMove::from_shakmaty(m));
        }
        out
    }

    /// A reordered list of all legal moves (see `move_ordering` for scoring).
    /// `tables` carry the search's ordering state (TT move comes from the
    /// caller separately).
    #[inline]
    pub fn legal_moves_ordered(&self, tables: &crate::move_ordering::OrderingTables) -> MoveList {
        let mut moves = self.legal_moves();
        crate::move_ordering::order_moves(&mut moves, self, tables, RawMove::NULL);
        moves
    }

    /// Moves that are captures or promotions (for quiescence search).
    #[inline]
    pub fn capture_moves(&self) -> MoveList {
        let board = self.chess.board();
        let mut out = MoveList::new();
        for m in self.chess.legal_moves() {
            let raw = RawMove::from_shakmaty(m);
            if raw.is_promotion() || raw.is_en_passant() || board.role_at(raw.to()).is_some() {
                out.push(raw);
            }
        }
        out
    }

    /// Standard perft. `depth == 0` counts 1 (the root itself).
    pub fn perft(&self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = self.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut nodes = 0;
        for m in moves.iter() {
            nodes += self.make_child(m).perft(depth - 1);
        }
        nodes
    }

    /// Hierarchical perft split by the first-move group (useful for
    /// validating and for test output).
    pub fn perft_split(&self, depth: u32) -> Vec<(RawMove, u64)> {
        let mut out = Vec::new();
        for m in self.legal_moves().iter() {
            out.push((m, self.make_child(m).perft(depth.saturating_sub(1))));
        }
        out
    }

    /// Polyglot opening-book key for this position.
    pub fn polyglot_key(&self) -> u64 {
        polyglot_key(&self.chess)
    }

    #[inline]
    pub fn chess(&self) -> &Chess {
        &self.chess
    }

    #[inline]
    pub fn board(&self) -> &shakmaty::Board {
        self.chess.board()
    }

    #[inline]
    pub fn turn(&self) -> Color {
        self.chess.turn()
    }

    #[inline]
    pub fn is_check(&self) -> bool {
        self.chess.is_check()
    }

    #[inline]
    pub fn halfmoves(&self) -> u32 {
        self.chess.halfmoves()
    }

    /// True when the 50-move rule already draws (no capture/pawn move for 100
    /// half-moves), or the position is a material draw.
    pub fn is_drawish(&self) -> bool {
        self.chess.halfmoves() >= 100 || self.chess.is_insufficient_material()
    }

    /// Convenience: castling rights of the given color and side.
    #[inline]
    pub fn has_castling_right(&self, color: Color, side: CastlingSide) -> bool {
        self.chess.castles().has(color, side)
    }

    /// Parse a move given as UCI text in the context of `self` and return the
    /// `RawMove` if legal.
    pub fn raw_move_from_uci(&self, uci: &str) -> Option<RawMove> {
        let umove: shakmaty::uci::UciMove = uci.parse().ok()?;
        umove.to_move(&self.chess).ok().map(RawMove::from_shakmaty)
    }

    /// Full legality check for a compact move (used for TT/book moves).
    /// This generates the legal move list internally, so keep it off the hot
    /// path — only TT moves already present in the generated list are played
    /// without it.
    pub fn raw_move_legal(&self, m: RawMove) -> bool {
        self.legal_moves().contains(m)
    }

    /// Shorthand validity check used by move-ordering callers.
    #[inline]
    pub fn is_mated(&self) -> bool {
        self.chess.is_checkmate()
    }
}

/// Sentinel move that never equals a real move; used as "no book move", etc.
pub const NULL_MOVE: RawMove = crate::types::RawMove::NULL;

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::zobrist::Zobrist64;

    #[test]
    fn incremental_hash_matches_full_during_random_play() {
        let mut pos = Position::startpos();
        let mut expected = pos.chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        assert_eq!(pos.hash, expected);
        // A deterministic pseudo-random walk (SplitMix-ish inline).
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for _ in 0..200 {
            if pos.is_drawish() {
                break;
            }
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            // Repeat positions would shrink legal moves; avoid infinite loop.
            let pick = (rnd() % moves.len() as u64) as usize;
            let m = moves.get(pick);
            let child = pos.make_child(m);
            expected = child.chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
            assert_eq!(child.hash, expected, "at fen {}", child.fen());
            pos = child;
        }
    }

    #[test]
    fn perft_startpos_known_values() {
        let pos = Position::startpos();
        assert_eq!(pos.perft(1), 20);
        assert_eq!(pos.perft(2), 400);
        assert_eq!(pos.perft(3), 8_902);
        assert_eq!(pos.perft(4), 197_281);
        // depth 5 is 4.8M nodes; kept in the integration suite instead.
    }

    #[test]
    fn perft_kiwipete_known_values() {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(pos.perft(1), 48);
        assert_eq!(pos.perft(2), 2_039);
        assert_eq!(pos.perft(3), 97_862);
    }

    #[test]
    fn perft_position3() {
        let pos = Position::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert_eq!(pos.perft(1), 14);
        assert_eq!(pos.perft(2), 191);
        assert_eq!(pos.perft(3), 2_812);
        assert_eq!(pos.perft(4), 43_238);
        assert_eq!(pos.perft(5), 674_624);
    }

    #[test]
    fn perft_position4() {
        let pos =
            Position::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
                .unwrap();
        assert_eq!(pos.perft(1), 6);
        assert_eq!(pos.perft(2), 264);
        assert_eq!(pos.perft(3), 9_467);
        assert_eq!(pos.perft(4), 422_333);
    }

    #[test]
    fn perft_position5() {
        let pos = Position::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")
            .unwrap();
        assert_eq!(pos.perft(1), 44);
        assert_eq!(pos.perft(2), 1_486);
        assert_eq!(pos.perft(3), 62_379);
    }

    #[test]
    fn perft_position6() {
        let pos = Position::from_fen(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        )
        .unwrap();
        assert_eq!(pos.perft(1), 46);
        assert_eq!(pos.perft(2), 2_079);
        assert_eq!(pos.perft(3), 89_890);
    }

    #[test]
    fn null_move_illegal_in_check() {
        // White to move, white king is in check by the rook on e2.
        let pos = Position::from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1").unwrap();
        assert!(pos.is_check());
        assert!(pos.null_move().is_none());
    }

    #[test]
    fn null_move_swaps_turn_and_clears_ep() {
        let pos =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
                .unwrap();
        let nulled = pos.null_move().expect("null move ok");
        assert_eq!(nulled.turn(), Color::Black);
        assert!(!nulled.is_check());
        // ep square is cleared.
        assert!(nulled.chess.maybe_ep_square().is_none());
    }

    #[test]
    fn play_uci_round_trip() {
        let pos = Position::startpos();
        let (child, raw) = pos.play_uci("e2e4").unwrap();
        assert_eq!(raw.to_uci(), "e2e4");
        assert_eq!(child.turn(), Color::Black);
        assert!(pos.play_uci("e7e5").is_err(), "illegal for white to move");
    }

    #[test]
    fn polyglot_key_startpos_matches_python_chess_reference() {
        // Reference values produced by python-chess (official random array).
        let pos = Position::startpos();
        assert_eq!(pos.polyglot_key(), 0x463b96181691fc9c);
    }

    #[test]
    fn polyglot_key_after_e4() {
        let pos = Position::startpos();
        let (child, _) = pos.play_uci("e2e4").unwrap();
        assert_eq!(child.polyglot_key(), 0x823c9b50fd114196);
    }

    #[test]
    fn fen_round_trip() {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let fen = pos.fen();
        let pos2 = Position::from_fen(&fen).unwrap();
        assert_eq!(pos.chess, pos2.chess);
        assert_eq!(pos.hash, pos2.hash);
    }

    #[test]
    fn castle_preserves_hash() {
        let pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let (o_o, raw) = pos.play_uci("e1g1").unwrap();
        assert!(raw.is_castle());
        assert_eq!(
            o_o.hash,
            o_o.chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal)
        );
        let (o_o_o, _) = pos.play_uci("e1c1").unwrap();
        assert_eq!(
            o_o_o.hash,
            o_o_o.chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal)
        );
        // Castling removes both rights; hashes differ bitwise from sibling lines.
        assert_ne!(o_o.hash, o_o_o.hash);
    }
}

//! [`Position`]: a `shakmaty::Chess` plus an incrementally maintained
//! Zobrist hash.
//!
//! `shakmaty` has no undo (`unmake`), so the search never mutates a position
//! in place: [`Position::make_child`] clones the current position, plays a
//! move on the clone and returns it. Clones are ~80 bytes, i.e. a cheap
//! `memcpy` on the stack, and the parent stays valid for sibling moves.

use anyhow::{Context, Result, anyhow};
use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{CastlingMode, CastlingSide, Chess, Color, EnPassantMode, Role, Square};
use shakmaty::{FromSetup as _, Position as _};

use crate::book::polyglot_key;
use crate::types::{MoveList, RawMove};

/// A position ready for search.
#[derive(Clone, Debug)]
pub struct Position {
    pub chess: Chess,
    /// Zobrist hash of `chess`, maintained incrementally when possible.
    /// Halfmove/fullmove counters are deliberately excluded (matches
    /// `zobrist_hash(EnPassantMode::Legal)`), so it is usable as the TT key
    /// and for repetition detection.
    pub hash: Zobrist64,
    /// Material keys used by the correction history (see
    /// [`crate::search::correction`]). Each one deliberately ignores pieces the
    /// others own, so all three are invariant under the movement of the pieces
    /// they exclude:
    ///
    /// * `pawn_key` — pawn placement only.
    /// * `minor_key` — knight and bishop placement only.
    /// * `non_pawn_key[c]` — every non-pawn piece of colour `c`.
    ///
    /// Maintained incrementally alongside `hash` (a move touches at most three
    /// squares plus one captured piece), with a `debug_assert` in
    /// [`Position::make_child`] that the incremental update agrees with a full
    /// recomputation.
    pub keys: MaterialKeys,
}

/// The three material keys that address the correction-history tables.
///
/// Kept as a plain 3-word struct so cloning a `Position` (which the search does
/// for every node) costs three more words.
///
/// The families partition the board the way Stockfish's do, and the overlap
/// between them is deliberate — it is not "three disjoint views":
///
/// * `pawn` — pawns only.
/// * `minor` — knights and bishops only (the king is deliberately *excluded*,
///   matching `minorPieceKey`).
/// * `non_pawn[color]` — everything that is not a pawn **of that colour**:
///   knight, bishop, rook, queen and king, matching `nonPawnKey`.
///
/// So a minor piece belongs to both `minor` and `non_pawn`, and a key
/// containing neither pawns nor minors says nothing about the placement.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MaterialKeys {
    pub pawn: u64,
    pub minor: u64,
    pub non_pawn: [u64; 2],
}

/// Deterministic per-(role, square) Zobrist keys for the material keys.
///
/// The correction history only needs collision resistance and run-to-run
/// stability, not bit-compatibility with Stockfish's own `Zobrist::psq`, so the
/// table is generated from a fixed-seed SplitMix64 instead of shipping 384
/// magic numbers. `LazyLock` keeps it out of the hot path: the table is built
/// once, then every lookup is a plain array read.
// Indexed by colour too: a white and a black pawn on the same square are
// different structures and must not share a key.
static MATERIAL_ZOBRIST: std::sync::LazyLock<[[u64; 64]; 14]> = std::sync::LazyLock::new(|| {
    let mut table = [[0u64; 64]; 14];
    let mut state = 0x5DEE_CE66_0B1C_2F35u64;
    for role in table.iter_mut() {
        for key in role.iter_mut() {
            // SplitMix64: a fixed seed makes the table identical in every
            // process, so a report is reproducible across runs and machines.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *key = z ^ (z >> 31);
        }
    }
    table
});

/// The key for `role` standing on `square`.
#[inline]
fn piece_key(role: Role, color: Color, square: Square) -> u64 {
    MATERIAL_ZOBRIST[color as usize * 7 + role as usize][square.to_usize()]
}

impl MaterialKeys {
    /// Computes all three keys from a board (O(pieces); only used when a
    /// position is built from scratch or a move cannot be applied by XOR).
    fn recompute(board: &shakmaty::Board) -> MaterialKeys {
        let mut keys = MaterialKeys::default();
        for sq in board.occupied() {
            let Some(piece) = board.piece_at(sq) else {
                continue;
            };
            let k = piece_key(piece.role, piece.color, sq);
            match piece.role {
                Role::Pawn => keys.pawn ^= k,
                Role::Knight | Role::Bishop => {
                    keys.minor ^= k;
                    keys.non_pawn[piece.color as usize] ^= k;
                }
                // Rook, queen **and king**. Stockfish's `nonPawnKey` is the
                // complement of `pawnKey` over the whole board, king included;
                // only `minorPieceKey` excludes the king.
                _ => keys.non_pawn[piece.color as usize] ^= k,
            }
        }
        keys
    }

    /// Removes a piece from the keys (used for both the captured victim and the
    /// moving piece's origin).
    #[inline]
    fn remove(&mut self, role: Role, color: Color, square: Square) {
        let k = piece_key(role, color, square);
        match role {
            Role::Pawn => self.pawn ^= k,
            Role::Knight | Role::Bishop => {
                self.minor ^= k;
                self.non_pawn[color as usize] ^= k;
            }
            // Rook, queen **and king**: `non_pawn` is the complement of `pawn`
            // over the whole board, exactly like Stockfish's `nonPawnKey`.
            _ => self.non_pawn[color as usize] ^= k,
        }
    }

    /// Applies `m` (assumed legal in `board`, which is the position *before* the
    /// move) to the keys. `us` is the side to move in that position.
    #[inline]
    fn apply(&mut self, board: &shakmaty::Board, us: Color, m: RawMove) {
        let from = m.from();
        let to = m.to();

        // 1. The captured victim, if any.
        if m.is_en_passant() {
            let cap = match us {
                Color::White => to.offset(-8),
                Color::Black => to.offset(8),
            };
            if let Some(cap) = cap {
                self.remove(Role::Pawn, !us, cap);
            }
        } else if let Some(victim) = board.piece_at(to) {
            self.remove(victim.role, victim.color, to);
        }

        // 2. The moving piece leaves `from` and lands on `to` in its new form.
        let Some(mover) = board.piece_at(from) else {
            return;
        };
        self.remove(mover.role, mover.color, from);
        let landed = m.promotion().unwrap_or(mover.role);
        self.add(landed, mover.color, to);
    }

    /// Places a piece on the keys.
    #[inline]
    fn add(&mut self, role: Role, color: Color, square: Square) {
        let k = piece_key(role, color, square);
        match role {
            Role::Pawn => self.pawn ^= k,
            Role::Knight | Role::Bishop => {
                self.minor ^= k;
                self.non_pawn[color as usize] ^= k;
            }
            // Rook, queen **and king**: `non_pawn` is the complement of `pawn`
            // over the whole board, exactly like Stockfish's `nonPawnKey`.
            _ => self.non_pawn[color as usize] ^= k,
        }
    }
}

impl Position {
    /// The standard chess starting position.
    pub fn startpos() -> Position {
        let chess = Chess::default();
        let hash = chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        Position {
            keys: MaterialKeys::recompute(chess.board()),
            chess,
            hash,
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
        Ok(Position {
            keys: MaterialKeys::recompute(chess.board()),
            chess,
            hash,
        })
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
        let keys = if m.is_castle() {
            MaterialKeys::recompute(chess.board())
        } else {
            let mut keys = self.keys;
            keys.apply(self.chess.board(), self.chess.turn(), m);
            keys
        };

        debug_assert_eq!(
            keys,
            MaterialKeys::recompute(chess.board()),
            "incremental material-key mismatch after {:?}",
            smove
        );

        Position { keys, chess, hash }
    }

    /// The number of pieces on the board.
    ///
    /// Used for the NNUE PSQT piece-count buckets and for dataset statistics;
    /// a bitboard popcount is exact and, unlike a `Setup` round-trip, free.
    #[inline]
    pub fn piece_count(&self) -> usize {
        self.chess.board().occupied().count()
    }

    /// Standard algebraic notation for `m` in this position.
    #[inline]
    pub fn san_of(&self, m: RawMove) -> String {
        San::from_move(self.chess(), m.to_shakmaty(self.chess.board())).to_string()
    }

    /// Parses a SAN move (`Nf3`, `exd5`, `O-O`, `e8=Q`, …) and plays it on a
    /// clone, returning the child position and the [`RawMove`] that was played.
    /// Errors on an illegal or ambiguous move while leaving `self` untouched.
    ///
    /// The lookup goes through shakmaty's own SAN renderer: the legal moves are
    /// enumerated and the one whose canonical SAN matches is played. That is a
    /// linear scan, which is why this is a *parsing* entry point beside
    /// [`Position::play_uci`] and not something the search ever calls — no node
    /// of a search reads SAN. It exists for recorded data: PGN, and the training
    /// dataset's principal variations.
    ///
    /// A trailing `+` or `#` is accepted and ignored. Whether a move gives
    /// check is a consequence of playing it rather than part of identifying it,
    /// and a dataset that omits the suffix must still replay.
    pub fn play_san(&self, san: &str) -> Result<(Position, RawMove)> {
        let want = san.trim().trim_end_matches(['+', '#']).trim();
        for m in self.chess.legal_moves() {
            if San::from_move(self.chess(), m).to_string() == want {
                let raw = RawMove::from_shakmaty(m);
                return Ok((self.make_child(raw), raw));
            }
        }
        Err(anyhow!(
            "illegal SAN move {san:?} in position {}",
            self.fen()
        ))
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

        let child_chess = {
            let mut c = self.chess.clone();
            c.play_unchecked(m);
            c
        };

        let keys = if raw.is_castle() {
            MaterialKeys::recompute(child_chess.board())
        } else {
            let mut keys = self.keys;
            keys.apply(self.chess.board(), self.chess.turn(), raw);
            keys
        };

        let child = Position {
            chess: child_chess,
            hash: self
                .chess
                .update_zobrist_hash::<Zobrist64>(self.hash, m, EnPassantMode::Legal)
                .unwrap_or_else(|| {
                    let mut c = self.chess.clone();
                    c.play_unchecked(m);
                    c.zobrist_hash::<Zobrist64>(EnPassantMode::Legal)
                }),
            keys,
        };
        debug_assert_eq!(
            child.hash,
            child.chess.zobrist_hash::<Zobrist64>(EnPassantMode::Legal)
        );
        debug_assert_eq!(
            child.keys,
            MaterialKeys::recompute(child.chess.board()),
            "incremental material-key mismatch after {uci}"
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
        // Passing changes no piece, so every material key survives untouched.
        Some(Position {
            keys: self.keys,
            chess,
            hash,
        })
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
        // Ordering tiers read the tunable material values; outside a search
        // (no `Searcher` in scope) the baseline parameter set is authoritative.
        crate::move_ordering::order_moves(
            &mut moves,
            self,
            tables,
            RawMove::NULL,
            crate::evaluation::default_params(),
        );
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
        // `make_child` carries a `debug_assert` that the incremental material
        // keys match a full recomputation, so this same walk validates them
        // too (including captures, en passant, promotions and castling).
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
            assert_eq!(
                child.keys,
                MaterialKeys::recompute(child.chess.board()),
                "material keys diverged at fen {}",
                child.fen()
            );
            pos = child;
        }
    }

    #[test]
    fn null_move_keeps_every_material_key() {
        // Passing relocates no piece, so the correction-history keys must be
        // bit-identical (only the turn-dependent `hash` changes).
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let nulled = pos.null_move().expect("not in check");
        assert_eq!(nulled.keys, pos.keys);
        assert_ne!(nulled.hash, pos.hash, "the turn is part of the TT key");
    }

    /// Plays the listed UCI moves from the start position, returning the last
    /// position. Panics on an illegal move, so a fixture typo fails loudly.
    fn play(moves: &[&str]) -> Position {
        let mut pos = Position::startpos();
        for m in moves {
            let (child, _) = pos.play_uci(m).expect("legal move in fixture");
            pos = child;
        }
        pos
    }

    #[test]
    fn keys_ignore_the_pieces_they_are_not_meant_to_describe() {
        // The point of three separate keys: each is invariant under the movement
        // of the pieces the others own. A pawn move leaves both `minor` and
        // `non_pawn` bit-identical; a knight move is visible in `minor` *and* in
        // `non_pawn`, because a minor piece belongs to both families.
        let start = Position::startpos();
        let after_knight = play(&["g1f3"]);
        assert_eq!(after_knight.keys.pawn, start.keys.pawn, "no pawn moved");
        assert_ne!(after_knight.keys.minor, start.keys.minor, "a knight moved");
        assert_ne!(
            after_knight.keys.non_pawn, start.keys.non_pawn,
            "a knight is a non-pawn piece"
        );

        let after_pawn = play(&["e2e4"]);
        assert_ne!(after_pawn.keys.pawn, start.keys.pawn, "a pawn moved");
        assert_eq!(
            after_pawn.keys.minor, start.keys.minor,
            "no minor piece moved"
        );
        assert_eq!(
            after_pawn.keys.non_pawn, start.keys.non_pawn,
            "a pawn is not a non-pawn piece"
        );

        // Captures must update the victim's key and leave the capturer's own
        // key family alone. 1.e4 d5 2.exd5 removes a black pawn.
        let after_capture = play(&["e2e4", "d7d5", "e4d5"]);
        let before_capture = play(&["e2e4", "d7d5"]);
        assert_ne!(after_capture.keys.pawn, before_capture.keys.pawn);
        assert_eq!(
            after_capture.keys.non_pawn, before_capture.keys.non_pawn,
            "a pawn exchange moves no non-pawn piece"
        );
    }

    #[test]
    fn a_minor_move_leaves_the_pawn_key_alone() {
        // 1.Na3 moves one piece and no pawn, so the pawn key must come back
        // bit-identical from the start position while `minor` — and `non_pawn`,
        // which a knight also belongs to — both move.
        let start = Position::startpos();
        let after = play(&["b1a3"]);
        assert_eq!(after.keys.pawn, start.keys.pawn, "no pawn moved");
        assert_ne!(after.keys.minor, start.keys.minor, "a knight moved");
        assert_ne!(
            after.keys.non_pawn, start.keys.non_pawn,
            "a knight is a non-pawn piece"
        );
    }

    #[test]
    fn a_pawn_move_also_moves_the_pawn_key() {
        // The mirror of the test above, and the reason the previous fixture was
        // wrong: a pawn that steps c2 -> c3 does **not** cancel out, because the
        // key covers the square too. The keys describe *placement*, not counts.
        let start = Position::startpos();
        let after = play(&["b1a3", "a7a6", "c2c3"]);
        assert_ne!(after.keys.pawn, start.keys.pawn, "two pawns moved");
    }

    #[test]
    fn a_king_move_disturbs_the_non_pawn_key_only() {
        // The one overlap rule that is easy to get backwards: the king is a
        // non-pawn piece (Stockfish's `nonPawnKey` is the complement of
        // `pawnKey` over the whole board) but is explicitly *not* a minor piece
        // (`minorPieceKey` stops at the bishop). So a king move moves exactly one
        // of the two.
        let start = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let after = start.make_child(start.raw_move_from_uci("e1e2").unwrap());
        assert_eq!(after.keys.pawn, start.keys.pawn, "no pawn moved");
        assert_eq!(
            after.keys.minor, start.keys.minor,
            "a king is not a minor piece"
        );
        assert_ne!(
            after.keys.non_pawn, start.keys.non_pawn,
            "a king is a non-pawn piece"
        );
    }

    #[test]
    fn promotion_and_capture_rewrite_the_right_keys() {
        // e7xe8=Q captures a rook: the pawn leaves the pawn key, the rook leaves
        // black's non-pawn key and the new queen enters white's.
        let pos = Position::from_fen("3r2k1/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promo = pos.raw_move_from_uci("e7d8q").unwrap();
        let after = pos.make_child(promo);
        let expected = MaterialKeys::recompute(after.chess.board());
        assert_eq!(after.keys, expected);
        assert_ne!(after.keys.pawn, pos.keys.pawn, "the pawn is gone");
        assert_ne!(
            after.keys.non_pawn[1], pos.keys.non_pawn[1],
            "black's rook was captured"
        );
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

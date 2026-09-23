//! Core value types shared across the engine: scores, depths, compact moves.
//!
//! Everything in here is `Copy`-able, stack-sized and cache friendly. There is
//! deliberately no heap allocation in the hot path.

use shakmaty::{Move, Role, Square};

/// Maximum search ply (both for the search stack and mate scoring).
pub const MAX_PLY: usize = 128;

/// Maximum iteratively deepening depth.
pub const MAX_DEPTH: i32 = 64;

/// Score of a checkmate, relative to the side to move at the node where the
/// score is produced. `MATE - ply` for "mate in `ply`", `-MATE + ply` for
/// "mated in `ply`".
pub const MATE: i32 = 32000;

/// Score larger than any real score, used as an upper/lower bound sentinel.
pub const INFINITE: i32 = 32001;

/// The score below which a simple `draw` (e.g. material draw) contribution
/// begins; used by endgame/eval draw handling.
pub const DRAW: i32 = 0;

/// Number of ulps from `MATE` inside which a score is treated as a mate score.
const MATE_THRESHOLD: i32 = 8000;

#[inline]
pub const fn mate_in(ply: i32) -> i32 {
    MATE - ply
}

#[inline]
pub const fn mated_in(ply: i32) -> i32 {
    -MATE + ply
}

#[inline]
pub const fn is_mate(score: i32) -> bool {
    score > MATE - MATE_THRESHOLD || score < -MATE + MATE_THRESHOLD
}

/// Converts a "mate in N plies" score to a human oriented `N`.
#[inline]
pub const fn mate_plies(score: i32) -> i32 {
    MATE - score.abs()
}

/// Search depth: plain `i32` on purpose (cheap arithmetic, no newtype
/// friction). TT entries store depth as a single `u8`.
pub type Depth = i32;

/// A compact, lossless encoding of a legal shakmaty move in 16 bits.
///
/// Layout:
///
/// ```text
/// bits  0..=5  from square (0..64)
/// bits  6..=11 to square   (0..64)
/// bits 12..=15 kind:
///   0 = normal move
///   1..=4 = promotion to Queen/Rook/Bishop/Knight
///   5 = en passant capture
///   6 = castling (king square in `from`, rook square in `to`)
/// ```
///
/// The encoding is stable and cheap to build; conversion to `shakmaty::Move`
/// needs the current board (for the moving piece and the captured piece).
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RawMove(u16);

impl RawMove {
    pub const NULL: RawMove = RawMove(0);

    /// Kind markers.
    pub const NORMAL: u16 = 0;
    pub const PROMO_QUEEN: u16 = 1;
    pub const PROMO_ROOK: u16 = 2;
    pub const PROMO_BISHOP: u16 = 3;
    pub const PROMO_KNIGHT: u16 = 4;
    pub const EN_PASSANT: u16 = 5;
    pub const CASTLE: u16 = 6;

    #[inline]
    pub const fn new(from: Square, to: Square, kind: u16) -> RawMove {
        RawMove(
            (kind << 12) | ((to.to_usize() as u16 & 0x3f) << 6) | (from.to_usize() as u16 & 0x3f),
        )
    }

    #[inline]
    pub const fn from(self) -> Square {
        Square::new((self.0 & 0x3f) as u32)
    }

    #[inline]
    pub const fn to(self) -> Square {
        Square::new(((self.0 >> 6) & 0x3f) as u32)
    }

    #[inline]
    pub const fn kind(self) -> u16 {
        self.0 >> 12
    }

    #[inline]
    pub const fn is_en_passant(self) -> bool {
        self.kind() == Self::EN_PASSANT
    }

    /// The raw 16-bit encoding (for packing into TT slots / logs).
    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Builds a move from a raw 16-bit encoding (used when decoding packed
    /// entries; the value must come from [`RawMove::raw`]).
    #[inline]
    pub const fn from_raw(v: u16) -> RawMove {
        RawMove(v)
    }

    #[inline]
    pub const fn is_castle(self) -> bool {
        self.kind() == Self::CASTLE
    }

    #[inline]
    pub const fn is_promotion(self) -> bool {
        matches!(
            self.kind(),
            Self::PROMO_QUEEN | Self::PROMO_ROOK | Self::PROMO_BISHOP | Self::PROMO_KNIGHT
        )
    }

    /// The promotion piece for a promotion move.
    #[inline]
    pub const fn promotion(self) -> Option<Role> {
        match self.kind() {
            Self::PROMO_QUEEN => Some(Role::Queen),
            Self::PROMO_ROOK => Some(Role::Rook),
            Self::PROMO_BISHOP => Some(Role::Bishop),
            Self::PROMO_KNIGHT => Some(Role::Knight),
            _ => None,
        }
    }

    /// A move is "tactical" if it is a capture, promotion or en passant.
    /// The capture test requires board context and lives on the board module;
    /// this covers the statically known subset (promotions, e.p., castle).
    #[inline]
    pub const fn is_tactical_kind(self) -> bool {
        self.is_promotion() || self.is_en_passant()
    }

    /// Encodes a `shakmaty::Move` into a `RawMove`. Only standard-chess
    /// variants (no drops) can appear.
    pub const fn from_shakmaty(m: Move) -> RawMove {
        match m {
            Move::Normal {
                from,
                to,
                promotion,
                ..
            } => match promotion {
                None => RawMove::new(from, to, RawMove::NORMAL),
                Some(role) => RawMove::new(from, to, RawMove::kind_for(role)),
            },
            Move::EnPassant { from, to } => RawMove::new(from, to, RawMove::EN_PASSANT),
            Move::Castle { king, rook } => RawMove::new(king, rook, RawMove::CASTLE),
            Move::Put { .. } => RawMove::NULL,
        }
    }

    /// Reconstitutes the `shakmaty::Move`; requires the board *before* the
    /// move so the moving/captured pieces can be looked up.
    #[inline]
    pub fn to_shakmaty(self, board: &shakmaty::Board) -> Move {
        let from = self.from();
        let to = self.to();
        match self.kind() {
            RawMove::PROMO_QUEEN
            | RawMove::PROMO_ROOK
            | RawMove::PROMO_BISHOP
            | RawMove::PROMO_KNIGHT => Move::Normal {
                role: Role::Pawn,
                from,
                capture: board.role_at(to),
                to,
                promotion: self.promotion(),
            },
            RawMove::EN_PASSANT => Move::EnPassant { from, to },
            RawMove::CASTLE => Move::Castle {
                king: from,
                rook: to,
            },
            _ => Move::Normal {
                role: board.role_at(from).expect("moving piece present"),
                from,
                capture: board.role_at(to),
                to,
                promotion: None,
            },
        }
    }

    /// UCI text form, e.g. `e2e4`, `e7e8q`, `e1g1`. Castling is sent as the king
    /// move per UCI convention (the internal encoding stores the rook square
    /// in `to`, the text form converts it to the king's destination).
    pub fn to_uci(self) -> String {
        let mut s = String::with_capacity(5);
        push_square(&mut s, self.from());
        push_square(&mut s, self.uci_to());
        if let Some(role) = self.promotion() {
            s.push(role.char());
        }
        s
    }

    /// The UCI "to" square: for castling this is the king's destination
    /// (g1/c1/g8/c8) rather than the stored rook square.
    #[inline]
    pub fn uci_to(self) -> Square {
        if self.is_castle() {
            let file = if self.to().file() == shakmaty::File::H {
                shakmaty::File::G
            } else {
                shakmaty::File::C
            };
            Square::from_coords(file, self.from().rank())
        } else {
            self.to()
        }
    }

    #[inline]
    const fn kind_for(role: Role) -> u16 {
        match role {
            Role::Queen => RawMove::PROMO_QUEEN,
            Role::Rook => RawMove::PROMO_ROOK,
            Role::Bishop => RawMove::PROMO_BISHOP,
            Role::Knight => RawMove::PROMO_KNIGHT,
            _ => RawMove::NORMAL,
        }
    }
}

/// Pushes `file` + `rank` UCI square text (e.g. `e4`) into `out`.
fn push_square(out: &mut String, sq: Square) {
    let (file, rank) = sq.coords();
    out.push((b'a' + u8::from(file)) as char);
    out.push((b'1' + u8::from(rank)) as char);
}

impl Default for RawMove {
    fn default() -> Self {
        RawMove::NULL
    }
}

/// A fixed-capacity stack list of `RawMove`s. Bounds are compile-time
/// constants, so there is no heap traffic while generating moves.
#[derive(Clone)]
pub struct MoveList {
    moves: [RawMove; MAX_MOVES],
    len: usize,
}

/// `270` mirrors shakmaty's own `MoveList` capacity, which is sized for a
/// pathological standard-chess position (`kBQQQQQQ/...`).
pub const MAX_MOVES: usize = 270;

impl MoveList {
    #[inline]
    pub fn new() -> MoveList {
        MoveList {
            moves: [RawMove::NULL; MAX_MOVES],
            len: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn push(&mut self, m: RawMove) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[RawMove] {
        &self.moves[..self.len]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [RawMove] {
        &mut self.moves[..self.len]
    }

    #[inline]
    pub fn get(&self, i: usize) -> RawMove {
        self.moves[i]
    }

    #[inline]
    pub fn set(&mut self, i: usize, m: RawMove) {
        self.moves[i] = m;
    }

    pub fn iter(&self) -> impl Iterator<Item = RawMove> + '_ {
        self.moves[..self.len].iter().copied()
    }

    #[inline]
    pub fn contains(&self, m: RawMove) -> bool {
        self.moves[..self.len].contains(&m)
    }
}

impl Default for MoveList {
    fn default() -> Self {
        MoveList::new()
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = RawMove;
    type IntoIter = core::iter::Copied<core::slice::Iter<'a, RawMove>>;

    fn into_iter(self) -> Self::IntoIter {
        self.moves[..self.len].iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{EnPassantMode, Position};

    #[test]
    fn raw_move_round_trips_uci_text() {
        let pos = shakmaty::Chess::default();
        for m in pos.legal_moves() {
            let raw = RawMove::from_shakmaty(m);
            let expected = m.to_uci(shakmaty::CastlingMode::Standard).to_string();
            assert_eq!(raw.to_uci(), expected, "{:?}", m);
        }
    }

    #[test]
    fn raw_move_round_trips_shakmaty() {
        let pos = shakmaty::Chess::default();
        for m in pos.legal_moves() {
            let raw = RawMove::from_shakmaty(m);
            let back = raw.to_shakmaty(pos.board());
            assert_eq!(back, m, "round trip failed");
        }
    }

    #[test]
    fn castling_encoding_uses_king_and_rook_squares() {
        let fen: shakmaty::fen::Fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".parse().unwrap();
        let pos: shakmaty::Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
        let castle: Vec<_> = pos
            .legal_moves()
            .iter()
            .copied()
            .filter(|m| m.is_castle())
            .collect();
        assert_eq!(castle.len(), 2);
        for m in castle {
            let raw = RawMove::from_shakmaty(m);
            assert_eq!(
                raw.to_uci(),
                m.to_uci(shakmaty::CastlingMode::Standard).to_string()
            );
            assert_eq!(raw.to_shakmaty(pos.board()), m);
        }
    }

    #[test]
    fn en_passant_round_trip() {
        let pos: shakmaty::Chess =
            crate::testutil::chess("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        let ep = pos
            .legal_moves()
            .iter()
            .copied()
            .find(|m| m.is_en_passant())
            .expect("en passant available");
        let raw = RawMove::from_shakmaty(ep);
        assert!(raw.is_en_passant());
        assert_eq!(raw.to_shakmaty(pos.board()), ep);
        assert_eq!(raw.to_uci(), "e5f6");
        // Zobrist sanity while we are here.
        let _ = pos.zobrist_hash::<shakmaty::zobrist::Zobrist64>(EnPassantMode::Legal);
    }

    #[test]
    fn mate_helpers() {
        assert!(is_mate(mate_in(3)));
        assert!(is_mate(mated_in(7)));
        assert_eq!(mate_plies(mate_in(3)), 3);
        assert!(!is_mate(0));
        assert!(!is_mate(900));
    }

    #[test]
    fn move_list_capacity() {
        let mut ml = MoveList::new();
        for i in 0..MAX_MOVES {
            ml.push(RawMove::new(
                Square::new((i % 64) as u32),
                Square::new(((i + 1) % 64) as u32),
                0,
            ));
        }
        assert_eq!(ml.len(), MAX_MOVES);
        assert!(ml.contains(ml.get(0)));
    }
}

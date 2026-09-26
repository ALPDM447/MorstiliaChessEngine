//! NNUE value types: colors, piece codes, fixed-capacity index lists and the
//! "dirty" change records the accumulator replays.
//!
//! The piece/color codes deliberately mirror Stockfish's own numbering, because
//! the `.nnue` feature indices are computed from them: a piece is
//! `1..=6` for White (`Pawn..King`) and `9..=14` for Black, with
//! `type_of(pc) == pc & 7` and `color_of(pc) == pc >> 3`. [`Color`] is
//! `White = 0, Black = 1` for the same reason — *shakmaty* numbers the
//! opposite way round, so [`Color::from_shakmaty`] converts explicitly rather
//! than transmuting.

use std::mem::MaybeUninit;

/// A chessboard bitboard (`a1` = bit 0 … `h8` = bit 63, the same square
/// numbering Stockfish's `Square` uses).
pub type Bitboard = u64;

/// Sentinel square for "no square" (Stockfish's `SQ_NONE`). It is *not* a
/// valid bitboard index — every accessor that takes a square must have
/// filtered this out already.
pub const SQ_NONE: usize = 64;

/// Side to move, in Stockfish's order (`White = 0`).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn other(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub const fn from_index(i: usize) -> Color {
        if i == 0 { Color::White } else { Color::Black }
    }

    /// shakmaty numbers `Black = 0, White = 1`; Stockfish the other way round.
    #[inline]
    pub const fn from_shakmaty(c: shakmaty::Color) -> Color {
        match c {
            shakmaty::Color::White => Color::White,
            shakmaty::Color::Black => Color::Black,
        }
    }
}

// --- piece types ------------------------------------------------------------

pub const PAWN: u8 = 1;
pub const KNIGHT: u8 = 2;
pub const BISHOP: u8 = 3;
pub const ROOK: u8 = 4;
pub const QUEEN: u8 = 5;
pub const KING: u8 = 6;

pub const NO_PIECE: u8 = 0;
pub const W_PAWN: u8 = PAWN;
pub const W_KNIGHT: u8 = KNIGHT;
pub const W_BISHOP: u8 = BISHOP;
pub const W_ROOK: u8 = ROOK;
pub const W_QUEEN: u8 = QUEEN;
pub const W_KING: u8 = KING;
pub const B_PAWN: u8 = PAWN + 8;
pub const B_KNIGHT: u8 = KNIGHT + 8;
pub const B_BISHOP: u8 = BISHOP + 8;
pub const B_ROOK: u8 = ROOK + 8;
pub const B_QUEEN: u8 = QUEEN + 8;
pub const B_KING: u8 = KING + 8;
pub const PIECE_NB: usize = 16;

/// Stockfish's material constants (`PawnValue` … `QueenValue`), indexed by
/// piece code. Used for `non_pawn_material()`.
pub const PIECE_VALUE: [i32; PIECE_NB] = [
    0, 208, 781, 825, 1276, 2538, 0, 0, //
    0, 208, 781, 825, 1276, 2538, 0, 0,
];

#[inline]
pub const fn type_of(pc: u8) -> u8 {
    pc & 7
}

#[inline]
pub const fn color_of(pc: u8) -> usize {
    (pc >> 3) as usize
}

#[inline]
pub const fn make_piece(c: usize, pt: u8) -> u8 {
    ((c as u8) << 3) | pt
}

/// The 12 "real" pieces in Stockfish's enumeration order (White `Pawn..King`,
/// then Black). Several lookup tables are built by walking exactly this
/// sequence, so the order is load-bearing.
pub const ALL_PIECES: [u8; 12] = [
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, //
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
];

// --- fixed-capacity list ----------------------------------------------------

/// A `push`-only list with a compile-time capacity — Stockfish's
/// `ValueList<T, N>`.
///
/// The backing array is left **uninitialized** and only the first `size`
/// elements are ever read, so creating one in the hot path costs no
/// initialization at all (Stockfish zeroes the whole array instead). All
/// element types are `Copy`, which is what makes the unchecked reads below
/// sound: a slot is written before it is ever exposed through `size`.
pub struct ValueList<T: Copy, const N: usize> {
    size: usize,
    values: [MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> ValueList<T, N> {
    #[inline]
    pub const fn new() -> Self {
        ValueList {
            size: 0,
            values: [MaybeUninit::uninit(); N],
        }
    }

    /// The live prefix. Never reads an unwritten slot.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: only slots below `size` are written before `size` is
        // incremented (see `push`), and `size` only ever grows to at most `N`.
        unsafe { std::slice::from_raw_parts(self.values.as_ptr() as *const T, self.size) }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Appends `v`, panicking in debug builds if the (documented) capacity is
    /// exceeded. The capacity bounds are derived from Stockfish's own analysis
    /// of the maximum feature change a single move can cause, so an overflow
    /// means a real bug rather than a tuning issue.
    #[inline]
    pub fn push(&mut self, v: T) {
        debug_assert!(self.size < N, "ValueList capacity {N} exceeded");
        if self.size < N {
            self.values[self.size].write(v);
            self.size += 1;
        }
    }
}

/// The feature sets address their weights with a `u16` index and drop anything
/// `>= Dimensions`, so the filter is specialised to that element type.
impl<const N: usize> ValueList<u16, N> {
    /// Appends `v` only when `v < limit`.
    #[inline]
    pub fn push_if_lt(&mut self, v: u16, limit: usize) {
        if (v as usize) < limit {
            self.push(v);
        }
    }
}

impl<T: Copy, const N: usize> Default for ValueList<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Manual because `MaybeUninit` implements neither. Copying the whole array
/// (not just the live prefix) keeps this a memcpy, which is what the dirty
/// records in a `MoveDirties` need to be `Copy`.
impl<T: Copy, const N: usize> Clone for ValueList<T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy, const N: usize> Copy for ValueList<T, N> {}

impl<T: Copy, const N: usize> core::fmt::Debug for ValueList<T, N>
where
    T: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: Copy, const N: usize> core::ops::Deref for ValueList<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

// --- dirty-piece / dirty-threat records -------------------------------------

/// The single piece a move changed, as Stockfish's `DirtyPiece`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DirtyPiece {
    /// Never `NO_PIECE`.
    pub pc: u8,
    /// Origin square.
    pub from: usize,
    /// Destination square; `SQ_NONE` for promotions (the added index then
    /// comes from `add_sq` instead).
    pub to: usize,
    /// `SQ_NONE` when unused.
    pub remove_sq: usize,
    /// `SQ_NONE` when unused.
    pub add_sq: usize,
    /// Only meaningful when `remove_sq != SQ_NONE`.
    pub remove_pc: u8,
    /// Only meaningful when `add_sq != SQ_NONE`.
    pub add_pc: u8,
}

impl DirtyPiece {
    #[inline]
    pub const fn new(pc: u8, from: usize, to: usize) -> Self {
        DirtyPiece {
            pc,
            from,
            to,
            remove_sq: SQ_NONE,
            add_sq: SQ_NONE,
            remove_pc: NO_PIECE,
            add_pc: NO_PIECE,
        }
    }
}

/// One `attacker → target` threat relationship that a move created or
/// destroyed, packed into a single `u32` exactly like Stockfish's
/// `DirtyThreat` (bit-for-bit the same layout, so the packing cost stays one
/// shift-and-or per emitted threat).
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct DirtyThreat(u32);

const PC_SQ_SHIFT: u32 = 0;
const THREATENED_SQ_SHIFT: u32 = 8;
const THREATENED_PC_SHIFT: u32 = 16;
const PC_SHIFT: u32 = 20;

impl DirtyThreat {
    #[inline]
    pub const fn new(
        pc: u8,
        threatened_pc: u8,
        pc_sq: usize,
        threatened_sq: usize,
        add: bool,
    ) -> Self {
        DirtyThreat(
            ((add as u32) << 31)
                | ((pc as u32) << PC_SHIFT)
                | ((threatened_pc as u32) << THREATENED_PC_SHIFT)
                | ((threatened_sq as u32) << THREATENED_SQ_SHIFT)
                | ((pc_sq as u32) << PC_SQ_SHIFT),
        )
    }

    /// The threatening piece.
    #[inline]
    pub const fn pc(self) -> u8 {
        ((self.0 >> PC_SHIFT) & 0xf) as u8
    }

    /// The threatened piece.
    #[inline]
    pub const fn threatened_pc(self) -> u8 {
        ((self.0 >> THREATENED_PC_SHIFT) & 0xf) as u8
    }

    #[inline]
    pub const fn threatened_sq(self) -> usize {
        ((self.0 >> THREATENED_SQ_SHIFT) & 0xff) as usize
    }

    /// The square the threatening piece stands on.
    #[inline]
    pub const fn pc_sq(self) -> usize {
        ((self.0 >> PC_SQ_SHIFT) & 0xff) as usize
    }

    #[inline]
    pub const fn add(self) -> bool {
        self.0 >> 31 != 0
    }
}

/// Every threat relationship the move touched.
///
/// Capacity 96 is Stockfish's own bound: a non-castling move changes at most
/// `(8 + 16) * 3 + 8 = 80` features, and a castling move at most 36; the
/// remaining 16 entries exist only so vector stores can overshoot harmlessly.
#[derive(Clone, Copy)]
pub struct DirtyThreats {
    pub list: ValueList<DirtyThreat, 96>,
}

impl DirtyThreats {
    #[inline]
    pub const fn new() -> Self {
        DirtyThreats {
            list: ValueList::new(),
        }
    }

    #[inline]
    pub fn push(&mut self, t: DirtyThreat) {
        self.list.push(t);
    }
}

impl Default for DirtyThreats {
    fn default() -> Self {
        Self::new()
    }
}

/// The pawn bitboards before and after a move, feeding the `PP_3Wide` feature
/// set (which only ever looks at pawns).
#[derive(Copy, Clone, Default)]
pub struct DirtyPawnPairs {
    pub before: [Bitboard; 2],
    pub after: [Bitboard; 2],
}

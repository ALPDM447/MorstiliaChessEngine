//! A lock-free transposition table.
//!
//! Layout: one 12-byte entry = a 32-bit key plus a packed 64-bit payload
//! (`mv | score | depth | bound | generation`), stored as two independent
//! aligned arrays of `AtomicU32`/`AtomicU64` so every write is a single-word
//! atomic store and readers never observe torn writes.
//!
//! # Concurrency notes
//!
//! Reads issue two independent loads; under multi-threaded search a torn
//! `(key, payload)` pair is theoretically possible (the new key with an old
//! payload). This is harmless by design: TT moves are only ever played when
//! they appear in the *generated legal move list*, so a foreign move is
//! silently rejected, and a stray bound just perturbs a bounded-time search
//! within a single iteration. Single-threaded search (the default) is fully
//! deterministic: `store`/`probe` are pure functions of insertion order.
//!
//! # Replacement
//!
//! Single-slot with generation aging: entries from an older search are
//! replaced eagerly; otherwise a same-key hit refreshes unconditionally and
//! a same-depth-or-deeper entry keeps the slot.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::types::{Depth, RawMove};

/// How confident a stored score is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    /// Score is exact.
    Exact = 0,
    /// Score is a lower bound (`>= beta` at the stored node).
    Lower = 1,
    /// Score is an upper bound (`<= alpha` at the stored node).
    Upper = 2,
}

impl From<u8> for Bound {
    #[inline]
    fn from(v: u8) -> Bound {
        match v {
            1 => Bound::Lower,
            2 => Bound::Upper,
            _ => Bound::Exact,
        }
    }
}

/// A decoded TT entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TtEntry {
    pub mv: RawMove,
    pub score: i32,
    pub depth: Depth,
    pub bound: Bound,
}

/// Packed 64-bit payload:
/// ```text
///  0..=15  move (RawMove)
/// 16..=31  score (i16)
/// 32..=39  depth (u8)
/// 40..=47  bound (u8)
/// 48..=55  generation (u8)
/// 56..=63  unused
/// ```
#[derive(Copy, Clone, Default)]
#[repr(align(8))]
struct Payload(u64);

impl Payload {
    #[inline]
    fn new(mv: RawMove, score: i32, depth: Depth, bound: Bound, gen_id: u8) -> Payload {
        // Mask the score through i16 → u16 so sign extension never bleeds
        // into the depth/bound/generation fields above bit 31.
        let score = score.clamp(i16::MIN as i32, i16::MAX as i32) as u16;
        Payload(
            u64::from(mv.raw())
                | (u64::from(score) << 16)
                | (u64::from(depth.clamp(0, 255) as u32) << 32)
                | (u64::from(bound as u8) << 40)
                | (u64::from(gen_id) << 48),
        )
    }

    #[inline]
    fn mv(&self) -> RawMove {
        RawMove::from_raw(self.0 as u16)
    }

    #[inline]
    fn score(&self) -> i32 {
        ((self.0 >> 16) & 0xffff) as u16 as i16 as i32
    }

    #[inline]
    fn depth(&self) -> Depth {
        ((self.0 >> 32) & 0xff) as Depth
    }

    #[inline]
    fn bound(&self) -> Bound {
        Bound::from(((self.0 >> 40) & 0xff) as u8)
    }

    #[inline]
    fn generation(&self) -> u8 {
        ((self.0 >> 48) & 0xff) as u8
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.depth() > 0 || self.0 != 0
    }
}

/// A sized, lock-free transposition table.
pub struct TranspositionTable {
    /// Entry array (usable capacity `mask + 1` entries; `mask` is a power of
    /// two minus one).
    keys: Box<[AtomicU32]>,
    data: Box<[AtomicU64]>,
    mask: usize,
    generation: u8,
    /// Total entries written since the last [`TranspositionTable::new_search`]
    /// (relaxed atomic; diagnostics only — never read in the hot path).
    stores: AtomicU64,
}

const ENTRY_BYTES: usize = 12;

/// Margin (in plies) below which a new entry surrenders the slot to an old
/// deep entry — "gold plating": a 14-ply entry from a previous search is
/// worth more than a fresh 6-ply one.
const DEPTH_MARGIN: Depth = 3;

impl TranspositionTable {
    /// Creates a table of at most `size_mb` megabytes (rounded down to a
    /// power-of-two entry count).
    pub fn new(size_mb: usize) -> TranspositionTable {
        let entries = (size_mb * 1024 * 1024 / ENTRY_BYTES)
            .max(1024)
            .next_power_of_two();
        Self::allocate(entries)
    }

    fn allocate(entries: usize) -> TranspositionTable {
        let entries = entries.max(1).next_power_of_two();
        let keys = (0..entries)
            .map(|_| AtomicU32::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let data = (0..entries)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        TranspositionTable {
            keys,
            data,
            mask: entries - 1,
            generation: 0,
            stores: AtomicU64::new(0),
        }
    }

    /// A tiny single-slot table for unit tests (verify replacement policies
    /// deterministically rather than relying on hash collisions).
    #[cfg(test)]
    fn with_entries(entries: usize) -> TranspositionTable {
        Self::allocate(entries)
    }

    /// Allocates a new table of `size_mb` MB, preserving nothing (used by
    /// `setoption Hash` resizes).
    pub fn resize(&mut self, size_mb: usize) {
        *self = TranspositionTable::new(size_mb);
    }

    pub fn clear(&mut self) {
        for k in self.keys.iter() {
            k.store(0, Ordering::Relaxed);
        }
        for d in self.data.iter() {
            d.store(0, Ordering::Relaxed);
        }
        self.generation = 0;
    }

    #[inline]
    pub fn entries(&self) -> usize {
        self.mask + 1
    }

    /// Approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.entries() * ENTRY_BYTES
    }

    /// Advances the generation (called at the start of each search).
    #[inline]
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.stores.store(0, Ordering::Relaxed);
    }

    /// Total entries written since the start of the current search. Aborted
    /// searches that never reach a `store` call report 0 regardless of how
    /// many nodes they visited.
    #[inline]
    pub fn stores(&self) -> u64 {
        self.stores.load(Ordering::Relaxed)
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        hash as usize & self.mask
    }

    /// Looks up a hash; returns the entry if the stored key matches.
    #[inline]
    pub fn probe(&self, hash: u64) -> Option<TtEntry> {
        let idx = self.index(hash);
        let key = self.keys[idx].load(Ordering::Relaxed);
        let data = Payload(self.data[idx].load(Ordering::Relaxed));
        if key == (hash >> 32) as u32 && data.is_valid() {
            Some(TtEntry {
                mv: data.mv(),
                score: data.score(),
                depth: data.depth(),
                bound: data.bound(),
            })
        } else {
            None
        }
    }

    /// Probes and returns only the stored best move (no depth/bound checks).
    #[inline]
    pub fn probe_move(&self, hash: u64) -> Option<RawMove> {
        let idx = self.index(hash);
        let key = self.keys[idx].load(Ordering::Relaxed);
        let data = Payload(self.data[idx].load(Ordering::Relaxed));
        if key == (hash >> 32) as u32 && data.is_valid() {
            Some(data.mv())
        } else {
            None
        }
    }

    /// Stores an entry. `score`, `depth` and `bound` are relative to the
    /// *storing* node; mate scores are adjusted on read by the caller.
    #[inline]
    pub fn store(&self, hash: u64, mv: RawMove, score: i32, depth: Depth, bound: Bound) {
        self.store_impl(hash, mv, score, depth, bound, self.generation)
    }

    fn store_impl(
        &self,
        hash: u64,
        mv: RawMove,
        score: i32,
        depth: Depth,
        bound: Bound,
        gen_id: u8,
    ) {
        let idx = self.index(hash);
        let key32 = (hash >> 32) as u32;

        if depth <= 0 {
            // Depth-0 entries (e.g. from qsearch) are not stored to avoid
            // flooding the table with low-quality data.
            return;
        }
        self.stores.fetch_add(1, Ordering::Relaxed);

        let old_key = self.keys[idx].load(Ordering::Relaxed);
        if old_key == key32 {
            // Same position: unconditional refresh is fine (depth ≥ other).
            self.data[idx].store(
                Payload::new(mv, score, depth, bound, gen_id).0,
                Ordering::Relaxed,
            );
            self.keys[idx].store(key32, Ordering::Relaxed);
            return;
        }

        let old = Payload(self.data[idx].load(Ordering::Relaxed));
        let replace = !old.is_valid()
            || (depth + DEPTH_MARGIN > old.depth()
                && (old.generation() != gen_id
                    || depth > old.depth()
                    || (depth == old.depth() && bound == Bound::Exact)));
        if replace {
            self.data[idx].store(
                Payload::new(mv, score, depth, bound, gen_id).0,
                Ordering::Relaxed,
            );
            self.keys[idx].store(key32, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn generation(&self) -> u8 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(base: u64) -> u64 {
        base ^ ((base & 0xff) << 32) // realistic 64-bit-ish keys
    }

    #[test]
    fn store_and_probe_exact() {
        let tt = TranspositionTable::with_entries(1);
        let hash = h(0x12345678);
        let mv = RawMove::new(shakmaty::Square::E2, shakmaty::Square::E4, 0);
        tt.store(hash, mv, 125, 6, Bound::Exact);
        let e = tt.probe(hash).unwrap();
        assert_eq!(e.mv, mv);
        assert_eq!(e.score, 125);
        assert_eq!(e.depth, 6);
        assert_eq!(e.bound, Bound::Exact);
    }

    #[test]
    fn same_key_refreshes() {
        let tt = TranspositionTable::with_entries(1);
        let hash = h(42);
        tt.store(
            hash,
            RawMove::new(shakmaty::Square::E2, shakmaty::Square::E4, 0),
            10,
            3,
            Bound::Exact,
        );
        tt.store(
            hash,
            RawMove::new(shakmaty::Square::D2, shakmaty::Square::D4, 0),
            -7,
            9,
            Bound::Lower,
        );
        let e = tt.probe(hash).unwrap();
        assert_eq!(e.mv.to_uci(), "d2d4");
        assert_eq!(e.score, -7);
        assert_eq!(e.depth, 9);
    }

    #[test]
    fn deeper_entry_wins_slot() {
        let tt = TranspositionTable::with_entries(1);
        let h1 = h(1);
        let h2 = h(2);
        tt.store(
            h1,
            RawMove::new(shakmaty::Square::E2, shakmaty::Square::E4, 0),
            1,
            2,
            Bound::Exact,
        );
        tt.store(
            h2,
            RawMove::new(shakmaty::Square::E2, shakmaty::Square::E4, 0),
            2,
            9,
            Bound::Exact,
        );
        assert!(tt.probe(h1).is_none());
        assert!(tt.probe(h2).is_some());
        // A shallow new entry does NOT evict the deep one.
        tt.store(
            h1,
            RawMove::new(shakmaty::Square::A2, shakmaty::Square::A3, 0),
            3,
            1,
            Bound::Exact,
        );
        assert!(tt.probe(h1).is_none());
        assert!(tt.probe(h2).is_some());
    }

    #[test]
    fn generations_expire_old_entries() {
        let mut tt = TranspositionTable::with_entries(1);
        let h1 = h(11);
        tt.store(
            h1,
            RawMove::new(shakmaty::Square::E2, shakmaty::Square::E4, 0),
            5,
            4,
            Bound::Exact,
        );
        tt.new_search(); // new generation
        let h2 = h(22);
        tt.store(
            h2,
            RawMove::new(shakmaty::Square::D2, shakmaty::Square::D4, 0),
            6,
            4,
            Bound::Exact,
        );
        // h1 (old generation) got evicted by h2.
        assert!(tt.probe(h1).is_none());
        assert!(tt.probe(h2).is_some());
    }

    #[test]
    fn gold_plating_protects_old_but_deep_entries() {
        let mut tt = TranspositionTable::with_entries(1);
        let h1 = h(111);
        tt.store(
            h1,
            RawMove::new(shakmaty::Square::G1, shakmaty::Square::F3, 0),
            5,
            14,
            Bound::Exact,
        );
        tt.new_search();
        let h2 = h(222);
        // h2 is shallower than h1's stored depth → should NOT evict.
        tt.store(
            h2,
            RawMove::new(shakmaty::Square::G1, shakmaty::Square::F3, 0),
            6,
            6,
            Bound::Exact,
        );
        assert!(tt.probe(h1).is_some(), "deep old entry survives");
        assert!(tt.probe(h2).is_none());
    }
}

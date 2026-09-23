//! Killer-move heuristic.
//!
//! When a quiet move produces a beta cutoff at a given ply, it is promoted to
//! a "killer": at the same ply in later sibling searches it is tried right
//! after the TT move and captures, before history-scored quiet moves. Two
//! killers per ply are kept, most recent first.

use crate::types::RawMove;

/// Number of killers tracked per ply.
pub const KILLERS_PER_PLY: usize = 2;

/// Total plies addressed; `MAX_PLY + 8` leaves headroom so callers do not
/// need to bound-check `ply` on every call (the table clamps internally).
const PLIES: usize = crate::types::MAX_PLY + 8;

/// Two killer moves per ply, `NULL` when unset.
#[derive(Clone, Debug)]
pub struct Killers {
    table: [[RawMove; KILLERS_PER_PLY]; PLIES],
}

impl Default for Killers {
    fn default() -> Self {
        Killers::new()
    }
}

impl Killers {
    pub fn new() -> Killers {
        Killers {
            table: [[RawMove::NULL; KILLERS_PER_PLY]; PLIES],
        }
    }

    /// Clears every killer (called on `ucinewgame`, and between iterations
    /// if a previous search was aborted mid-iteration).
    pub fn reset(&mut self) {
        *self = Killers::new();
    }

    /// The two killers at `ply` as `(most_recent, older)` — either may be
    /// [`RawMove::NULL`] when unset.
    #[inline]
    pub fn get(&self, ply: usize) -> (RawMove, RawMove) {
        let entry = &self.table[ply.min(PLIES - 1)];
        (entry[0], entry[1])
    }

    /// Records a quiet move that caused a beta cutoff at `ply`. Captures and
    /// promotions are already ordered above killers by MVV-LVA, so the
    /// caller should not store them here (a duplicate store is a no-op either
    /// way).
    #[inline]
    pub fn store(&mut self, ply: usize, m: RawMove) {
        let entry = &mut self.table[ply.min(PLIES - 1)];
        if entry[0] == m {
            return; // already the most recent killer
        }
        if entry[1] == m {
            // second killer re-promoted to first, no push-down needed
            entry[1] = entry[0];
            entry[0] = m;
            return;
        }
        entry[1] = entry[0];
        entry[0] = m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Square;

    fn mv(from: Square, to: Square) -> RawMove {
        RawMove::new(from, to, 0)
    }

    #[test]
    fn stores_most_recent_first() {
        let mut k = Killers::new();
        let a = mv(Square::E2, Square::E4);
        let b = mv(Square::D2, Square::D4);
        k.store(3, a);
        assert_eq!(k.get(3), (a, RawMove::NULL));
        k.store(3, b);
        assert_eq!(k.get(3), (b, a));
    }

    #[test]
    fn duplicate_store_is_a_noop() {
        let mut k = Killers::new();
        let a = mv(Square::E2, Square::E4);
        let b = mv(Square::G1, Square::F3);
        k.store(5, a);
        k.store(5, b);
        k.store(5, a); // already first killer -> unchanged
        assert_eq!(k.get(5), (a, b));
        k.store(5, b); // second killer re-promotes to first
        assert_eq!(k.get(5), (b, a));
    }

    #[test]
    fn plies_are_independent() {
        let mut k = Killers::new();
        let a = mv(Square::E2, Square::E4);
        let b = mv(Square::C2, Square::C4);
        k.store(0, a);
        k.store(9, b);
        assert_eq!(k.get(0), (a, RawMove::NULL));
        assert_eq!(k.get(9), (b, RawMove::NULL));
    }

    #[test]
    fn huge_ply_clamps() {
        let mut k = Killers::new();
        let a = mv(Square::E2, Square::E4);
        k.store(usize::MAX, a);
        assert_eq!(k.get(usize::MAX), (a, RawMove::NULL));
    }

    #[test]
    fn reset_clears() {
        let mut k = Killers::new();
        k.store(4, mv(Square::E2, Square::E4));
        k.reset();
        assert_eq!(k.get(4), (RawMove::NULL, RawMove::NULL));
    }
}

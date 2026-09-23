//! Endgame tablebase integration (placeholder).
//!
//! The engine ships without byte tables, so this module provides the trait
//! and the plumbing (a configurable Syzygy path, a no-op probe) that a real
//! tablebase backend can fill in later — the search only ever calls
//! `Tablebase::probe`, never the backend details.

use shakmaty::{Color, Move};

use crate::board::Position;

/// `None` means "no tablebase configured"; `Some(Wdl)` wraps a win/draw/loss
/// with a distance (in plies) to the zeroing move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TablebaseResult {
    Win(Color, u32),
    Draw,
}

/// A tablebase backend: probe a position for its outcome. Placeholder
/// implementation always returns `None` (no tables loaded).
pub trait Tablebase {
    /// Probes a position that has at most the tablebase's piece limit.
    /// Returns `None` when the position is unknown / tables are missing.
    fn probe(&self, pos: &Position) -> Option<TablebaseResult>;

    /// The best move in `legal` according to the probe (placeholder: the
    /// first legal move).
    fn probe_root(&self, pos: &Position, legal: &[Move]) -> Option<Move> {
        let result = self.probe(pos)?;
        // With no tables nothing is reachable; keep the signature honest.
        let _ = result;
        legal.first().copied()
    }
}

/// The no-op tablebase: always probes `None`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoTablebase;

impl Tablebase for NoTablebase {
    #[inline]
    fn probe(&self, _pos: &Position) -> Option<TablebaseResult> {
        None
    }
}

/// Configurable wrapper that can later be switched to a real Syzygy backend
/// without touching the search.
#[derive(Debug, Default, Clone)]
pub struct Endgame {
    /// Path to a Syzygy directory (unused by the placeholder backend).
    pub path: Option<String>,
    pub backend: NoTablebase,
}

impl Endgame {
    pub fn new(path: Option<String>) -> Endgame {
        Endgame {
            path,
            backend: NoTablebase,
        }
    }
}

impl Tablebase for Endgame {
    #[inline]
    fn probe(&self, pos: &Position) -> Option<TablebaseResult> {
        self.backend.probe(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_probe_is_none() {
        let eg = Endgame::new(None);
        assert_eq!(eg.probe(&Position::startpos()), None);
        assert_eq!(NoTablebase.probe(&Position::startpos()), None);
    }
}

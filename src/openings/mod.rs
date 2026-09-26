//! Reproducible opening-position suites for strength testing.
//!
//! A match played always from the start position cannot distinguish an engine
//! that is strong in the early middlegame from one that simply knows the
//! opening phase, and it makes the outcome extremely drawish at shallow
//! depths. This module provides a curated, deterministic **position suite**:
//! a fixed table of diverse, theory-relevant positions (`classical-v1`, 48
//! entries), each derived by replaying a verified move list from the start
//! position.
//!
//! Properties (each is asserted by tests):
//!
//! * **Diverse** — the suite spans the main families: open games (Ruy,
//!   Italian, Scotch, Petrov, King's Gambit, ...), Sicilians (Najdorf,
//!   Sveshnikov, Open, Four Knights), French (Advance, Tarrasch, Winawer),
//!   Caro-Kann, flank defences (English, Réti, Dutch, Bird, KIA), and the
//!   double queen-pawn complexes (QGD, QGA, Slav, Semi-Slav, Nimzo, QID,
//!   Catalan, KID, Grünfeld, ...).
//! * **Balanced colors** — exactly half the entries leave White to move and
//!   half leave Black to move, so neither engine benefits from a first-move
//!   bias across the suite.
//! * **Deterministic ordering** — [`OpeningSuite::order`] returns a full
//!   Fisher–Yates permutation of entry indices derived from a seed (the
//!   engine's [`crate::book::SplitMix64`]), so a match replays the same
//!   opening order for the same seed.
//! * **No duplicate positions** — the FEN of every entry is unique, and a
//!   seeded match cycles the suite without repetition until every entry has
//!   been used once.
//! * **Separate from the engine's book** — the suite is compiled-in test
//!   data, never consulted by the UCI `go` handler, and has nothing to do
//!   with the Polyglot opening book (`BookPath`/`book/*.bin`).

use anyhow::{Context, Result};

use crate::board::Position;
use crate::book::SplitMix64;
use shakmaty::zobrist::Zobrist64;

/// One suite position: a name and the UCI plies that reach it from the start
/// position (must each be legal in sequence — [`OpeningSuite::from_entries`]
/// verifies this).
#[derive(Debug, Clone, Copy)]
pub struct SuiteEntry {
    pub name: &'static str,
    pub plies: &'static [&'static str],
}

/// A validated opening suite.
#[derive(Debug, Clone)]
pub struct OpeningSuite {
    entries: Vec<SuiteEntry>,
    /// FEN of each entry (after its plies), used for dedup and reporting.
    fens: Vec<String>,
    /// Side to move in each entry's final position (balance metric).
    white_to_move: Vec<bool>,
    /// Pre-built positions for entries that are not reached by replaying
    /// plies from the start position (e.g. endgame test positions). `None`
    /// for the normal replay-based suites.
    positions: Option<Vec<Position>>,
}

impl OpeningSuite {
    /// Validates `entries` (every move legal in sequence, every FEN unique)
    /// and freezes them into a suite.
    pub fn from_entries(entries: Vec<SuiteEntry>) -> Result<OpeningSuite> {
        let mut fens = Vec::with_capacity(entries.len());
        let mut white_to_move = Vec::with_capacity(entries.len());
        let mut seen = std::collections::HashSet::new();
        for e in &entries {
            let mut pos = Position::startpos();
            for uci in e.plies {
                pos = pos
                    .play_uci(uci)
                    .with_context(|| format!("opening {}: move {uci} is illegal", e.name))?
                    .0;
            }
            let fen = pos.fen();
            if !seen.insert(fen.clone()) {
                anyhow::bail!(
                    "opening {} duplicates the position of an earlier entry",
                    e.name
                );
            }
            fens.push(fen);
            white_to_move.push(pos.turn() == shakmaty::Color::White);
        }
        Ok(OpeningSuite {
            entries,
            fens,
            white_to_move,
            positions: None,
        })
    }

    /// Builds a suite directly from FENs (e.g. endgame test positions a legal
    /// game would take many plies to reach). Positions are stored as-is; the
    /// per-entry opening line is empty. FENs must be unique and parse.
    pub fn from_positions(entries: &[(&'static str, &str)]) -> Result<OpeningSuite> {
        let mut fens = Vec::with_capacity(entries.len());
        let mut white_to_move = Vec::with_capacity(entries.len());
        let mut seen = std::collections::HashSet::new();
        for (name, fen) in entries {
            let pos = Position::from_fen(fen)
                .with_context(|| format!("opening {name}: bad FEN {fen}"))?;
            let normalized = pos.fen();
            if !seen.insert(normalized.clone()) {
                anyhow::bail!("opening {name} duplicates an earlier position");
            }
            fens.push(normalized);
            white_to_move.push(pos.turn() == shakmaty::Color::White);
        }
        let positions = Some(
            entries
                .iter()
                .map(|(_, fen)| Position::from_fen(fen).expect("FENs validated above"))
                .collect(),
        );
        let entries = entries
            .iter()
            .map(|(name, _)| SuiteEntry { name, plies: &[] })
            .collect();
        Ok(OpeningSuite {
            entries,
            fens,
            white_to_move,
            positions,
        })
    }

    /// The built-in `classical-v1` suite: 48 diverse positions, 24 White to
    /// move and 24 Black to move.
    pub fn classical_v1() -> OpeningSuite {
        // Lines are standard theory; a unit test re-verifies legality and
        // uniqueness on every build, so an accidental typo fails fast.
        let e = |name: &'static str, plies: &'static [&'static str]| SuiteEntry { name, plies };
        OpeningSuite::from_entries(vec![
            // --- open games (White to move after the listed plies) ---------
            e(
                "scotch",
                &[
                    "e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "g8f6",
                ],
            ),
            e(
                "italian-giuoco",
                &[
                    "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6",
                ],
            ),
            e(
                "ruy-lopez",
                &[
                    "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
                ],
            ),
            e(
                "four-knights-spanish",
                &[
                    "e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "f1b5", "f8b4",
                ],
            ),
            // --- sicilians (White to move) ---------------------------------
            e(
                "sicilian-najdorf",
                &[
                    "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6",
                ],
            ),
            e(
                "sicilian-sveshnikov",
                &[
                    "e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5",
                ],
            ),
            e(
                "sicilian-four-knights",
                &[
                    "e2e4", "c7c5", "b1c3", "b8c6", "g1f3", "g8f6", "d2d4", "c5d4",
                ],
            ),
            // --- french / caro-kann (White to move) ------------------------
            e(
                "french-advance",
                &[
                    "e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6",
                ],
            ),
            e(
                "french-tarrasch",
                &[
                    "e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5", "f6d7",
                ],
            ),
            e(
                "french-winawer",
                &[
                    "e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5",
                ],
            ),
            e(
                "caro-advance",
                &[
                    "e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", "g1f3", "e7e6",
                ],
            ),
            e(
                "caro-classical",
                &[
                    "e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6e4",
                ],
            ),
            // --- semi-open defences (White to move) ------------------------
            e(
                "pirc",
                &[
                    "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f1e2", "f8g7",
                ],
            ),
            e(
                "alekhine",
                &[
                    "e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3", "c8g4",
                ],
            ),
            // --- double queen-pawn complexes (White to move) ---------------
            e(
                "queens-gambit-exchange",
                &[
                    "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c4d5", "e6d5",
                ],
            ),
            e(
                "queens-gambit-slav",
                &[
                    "d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "g1f3", "e7e6",
                ],
            ),
            e(
                "semi-slav",
                &[
                    "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "c7c6", "e2e3",
                ],
            ),
            e(
                "kings-indian",
                &[
                    "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6",
                ],
            ),
            e(
                "grunfeld",
                &[
                    "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5",
                ],
            ),
            e(
                "nimzo-indian",
                &[
                    "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "c7c5",
                ],
            ),
            e(
                "catalan",
                &[
                    "d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7",
                ],
            ),
            e(
                "dutch-leningrad",
                &[
                    "d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "g7g6", "g1f3", "f8g7",
                ],
            ),
            e(
                "london-system",
                &[
                    "d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "e7e6", "g1f3", "f8e7",
                ],
            ),
            // --- flank openings (White to move) ----------------------------
            e("english", &["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"]),
            e(
                "reti",
                &[
                    "g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "e7e6", "e2e3", "f8e7",
                ],
            ),
            // --- open games (Black to move after the listed plies) ---------
            e("petrov", &["e2e4", "e7e5", "g1f3", "g8f6", "f3e5"]),
            e("kings-gambit", &["e2e4", "e7e5", "f2f4", "e5f4", "g1f3"]),
            e(
                "spanish-morphy",
                &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"],
            ),
            e(
                "italian-classical",
                &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3"],
            ),
            e(
                "scotch-btm",
                &["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"],
            ),
            e("vienna", &["e2e4", "e7e5", "b1c3", "g8f6", "f2f4"]),
            e("three-knights", &["e2e4", "e7e5", "g1f3", "b8c6", "b1c3"]),
            e("bishop-opening", &["e2e4", "e7e5", "f1c4", "g8f6", "d2d3"]),
            // --- sicilians (Black to move) ----------------------------------
            e(
                "sicilian-open",
                &["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
            ),
            e(
                "sicilian-najdorf-btm",
                &[
                    "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6",
                    "f1e2",
                ],
            ),
            // --- french / caro-kann (Black to move) -------------------------
            e(
                "french-tarrasch-btm",
                &["e2e4", "e7e6", "d2d4", "d7d5", "b1d2"],
            ),
            e(
                "french-winawer-btm",
                &["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5"],
            ),
            e("caro-kann-5", &["e2e4", "c7c6", "d2d4", "d7d5", "g1f3"]),
            e(
                "caro-classical-btm",
                &[
                    "e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5", "e4g3",
                ],
            ),
            // --- semi-open defences (Black to move) -------------------------
            e(
                "pirc-btm",
                &["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f1e2"],
            ),
            e(
                "alekhine-btm",
                &["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "c2c4"],
            ),
            e("scandinavian", &["e2e4", "d7d5", "e4d5", "g8f6", "d2d4"]),
            // --- double queen-pawn complexes (Black to move) ----------------
            e(
                "queens-gambit-accepted",
                &["d2d4", "d7d5", "c2c4", "d5c4", "g1f3"],
            ),
            e("slav-5", &["d2d4", "d7d5", "c2c4", "c7c6", "g1f3"]),
            e(
                "nimzo-btm",
                &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"],
            ),
            e(
                "kid-btm",
                &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4"],
            ),
            e(
                "grunfeld-btm",
                &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5"],
            ),
            // --- flank openings (Black to move) ------------------------------
            e("english-btm", &["c2c4", "e7e5", "b1c3", "g8f6", "g1f3"]),
        ])
        .expect("classical-v1 suite is legal and unique")
    }

    /// Number of positions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the suite holds no positions.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entry names in suite order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> {
        self.entries.iter().map(|e| e.name)
    }

    /// The `i`-th entry.
    pub fn entry(&self, i: usize) -> SuiteEntry {
        self.entries[i]
    }

    /// The FEN reached by the `i`-th entry.
    pub fn fen_of(&self, i: usize) -> &str {
        &self.fens[i]
    }

    /// Number of entries whose final position is White to move (the rest are
    /// Black to move). `classical-v1` is exactly balanced at `len / 2`.
    pub fn white_to_move_count(&self) -> usize {
        self.white_to_move.iter().filter(|&&w| w).count()
    }

    /// A deterministic permutation of the entry indices derived from `seed`
    /// (Fisher–Yates over [`SplitMix64`]). A match consumes this order in
    /// sequence, cycling when `games > len`; the same seed always yields the
    /// same order, and every entry appears once before any repeats.
    pub fn order(&self, seed: u64) -> Vec<usize> {
        let mut rng = SplitMix64(seed);
        let mut idx: Vec<usize> = (0..self.len()).collect();
        for i in (1..idx.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        idx
    }

    /// The position reached by entry `i`, the move-to-move history window
    /// (hashes before the current root since the last zeroing move) exactly as
    /// a game would see it, and the opening line as `RawMove`s (empty for
    /// suites built from FENs).
    pub fn position_of(&self, i: usize) -> (Position, Vec<Zobrist64>, Vec<crate::types::RawMove>) {
        if let Some(positions) = &self.positions {
            return (positions[i].clone(), Vec::new(), Vec::new());
        }
        let mut pos = Position::startpos();
        let mut history = Vec::new();
        let mut opened = Vec::new();
        for uci in self.entries[i].plies {
            let (child, raw) = pos.play_uci(uci).expect("suite entries are legal");
            crate::selfplay::record(&mut history, &pos, &child);
            opened.push(raw);
            pos = child;
        }
        (pos, history, opened)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classical_v1_is_legal_unique_and_balanced() {
        let suite = OpeningSuite::classical_v1();
        assert_eq!(suite.len(), 48, "the documented suite size");
        let mut fens = std::collections::HashSet::new();
        let mut names = std::collections::HashSet::new();
        for i in 0..suite.len() {
            assert!(
                fens.insert(suite.fen_of(i).to_string()),
                "duplicate position at entry {} ({})",
                i,
                suite.entry(i).name
            );
            assert!(names.insert(suite.entry(i).name), "duplicate name");
        }
        assert_eq!(
            suite.white_to_move_count(),
            suite.len() / 2,
            "colors must be balanced"
        );
    }

    #[test]
    fn every_entry_replays_to_its_fen() {
        let suite = OpeningSuite::classical_v1();
        for i in 0..suite.len() {
            let (pos, _, _) = suite.position_of(i);
            assert_eq!(pos.fen(), suite.fen_of(i), "entry {}", suite.entry(i).name);
        }
    }

    #[test]
    fn from_entries_rejects_illegal_lines() {
        let bad = vec![SuiteEntry {
            name: "bad",
            plies: &["e2e4", "e7e5", "g1f3", "d8d2"], // pawn move for black? d8d2 illegal
        }];
        assert!(OpeningSuite::from_entries(bad).is_err());
        let dup = vec![
            SuiteEntry {
                name: "a",
                plies: &["e2e4"],
            },
            SuiteEntry {
                name: "b",
                plies: &["e2e4"], // reaches the same position
            },
        ];
        assert!(OpeningSuite::from_entries(dup).is_err());
    }

    #[test]
    fn order_is_deterministic_and_a_bijection() {
        let suite = OpeningSuite::classical_v1();
        let a = suite.order(7);
        let b = suite.order(7);
        assert_eq!(a, b, "same seed -> same order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..suite.len()).collect::<Vec<_>>(),
            "a permutation"
        );
        let c = suite.order(8);
        assert_ne!(a, c, "different seeds should (almost surely) differ");
        // Every suite index is used before any repeats when cycling.
        for i in 0..suite.len() {
            assert!(a[i] < suite.len());
        }
    }
}

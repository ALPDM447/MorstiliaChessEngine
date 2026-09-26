//! Morstilia — a strong classical (pre-NNUE) UCI chess engine built on shakmaty.
//!
//! Architecture (module tree):
//!
//! * `board` — `Position`: `shakmaty::Chess` + incremental Zobrist hash,
//!   perft, FEN.
//! * `types` — compact `RawMove`, fixed-capacity `MoveList`, depths/scores.
//! * `move_ordering` — TT move, MVV-LVA, killers, history/countermove/
//!   continuation, SEE.
//! * `tt` — lock-free transposition table.
//! * `evaluation` — tapered classical evaluation (material, PST, pawns,
//!   mobility, king safety, threats, passed pawns, phase).
//! * `nnue` — real Stockfish 19 NNUE inference: `.nnue` parsing, the three
//!   feature sets, the incrementally maintained accumulator and the forward
//!   pass, behind the same evaluation interface the classical evaluator uses.
//! * `book` — Polyglot opening book.
//! * `search` — iterative deepening, alpha-beta/PVS, quiescence, pruning,
//!   reductions, time management.
//! * `threading`, `endgame`, `uci`, `config` — multi-thread infra, Syzygy
//!   scaffolding, the UCI protocol loop and engine options.
//! * `rating` — W/D/L → Elo + confidence interval and the trinomial SPRT.
//! * `regression` — tactic/determinism/eval-delta protection for tuned sets.
//! * `selfplay` — deterministic seeded engine self-play with PGN records.
//! * `openings` — reproducible opening-position suites for matches.
//! * `matchplay` — engine-vs-engine matches: time controls, color
//!   alternation, parallel games, SPRT auto-stop, JSON match reports.
//! * `tuning` — SPSA evaluation tuning over `config/baseline_eval.toml`.

pub mod board;
pub mod book;
pub mod config;
pub mod endgame;
pub mod evaluation;
pub mod matchplay;
pub mod move_ordering;
pub mod nnue;
pub mod openings;
pub mod rating;
pub mod regression;
pub mod search;
pub mod selfplay;
pub mod threading;
pub mod tt;
pub mod tuning;
pub mod types;
pub mod uci;

#[cfg(test)]
pub mod testutil;

/// Re-exported for ergonomics in binaries and tests.
pub use board::Position;
pub use types::{MoveList, RawMove};

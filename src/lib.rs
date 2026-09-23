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
//! * `book` — Polyglot opening book.
//! * `search` — iterative deepening, alpha-beta/PVS, quiescence, pruning,
//!   reductions, time management.
//! * `threading`, `endgame`, `uci`, `config` — multi-thread infra, Syzygy
//!   scaffolding, the UCI protocol loop and engine options.

pub mod board;
pub mod book;
pub mod config;
pub mod endgame;
pub mod evaluation;
pub mod move_ordering;
pub mod search;
pub mod threading;
pub mod tt;
pub mod types;
pub mod uci;

#[cfg(test)]
pub mod testutil;

/// Re-exported for ergonomics in binaries and tests.
pub use board::Position;
pub use types::{MoveList, RawMove};

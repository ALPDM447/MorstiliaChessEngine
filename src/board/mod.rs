//! Board module: the engine's wrapping of `shakmaty::Chess`.
//!
//! The engine deliberately does *not* implement its own chess logic —
//! `shakmaty` is the single source of truth for board state, FENs and legal
//! move generation. This module adds the pieces shakmaty does not provide for
//! a search engine: an incrementally maintained Zobrist hash, compact move
//! lists, perft, and the Polyglot opening-book key.

pub mod state;

pub use state::Position;

/// Re-exported for convenience so callers do not need both imports.
pub use shakmaty::{Board, CastlingMode, CastlingSide, Color, Role, Square};

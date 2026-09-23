//! Shared test-only helpers (compiled only under `#[cfg(test)]`).
//!
//! shakmaty does not implement `FromStr for Chess`, so unit tests across the
//! crate need a common way to build positions from FEN strings.

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, EnPassantMode};

/// Parses a FEN string into a `shakmaty::Chess` (standard castling rights).
pub fn chess(fen_str: &str) -> Chess {
    let fen: Fen = fen_str.parse().expect("invalid FEN in test");
    fen.into_position(CastlingMode::Standard)
        .unwrap_or_else(|e| panic!("illegal FEN {fen_str:?} in test: {e}"))
}

/// The FEN string of the standard starting position.
pub fn startpos_fen() -> String {
    Fen::from_position(&Chess::default(), EnPassantMode::Legal).to_string()
}

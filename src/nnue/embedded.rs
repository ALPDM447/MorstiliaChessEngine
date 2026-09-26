//! Embedded default NNUE network (Stockfish 19's `nn-1a298aa575a0.nnue`).
//!
//! This module embeds the default network at compile time using `include_bytes!`,
//! so the engine can run NNUE without requiring an external `.nnue` file.

use std::io::Cursor;

use crate::nnue::format::Reader;
use crate::nnue::network::{Network, load_from};

/// The embedded Stockfish 19 default network (`nn-1a298aa575a0.nnue`).
///
/// This is ~98 MB and will increase the executable size accordingly.
const EMBEDDED_NETWORK_BYTES: &[u8] = include_bytes!("../../nnue/nn-1a298aa575a0.nnue");

/// Loads the embedded default network.
///
/// This uses the same validation (header hash, section hashes, EOF check) as
/// [`crate::nnue::network::load_network`], but reads from the compile-time
/// embedded bytes instead of a file.
///
/// Returns `None` if the embedded network fails validation (should never happen
/// unless the embedded bytes are corrupted at compile time).
pub fn load_embedded_network() -> Option<Network> {
    let cursor = Cursor::new(EMBEDDED_NETWORK_BYTES);
    let mut reader = Reader::new(cursor);
    load_from(&mut reader).ok()
}

/// Returns the size of the embedded network in bytes.
pub fn embedded_network_size() -> usize {
    EMBEDDED_NETWORK_BYTES.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::NETWORK_HASH;

    #[test]
    fn embedded_network_loads_successfully() {
        let net = load_embedded_network().expect("embedded network must load");
        assert_eq!(net.hash(), NETWORK_HASH);
    }

    #[test]
    fn embedded_network_has_expected_size() {
        // The known size of nn-1a298aa575a0.nnue
        assert_eq!(embedded_network_size(), 98_511_183);
    }
}

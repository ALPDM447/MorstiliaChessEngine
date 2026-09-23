//! UCI (Universal Chess Interface) support.
//!
//! [`parser`] turns raw stdin lines into typed [`Command`]s; [`protocol`]
//! hosts the engine driver that reacts to them. The (later) `main.rs` loop
//! is just: read a line → parse → hand to the engine.

pub mod parser;
pub mod protocol;

pub use parser::{Command, GoParams, parse};
pub use protocol::UciEngine;

//! Deterministic engine-vs-engine match driver (Stages 5 + 8).
//!
//! Thin wrapper around [`morstilia::matchplay::cli`] (shared with the
//! `morstilia-selfplay` alias): seeded reproducible matches, Elo + CI + SPRT
//! statistics, JSON match reports and baseline freezing. See `--help` for the
//! full option list.

fn main() {
    morstilia::matchplay::cli::main();
}

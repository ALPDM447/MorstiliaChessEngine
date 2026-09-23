//! Engine options and configuration.
//!
//! UCI `setoption` values are mirrored here (in-memory `EngineConfig`); a
//! `config.toml` next to the executable can seed defaults. Values are plain
//! and serializable so a GUI-exposed config stays transparent.

use serde::{Deserialize, Serialize};

/// Default transposition table size in megabytes.
pub const DEFAULT_HASH_MB: usize = 64;

/// Default number of search threads.
pub const DEFAULT_THREADS: usize = 1;

/// Default search depth limit (0 = unlimited).
pub const DEFAULT_SEARCH_DEPTH: i32 = 0;

/// Engine option defaults and state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// TT size in megabytes.
    pub hash_mb: usize,
    /// Number of search threads used on `go`.
    pub threads: usize,
    /// If > 0, an absolute depth ceiling regardless of time budget.
    pub search_depth: i32,
    /// Whether the opening book is consulted on `go`.
    pub book_enabled: bool,
    /// Path to a Polyglot book; empty = auto-detect next to the executable.
    pub book_path: String,
    /// Path to a Syzygy directory; empty = not configured.
    pub syzygy_path: String,
    /// Multi-PV count (how many root lines to report).
    pub multi_pv: usize,
    /// Ponder is accepted by the protocol but the engine does not search it.
    pub ponder: bool,
    /// Print `info string` diagnostics to stderr instead of stdout.
    pub debug: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            hash_mb: DEFAULT_HASH_MB,
            threads: DEFAULT_THREADS,
            search_depth: DEFAULT_SEARCH_DEPTH,
            book_enabled: true,
            book_path: String::new(),
            syzygy_path: String::new(),
            multi_pv: 1,
            ponder: false,
            debug: false,
        }
    }
}

impl EngineConfig {
    /// Named UCI option → value. Unknown options are ignored (per UCI) and
    /// return `false`.
    pub fn set_option(&mut self, name: &str, value: Option<&str>) -> bool {
        match name {
            "Hash" => {
                if let Some(v) = value.and_then(|s| s.parse::<usize>().ok()) {
                    self.hash_mb = v.max(1);
                    return true;
                }
                false
            }
            "Threads" => {
                if let Some(v) = value.and_then(|s| s.parse::<usize>().ok()) {
                    self.threads = v.clamp(1, 1024);
                    return true;
                }
                false
            }
            "SearchDepth" => {
                if let Some(v) = value.and_then(|s| s.parse::<i32>().ok()) {
                    self.search_depth = v.max(0);
                    return true;
                }
                false
            }
            "BookEnabled" => {
                if let Some(v) = value {
                    self.book_enabled = matches!(v, "true" | "True" | "1" | "yes");
                    return true;
                }
                false
            }
            "BookPath" => {
                if let Some(v) = value {
                    self.book_path = v.to_string();
                    return true;
                }
                false
            }
            "SyzygyPath" => {
                if let Some(v) = value {
                    self.syzygy_path = v.to_string();
                    return true;
                }
                false
            }
            "MultiPV" => {
                if let Some(v) = value.and_then(|s| s.parse::<usize>().ok()) {
                    self.multi_pv = v.max(1);
                    return true;
                }
                false
            }
            "Debug" => {
                if let Some(v) = value {
                    self.debug = matches!(v, "true" | "True" | "1" | "yes");
                    return true;
                }
                false
            }
            "Ponder" => {
                if let Some(v) = value {
                    self.ponder = matches!(v, "true" | "True" | "1" | "yes");
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    /// A `setoption`-style description of every option (for `uci` output).
    pub fn option_log(&self) -> Vec<String> {
        vec![
            format!(
                "option name Hash type spin default {} min 1 max 1048576",
                self.hash_mb
            ),
            format!(
                "option name Threads type spin default {} min 1 max 1024",
                self.threads
            ),
            format!(
                "option name SearchDepth type spin default {} min 0 max 64",
                self.search_depth
            ),
            format!(
                "option name BookEnabled type check default {}",
                self.book_enabled
            ),
            format!(
                "option name BookPath type string default {}",
                self.book_path
            ),
            format!(
                "option name SyzygyPath type string default {}",
                self.syzygy_path
            ),
            format!(
                "option name MultiPV type spin default {} min 1 max 4",
                self.multi_pv
            ),
            format!("option name Debug type check default {}", self.debug),
            format!("option name Ponder type check default {}", self.ponder),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = EngineConfig::default();
        assert_eq!(c.hash_mb, 64);
        assert_eq!(c.threads, 1);
        assert!(c.book_enabled);
    }

    #[test]
    fn set_option_parses_values() {
        let mut c = EngineConfig::default();
        assert!(c.set_option("Hash", Some("128")));
        assert_eq!(c.hash_mb, 128);
        assert!(c.set_option("Threads", Some("4")));
        assert_eq!(c.threads, 4);
        assert!(c.set_option("SearchDepth", Some("12")));
        assert_eq!(c.search_depth, 12);
        assert!(!c.set_option("NoSuchOption", Some("x")));
        // Hash=0 floors to 1.
        assert!(c.set_option("Hash", Some("0")));
        assert_eq!(c.hash_mb, 1);
    }

    #[test]
    fn option_log_is_uci_shaped() {
        let c = EngineConfig::default();
        let log = c.option_log();
        assert!(log.iter().all(|l| l.starts_with("option name ")));
        assert!(log.iter().any(|l| l.contains("type spin")));
        assert!(log.iter().any(|l| l.contains("type check")));
    }
}

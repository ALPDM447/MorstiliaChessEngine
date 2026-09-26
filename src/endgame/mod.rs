//! Syzygy endgame tablebase integration.
//!
//! The engine probes WDL + DTZ through `shakmaty_syzygy` (the official
//! shakmaty-ecosystem backend), wrapped in [`Syzygy`]:
//!
//! * *configurable path* — [`Syzygy::load`] reads a directory of `.rtbw` /
//!   `.rtbz` tables (auto-detecting whatever the directory holds, like
//!   Stockfish) or a single table file. A missing or unreadable path is
//!   never fatal: it produces [`LoadReport`] warnings and an inert
//!   [`Syzygy`] that answers `None` to every probe;
//! * *piece-count gate* — probes only fire for positions with at most
//!   [`Syzygy::max_pieces`] pieces, so the midgame hot path never touches
//!   the tablebase at all;
//! * *50-move / DTZ semantics* — every probe is resolved together with the
//!   position's halfmove counter (`probe_wdl` returns an `AmbiguousWdl`
//!   that was folded with `halfmoves`), so "cursed" wins and "blessed"
//!   losses are detected exactly, and [`Syzygy::probe_dtz`] gives the DTZ
//!   used by root move selection;
//! * *graceful degradation* — unknown material, positions with castling
//!   rights, too many pieces, or a WDL-only table set (no `.rtbz`) all
//!   degrade to "no tablebase information" instead of an error; the WDL-only
//!   fallback assumes the position was reached right after a zeroing move
//!   (the library's `probe_wdl_after_zeroing`), i.e. it is blind to the
//!   halfmove counter;
//! * *thread safety* — the underlying `Tablebase<Chess>` is `Send` but not
//!   `Sync` (it owns lookup maps with a lazy `OnceCell` per material), so it
//!   lives behind a `Mutex`; probing is gated on the piece count before the
//!   lock is ever taken, keeping it off the normal search hot path.
//!
//! Score semantics: a tablebase outcome is *absolute* and its magnitude is
//! what matters. The engine maps them into a dedicated band between any real
//! evaluation and the mate zone ([`TB_WIN`], [`TB_CURSED`]), so a tablebase
//! win is never reported as a checkmate and vice versa.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use shakmaty::Chess;
use shakmaty::Position as _; // `Chess::board()` (piece-count gate) comes from this trait.
use shakmaty_syzygy::{AmbiguousWdl, Tablebase, Wdl};

use crate::types::RawMove;

/// Score of an *unconditional* win, from the side to move. Puts every honest
/// tablebase win above any static evaluation (~±5000) while staying far
/// below the mate zone (`is_mate` starts at ±24000): a TB win is never
/// confused with a mate score.
pub const TB_WIN: i32 = 20_000;

/// Magnitude of a *cursed* win / *blessed* loss — a decisive result the
/// 50-move rule can still frustrate. Strictly between a draw and an honest
/// win so the search always prefers the unconditional result.
pub const TB_CURSED: i32 = 10_000;

/// The resolved outcome of a probed position, from the side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbOutcome {
    /// Unconditional win.
    Win,
    /// Win that can be frustrated by the 50-move rule (or the ambiguous
    /// edge case, resolved conservatively).
    CursedWin,
    /// Unconditional draw.
    Draw,
    /// Loss that can be saved by the 50-move rule.
    BlessedLoss,
    /// Unconditional loss.
    Loss,
}

impl TbOutcome {
    /// The search score of this outcome (see [`wdl_score`]).
    #[inline]
    pub fn score(self) -> i32 {
        wdl_score(self)
    }
}

/// Maps a resolved WDL outcome to the engine's score band. The bands are
/// negation-symmetric ([`wdl_score(Win)`] == `-wdl_score(Loss)`), so the
/// alpha-beta negamax never drifts off the band: a won leaf and the negation
/// of the *opponent's* lost leaf produce the same score for the winner.
#[inline]
pub const fn wdl_score(wdl: TbOutcome) -> i32 {
    match wdl {
        TbOutcome::Win => TB_WIN - 1,
        TbOutcome::CursedWin => TB_CURSED,
        TbOutcome::Draw => 0,
        TbOutcome::BlessedLoss => -TB_CURSED,
        TbOutcome::Loss => -(TB_WIN - 1),
    }
}

/// What [`Syzygy::load`] found (or failed to find) for a `SyzygyPath`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadReport {
    /// The configured path (as given).
    pub path: String,
    /// Number of table files successfully registered.
    pub files: usize,
    /// Largest piece count the loaded set covers (`0` = no tables).
    pub max_pieces: usize,
    /// Non-fatal problems (missing path, corrupt files, ...).
    pub warnings: Vec<String>,
}

/// A thread-safe handle to the loaded Syzygy tables plus the probe
/// instrumentation. Sharing one instance across the Lazy-SMP workers is
/// intentional: probes are pure, the counters are lock-free atomics, and the
/// table collection itself is guarded by the piece-count gate.
pub struct Syzygy {
    inner: Option<Arc<Mutex<Tablebase<Chess>>>>,
    max_pieces: usize,
    loaded_files: usize,
    probes: AtomicU64,
    hits: AtomicU64,
    wins: AtomicU64,
    draws: AtomicU64,
    losses: AtomicU64,
    cursed: AtomicU64,
}

impl std::fmt::Debug for Syzygy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Syzygy")
            .field("loaded", &self.is_loaded())
            .field("max_pieces", &self.max_pieces)
            .field("loaded_files", &self.loaded_files)
            .finish()
    }
}

impl Default for Syzygy {
    fn default() -> Self {
        Syzygy::none()
    }
}

impl Syzygy {
    /// An inert tablebase: no tables, every probe answers `None` and the
    /// piece-count gate keeps the search off it entirely.
    pub fn none() -> Syzygy {
        Syzygy {
            inner: None,
            max_pieces: 0,
            loaded_files: 0,
            probes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            wins: AtomicU64::new(0),
            draws: AtomicU64::new(0),
            losses: AtomicU64::new(0),
            cursed: AtomicU64::new(0),
        }
    }

    /// Loads every Syzygy table under `path` (`SyzygyPath`). `path` may name
    /// a directory (auto-detected `.rtbw`/`.rtbz` files, the normal case) or
    /// one table file. An empty, missing or unreadable path yields `none()`
    /// plus warnings; the engine then probes nothing until a valid path is
    /// configured.
    pub fn load(path: &str) -> (Syzygy, LoadReport) {
        let mut report = LoadReport {
            path: path.to_string(),
            files: 0,
            max_pieces: 0,
            warnings: Vec::new(),
        };
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return (Syzygy::none(), report);
        }
        let p = Path::new(trimmed);
        let mut tb = Tablebase::<Chess>::new();
        let files = if p.is_dir() {
            match tb.add_directory(p) {
                Ok(n) => n,
                Err(e) => {
                    report.warnings.push(format!(
                        "cannot read tablebase directory {}: {e}",
                        p.display()
                    ));
                    return (Syzygy::none(), report);
                }
            }
        } else if p.is_file() {
            match tb.add_file(p) {
                Ok(()) => 1,
                Err(e) => {
                    report
                        .warnings
                        .push(format!("cannot add tablebase file {}: {e}", p.display()));
                    return (Syzygy::none(), report);
                }
            }
        } else {
            report
                .warnings
                .push(format!("tablebase path not found: {}", p.display()));
            return (Syzygy::none(), report);
        };
        report.files = files;
        report.max_pieces = tb.max_pieces();
        let syzygy = Syzygy {
            inner: if files > 0 {
                Some(Arc::new(Mutex::new(tb)))
            } else {
                None
            },
            max_pieces: report.max_pieces,
            loaded_files: files,
            ..Syzygy::none()
        };
        (syzygy, report)
    }

    /// Whether at least one table file was registered.
    #[inline]
    pub fn is_loaded(&self) -> bool {
        self.inner.is_some()
    }

    /// Largest piece count the loaded set covers (`0` = no tables). This is
    /// the auto-detection result of the load.
    #[inline]
    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    /// Number of table files registered.
    #[inline]
    pub fn loaded_files(&self) -> usize {
        self.loaded_files
    }

    /// Total probe attempts that reached the tables (in range, tables
    /// present). Diagnostic only.
    #[inline]
    pub fn probes(&self) -> u64 {
        self.probes.load(Ordering::Relaxed)
    }

    /// Successful probes (an outcome was produced — counting the Dtz and
    /// best-move helpers too). Diagnostic only.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Probes resolved to an unconditional win / draw / loss / cursed result
    /// (WDL probes only). Diagnostic only.
    #[inline]
    pub fn wins(&self) -> u64 {
        self.wins.load(Ordering::Relaxed)
    }
    #[inline]
    pub fn draws(&self) -> u64 {
        self.draws.load(Ordering::Relaxed)
    }
    #[inline]
    pub fn losses(&self) -> u64 {
        self.losses.load(Ordering::Relaxed)
    }
    #[inline]
    pub fn cursed(&self) -> u64 {
        self.cursed.load(Ordering::Relaxed)
    }

    /// WDL probe of `chess` from the side to move, resolved with the
    /// position's own halfmove counter. `None` = no tablebase information
    /// (tables missing for this material, position with castling rights, too
    /// many pieces, or no tables loaded at all).
    ///
    /// When the DTZ tables for the material are missing, falls back to the
    /// WDL-only probe (`probe_wdl_after_zeroing`), which assumes the position
    /// was reached right after a zeroing move — i.e. it is blind to the
    /// halfmove counter (documented limitation of WDL-only deployments).
    #[inline]
    pub fn probe_wdl(&self, chess: &Chess) -> Option<TbOutcome> {
        if !self.in_range(chess) {
            return None;
        }
        self.probes.fetch_add(1, Ordering::Relaxed);
        let result = match &self.inner {
            Some(tb) => match tb.lock() {
                Ok(tb) => match tb.probe_wdl(chess) {
                    Ok(wdl) => Some(resolve_ambiguous(wdl)),
                    Err(_) => tb.probe_wdl_after_zeroing(chess).ok().map(resolve_wdl),
                },
                Err(_) => None, // poisoned by a panicking worker: treat as a miss
            },
            None => None,
        };
        self.count(result);
        result
    }

    /// DTZ probe of `chess` from the side to move: positive = winning for
    /// the side to move, negative = losing, `0` = drawn (in plies to the
    /// next zeroing move). `None` = no information (needs both `.rtbw` and
    /// `.rtbz` for the material). Used by root move selection and by the
    /// halfmove-sensitive diagnostics.
    #[inline]
    pub fn probe_dtz(&self, chess: &Chess) -> Option<i32> {
        if !self.in_range(chess) {
            return None;
        }
        self.probes.fetch_add(1, Ordering::Relaxed);
        let result = match &self.inner {
            Some(tb) => match tb.lock() {
                Ok(tb) => match tb.probe_dtz(chess) {
                    Ok(dtz) => Some(dtz.ignore_rounding().0),
                    Err(_) => None,
                },
                Err(_) => None,
            },
            None => None,
        };
        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// The DTZ-optimal move from `chess`, when the root position itself is
    /// covered by the tables: the fastest win / longest resistance under the
    /// 50-move rule (the library's `best_move`). Returns the move plus its
    /// signed DTZ. `None` = no decisive move known (draw or no tables).
    #[inline]
    pub fn root_best_move(&self, chess: &Chess) -> Option<(RawMove, i32)> {
        if !self.in_range(chess) {
            return None;
        }
        self.probes.fetch_add(1, Ordering::Relaxed);
        let result = match &self.inner {
            Some(tb) => match tb.lock() {
                Ok(tb) => match tb.best_move(chess) {
                    Ok(Some((m, dtz))) => {
                        Some((RawMove::from_shakmaty(m), dtz.ignore_rounding().0))
                    }
                    Ok(None) => None,
                    Err(_) => None,
                },
                Err(_) => None,
            },
            None => None,
        };
        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Piece-count gate: only positions with at most [`Syzygy::max_pieces`]
    /// (and at least the two kings) can possibly be covered.
    #[inline]
    fn in_range(&self, chess: &Chess) -> bool {
        if self.inner.is_none() || self.max_pieces == 0 {
            return false;
        }
        chess.board().occupied().count() <= self.max_pieces
    }

    fn count(&self, outcome: Option<TbOutcome>) {
        if let Some(wdl) = outcome {
            self.hits.fetch_add(1, Ordering::Relaxed);
            match wdl {
                TbOutcome::Win => {
                    self.wins.fetch_add(1, Ordering::Relaxed);
                }
                TbOutcome::Draw => {
                    self.draws.fetch_add(1, Ordering::Relaxed);
                }
                TbOutcome::Loss => {
                    self.losses.fetch_add(1, Ordering::Relaxed);
                }
                TbOutcome::CursedWin | TbOutcome::BlessedLoss => {
                    self.cursed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Conservative resolution of the 50-move ambiguity: the "maybe" edge values
/// are never read as unconditional results, so the engine cannot *overstate*
/// a win (a maybe-win scores like the cursed win it may actually be).
fn resolve_ambiguous(wdl: AmbiguousWdl) -> TbOutcome {
    match wdl {
        AmbiguousWdl::Win => TbOutcome::Win,
        AmbiguousWdl::MaybeWin | AmbiguousWdl::CursedWin => TbOutcome::CursedWin,
        AmbiguousWdl::Draw => TbOutcome::Draw,
        AmbiguousWdl::BlessedLoss | AmbiguousWdl::MaybeLoss => TbOutcome::BlessedLoss,
        AmbiguousWdl::Loss => TbOutcome::Loss,
    }
}

/// The unambiguous, halfmove-blind WDL family (WDL-only fallback).
fn resolve_wdl(wdl: Wdl) -> TbOutcome {
    match wdl {
        Wdl::Loss => TbOutcome::Loss,
        Wdl::BlessedLoss => TbOutcome::BlessedLoss,
        Wdl::Draw => TbOutcome::Draw,
        Wdl::CursedWin => TbOutcome::CursedWin,
        Wdl::Win => TbOutcome::Win,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use std::path::PathBuf;

    fn tb_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/syzygy")
    }

    fn tb() -> Syzygy {
        let (tb, report) = Syzygy::load(tb_dir().to_str().unwrap());
        assert_eq!(report.max_pieces, 4, "3+4 piece set expected: {report:?}");
        tb
    }

    /// Compile-time proof that one `Syzygy` can be shared across the
    /// Lazy-SMP workers (a `&Syzygy` used from every worker thread).
    fn assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn syzygy_is_shareable_across_workers() {
        assert_send_sync::<Syzygy>();
        assert_send_sync::<crate::search::SearchShared>();
    }

    #[test]
    fn load_detects_the_available_tables() {
        let (tb, report) = Syzygy::load(tb_dir().to_str().unwrap());
        assert!(tb.is_loaded());
        assert_eq!(report.files, 20, "10 materials × rtbw+rtbz -> 20 files");
        assert_eq!(tb.loaded_files(), report.files);
        assert_eq!(tb.max_pieces(), 4, "only 3 and 4 piece tables present");
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn none_and_empty_paths_are_inert() {
        for path in ["", "   "] {
            let (tb, report) = Syzygy::load(path);
            assert!(!tb.is_loaded());
            assert_eq!(tb.max_pieces(), 0);
            assert_eq!(report.files, 0);
        }
        let none = Syzygy::none();
        assert!(!none.is_loaded());
        assert_eq!(none.max_pieces(), 0);
        assert_eq!(none.probe_wdl(&Position::startpos().chess), None);
        assert_eq!(none.probe_dtz(&Position::startpos().chess), None);
        assert_eq!(none.root_best_move(&Position::startpos().chess), None);
        assert_eq!(none.probes(), 0);
    }

    #[test]
    fn missing_path_is_graceful_and_warns() {
        let (tb, report) = Syzygy::load("/definitely/not/a/real/tablebase/path");
        assert!(!tb.is_loaded());
        assert_eq!(tb.max_pieces(), 0);
        assert_eq!(report.files, 0);
        assert!(!report.warnings.is_empty(), "missing path must warn");
        let start = Position::startpos();
        assert_eq!(tb.probe_wdl(&start.chess), None);
        assert_eq!(tb.probe_dtz(&start.chess), None);
        assert_eq!(
            tb.probe_wdl(&start.chess),
            None,
            "no probes counted unless in range"
        );
    }

    #[test]
    fn single_table_file_can_be_loaded_directly() {
        let path = tb_dir().join("KQvK.rtbw");
        let (tb, report) = Syzygy::load(path.to_str().unwrap());
        assert!(tb.is_loaded());
        assert_eq!(report.files, 1);
        assert_eq!(tb.max_pieces(), 3);
    }

    #[test]
    fn corrupt_payload_probes_as_a_miss() {
        // Write a table-sized file with the right size but garbage content
        // (the loader checks the header magic lazily, at probe/open time, so
        // this must degrade to a probe miss — never a panic).
        let dir = std::env::temp_dir().join("morstilia-tb-corrupt-test");
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = vec![0u8; 16 + 64];
        bytes[0..4].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        let file = dir.join("KQvK.rtbw");
        std::fs::write(&file, &bytes).unwrap();
        let (tb, report) = Syzygy::load(dir.to_str().unwrap());
        assert_eq!(
            report.files, 1,
            "size is a multiple of 64+16, loader accepts"
        );
        assert_eq!(tb.max_pieces(), 3);
        // The header magic is wrong, so probing this material is a miss —
        // never a panic.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        assert_eq!(tb.probe_wdl(&pos.chess), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- WDL outcomes ------------------------------------------------------

    #[test]
    fn known_win_queen_endgame() {
        let tb = tb();
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        assert_eq!(tb.probe_wdl(&pos.chess), Some(TbOutcome::Win));
    }

    #[test]
    fn known_win_rook_and_double_pawn() {
        let tb = tb();
        let kr = Position::from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 0 1").unwrap();
        assert_eq!(tb.probe_wdl(&kr.chess), Some(TbOutcome::Win));
        let pp = Position::from_fen("4k3/8/8/8/8/4P3/4P3/4K3 w - - 0 1").unwrap();
        assert_eq!(tb.probe_wdl(&pp.chess), Some(TbOutcome::Win));
        let bn = Position::from_fen("4k3/8/8/8/8/8/8/4KB1N w - - 0 1").unwrap();
        assert_eq!(
            tb.probe_wdl(&bn.chess),
            Some(TbOutcome::Win),
            "KBNvK is a forced mate"
        );
    }

    #[test]
    fn known_draws() {
        let tb = tb();
        let bk = Position::from_fen("4k3/8/8/8/8/8/8/4B1K1 w - - 0 1").unwrap();
        assert_eq!(
            tb.probe_wdl(&bk.chess),
            Some(TbOutcome::Draw),
            "KBvK is drawn"
        );
        let nk = Position::from_fen("4k3/8/8/8/8/8/8/4N1K1 w - - 0 1").unwrap();
        assert_eq!(
            tb.probe_wdl(&nk.chess),
            Some(TbOutcome::Draw),
            "KNvK is drawn"
        );
        let rr = Position::from_fen("4k2r/8/8/8/8/8/8/3RK3 w - - 0 1").unwrap();
        assert_eq!(
            tb.probe_wdl(&rr.chess),
            Some(TbOutcome::Draw),
            "KRvKR is drawn"
        );
        let wrong_pawn = Position::from_fen("7k/8/8/8/8/8/7P/7K w - - 0 1").unwrap();
        assert_eq!(
            tb.probe_wdl(&wrong_pawn.chess),
            Some(TbOutcome::Draw),
            "wrong rook pawn"
        );
    }

    #[test]
    fn known_loss_for_the_side_to_move() {
        let tb = tb();
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1").unwrap();
        assert_eq!(tb.probe_wdl(&pos.chess), Some(TbOutcome::Loss));
    }

    #[test]
    fn winning_pawn_endgame_with_king_support() {
        let tb = tb();
        // King on the 6th rank in front of the pawn (white Ke6, Pe5 vs Ke8):
        // a textbook KPK win.
        let pos = Position::from_fen("4k3/8/4K3/4P3/8/8/8/8 w - - 0 1").unwrap();
        assert_eq!(tb.probe_wdl(&pos.chess), Some(TbOutcome::Win));
    }

    // --- DTZ ---------------------------------------------------------------

    #[test]
    fn dtz_signs_follow_the_side_to_move() {
        let tb = tb();
        let white = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let d = tb.probe_dtz(&white.chess).unwrap();
        assert!(d > 0, "winning side to move -> positive dtz, got {d}");
        let black = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1").unwrap();
        let d = tb.probe_dtz(&black.chess).unwrap();
        assert!(d < 0, "losing side to move -> negative dtz, got {d}");
    }

    #[test]
    fn dtz_is_bounded_and_positive_for_krkv() {
        let tb = tb();
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 0 1").unwrap();
        let d = tb.probe_dtz(&pos.chess).unwrap();
        assert!(
            (1..=100).contains(&d),
            "KRvK win converts within the 50-move window: got {d}"
        );
    }

    // --- 50-move / halfmove semantics --------------------------------------

    #[test]
    fn halfmove_counter_shapes_the_resolution() {
        let tb = tb();
        // Same board, two halfmove counters. KRvK needs >= 2 plies before the
        // next zeroing move for *this* pair, so 99+dtz crosses the 100-ply
        // line and the win turns "cursed".
        let fresh = Position::from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 0 40").unwrap();
        let pressed = Position::from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 99 40").unwrap();
        assert_eq!(tb.probe_wdl(&fresh.chess), Some(TbOutcome::Win));
        let dtz = tb.probe_dtz(&fresh.chess).unwrap();
        assert!(dtz >= 2, "test needs dtz >= 2, got {dtz}");
        assert_eq!(
            tb.probe_wdl(&pressed.chess),
            Some(TbOutcome::CursedWin),
            "99 halfmoves + dtz {dtz} crosses 100: the win is only cursed"
        );
    }

    #[test]
    fn wdl_only_tables_probe_without_dtz() {
        // A directory holding only .rtbw files must still probe (the
        // after-zeroing fallback), while DTZ probes answer None.
        let dir = std::env::temp_dir().join("morstilia-tb-wdl-only");
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["KQvK.rtbw", "KRvK.rtbw"] {
            let src = tb_dir().join(name);
            let dst = dir.join(name);
            std::fs::copy(&src, &dst).unwrap();
        }
        let (tb, report) = Syzygy::load(dir.to_str().unwrap());
        assert_eq!(report.files, 2);
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        assert_eq!(tb.probe_wdl(&pos.chess), Some(TbOutcome::Win));
        assert_eq!(tb.probe_dtz(&pos.chess), None, "no DTZ tables present");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- range and coverage gates -------------------------------------------

    #[test]
    fn unsupported_piece_count_is_a_miss() {
        let tb = tb();
        // 5 pieces: KQ+R vs K+R — beyond the 4-piece limit.
        let pos = Position::from_fen("4k2r/8/8/8/8/8/2R5/3QK3 w - - 0 1").unwrap();
        assert_eq!(shakmaty::Position::board(&pos.chess).occupied().count(), 5);
        assert_eq!(tb.probe_wdl(&pos.chess), None);
        assert_eq!(tb.probe_dtz(&pos.chess), None);
        assert_eq!(tb.root_best_move(&pos.chess), None);
    }

    #[test]
    fn uncovered_material_within_range_is_a_miss() {
        // KQvKB is a standard 4-piece table that we simply did not include:
        // within the piece range, but no table file -> graceful miss.
        let tb = tb();
        let pos = Position::from_fen("4kb2/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        assert_eq!(shakmaty::Position::board(&pos.chess).occupied().count(), 4);
        assert_eq!(tb.probe_wdl(&pos.chess), None);
    }

    #[test]
    fn castling_rights_position_is_a_miss() {
        let tb = tb();
        // KRvKR with white still holding K-side castling rights: Syzygy
        // tables do not contain positions with castling rights.
        let pos = Position::from_fen("4k2r/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        assert_eq!(shakmaty::Position::board(&pos.chess).occupied().count(), 4);
        assert_eq!(
            tb.probe_wdl(&pos.chess),
            None,
            "castling rights -> no probe"
        );
    }

    // --- score band ---------------------------------------------------------

    #[test]
    fn score_bands_are_ordered_and_not_mates() {
        let bands = [
            (TbOutcome::Win, 19_999),
            (TbOutcome::CursedWin, 10_000),
            (TbOutcome::Draw, 0),
            (TbOutcome::BlessedLoss, -10_000),
            (TbOutcome::Loss, -19_999),
        ];
        for (outcome, expected) in bands {
            assert_eq!(outcome.score(), expected);
            assert_eq!(wdl_score(outcome), expected);
            assert!(
                !crate::types::is_mate(expected),
                "a TB score is never a mate score"
            );
        }
        assert!(wdl_score(TbOutcome::Win) > wdl_score(TbOutcome::CursedWin));
        assert!(wdl_score(TbOutcome::CursedWin) > wdl_score(TbOutcome::Draw));
        assert!(wdl_score(TbOutcome::Draw) > wdl_score(TbOutcome::BlessedLoss));
        assert!(wdl_score(TbOutcome::BlessedLoss) > wdl_score(TbOutcome::Loss));
    }

    // --- root helpers -------------------------------------------------------

    #[test]
    fn root_best_move_is_legal_and_wins() {
        let tb = tb();
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let (m, dtz) = tb.root_best_move(&pos.chess).expect("KQvK is covered");
        assert!(m != RawMove::NULL);
        let legal: Vec<RawMove> = pos.legal_moves().iter().collect();
        assert!(legal.contains(&m), "best move must be legal");
        // `best_move` reports the DTZ *of the resulting position* (from the
        // side to move there, i.e. the opponent): negative = the opponent is
        // now losing.
        assert!(dtz < 0, "opponent faces a losing DTZ, got {dtz}");
        // Playing the TB mainline must keep the win.
        let child = pos.make_child(m);
        assert_eq!(
            tb.probe_wdl(&child.chess),
            Some(TbOutcome::Loss),
            "after white's winning move black still loses"
        );
    }

    #[test]
    fn root_best_move_of_a_draw_keeps_the_draw() {
        let tb = tb();
        // KRvKR is a draw; the DB-optimal move exists (dtz 0) and must not
        // blunder into a loss.
        let draw = Position::from_fen("4k2r/8/8/8/8/8/8/3RK3 w - - 0 1").unwrap();
        let (m, dtz) = tb
            .root_best_move(&draw.chess)
            .expect("drawn root still has a DB-optimal move");
        assert_eq!(dtz, 0, "drawn root -> dtz 0");
        let child = draw.make_child(m);
        assert_eq!(
            tb.probe_wdl(&child.chess),
            Some(TbOutcome::Draw),
            "the DB-optimal move holds the draw"
        );
    }

    #[test]
    fn root_best_move_is_none_out_of_range() {
        let tb = tb();
        let big = Position::startpos();
        assert_eq!(
            tb.root_best_move(&big.chess),
            None,
            "32 pieces is out of range"
        );
    }
}

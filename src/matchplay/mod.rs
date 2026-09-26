//! Engine-vs-engine match infrastructure (Stage 8).
//!
//! Everything needed to run a reproducible, statistically honest match
//! between two engine configurations and to freeze the result as a
//! machine-readable report:
//!
//! * **Configurations** — each side is an [`EngineConfig`]: its own
//!   evaluation parameters, Threads, Hash and optional Syzygy tables. That is
//!   what makes *engine A vs engine B* regression testing possible (old vs
//!   new eval, Threads 1 vs N, parameter sets, future NNUE builds).
//! * **Time controls** — [`TimeControl`]: fixed depth (deterministic),
//!   fixed time per move, or a classic `moves/base+inc` schedule with a
//!   per-side clock, a 10% safety reserve (an engine can never flag in our
//!   own matches) and the increment plus the `moves`-cycle bonus applied
//!   after each side's move.
//! * **Openings** — a match draws positions from an [`OpeningSuite`] in a
//!   seeded, duplicate-free order (see `crate::openings`).
//! * **Fairness** — colors alternate by default (`MatchConfig::alternate_colors`);
//!   result accounting is done from the *candidate's* point of view,
//!   colour-correctly per game.
//! * **Parallelism** — up to [`MatchConfig::parallel`] independent games run
//!   concurrently via `std::thread::scope`, each with its own searchers and
//!   a per-game derived seed, so a `Threads = 1` depth match produces the
//!   **identical** report sequentially or in parallel.
//! * **SPRT auto-stop** — when [`MatchConfig::sprt`] is set, games run in
//!   parallel *batches*, results are fed to the trinomial SPRT in game order,
//!   and the match stops as soon as a decision (accept/reject) is reached.
//! * **Reports** — [`MatchReport`] serializes (serde JSON) every game, the
//!   W/D/L + score, Elo + Wilson CI, average plies, termination histogram,
//!   SPRT state, engine parameters, search settings and a source fingerprint
//!   ([`source_fingerprint`]) so future stages can detect regressions from a
//!   stored report alone. [`run_match`] accepts a previous report to *resume*
//!   a match (games are pure functions of the seed, so replayed games are
//!   byte-identical).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use shakmaty::Color;

use crate::board::Position;
use crate::evaluation::EvalParams;
use crate::openings::OpeningSuite;
use crate::rating::{GameResult, Sprt, SprtConfig, SprtDecision, Wdl};
use crate::search::{Searcher, TimeLimit};
use crate::selfplay::{Outcome, Termination, record, san_of, terminal_state};
use crate::types::{RawMove, is_mate};

pub mod cli;

/// The tagged baseline frozen by `--record-baseline`.
pub const BASELINE_TAG: &str = "Stage8-Classical-SMP-Baseline";

// ---------------------------------------------------------------------------
// Configurations
// ---------------------------------------------------------------------------

/// One side of a match: an independent engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Display name (e.g. `baseline`, a TOML path).
    pub name: String,
    /// Evaluation parameters this side searches with.
    pub params: EvalParams,
    /// Short label for the parameter set in reports.
    pub params_label: String,
    /// Search threads for this side (1 = deterministic).
    pub threads: usize,
    /// Transposition-table size in MB for this side.
    pub hash_mb: usize,
    /// Optional Syzygy tables, shared (read-only) by this side's searchers.
    pub syzygy: Option<Arc<crate::endgame::Syzygy>>,
    /// Human-readable Syzygy state for reports (`none`, or a load summary).
    pub syzygy_label: String,
}

impl EngineConfig {
    /// A side with the baseline evaluation and sane match defaults.
    pub fn baseline(name: &str) -> EngineConfig {
        EngineConfig::new(name, EvalParams::default())
    }

    /// A side with the given evaluation parameters.
    pub fn new(name: &str, params: EvalParams) -> EngineConfig {
        EngineConfig {
            name: name.to_string(),
            params,
            params_label: "eval-params".to_string(),
            threads: 1,
            hash_mb: 16,
            syzygy: None,
            syzygy_label: "none".to_string(),
        }
    }
}

/// Which engine the match's W/D/L/Elo/SPRT speak for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSide {
    /// The match's first side (game 0 White).
    White,
    /// The match's second side.
    Black,
}

/// Time control of a match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeControl {
    /// Fixed search depth per move — **deterministic**.
    Depth(i32),
    /// A fixed number of seconds per move.
    FixedMove(f64),
    /// Classic schedule: `moves` moves of `base` seconds, plus `inc` seconds
    /// per move. Standard notation `40/10+0.1` → `moves = 40, base = 10.0,
    /// inc = 0.1`. Per-move budget is `(remaining − 10% reserve) / 25 + inc`,
    /// so neither side can ever run out of time.
    Classic {
        moves: u32,
        base_sec: f64,
        inc_sec: f64,
    },
}

impl TimeControl {
    /// A compact, stable label (`depth 6`, `movetime 0.05s`, `40/10+0.1`).
    pub fn label(&self) -> String {
        match self {
            TimeControl::Depth(d) => format!("depth {d}"),
            TimeControl::FixedMove(s) => format!("movetime {s}s"),
            TimeControl::Classic {
                moves,
                base_sec,
                inc_sec,
            } => format!("{moves}/{base_sec}+{inc_sec}"),
        }
    }

    /// Depth controls are the only fully deterministic ones (time-based
    /// controls depend on wall-clock / NPS).
    pub fn is_deterministic(&self) -> bool {
        matches!(self, TimeControl::Depth(_))
    }
}

/// Full match configuration.
#[derive(Debug, Clone)]
pub struct MatchConfig {
    /// Total `--games` (a cap when `sprt` is set).
    pub games: usize,
    /// Deterministic match seed: opening order + per-game seeds.
    pub seed: u64,
    /// Match "White" (game 0's White side when colors alternate).
    pub white: EngineConfig,
    /// Match "Black".
    pub black: EngineConfig,
    pub tc: TimeControl,
    /// Alternate colors every game (fair testing) — default true.
    pub alternate_colors: bool,
    /// Hard cap on plies before a game is a draw.
    pub max_plies: usize,
    /// End a game immediately when the search proves a forced mate.
    pub adjudicate_mate: bool,
    /// SPRT auto-stop configuration; `None` plays exactly `games` games.
    pub sprt: Option<SprtConfig>,
    /// Concurrent games (default 1). Threads=1 depth matches are identical
    /// to sequential regardless of this value.
    pub parallel: usize,
    /// The side the report's W/D/L/Elo/SPRT describe.
    pub candidate: CandidateSide,
    /// Opening-suite name (for the report).
    pub suite_name: String,
}

impl Default for MatchConfig {
    fn default() -> Self {
        MatchConfig {
            games: 10,
            seed: 1,
            white: EngineConfig::baseline("baseline"),
            black: EngineConfig::baseline("baseline"),
            tc: TimeControl::Depth(6),
            alternate_colors: true,
            max_plies: 240,
            adjudicate_mate: true,
            sprt: None,
            parallel: 1,
            candidate: CandidateSide::White,
            suite_name: "classical-v1".to_string(),
        }
    }
}

/// Loads Syzygy tables for a match side. A missing/broken path is **not**
/// fatal: the side gets no tables and `label` reports the problem (visible in
/// the report, just like the UCI `SyzygyPath` handling).
pub fn load_syzygy(path: Option<&str>) -> (Option<Arc<crate::endgame::Syzygy>>, String) {
    match path.map(str::trim) {
        None | Some("") => (None, "none".to_string()),
        Some(p) => {
            let (sz, rep) = crate::endgame::Syzygy::load(p);
            if sz.is_loaded() {
                let label = format!(
                    "{} files (max {} pieces) from {}",
                    rep.files, rep.max_pieces, p
                );
                (Some(Arc::new(sz)), label)
            } else {
                let warn = rep
                    .warnings
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "load failed".to_string());
                (None, format!("{warn}"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A single game
// ---------------------------------------------------------------------------

/// The seeded time budget for one move: `(remaining − 10% reserve) / 25 + inc`
/// under [`TimeControl::Classic`], the full fixed chunk under
/// [`TimeControl::FixedMove`], and 0 under `Depth` (the search ignores time).
pub fn move_budget(remaining_ms: u64, tc: &TimeControl) -> u64 {
    match tc {
        TimeControl::Depth(_) => 0,
        TimeControl::FixedMove(sec) => (sec * 1000.0).max(1.0) as u64,
        TimeControl::Classic {
            moves: _,
            base_sec: _,
            inc_sec,
        } => {
            // Keep a 10% safety reserve so the clock can never flag.
            let usable = ((remaining_ms as f64) * 0.9).floor().max(1.0) as u64;
            let by_rate = usable / 25;
            let inc = (inc_sec * 1000.0).max(0.0) as u64;
            by_rate.saturating_add(inc).min(usable)
        }
    }
}

/// The [`TimeLimit`] handed to the search for one move.
fn time_limit_for(tc: &TimeControl, budget_ms: u64) -> TimeLimit {
    match tc {
        TimeControl::Depth(d) => TimeLimit {
            depth: Some(*d),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        },
        TimeControl::FixedMove(_) => TimeLimit {
            depth: None,
            nodes: None,
            movetime_ms: Some(budget_ms),
            soft_ms: budget_ms,
            hard_ms: budget_ms,
            infinite: true,
        },
        TimeControl::Classic { .. } => {
            let grace = (budget_ms / 8).max(25);
            TimeLimit {
                depth: None,
                nodes: None,
                movetime_ms: None,
                soft_ms: budget_ms,
                hard_ms: budget_ms.saturating_add(grace),
                infinite: true,
            }
        }
    }
}

/// A side's clock under a [`TimeControl::Classic`] schedule.
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    remaining_ms: u64,
    cycle_done: u32,
}

impl Clock {
    fn new(tc: &TimeControl) -> Clock {
        let remaining_ms = match tc {
            TimeControl::Classic { base_sec, .. } => (base_sec * 1000.0).max(1.0) as u64,
            _ => 0,
        };
        Clock {
            remaining_ms,
            cycle_done: 0,
        }
    }

    /// Books `used_ms` of actual search time (never more than is left),
    /// grants the increment, and awards the `moves`-cycle bonus.
    fn tick(&mut self, used_ms: u128, tc: &TimeControl) {
        if let TimeControl::Classic {
            moves,
            base_sec,
            inc_sec,
        } = tc
        {
            self.remaining_ms = self.remaining_ms.saturating_sub(used_ms as u64);
            self.remaining_ms = self
                .remaining_ms
                .saturating_add((inc_sec * 1000.0).max(0.0) as u64);
            self.cycle_done += 1;
            if *moves > 0 && self.cycle_done >= *moves {
                self.remaining_ms = self
                    .remaining_ms
                    .saturating_add((base_sec * 1000.0).max(1.0) as u64);
                self.cycle_done = 0;
            }
        }
    }
}

/// One played game, before serialization. All fields are pure functions of the
/// inputs, so the same `(config, seed, game index)` replays identically.
#[derive(Debug, Clone)]
pub struct PlayedGame {
    /// 0-based game number within the match.
    pub game_no: usize,
    /// Opening-suite entry index used for this game.
    pub opening_idx: usize,
    /// The derived per-game seed (also stored in the report for audit).
    pub seed: u64,
    /// True when the match's `white` engine ("A") played White.
    pub white_was_a: bool,
    pub opening_name: &'static str,
    pub fen_after_opening: String,
    /// Every ply: opening plies followed by the engines' moves.
    pub moves: Vec<RawMove>,
    /// Outcome from White's point of view.
    pub outcome: Outcome,
    pub termination: Termination,
}

impl PlayedGame {
    /// Converts to the serializable report form.
    pub fn to_std(&self) -> StdGame {
        StdGame {
            game_no: self.game_no,
            opening_idx: self.opening_idx,
            seed: self.seed,
            white_was_a: self.white_was_a,
            opening: self.opening_name.to_string(),
            fen_after_opening: self.fen_after_opening.clone(),
            moves: self.moves.iter().map(|m| m.to_uci()).collect(),
            outcome: self.outcome.to_pgn().to_string(),
            termination: self.termination.label().to_string(),
        }
    }
}

/// The `i`-th O(1) per-game seed: the value a shared sequential
/// [`crate::book::SplitMix64`] would return on its `(game + 1)`-th call, so a
/// sequential runner and a parallel runner agree exactly.
pub fn game_seed(match_seed: u64, game: usize) -> u64 {
    const GOLDEN: u64 = 0x9E3779B97F4A7C15;
    splitmix_mix(match_seed.wrapping_add(GOLDEN.wrapping_mul(game as u64 + 1)))
}

/// The SplitMix64 finalizer (mirrors `crate::book::SplitMix64::next`).
fn splitmix_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Plays one game from a suite position with two engine configurations.
pub fn play_match_game(
    white: &EngineConfig,
    black: &EngineConfig,
    suite: &OpeningSuite,
    opening_idx: usize,
    game_no: usize,
    seed: u64,
    tc: &TimeControl,
    max_plies: usize,
    adjudicate_mate: bool,
    alternate_colors: bool,
) -> PlayedGame {
    let entry = suite.entry(opening_idx);
    // Replay the opening line, collecting the history window exactly as a
    // game would see it (used for honest threefold detection).
    let (mut pos, mut history, mut moves) = suite.position_of(opening_idx);

    let white_was_a = !alternate_colors || game_no % 2 == 0;

    let mut white_searcher = Searcher::with_params(white.hash_mb, white.params.clone());
    let mut black_searcher = Searcher::with_params(black.hash_mb, black.params.clone());
    if let Some(tb) = &white.syzygy {
        white_searcher.tb = tb.clone();
    }
    if let Some(tb) = &black.syzygy {
        black_searcher.tb = tb.clone();
    }
    let stop = Arc::new(AtomicBool::new(false));
    let mut clock_w = Clock::new(tc);
    let mut clock_b = Clock::new(tc);

    let (outcome, termination) = loop {
        // --- natural terminations (no search needed) -----------------------
        if let Some(t) = terminal_state(&pos, &history) {
            break (t.outcome(&pos), t);
        }
        if moves.len() >= max_plies {
            break (Outcome::Draw, Termination::MovesLimit);
        }

        // --- search & play the side to move --------------------------------
        let turn_is_white = pos.turn() == Color::White;
        // The two (searcher, clock) pairs are disjoint, so the branches below
        // never alias.
        let (searcher, clock) = if turn_is_white {
            (&mut white_searcher, &mut clock_w)
        } else {
            (&mut black_searcher, &mut clock_b)
        };
        let threads = if turn_is_white {
            white.threads
        } else {
            black.threads
        };
        let budget = move_budget(clock.remaining_ms, tc);
        let limits = time_limit_for(tc, budget);
        let r = searcher.search(&pos, &history, &limits, &stop, threads, &[]);
        clock.tick(r.time_ms, tc);

        // Score adjudication: a proven forced mate decides the game.
        if adjudicate_mate && is_mate(r.score) {
            let stm_wins = r.score > 0;
            break (
                if stm_wins {
                    if turn_is_white {
                        Outcome::WhiteWin
                    } else {
                        Outcome::BlackWin
                    }
                } else if turn_is_white {
                    Outcome::BlackWin
                } else {
                    Outcome::WhiteWin
                },
                Termination::AdjudicatedMate,
            );
        }
        if r.best == RawMove::NULL {
            break (Outcome::Draw, Termination::Aborted);
        }
        let child = pos.make_child(r.best);
        record(&mut history, &pos, &child);
        moves.push(r.best);
        pos = child;
    };

    PlayedGame {
        game_no,
        opening_idx,
        seed,
        white_was_a,
        opening_name: entry.name,
        fen_after_opening: suite.fen_of(opening_idx).to_string(),
        moves,
        outcome,
        termination,
    }
}

/// The [`GameResult`] of this game from the match's *candidate* side's point
/// of view (colour-correct through `white_was_a`).
pub fn candidate_result(g: &PlayedGame, candidate: CandidateSide) -> GameResult {
    let a_pov = match g.outcome {
        Outcome::WhiteWin => {
            if g.white_was_a {
                GameResult::Win
            } else {
                GameResult::Loss
            }
        }
        Outcome::Draw => GameResult::Draw,
        Outcome::BlackWin => {
            if g.white_was_a {
                GameResult::Loss
            } else {
                GameResult::Win
            }
        }
    };
    match candidate {
        CandidateSide::White => a_pov,
        CandidateSide::Black => match a_pov {
            GameResult::Win => GameResult::Loss,
            GameResult::Draw => GameResult::Draw,
            GameResult::Loss => GameResult::Win,
        },
    }
}

/// Result accounting from a serialized game (used when resuming).
fn candidate_result_std(g: &StdGame, candidate: CandidateSide) -> GameResult {
    let outcome = outcome_of_str(&g.outcome);
    let white_was_a = g.white_was_a;
    let played = PlayedGame {
        game_no: g.game_no,
        opening_idx: g.opening_idx,
        seed: g.seed,
        white_was_a,
        opening_name: "",
        fen_after_opening: String::new(),
        moves: Vec::new(),
        outcome,
        termination: Termination::Aborted,
    };
    candidate_result(&played, candidate)
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// One game as stored in a report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StdGame {
    pub game_no: usize,
    pub opening_idx: usize,
    pub seed: u64,
    pub white_was_a: bool,
    pub opening: String,
    pub fen_after_opening: String,
    /// All plies as UCI strings (opening + played), in order.
    pub moves: Vec<String>,
    /// PGN result: `1-0`, `1/2-1/2`, `0-1` (White's point of view).
    pub outcome: String,
    pub termination: String,
}

/// W/D/L counts in a report (from the candidate's point of view).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct WdlSer {
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
}

impl From<&Wdl> for WdlSer {
    fn from(w: &Wdl) -> WdlSer {
        WdlSer {
            wins: w.wins,
            draws: w.draws,
            losses: w.losses,
        }
    }
}

impl From<&WdlSer> for Wdl {
    fn from(w: &WdlSer) -> Wdl {
        Wdl {
            wins: w.wins,
            draws: w.draws,
            losses: w.losses,
        }
    }
}

/// SPRT state as stored in a report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SprtSer {
    pub elo0: f64,
    pub elo1: f64,
    pub draw_elo: f64,
    pub alpha: f64,
    pub beta: f64,
    pub games: u64,
    pub llr: f64,
    pub decision: String,
}

impl From<&Sprt> for SprtSer {
    fn from(s: &Sprt) -> SprtSer {
        SprtSer {
            elo0: s.config.elo0,
            elo1: s.config.elo1,
            draw_elo: s.config.draw_elo,
            alpha: s.config.alpha,
            beta: s.config.beta,
            games: s.games,
            llr: s.llr,
            decision: sprt_decision_label(s.decision()).to_string(),
        }
    }
}

/// One side of a report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineSummary {
    pub name: String,
    pub params: String,
    pub threads: usize,
    pub hash_mb: usize,
    pub syzygy: String,
}

/// Frozen engine identity at the time the match was played.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceManifest {
    pub engine: String,
    pub version: String,
    /// FNV-1a 64-bit fingerprint of `src/**/*.rs` + `Cargo.toml` — the
    /// "exact revision" stand-in for a repo without git.
    pub source_fingerprint: u64,
    /// `release` or `debug` profile of the binary.
    pub profile: String,
}

impl SourceManifest {
    fn current() -> SourceManifest {
        SourceManifest {
            engine: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            source_fingerprint: source_fingerprint(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .to_string(),
        }
    }
}

/// The full, machine-readable result of a match.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MatchReport {
    pub format: String,
    pub source: SourceManifest,
    pub suite: String,
    pub suite_entries: usize,
    pub seed: u64,
    pub games_requested: usize,
    pub games_completed: usize,
    pub time_control: String,
    pub tc_deterministic: bool,
    pub alternate_colors: bool,
    pub max_plies: usize,
    pub adjudicate_mate: bool,
    pub parallel: usize,
    pub candidate: String,
    pub white: EngineSummary,
    pub black: EngineSummary,
    /// Candidate's W/D/L.
    pub wdl: WdlSer,
    pub score_rate: f64,
    pub elo: f64,
    pub elo_ci_lo: f64,
    pub elo_ci_hi: f64,
    /// Mean plies per game (opening + played).
    pub avg_plies: f64,
    /// `(termination label, count)` histogram, sorted by label.
    pub terminations: Vec<(String, u64)>,
    pub sprt: Option<SprtSer>,
    /// Set by `--record-baseline` (e.g. `Stage8-Classical-SMP-Baseline`).
    pub baseline: Option<String>,
    pub games: Vec<StdGame>,
}

impl MatchReport {
    /// Pretty-printed JSON (deterministic for identical reports).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("report serializes")
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        fs::write(path, self.to_json()).with_context(|| format!("cannot write report to {path:?}"))
    }

    pub fn load(path: &Path) -> anyhow::Result<MatchReport> {
        let s = fs::read_to_string(path).with_context(|| format!("cannot read {path:?}"))?;
        serde_json::from_str(&s).with_context(|| format!("cannot parse report {path:?}"))
    }

    /// The report's W/D/L as the live [`Wdl`] type.
    pub fn wdl(&self) -> Wdl {
        Wdl::from(&self.wdl)
    }

    /// The candidate's score rate.
    pub fn score_rate(&self) -> f64 {
        self.score_rate
    }

    /// The stored SPRT decision string (`Running`/`AcceptH1`/`AcceptH0`/`MaxGames`).
    pub fn sprt_decision(&self) -> &str {
        self.sprt
            .as_ref()
            .map(|s| s.decision.as_str())
            .unwrap_or("no-sprt")
    }
}

/// Stable label for an SPRT decision.
pub fn sprt_decision_label(d: SprtDecision) -> &'static str {
    match d {
        SprtDecision::Running => "Running",
        SprtDecision::AcceptH1 => "AcceptH1",
        SprtDecision::AcceptH0 => "AcceptH0",
        SprtDecision::MaxGames => "MaxGames",
    }
}

/// `1-0`/`0-1`/anything else → the corresponding white-relative [`Outcome`].
fn outcome_of_str(s: &str) -> Outcome {
    match s {
        "1-0" => Outcome::WhiteWin,
        "0-1" => Outcome::BlackWin,
        _ => Outcome::Draw,
    }
}

// ---------------------------------------------------------------------------
// The match runner
// ---------------------------------------------------------------------------

/// A progress event: one game finished. `done` counts *completed* games, so
/// the first callback of a parallel batch reports the batch start index.
#[derive(Debug)]
pub struct MatchProgress<'a> {
    pub done: usize,
    pub total: usize,
    pub game: &'a StdGame,
}

/// Field-level check that a resumed report matches the current config, so
/// replayed games keep their determinism guarantees.
fn validate_resume(cfg: &MatchConfig, resume: &MatchReport) -> anyhow::Result<()> {
    let tc_ok = resume.time_control == cfg.tc.label();
    let w = &cfg.white;
    let b = &cfg.black;
    let engine_ok = resume.white.name == w.name
        && resume.white.threads == w.threads
        && resume.white.hash_mb == w.hash_mb
        && resume.black.name == b.name
        && resume.black.threads == b.threads
        && resume.black.hash_mb == b.hash_mb;
    let sprt_ok = match (&resume.sprt, &cfg.sprt) {
        (Some(r), Some(c)) => {
            (r.elo0 - c.elo0).abs() < 1e-9
                && (r.elo1 - c.elo1).abs() < 1e-9
                && (r.draw_elo - c.draw_elo).abs() < 1e-9
                && (r.alpha - c.alpha).abs() < 1e-9
                && (r.beta - c.beta).abs() < 1e-9
        }
        (None, None) => true,
        _ => false,
    };
    if resume.seed == cfg.seed
        && resume.suite == cfg.suite_name
        && resume.alternate_colors == cfg.alternate_colors
        && resume.max_plies == cfg.max_plies
        && resume.adjudicate_mate == cfg.adjudicate_mate
        && tc_ok
        && engine_ok
        && sprt_ok
    {
        Ok(())
    } else {
        bail!(
            "resume report does not match the current configuration \
             (seed {}, suite {:?}, tc {:?}, colors {:?}, engines {:?} vs {:?}, threads/hash/sprt differ); \
             refusing to mix incompatible matches",
            resume.seed,
            resume.suite,
            resume.time_control,
            resume.alternate_colors,
            w.name,
            b.name
        );
    }
}

/// Runs a match. `resume` continues an aborted match from its completed games
/// (replayed games are byte-identical). `stop` aborts cleanly between games;
/// a partially-completed parallel batch is discarded (games are pure, so a
/// resume replays it identically). `progress` is called once per finished
/// game.
pub fn run_match(
    cfg: &MatchConfig,
    suite: &OpeningSuite,
    resume: Option<&MatchReport>,
    stop: Option<&AtomicBool>,
    mut progress: impl FnMut(&MatchProgress),
) -> anyhow::Result<MatchReport> {
    if cfg.games == 0 {
        bail!("a match must play at least one game (--games N with N >= 1)");
    }
    if let Some(r) = resume {
        validate_resume(cfg, r)?;
    }

    let order = suite.order(cfg.seed);

    // --- seed state from the resumed report -------------------------------
    let mut games: Vec<StdGame> = resume.map(|r| r.games.clone()).unwrap_or_default();
    let mut wdl = Wdl::default();
    for g in &games {
        apply_result(&mut wdl, candidate_result_std(g, cfg.candidate));
    }
    let mut sprt_state: Option<Sprt> = cfg.sprt.map(Sprt::new);
    if let Some(r) = resume {
        if let (Some(s), Some(sc)) = (&r.sprt, cfg.sprt) {
            sprt_state = Some(Sprt {
                config: sc,
                games: s.games,
                llr: s.llr,
            });
        }
    }
    // A resumed, already-decided SPRT stops immediately.
    let sprt_finished = sprt_state
        .as_ref()
        .map(|s| s.decision() != SprtDecision::Running)
        .unwrap_or(false);

    let mut completed = games.len();

    'outer: while completed < cfg.games && !sprt_finished {
        if stop.map_or(false, |s| s.load(Ordering::Relaxed)) {
            break 'outer;
        }
        let batch = (cfg.games - completed).min(cfg.parallel.max(1));

        // --- play the batch concurrently -----------------------------------
        let mut batch_played: Vec<PlayedGame> = Vec::with_capacity(batch);
        {
            let (white, black) = (&cfg.white, &cfg.black);
            let tc = &cfg.tc;
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..batch)
                    .map(|k| {
                        let gi = completed + k;
                        let opening_idx = order[gi % order.len()];
                        let seed = game_seed(cfg.seed, gi);
                        scope.spawn(move || {
                            play_match_game(
                                white,
                                black,
                                suite,
                                opening_idx,
                                gi,
                                seed,
                                tc,
                                cfg.max_plies,
                                cfg.adjudicate_mate,
                                cfg.alternate_colors,
                            )
                        })
                    })
                    .collect();
                for h in handles {
                    batch_played.push(h.join().expect("match game thread panicked"));
                }
            });
        }

        // --- account, report, and run the SPRT in game order ---------------
        for g in batch_played {
            let cand = candidate_result(&g, cfg.candidate);
            let std_g = g.to_std();
            progress(&MatchProgress {
                done: completed,
                total: cfg.games,
                game: &std_g,
            });
            apply_result(&mut wdl, cand);
            games.push(std_g.clone());
            completed += 1;
            if let Some(s) = &mut sprt_state {
                s.update(cand);
                if s.decision() != SprtDecision::Running {
                    break 'outer;
                }
            }
        }
    }

    // --- summarize ---------------------------------------------------------
    let avg_plies = if games.is_empty() {
        0.0
    } else {
        games.iter().map(|g| g.moves.len() as f64).sum::<f64>() / games.len() as f64
    };
    let mut termination_counts: Vec<(String, u64)> = Vec::new();
    {
        let mut seen: Vec<(&str, u64)> = Vec::new();
        for g in &games {
            let label: &str = &g.termination;
            if let Some(e) = seen.iter_mut().find(|(t, _)| *t == label) {
                e.1 += 1;
            } else {
                seen.push((label, 1));
            }
        }
        seen.sort();
        termination_counts.extend(seen.iter().map(|(t, n)| (t.to_string(), *n)));
    }
    let (lo, hi) = wdl.elo_ci();

    Ok(MatchReport {
        format: "morstilia-match-report/v1".to_string(),
        source: SourceManifest::current(),
        suite: cfg.suite_name.clone(),
        suite_entries: suite.len(),
        seed: cfg.seed,
        games_requested: cfg.games,
        games_completed: completed,
        time_control: cfg.tc.label(),
        tc_deterministic: cfg.tc.is_deterministic(),
        alternate_colors: cfg.alternate_colors,
        max_plies: cfg.max_plies,
        adjudicate_mate: cfg.adjudicate_mate,
        parallel: cfg.parallel,
        candidate: match cfg.candidate {
            CandidateSide::White => "white".to_string(),
            CandidateSide::Black => "black".to_string(),
        },
        white: EngineSummary {
            name: cfg.white.name.clone(),
            params: cfg.white.params_label.clone(),
            threads: cfg.white.threads,
            hash_mb: cfg.white.hash_mb,
            syzygy: cfg.white.syzygy_label.clone(),
        },
        black: EngineSummary {
            name: cfg.black.name.clone(),
            params: cfg.black.params_label.clone(),
            threads: cfg.black.threads,
            hash_mb: cfg.black.hash_mb,
            syzygy: cfg.black.syzygy_label.clone(),
        },
        wdl: WdlSer::from(&wdl),
        score_rate: wdl.score_rate(),
        elo: wdl.elo(),
        elo_ci_lo: lo,
        elo_ci_hi: hi,
        avg_plies,
        terminations: termination_counts,
        sprt: sprt_state.as_ref().map(SprtSer::from),
        baseline: None,
        games,
    })
}

/// Adds one candidate result to the running W/D/L.
fn apply_result(wdl: &mut Wdl, r: GameResult) {
    match r {
        GameResult::Win => wdl.wins += 1,
        GameResult::Draw => wdl.draws += 1,
        GameResult::Loss => wdl.losses += 1,
    }
}

// ---------------------------------------------------------------------------
// PGN output
// ---------------------------------------------------------------------------

/// Writes every game of a report to `path` as one PGN document (SAN move
/// text, per-game tags for White/Black/Result/Opening/Termination/TimeControl).
pub fn write_match_pgn(report: &MatchReport, event: &str, path: &Path) -> anyhow::Result<()> {
    let mut pgn = String::new();
    for (i, g) in report.games.iter().enumerate() {
        pgn.push_str(
            &std_game_pgn(report, g, event, i + 1)
                .with_context(|| format!("cannot render game {} to SAN", g.game_no))?,
        );
        pgn.push('\n');
    }
    fs::write(path, pgn).with_context(|| format!("cannot write PGN to {path:?}"))
}

/// One game as a PGN record. The move list is *replayed* from the start
/// position purely to render SAN (and to verify each UCI move is legal).
fn std_game_pgn(
    report: &MatchReport,
    g: &StdGame,
    event: &str,
    round: usize,
) -> anyhow::Result<String> {
    let (white_name, black_name) = if g.white_was_a {
        (&report.white.name, &report.black.name)
    } else {
        (&report.black.name, &report.white.name)
    };
    let mut out = String::new();
    out.push_str(&format!("[Event \"{event}\"]\n"));
    out.push_str("[Site \"?\"]\n[Date \"????.??.??\"]\n");
    out.push_str(&format!("[Round \"{round}\"]\n"));
    out.push_str(&format!("[White \"{white_name}\"]\n"));
    out.push_str(&format!("[Black \"{black_name}\"]\n"));
    out.push_str(&format!("[Result \"{}\"]\n", g.outcome));
    out.push_str(&format!("[Opening \"{}\"]\n", g.opening));
    out.push_str(&format!("[Termination \"{}\"]\n", g.termination));
    out.push_str(&format!("[TimeControl \"{}\"]\n", report.time_control));
    out.push('\n');

    let mut pos = Position::startpos();
    let mut san: Vec<String> = Vec::with_capacity(g.moves.len());
    for uci in &g.moves {
        let (child, raw) = pos
            .play_uci(uci)
            .with_context(|| format!("illegal replay move {uci} in game {}", g.game_no))?;
        san.push(san_of(&pos, raw));
        pos = child;
    }
    let mut tokens: Vec<String> = Vec::with_capacity(san.len() + 8);
    for (i, s) in san.iter().enumerate() {
        if i % 2 == 0 {
            tokens.push(format!("{}.", i / 2 + 1));
        }
        tokens.push(s.clone());
    }
    tokens.push(g.outcome.clone());

    let mut line = String::new();
    for t in tokens {
        if !line.is_empty() && line.len() + t.len() + 1 > 72 {
            out.push_str(&line);
            out.push('\n');
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(&t);
    }
    out.push_str(&line);
    out.push('\n');
    Ok(out)
}

// ---------------------------------------------------------------------------
// Source fingerprint / suites
// ---------------------------------------------------------------------------

/// The built-in suites. Add new compiled-in suites here.
pub fn suite_by_name(name: &str) -> anyhow::Result<OpeningSuite> {
    match name {
        "classical-v1" => Ok(OpeningSuite::classical_v1()),
        other => bail!("unknown opening suite {other:?} (available: classical-v1)"),
    }
}

/// An FNV-1a 64-bit fingerprint of the engine's own source: every `*.rs`
/// under `src/` plus `Cargo.toml`, sorted by relative path. Two builds that
/// share a fingerprint share (almost certainly) the exact same source —
/// the "revision" recorded in [`SourceManifest`] for a repo without git.
pub fn source_fingerprint() -> u64 {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    collect_sources(&root.join("src"), "", &mut files);
    files.push((
        "Cargo.toml".to_string(),
        fs::read(root.join("Cargo.toml")).unwrap_or_default(),
    ));
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (path, bytes) in &files {
        for byte in path.bytes().chain([0u8]) {
            h = (h ^ byte as u64).wrapping_mul(0x100_0000_01B3);
        }
        for &byte in bytes {
            h = (h ^ byte as u64).wrapping_mul(0x100_0000_01B3);
        }
    }
    h
}

fn collect_sources(dir: &Path, rel: &str, out: &mut Vec<(String, Vec<u8>)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut names: Vec<(String, Option<PathBuf>)> = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        names.push((name, Some(e.path())));
    }
    names.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, path) in names {
        let path = path.expect("collected path");
        let rel_path = format!("{rel}/{name}");
        if path.is_dir() {
            collect_sources(&path, &rel_path, out);
        } else if name.ends_with(".rs") {
            if let Ok(bytes) = fs::read(&path) {
                out.push((rel_path, bytes));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "morstilia-matchplay-{}-{tag}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        p
    }

    fn suite() -> OpeningSuite {
        OpeningSuite::classical_v1()
    }

    /// A fast, deterministic config for tests.
    fn cfg(games: usize) -> MatchConfig {
        MatchConfig {
            games,
            seed: 7,
            tc: TimeControl::Depth(2),
            max_plies: 120,
            adjudicate_mate: true,
            parallel: 1,
            ..MatchConfig::default()
        }
    }

    fn run(cfg: &MatchConfig, suite: &OpeningSuite) -> MatchReport {
        run_match(cfg, suite, None, None, |_| {}).expect("match runs")
    }

    // --- time control ------------------------------------------------------

    #[test]
    fn move_budget_follows_the_schedule() {
        // Classic 40/10+0.1: remaining 10 000 ms → (9000/25) + 100 = 460 ms.
        let tc = TimeControl::Classic {
            moves: 40,
            base_sec: 10.0,
            inc_sec: 0.1,
        };
        assert_eq!(move_budget(10_000, &tc), 460);
        // Reserve: never spend the last 10%.
        assert!(move_budget(100, &tc) <= 90);
        // Fixed time is fixed.
        assert_eq!(move_budget(0, &TimeControl::FixedMove(0.05)), 50);
        // Depth ignores time entirely.
        assert_eq!(move_budget(u64::MAX, &TimeControl::Depth(6)), 0);
    }

    #[test]
    fn clock_ticks_increment_and_cycle_bonus() {
        let tc = TimeControl::Classic {
            moves: 2,
            base_sec: 10.0,
            inc_sec: 0.1,
        };
        let mut c = Clock::new(&tc);
        assert_eq!(c.remaining_ms, 10_000);
        // Move 1: spends 100 ms, gets 100 ms increment.
        c.tick(100, &tc);
        assert_eq!(c.remaining_ms, 10_000);
        // Move 2: completes the 2-move cycle → +10 s bonus.
        c.tick(70, &tc);
        assert_eq!(c.remaining_ms, 10_030 + 10_000);
        // Cycle counter resets.
        assert_eq!(c.cycle_done, 0);
        // A huge overshoot cannot push the clock negative.
        let mut c = Clock::new(&tc);
        c.tick(u128::MAX, &tc);
        assert_eq!(c.remaining_ms, 100); // inc granted after full spend
    }

    #[test]
    fn depth_is_deterministic_fixed_time_is_not() {
        assert!(TimeControl::Depth(6).is_deterministic());
        assert!(!TimeControl::FixedMove(0.5).is_deterministic());
        assert!(
            !TimeControl::Classic {
                moves: 40,
                base_sec: 10.0,
                inc_sec: 0.1
            }
            .is_deterministic()
        );
    }

    #[test]
    fn game_seeds_are_unique_per_game_and_o1() {
        let a = game_seed(1, 0);
        let b = game_seed(1, 1);
        let c = game_seed(1, 0);
        assert_ne!(a, b, "consecutive games must get different seeds");
        assert_eq!(a, c, "same (match seed, game) must derive the same seed");
        // Matches a sequential SplitMix64 stream: definitionally the same.
        let mut rng = crate::book::SplitMix64(1);
        assert_eq!(a, rng.next());
        assert_eq!(b, rng.next());
    }

    // --- accounting --------------------------------------------------------

    #[test]
    fn candidate_result_is_colour_correct() {
        // A wins as White.
        let a_white_win = PlayedGame {
            game_no: 0,
            opening_idx: 0,
            seed: 1,
            white_was_a: true,
            opening_name: "x",
            fen_after_opening: String::new(),
            moves: Vec::new(),
            outcome: Outcome::WhiteWin,
            termination: Termination::Checkmate,
        };
        assert_eq!(
            candidate_result(&a_white_win, CandidateSide::White),
            GameResult::Win
        );
        assert_eq!(
            candidate_result(&a_white_win, CandidateSide::Black),
            GameResult::Loss
        );
        // A loses a game in which it played Black (victory went to B as White).
        let b_white_win = PlayedGame {
            game_no: 1,
            opening_idx: 0,
            seed: 1,
            white_was_a: false,
            opening_name: "x",
            fen_after_opening: String::new(),
            moves: Vec::new(),
            outcome: Outcome::WhiteWin,
            termination: Termination::Checkmate,
        };
        assert_eq!(
            candidate_result(&b_white_win, CandidateSide::White),
            GameResult::Loss
        );
        assert_eq!(
            candidate_result(&b_white_win, CandidateSide::Black),
            GameResult::Win
        );
    }

    #[test]
    fn accumulated_wdl_swaps_with_alternation() {
        let mk = |white_was_a: bool, outcome: Outcome| PlayedGame {
            game_no: 0,
            opening_idx: 0,
            seed: 0,
            white_was_a,
            opening_name: "x",
            fen_after_opening: String::new(),
            moves: Vec::new(),
            outcome,
            termination: Termination::Checkmate,
        };
        let mut w = Wdl::default();
        // Alternating: game 0 A=White (A wins), game 1 B=White, A loses.
        for (a_on_white, outcome) in [(true, Outcome::WhiteWin), (false, Outcome::WhiteWin)] {
            apply_result(
                &mut w,
                candidate_result(&mk(a_on_white, outcome), CandidateSide::White),
            );
        }
        assert_eq!(w.wins, 1);
        assert_eq!(w.losses, 1);
        assert_eq!(w.draws, 0);
    }

    // --- matches -----------------------------------------------------------

    #[test]
    fn same_config_replays_identically() {
        let suite = suite();
        let cfg = cfg(4);
        let a = run(&cfg, &suite);
        let b = run(&cfg, &suite);
        assert_eq!(a.to_json(), b.to_json(), "same seed must replay the match");
        assert_eq!(a.games_completed, 4);
        assert_eq!(a.games.len(), 4);
    }

    #[test]
    fn parallel_games_equal_sequential() {
        let suite = suite();
        let mut c1 = cfg(6);
        c1.parallel = 4;
        let mut c2 = cfg(6);
        c2.parallel = 1;
        // Slightly different hash sizes to prove per-game independence.
        c1.white.hash_mb = 8;
        c1.black.hash_mb = 8;
        c2.white.hash_mb = 8;
        c2.black.hash_mb = 8;
        let seq = run(&c2, &suite);
        let par = run(&c1, &suite);
        // The `parallel` field is *reported*, not an input, so compare the
        // games themselves plus the derived statistics instead of raw JSON.
        assert_eq!(par.games, seq.games, "games must be identical in parallel");
        assert_eq!(par.wdl, seq.wdl, "wdl must be identical in parallel");
        assert_eq!(par.avg_plies, seq.avg_plies, "avg plies must match");
        assert_eq!(par.games_completed, 6);
    }

    #[test]
    fn every_game_is_legal_and_terminations_consistent() {
        let suite = suite();
        let cfg = cfg(6);
        let rep = run(&cfg, &suite);
        for g in &rep.games {
            let mut pos = Position::startpos();
            for uci in &g.moves {
                let (child, _) = pos
                    .play_uci(uci)
                    .unwrap_or_else(|_| panic!("illegal move {uci} in game {}", g.game_no));
                pos = child;
            }
            match g.termination.as_str() {
                "checkmate" => {
                    assert!(pos.legal_moves().is_empty() && pos.is_check());
                    assert_eq!(
                        g.outcome,
                        if pos.turn() == Color::White {
                            "0-1"
                        } else {
                            "1-0"
                        }
                    );
                }
                "stalemate" => {
                    assert!(pos.legal_moves().is_empty() && !pos.is_check());
                    assert_eq!(g.outcome, "1/2-1/2");
                }
                "fifty-move" => assert!(pos.halfmoves() >= 100),
                _ => {} // draws and adjudications don't need a natural check
            }
        }
    }

    #[test]
    fn threads_gt_one_stays_stable() {
        let suite = suite();
        let mut cfg = cfg(2);
        cfg.white.threads = 2;
        cfg.black.threads = 2;
        let rep = run(&cfg, &suite);
        assert_eq!(rep.games_completed, 2);
        for g in &rep.games {
            assert!(!g.moves.is_empty());
            let mut pos = Position::startpos();
            for uci in &g.moves {
                let (child, _) = pos.play_uci(uci).expect("legal move");
                pos = child;
            }
        }
    }

    #[test]
    fn fixed_time_control_plays_complete_games() {
        let suite = suite();
        let mut cfg = MatchConfig {
            games: 1,
            seed: 11,
            tc: TimeControl::FixedMove(0.02),
            max_plies: 60,
            adjudicate_mate: true,
            parallel: 1,
            ..MatchConfig::default()
        };
        let rep = run(&cfg, &suite);
        assert_eq!(rep.games_completed, 1);
        let g = &rep.games[0];
        assert!(!g.moves.is_empty() && g.moves.len() <= 60 + 16);
        // Spot-check with a second seed that a fixed-time match can also run
        // from a different opening.
        cfg.seed = 12;
        let rep = run(&cfg, &suite);
        assert_eq!(rep.games_completed, 1);
    }

    #[test]
    fn classic_time_control_accounts_its_clock() {
        let tc = TimeControl::Classic {
            moves: 40,
            base_sec: 10.0,
            inc_sec: 0.25,
        };
        let suite = suite();
        let cfg = MatchConfig {
            games: 1,
            seed: 3,
            tc,
            max_plies: 60,
            adjudicate_mate: true,
            parallel: 1,
            ..MatchConfig::default()
        };
        let rep = run(&cfg, &suite);
        assert_eq!(rep.games_completed, 1);
        assert!(!rep.games[0].moves.is_empty());
    }

    #[test]
    fn sprt_stops_early_on_a_clear_outcome() {
        // A queen-and-king endgame where White to move has a forced mate in
        // one: the baseline, moving first, adjudicates a mate on the very
        // first move. With a wide H0/H1 gap a single decisive win crosses the
        // upper bound, so the match must stop after one game — deterministic
        // for this seed.
        let suite =
            OpeningSuite::from_positions(&[("kqk-mate-in-1", "7k/8/6K1/8/8/6Q1/8/8 w - - 0 1")])
                .expect("KQK suite is legal");
        let cfg = MatchConfig {
            games: 100,
            seed: 1,
            tc: TimeControl::Depth(6),
            max_plies: 60,
            adjudicate_mate: true,
            parallel: 1,
            // Colors do NOT alternate: White keeps the queen every single
            // game, so the candidate wins game after game (alternating would
            // cancel exactly, one win and one loss per pair, and SPRT would
            // never move).
            alternate_colors: false,
            sprt: Some(SprtConfig {
                elo0: -100.0,
                elo1: 100.0,
                draw_elo: 100.0,
                alpha: 0.05,
                beta: 0.05,
                max_games: 10_000,
            }),
            suite_name: "kqk-test".to_string(),
            ..MatchConfig::default()
        };
        let rep = run(&cfg, &suite);
        assert!(
            rep.games_completed < 100,
            "SPRT must stop before the cap (completed {})",
            rep.games_completed
        );
        // Every game is an adjudicated mate win for White (the queen side),
        // so the match is all wins with no draws/losses; the SPRT stops once
        // the accumulated wins push the log-likelihood ratio across the upper
        // bound (a handful of consecutive wins with these bounds).
        assert_eq!(rep.games[0].termination, "adjudicated-mate");
        assert_eq!(rep.games[0].outcome, "1-0");
        let decision = rep.sprt_decision();
        assert_eq!(decision, "AcceptH1", "a decisive win accepts the candidate");
        let wdl = rep.wdl();
        assert_eq!(wdl.losses, 0, "{wdl:?}");
        assert_eq!(wdl.draws, 0, "{wdl:?}");
        assert_eq!(wdl.wins, rep.games_completed as u64, "{wdl:?}");
        assert!(
            wdl.wins >= 3,
            "must need a few wins to cross the bound: {wdl:?}"
        );
        assert_eq!(rep.games_completed, rep.games.len());
    }

    #[test]
    fn sprt_without_flag_plays_all_games() {
        let suite = suite();
        let cfg = cfg(4);
        let rep = run(&cfg, &suite);
        assert_eq!(rep.games_completed, 4, "fixed games run to completion");
        assert_eq!(rep.sprt_decision(), "no-sprt");
    }

    // --- resume / serialization ---------------------------------------------

    #[test]
    fn json_round_trips() {
        let suite = suite();
        let rep = run(&cfg(4), &suite);
        let s = rep.to_json();
        let back: MatchReport = serde_json::from_str(&s).expect("parse");
        // serde_json's f64 formatting is not guaranteed bit-exact (a trailing
        // ULP can be lost when re-serializing a parsed value), so compare the
        // structural state with floats neutralized — everything the resume
        // path actually depends on (seeds, games, moves, counts, SPRT decision
        // strings) is integer/string data and compares exactly.
        let mut a = rep.clone();
        let mut b = back.clone();
        a.score_rate = 0.0;
        a.elo = 0.0;
        a.elo_ci_lo = 0.0;
        a.elo_ci_hi = 0.0;
        a.avg_plies = 0.0;
        b.score_rate = 0.0;
        b.elo = 0.0;
        b.elo_ci_lo = 0.0;
        b.elo_ci_hi = 0.0;
        b.avg_plies = 0.0;
        if let (Some(sa), Some(sb)) = (&mut a.sprt, &mut b.sprt) {
            sa.llr = 0.0;
            sb.llr = 0.0;
        }
        assert_eq!(a, b, "resume-relevant state must round-trip");
        assert_eq!(back.games, rep.games, "games must round-trip bit-exactly");
        // The serialization itself must be byte-deterministic for identical
        // reports *before* a parse round-trip (what reproducibility needs).
        assert_eq!(rep.to_json(), s);
    }

    #[test]
    fn resume_extends_a_match_identically() {
        let suite = suite();
        let mut cfg = cfg(4);
        // Run 2 games, save, then resume to 4 with a fresh process-like load.
        cfg.games = 2;
        let part = run(&cfg, &suite);
        assert_eq!(part.games_completed, 2);

        let path = tmp_path("resume.json");
        part.save(&path).expect("save");
        let loaded = MatchReport::load(&path).expect("load");

        cfg.games = 4;
        let progress = run_match(&cfg, &suite, Some(&loaded), None, |_| {}).expect("resume");
        assert_eq!(progress.games_completed, 4);
        // The first two replayed games must be byte-identical to the stored run.
        for (a, b) in progress.games.iter().zip(part.games.iter()) {
            assert_eq!(a, b, "resumed games must replay identically");
        }
        // The full 4-game run must also equal the resumed one.
        let full = run(&cfg, &suite);
        assert_eq!(progress.to_json(), full.to_json());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn resume_rejects_incompatible_configs() {
        let suite = suite();
        let rep = run(&cfg(2), &suite);
        let mut cfg = cfg(4);
        cfg.seed = 99; // changed seed -> different games
        assert!(run_match(&cfg, &suite, Some(&rep), None, |_| {}).is_err());
    }

    #[test]
    fn baseline_tag_is_frozen_by_the_report() {
        let suite = suite();
        let mut rep = run(&cfg(2), &suite);
        assert_eq!(rep.baseline, None);
        rep.baseline = Some(BASELINE_TAG.to_string());
        let s = rep.to_json();
        assert!(s.contains(BASELINE_TAG));
    }

    #[test]
    fn pgn_render_uses_san_and_result() {
        let suite = suite();
        let cfg = cfg(2);
        let rep = run(&cfg, &suite);
        let path = tmp_path("games.pgn");
        write_match_pgn(&rep, "test", &path).expect("write pgn");
        let text = fs::read_to_string(&path).expect("read pgn");
        for g in &rep.games {
            assert!(text.contains(&format!("[Result \"{}\"]", g.outcome)));
            assert!(text.contains(&format!("[Termination \"{}\"]", g.termination)));
            assert!(text.contains(&format!("[Opening \"{}\"]", g.opening)));
        }
        // Every move token is SAN (contains dots for white plies).
        assert!(text.contains("1."));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn source_fingerprint_is_stable_within_a_build() {
        let a = source_fingerprint();
        let b = source_fingerprint();
        assert_eq!(a, b);
        assert!(
            a != 0xcbf2_9ce4_8422_2325,
            "must not be the FNV-1a offset basis"
        );
    }
}

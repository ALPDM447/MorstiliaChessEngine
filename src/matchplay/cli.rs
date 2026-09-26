//! Command-line driver for engine-vs-engine matches (Stage 8).
//!
//! `selfplay` (alias `morstilia-selfplay`) runs a reproducible match between
//! two engine configurations and reports the honest strength signal:
//!
//! ```text
//! W/D/L counts, score rate, logistic Elo + 95% Wilson confidence interval,
//! average game length, termination histogram and a trinomial SPRT verdict
//! (never NPS/depth/node totals).
//! ```
//!
//! Beyond the Stage-5 fixed-depth self-play, the driver adds:
//!
//! * time controls (`--tc depth=N|movetime=S|M/B+I`);
//! * per-side Threads/Hash (`--white-threads`/`--black-threads`, ...);
//! * a reproducible opening suite (`--suite classical-v1`, seeded order,
//!   duplicate-free, color-balanced — separate from the Polyglot book);
//! * alternating colors (`--alternate on`, the default);
//! * parallel games (`--parallel N`, result-identical to sequential at
//!   `Threads = 1`);
//! * SPRT auto-stop (`--sprt` stops as soon as a decision is reached);
//! * a machine-readable JSON report (`--report match.json`);
//! * resume/restart (`--resume match.json` continues an aborted match
//!   byte-identically);
//! * baseline freezing (`--record-baseline match.json` tags the report
//!   `Stage8-Classical-SMP-Baseline`).
//!
//! The same `--seed` replays the identical match at Threads = 1 depth
//! control. Diagnostics (per-game progress, file notes, the optional tune
//! dataset line) go to **stderr**; the summary goes to **stdout**.

use std::path::Path;

use anyhow::{Context, bail};

use crate::board::Position;
use crate::evaluation::EvalParams;
use crate::matchplay::{
    BASELINE_TAG, CandidateSide, EngineConfig, MatchConfig, MatchReport, TimeControl, load_syzygy,
    run_match, suite_by_name, write_match_pgn,
};
use crate::rating::SprtConfig;
use crate::selfplay::{GameRecord, Outcome, Termination};

const USAGE: &str = "\
Morstilia — deterministic engine-vs-engine matches and strength measurement.

Usage:
  selfplay [options]

Options:
  --games N             number of games (default 10; a cap when --sprt)
  --tc SPEC             time control: depth=N | movetime=S | M/B+I
                        e.g. --tc 40/10+0.1 (default depth 6)
  --depth N             shorthand for --tc depth=N
  --seed N              deterministic opening/game seed (default 1)
  --suite NAME          opening suite (default classical-v1: 48 positions, 24
                        white-to-move and 24 black-to-move)
  --alternate on|off    alternate colors every game (default on)
  --parallel N          concurrent games (default 1; Threads=1 depth matches
                        are byte-identical to sequential at any N)
  --white FILE          eval params TOML for side A (default: baseline)
  --black FILE          eval params TOML for side B (default: baseline)
  --hash MB             TT size per side (default 16)
  --white-hash MB       TT size for side A
  --black-hash MB       TT size for side B
  --threads N           search threads per side (default 1; >1 is Lazy SMP
                        strength testing and is NOT deterministic)
  --white-threads N     search threads for side A
  --black-threads N     search threads for side B
  --candidate SIDE      report POV for W/D/L + Elo + SPRT: white|black
                        (default white; with --alternate the candidate plays
                        both colors)
  --max-plies N         hard cap before a game is a draw (default 240)
  --adjudicate-mate B   end when the search proves a forced mate (default on)
  --syzygy DIR          Syzygy tables for both sides (optional; a bad path is
                        reported but never fatal)
  --sprt                auto-stop when the SPRT decides (default off: play
                        exactly --games and print a diagnostic SPRT)
  --sprt-elo0 F         H0 advantage of the SPRT (default -2.0)
  --sprt-elo1 F         H1 advantage of the SPRT (default +3.0)
  --sprt-alpha F        false-positive rate (default 0.05)
  --sprt-beta F         false-negative rate (default 0.05)
  --sprt-draw-elo F     draw threshold of the outcome model (default 100.0)
  --compare A.json B.json
                        A-vs-B regression: load two machine-readable reports
                        (same suite/seed/time control, same candidate side)
                        and print the score-rate delta, independent-binomial
                        95% CI and verdict (improved|similar|regressed) as
                        machine-readable JSON
  --out FILE            write the match as PGN (SAN move text)
  --report FILE         write the machine-readable JSON report
  --resume FILE         continue a previous --report (must match config)
  --record-baseline F   freeze this match as the Stage-8 baseline report
  --event NAME          PGN event tag (default 'Morstilia match')
  --help                this text

Examples:
  selfplay --games 1000 --depth 10 --seed 42 --report results/a10.json
  selfplay --games 100 --tc 40/10+0.1 --parallel 8 --report results/tc.json
  selfplay --resume results/a10.json --games 2000 --report results/a10-full.json
  selfplay --candidate white --sprt --games 1000 \\
           --white tuned.toml --black baseline --sprt-elo0 -2 --sprt-elo1 3
  selfplay --compare new.json baseline.json   # A-vs-B regression verdict

The match is deterministic for a given seed at Threads = 1 depth control:
same inputs, same games. Strength is reported as W/D/L + Elo + 95% CI +
SPRT — never nodes or NPS.
";

/// Entry point shared by the `selfplay` and `morstilia-selfplay` binaries.
pub fn main() {
    let prog = std::env::args()
        .next()
        .and_then(|p| {
            Path::new(&p)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| "selfplay".to_string());
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("{prog}: {e:#}");
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
}

fn run(args: &[String]) -> anyhow::Result<()> {
    // --- defaults -----------------------------------------------------------
    let mut games_n = 10usize;
    let mut depth_opt: Option<i32> = None;
    let mut tc_opt: Option<TimeControl> = None;
    let mut seed = 1u64;
    let mut white_path: Option<String> = None;
    let mut black_path: Option<String> = None;
    let mut hash = 16usize;
    let mut white_hash: Option<usize> = None;
    let mut black_hash: Option<usize> = None;
    let mut threads = 1usize;
    let mut white_threads: Option<usize> = None;
    let mut black_threads: Option<usize> = None;
    let mut candidate = "white".to_string();
    let mut suite_name = "classical-v1".to_string();
    let mut alternate = true;
    let mut parallel = 1usize;
    let mut adjudicate = true;
    let mut max_plies = 240usize;
    let mut syzygy: Option<String> = None;
    let mut sprt_enabled = false;
    let (mut elo0, mut elo1, mut alpha, mut beta, mut draw_elo) = {
        let d = SprtConfig::default();
        (d.elo0, d.elo1, d.alpha, d.beta, d.draw_elo)
    };
    let mut out: Option<String> = None;
    let mut report_path: Option<String> = None;
    let mut resume_path: Option<String> = None;
    let mut record_baseline: Option<String> = None;
    let mut compare: Option<(String, String)> = None;
    let mut event = "Morstilia match".to_string();

    // --- parse --------------------------------------------------------------
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            "--compare" => {
                let a = args
                    .get(i + 1)
                    .with_context(|| "--compare needs two report paths")?
                    .clone();
                let b = args
                    .get(i + 2)
                    .with_context(|| "--compare needs two report paths")?
                    .clone();
                compare = Some((a, b));
                // Advance past `--compare`, path A and path B.
                i += 3;
            }
            _ => {
                // `--name=value` inline form: match on the key so both spellings
                // reach the same arm below.
                let (key, inline) = match flag.split_once('=') {
                    Some((k, v)) => (k.to_string(), Some(v.to_string())),
                    None => (flag.clone(), None),
                };
                // Bare boolean flags take no value: `flag_value` would consume
                // (and discard) the *next* token, silently swallowing the flag
                // that follows (`--sprt --report x` -> `unknown argument: x`).
                let v = if matches!(
                    key.as_str(),
                    "--sprt" | "--no-alternate" | "--no-adjudicate"
                ) {
                    i += 1;
                    inline.unwrap_or_default()
                } else {
                    flag_value(args, &mut i, &key)
                        .with_context(|| format!("unexpected argument: {flag} (use --help)"))?
                };
                let parse_u = |what: &str, v: &str| -> anyhow::Result<usize> {
                    v.parse()
                        .with_context(|| format!("{what} must be an integer"))
                };
                match key.as_str() {
                    "--games" => games_n = parse_u("--games", &v)?,
                    "--depth" => depth_opt = Some(v.parse().context("--depth must be an integer")?),
                    "--tc" => tc_opt = Some(parse_tc(&v)?),
                    "--seed" => seed = v.parse().context("--seed must be an integer")?,
                    "--suite" => suite_name = v,
                    "--alternate" => alternate = parse_bool("--alternate", &v)?,
                    "--no-alternate" => alternate = false,
                    "--parallel" => parallel = parse_u("--parallel", &v)?,
                    "--white" => white_path = Some(v),
                    "--black" => black_path = Some(v),
                    "--hash" => hash = parse_u("--hash", &v)?,
                    "--white-hash" => white_hash = Some(parse_u("--white-hash", &v)?),
                    "--black-hash" => black_hash = Some(parse_u("--black-hash", &v)?),
                    "--threads" => threads = parse_u("--threads", &v)?,
                    "--white-threads" => white_threads = Some(parse_u("--white-threads", &v)?),
                    "--black-threads" => black_threads = Some(parse_u("--black-threads", &v)?),
                    "--candidate" => candidate = v,
                    "--max-plies" => max_plies = parse_u("--max-plies", &v)?,
                    "--adjudicate-mate" => adjudicate = parse_bool("--adjudicate-mate", &v)?,
                    "--no-adjudicate" => adjudicate = false,
                    "--syzygy" => syzygy = Some(v),
                    "--sprt" => sprt_enabled = true,
                    "--sprt-elo0" => elo0 = v.parse().context("--sprt-elo0 must be a number")?,
                    "--sprt-elo1" => elo1 = v.parse().context("--sprt-elo1 must be a number")?,
                    "--sprt-alpha" => alpha = v.parse().context("--sprt-alpha must be a number")?,
                    "--sprt-beta" => beta = v.parse().context("--sprt-beta must be a number")?,
                    "--sprt-draw-elo" => {
                        draw_elo = v.parse().context("--sprt-draw-elo must be a number")?
                    }
                    "--out" => out = Some(v),
                    "--report" => report_path = Some(v),
                    "--resume" => resume_path = Some(v),
                    "--record-baseline" => record_baseline = Some(v),
                    "--event" => event = v,
                    other => bail!("unknown argument: {other}"),
                }
            }
        }
    }

    // --- compare mode: no match is played ----------------------------------
    if let Some((path_a, path_b)) = compare {
        let ra = MatchReport::load(Path::new(&path_a))?;
        let rb = MatchReport::load(Path::new(&path_b))?;
        let cmp = crate::regression::compare_match_reports(&ra, &rb);
        println!(
            "{}",
            serde_json::to_string_pretty(&cmp).expect("comparison serializes")
        );
        return Ok(());
    }

    // --- validate -----------------------------------------------------------
    if games_n == 0 {
        bail!("--games must be >= 1");
    }
    if parallel == 0 {
        bail!("--parallel must be >= 1");
    }
    if !matches!(candidate.as_str(), "white" | "black") {
        bail!("--candidate must be 'white' or 'black'");
    }
    let tc = match tc_opt {
        Some(t) => t,
        None => TimeControl::Depth(depth_opt.unwrap_or(6)),
    };
    if let TimeControl::Depth(d) = tc {
        if d < 1 {
            bail!("--depth must be >= 1");
        }
    }
    if max_plies < 1 {
        bail!("--max-plies must be >= 1");
    }

    // --- build the two sides ------------------------------------------------
    let suite = suite_by_name(&suite_name)?;

    let white_params = load_params(white_path.as_deref())?;
    let mut white = EngineConfig::new(name_of(white_path.as_deref()), white_params);
    white.threads = white_threads.unwrap_or(threads);
    white.hash_mb = white_hash.unwrap_or(hash);
    white.params_label = white_path.clone().unwrap_or_else(|| "baseline".to_string());
    let (wb, wbl) = load_syzygy(syzygy.as_deref());
    white.syzygy = wb;
    white.syzygy_label = wbl;

    let black_params = load_params(black_path.as_deref())?;
    let mut black = EngineConfig::new(name_of(black_path.as_deref()), black_params);
    black.threads = black_threads.unwrap_or(threads);
    black.hash_mb = black_hash.unwrap_or(hash);
    black.params_label = black_path.clone().unwrap_or_else(|| "baseline".to_string());
    let (bb, bbl) = load_syzygy(syzygy.as_deref());
    black.syzygy = bb;
    black.syzygy_label = bbl;

    let sprt_cfg = SprtConfig {
        elo0,
        elo1,
        draw_elo,
        alpha,
        beta,
        max_games: games_n as u64,
    };
    let cfg = MatchConfig {
        games: games_n,
        seed,
        white,
        black,
        tc,
        alternate_colors: alternate,
        max_plies,
        adjudicate_mate: adjudicate,
        sprt: if sprt_enabled { Some(sprt_cfg) } else { None },
        parallel,
        candidate: if candidate == "white" {
            CandidateSide::White
        } else {
            CandidateSide::Black
        },
        suite_name: suite_name.clone(),
    };

    let resume = match resume_path.as_deref() {
        Some(p) => Some(MatchReport::load(Path::new(p))?),
        None => None,
    };

    // --- play ---------------------------------------------------------------
    eprintln!(
        "match: {} vs {}  (candidate: {})  suite: {} ({} positions)  seed {}",
        cfg.white.name,
        cfg.black.name,
        candidate,
        suite_name,
        suite.len(),
        seed
    );
    eprintln!(
        "time control: {}  threads {} / {}  tt {} / {} MB  alternate colors: {}  parallel: {}",
        tc.label(),
        cfg.white.threads,
        cfg.black.threads,
        cfg.white.hash_mb,
        cfg.black.hash_mb,
        alternate,
        parallel
    );
    if resume.is_some() {
        eprintln!("resuming from completed games (replayed games are byte-identical)");
    }

    let report = run_match(&cfg, &suite, resume.as_ref(), None, |p| {
        eprintln!(
            "game {:3}/{:3}: {}  {}  plies {:3}  ({})",
            p.done + 1,
            p.total,
            p.game.outcome,
            p.game.opening,
            p.game.moves.len(),
            p.game.termination
        );
    })?;

    // --- stdout summary -----------------------------------------------------
    let wdl = report.wdl();
    println!(
        "match: {} vs {}  (candidate: {})",
        report.white.name, report.black.name, report.candidate
    );
    println!(
        "suite: {} ({} positions, seeded order from seed {})",
        report.suite, report.suite_entries, report.seed
    );
    println!(
        "time control: {}  threads {} / {}  tt {} / {} MB",
        report.time_control,
        report.white.threads,
        report.black.threads,
        report.white.hash_mb,
        report.black.hash_mb
    );
    println!(
        "games: {} completed of {} requested  colors alternate: {}",
        report.games_completed, report.games_requested, report.alternate_colors
    );
    println!("{wdl}");
    println!(
        "score {:.3}  elo {}  95% CI [{}, {}]",
        wdl.score_rate(),
        fmt_elo(wdl.elo()),
        fmt_elo(report.elo_ci_lo),
        fmt_elo(report.elo_ci_hi)
    );
    println!("avg game length {:.1} plies", report.avg_plies);
    if !report.terminations.is_empty() {
        let hist = report
            .terminations
            .iter()
            .map(|(t, n)| format!("{t}: {n}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("terminations: {hist}");
    }
    match &report.sprt {
        Some(s) => println!(
            "sprt: elo0 {} elo1 {} alpha {} beta {}  llr {:+.3}  games {}  decision {}",
            s.elo0, s.elo1, s.alpha, s.beta, s.llr, s.games, s.decision
        ),
        None => println!(
            "sprt: not enabled (played exactly {} games; add --sprt to auto-stop)",
            report.games_requested
        ),
    }
    if let Some(tag) = &report.baseline {
        println!("baseline: frozen as {tag}");
    } else if record_baseline.is_some() {
        // The freeze is applied to the saved copy; announce it here too so
        // the summary is unambiguous about what got frozen.
        println!("baseline: frozen as {BASELINE_TAG}");
    }

    // --- files --------------------------------------------------------------
    if let Some(p) = &out {
        write_match_pgn(&report, &event, Path::new(p)).with_context(|| "cannot write PGN")?;
        eprintln!("wrote {} games to {p}", report.games_completed);
    }
    if let Some(p) = record_baseline {
        let mut rep = report.clone();
        rep.baseline = Some(BASELINE_TAG.to_string());
        rep.save(Path::new(&p))?;
        eprintln!(
            "baseline frozen: {BASELINE_TAG} -> {p} (seed {}, tc {}, games {}, {} games)",
            rep.seed, rep.time_control, rep.games_completed, rep.suite
        );
    }
    if let Some(p) = report_path {
        report
            .save(Path::new(&p))
            .with_context(|| format!("cannot write report to {p:?}"))?;
        eprintln!("wrote match report to {p}");
    } else if resume_path.is_some() {
        // Resume without an explicit --report: refresh the original file.
        let p = resume_path.as_deref().expect("checked");
        report.save(Path::new(p))?;
        eprintln!("updated resume report {p}");
    }

    // Tune linking (informational, matches the Stage-5 workflow): only in the
    // classic fixed-depth, fixed-color mode where white-relative records and a
    // single opening stream have their original meaning.
    if !alternate {
        if let TimeControl::Depth(d) = tc {
            let records = std_games_to_records(&report);
            let ds = crate::tuning::dataset::DataSet::from_selfplay_games(&records).with_seed(seed);
            eprintln!(
                "dataset: {} entries from {} games (seed {seed}, depth {d}) -> pipe into tune",
                ds.len(),
                records.len()
            );
        }
    }
    Ok(())
}

/// `40/10+0.1`, `depth 8`, `depth=N`, `movetime 0.5` → the corresponding
/// [`TimeControl`]. The documented `name=value` spelling is accepted next to
/// `name value`.
fn parse_tc(s: &str) -> anyhow::Result<TimeControl> {
    let t = s.trim();
    if let Some(d) = t.strip_prefix("depth") {
        let d: i32 = d
            .trim_start_matches(['=', ' ', '\t'])
            .parse()
            .with_context(|| format!("bad depth control {s:?}"))?;
        if d < 1 {
            bail!("depth must be >= 1");
        }
        return Ok(TimeControl::Depth(d));
    }
    if let Some(x) = t.strip_prefix("movetime") {
        let sec: f64 = x
            .trim_start_matches(['=', ' ', '\t'])
            .parse()
            .with_context(|| format!("bad movetime control {s:?}"))?;
        if !(sec > 0.0) {
            bail!("movetime must be positive");
        }
        return Ok(TimeControl::FixedMove(sec));
    }
    // Classic `M/B+I`: exactly one `/` and one `+`.
    let (moves_part, base_part) = t
        .split_once('/')
        .with_context(|| format!("bad time control {s:?} (use depth=N, movetime=S or M/B+I)"))?;
    let (base, inc) = base_part
        .split_once('+')
        .with_context(|| format!("bad time control {s:?} (missing increment, use M/B+I)"))?;
    let moves: u32 = moves_part
        .trim()
        .parse()
        .with_context(|| format!("bad move count in {s:?}"))?;
    let base_sec: f64 = base
        .trim()
        .parse()
        .with_context(|| format!("bad base time in {s:?}"))?;
    let inc_sec: f64 = inc
        .trim()
        .parse()
        .with_context(|| format!("bad increment in {s:?}"))?;
    if moves == 0 || !(base_sec > 0.0) || inc_sec < 0.0 {
        bail!("bad time control {s:?}");
    }
    Ok(TimeControl::Classic {
        moves,
        base_sec,
        inc_sec,
    })
}

fn parse_bool(what: &str, v: &str) -> anyhow::Result<bool> {
    match v.trim() {
        "on" | "true" | "yes" | "1" => Ok(true),
        "off" | "false" | "no" | "0" => Ok(false),
        other => bail!("{what} must be on|off, got {other:?}"),
    }
}

/// `None`/empty → the baseline defaults; a path must load cleanly.
fn load_params(path: Option<&str>) -> anyhow::Result<EvalParams> {
    match path {
        None | Some("") => Ok(EvalParams::default()),
        Some(p) => {
            EvalParams::load(p).with_context(|| format!("cannot load eval params from {p:?}"))
        }
    }
}

fn name_of(path: Option<&str>) -> &str {
    match path {
        None | Some("") => "baseline",
        Some(p) => p,
    }
}

fn fmt_elo(e: f64) -> String {
    if e.is_infinite() {
        if e > 0.0 {
            "+inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        format!("{e:+.1}")
    }
}

/// Rebuilds [`GameRecord`]s from a report's serialized games (for the
/// informational tune-dataset line). Every UCI move is re-checked as legal.
fn std_games_to_records(report: &MatchReport) -> Vec<GameRecord> {
    report
        .games
        .iter()
        .map(|g| {
            let mut pos = Position::startpos();
            let mut moves = Vec::with_capacity(g.moves.len());
            for uci in &g.moves {
                let (child, raw) = pos.play_uci(uci).expect("report moves are legal");
                moves.push(raw);
                pos = child;
            }
            let outcome = match g.outcome.as_str() {
                "1-0" => Outcome::WhiteWin,
                "0-1" => Outcome::BlackWin,
                _ => Outcome::Draw,
            };
            GameRecord {
                moves,
                outcome,
                termination: Termination::Aborted,
                opening: g.opening.clone(),
            }
        })
        .collect()
}

/// Reads `--name value` or `--name=value` starting at `*i`, advancing past
/// the consumed tokens. Returns `None` when the next token is not `name`.
fn flag_value(args: &[String], i: &mut usize, name: &str) -> Option<String> {
    let arg = args.get(*i)?;
    if let Some(v) = arg.strip_prefix(&format!("{name}=")) {
        *i += 1;
        return Some(v.to_string());
    }
    if arg == name {
        *i += 1;
        let v = args.get(*i).cloned();
        *i += usize::from(v.is_some());
        return v;
    }
    None
}

/// Parsing is exercised directly (no subprocess needed for the format rules).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::matchplay::TimeControl;

    #[test]
    fn tc_parsing() {
        assert_eq!(parse_tc("depth 8").unwrap(), TimeControl::Depth(8));
        assert_eq!(parse_tc("depth=5").unwrap(), TimeControl::Depth(5));
        assert_eq!(parse_tc("depth5").unwrap(), TimeControl::Depth(5));
        assert_eq!(
            parse_tc("movetime 0.5").unwrap(),
            TimeControl::FixedMove(0.5)
        );
        assert_eq!(
            parse_tc("movetime=0.25").unwrap(),
            TimeControl::FixedMove(0.25)
        );
        assert_eq!(
            parse_tc("40/10+0.1").unwrap(),
            TimeControl::Classic {
                moves: 40,
                base_sec: 10.0,
                inc_sec: 0.1
            }
        );
        assert!(parse_tc("garbage").is_err());
        assert!(parse_tc("40/10").is_err(), "missing increment");
        assert!(parse_tc("0/10+1").is_err(), "zero moves");
    }

    #[test]
    fn bool_parsing() {
        assert!(parse_bool("x", "on").unwrap());
        assert!(parse_bool("x", "1").unwrap());
        assert!(!parse_bool("x", "off").unwrap());
        assert!(parse_bool("x", "maybe").is_err());
    }

    #[test]
    fn flag_value_handles_both_forms() {
        let args = vec![
            "--games".to_string(),
            "4".to_string(),
            "--seed=9".to_string(),
        ];
        let mut i = 0;
        assert_eq!(flag_value(&args, &mut i, "--games").unwrap(), "4");
        assert_eq!(flag_value(&args, &mut i, "--seed").unwrap(), "9");
        assert_eq!(i, 3);
        assert_eq!(flag_value(&args, &mut i, "--nope"), None);
    }
}

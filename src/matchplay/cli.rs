//! Command-line driver for engine-vs-engine matches (Stage 8).
//!
//! `selfplay` runs a reproducible match between two engine configurations
//! and reports the honest strength signal:
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
use std::sync::Arc;

use anyhow::{Context, bail};

use crate::board::Position;
use crate::evaluation::EvalParams;
use crate::matchplay::{
    BASELINE_TAG, CandidateSide, EngineConfig, EngineEvaluator, MatchConfig, MatchReport,
    TimeControl, load_syzygy, run_match, suite_by_name, write_match_pgn,
};
use crate::rating::SprtConfig;
use crate::search::params::{SearchGates, SearchParams};
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
  --search-params FILE  search params TOML for both sides (default: the built-in
                        Stockfish 19 defaults). The singular/multi-cut margins,
                        the modern LMR formula, IIR and the pruning thresholds.
  --white-search-params FILE
  --black-search-params FILE
                        per-side search params; overrides --search-params for
                        that side. This is the A/B axis for a search-parameter
                        experiment: two files, one suite, one seed.
  --search-gate NAME=BOOL
                        switch one ported mechanism off. Repeatable and applies
                        to both sides unless prefixed, e.g.
                        --white-search-gate singular=false.
  --eval MODE           nnue|classical for both sides (default nnue)
  --white-eval MODE     per-side evaluator
  --black-eval MODE
  --nnue FILE           use a different .nnue net for both NNUE sides
  --white-nnue FILE     per-side net
  --black-nnue FILE
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
  --ab-sf19             the documented Stockfish-19 A/B: white = every
                        mechanism OFF (the pre-port search), black = the
                        shipped SF19 defaults, candidate = black. Refuses to
                        be combined with any --search-params/--search-gate
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
  selfplay --games 2000 --depth 12 --seed 7 \\
           --search-params tuned-search.toml \\
           --report results/sf19-search.json   # whole-engine A/B
  selfplay --games 2000 --depth 12 --seed 7 \\
           --white-search-gate singular=false \\
           --report results/no-singular.json   # single-mechanism A/B
  selfplay --games 2000 --depth 12 --seed 7 --ab-sf19 --sprt \\
           --report results/sf19-ab.json       # the headline A/B
  selfplay --compare new.json baseline.json   # A-vs-B regression verdict

The match is deterministic for a given seed at Threads = 1 depth control:
same inputs, same games. Strength is reported as W/D/L + Elo + 95% CI +
SPRT — never nodes or NPS.
";

/// Entry point of the `selfplay` binary.
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
    let mut sp_path: Option<String> = None;
    let mut white_sp_path: Option<String> = None;
    let mut black_sp_path: Option<String> = None;
    let mut white_gates: Vec<(String, bool)> = Vec::new();
    let mut black_gates: Vec<(String, bool)> = Vec::new();
    let mut eval_mode: Option<EngineEvaluator> = None;
    let mut white_eval: Option<EngineEvaluator> = None;
    let mut black_eval: Option<EngineEvaluator> = None;
    let mut net_path: Option<String> = None;
    let mut white_net: Option<String> = None;
    let mut black_net: Option<String> = None;
    let mut ab_sf19 = false;
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
                    "--search-params" => sp_path = Some(v),
                    "--white-search-params" => white_sp_path = Some(v),
                    "--black-search-params" => black_sp_path = Some(v),
                    "--search-gate" => white_gates.push(parse_gate(&v)?),
                    "--white-search-gate" => white_gates.push(parse_gate(&v)?),
                    "--black-search-gate" => black_gates.push(parse_gate(&v)?),
                    "--eval" => eval_mode = Some(parse_evaluator(&v)?),
                    "--white-eval" => white_eval = Some(parse_evaluator(&v)?),
                    "--black-eval" => black_eval = Some(parse_evaluator(&v)?),
                    "--nnue" => net_path = Some(v),
                    "--white-nnue" => white_net = Some(v),
                    "--black-nnue" => black_net = Some(v),
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
                    "--ab-sf19" => ab_sf19 = true,
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
    if let TimeControl::Depth(d) = tc
        && d < 1
    {
        bail!("--depth must be >= 1");
    }
    if max_plies < 1 {
        bail!("--max-plies must be >= 1");
    }
    // `--ab-sf19` *defines* both sides' search parameters. Letting an explicit
    // parameter file be silently overridden (or the reverse) is exactly the
    // kind of quiet precedence rule that makes an A/B report mean something
    // other than what it says, so the combination is refused outright.
    if ab_sf19
        && (sp_path.is_some()
            || white_sp_path.is_some()
            || black_sp_path.is_some()
            || !white_gates.is_empty()
            || !black_gates.is_empty())
    {
        bail!(
            "--ab-sf19 sets both sides' search parameters itself; it cannot be combined \
             with --search-params/--white-search-params/--black-search-params or \
             --search-gate/--white-search-gate/--black-search-gate"
        );
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
    {
        let path = white_sp_path.as_deref().or(sp_path.as_deref());
        let sp = load_search_params(path, &white_gates)?;
        eprintln!(
            "search params (white): {}  fingerprint {}",
            path.unwrap_or("built-in defaults"),
            sp.fingerprint()
        );
        white = white
            .with_search_params(sp)
            .with_search_params_label(path.unwrap_or("built-in defaults"));
    }

    let black_params = load_params(black_path.as_deref())?;
    let mut black = EngineConfig::new(name_of(black_path.as_deref()), black_params);
    black.threads = black_threads.unwrap_or(threads);
    black.hash_mb = black_hash.unwrap_or(hash);
    black.params_label = black_path.clone().unwrap_or_else(|| "baseline".to_string());
    let (bb, bbl) = load_syzygy(syzygy.as_deref());
    black.syzygy = bb;
    black.syzygy_label = bbl;
    {
        let path = black_sp_path.as_deref().or(sp_path.as_deref());
        let sp = load_search_params(path, &black_gates)?;
        eprintln!(
            "search params (black): {}  fingerprint {}",
            path.unwrap_or("built-in defaults"),
            sp.fingerprint()
        );
        black = black
            .with_search_params(sp)
            .with_search_params_label(path.unwrap_or("built-in defaults"));
    }

    // --- evaluator ---------------------------------------------------------
    // The net is decoded once here, never per game: 100 MB of LEB128 inside a
    // measured match would dominate every time control.
    let white_eval = white_eval.or(eval_mode).unwrap_or_default();
    let black_eval = black_eval.or(eval_mode).unwrap_or_default();
    if white_eval != black_eval {
        eprintln!(
            "matchplay: evaluator mismatch ({} vs {}): this is a deliberate cross-evaluator A/B, not a search A/B",
            white_eval.as_str(),
            black_eval.as_str()
        );
    }
    let (white_net, black_net) = load_nets(
        net_path.as_deref(),
        white_net.as_deref(),
        black_net.as_deref(),
    )?;
    if (white_eval == EngineEvaluator::Nnue) != (white_net.is_some())
        || (black_eval == EngineEvaluator::Nnue) != (black_net.is_some())
    {
        bail!(
            "NNUE was requested but no net could be loaded (pass --nnue FILE, or --eval classical)"
        );
    }
    // One decoded net shared by `Arc` means *provably* identical weights; two
    // separate decodes mean the A/B measures the nets, not the search. Either
    // way the report says which, and a mismatch is announced here.
    let nets_identical = match (&white_net, &black_net) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    };
    if !nets_identical {
        eprintln!(
            "matchplay: WARNING: the two sides use *different* NNUE nets; the result measures the nets, not the search"
        );
    }
    eprintln!(
        "evaluator: {} vs {}{}",
        white_eval.as_str(),
        black_eval.as_str(),
        if white_eval == EngineEvaluator::Nnue {
            format!("  net hash {:#010x}", crate::nnue::NETWORK_HASH)
        } else {
            String::new()
        }
    );
    white = white.with_evaluator(white_eval, white_net);
    black = black.with_evaluator(black_eval, black_net);

    // --- the documented SF19 A/B --------------------------------------------
    // White becomes the pre-port search (every gate off) and Black the shipped
    // Stockfish 19 defaults, and Black becomes the candidate. The point of
    // spelling it out as a single flag is that *this* is the comparison the
    // port is claiming to be worth something, and expressing it as
    // `--search-gate singular=false --search-gate iir=false ...` invites a
    // typo that silently changes the baseline into a third configuration.
    if ab_sf19 {
        let mut baseline = SearchParams::default();
        baseline.gates = SearchGates::all_off();
        white = white
            .with_search_params(baseline)
            .with_search_params_label("baseline (every SF19 mechanism off)");
        black = black
            .with_search_params(SearchParams::default())
            .with_search_params_label("SF19 defaults (every mechanism on)");
        candidate = "black".to_string();
        eprintln!(
            "A/B: white = {} / black = {}",
            white.search_params_label, black.search_params_label
        );
        eprintln!("     candidate side forced to black");
    }

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
    eprintln!(
        "search params: white [{}] black [{}]",
        cfg.white.search_params_label, cfg.black.search_params_label
    );
    eprintln!(
        "gates:        white [{}]",
        crate::matchplay::gates_fingerprint(&cfg.white.search_params.gates)
    );
    eprintln!(
        "gates:        black [{}]",
        crate::matchplay::gates_fingerprint(&cfg.black.search_params.gates)
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
    println!(
        "search params: white [{}] black [{}]",
        report.white.search_params, report.black.search_params
    );
    println!(
        "evaluator: {} vs {}{}",
        report.white.evaluator,
        report.black.evaluator,
        report
            .white
            .net
            .as_ref()
            .map_or_else(String::new, |n| format!(
                "  net hash {:#010x} ({})",
                n.hash, n.file
            ))
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
    if !alternate && let TimeControl::Depth(d) = tc {
        let records = std_games_to_records(&report);
        let ds = crate::tuning::dataset::DataSet::from_selfplay_games(&records).with_seed(seed);
        eprintln!(
            "dataset: {} entries from {} games (seed {seed}, depth {d}) -> pipe into tune",
            ds.len(),
            records.len()
        );
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
        // `!(x > 0.0)` and not `x <= 0.0`: the negated form is also true for
        // NaN, so `--movetime nan` is rejected rather than silently accepted.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
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
    // As above: `!(base_sec > 0.0)` rejects NaN, `base_sec <= 0.0` would not.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
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

/// Parses `NAME=BOOL` for a `--*-search-gate` flag.
fn parse_gate(v: &str) -> anyhow::Result<(String, bool)> {
    let (name, on) = v
        .rsplit_once('=')
        .with_context(|| format!("--search-gate needs NAME=BOOL, got {v:?}"))?;
    let on = SearchGates::parse_gate_value(on)
        .with_context(|| format!("--search-gate {name}: {on:?} is not a boolean"))?;
    Ok((name.trim().to_string(), on))
}

fn parse_evaluator(v: &str) -> anyhow::Result<EngineEvaluator> {
    match v.trim().to_ascii_lowercase().as_str() {
        "nnue" => Ok(EngineEvaluator::Nnue),
        "classical" => Ok(EngineEvaluator::Classical),
        other => bail!("--eval must be nnue or classical, got {other:?}"),
    }
}

/// Decodes a net once, for one side. `None`/empty path means the net embedded
/// in the binary. A requested-but-unusable net is a hard error here: unlike over
/// UCI there is no interactive session to fall back in, and a match that
/// quietly played classical while the report said `nnue` would be worthless.
fn load_net(path: Option<&str>) -> anyhow::Result<Arc<crate::nnue::network::Network>> {
    let loaded = match path {
        None | Some("") => {
            crate::nnue::load_embedded_network().context("the embedded NNUE net failed to load")?
        }
        Some(p) => {
            let resolved = crate::nnue::resolve_net_path(p);
            crate::nnue::load_network(&resolved)
                .with_context(|| format!("cannot load the NNUE net at {}", resolved.display()))?
        }
    };
    Ok(Arc::new(loaded))
}

/// Decodes the *distinct* nets both sides ask for.
///
/// Two different paths means two different weight sets, and a search-parameter
/// A/B across them is not an A/B at all — so the mismatch is reported on stderr
/// rather than being quietly accepted. `None` means no side wants a net.
fn load_nets(
    common: Option<&str>,
    white: Option<&str>,
    black: Option<&str>,
) -> anyhow::Result<(
    Option<Arc<crate::nnue::network::Network>>,
    Option<Arc<crate::nnue::network::Network>>,
)> {
    let white_path = white.or(common);
    let black_path = black.or(common);
    if white_path.is_none() && black_path.is_none() {
        return Ok((None, None));
    }
    if white_path == black_path {
        let net = load_net(white_path)?;
        return Ok((Some(Arc::clone(&net)), Some(net)));
    }
    Ok((Some(load_net(white_path)?), Some(load_net(black_path)?)))
}

/// Resolves a search-parameter file plus the gates requested on the command
/// line. An unknown gate name is an error: a typo that silently leaves the
/// mechanism on is exactly the failure an A/B run cannot detect afterwards.
fn load_search_params(
    path: Option<&str>,
    gates: &[(String, bool)],
) -> anyhow::Result<SearchParams> {
    let mut sp = match path {
        None | Some("") => SearchParams::default(),
        Some(p) => SearchParams::load(p)
            .with_context(|| format!("cannot load search params from {p:?}"))?,
    };
    for (name, on) in gates {
        if !sp.gates.set(name, *on) {
            bail!(
                "unknown search gate {name:?}; known gates: {}",
                SearchGates::NAMES.join(", ")
            );
        }
    }
    Ok(sp)
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

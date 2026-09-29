//! NNUE training driver.
//!
//! Everything this binary does is *about the data*, not about the weights:
//! build a config, generate a labelled dataset from self-play, audit it, split
//! it, and compare a candidate net against the shipped one. It deliberately
//! contains no optimizer — see [`morstilia::training`] for why the engine owns
//! the data policy but not the training loop.
//!
//! Every mode is deterministic for a `--seed`. The default config, the games,
//! the samples and the dataset file are reproducible from `--config` and
//! `--seed` alone.
//!
//! Usage:
//! ```text
//! train --print-default-config          write a commented starter config
//! train --generate [--config F] [--seed N] [--games N] [--out F]
//! train --inspect FILE                 audit a dataset (WDL, health, buckets)
//! train --split FILE [--test-fraction F] [--seed N] [--out-train F] [--out-test F]
//! train --compare CAND REF --data FILE  candidate net vs reference on a dataset
//! train --list                         the dataset format and the config schema
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, bail};

use morstilia::search::TimeLimit;
use morstilia::training::config::TrainingConfig;
use morstilia::training::dataset::Dataset;
use morstilia::training::{Run, Teacher};

const USAGE: &str = "\
Morstilia — NNUE training data driver.

Usage:
  train --print-default-config
  train --generate [--config FILE] [--seed N] [--games N] [--depth N]
                 [--out FILE] [--eval-only]
  train --inspect FILE [--sample N]
  train --split FILE [--test-fraction F] [--seed N]
                   [--out-train FILE] [--out-test FILE]
  train --compare CANDIDATE REFERENCE --data FILE [--depth N] [--limit N]
  train --list

Options:
  --config FILE       training config TOML (default: built-in defaults)
  --seed N            generation seed; overrides the config
  --games N           games to play; overrides the config
  --depth N           self-play depth per move; overrides the config
  --teacher-depth N   teacher search depth; overrides the config
  --out FILE          dataset output (default: config data_path)
  --sample N          how many positions to audit in --inspect
  --test-fraction F   held-out fraction for --split
  --out-train FILE    training half output for --split
  --out-test FILE     test half output for --split
  --compare A B       score candidate net A against reference net B on a dataset
  --data FILE         the dataset --compare reads
  --limit N           positions for --compare (0 = all)
  --eval-only         print the config's summary and exit
  --print-default-config
  --list              print the dataset format and config schema
  --help              this text
";

fn main() {
    if let Err(e) = run() {
        eprintln!("train: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            return Ok(());
        }
        Some("--print-default-config") => {
            print!("{}", starter_config()?);
            return Ok(());
        }
        Some("--list") => {
            list();
            return Ok(());
        }
        Some("--inspect") => {
            let path = need(&args, "--inspect")?;
            return inspect(&path, flag_usize(&args, "--sample").unwrap_or(0));
        }
        Some("--split") => {
            let path = need(&args, "--split")?;
            return split(&args, &path);
        }
        Some("--compare") => {
            let cand = need(&args, "--compare")?;
            let reference = value_after(&args, "--compare", 1)
                .context("--compare needs CANDIDATE and REFERENCE")?;
            return compare(&args, &cand, &reference);
        }
        _ => {}
    }
    if args.iter().any(|a| a == "--generate") {
        return generate(&args);
    }
    if args.iter().any(|a| a == "--eval-only") {
        let cfg = load_config(&args)?;
        print_config(&cfg);
        return Ok(());
    }
    print!("{USAGE}");
    Ok(())
}

// ---------------------------------------------------------------------------
// modes
// ---------------------------------------------------------------------------

/// Builds and labels a dataset, then writes and audits it.
fn generate(args: &[String]) -> anyhow::Result<()> {
    let mut cfg = load_config(args)?;
    if let Some(seed) = flag_u64(args, "--seed") {
        cfg.generation.seed = seed;
    }
    if let Some(games) = flag_usize(args, "--games") {
        cfg.generation.games = games;
    }
    if let Some(depth) = flag_i32(args, "--depth") {
        cfg.generation.depth = depth;
    }
    if let Some(d) = flag_i32(args, "--teacher-depth") {
        cfg.teacher.depth = d;
    }
    if let Some(out) = flag_str(args, "--out") {
        cfg.data_path = PathBuf::from(out);
    }
    cfg.validate()?;

    print_config(&cfg);
    eprintln!(
        "generating {} games of self-play at depth {} ...",
        cfg.generation.games, cfg.generation.depth
    );
    let run = Run::from_config(cfg)?;
    let t0 = std::time::Instant::now();
    let (ds, report, health) = run.generate(&mut |n, r| {
        // Progress on stderr, so stdout stays a clean artefact stream. `n` and
        // `r.games` are the same number today; using `n` keeps the line honest
        // if the callback is ever driven per *sample* instead of per game.
        eprintln!(
            "  game {n:>6}  {} samples so far  ({:.0}s)",
            r.samples,
            t0.elapsed().as_secs_f64()
        );
    })?;
    eprintln!("{}", report.summary());
    eprintln!("{}", health.summary());

    let path = &run.config.data_path;
    ds.save(path)
        .with_context(|| format!("cannot write the dataset to {}", path.display()))?;
    let (w, d, l) = ds.wdl();
    eprintln!(
        "wrote {} samples to {} (W {w} D {d} L {l}, {:.1}s)",
        ds.len(),
        path.display(),
        t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "next: train --inspect {} --split {} --test-fraction {}",
        path.display(),
        path.display(),
        run.config.split.test_fraction
    );
    Ok(())
}

/// Audits an existing dataset.
fn inspect(path: &str, sample: usize) -> anyhow::Result<()> {
    let ds = Dataset::load(path).context("cannot load the dataset")?;
    println!("{}", ds.stats());
    let (w, d, l) = ds.wdl();
    println!("  W/D/L            {w} / {d} / {l}");
    println!("  draw rate        {:.1}%", 100.0 * ds.draw_rate());
    println!("  mean PV length   {:.2}", ds.mean_pv_len());
    let with_pv = ds.samples().iter().filter(|s| !s.pv.is_empty()).count();
    println!(
        "  with a PV        {with_pv} / {} ({:.1}%)",
        ds.len(),
        100.0 * with_pv as f64 / ds.len().max(1) as f64
    );

    let health = if sample > 0 && sample < ds.len() {
        // A sampled audit: the same code path on a prefix, so the report means
        // the same thing it would for a whole corpus.
        let mut sub = Dataset::empty(format!("{path} (first {sample})"));
        for s in ds.samples().iter().take(sample) {
            sub.push(s.clone());
        }
        sub.validate()
    } else {
        ds.validate()
    };
    let scope = if sample > 0 && sample < ds.len() {
        format!("first {sample} of {} (full audit: drop --sample)", sample)
    } else {
        format!("all {}", ds.len())
    };
    println!(
        "  health ({scope}): {}",
        health.summary().trim_start_matches("dataset health: ")
    );

    println!("  WDL by piece bucket (bucket = pieces/4):");
    for (bucket, (bw, bd, bl)) in ds.wdl_by_piece_bucket(4) {
        let total = bw + bd + bl;
        let pct = |n: usize| 100.0 * n as f64 / total.max(1) as f64;
        println!(
            "    {:>3} pieces  n={total:<6} W {pct_w:>5.1}%  D {pct_d:>5.1}%  L {pct_l:>5.1}%",
            bucket * 4,
            pct_w = pct(bw),
            pct_d = pct(bd),
            pct_l = pct(bl)
        );
    }

    for s in ds.samples().iter().take(3) {
        println!("  sample {} {:?}", s.label.as_char(), s.fen);
        if !s.pv.is_empty() {
            println!("    pv  {}", s.pv.join(" "));
        }
    }
    Ok(())
}

/// Writes the train/test halves as two files.
fn split(args: &[String], path: &str) -> anyhow::Result<()> {
    let ds = Dataset::load(path).context("cannot load the dataset")?;
    let fraction = flag_f64(args, "--test-fraction").unwrap_or(0.1);
    let seed = flag_u64(args, "--seed").unwrap_or(1);
    let (train, test) = ds.split(fraction, seed);
    if train.is_empty() {
        bail!("the split left no training data (test fraction {fraction})");
    }
    let train_path = flag_str(args, "--out-train")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(path).with_extension("train.txt"));
    let test_path = flag_str(args, "--out-test")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(path).with_extension("test.txt"));
    train.save(&train_path)?;
    test.save(&test_path)?;
    println!(
        "split {fraction:.3} (seed {seed}): {} train -> {}   {} test -> {}",
        train.len(),
        train_path.display(),
        test.len(),
        test_path.display()
    );
    Ok(())
}

/// Scores a candidate net against a reference net on a dataset.
///
/// This is the *acceptance test*, and it is deliberately conservative: a net
/// that scores better than the reference on the data it was trained on has
/// proved nothing, so the numbers reported here are diagnostics for a human,
/// never a pass/fail. Real acceptance is a match.
fn compare(args: &[String], candidate: &str, reference: &str) -> anyhow::Result<()> {
    let data = flag_str(args, "--data").context("--compare needs --data FILE")?;
    let depth = flag_i32(args, "--depth").unwrap_or(6);
    let limit = flag_usize(args, "--limit").unwrap_or(512);

    let ds = Dataset::load(&data).context("cannot load the dataset")?;
    let cand = load_net(candidate)?;
    let refr = load_net(reference)?;
    println!(
        "candidate  {candidate}  hash {:#010x}",
        morstilia::nnue::NETWORK_HASH
    );
    println!("reference  {reference}");
    println!(
        "dataset    {} ({} positions, depth {depth}, limit {limit})",
        ds.len(),
        data
    );

    let mut a = Teacher::builtin(depth);
    a.set_nnue(Some(Arc::clone(&cand)));
    let mut b = Teacher::builtin(depth);
    b.set_nnue(Some(refr));

    let n = limit.min(ds.len()).max(1);
    let mut agree = 0usize;
    let mut sum_abs = 0i64;
    let mut max_abs = 0i32;
    let mut label_mismatch = 0usize;
    for s in ds.samples().iter().take(n) {
        let Ok(pos) = s.position() else { continue };
        // A terminal position has no opinion to compare.
        if pos.legal_moves().is_empty() {
            continue;
        }
        let limits = TimeLimit {
            depth: Some(depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let sa = a.searcher_mut().search(&pos, &[], &limits, &stop, 1, &[]);
        let sb = b.searcher_mut().search(&pos, &[], &limits, &stop, 1, &[]);
        let d = (sa.score - sb.score).abs();
        sum_abs += i64::from(d);
        max_abs = max_abs.max(d);
        if sa.best == sb.best {
            agree += 1;
        }
        if (sa.score > 0) != (s.label == morstilia::training::Label::Win) {
            label_mismatch += 1;
        }
    }
    println!();
    println!("positions compared   {n}");
    println!(
        "same best move       {agree} ({:.1}%)",
        100.0 * agree as f64 / n as f64
    );
    println!("mean |score delta|   {:.2} cp", sum_abs as f64 / n as f64);
    println!("max  |score delta|   {max_abs} cp");
    println!(
        "label disagreements {label_mismatch} ({:.1}%)",
        100.0 * label_mismatch as f64 / n as f64
    );
    println!();
    println!("NOTE: this is a diagnostic, not an acceptance test. A net that beats its");
    println!("      reference on its own training data has proved nothing. Validate with");
    println!("      a match, and only then update the shipped net.");
    Ok(())
}

/// Prints the dataset format and the config schema.
fn list() {
    println!("dataset format v{}", morstilia::training::DATA_FORMAT);
    println!();
    println!("  <label> <FEN>[  <pv in san> ...]");
    println!();
    println!("  label   W (side to move wins) | D (draw) | L (side to move loses)");
    println!("  FEN     the position, fields separated by single spaces");
    println!("  pv      the teacher's principal variation, SAN, two spaces after the FEN");
    println!();
    println!("  comments start with '#'; blank lines are skipped; anything else is an");
    println!("  error rather than a silently dropped row.");
    println!();
    println!("config schema (all sections are required; unknown keys are rejected):");
    print!("{}", starter_config().unwrap_or_default());
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// A starter config with every field present and a comment on each.
fn starter_config() -> anyhow::Result<String> {
    let c = TrainingConfig::default();
    let body = c.to_toml_string()?;
    Ok(format!(
        "{}\n\
         # Morstilia NNUE training configuration.\n\
         #\n\
         # Everything below is optional in the sense that the built-in defaults are\n\
         # exactly these values, so an empty file is a valid config. Unknown keys are\n\
         # REJECTED rather than ignored, so a stale file fails loudly instead of being\n\
         # read as if it had been honoured.\n\
         #\n\
         # The engine owns the data policy and the validation policy. It does not own\n\
         # the optimizer: there is no learning rate or loss here on purpose.\n",
        body
    ))
}

fn load_config(args: &[String]) -> anyhow::Result<TrainingConfig> {
    match flag_str(args, "--config") {
        Some(p) => TrainingConfig::load(p).context("cannot load the training config"),
        None => Ok(TrainingConfig::default()),
    }
}

fn print_config(c: &TrainingConfig) {
    println!("data             {}", c.data_path.display());
    println!("out net          {}", c.out_net.display());
    println!("description      {}", c.description);
    println!(
        "generation       {} games, depth {}, stride {}, {} samples/game",
        c.generation.games,
        c.generation.depth,
        c.generation.stride_plies,
        c.generation.samples_per_game
    );
    println!(
        "  filters        min_pieces {}, include_checks {}, keep_draws {}",
        c.generation.min_pieces, c.generation.include_checks, c.generation.keep_draws
    );
    println!(
        "  suite/seed     {} / {}",
        c.generation.suite, c.generation.seed
    );
    println!(
        "teacher          {} at depth {}, max score {} cp{}",
        c.teacher.kind.as_str(),
        c.teacher.depth,
        c.teacher.max_score_cp,
        match &c.teacher.net {
            Some(p) => format!(", net {}", p.display()),
            None => ", classical evaluation".to_string(),
        }
    );
    println!(
        "split            {:.1}% test, seed {}",
        100.0 * c.split.test_fraction,
        c.split.seed
    );
    println!(
        "validation       {} positions, <= {} cp delta, <= {:.1}% churn",
        c.validation.tactical_positions,
        c.validation.max_score_delta_cp,
        100.0 * c.validation.max_best_move_churn
    );
    println!(
        "syzygy           {}",
        if c.syzygy_path.as_os_str().is_empty() {
            "(none)".to_string()
        } else {
            c.syzygy_path.display().to_string()
        }
    );
    println!();
}

fn load_net(path: &str) -> anyhow::Result<Arc<morstilia::nnue::network::Network>> {
    let resolved = morstilia::nnue::resolve_net_path(path);
    let net = morstilia::nnue::load_network(&resolved)
        .with_context(|| format!("cannot load the net at {}", resolved.display()))?;
    Ok(Arc::new(net))
}

fn need(args: &[String], flag: &str) -> anyhow::Result<String> {
    value_after(args, flag, 0).with_context(|| format!("{flag} needs a file name"))
}

fn value_after(args: &[String], flag: &str, skip: usize) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1 + skip))
        .cloned()
}

fn flag_str(args: &[String], flag: &str) -> Option<String> {
    value_after(args, flag, 0)
}

fn flag_usize(args: &[String], flag: &str) -> Option<usize> {
    value_after(args, flag, 0).and_then(|v| v.parse().ok())
}

fn flag_u64(args: &[String], flag: &str) -> Option<u64> {
    value_after(args, flag, 0).and_then(|v| v.parse().ok())
}

fn flag_i32(args: &[String], flag: &str) -> Option<i32> {
    value_after(args, flag, 0).and_then(|v| v.parse().ok())
}

fn flag_f64(args: &[String], flag: &str) -> Option<f64> {
    value_after(args, flag, 0).and_then(|v| v.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn flag_lookups_find_the_value_after_the_flag() {
        let a = args(&["--generate", "--seed", "42", "--games", "8"]);
        assert_eq!(flag_u64(&a, "--seed"), Some(42));
        assert_eq!(flag_usize(&a, "--games"), Some(8));
        assert_eq!(flag_str(&a, "--seed").as_deref(), Some("42"));
        assert_eq!(flag_u64(&a, "--missing"), None);
    }

    #[test]
    fn a_flag_with_no_value_is_not_a_flag() {
        let a = args(&["--seed"]);
        assert_eq!(flag_u64(&a, "--seed"), None);
        assert!(need(&a, "--seed").is_err());
    }

    #[test]
    fn the_starter_config_is_a_valid_config() {
        let text = starter_config().unwrap();
        // Strip the leading comment block: `from_toml_str` accepts comments, so
        // the whole thing must parse.
        let cfg = TrainingConfig::from_toml_str(&text).expect("starter config must parse");
        cfg.validate().unwrap();
    }

    #[test]
    fn the_starter_config_round_trips() {
        let cfg = TrainingConfig::default();
        let back = TrainingConfig::from_toml_str(&cfg.to_toml_string().unwrap()).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn config_overrides_come_from_the_command_line() {
        let mut cfg = TrainingConfig::default();
        let a = args(&["--seed", "7", "--games", "3", "--depth", "5"]);
        cfg.generation.seed = flag_u64(&a, "--seed").unwrap();
        cfg.generation.games = flag_usize(&a, "--games").unwrap();
        cfg.generation.depth = flag_i32(&a, "--depth").unwrap();
        assert_eq!(
            (
                cfg.generation.seed,
                cfg.generation.games,
                cfg.generation.depth
            ),
            (7, 3, 5)
        );
    }
}

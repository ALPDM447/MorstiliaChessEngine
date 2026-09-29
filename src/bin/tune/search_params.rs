//! Search-parameter tuning mode for the `tune` driver.
//!
//! The evaluation mode in `tune.rs` descends a Texel loss over a fixed
//! position set. That objective does not exist for the search: a search
//! parameter changes *which move is played* and *how long the game lasts*, so
//! the only honest objective is a game result. This module drives
//! [`morstilia::tuning::search::spsa_search`] over the flat
//! [`SearchParams`] vector with a deterministic fixed-game gauntlet, and
//! leaves the *validation* to `selfplay --sprt` — the run that produced a
//! parameter set and the run that decides whether it is real are separate
//! steps, deliberately.
//!
//! Everything is deterministic for a `--seed`: the same seed replays the same
//! gauntlet schedule, the same SPSA trajectory, the same checkpoints and the
//! same export.

use std::fs;
use std::sync::Arc;

use anyhow::{Context, bail};

use morstilia::book::SplitMix64;
use morstilia::matchplay::{EngineEvaluator, suite_by_name};
use morstilia::search::params::{SearchGates, SearchParams};
use morstilia::tuning::SpsaConfig;
use morstilia::tuning::search::{Gauntlet, SearchSubset, spsa_search_with_callback};

/// Everything the search-tuning mode needs. Parsed by the parent so `--help`
/// and the shared smoke preset stay in one place.
pub struct Flags {
    pub seed: u64,
    pub iterations: usize,
    pub subset: Vec<String>,
    pub in_path: Option<String>,
    pub out: String,
    pub checkpoint_every: usize,
    pub checkpoints_dir: String,
    pub gain: f64,
    pub perturbation: f64,
    pub hash: usize,
    /// Games per objective evaluation (the gauntlet width).
    pub games: usize,
    /// Search depth per move inside the gauntlet.
    pub depth: i32,
    /// Evaluator for both sides of the gauntlet.
    pub evaluator: EngineEvaluator,
    /// Reference search parameters: the engine the candidate is measured
    /// against. Defaults to the built-in set when `--reference` is absent.
    pub reference_path: Option<String>,
    /// Gates to force off, `name=bool`.
    pub gates: Vec<(String, bool)>,
    /// Use the cheap fixed-position proxy instead of the game gauntlet. Only
    /// for plumbing smoke tests; a strength claim needs the gauntlet.
    pub proxy: bool,
    /// FENs for the proxy objective.
    pub proxy_positions: Vec<String>,
}

/// Runs the search-parameter SPSA loop and exports the winning set.
pub fn run(f: &Flags) -> anyhow::Result<()> {
    let baseline = match &f.in_path {
        Some(p) => SearchParams::load(p).with_context(|| format!("cannot load {p:?}"))?,
        None => SearchParams::default(),
    };
    let baseline_src = f.in_path.as_deref().unwrap_or("built-in defaults");
    let mut baseline = baseline;
    for (name, on) in &f.gates {
        if !baseline.gates.set(name, *on) {
            bail!(
                "unknown search gate {name:?}; known gates: {}",
                SearchGates::NAMES.join(", ")
            );
        }
    }
    eprintln!(
        "search baseline: {baseline_src} ({} numeric params, gates {})",
        baseline.param_count(),
        morstilia::matchplay::gates_fingerprint(&baseline.gates)
    );

    let reference = match &f.reference_path {
        Some(p) => SearchParams::load(p).with_context(|| format!("cannot load reference {p:?}"))?,
        None => SearchParams::default(),
    };
    if let Some(p) = &f.reference_path {
        eprintln!("search reference: {p} ({})", reference.fingerprint());
    } else {
        eprintln!(
            "search reference: built-in defaults ({}) — the candidate is measured against them",
            reference.fingerprint()
        );
    }

    // The suite is validated up front so a typo fails before hours of games.
    let suite_len = suite_by_name("classical-v1")
        .context("the default opening suite is unavailable")?
        .len();
    eprintln!(
        "gauntlet: {} games per evaluation, depth {}, hash {} MB, suite classical-v1 ({} openings)",
        f.games, f.depth, f.hash, suite_len
    );

    let net = if f.evaluator == EngineEvaluator::Nnue {
        let loaded = morstilia::nnue::load_embedded_network()
            .context("the embedded NNUE net failed to load (use --search-eval classical)")?;
        eprintln!(
            "evaluator: nnue (net hash {:#010x})",
            morstilia::nnue::NETWORK_HASH
        );
        Some(Arc::new(loaded))
    } else {
        eprintln!("evaluator: classical");
        None
    };

    let gauntlet = Gauntlet {
        depth: f.depth,
        hash_mb: f.hash,
        games: f.games,
        reference_search_params: reference,
        evaluator: f.evaluator,
        net: net.clone(),
        ..Gauntlet::default()
    };
    if f.games == 0 {
        bail!("--gauntlet-games must be >= 1");
    }
    if f.games % 2 != 0 {
        // Not fatal, but the schedule would be colour-imbalanced, which
        // systematically biases whichever engine is the baseline. Say so
        // rather than let it look like a real result.
        eprintln!(
            "tune: WARNING: --gauntlet-games {} is odd, so the schedule is not colour-balanced",
            f.games
        );
    }

    let mut proxy = None;
    if f.proxy {
        if f.proxy_positions.is_empty() {
            bail!("--search-proxy needs at least one --search-fen FEN");
        }
        let mut positions = f.proxy_positions.clone();
        // The default fixed set is the start position and two simple middlegame
        // positions: enough to prove the plumbing, nowhere near enough to
        // measure strength. Stated here so nobody mistakes it for a dataset.
        if positions.len() == 1 {
            positions
                .push("r1bq1rk1/ppp2ppp/2n5/3p4/3P4/2P5/PP1B1PPP/RNBQ1RK1 w - - 0 8".to_string());
            positions.push(
                "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9".to_string(),
            );
        }
        eprintln!(
            "objective: PROXY over {} fixed positions — plumbing smoke test, not a strength measurement",
            positions.len()
        );
        proxy = Some(morstilia::tuning::search::ProxyObjective {
            depth: f.depth.min(8),
            hash_mb: f.hash,
            positions,
            evaluator: f.evaluator,
            net: net.clone(),
        });
    }

    let subset =
        SearchSubset::from_prefixes(&f.subset.iter().map(String::as_str).collect::<Vec<_>>());
    if subset.selected_count() == 0 {
        bail!(
            "--search-subset matched none of the {} numeric search parameters; run with \
             `--search-list` to see the names",
            SearchParams::bounds().len()
        );
    }
    eprintln!(
        "search subset: {} of {} numeric params",
        subset.selected_count(),
        SearchParams::bounds().len()
    );

    let cfg = SpsaConfig {
        a: f.gain,
        c: f.perturbation,
        iterations: f.iterations,
        ..SpsaConfig::default()
    };
    if f.checkpoint_every > 0 {
        fs::create_dir_all(&f.checkpoints_dir)
            .with_context(|| format!("cannot create {:?}", f.checkpoints_dir))?;
    }

    // A fresh stream derived from --seed: the dataset path's stream is spent.
    let mut rng = SplitMix64(f.seed ^ 0x5EA2_C4_5EA2_C4);
    let mut ckpt_err: Option<anyhow::Error> = None;

    let report = {
        let gauntlet_ref = &gauntlet;
        let mut objective: Box<dyn morstilia::tuning::search::SearchLoss> = match &proxy {
            Some(p) => Box::new(p.clone()),
            None => Box::new(GauntletObjective {
                inner: gauntlet_ref.clone(),
            }),
        };
        spsa_search_with_callback(
            &baseline,
            &subset,
            &cfg,
            objective.as_mut(),
            &mut rng,
            |k, loss, best| {
                let it = k + 1;
                let want_checkpoint =
                    f.checkpoint_every > 0 && (it % f.checkpoint_every == 0 || it == f.iterations);
                if want_checkpoint {
                    let path = format!("{}/search_it{:06}.toml", f.checkpoints_dir, it);
                    if let Err(e) = best.save(&path) {
                        ckpt_err = Some(e);
                        return;
                    }
                    eprintln!("checkpoint {it:>6}: best loss {loss:.6} -> {path}");
                }
                if it % 10 == 0 || it == f.iterations {
                    eprintln!("iteration {it:>6}: best loss {loss:.6}");
                }
            },
        )?
    };
    if let Some(e) = ckpt_err {
        return Err(e);
    }

    report
        .best
        .save(&f.out)
        .with_context(|| format!("cannot export tuned search params to {:?}", f.out))?;

    println!(
        "spsa: {} iterations  {}",
        report.iterations_run,
        report.summary()
    );
    println!(
        "tuning: updated {} of {} selected params  exported -> {}",
        report.updated_count,
        subset.selected_count(),
        f.out
    );

    // Name the parameters that actually moved. A tuning run whose diff is empty
    // has learned nothing, and the reader should not have to diff two TOMLs to
    // discover that.
    let defs = SearchParams::bounds();
    let base_vec = baseline.to_vec();
    let mut moved = 0usize;
    for (i, d) in defs.iter().enumerate() {
        let before = base_vec[i];
        let after = report.best_params[i];
        if (before as i64) != (after as i64) {
            moved += 1;
            println!("  {:<34} {} -> {}", d.name, fmt(before), fmt(after));
        }
    }
    if moved == 0 {
        println!("  (nothing moved: the objective was flat or every step was clamped)");
    }

    println!(
        "validate: this file is a *candidate*. Confirm it with \
         `selfplay --search-params {} --sprt` before believing it.",
        f.out
    );
    Ok(())
}

/// Newtype so the boxed trait object can carry the gauntlet by value.
struct GauntletObjective {
    inner: Gauntlet,
}

impl morstilia::tuning::search::SearchLoss for GauntletObjective {
    fn loss(&mut self, sp: &SearchParams, k: usize) -> anyhow::Result<f64> {
        // The base seed is fixed so the *iteration* index is the only thing
        // that varies between the `+` and the `−` evaluation; that is what
        // makes the two-sided gradient an estimate of the same game schedule
        // rather than of two different ones.
        morstilia::tuning::search::Gauntlet::loss(&self.inner, sp, k, 0x5EED_0000_0000_0001)
    }
}

/// `--compare-search A B`: the search-parameter counterpart of
/// `tune --compare A B`. Also prints the gate diff, because a file pair can be
/// numerically identical and still search differently.
pub fn compare(a: &str, b: &str) -> anyhow::Result<()> {
    let pa = SearchParams::load(a).with_context(|| format!("cannot load {a:?}"))?;
    let pb = SearchParams::load(b).with_context(|| format!("cannot load {b:?}"))?;
    let defs = SearchParams::bounds();
    let va = pa.to_vec();
    let vb = pb.to_vec();
    if va.len() != defs.len() || vb.len() != defs.len() {
        bail!(
            "search parameter vector size mismatch ({} vs {}, {} in bounds)",
            va.len(),
            vb.len(),
            defs.len()
        );
    }

    let mut n_changed = 0usize;
    let mut max_delta = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut per_prefix: std::collections::BTreeMap<&str, (usize, f64)> =
        std::collections::BTreeMap::new();
    for i in 0..defs.len() {
        if (va[i] as i64) == (vb[i] as i64) {
            continue;
        }
        let d = (vb[i] - va[i]).abs();
        n_changed += 1;
        max_delta = max_delta.max(d);
        sum_abs += d;
        // Group by the leading `_`-separated family so a singular-only or
        // LMR-only diff is readable at a glance.
        let group = defs[i].name.split('_').next().unwrap_or("?");
        let e = per_prefix.entry(group).or_insert((0, 0.0));
        e.0 += 1;
        e.1 = e.1.max(d);
        println!(
            "  {:<34} {} -> {} ({:+})",
            defs[i].name,
            fmt(va[i]),
            fmt(vb[i]),
            vb[i] - va[i]
        );
    }

    println!("compare-search: {a} vs {b}");
    if n_changed == 0 {
        println!(
            "identical: {} / {} numeric parameters match",
            defs.len(),
            defs.len()
        );
    } else {
        println!("group summary:");
        for (g, (n, mx)) in &per_prefix {
            println!("  {g:<14} changed {n:>4}  max |d| {mx:>6.1}");
        }
        println!(
            "changed: {n_changed} / {} numeric params  max |d| {max_delta:.1}  mean |d| {:.2}",
            defs.len(),
            sum_abs / n_changed as f64
        );
    }

    // The gate diff is a separate axis on purpose: two files can be
    // numerically identical and behave completely differently.
    let ga = morstilia::matchplay::gates_fingerprint(&pa.gates);
    let gb = morstilia::matchplay::gates_fingerprint(&pb.gates);
    if ga == gb {
        println!("gates: identical  {ga}");
    } else {
        println!("gates differ:");
        for name in SearchGates::NAMES {
            let (x, y) = (pa.gates.get(name), pb.gates.get(name));
            if x != y {
                println!("  {name:<22} {} -> {}", on_off(x), on_off(y));
            }
        }
    }
    Ok(())
}

/// `--search-list`: the tunable names, with bounds, so `--search-subset` is
/// discoverable from the binary itself.
pub fn list() {
    let defs = SearchParams::bounds();
    println!(
        "search parameters ({} numeric, {} gates):",
        defs.len(),
        SearchGates::NAMES.len()
    );
    for d in &defs {
        println!("  {:<34} [{}, {}]", d.name, fmt(d.min), fmt(d.max));
    }
    println!("gates (--search-gate name=bool):");
    for g in SearchGates::NAMES {
        println!("  {g}");
    }
}

fn on_off(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "on",
        Some(false) => "off",
        None => "unknown",
    }
}

fn fmt(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Parses `NAME=BOOL`. An unknown name is an error: a typo that silently left
/// a mechanism on is precisely the failure an A/B run cannot detect.
pub fn parse_gate(v: &str) -> anyhow::Result<(String, bool)> {
    let (name, on) = v
        .rsplit_once('=')
        .with_context(|| format!("--search-gate needs NAME=BOOL, got {v:?}"))?;
    let on = SearchGates::parse_gate_value(on)
        .with_context(|| format!("--search-gate {name}: {on:?} is not a boolean"))?;
    if !SearchGates::NAMES.contains(&name.trim()) {
        bail!(
            "unknown search gate {:?}; known gates: {}",
            name,
            SearchGates::NAMES.join(", ")
        );
    }
    Ok((name.trim().to_string(), on))
}

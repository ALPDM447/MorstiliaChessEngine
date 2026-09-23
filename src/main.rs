//! Morstilia binary.
//!
//! The default mode is a UCI loop over stdin — the standard way GUI
//! front-ends drive the engine. A few one-shot CLI tools cover manual and
//! scripted use:
//!
//! * `--perft <depth> [--fen <fen>]` — grouped perft node counts.
//! * `--fen <fen> --depth N [--threads N] [--hash MB]` — one-shot positional
//!   search; prints the same `info ...` / `bestmove ...` lines UCI would.
//! * `--bench [--depth N] [--threads N] [--hash MB]` — a fixed position suite
//!   with per-position node counts and an aggregate NPS figure.
//! * `--version`, `--help`.
//!
//! Arguments take the `--name value` or `--name=value` form.

use std::io::BufRead;

use anyhow::{Context, bail};

use morstilia::board::Position;
use morstilia::config::{DEFAULT_HASH_MB, DEFAULT_THREADS};
use morstilia::uci::UciEngine;

const USAGE: &str = "\
Morstilia — a strong classical UCI chess engine.

Usage:
  morstilia [--mode uci]                 UCI loop over stdin (default)
  morstilia --perft <depth> [--fen FEN]  perft node counts
  morstilia --fen FEN --depth N [--threads N] [--hash MB]   one-shot search
  morstilia --bench [--depth N] [--threads N] [--hash MB] [--stats]   fixed benchmark
  morstilia --version                    print engine identity
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("morstilia: {e:#}");
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
}

fn run(args: &[String]) -> anyhow::Result<()> {
    let mut i = 0;
    // `--mode uci` is kept for CLI compatibility with the previous Python
    // engine; UCI is also the default when no arguments are given.
    if args.first().map(String::as_str) == Some("--mode") {
        if !args.get(1).is_some_and(|m| m == "uci") {
            bail!("only --mode uci is supported");
        }
        i = 2;
    }
    match args.get(i).map(String::as_str) {
        None | Some("uci") => uci_loop(),
        Some("--version" | "-v") => {
            print_version();
            Ok(())
        }
        Some("--help" | "-h") => {
            print!("{USAGE}");
            Ok(())
        }
        Some("--perft") => cmd_perft(args, i),
        Some("--fen") => cmd_search(args, i),
        Some("--bench") => cmd_bench(args, i),
        Some(other) => bail!("unknown argument: {other}"),
    }
}

/// The default driver: read lines from stdin, dispatch, exit on `quit`/EOF.
fn uci_loop() -> anyhow::Result<()> {
    let mut engine = UciEngine::new();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line.context("failed to read stdin")?;
        if line.trim().is_empty() {
            continue;
        }
        if !engine.handle_line(&line) {
            break;
        }
    }
    engine.stop_and_join();
    Ok(())
}

/// `--perft <depth> [--fen FEN]`: prints one line per root move plus a total.
fn cmd_perft(args: &[String], i: usize) -> anyhow::Result<()> {
    let mut j = i + 1;
    let depth: u32 = if let Some(v) = flag_value(args, &mut j, "--depth") {
        v.parse().context("invalid perft depth")?
    } else if let Some(s) = args.get(j).filter(|s| s.parse::<u32>().is_ok()) {
        j += 1;
        s.parse().expect("checked")
    } else {
        bail!("--perft needs a depth (e.g. `--perft 5`)");
    };
    let mut fen = None;
    while j < args.len() {
        if let Some(v) = flag_value(args, &mut j, "--fen") {
            fen = Some(v);
        } else {
            bail!("unexpected argument: {}", args[j]);
        }
    }
    let pos = match fen {
        Some(f) => Position::from_fen(&f).with_context(|| format!("invalid FEN {f:?}"))?,
        None => Position::startpos(),
    };
    if depth == 0 {
        println!("1");
        return Ok(());
    }
    let mut total = 0u64;
    for (m, n) in pos.perft_split(depth) {
        println!("{}: {n}", m.to_uci());
        total += n;
    }
    println!("total: {total}");
    Ok(())
}

/// `--fen FEN --depth N`: run a real search and print its `info`/`bestmove`
/// output (exactly `depth` plies, no time cap). The opening book is
/// disabled so results are deterministic across machines.
fn cmd_search(args: &[String], i: usize) -> anyhow::Result<()> {
    let fen = args
        .get(i + 1)
        .filter(|f| !f.starts_with('-'))
        .context("--fen needs a FEN string, e.g. --fen \"r6k/... \"")?;
    let mut j = i + 2;
    let mut depth = DEFAULT_BENCH_DEPTH;
    let mut threads = DEFAULT_THREADS;
    let mut hash = DEFAULT_HASH_MB;
    while j < args.len() {
        let flag = args[j].clone();
        match flag_value(args, &mut j, &flag) {
            Some(v) if flag == "--depth" => {
                depth = v.parse().context("--depth must be an integer")?
            }
            Some(v) if flag == "--threads" => {
                threads = v.parse().context("--threads must be an integer")?
            }
            Some(v) if flag == "--hash" => {
                hash = v.parse().context("--hash must be an integer (MB)")?
            }
            _ => bail!("unexpected argument: {flag}"),
        }
    }
    let out = search_once(fen, depth, threads, hash)?;
    print!("{out}");
    Ok(())
}

/// Runs one search to exactly `depth` plies and returns its `info ...` /
/// `bestmove ...` transcript (the opening book is never consulted, so
/// results are deterministic across machines).
fn search_once(fen: &str, depth: i32, threads: usize, hash: usize) -> anyhow::Result<String> {
    Ok(search_once_with_result(fen, depth, threads, hash)?.0)
}

/// [`search_once`] plus the full [`SearchResult`] (instrumentation counters)
/// for the bench's `--stats` output.
///
/// A `UciEngine` is deliberately *not* used here: routing `go depth N`
/// through the engine inherits the 5s default UCI budget and quietly stops
/// short on hard positions, so `--depth N` could never reach depth N — a
/// reproducibility trap for the fixed-depth benchmark. This searches with a
/// pure depth limit and formats the transcript in the same shape UCI would,
/// so [`parse_info`] treats them identically.
fn search_once_with_result(
    fen: &str,
    depth: i32,
    threads: usize,
    hash: usize,
) -> anyhow::Result<(String, Option<morstilia::search::SearchResult>)> {
    use morstilia::search::{Searcher, TimeLimit};
    use morstilia::types::{is_mate, mate_plies};

    let pos = Position::from_fen(fen).context("invalid FEN")?;
    let limits = TimeLimit {
        depth: Some(depth),
        nodes: None,
        movetime_ms: None,
        soft_ms: 0,
        hard_ms: 0,
        infinite: true,
    };
    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut searcher = Searcher::new(hash);
    let r = searcher.search(&pos, &[], &limits, &stop, threads.max(1), &[]);

    let mut out = String::new();
    if r.is_none() {
        out.push_str("bestmove 0000\n");
        return Ok((out, Some(r)));
    }
    let score = if is_mate(r.score) {
        format!("mate {}", mate_plies(r.score))
    } else {
        format!("cp {}", r.score)
    };
    let pv: Vec<String> = r.pv.iter().map(|m| m.to_uci()).collect();
    out.push_str(&format!(
        "info depth {} score {} nodes {} nps {} time {} pv {}\n",
        r.depth,
        score,
        r.nodes,
        r.nps(),
        r.time_ms,
        pv.join(" ")
    ));
    out.push_str(&format!("bestmove {}\n", r.best.to_uci()));
    Ok((out, Some(r)))
}

/// `--bench [--depth N] [--threads N] [--hash MB] [--stats]`: searches a fixed
/// set of positions and prints per-position nodes/time/NPS plus an aggregate.
/// With `--stats` each line is followed by its instrumentation counters.
fn cmd_bench(args: &[String], i: usize) -> anyhow::Result<()> {
    let mut j = i + 1;
    let mut depth = DEFAULT_BENCH_DEPTH;
    let mut threads = DEFAULT_THREADS;
    let mut hash = DEFAULT_HASH_MB;
    let mut stats = false;
    while j < args.len() {
        let flag = args[j].clone();
        if flag == "--stats" {
            stats = true;
            j += 1;
            continue;
        }
        match flag_value(args, &mut j, &flag) {
            Some(v) if flag == "--depth" => {
                depth = v.parse().context("--depth must be an integer")?
            }
            Some(v) if flag == "--threads" => {
                threads = v.parse().context("--threads must be an integer")?
            }
            Some(v) if flag == "--hash" => {
                hash = v.parse().context("--hash must be an integer (MB)")?
            }
            _ => bail!("unexpected argument: {flag}"),
        }
    }
    let positions: &[(&str, &str)] = &[
        (
            "startpos",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        (
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ),
        (
            "middlegame",
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
        ),
        ("endgame", "8/8/4k3/3pN3/3P4/8/4K3/8 w - - 0 1"),
    ];
    println!("bench: depth {depth} threads {threads} hash {hash} MB");
    let mut tot_nodes = 0u64;
    let mut tot_time: u128 = 0;
    for (name, fen) in positions {
        let (output, result) = search_once_with_result(fen, depth, threads, hash)?;
        let (nodes, time_ms, best) = parse_info(&output);
        tot_nodes += nodes;
        tot_time += time_ms;
        let nps = if time_ms == 0 {
            0
        } else {
            nodes * 1000 / time_ms as u64
        };
        println!("{name:12} nodes {nodes:>10} time {time_ms:>6} ms nps {nps:>9} best {best}");
        if stats {
            if let Some(r) = result.as_ref().filter(|r| !r.is_none()) {
                let s = &r.stats;
                println!(
                    "  stats depth {} qnodes {} tt_probe {} tt_hit {:.1}% tt_cut {:.1}% \
                     beta_cut {} first_cut {:.1}% avg_moves {:.2} see {} prune {:.1}% tt_stores {}",
                    r.depth,
                    s.qsearch_nodes,
                    s.tt_probes,
                    s.tt_hit_pct(),
                    s.tt_cutoff_pct(),
                    s.beta_cutoffs,
                    s.first_move_cutoff_pct(),
                    s.avg_moves_until_cutoff(),
                    s.see_calls,
                    s.see_prune_pct(),
                    r.tt_stores,
                );
                println!(
                    "  prune null {}/{} lmr {}/{} fut {} rfp {} razor {}/{} probcut {}/{} \
                     pruned {} ebf {:.2}",
                    s.null_cutoffs,
                    s.null_probes,
                    s.lmr_researched,
                    s.lmr_reduced,
                    s.futility_pruned,
                    s.rfp_pruned,
                    s.razor_cutoffs,
                    s.razor_attempts,
                    s.probcut_cutoffs,
                    s.probcut_attempts,
                    s.total_pruned(),
                    r.ebf(),
                );
            }
        }
    }
    println!(
        "total     nodes {tot_nodes:>10} time {tot_time:>6} ms nps {}",
        if tot_time == 0 {
            0
        } else {
            tot_nodes * 1000 / tot_time as u64
        }
    );
    Ok(())
}

/// Pulls `nodes`, `time` and `bestmove` out of a captured search transcript
/// (parsing its own `info`/`bestmove` lines).
fn parse_info(output: &str) -> (u64, u128, String) {
    let mut nodes = 0u64;
    let mut time_ms = 0u128;
    let mut best = "0000".to_string();
    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("info ") {
            let tokens: Vec<&str> = rest.split_whitespace().collect();
            for pair in tokens.windows(2) {
                match pair[0] {
                    "nodes" => nodes = pair[1].parse().unwrap_or(nodes),
                    "time" => time_ms = pair[1].parse().unwrap_or(time_ms),
                    _ => {}
                }
            }
        } else if let Some(m) = line.strip_prefix("bestmove ") {
            best = m.to_string();
        }
    }
    (nodes, time_ms, best)
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

fn print_version() {
    println!("{}", env!("MORSTILIA_NAME"));
    println!("author {}", env!("MORSTILIA_ID_AUTHOR"));
}

/// Default `go depth` for the one-shot and bench commands.
const DEFAULT_BENCH_DEPTH: i32 = 10;

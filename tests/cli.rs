//! End-to-end tests: drive the compiled `morstilia` binary as a subprocess.
//!
//! These exercise the real CLI entry point (`main.rs`) — perft, one-shot
//! searches, the UCI loop and the stdout contract — rather than the library
//! internals (which the unit tests cover).

use std::io::Write;
use std::process::{Command, Stdio};

/// Path to the binary under test (set by Cargo for integration tests).
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_morstilia")
}

/// Runs the CLI with `args` and returns stdout, failing on a non-zero exit.
fn run(args: &[&str]) -> String {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("spawn engine");
    assert!(
        out.status.success(),
        "non-zero exit ({:?}): {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("stdout is utf-8")
}

/// Feeds `lines` to the engine's UCI stdin loop and returns stdout.
fn drive_uci(lines: &[&str]) -> String {
    let mut child = Command::new(bin())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");
    {
        let mut stdin = child.stdin.take().unwrap();
        for l in lines {
            writeln!(stdin, "{l}").unwrap();
        }
        // Dropping stdin sends EOF, ending the UCI loop.
    }
    let out = child.wait_with_output().expect("wait for engine");
    assert!(
        out.status.success(),
        "non-zero exit ({:?}): {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("stdout is utf-8")
}

/// The `bestmove <m>` value from a transcript.
fn bestmove(out: &str) -> String {
    out.lines()
        .find_map(|l| l.strip_prefix("bestmove "))
        .map(str::to_string)
        .unwrap_or_else(|| panic!("no bestmove line in: {out}"))
}

/// Drops the non-deterministic `time <ms>` / `nps <x>` tokens so two runs of
/// the same deterministic search can be compared byte-for-byte.
fn normalized(out: &str) -> String {
    out.lines()
        .map(|l| {
            let mut kept = Vec::new();
            let mut it = l.split_whitespace().peekable();
            while let Some(t) = it.next() {
                if (t == "time" || t == "nps")
                    && it.peek().is_some_and(|v| v.parse::<u64>().is_ok())
                {
                    let _ = it.next();
                    continue;
                }
                kept.push(t);
            }
            kept.join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// --- CLI ---------------------------------------------------------------

#[test]
fn perft_startpos_via_cli_matches_known_values() {
    for (depth, expected) in [(1u32, 20u64), (2, 400), (3, 8902), (4, 197_281)] {
        let out = run(&["--perft", &depth.to_string()]);
        assert!(
            out.ends_with(&format!("total: {expected}\n")),
            "perft({depth}) got: {out}"
        );
    }
}

#[test]
fn perft_kiwipete_via_cli_matches_known_values() {
    let out = run(&[
        "--perft",
        "3",
        "--fen",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]);
    assert!(out.ends_with("total: 97862\n"), "got: {out}");
}

#[test]
fn bench_stats_flag_prints_instrumentation() {
    // `--bench --stats` must surface the aggregate counters per position —
    // the smoke check for the whole instrumentation pipeline, from counters
    // in the search up through aggregation and formatting. This includes the
    // Stage-3 pruning statistics (null move, LMR, futility, razor, ProbCut,
    // total pruned and the effective branching factor).
    let out = run(&["--bench", "--depth", "3", "--stats"]);
    assert!(
        out.contains("stats "),
        "per-position stats block missing: {out}"
    );
    for key in [
        "tt_probe",
        "tt_stores",
        "first_cut",
        "beta_cut",
        "see ",
        "null ",
        "lmr ",
        "fut ",
        "rfp ",
        "razor ",
        "probcut ",
        "pruned ",
        "ebf ",
    ] {
        assert!(out.contains(key), "stats must include {key:?}: {out}");
    }
    assert!(
        out.contains("best "),
        "position lines must still print: {out}"
    );
}

#[test]
fn one_shot_search_reports_depth_and_bestmove() {
    let out = run(&[
        "--fen",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "--depth",
        "4",
    ]);
    assert!(out.contains("info depth 4"), "got: {out}");
    let best = bestmove(&out);
    assert!(best.len() >= 4 && best != "0000", "bad bestmove {best:?}");
}

#[test]
fn finds_mate_in_two() {
    let out = run(&[
        "--fen",
        "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1",
        "--depth",
        "6",
    ]);
    assert!(out.contains("score mate"), "expected a mate score: {out}");
    let best = bestmove(&out);
    // 1. Rd8+! Rxd8 2. Rxd8# — either rook to d8 mates in two.
    assert!(best.ends_with("d8"), "expected Rd8+, got {best}");
}

#[test]
fn threads_one_search_is_deterministic() {
    let args = [
        "--fen",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "--depth",
        "4",
        "--threads",
        "1",
    ];
    let a = run(&args);
    let b = run(&args);
    assert_eq!(
        normalized(&a),
        normalized(&b),
        "Threads=1 must be deterministic"
    );
    assert!(a.contains("bestmove "), "got: {a}");
}

#[test]
fn multithreaded_search_still_returns_a_move() {
    let out = run(&[
        "--fen",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "--depth",
        "4",
        "--threads",
        "2",
    ]);
    let best = bestmove(&out);
    assert!(best.len() >= 4 && best != "0000", "bad bestmove {best:?}");
}

#[test]
fn version_prints_identity() {
    let out = run(&["--version"]);
    assert!(out.starts_with("Morstilia "), "got: {out}");
    assert!(out.contains("author"), "got: {out}");
}

#[test]
fn bad_invocation_is_an_error() {
    let out = Command::new(bin())
        .args(["--bogus"])
        .output()
        .expect("spawn engine");
    assert_eq!(out.status.code(), Some(2));
}

// --- UCI protocol over a subprocess -------------------------------------

#[test]
fn uci_handshake_produces_clean_stdout() {
    let out = drive_uci(&["uci", "isready", "position startpos", "go depth 2", "quit"]);
    let lines: Vec<&str> = out.lines().collect();
    assert!(lines.iter().any(|l| *l == "uciok"), "no uciok: {out}");
    assert!(lines.iter().any(|l| *l == "readyok"), "no readyok: {out}");
    assert!(
        lines.iter().any(|l| l.starts_with("bestmove ")),
        "no bestmove: {out}"
    );

    // The UCI stdout contract: only these line kinds may ever appear.
    for l in &lines {
        assert!(
            l.starts_with("id ")
                || l.starts_with("option ")
                || *l == "uciok"
                || *l == "readyok"
                || l.starts_with("info ")
                || l.starts_with("bestmove "),
            "leaked line: {l:?}"
        );
    }
}

#[test]
fn go_infinite_stops_with_a_bestmove() {
    let out = drive_uci(&[
        "position startpos",
        "go infinite",
        "stop",
        "isready",
        "quit",
    ]);
    assert!(
        out.lines().any(|l| l.starts_with("bestmove ")),
        "expected a bestmove after stop: {out}"
    );
    assert!(
        out.lines().any(|l| l == "readyok"),
        "expected readyok after stop: {out}"
    );
}

#[test]
fn invalid_moves_do_not_leak_to_stdout() {
    let out = drive_uci(&[
        "position startpos moves e2e5", // illegal: silently ignored
        "go depth 2",
        "quit",
    ]);
    // The engine still searches the untouched start position and answers.
    assert!(
        out.lines().any(|l| l.starts_with("bestmove ")),
        "got: {out}"
    );
}

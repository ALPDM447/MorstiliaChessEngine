//! End-to-end tests for the engine-vs-engine match driver (Stage 8).
//!
//! These drive the compiled `morstilia-selfplay` binary as a subprocess:
//! deterministic small matches, JSON match reports, resume continuation,
//! baseline freezing and A-vs-B comparison — the same commands documented for
//! the real baseline runs.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_morstilia-selfplay")
}

/// Runs the CLI with `args`, asserting a zero exit, and returns stdout.
fn run(args: &[&str]) -> String {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("spawn match driver");
    assert!(
        out.status.success(),
        "non-zero exit ({:?}): {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("stdout is utf-8")
}

/// A per-process temp file so parallel tests never collide.
fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("morstilia_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir.join(name)
}

fn parse(path: &PathBuf) -> serde_json::Value {
    let s = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("read {}", path.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

#[test]
fn small_deterministic_match_writes_a_report() {
    let rep = tmp("match.json");
    let out = run(&[
        "--games",
        "2",
        "--depth",
        "2",
        "--seed",
        "42",
        "--report",
        rep.to_str().unwrap(),
    ]);
    assert!(out.contains("games: 2 completed of 2"), "summary: {out}");

    let r = parse(&rep);
    assert_eq!(r["format"], "morstilia-match-report/v1");
    assert_eq!(r["games_completed"], 2);
    assert_eq!(r["games_requested"], 2);
    assert_eq!(r["seed"], 42);
    assert_eq!(r["time_control"], "depth 2");
    assert_eq!(r["tc_deterministic"], true);
    assert_eq!(r["suite"], "classical-v1");
    assert_eq!(r["suite_entries"], 48);
    assert_eq!(r["candidate"], "white");
    assert_eq!(r["alternate_colors"], true);
    assert_ne!(r["source"]["source_fingerprint"].as_u64().unwrap(), 0);
    assert!(r["source"]["profile"].is_string());

    // W/D/L must exactly account for every completed game.
    let wdl = &r["wdl"];
    let n = wdl["wins"].as_u64().unwrap()
        + wdl["draws"].as_u64().unwrap()
        + wdl["losses"].as_u64().unwrap();
    assert_eq!(n, r["games_completed"].as_u64().unwrap());
    // Score rate consistent with the W/D/L.
    let s = r["score_rate"].as_f64().unwrap();
    let expect = (wdl["wins"].as_f64().unwrap() + 0.5 * wdl["draws"].as_f64().unwrap()) / n as f64;
    assert!((s - expect).abs() < 1e-9);

    // Every game is a legal, opening-prefixed move list with an outcome.
    let games = r["games"].as_array().unwrap();
    assert_eq!(games.len(), 2);
    for g in games {
        assert!(
            g["moves"].as_array().unwrap().len() >= 2,
            "opening plies recorded"
        );
        assert!(["1-0", "0-1", "1/2-1/2"].contains(&g["outcome"].as_str().unwrap()));
    }
    assert!(r["terminations"].as_array().unwrap().len() >= 1);
    assert!(r["avg_plies"].as_f64().unwrap() > 0.0);
}

#[test]
fn same_seed_replays_the_identical_match() {
    let a = tmp("replay_a.json");
    let b = tmp("replay_b.json");
    run(&[
        "--games",
        "4",
        "--depth",
        "2",
        "--seed",
        "9",
        "--report",
        a.to_str().unwrap(),
    ]);
    run(&[
        "--games",
        "4",
        "--depth",
        "2",
        "--seed",
        "9",
        "--report",
        b.to_str().unwrap(),
    ]);
    let sa = std::fs::read_to_string(&a).unwrap();
    let sb = std::fs::read_to_string(&b).unwrap();
    assert_eq!(sa, sb, "same seed + config must replay byte-identically");
}

#[test]
fn resume_extends_a_match_identically() {
    let r1 = tmp("resume_r1.json");
    let r2 = tmp("resume_r2.json");
    run(&[
        "--games",
        "2",
        "--depth",
        "2",
        "--seed",
        "7",
        "--report",
        r1.to_str().unwrap(),
    ]);
    run(&[
        "--games",
        "4",
        "--depth",
        "2",
        "--seed",
        "7",
        "--resume",
        r1.to_str().unwrap(),
        "--report",
        r2.to_str().unwrap(),
    ]);
    let a = parse(&r1);
    let b = parse(&r2);
    assert_eq!(a["games_completed"], 2);
    assert_eq!(b["games_completed"], 4);
    // The continued report's first games are byte-identical to the source.
    let first_two = b["games"].as_array().unwrap();
    assert_eq!(a["games"].as_array().unwrap(), &first_two[..2]);
}

#[test]
fn resume_rejects_an_incompatible_config() {
    let r1 = tmp("resume_x.json");
    run(&[
        "--games",
        "2",
        "--depth",
        "2",
        "--seed",
        "5",
        "--report",
        r1.to_str().unwrap(),
    ]);
    // Different seed: the resumed match would silently change meaning.
    let out = Command::new(bin())
        .args([
            "--games",
            "4",
            "--depth",
            "2",
            "--seed",
            "6",
            "--resume",
            r1.to_str().unwrap(),
            "--report",
            tmp("resume_bad.json").to_str().unwrap(),
        ])
        .output()
        .expect("spawn match driver");
    assert!(!out.status.success(), "incompatible resume must be refused");
}

#[test]
fn record_baseline_stamps_the_frozen_tag() {
    let rep = tmp("baseline.json");
    let out = run(&[
        "--games",
        "1",
        "--depth",
        "2",
        "--seed",
        "3",
        "--record-baseline",
        rep.to_str().unwrap(),
    ]);
    assert!(
        out.contains("Stage8-Classical-SMP-Baseline"),
        "print: {out}"
    );
    let r = parse(&rep);
    assert_eq!(r["baseline"], "Stage8-Classical-SMP-Baseline");
    assert_eq!(r["games_completed"], 1);
    // The frozen report still carries the full reproducible configuration.
    assert_eq!(r["seed"], 3);
    assert_eq!(r["time_control"], "depth 2");
    assert_eq!(r["suite"], "classical-v1");
}

/// A bare boolean flag (`--sprt`) used to consume the *next* token through
/// `flag_value`, swallowing the following option (`--sprt --report x` →
/// `unknown argument: x`). This is the end-to-end guard for that regression.
#[test]
fn bare_sprt_flag_does_not_swallow_the_next_option() {
    let rep = tmp("sprt.json");
    let out = run(&[
        "--games",
        "2",
        "--depth",
        "2",
        "--seed",
        "13",
        "--sprt",
        "--report",
        rep.to_str().unwrap(),
    ]);
    assert!(out.contains("sprt:"), "SPRT status printed: {out}");
    let r = parse(&rep);
    let n = r["games_completed"].as_u64().unwrap();
    assert!(n >= 1 && n <= 2, "plays up to the cap, never more: {n}");
    assert!(r["sprt"].is_object(), "report carries the SPRT block: {r}");
}

#[test]
fn compare_mode_prints_a_machine_readable_verdict() {
    let a = tmp("cmp_a.json");
    let b = tmp("cmp_b.json");
    run(&[
        "--games",
        "8",
        "--depth",
        "2",
        "--seed",
        "11",
        "--report",
        a.to_str().unwrap(),
    ]);
    run(&[
        "--games",
        "8",
        "--depth",
        "2",
        "--seed",
        "11",
        "--report",
        b.to_str().unwrap(),
    ]);
    let out = run(&["--compare", a.to_str().unwrap(), b.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("compare prints JSON");
    // Two identical runs trivially regress neither way.
    assert_eq!(v["verdict"], "similar");
    assert_eq!(v["score_delta"].as_f64().unwrap(), 0.0);
    assert_eq!(v["a_games"], 8);
    assert_eq!(v["b_games"], 8);
    assert_eq!(v["comparable"], true);
}

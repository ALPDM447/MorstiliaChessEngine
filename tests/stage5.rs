//! Stage-5 regression suite: parameter-file parity, the dataset pipeline
//! (FEN/PGN loading, deterministic shuffle and train/test split, dataset
//! statistics), the parameter-comparison tool and evaluation colour
//! symmetry under any loaded parameter set.
//!
//! An optional *candidate* parameter set (e.g. a fresh SPSA export) can be
//! checked by exporting `MORSTILIA_EVAL_PARAMS=/path/to/tuned.toml` before
//! running `cargo test`; without it, only the shipped baseline is tested.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use morstilia::evaluation::{EvalParams, Evaluator};
use morstilia::tuning::dataset::DataSet;

fn manifest(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Mirrors a FEN across the board diagonal (colour-flip symmetry):
/// every piece changes colour, squares swap ranks, the side to move and
/// the castling rights flip. White-relative evaluations must be unchanged.
fn mirror_fen(fen: &str) -> String {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    assert!(fields.len() >= 4, "not a FEN: {fen}");
    // Board: reverse rank order, swap piece case.
    let ranks: Vec<&str> = fields[0].split('/').collect();
    assert_eq!(ranks.len(), 8, "bad board part in {fen}");
    let flipped: Vec<String> = ranks
        .iter()
        .rev()
        .map(|r| {
            r.chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else if c.is_ascii_lowercase() {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    }
                })
                .collect::<String>()
        })
        .collect();
    let side = if fields[1] == "w" { "b" } else { "w" };
    let castle: String = fields[2]
        .chars()
        .map(|c| {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else {
                c
            }
        })
        .collect();
    let ep = if fields[3] == "-" {
        "-".to_string()
    } else {
        let mut chars = fields[3].chars();
        let file = chars.next().expect("ep file");
        let rank = chars.next().expect("ep rank");
        let mirrored = match rank {
            '3' => '6',
            '6' => '3',
            other => other,
        };
        format!("{file}{mirrored}")
    };
    format!(
        "{} {} {} {} {}",
        flipped.join("/"),
        side,
        castle,
        ep,
        fields.get(4).copied().unwrap_or("0")
    )
}

/// The FEN battery every parameter set is checked against (tactics,
/// symmetry, endgames, opening positions).
const SYMMETRY_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/8/8/8/8/2k5/2P5/2K5 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/8/8/8/8/2k5/3R4/2K5 w - - 0 1",
    "r2q1rk1/1p2bppp/p2pbn2/4p3/4P3/1PN1BP2/PQPP2PP/R3KB1R w KQ - 0 1",
];

/// Asserts colour symmetry (white-relative eval is mirror-invariant) and
/// round-trip TOML parity for one parameter set.
fn assert_params_sane(params: &EvalParams, label: &str) {
    let ev = Evaluator;
    for fen in SYMMETRY_FENS {
        let pos = morstilia::Position::from_fen(fen).expect("battery FEN");
        let mirrored = morstilia::Position::from_fen(&mirror_fen(fen)).expect("mirror FEN");
        // Side-to-move-relative evaluation is invariant under a colour-flip
        // mirror (both the position and the point of view flip sign).
        let a = ev.evaluate_with(&pos, params);
        let b = ev.evaluate_with(&mirrored, params);
        assert_eq!(
            a, b,
            "{label}: stm-relative eval not colour-symmetric for {fen:?}"
        );
        // White-relative evaluation negates: whatever White had, Black has
        // after the flip.
        let wa = ev.evaluate_white_with(&pos, params);
        let wb = ev.evaluate_white_with(&mirrored, params);
        assert_eq!(
            wa, -wb,
            "{label}: white-relative eval not mirror-antisymmetric for {fen:?}"
        );
    }
    // TOML round-trip must be lossless (integer parameters only).
    let text = params.to_toml_string().expect("serialize params");
    let back = EvalParams::from_toml_str(&text).expect("deserialize params");
    assert_eq!(
        params.to_vec(),
        back.to_vec(),
        "{label}: TOML round-trip changed parameters"
    );
}

#[test]
fn baseline_toml_matches_builtin_defaults() {
    let path = manifest("config/baseline_eval.toml");
    let loaded = EvalParams::load(path.to_str().expect("utf-8 path"))
        .expect("config/baseline_eval.toml must load");
    let a = loaded.to_vec();
    let b = EvalParams::default().to_vec();
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x, y,
            "shipped baseline differs from built-in default at parameter #{i}"
        );
    }
    assert_params_sane(&loaded, "baseline");
}

#[test]
fn fen_file_supports_both_line_orders_and_targets() {
    let path = manifest("tests/data/eval_positions.txt");
    let ds = DataSet::from_fen_file(path.to_str().expect("utf-8 path"))
        .expect("sample position file must load");
    assert_eq!(ds.len(), 8, "sample file has 8 positions");
    assert!(ds.source.starts_with("fen "));
    // result-first lines: 0.5, 1, 0, 0.5, 0.75
    assert_eq!(ds.entries[0].result, 0.5);
    assert_eq!(ds.entries[1].result, 1.0);
    assert_eq!(ds.entries[2].result, 0.0);
    assert_eq!(ds.entries[3].result, 0.5);
    assert_eq!(ds.entries[4].result, 0.75, "fractional target scores work");
    // FEN-first lines: 1.0, 0.5, PGN '1-0'
    assert_eq!(ds.entries[5].result, 1.0);
    assert_eq!(ds.entries[6].result, 0.5);
    assert_eq!(ds.entries[7].result, 1.0);
    // The same position written in both orders parses identically.
    assert_eq!(
        ds.entries[1].pos.fen(),
        ds.entries[7].pos.fen(),
        "result-first and FEN-first forms must agree"
    );
}

#[test]
fn fen_file_rejects_garbage_loudly() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("morstilia_stage5_bad_{}.txt", std::process::id()));
    fs::write(&path, "0.5 not-a-fen w - - 0 1\n").expect("write temp file");
    let err = DataSet::from_fen_file(path.to_str().expect("utf-8 path"));
    assert!(err.is_err(), "invalid FEN must be an error");
    let _ = fs::remove_file(&path);
}

#[test]
fn pgn_file_replays_every_finished_game() {
    let path = manifest("tests/data/sample.pgn");
    let ds =
        DataSet::from_pgn_file(path.to_str().expect("utf-8 path")).expect("sample PGN must load");
    assert_eq!(
        ds.source,
        format!("pgn {} (3 games, 0 skipped, 0 truncated)", path.display()),
        "every SAN of the sample must replay without truncation"
    );
    // Game 1: 7 plies → 8 positions. Game 2: 8 plies → 9. Game 3 (FEN
    // header): 4 plies → 5.
    assert_eq!(ds.len(), 8 + 9 + 5, "positions per replayed game");
    let s = ds.stats();
    assert_eq!(s.white_wins, 8, "game 1 targets (1-0)");
    assert_eq!(s.draws, 5, "game 3 targets (1/2-1/2)");
    assert_eq!(s.black_wins, 9, "game 2 targets (0-1)");
    // The last position of game 1 is the mate after 4.Qxf7#.
    assert!(ds.entries[7].pos.is_mated(), "game 1 must end in checkmate");
    // Game 3 starts from its [FEN] header, not from the initial position.
    let expect =
        morstilia::Position::from_fen("8/8/8/8/8/2k5/2P5/2K5 w - - 0 1").expect("setup FEN");
    assert_eq!(
        ds.entries[17].pos.fen(),
        expect.fen(),
        "[FEN] header honoured"
    );
}

#[test]
fn train_test_split_is_deterministic_and_partitions_the_data() {
    let path = manifest("tests/data/eval_positions.txt");
    let ds = DataSet::from_fen_file(path.to_str().expect("utf-8 path")).expect("load");
    let (tr1, te1) = ds.train_test_split(0.5, 7);
    let (tr2, te2) = ds.train_test_split(0.5, 7);
    assert_eq!(tr1.len() + te1.len(), ds.len());
    assert_eq!(tr1.len(), 4, "half of 8 entries");
    assert_eq!(te1.len(), 4);
    // The fraction is the *test* share: 0.25 → 6 train / 2 test.
    let (trq, teq) = ds.train_test_split(0.25, 7);
    assert_eq!(trq.len(), 6, "train gets 1 - fraction");
    assert_eq!(teq.len(), 2, "test gets exactly the fraction");
    assert_eq!(trq.len() + teq.len(), ds.len());
    // Same seed → identical partition.
    assert_eq!(
        tr1.entries.iter().map(|e| e.pos.fen()).collect::<Vec<_>>(),
        tr2.entries.iter().map(|e| e.pos.fen()).collect::<Vec<_>>()
    );
    assert_eq!(
        te1.entries.iter().map(|e| e.pos.fen()).collect::<Vec<_>>(),
        te2.entries.iter().map(|e| e.pos.fen()).collect::<Vec<_>>()
    );
    // Train ∪ test == the whole dataset (as a multiset), i.e. nothing is
    // lost or duplicated across the split.
    let mut all: Vec<String> = ds.entries.iter().map(|e| e.pos.fen()).collect();
    let mut joined: Vec<String> = tr1
        .entries
        .iter()
        .chain(te1.entries.iter())
        .map(|e| e.pos.fen())
        .collect();
    all.sort();
    joined.sort();
    assert_eq!(all, joined, "split must partition the dataset");
    assert!(tr1.source.contains("train") && te1.source.contains("test"));
}

#[test]
fn shuffle_is_reproducible_from_the_seed() {
    let path = manifest("tests/data/eval_positions.txt");
    let mut a = DataSet::from_fen_file(path.to_str().expect("utf-8 path")).expect("load");
    let mut b = a.clone();
    use morstilia::book::SplitMix64;
    a.shuffle_with(&mut SplitMix64(99));
    b.shuffle_with(&mut SplitMix64(99));
    let fa: Vec<String> = a.entries.iter().map(|e| e.pos.fen()).collect();
    let fb: Vec<String> = b.entries.iter().map(|e| e.pos.fen()).collect();
    assert_eq!(fa, fb);
    // Same multiset as unshuffled.
    let mut orig: Vec<String> = DataSet::from_fen_file(
        manifest("tests/data/eval_positions.txt")
            .to_str()
            .expect("utf-8 path"),
    )
    .expect("load")
    .entries
    .iter()
    .map(|e| e.pos.fen())
    .collect();
    let mut shuf = fa.clone();
    orig.sort();
    shuf.sort();
    assert_eq!(orig, shuf);
}

#[test]
fn dataset_stats_report_balanced_targets() {
    let path = manifest("tests/data/eval_positions.txt");
    let ds = DataSet::from_fen_file(path.to_str().expect("utf-8 path")).expect("load");
    let s = ds.stats();
    assert_eq!(s.entries, 8);
    assert_eq!(s.white_wins, 4, "results > 0.5");
    assert_eq!(s.draws, 3, "results == 0.5");
    assert_eq!(s.black_wins, 1, "results < 0.5");
    assert!((s.avg_result - 5.25 / 8.0).abs() < 1e-9, "{}", s.avg_result);
    assert!((0.0..=24.0).contains(&s.avg_phase));
    assert!(s.avg_material_cp >= 0.0);
    let line = s.to_string();
    assert!(!line.contains('\n') && line.contains("8 entries"), "{line}");
}

/// Runs `tune --compare A B` and returns its stdout.
fn tune_compare(a: &str, b: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_tune"))
        .args(["--compare", a, b])
        .output()
        .expect("spawn tune");
    assert!(
        out.status.success(),
        "tune --compare failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

#[test]
fn compare_cli_identifies_equal_and_differing_parameter_files() {
    let base = manifest("config/baseline_eval.toml");
    let base_str = base.to_str().expect("utf-8 path");

    // Identical files.
    let same = tune_compare(base_str, base_str);
    let n = EvalParams::default().param_count();
    assert!(
        same.contains(&format!("identical: {n} / {n} parameters match")),
        "got: {same}"
    );

    // Exactly one changed parameter (material.pawn 100 → 101).
    let text = fs::read_to_string(&base).expect("read baseline");
    let modified = text
        .replace("piece_values = [100,", "piece_values = [101,")
        .replace("piece_values=[100,", "piece_values=[101,");
    assert_ne!(modified, text, "baseline must contain piece_values = [100,");
    let path =
        std::env::temp_dir().join(format!("morstilia_stage5_cmp_{}.toml", std::process::id()));
    fs::write(&path, modified).expect("write modified params");
    let diff = tune_compare(base_str, path.to_str().expect("utf-8 path"));
    let _ = fs::remove_file(&path);
    assert!(diff.contains("material.pawn"), "got: {diff}");
    assert!(diff.contains("100 -> 101"), "got: {diff}");
    assert!(diff.contains("changed: 1 / "), "got: {diff}");
    assert!(diff.contains("material"), "group summary missing: {diff}");
}

/// Every parameter set that opts in (baseline by default, plus a candidate
/// via `MORSTILIA_EVAL_PARAMS`) must keep colour symmetry and lossless
/// TOML round-trips — the §7 symmetry regression for tuned candidates.
#[test]
fn candidate_parameters_keep_symmetry_and_roundtrip() {
    assert_params_sane(&EvalParams::default(), "built-in baseline");
    if let Ok(path) = std::env::var("MORSTILIA_EVAL_PARAMS") {
        let cand = EvalParams::load(&path)
            .unwrap_or_else(|e| panic!("MORSTILIA_EVAL_PARAMS={path:?} failed to load: {e:#}"));
        assert_params_sane(&cand, &format!("candidate {path}"));
    }
}

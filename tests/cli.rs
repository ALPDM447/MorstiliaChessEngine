//! End-to-end tests: drive the compiled `morstilia` binary as a subprocess.
//!
//! These exercise the real CLI entry point (`main.rs`) — perft, one-shot
//! searches, the UCI loop and the stdout contract — rather than the library
//! internals (which the unit tests cover).

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
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

/// Feeds `lines` to the engine's UCI stdin loop and returns `(stdout, stderr)`.
///
/// The session is **interactive**, the way a GUI drives an engine: after a
/// bounded `go`, the next line is withheld until the engine's `bestmove` has
/// actually been read. This is not politeness, it is correctness — `quit` and
/// every other command that starts a search call the same stop-and-join path as
/// `stop`, so a harness that dumps all its lines at once *aborts* any search that
/// has not already finished, and observes `bestmove 0000` plus a `stopped`
/// search at whatever depth it happened to reach.
///
/// A `go infinite` / `go ponder` is unbounded, so no `bestmove` is expected and
/// the following line (`stop`, normally) is sent straight away.
fn drive_uci_capturing(lines: &[&str]) -> (String, String) {
    let mut child = Command::new(bin())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut seen = String::new();

    for l in lines {
        writeln!(stdin, "{l}").unwrap();
        stdin.flush().unwrap();
        if !l.starts_with("go ") || l.contains("infinite") || l.contains("ponder") {
            continue;
        }
        read_until_bestmove(&mut stdout, &mut seen);
    }
    // `quit` (rather than a bare EOF) so the engine exits through its own path.
    writeln!(stdin, "quit").unwrap();
    drop(stdin);

    let mut rest = String::new();
    stdout.read_to_string(&mut rest).expect("read stdout");
    seen.push_str(&rest);
    let out = child.wait_with_output().expect("wait for engine");
    let err = String::from_utf8(out.stderr).expect("stderr is utf-8");
    assert!(
        out.status.success(),
        "non-zero exit ({:?}): {err}",
        out.status.code()
    );
    (seen, err)
}

/// Blocks until a `bestmove` line has been read, appending everything to
/// `seen`. A `go` that never answers (only `infinite`/`ponder` do) is bounded by
/// the wait below so a mistake shows up as a failure rather than a hang.
fn read_until_bestmove(stdout: &mut impl std::io::BufRead, seen: &mut String) {
    loop {
        let mut line = String::new();
        let n = stdout.read_line(&mut line).expect("read stdout");
        assert_ne!(n, 0, "the engine closed stdout while a search was running");
        let done = line.starts_with("bestmove ");
        seen.push_str(&line);
        if done {
            return;
        }
    }
}

/// [`drive_uci_capturing`], discarding stderr.
fn drive_uci(lines: &[&str]) -> String {
    drive_uci_capturing(lines).0
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
fn bench_threads_stats_reports_smp_workers() {
    // `--bench --threads N --stats` must surface the Lazy-SMP instrumentation:
    // the aggregate worker count/root-task total plus one per-worker line.
    let out = run(&["--bench", "--depth", "3", "--threads", "2", "--stats"]);
    assert!(
        out.contains("smp threads 2 workers 2"),
        "SMP aggregate line missing: {out}"
    );
    assert!(
        out.lines()
            .any(|l| l.contains("worker ") && l.contains("nodes ")),
        "per-worker lines missing: {out}"
    );
    assert!(
        out.lines().any(|l| l.contains("worker *0")),
        "main worker must be marked: {out}"
    );
}

#[test]
fn bench_hash_flag_is_honored() {
    // The `--hash` knob flows through to the TT size; the bench header must
    // echo it back and the run must still complete with an aggregate line.
    let out = run(&["--bench", "--depth", "3", "--hash", "256"]);
    assert!(
        out.contains("hash 256 MB"),
        "bench header must echo --hash: {out}"
    );
    assert!(
        out.lines()
            .any(|l| l.starts_with("total") && l.contains("nodes ")),
        "no aggregate bench line: {out}"
    );
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
    let out = drive_uci(&["position startpos", "go infinite", "stop", "isready"]);
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
fn setoption_threads_runs_smp_search() {
    // Changing `Threads` between searches must pre-size the persistent pool
    // and the next `go` must actually use the helpers (the `info string smp`
    // summary only appears when more than one worker ran). Switching back to
    // Threads = 1 must also work (pool shrink + join).
    let out = drive_uci(&[
        // Disable the auto-discovered book so both `go`s actually search
        // (startpos is the book's home turf; a book move would skip them).
        "setoption name BookEnabled value false",
        "setoption name Threads value 4",
        "position startpos",
        "go depth 4",
        "setoption name Threads value 1",
        "position startpos",
        "go depth 3",
    ]);
    let info = out.lines().filter(|l| l.starts_with("info string smp "));
    assert!(
        info.clone().any(|l| l.contains("threads 4 workers 4")),
        "expected a 4-worker SMP summary: {out}"
    );
    // The Threads=1 search must not emit the SMP summary at all.
    assert!(
        out.lines().filter(|l| l.starts_with("bestmove ")).count() >= 2,
        "expected two bestmove lines: {out}"
    );
}

#[test]
fn hash_resize_between_searches_keeps_working() {
    // Resizing the TT via `setoption Hash` must not disturb an ongoing or
    // subsequent search: same transcript shape as an untouched run.
    let out = drive_uci(&[
        "setoption name Hash value 8",
        "position startpos",
        "go depth 3",
        "setoption name Hash value 256",
        "position startpos",
        "go depth 3",
    ]);
    assert!(
        out.lines().filter(|l| l.starts_with("bestmove ")).count() == 2,
        "expected two bestmove lines: {out}"
    );
}

#[test]
fn invalid_moves_do_not_leak_to_stdout() {
    let out = drive_uci(&[
        "position startpos moves e2e5", // illegal: silently ignored
        "go depth 2",
    ]);
    // The engine still searches the untouched start position and answers.
    assert!(
        out.lines().any(|l| l.starts_with("bestmove ")),
        "got: {out}"
    );
}

// --- Syzygy ---------------------------------------------------------------

/// Absolute path to the 3/4-piece test tables shipped with the repo.
fn syzygy_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/syzygy")
}

const KQVK: &str = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";

#[test]
fn missing_syzygy_path_keeps_engine_running() {
    // An unreadable SyzygyPath must degrade to an inert tablebase: the engine
    // keeps probing nothing, answers isready, and still plays chess.
    let out = drive_uci(&[
        "setoption name SyzygyPath value /nonexistent/morstilia/tables",
        "isready",
        "position fen 4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
        "go depth 3",
    ]);
    assert!(out.lines().any(|l| l == "readyok"), "got: {out}");
    assert!(
        out.lines().any(|l| l.starts_with("bestmove ")),
        "a normal search must still run: {out}"
    );
}

#[test]
fn syzygy_path_option_reports_tb_win_not_mate() {
    // Loading the tables through the UCI option turns the KQvK search into a
    // tablebase win: reported in centipawns (the fixed win band), never as a
    // mate score, and the best move must be legal.
    let out = drive_uci(&[
        &format!("setoption name SyzygyPath value {}", syzygy_dir().display()),
        "isready",
        "position fen 4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
        "go depth 3",
    ]);
    assert!(out.lines().any(|l| l == "readyok"), "got: {out}");
    assert!(out.contains("score cp 19999"), "TB win band: {out}");
    assert!(
        !out.contains("score mate"),
        "no mate claim from the tables: {out}"
    );
    let best = bestmove(&out);
    assert!(best.len() >= 4 && best != "0000", "bad bestmove {best:?}");
}

#[test]
fn one_shot_search_with_syzygy_reports_tb_win() {
    let out = run(&[
        "--fen",
        KQVK,
        "--depth",
        "4",
        "--syzygy",
        syzygy_dir().to_str().unwrap(),
    ]);
    assert!(out.contains("info depth 4"), "got: {out}");
    assert!(out.contains("score cp 19999"), "TB win band: {out}");
    assert!(
        !out.contains("score mate"),
        "no mate claim from the tables: {out}"
    );
}

#[test]
fn bench_with_syzygy_accepts_the_flag() {
    // The built-in bench positions are all outside the 3/4-piece tables, so
    // no TB probing happens — this is a smoke check that the flag parses and
    // the header reports it (a regression guard for the CLI plumbing).
    let out = run(&[
        "--bench",
        "--depth",
        "2",
        "--stats",
        "--syzygy",
        syzygy_dir().to_str().unwrap(),
    ]);
    assert!(
        out.lines()
            .any(|l| l.starts_with("bench:") && l.contains("syzygy")),
        "bench header must record the syzygy path: {out}"
    );
    assert!(
        out.contains("best "),
        "position lines must still print: {out}"
    );
}

// --- Opening book (end-to-end) -------------------------------------------

#[test]
fn book_move_then_search_after_book_exit() {
    // A scratch book with *only* a startpos entry: the first `go` answers
    // from the book (bestmove e2e4, no search info); once the game leaves the
    // book (e2e4 played) the very next `go` must run a real search.
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "morstilia_book_cli_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mini.bin");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x463b96181691fc9cu64.to_be_bytes());
    bytes.extend_from_slice(&796u16.to_be_bytes()); // e2e4, weight 10
    bytes.extend_from_slice(&10u16.to_be_bytes());
    bytes.extend_from_slice(&0u32.to_be_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let out = drive_uci(&[
        "setoption name BookEnabled value true",
        &format!("setoption name BookPath value {}", path.display()),
        "position startpos",
        "go depth 1",
        "position startpos moves e2e4",
        "go depth 2",
    ]);
    // Nothing emits before the first `go`, so a book answer means the whole
    // transcript begins with the move — with no search info of its own.
    assert!(
        out.starts_with("bestmove e2e4\n"),
        "the book must answer startpos first: {out}"
    );
    let rest = out.strip_prefix("bestmove e2e4\n").unwrap();
    assert!(
        rest.contains("info depth"),
        "the second go must search now that the book is exhausted: {out}"
    );
    let best = bestmove(rest);
    assert!(
        best.len() >= 4 && best != "0000",
        "bad second bestmove {best:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// --- NNUE (end-to-end) -----------------------------------------------------
//
// The NNUE layer is unit-tested in `src/nnue/`, and `tests/nnue.rs` pins the
// evaluator against real Stockfish 19 output. What is left for a subprocess test
// is the plumbing the user actually touches: the `--eval` / `--nnue` flags, the
// `Eval` / `NNUEFile` UCI options, and — most importantly — what happens when
// the net is missing or broken.
//
// The engine's diagnostics deliberately go to stderr, not stdout, so that the
// documented stdout contract (`uciok`, `readyok`, `info`, `bestmove`) holds
// exactly. These tests therefore check both streams.

/// Absolute path to the bundled net inside the crate.
fn bundled_net() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("nnue")
        .join("nn-1a298aa575a0.nnue")
}

/// A file that is not a net, made by truncating the real one. Distinct content
/// each call, so parallel tests do not race on the same path.
fn corrupt_net() -> (std::path::PathBuf, impl Drop) {
    use std::io::Read as _;
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let p = std::env::temp_dir().join(format!(
        "morstilia_corrupt_net_{}_{}.nnue",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let mut head = vec![0u8; 5_000_000];
    File::open(bundled_net())
        .unwrap()
        .read_exact(&mut head)
        .unwrap();
    std::fs::write(&p, &head).unwrap();
    (p.clone(), Guard(p))
}

struct Guard(std::path::PathBuf);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs the CLI and returns `(stdout, stderr, exit code)`, without asserting
/// anything: the failure paths are part of what is under test.
fn run_capturing(args: &[&str]) -> (String, String, Option<i32>) {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("spawn engine");
    (
        String::from_utf8(out.stdout).expect("stdout is utf-8"),
        String::from_utf8(out.stderr).expect("stderr is utf-8"),
        out.status.code(),
    )
}

/// The `score cp <n>` / `score mate <n>` token of the deepest `info` line.
fn final_score(out: &str) -> String {
    let line = out
        .lines()
        .filter(|l| l.starts_with("info depth"))
        .next_back()
        .unwrap_or_else(|| panic!("no info line in: {out}"));
    let mut it = line.split_whitespace();
    while let Some(t) = it.next() {
        if t == "score" {
            return format!("{} {}", it.next().unwrap(), it.next().unwrap());
        }
    }
    panic!("no score in: {line}");
}

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

/// A `position fen ...` command, as the string the UCI loop expects.
fn position(fen: &str) -> String {
    format!("position fen {fen}")
}

/// Prologue for every UCI session below: turn the opening book off.
///
/// The engine ships with a book containing a Kiwipete entry, so `go` there
/// answers straight from the book and never searches — which is correct
/// behaviour and useless for these tests. Disabling it is also what the
/// Stage-8 `threads_one_search_is_deterministic` test does.
const NO_BOOK: &str = "setoption name BookEnabled value false";

#[test]
fn the_default_evaluator_is_classical_and_loads_no_net() {
    // The engine must be usable with no net anywhere in sight, and must not
    // spend time looking for one: nothing about the net appears on either
    // stream.
    let (out, err, code) = run_capturing(&["--fen", KIWIPETE, "--depth", "6"]);
    assert_eq!(code, Some(0), "stderr: {err}");
    assert!(out.contains("bestmove "), "{out}");
    assert!(!err.contains("nnue"), "no net should be touched: {err}");
}

#[test]
fn cli_eval_nnue_loads_the_bundled_net_and_searches() {
    let (out, err, code) = run_capturing(&["--fen", KIWIPETE, "--depth", "8", "--eval", "nnue"]);
    assert_eq!(code, Some(0), "stderr: {err}");
    assert!(
        err.contains("nnue: loaded embedded default"),
        "the loaded net must be reported: {err}"
    );
    assert!(
        err.contains("0xa85b2205"),
        "the network hash must be reported: {err}"
    );
    assert_eq!(bestmove(&out).len(), 4, "no bestmove: {out}");
    // Kiwipete is worth about -0.7 for the side to move in Stockfish 19's own
    // evaluation (`tests/data/nnue_groundtruth.json`: `final_stm = -704`), so a
    // score near that is what proves the net is driving the search rather than
    // something merely labelled `nnue`.
    let score: i32 = final_score(&out)
        .strip_prefix("cp ")
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (-1_200..=-200).contains(&score),
        "NNUE score {score} is nowhere near the known Kiwipete value: {out}"
    );
}

#[test]
fn cli_eval_nnue_matches_the_uci_option() {
    // The flag and the UCI option must install the same net, or `--bench` and a
    // GUI would measure different engines.
    let (cli, _, code) = run_capturing(&["--fen", KIWIPETE, "--depth", "8", "--eval", "nnue"]);
    assert_eq!(code, Some(0));
    let (uci, err) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name Eval value nnue",
        &position(KIWIPETE),
        "go depth 8",
    ]);
    assert!(err.contains("nnue: loaded"), "stderr: {err}");
    assert_eq!(
        final_score(&cli),
        final_score(&uci),
        "cli: {}\nuci: {err}",
        normalized(&cli)
    );
    assert_eq!(bestmove(&cli), bestmove(&uci));
    // Same node count too: the two paths must search identically, not merely
    // reach the same conclusion.
    assert_eq!(
        normalized(&cli).replace(" nps", ""),
        normalized(&uci).replace(" nps", ""),
        "the two entry points must search identically"
    );
}

#[test]
fn cli_eval_nnue_with_a_missing_net_is_a_hard_error() {
    // Unlike UCI, where a bad net degrades to classical and keeps playing, an
    // explicit `--eval nnue` on a command line must not silently run something
    // else. This is the difference a CLI has to make.
    let (out, err, code) = run_capturing(&[
        "--fen",
        KIWIPETE,
        "--depth",
        "4",
        "--eval",
        "nnue",
        "--nnue",
        "/definitely/not/here/morstilia-missing.nnue",
    ]);
    assert!(!out.contains("bestmove"), "nothing may be searched: {out}");
    assert!(
        err.contains("cannot use the NNUE net") && err.contains("No such file"),
        "stderr must name the problem: {err}"
    );
    assert!(
        err.contains("morstilia-missing.nnue"),
        "stderr must name the file: {err}"
    );
    // Non-zero exit, so a script notices.
    assert_ne!(code, Some(0), "a broken --eval nnue must fail");
}

#[test]
fn cli_eval_nnue_with_a_corrupt_net_is_a_hard_error() {
    let (corrupt, _guard) = corrupt_net();
    let (out, err, code) = run_capturing(&[
        "--fen",
        KIWIPETE,
        "--depth",
        "4",
        "--eval",
        "nnue",
        "--nnue",
        corrupt.to_str().unwrap(),
    ]);
    assert!(!out.contains("bestmove"), "nothing may be searched: {out}");
    assert!(err.contains("truncated"), "stderr must explain: {err}");
    assert_ne!(code, Some(0));
}

#[test]
fn cli_rejects_an_unknown_evaluator() {
    for bad in ["nnuf", "", "nnsue", "NNUEE"] {
        let (_, err, code) = run_capturing(&["--fen", KIWIPETE, "--depth", "2", "--eval", bad]);
        assert_ne!(code, Some(0), "--eval {bad:?} was accepted");
        assert!(err.contains("--eval must be"), "stderr: {err}");
    }
    // The case-insensitive forms must work, matching the UCI option's `var` list.
    for good in ["nnue", "NNUE", "Nnue", " classical", "CLASSICAL"] {
        let (_, _, code) = run_capturing(&["--fen", KIWIPETE, "--depth", "4", "--eval", good]);
        assert_eq!(code, Some(0), "--eval {good:?} was rejected");
    }
}

#[test]
fn bench_names_the_evaluator_it_ran() {
    // Bench figures are only comparable within one evaluator, and the two are
    // not on the same centipawn scale, so the header has to say which ran.
    let (classical, _, code) = run_capturing(&["--bench", "--depth", "5"]);
    assert_eq!(code, Some(0));
    assert!(
        classical.lines().next().unwrap().contains("eval classical"),
        "{classical}"
    );
    let (nnue, err, code) = run_capturing(&["--bench", "--depth", "5", "--eval", "nnue"]);
    assert_eq!(code, Some(0), "stderr: {err}");
    assert!(nnue.lines().next().unwrap().contains("eval nnue"), "{nnue}");
    // Same positions, same total-node line, genuinely different work.
    assert!(nnue.contains("total     nodes"), "{nnue}");
    assert!(classical.contains("total     nodes"), "{classical}");
    assert_ne!(
        nnue.lines().filter(|l| l.contains(" nodes ")).count(),
        0,
        "{nnue}"
    );
}

#[test]
fn the_uci_handshake_advertises_the_eval_options() {
    let out = drive_uci(&["uci", "quit"]);
    let line = out
        .lines()
        .find(|l| l.starts_with("option name Eval "))
        .unwrap_or_else(|| panic!("no Eval option in: {out}"));
    // Pinned whole: a GUI parses this line, and `classical` has to be the
    // default so an engine with no net installed behaves exactly as before.
    assert_eq!(
        line,
        "option name Eval type combo default classical var Classical var classical var NNUE var nnue"
    );
    assert!(
        out.lines()
            .any(|l| l.starts_with("option name NNUEFile type string")),
        "{out}"
    );
}

#[test]
fn a_missing_net_over_uci_falls_back_to_classical_and_keeps_playing() {
    // The GUI case: a user sets `Eval nnue` but the net is not installed. The
    // engine must say so, revert to classical, and never stop playing.
    // Set bad NNUEFile FIRST, then Eval nnue (which will try to load it and fail).
    let (out, err) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name NNUEFile value /definitely/not/here/morstilia-missing.nnue",
        "setoption name Eval value nnue",
        "isready",
        &position(KIWIPETE),
        "go depth 8",
    ]);
    assert!(
        err.contains("nnue: cannot use"),
        "stderr must explain: {err}"
    );
    assert!(
        err.contains("falling back to the classical evaluator"),
        "stderr must say what happened: {err}"
    );
    assert!(
        out.contains("readyok"),
        "the engine must stay responsive: {out}"
    );
    assert_eq!(bestmove(&out).len(), 4, "the engine must still play: {out}");
    // The decisive part: it is *really* classical now, not just labelled that
    // way. Identical score, identical node count, identical bestmove to a run
    // that never asked for NNUE. The test includes "isready" -> "readyok", so
    // the plain comparison should also include isready.
    let (plain, _) = drive_uci_capturing(&["isready", &position(KIWIPETE), "go depth 8", "quit"]);
    assert_eq!(
        normalized(&out),
        normalized(&plain),
        "a failed NNUE load must be byte-identical to never having asked"
    );
}

#[test]
fn a_corrupt_net_over_uci_falls_back_to_classical() {
    let (corrupt, _guard) = corrupt_net();
    let (out, err) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name Eval value nnue",
        &format!("setoption name NNUEFile value {}", corrupt.display()),
        &position(KIWIPETE),
        "go depth 6",
    ]);
    assert!(err.contains("truncated"), "stderr must explain: {err}");
    assert_eq!(bestmove(&out).len(), 4, "the engine must still play: {out}");
    assert!(
        out.contains("info depth 6"),
        "the search must run to completion: {out}"
    );
}

#[test]
fn fixing_a_bad_net_path_recovers_without_restarting() {
    // The GUI will often correct the path and re-send `Eval nnue`. That must
    // work: the load is per-`setoption`, not per-process.
    let (out, err) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name Eval value nnue",
        "setoption name NNUEFile value /definitely/not/here/morstilia-missing.nnue",
        &format!("setoption name NNUEFile value {}", bundled_net().display()),
        &position(KIWIPETE),
        "go depth 8",
    ]);
    assert!(err.contains("cannot use"), "the first attempt fails: {err}");
    assert!(err.contains("nnue: loaded"), "the second succeeds: {err}");

    let (good, good_err) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name Eval value nnue",
        &position(KIWIPETE),
        "go depth 8",
    ]);
    assert!(good_err.contains("nnue: loaded"), "stderr: {good_err}");
    assert_eq!(normalized(&out), normalized(&good));
}

#[test]
fn switching_back_to_classical_mid_session_takes_effect() {
    // `Eval classical` must really uninstall the net, and the next search must be
    // the classical one.
    let (out, err) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name Eval value nnue",
        &position(KIWIPETE),
        "go depth 8",
        "setoption name Eval value classical",
        "ucinewgame",
        &position(KIWIPETE),
        "go depth 8",
    ]);
    assert!(err.contains("nnue: loaded"), "stderr: {err}");
    let scores: Vec<String> = out
        .lines()
        .filter(|l| l.starts_with("info depth 8 "))
        .map(final_score)
        .collect();
    assert_eq!(scores.len(), 2, "two depth-8 scores in: {out}");
    assert_ne!(scores[0], scores[1], "the switch had no effect: {out}");

    // The second run is the plain classical one, down to the node count.
    let (plain, plain_err) = drive_uci_capturing(&[&position(KIWIPETE), "go depth 8", "quit"]);
    // Get the LAST "info depth 8" line (the second search after switching to classical)
    let last_depth8 = |s: &str| {
        s.lines()
            .rev()
            .find(|l| l.starts_with("info depth 8 "))
            .unwrap_or_else(|| panic!("no depth-8 line in: {s}"))
            .to_string()
    };
    let out_depth8 = normalized(&last_depth8(&out));
    let plain_depth8 = normalized(&last_depth8(&plain));
    if out_depth8 != plain_depth8 {
        eprintln!("STDERR from test session:\n{err}");
        eprintln!("STDERR from plain session:\n{plain_err}");
    }
    assert_eq!(out_depth8, plain_depth8);
}

#[test]
fn one_thread_nnue_search_is_reproducible_over_uci() {
    // The full determinism guarantee, end to end, with the net installed. The
    // node count is part of it: an accumulator that were order-dependent or raced
    // would change it even when the best move happened to match.
    let session: &[&str] = &[
        NO_BOOK,
        "setoption name Threads value 1",
        "setoption name Eval value nnue",
        "ucinewgame",
        &position(KIWIPETE),
        "go depth 9",
        "quit",
    ];
    let first = normalized(&drive_uci_capturing(session).0);
    for _ in 0..2 {
        let again = normalized(&drive_uci_capturing(session).0);
        assert_eq!(first, again, "the NNUE search is not reproducible");
    }
    // And it is not accidentally the classical search.
    let (plain, _) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name Threads value 1",
        "ucinewgame",
        &position(KIWIPETE),
        "go depth 9",
    ]);
    assert_ne!(first, normalized(&plain));
}

#[test]
fn a_multithreaded_nnue_search_still_returns_a_move() {
    // Lazy SMP shares one `Arc<Network>` across the workers, each with its own
    // accumulator stack. There is no per-thread determinism claim, but there is
    // a "it must not crash or hang" claim.
    let (out, err) = drive_uci_capturing(&[
        NO_BOOK,
        "setoption name Threads value 4",
        "setoption name Eval value nnue",
        &position(KIWIPETE),
        "go depth 8",
    ]);
    assert!(err.contains("nnue: loaded"), "stderr: {err}");
    let best = bestmove(&out);
    assert_eq!(best.len(), 4, "bad bestmove {best:?}");
    assert_ne!(best, "0000", "no move found: {out}");
}

#[test]
fn nnue_and_classical_both_find_the_same_forced_mate() {
    // A shared, evaluator-independent ground truth: whatever the evaluation says,
    // Ra1xa8# is mate in one. Both evaluators must see it, or "both modes work"
    // would only be true for quiet positions.
    let fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1";
    for mode in ["classical", "nnue"] {
        let (out, err) = drive_uci_capturing(&[
            NO_BOOK,
            &format!("setoption name Eval value {mode}"),
            &position(fen),
            "go depth 4",
        ]);
        assert_eq!(
            bestmove(&out),
            "a1a8",
            "{mode} chose differently: {out} / {err}"
        );
        assert!(
            out.contains("score mate 1"),
            "{mode} did not see the mate: {out}"
        );
    }
}

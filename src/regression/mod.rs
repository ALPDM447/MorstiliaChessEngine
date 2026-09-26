//! Regression protection for evaluation tuning.
//!
//! Tuning changes 864 constants, so it can silently wreck tactics the rest of
//! the engine provably solves. This module pins the inviolable properties and
//! measures everything else:
//!
//! * **Determinism** — a `Threads = 1` search with the *tuned* parameter set
//!   must be fully reproducible (identical nodes, score and best move across
//!   two identical runs). The engine's determinism is a hard invariant, and
//!   tuned parameters must not break it.
//! * **Known tactics** — a fixed suite of positions whose correct best move
//!   the engine provably finds (forced mates, hanging material). If tuning
//!   regresses any of these, the tuned set is rejected out of hand.
//! * **Evaluation deltas** — per-position baseline-vs-tuned evaluation
//!   comparison (max and average absolute delta) over the same suite so the
//!   magnitude of the change is visible before any match is played.
//! * **Match regression (A vs B)** — [`compare_match_reports`] turns two
//!   machine-readable match reports (same suite / seed / time control, each
//!   candidate anchored against the same shared baseline) into a score-rate
//!   delta with an independent-binomial confidence interval and a verdict:
//!   `improved`, `similar`, `regressed` or `incomparable`. This is the same
//!   gate used for search/Threads/param/NNUE comparisons — no Elo is ever
//!   quoted from NPS, depth or node totals.

use crate::evaluation::EvalParams;
use crate::matchplay::{MatchReport, WdlSer};
use crate::rating::Wdl;
use crate::search::{Searcher, TimeLimit};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A position the engine must keep resolving correctly after tuning.
pub struct TacticCase {
    pub name: &'static str,
    pub fen: &'static str,
    /// Fixed search depth for the check.
    pub depth: i32,
    /// The best move the (baseline) engine provably plays at `depth`.
    pub expect_best: &'static str,
}

/// The tactic battery. Values mirror the search unit tests' proven results:
/// `finds_forced_mate_in_two` (d2d8 at depth 4) and `wins_hanging_queen`
/// (f1f2 at depth 3), plus the fixed-depth bench's best moves are computed
/// live against the baseline.
pub const TACTIC_CASES: &[TacticCase] = &[
    TacticCase {
        name: "mate-in-two",
        fen: "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1",
        depth: 4,
        expect_best: "d2d8",
    },
    TacticCase {
        name: "hanging-queen",
        fen: "6k1/8/8/8/8/8/5q2/5Q1K w - - 0 1",
        depth: 3,
        expect_best: "f1f2",
    },
];

/// Per-case regression result.
#[derive(Debug, Clone)]
pub struct CaseReport {
    pub name: String,
    pub fen: String,
    /// Baseline (default) evaluation of the position, side-to-move relative.
    pub baseline_eval: i32,
    /// Tuned evaluation.
    pub tuned_eval: i32,
    /// Best move the tuned set finds at `depth`.
    pub tuned_best: String,
    /// True when `tuned_best` matches the expected move.
    pub tactic_ok: bool,
    /// True when two identical tuned searches produced identical nodes.
    pub deterministic: bool,
    /// Node counts of the two determinism runs.
    pub nodes: (u64, u64),
}

/// Aggregate report of a [`run_regression`].
#[derive(Debug, Clone, Default)]
pub struct RegressionReport {
    pub cases: Vec<CaseReport>,
    /// Every tactic expected best move matched.
    pub all_tactics_ok: bool,
    /// Every tuned search was reproducible.
    pub all_deterministic: bool,
    /// Max |baseline_eval − tuned_eval| over the suite.
    pub max_eval_delta: i32,
    /// Mean |baseline_eval − tuned_eval| over the suite.
    pub avg_eval_delta: f64,
}

/// Runs the full regression battery: searches the tuned set twice per case at
/// `Threads = 1` (determinism), compares best moves against the known-good
/// tactics, and diff-prints the evaluation deltas.
pub fn run_regression(baseline: &EvalParams, tuned: &EvalParams) -> RegressionReport {
    let mut report = RegressionReport::default();
    let mut deltas: Vec<i32> = Vec::new();
    let evaluator = crate::evaluation::Evaluator;

    for case in TACTIC_CASES {
        let pos = crate::board::Position::from_fen(case.fen)
            .unwrap_or_else(|_| panic!("bad regression FEN for {}", case.name));
        let baseline_eval = evaluator.evaluate_with(&pos, baseline);
        let tuned_eval = evaluator.evaluate_with(&pos, tuned);

        let run = |params: &EvalParams| -> (String, u64) {
            let mut searcher = Searcher::with_params(16, params.clone());
            let stop = std::sync::atomic::AtomicBool::new(false);
            let limits = TimeLimit {
                depth: Some(case.depth),
                nodes: None,
                movetime_ms: None,
                soft_ms: 0,
                hard_ms: 0,
                infinite: true,
            };
            let r = searcher.search(&pos, &[], &limits, &Arc::new(stop), 1, &[]);
            (r.best.to_uci(), r.nodes)
        };
        let (a_best, a_nodes) = run(tuned);
        let (b_best, b_nodes) = run(tuned);

        let delta = (tuned_eval - baseline_eval).abs();
        deltas.push(delta);
        report.cases.push(CaseReport {
            name: case.name.to_string(),
            fen: case.fen.to_string(),
            baseline_eval,
            tuned_eval,
            tuned_best: a_best.clone(),
            tactic_ok: a_best == case.expect_best,
            deterministic: a_nodes == b_nodes && a_best == b_best,
            nodes: (a_nodes, b_nodes),
        });
    }

    report.all_tactics_ok = report.cases.iter().all(|c| c.tactic_ok);
    report.all_deterministic = report.cases.iter().all(|c| c.deterministic);
    report.max_eval_delta = deltas.iter().copied().max().unwrap_or(0);
    report.avg_eval_delta =
        deltas.iter().map(|&d| d as f64).sum::<f64>() / deltas.len().max(1) as f64;
    report
}

/// The benchmark position set reused by both the bench tool and tuning
/// diagnostics, as `(name, fen)` pairs.
pub fn bench_positions() -> &'static [(&'static str, &'static str)] {
    &[
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
    ]
}

/// Machine-readable result of an A-vs-B regression comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatchComparison {
    /// Candidate names (report `candidate` POV) of the two runs.
    pub a: String,
    pub b: String,
    pub a_games: u64,
    pub b_games: u64,
    /// Candidate W/D/L of each run.
    pub a_wdl: WdlSer,
    pub b_wdl: WdlSer,
    /// Candidate score rates `(wins + draws/2) / games`.
    pub a_score: f64,
    pub b_score: f64,
    /// `a_score - b_score`.
    pub score_delta: f64,
    /// 95% confidence interval on the score-rate difference (independent
    /// binomial, Newcombe hybrid on the two Wilson intervals).
    pub ci_lo: f64,
    pub ci_hi: f64,
    /// Logistic Elo difference `a.elo - b.elo` (both runs describe the same
    /// shared baseline opponent).
    pub elo_delta: f64,
    /// True when both runs used the same suite/seed/time control/candidate
    /// (only then are the deltas meaningful).
    pub comparable: bool,
    /// `improved` | `similar` | `regressed` | `insufficient` | `incomparable`.
    pub verdict: String,
    /// Why the runs are (or are not) comparable.
    pub note: String,
}

/// Compares two match reports for an A-vs-B regression.
///
/// The two runs must describe the **same matchup** — identical suite, seed,
/// time control, color alternation and candidate side — so that the only
/// difference is the tested parameter set (eval, search, threads, …). Each
/// report's W/D/L/Elo is already candidate-anchored against the same shared
/// baseline opponent, so `score_delta` / `elo_delta` directly measure A vs B.
///
/// The verdict uses a 95% *independent-binomial* confidence interval on the
/// score-rate difference (Newcombe hybrid over the two Wilson intervals, the
/// same convention as [`Wdl::score_ci`]):
///
/// * `improved` — the whole interval lies above 0;
/// * `regressed` — the whole interval lies below 0;
/// * `similar` — the interval straddles 0 (no detectable difference);
/// * `insufficient` — one record has no games;
/// * `incomparable` — the run configurations differ (same-config A/B testing
///   requires a shared seed/suite/time control).
///
/// No Elo is ever derived from NPS, depth or nodes — only W/D/L counts.
pub fn compare_match_reports(a: &MatchReport, b: &MatchReport) -> MatchComparison {
    let comparable = a.suite == b.suite
        && a.seed == b.seed
        && a.time_control == b.time_control
        && a.alternate_colors == b.alternate_colors
        && a.candidate == b.candidate;
    let wa: Wdl = a.wdl();
    let wb: Wdl = b.wdl();
    let (pa, pb) = (wa.score_rate(), wb.score_rate());

    let mut out = MatchComparison {
        a: a.white.name.clone(),
        b: b.white.name.clone(),
        a_games: wa.games(),
        b_games: wb.games(),
        a_wdl: a.wdl.clone(),
        b_wdl: b.wdl.clone(),
        a_score: pa,
        b_score: pb,
        score_delta: pa - pb,
        ci_lo: 0.0,
        ci_hi: 0.0,
        elo_delta: a.elo - b.elo,
        comparable,
        verdict: String::new(),
        note: String::new(),
    };

    if !comparable {
        out.verdict = "incomparable".to_string();
        let note = format!(
            "runs are not the same matchup: suite {:?} vs {:?}, seed {} vs {}, tc {:?} vs {:?}, \
             alternate {} vs {}, candidate {:?} vs {:?}",
            a.suite,
            b.suite,
            a.seed,
            b.seed,
            a.time_control,
            b.time_control,
            a.alternate_colors,
            b.alternate_colors,
            a.candidate,
            b.candidate
        );
        out.note = note;
        return out;
    }
    if wa.games() == 0 || wb.games() == 0 {
        out.verdict = "insufficient".to_string();
        out.note = format!("A has {} games, B has {} games", wa.games(), wb.games());
        return out;
    }

    // 95% two-sided Wilson intervals on each score rate.
    let (la, ua) = wa.score_ci(1.96);
    let (lb, ub) = wb.score_ci(1.96);
    // Newcombe hybrid interval for the difference of two independent
    // proportions.
    let d = pa - pb;
    out.ci_lo = d - ((pa - la).powi(2) + (ub - pb).powi(2)).sqrt();
    out.ci_hi = d + ((ua - pa).powi(2) + (pb - lb).powi(2)).sqrt();

    // serde_json cannot represent ±∞ (it would serialize as `null`), and a
    // perfect/blank record makes the raw delta infinite, so clamp the
    // machine-readable value; the note keeps the honest bound.
    let ed = out.elo_delta;
    out.elo_delta = if ed.is_finite() {
        ed
    } else if ed.is_sign_positive() {
        9999.0
    } else {
        -9999.0
    };

    out.verdict = if out.ci_lo > 0.0 {
        "improved".to_string()
    } else if out.ci_hi < 0.0 {
        "regressed".to_string()
    } else {
        "similar".to_string()
    };
    out.note = format!(
        "score delta {d:+.3} (95% CI [{}, {}]) over {} vs {} games",
        out.ci_lo,
        out.ci_hi,
        wa.games(),
        wb.games()
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matchplay::{EngineSummary, SourceManifest};

    /// A minimal, valid report describing one candidate (named `name`) against
    /// a shared baseline with the given W/D/L. Both reports in a comparison
    /// use the same candidate *side* (`white`), so the runs are comparable.
    fn report_with(
        wins: u64,
        draws: u64,
        losses: u64,
        name: &str,
        suite: &str,
        seed: u64,
        tc: &str,
    ) -> MatchReport {
        let wdl = WdlSer {
            wins,
            draws,
            losses,
        };
        let w: Wdl = Wdl::from(&wdl);
        let (lo, hi) = w.elo_ci();
        MatchReport {
            format: "morstilia-match-report/v1".to_string(),
            source: SourceManifest {
                engine: "morstilia".to_string(),
                version: "7.0.0".to_string(),
                source_fingerprint: 0,
                profile: "debug".to_string(),
            },
            suite: suite.to_string(),
            suite_entries: 48,
            seed,
            games_requested: w.games() as usize,
            games_completed: w.games() as usize,
            time_control: tc.to_string(),
            tc_deterministic: true,
            alternate_colors: true,
            max_plies: 240,
            adjudicate_mate: true,
            parallel: 1,
            candidate: "white".to_string(),
            white: EngineSummary {
                name: name.to_string(),
                params: "eval".to_string(),
                threads: 1,
                hash_mb: 16,
                syzygy: "none".to_string(),
            },
            black: EngineSummary {
                name: "baseline".to_string(),
                params: "baseline".to_string(),
                threads: 1,
                hash_mb: 16,
                syzygy: "none".to_string(),
            },
            wdl,
            score_rate: w.score_rate(),
            elo: w.elo(),
            elo_ci_lo: lo,
            elo_ci_hi: hi,
            avg_plies: 60.0,
            terminations: vec![("checkmate".to_string(), wins + losses)],
            sprt: None,
            baseline: None,
            games: vec![],
        }
    }

    #[test]
    fn baseline_passes_its_own_battery() {
        let baseline = EvalParams::default();
        let report = run_regression(&baseline, &baseline);
        assert!(
            report.all_tactics_ok,
            "baseline must solve its own tactics: {report:?}"
        );
        assert!(
            report.all_deterministic,
            "Threads=1 must be deterministic even via the regression path"
        );
        assert_eq!(report.max_eval_delta, 0, "identical params → zero deltas");
    }

    #[test]
    fn heavy_perturbation_reports_deltas_but_keeps_determinism() {
        let baseline = EvalParams::default();
        let mut noisy = baseline.clone();
        // Shove the rook value +40 (far outside any sane tuning step). The
        // mate-in-two case is white-up-a-rook, so its evaluation must move.
        noisy.piece_values[3] += 40;
        let report = run_regression(&baseline, &noisy);
        assert!(report.max_eval_delta > 0, "deltas must be visible");
        assert!(
            report.all_deterministic,
            "any parameter set must still search deterministically"
        );
    }

    // --- A-vs-B match regression -------------------------------------------

    #[test]
    fn identical_reports_are_similar() {
        let a = report_with(6, 2, 2, "A", "classical-v1", 7, "depth 6");
        let b = report_with(6, 2, 2, "B", "classical-v1", 7, "depth 6");
        let c = compare_match_reports(&a, &b);
        assert_eq!(c.verdict, "similar");
        assert_eq!(c.score_delta, 0.0);
        assert!(
            c.ci_lo <= 0.0 && c.ci_hi >= 0.0,
            "{:?}..{:?}",
            c.ci_lo,
            c.ci_hi
        );
    }

    #[test]
    fn dominant_a_is_improved() {
        let a = report_with(10, 0, 0, "A", "classical-v1", 7, "depth 6");
        let b = report_with(0, 0, 10, "B", "classical-v1", 7, "depth 6");
        let c = compare_match_reports(&a, &b);
        assert_eq!(c.verdict, "improved");
        assert!(c.ci_lo > 0.0, "interval must be entirely positive");
        assert_eq!(c.score_delta, 1.0);
        assert!(c.elo_delta > 0.0, "the better side must be ahead in Elo");
    }

    #[test]
    fn dominant_b_is_regressed() {
        let a = report_with(0, 0, 10, "A", "classical-v1", 7, "depth 6");
        let b = report_with(10, 0, 0, "B", "classical-v1", 7, "depth 6");
        let c = compare_match_reports(&a, &b);
        assert_eq!(c.verdict, "regressed");
        assert!(c.ci_hi < 0.0, "interval must be entirely negative");
    }

    #[test]
    fn small_noisy_gap_is_similar() {
        let a = report_with(6, 0, 4, "A", "classical-v1", 7, "depth 6");
        let b = report_with(5, 0, 5, "B", "classical-v1", 7, "depth 6");
        let c = compare_match_reports(&a, &b);
        assert_eq!(
            c.verdict, "similar",
            "10 games cannot resolve 0.1 score diff"
        );
    }

    #[test]
    fn mismatched_runs_are_incomparable() {
        let a = report_with(6, 2, 2, "A", "classical-v1", 7, "depth 6");
        let b = report_with(6, 2, 2, "B", "classical-v1", 8, "depth 6");
        let c = compare_match_reports(&a, &b);
        assert_eq!(c.verdict, "incomparable");
        assert!(c.note.contains("seed"), "note explains why: {}", c.note);
    }

    #[test]
    fn empty_record_is_insufficient() {
        let a = report_with(0, 0, 0, "A", "classical-v1", 7, "depth 6");
        let b = report_with(5, 0, 5, "B", "classical-v1", 7, "depth 6");
        let c = compare_match_reports(&a, &b);
        assert_eq!(c.verdict, "insufficient");
    }

    #[test]
    fn comparison_is_machine_readable_json() {
        let a = report_with(10, 0, 0, "A", "classical-v1", 7, "depth 6");
        let b = report_with(0, 0, 10, "B", "classical-v1", 7, "depth 6");
        let s = serde_json::to_string_pretty(&compare_match_reports(&a, &b)).expect("serializes");
        let back: MatchComparison = serde_json::from_str(&s).expect("parses");
        assert_eq!(back.verdict, "improved");
        assert_eq!(back.a_score, 1.0);
    }
}

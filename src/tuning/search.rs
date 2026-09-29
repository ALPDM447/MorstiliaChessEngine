//! SPSA tuning for the *search* parameters.
//!
//! [`tuning::tune`](super::tune) optimises the classical evaluation against a
//! fixed position set with a Texel loss. That objective is wrong for search
//! parameters: the evaluation is not what changes, so there is no scalar
//! "loss of this position" to descend — a search parameter only shows up as
//! *which move got played* and *how long the game lasted*.
//!
//! So this module optimises the honest signal: **a fixed, deterministic
//! self-play gauntlet**. For every candidate parameter set, the same seed
//! schedule plays the same games against a frozen reference engine, and the
//! objective is the candidate's score deficit:
//!
//! ```text
//! loss = 1 − score_rate      (0.0 = draws everything, 1.0 = loses everything)
//! ```
//!
//! # Why a gauntlet and not a bench
//!
//! Node counts, NPS and "score of the PV at depth 12" are all proxies that can
//! move the wrong way. A gauntlet is the same measurement an A/B test makes,
//! just run in a loop — which is why [`crate::matchplay`]'s SPRT remains the
//! *validation* step afterwards and this is only the *search* for a candidate.
//! A parameter set that wins the gauntlet and loses the SPRT was noise, and the
//! report says so.
//!
//! # Determinism
//!
//! [`spsa_search`] is a pure function of
//! `(base, subset, config, objective, rng)`. The only randomness is the
//! caller's [`crate::book::SplitMix64`], and the gauntlet itself is seeded
//! from that same run's index rather than from wall-clock or thread
//! interleaving. The same seed replays the identical parameter trajectory *and*
//! the identical game schedule — the self-play determinism contract, applied to
//! search tuning.
//!
//! # What is *not* tuned here
//!
//! The `[gates]` switches are booleans, not numbers, and are deliberately
//! outside the SPSA vector. They are switched one at a time and measured with
//! [`crate::matchplay`] SPRT: an SPSA gradient over a discontinuous 0/1
//! objective is noise, and pretending otherwise would produce a "tuned" file
//! whose gate set is arbitrary.

use std::sync::Arc;

use crate::book::SplitMix64;
use crate::evaluation::EvalParams;
use crate::matchplay::{
    CandidateSide, EngineConfig, EngineEvaluator, MatchConfig, TimeControl, run_match,
    suite_by_name,
};
use crate::nnue::network::Network;
use crate::openings::OpeningSuite;
use crate::search::params::SearchParams;
use crate::search::{Searcher, TimeLimit};
use crate::tuning::SpsaConfig;

// ---------------------------------------------------------------------------
// Subset selection
// ---------------------------------------------------------------------------

/// Which search parameters the tuner is allowed to change, selected by prefix.
///
/// Mirrors [`super::ParamSubset`] but over [`SearchParams::bounds`]: the
/// parameter names are `singular_beta_base`, `lmr_reduction_modern`, …, so
/// `"singular"` selects the whole singular family and `"lmr_"` the LMR family.
#[derive(Debug, Clone, Default)]
pub struct SearchSubset {
    prefixes: Vec<String>,
}

impl SearchSubset {
    /// Every numeric search parameter is tunable.
    pub fn all() -> SearchSubset {
        SearchSubset::default()
    }

    /// Only parameters whose name equals a prefix or starts with `prefix.`.
    ///
    /// Note the `singular_beta_base` naming: there are no dots in the search
    /// parameter names, so a bare prefix (`"singular"`) is the useful form and
    /// both spellings are accepted.
    pub fn from_prefixes(prefixes: &[&str]) -> SearchSubset {
        SearchSubset {
            prefixes: prefixes.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn is_all(&self) -> bool {
        self.prefixes.is_empty()
    }

    /// True when `name` is selected (exact match, prefix match, or the
    /// `prefix.name` spelling).
    pub fn matches(&self, name: &str) -> bool {
        self.prefixes.is_empty()
            || self
                .prefixes
                .iter()
                .any(|p| name == p || name.starts_with(p.as_str()))
    }

    /// The boolean mask over [`SearchParams::bounds`] order.
    pub fn mask(&self) -> Vec<bool> {
        SearchParams::bounds()
            .iter()
            .map(|d| self.matches(d.name))
            .collect()
    }

    pub fn selected_count(&self) -> usize {
        self.mask().iter().filter(|&&b| b).count()
    }

    /// The selected parameter names, in bounds order — what a report lists.
    pub fn selected_names(&self) -> Vec<String> {
        SearchParams::bounds()
            .iter()
            .filter(|d| self.matches(d.name))
            .map(|d| d.name.to_string())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The objective
// ---------------------------------------------------------------------------

/// A frozen game between one candidate configuration and one reference.
///
/// A single point, the smallest honest unit: at fixed depth, single-threaded,
/// from a named opening, with a named seed, exactly reproducible.
#[derive(Debug, Clone)]
pub struct GauntletGame {
    /// Index into the opening suite.
    pub opening: usize,
    /// Which color the candidate plays.
    pub candidate: CandidateSide,
    /// Per-game seed (mixed with the SPSA iteration index, never reused).
    pub seed: u64,
}

/// The fixed schedule of games every candidate plays.
#[derive(Debug, Clone)]
pub struct Gauntlet {
    /// Name of the opening suite (`matchplay::suite_by_name`).
    pub suite: String,
    /// Search depth per move. Depth control is the only time control that is
    /// fully deterministic; a `movetime` control would make the objective
    /// depend on machine load and silently tune noise.
    pub depth: i32,
    /// Transposition table, MB, per side.
    pub hash_mb: usize,
    /// Maximum plies per game before an adjudicated draw.
    pub max_plies: usize,
    /// Total games per objective evaluation. Must be even: the schedule
    /// alternates colors so a candidate is never the color that happens to
    /// draw more.
    pub games: usize,
    /// Games to stop after when the schedule is truncated. `None` = all.
    pub truncate_after: Option<usize>,
    /// The reference engine's own search parameters. Frozen: a moving
    /// reference would make the gradient a difference of two moving targets.
    pub reference_search_params: SearchParams,
    /// The reference engine's evaluation parameters.
    pub reference_eval_params: EvalParams,
    /// Evaluator for both sides.
    pub evaluator: EngineEvaluator,
    /// The decoded net, shared by every game and every thread.
    pub net: Option<Arc<Network>>,
}

impl Default for Gauntlet {
    fn default() -> Self {
        Gauntlet {
            suite: "classical-v1".to_string(),
            depth: 10,
            hash_mb: 16,
            max_plies: 200,
            games: 8,
            truncate_after: None,
            reference_search_params: SearchParams::default(),
            reference_eval_params: EvalParams::default(),
            evaluator: EngineEvaluator::Nnue,
            net: None,
        }
    }
}

impl Gauntlet {
    /// The exact game schedule for SPSA iteration `k`.
    ///
    /// Deterministic in `(games, k)`: game `i` of iteration `k` uses opening
    /// `i % suite_len`, color alternating on `i`, and a seed mixed from
    /// `(base_seed, k, i)`. The `k` mixing is what stops the same opening with
    /// the same color from being the *only* evidence in both the `+` and the
    /// `−` evaluation — without it, a candidate's whole gradient could come
    /// from one opening's horizon effect.
    pub fn schedule(&self, suite_len: usize, base_seed: u64, k: usize) -> Vec<GauntletGame> {
        let n = self.truncate_after.unwrap_or(self.games).min(self.games);
        (0..n)
            .map(|i| GauntletGame {
                opening: if suite_len == 0 { 0 } else { i % suite_len },
                candidate: if i % 2 == 0 {
                    CandidateSide::White
                } else {
                    CandidateSide::Black
                },
                seed: mix3(base_seed, k as u64, i as u64),
            })
            .collect()
    }

    /// Plays the schedule for iteration `k` and returns the candidate's
    /// score deficit, `1 − score_rate` in `0.0 ..= 2.0`.
    ///
    /// `0.0` means the candidate drew every game; `1.0` means it split them
    /// evenly; `2.0` means it lost every game. A *lower* loss is better, which
    /// is what the SPSA loop minimises.
    pub fn loss(&self, candidate: &SearchParams, k: usize, base_seed: u64) -> anyhow::Result<f64> {
        let suite: OpeningSuite = suite_by_name(&self.suite)?;
        let suite_len = suite.len();
        let games = self.schedule(suite_len, base_seed, k);
        if games.is_empty() {
            // An empty schedule has no information; returning the neutral
            // value (1.0 = an even split) makes the gradient zero rather than
            // inventing evidence.
            return Ok(1.0);
        }

        let mut score = 0.0f64;
        for g in &games {
            // Build both sides first, then place them by color: the
            // candidate's *color* is the schedule's job, so a game is
            // identified by which engine sits on which side rather than by a
            // special-cased `MatchConfig`.
            let mut cand = EngineConfig::new("candidate", self.reference_eval_params.clone())
                .with_search_params(candidate.clone())
                .with_evaluator(self.evaluator, self.net.clone());
            cand.hash_mb = self.hash_mb;
            let mut refr = EngineConfig::new("reference", self.reference_eval_params.clone())
                .with_search_params(self.reference_search_params.clone())
                .with_evaluator(self.evaluator, self.net.clone());
            refr.hash_mb = self.hash_mb;
            let (white, black) = match g.candidate {
                CandidateSide::White => (cand, refr),
                CandidateSide::Black => (refr, cand),
            };

            let cfg = MatchConfig {
                games: 1,
                seed: g.seed,
                white,
                black,
                tc: TimeControl::Depth(self.depth),
                // `alternate_colors` is irrelevant for a one-game match whose
                // sides are already placed; leaving it off documents that the
                // schedule, not this flag, is what balances colors.
                alternate_colors: false,
                max_plies: self.max_plies,
                adjudicate_mate: true,
                sprt: None,
                parallel: 1,
                candidate: g.candidate,
                suite_name: self.suite.clone(),
            };
            let report = run_match(&cfg, &suite, None, None, |_| {})?;
            score += report.score_rate;
        }
        Ok(1.0 - score / games.len() as f64)
    }
}

/// A cheap, *non-game* objective for smoke tests and for the `--iterations 0`
/// sanity path: fixed-depth searches over a fixed position list, scored by the
/// mean static evaluation of the chosen best move (higher is better, so the
/// loss is its negation).
///
/// It is honest about what it is: a **proxy**. A real strength claim needs
/// [`Gauntlet::loss`]; this exists so the SPSA loop itself is testable without
/// playing a single game, and so a smoke run can prove the plumbing end to end
/// in under a second.
#[derive(Debug, Clone)]
pub struct ProxyObjective {
    /// Fixed-depth search per position.
    pub depth: i32,
    /// Transposition table, MB.
    pub hash_mb: usize,
    /// The positions, as FEN strings, searched in order.
    pub positions: Vec<String>,
    /// Evaluator for the searches.
    pub evaluator: EngineEvaluator,
    /// The decoded net, shared across positions.
    pub net: Option<Arc<Network>>,
}

impl Default for ProxyObjective {
    fn default() -> Self {
        ProxyObjective {
            depth: 8,
            hash_mb: 16,
            positions: Vec::new(),
            evaluator: EngineEvaluator::Nnue,
            net: None,
        }
    }
}

impl ProxyObjective {
    /// Mean static evaluation of the played move, negated so lower is better.
    pub fn loss(&self, sp: &SearchParams) -> anyhow::Result<f64> {
        if self.positions.is_empty() {
            return Ok(0.0);
        }
        let mut searcher = Searcher::with_params(self.hash_mb, EvalParams::default());
        searcher.set_search_params(sp.clone());
        searcher.set_nnue(self.net.clone());
        // A single searcher is reused across the position list, so the second
        // position already benefits from a warm TT. That is *wrong* for a
        // strength proxy — it would measure how well the candidate exploits a
        // table filled by the previous position — so the table is cleared per
        // position instead. `new_search()` ages the generation, which is
        // exactly "keep the memory, drop the certainty"; for a proxy that
        // wants neither, resizing is the honest choice and is cheap at 16 MB.
        let mut total = 0.0f64;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let limits = TimeLimit {
            depth: Some(self.depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        for fen in &self.positions {
            let pos = crate::board::Position::from_fen(fen)
                .map_err(|e| anyhow::anyhow!("proxy objective: invalid FEN {fen:?}: {e}"))?;
            searcher.clear_tt();
            let r = searcher.search(&pos, &[], &limits, &stop, 1, &[]);
            // The score is from the side to move; negating it makes "a better
            // move for the side to move" a lower loss.
            total += -(r.score as f64);
        }
        Ok(total / self.positions.len() as f64)
    }
}

/// Anything that can score a candidate parameter set. Lower loss is better.
pub trait SearchLoss {
    /// The loss of `sp` at SPSA iteration `k` (the index only matters to
    /// stochastic objectives such as [`Gauntlet`]).
    fn loss(&mut self, sp: &SearchParams, k: usize) -> anyhow::Result<f64>;
}

impl SearchLoss for Gauntlet {
    fn loss(&mut self, sp: &SearchParams, k: usize) -> anyhow::Result<f64> {
        // The base seed is fixed at construction time via `with_seed`; a
        // `Gauntlet` value carries no RNG, so it is `&self` + a fixed seed.
        Gauntlet::loss(self, sp, k, 0x5EED_0000_0000_0001)
    }
}

impl SearchLoss for ProxyObjective {
    fn loss(&mut self, sp: &SearchParams, _k: usize) -> anyhow::Result<f64> {
        ProxyObjective::loss(self, sp)
    }
}

// ---------------------------------------------------------------------------
// The SPSA loop
// ---------------------------------------------------------------------------

/// The outcome of an [`spsa_search`] run.
#[derive(Debug, Clone)]
pub struct SearchTuneReport {
    /// Loss of the starting point.
    pub loss_before: f64,
    /// Lowest loss seen; `best_params` reproduces it.
    pub loss_best: f64,
    /// Loss of the final iterate (un-smoothed; can exceed `loss_best`).
    pub loss_final: f64,
    /// The winning flat vector, [`SearchParams::bounds`] order.
    pub best_params: Vec<f64>,
    /// The final iterate's flat vector.
    pub final_params: Vec<f64>,
    /// The winning parameter set, decoded.
    pub best: SearchParams,
    /// Number of parameters whose *exported* value changed.
    pub updated_count: usize,
    pub iterations_run: usize,
    /// `(iteration, best-so-far loss)` — the convergence curve.
    pub samples: Vec<(usize, f64)>,
    /// Every `+`/`−` loss pair, for diagnosing a noisy objective.
    pub pairs: Vec<(f64, f64)>,
}

impl SearchTuneReport {
    /// Relative loss reduction `(before − best) / before`.
    pub fn relative_improvement(&self) -> f64 {
        if self.loss_before == 0.0 {
            0.0
        } else {
            (self.loss_before - self.loss_best) / self.loss_before
        }
    }

    /// A one-line summary for a report or a log.
    pub fn summary(&self) -> String {
        format!(
            "loss {:.6} -> {:.6} ({:+.2}%, {} of {} params moved, {} iterations)",
            self.loss_before,
            self.loss_best,
            100.0 * self.relative_improvement(),
            self.updated_count,
            self.best_params.len(),
            self.iterations_run
        )
    }
}

/// Runs the SPSA loop over the flat [`SearchParams`] vector with no callback.
pub fn spsa_search(
    base: &SearchParams,
    subset: &SearchSubset,
    cfg: &SpsaConfig,
    objective: &mut dyn SearchLoss,
    rng: &mut SplitMix64,
) -> anyhow::Result<SearchTuneReport> {
    spsa_search_with_callback(base, subset, cfg, objective, rng, |_, _, _| {})
}

/// [`spsa_search`] with a per-iteration callback
/// `on_sample(k, best_loss, best)`, so a long run can checkpoint the winning
/// parameter set to a TOML file. Because the loop is deterministic, a
/// checkpoint written at a fixed `k` is reproducible from the same seed.
pub fn spsa_search_with_callback(
    base: &SearchParams,
    subset: &SearchSubset,
    cfg: &SpsaConfig,
    objective: &mut dyn SearchLoss,
    rng: &mut SplitMix64,
    mut on_sample: impl FnMut(usize, f64, &SearchParams),
) -> anyhow::Result<SearchTuneReport> {
    let defs = SearchParams::bounds();
    let n = defs.len();
    let mask = subset.mask();
    if mask.len() != n {
        anyhow::bail!("internal: search-parameter mask/bounds length mismatch");
    }
    if !mask.iter().any(|&b| b) {
        anyhow::bail!(
            "the prefix selection matched none of the {} search parameters; known names: {}",
            n,
            SearchParams::bounds()
                .iter()
                .map(|d| d.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let base_vec = base.to_vec();
    let loss_before = objective.loss(base, 0)?;

    let mut theta = base_vec.clone();
    let mut best = theta.clone();
    let mut best_loss = loss_before;
    let mut samples = vec![(0usize, loss_before)];
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(cfg.iterations);

    for k in 0..cfg.iterations {
        // ±1 on the selected parameters, 0 elsewhere: a component with
        // `delta == 0` contributes no gradient and is never stepped, so an
        // unselected parameter is bit-identical to the base set.
        let mut delta = vec![0.0f64; n];
        for i in 0..n {
            delta[i] = if mask[i] && (rng.next() & 1 == 0) {
                -1.0
            } else if mask[i] {
                1.0
            } else {
                0.0
            };
        }
        let ck = cfg.perturb(k);
        let mut plus = theta.clone();
        let mut minus = theta.clone();
        for i in 0..n {
            if delta[i] != 0.0 {
                plus[i] = (theta[i] + ck * delta[i]).clamp(defs[i].min, defs[i].max);
                minus[i] = (theta[i] - ck * delta[i]).clamp(defs[i].min, defs[i].max);
            }
        }
        let mut sp_plus = base.clone();
        sp_plus.from_vec(&plus);
        let mut sp_minus = base.clone();
        sp_minus.from_vec(&minus);
        let l_plus = objective.loss(&sp_plus, k)?;
        let l_minus = objective.loss(&sp_minus, k)?;
        pairs.push((l_plus, l_minus));

        // Two-sided gradient estimate, then a normalised step: the largest
        // component moves by `a_k`, everything else proportionally. The same
        // normalisation the evaluation tuner uses, so a search-parameter loss
        // (a score *deficit*, order 1e-3) gets a usable step size without a
        // hand-tuned per-objective gain.
        let ak = cfg.step(k);
        let mut grad = vec![0.0f64; n];
        let mut g_scale = 0.0f64;
        for i in 0..n {
            if delta[i] != 0.0 {
                grad[i] = (l_plus - l_minus) / (2.0 * ck * delta[i]);
                g_scale = g_scale.max(grad[i].abs());
            }
        }
        if g_scale > 0.0 {
            for i in 0..n {
                if grad[i] != 0.0 {
                    theta[i] = (theta[i] - ak * grad[i] / g_scale).clamp(defs[i].min, defs[i].max);
                }
            }
        }

        let mut sp_cur = base.clone();
        sp_cur.from_vec(&theta);
        let l_cur = objective.loss(&sp_cur, k)?;
        if l_cur < best_loss {
            best_loss = l_cur;
            best = theta.clone();
        }
        let mut sp_best = base.clone();
        sp_best.from_vec(&best);
        samples.push((k + 1, best_loss));
        on_sample(k, best_loss, &sp_best);
    }

    let mut sp_final = base.clone();
    sp_final.from_vec(&theta);
    let mut sp_best = base.clone();
    sp_best.from_vec(&best);
    let updated_count = (0..n)
        .filter(|&i| best[i] as i64 != base_vec[i] as i64)
        .count();

    Ok(SearchTuneReport {
        loss_before,
        loss_best: best_loss,
        loss_final: objective.loss(&sp_final, cfg.iterations)?,
        best_params: best,
        final_params: theta,
        best: sp_best,
        updated_count,
        iterations_run: cfg.iterations,
        samples,
        pairs,
    })
}

/// Mixes three `u64`s into a seed. SplitMix64's finalizer, applied twice, so
/// consecutive `(k, i)` pairs cannot produce correlated schedules.
fn mix3(a: u64, b: u64, c: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c.wrapping_add(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        SearchParams::bounds()
            .iter()
            .map(|d| d.name.to_string())
            .collect()
    }

    #[test]
    fn subset_all_selects_every_numeric_parameter() {
        let all = SearchSubset::all();
        assert!(all.is_all());
        assert_eq!(all.selected_count(), SearchParams::bounds().len());
        assert_eq!(all.selected_names().len(), all.selected_count());
        // The gates are booleans and must NOT be in the flat vector, or an
        // SPSA step could silently switch a mechanism off.
        for n in all.selected_names() {
            assert!(
                !crate::search::params::SearchGates::NAMES.contains(&n.as_str()),
                "gate {n} leaked into the numeric vector"
            );
        }
    }

    #[test]
    fn subset_prefix_selects_families() {
        let s = SearchSubset::from_prefixes(&["singular"]);
        let picked = s.selected_names();
        assert!(!picked.is_empty(), "singular prefix selected nothing");
        for n in &picked {
            assert!(n.starts_with("singular"), "{n} should not be selected");
        }
        assert!(names().iter().any(|n| n.starts_with("singular")));
    }

    #[test]
    fn subset_rejects_a_prefix_that_matches_nothing() {
        let s = SearchSubset::from_prefixes(&["definitely_not_a_parameter"]);
        assert_eq!(s.selected_count(), 0);
    }

    #[test]
    fn subset_mask_matches_bounds_length() {
        for pre in [
            SearchSubset::all(),
            SearchSubset::from_prefixes(&["lmr"]),
            SearchSubset::from_prefixes(&["tt_probcut"]),
        ] {
            assert_eq!(pre.mask().len(), SearchParams::bounds().len());
        }
    }

    #[test]
    fn mix3_is_total_and_seed_sensitive() {
        assert_ne!(mix3(1, 2, 3), mix3(1, 2, 4));
        assert_ne!(mix3(1, 2, 3), mix3(2, 2, 3));
        assert_ne!(mix3(0, 0, 0), mix3(1, 0, 0));
    }

    #[test]
    fn gauntlet_schedule_is_deterministic_and_colour_balanced() {
        let g = Gauntlet {
            games: 8,
            ..Gauntlet::default()
        };
        let a = g.schedule(6, 42, 0);
        let b = g.schedule(6, 42, 0);
        assert_eq!(a.len(), 8);
        // Same (suite_len, seed, k) → identical schedule.
        assert_eq!(
            a.iter().map(|x| x.seed).collect::<Vec<_>>(),
            b.iter().map(|x| x.seed).collect::<Vec<_>>()
        );
        // Exactly half white, half black.
        assert_eq!(
            a.iter()
                .filter(|x| x.candidate == CandidateSide::White)
                .count(),
            4
        );
        // Distinct seeds: a repeated seed would replay an identical game.
        let mut seeds: Vec<u64> = a.iter().map(|x| x.seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), a.len());
        // A different iteration must not reuse the schedule.
        let c = g.schedule(6, 42, 1);
        assert_ne!(
            a.iter().map(|x| x.seed).collect::<Vec<_>>(),
            c.iter().map(|x| x.seed).collect::<Vec<_>>()
        );
    }

    #[test]
    fn gauntlet_schedule_wraps_the_suite() {
        let g = Gauntlet {
            games: 10,
            truncate_after: Some(5),
            ..Gauntlet::default()
        };
        let s = g.schedule(3, 1, 0);
        assert_eq!(s.len(), 5);
        assert!(s.iter().all(|x| x.opening < 3));
    }

    /// An objective that is exactly a linear function of the flat vector, so
    /// the SPSA loop's recovery of the gradient is testable exactly.
    struct LinearObjective {
        /// Loss = `sum_i w_i * theta_i`.
        w: Vec<f64>,
        calls: usize,
    }

    impl SearchLoss for LinearObjective {
        fn loss(&mut self, sp: &SearchParams, _k: usize) -> anyhow::Result<f64> {
            self.calls += 1;
            let v = sp.to_vec();
            Ok(v.iter().zip(&self.w).map(|(a, b)| a * b).sum())
        }
    }

    #[test]
    fn spsa_moves_uphill_on_a_linear_objective() {
        let n = SearchParams::bounds().len();
        let w: Vec<f64> = (0..n).map(|i| (i % 7) as f64 * 0.1).collect();
        let mut obj = LinearObjective {
            w: w.clone(),
            calls: 0,
        };
        let base = SearchParams::default();
        let mut rng = SplitMix64(0xA5A5);
        let cfg = SpsaConfig {
            iterations: 400,
            a: 20.0,
            c: 4.0,
            ..SpsaConfig::default()
        };
        let report = spsa_search(&base, &SearchSubset::all(), &cfg, &mut obj, &mut rng).unwrap();
        // Three losses per iteration: `+`, `−`, then the current iterate.
        assert_eq!(obj.calls, 1 + 3 * cfg.iterations);
        // The base vector sits inside the bounds, so the step can only move
        // downhill on a linear objective; anything else means the sign or the
        // normalisation is wrong.
        assert!(
            report.loss_best < report.loss_before,
            "SPSA moved the wrong way: {} -> {}",
            report.loss_before,
            report.loss_best
        );
        assert!(report.updated_count > 0);
    }

    #[test]
    fn spsa_is_reproducible_for_a_seed() {
        let n = SearchParams::bounds().len();
        let mk = || LinearObjective {
            w: (0..n).map(|i| (i % 5) as f64).collect(),
            calls: 0,
        };
        let cfg = SpsaConfig {
            iterations: 25,
            ..SpsaConfig::default()
        };
        let run = || {
            let mut obj = mk();
            let mut rng = SplitMix64(7);
            spsa_search(
                &SearchParams::default(),
                &SearchSubset::all(),
                &cfg,
                &mut obj,
                &mut rng,
            )
            .unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.best_params, b.best_params);
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.pairs, b.pairs);
        assert_eq!(a.best.fingerprint(), b.best.fingerprint());
    }

    #[test]
    fn spsa_never_steps_an_unselected_parameter() {
        struct CountingObjective;
        impl SearchLoss for CountingObjective {
            fn loss(&mut self, sp: &SearchParams, _k: usize) -> anyhow::Result<f64> {
                Ok(sp.to_vec().iter().sum::<f64>() * 0.0)
            }
        }
        let base = SearchParams::default();
        let base_vec = base.to_vec();
        let subset = SearchSubset::from_prefixes(&["singular"]);
        let mask = subset.mask();
        let mut obj = CountingObjective;
        let mut rng = SplitMix64(11);
        let cfg = SpsaConfig {
            iterations: 50,
            ..SpsaConfig::default()
        };
        let report = spsa_search(&base, &subset, &cfg, &mut obj, &mut rng).unwrap();
        for i in 0..report.best_params.len() {
            if !mask[i] {
                assert_eq!(
                    report.best_params[i],
                    base_vec[i],
                    "unselected parameter {} was modified",
                    SearchParams::bounds()[i].name
                );
            }
        }
    }

    #[test]
    fn spsa_reports_an_empty_selection_instead_of_silently_doing_nothing() {
        struct Zero;
        impl SearchLoss for Zero {
            fn loss(&mut self, _sp: &SearchParams, _k: usize) -> anyhow::Result<f64> {
                Ok(0.0)
            }
        }
        let mut obj = Zero;
        let mut rng = SplitMix64(1);
        let err = spsa_search(
            &SearchParams::default(),
            &SearchSubset::from_prefixes(&["nope_not_here"]),
            &SpsaConfig {
                iterations: 1,
                ..SpsaConfig::default()
            },
            &mut obj,
            &mut rng,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("matched none"),
            "unhelpful error: {err}"
        );
    }

    #[test]
    fn zero_iterations_reports_the_baseline_loss_unchanged() {
        struct Const(f64);
        impl SearchLoss for Const {
            fn loss(&mut self, _sp: &SearchParams, _k: usize) -> anyhow::Result<f64> {
                Ok(self.0)
            }
        }
        let mut obj = Const(0.375);
        let mut rng = SplitMix64(3);
        let base = SearchParams::default();
        let report = spsa_search(
            &base,
            &SearchSubset::all(),
            &SpsaConfig {
                iterations: 0,
                ..SpsaConfig::default()
            },
            &mut obj,
            &mut rng,
        )
        .unwrap();
        assert_eq!(report.iterations_run, 0);
        assert_eq!(report.loss_before, 0.375);
        assert_eq!(report.loss_best, 0.375);
        assert_eq!(report.updated_count, 0);
        assert_eq!(report.best.fingerprint(), base.fingerprint());
        assert!(report.summary().contains("0 of"));
    }

    #[test]
    fn the_tuned_vector_survives_a_toml_round_trip() {
        // The whole point of the flat vector is that the tuner and the file
        // agree; a `from_vec` → `save` → `load` mismatch would silently
        // discard a tuning run.
        let mut sp = SearchParams::default();
        let defs = SearchParams::bounds();
        let mut v = sp.to_vec();
        for (i, d) in defs.iter().enumerate() {
            v[i] = (d.min * 0.5 + d.max * 0.5).round();
        }
        sp.from_vec(&v);
        let toml_text = sp.to_toml_string().unwrap();
        let back = SearchParams::from_toml_str(&toml_text).unwrap();
        assert_eq!(back.to_vec(), sp.to_vec());
        assert_eq!(back.fingerprint(), sp.fingerprint());
    }
}

//! SPSA-based evaluation tuning (Stage 5).
//!
//! [`tune`] runs the classical Simultaneous Perturbation Stochastic
//! Approximation loop over the flat [`EvalParams`] vector: every iteration
//! picks a random ±1 perturbation `Δ`, evaluates the dataset at
//! `θ + c_k·Δ` and `θ − c_k·Δ` (both clamped to [`EvalParams::bounds`]), forms
//! the two-sided gradient estimate
//!
//! ```text
//! ĝ_i = (loss⁺ − loss⁻) / (2·c_k·Δ_i)
//! ```
//!
//! and steps `θ_i -= a_k·ĝ_i` before projecting back into bounds.
//!
//! # Determinism
//!
//! [`tune`] is a pure function of `(base, dataset, subset, config, rng)`; the
//! only randomness is a caller-owned [`crate::book::SplitMix64`], so the same
//! seed replays the identical parameter trajectory and the identical
//! checkpoint/export files. This mirrors the self-play determinism contract:
//! dataset and tuning are both fully reproducible.
//!
//! # Step-size normalisation
//!
//! The raw two-sided gradient of the *mean* Texel loss is tiny (a loss
//! derivative of ~1e-3 per centipawn per position, averaged over the
//! dataset), so a fixed absolute step would need a hand-tuned gain per dataset
//! size. To keep the tool meaningful at any scale — a 4-entry smoke dataset or
//! a 10k-position real one — the step is *normalised* by the observed gradient
//! scale: the largest-magnitude component moves by `a_k` centipawns each
//! iteration and every other component proportionally. The direction of the
//! step still comes from the true loss measurements; only the magnitude is
//! dataset-independent. `a_k = a/(A + k + 1)^α` and `c_k = c/(k + 1)^γ` follow
//! the classic SPSA decay schedule.
//!
//! [`ParamSubset`] restricts the active parameter set by dotted prefix
//! (`"material"`, `"pst"`, `"mobility"`, …), so a first run can tune just the
//! material values before releasing the full 864-parameter vector.

pub mod dataset;

use crate::book::SplitMix64;
use crate::evaluation::EvalParams;
use dataset::DataSet;

/// SPSA gain schedule. Defaults are the classic Spall values with a
/// normalised-step `a` sized in centipawns (see module docs).
#[derive(Debug, Clone, Copy)]
pub struct SpsaConfig {
    /// Step-size gain `a` (centipawns per iteration at `k = 0` for the
    /// strongest-gradient parameter).
    pub a: f64,
    /// Perturbation gain `c` (the ± perturbation magnitude at `k = 0`).
    pub c: f64,
    /// The moving-average horizon `A`: keeps early steps from blowing up.
    pub a_horizon: f64,
    /// Step-size decay exponent (classic 0.602).
    pub alpha: f64,
    /// Perturbation decay exponent (classic 0.101).
    pub gamma: f64,
    /// Total SPSA iterations.
    pub iterations: usize,
}

impl Default for SpsaConfig {
    fn default() -> Self {
        SpsaConfig {
            a: 5.0,
            c: 2.0,
            a_horizon: 100.0,
            alpha: 0.602,
            gamma: 0.101,
            iterations: 10_000,
        }
    }
}

impl SpsaConfig {
    /// The step-size at iteration `k`: `a / (A + k + 1)^α`.
    fn step(&self, k: usize) -> f64 {
        self.a / (self.a_horizon + k as f64 + 1.0).powf(self.alpha)
    }

    /// The perturbation magnitude at iteration `k`: `c / (k + 1)^γ`.
    fn perturb(&self, k: usize) -> f64 {
        self.c / (k as f64 + 1.0).powf(self.gamma)
    }
}

/// Which parameters the tuner is allowed to change, selected by dotted
/// prefix. `ParamSubset::all()` (an empty prefix list) tunes everything.
#[derive(Debug, Clone, Default)]
pub struct ParamSubset {
    prefixes: Vec<String>,
}

impl ParamSubset {
    /// Every parameter is tunable.
    pub fn all() -> ParamSubset {
        ParamSubset::default()
    }

    /// Only parameters whose [`ParamDef::name`] equals a prefix or starts
    /// with `prefix.` are tunable (e.g. `"material"` selects
    /// `material.pawn`, `"pst"` all 768 square tables).
    pub fn from_prefixes(prefixes: &[&str]) -> ParamSubset {
        ParamSubset {
            prefixes: prefixes.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn is_all(&self) -> bool {
        self.prefixes.is_empty()
    }

    /// True when `name` is selected (exact prefix match or `prefix.`-joined).
    pub fn matches(&self, name: &str) -> bool {
        self.prefixes.is_empty()
            || self
                .prefixes
                .iter()
                .any(|p| name == p || name.strip_prefix(p).is_some_and(|r| r.starts_with('.')))
    }

    /// The boolean mask over [`EvalParams::bounds`] order.
    pub fn mask(&self) -> Vec<bool> {
        EvalParams::bounds()
            .iter()
            .map(|d| self.matches(d.name))
            .collect()
    }

    /// How many of the 864 parameters are selected.
    pub fn selected_count(&self) -> usize {
        self.mask().iter().filter(|&&b| b).count()
    }
}

/// The outcome of a [`tune`] run. `loss_*` values are the *mean* Texel loss
/// over the dataset (see [`DataSet::texel_loss`]).
#[derive(Debug, Clone)]
pub struct TuneReport {
    /// Loss of the baseline parameters (the starting point).
    pub loss_before: f64,
    /// Lowest loss encountered; [`TuneReport::best_params`] reproduce it.
    pub loss_best: f64,
    /// Loss of the final iterate (un-smoothed; may exceed `loss_best`).
    pub loss_final: f64,
    /// Parameters that reproduce `loss_best` (flat vector, bounds order).
    pub best_params: Vec<f64>,
    /// The final iterate's parameters (flat vector, bounds order).
    pub final_params: Vec<f64>,
    /// Number of parameters whose *exported* value changed vs the baseline
    /// (an `i32`-quantised comparison, matching what `EvalParams::from_vec`
    /// writes into a TOML file).
    pub updated_count: usize,
    /// Number of SPSA iterations actually run (always `config.iterations`).
    pub iterations_run: usize,
    /// `(iteration, best-so-far loss)` sampled every iteration — the
    /// convergence curve, deterministic for a seed.
    pub samples: Vec<(usize, f64)>,
}

impl TuneReport {
    /// Relative loss reduction `(before − best) / before`; `0.0` when the
    /// baseline loss is already zero.
    pub fn relative_improvement(&self) -> f64 {
        if self.loss_before == 0.0 {
            0.0
        } else {
            (self.loss_before - self.loss_best) / self.loss_before
        }
    }
}

/// Runs the SPSA loop with no checkpoint callback.
pub fn tune(
    base: &EvalParams,
    ds: &DataSet,
    subset: &ParamSubset,
    cfg: &SpsaConfig,
    rng: &mut SplitMix64,
) -> TuneReport {
    tune_with_callback(base, ds, subset, cfg, rng, |_, _, _| {})
}

/// [`tune`] with a per-iteration callback `on_sample(k, best_loss, best)` —
/// `k` is the 0-based iteration index, `best` the best-so-far parameter set.
/// The caller decides what to persist (checkpoint TOML files, progress
/// lines); because the loop is deterministic, any checkpoint written at a
/// fixed `k` is reproducible from the same seed.
pub fn tune_with_callback(
    base: &EvalParams,
    ds: &DataSet,
    subset: &ParamSubset,
    cfg: &SpsaConfig,
    rng: &mut SplitMix64,
    mut on_sample: impl FnMut(usize, f64, &EvalParams),
) -> TuneReport {
    let defs = EvalParams::bounds();
    let n = defs.len();
    let mask = subset.mask();
    debug_assert_eq!(n, mask.len());

    let base_vec = base.to_vec();
    let loss_before = ds.texel_loss(base);

    let mut theta = base_vec.clone();
    let mut best = theta.clone();
    let mut best_loss = loss_before;
    let mut samples = vec![(0usize, loss_before)];
    let mut delta = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n];

    for k in 0..cfg.iterations {
        // Two-sided random perturbation, ±1 on the selected parameters.
        for i in 0..n {
            delta[i] = if mask[i] {
                if rng.next() & 1 == 0 { -1.0 } else { 1.0 }
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
        let l_plus = ds.texel_loss(&params_of(base, &plus));
        let l_minus = ds.texel_loss(&params_of(base, &minus));

        // Gradient estimate, normalised by the observed gradient scale so
        // the step magnitude is independent of dataset size.
        let ak = cfg.step(k);
        let mut g_scale = 0.0f64;
        for i in 0..n {
            if delta[i] != 0.0 {
                let g = (l_plus - l_minus) / (2.0 * ck * delta[i]);
                grad[i] = g;
                g_scale = g_scale.max(g.abs());
            } else {
                grad[i] = 0.0;
            }
        }
        if g_scale > 0.0 {
            for i in 0..n {
                if grad[i] != 0.0 {
                    theta[i] = (theta[i] - ak * grad[i] / g_scale).clamp(defs[i].min, defs[i].max);
                }
            }
        }

        // Best-so-far tracking (the iterate's true loss).
        let cur_params = params_of(base, &theta);
        let l_cur = ds.texel_loss(&cur_params);
        if l_cur < best_loss {
            best_loss = l_cur;
            best = theta.clone();
        }
        let best_params = params_of(base, &best);
        samples.push((k + 1, best_loss));
        on_sample(k, best_loss, &best_params);
    }

    let final_params = params_of(base, &theta);
    let updated_count = (0..n)
        .filter(|&i| best[i] as i32 != base_vec[i] as i32)
        .count();

    TuneReport {
        loss_before,
        loss_best: best_loss,
        loss_final: ds.texel_loss(&final_params),
        best_params: best,
        final_params: theta,
        updated_count,
        iterations_run: cfg.iterations,
        samples,
    }
}

/// Builds an [`EvalParams`] carrying the flat vector `theta` (all other
/// fields cloned from `base`).
fn params_of(base: &EvalParams, theta: &[f64]) -> EvalParams {
    let mut p = base.clone();
    p.from_vec(theta);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::tuning::dataset::EvalEntry;

    /// A tiny dataset with one clean signal: White is up a single pawn in a
    /// bare-kings endgame, all targets `1.0`. The evaluation is dominated by
    /// `material.pawn` (all other material terms are zero), so pushing the
    /// pawn value toward its upper bound must visibly reduce the loss.
    fn pawn_up_dataset() -> DataSet {
        let fens = [
            "4k3/8/8/8/8/8/2P5/4K3 w - - 0 1",
            "4k3/8/8/8/8/2P5/8/4K3 w - - 0 1",
            "4k3/8/8/8/3P4/8/8/4K3 w - - 0 1",
            "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
        ];
        let entries = fens
            .iter()
            .map(|fen| EvalEntry {
                pos: Position::from_fen(fen).expect("valid pawn-up FEN"),
                result: 1.0,
            })
            .collect();
        DataSet {
            entries,
            seed: 0,
            source: "test: white up a pawn".to_string(),
        }
    }

    #[test]
    fn subset_matches_dotted_prefixes_only() {
        let s = ParamSubset::from_prefixes(&["material", "mobility"]);
        assert!(s.matches("material.pawn"));
        assert!(s.matches("material.queen"));
        assert!(s.matches("mobility.knight.mg"));
        assert!(!s.matches("pst.pawn.mg.0"));
        assert!(!s.matches("king.shield_cap"));
        assert!(!s.matches("materialism"), "exact prefix, not substring");
        assert!(ParamSubset::all().matches("anything.at.all"));
        assert!(ParamSubset::all().is_all());
    }

    #[test]
    fn subset_mask_aligns_with_bounds() {
        let m = ParamSubset::from_prefixes(&["material"]).mask();
        let defs = EvalParams::bounds();
        assert_eq!(m.len(), defs.len());
        assert!(m[0], "material.pawn must be selected");
        assert!(m[4], "material.queen must be selected");
        assert!(!m[7], "pawns.doubled.mg must not be selected");
        assert_eq!(
            ParamSubset::from_prefixes(&["material"]).selected_count(),
            5,
            "the five material values; `bishop_pair` has its own prefix"
        );
        assert_eq!(ParamSubset::all().selected_count(), defs.len());
    }

    #[test]
    fn spsa_reduces_loss_on_clean_signal() {
        let base = EvalParams::default();
        let ds = pawn_up_dataset();
        let cfg = SpsaConfig {
            iterations: 300,
            ..SpsaConfig::default()
        };
        let report = tune(
            &base,
            &ds,
            &ParamSubset::from_prefixes(&["material"]),
            &cfg,
            &mut SplitMix64(1),
        );
        assert!(report.iterations_run == 300);
        assert!(
            report.loss_best < report.loss_before - 0.02,
            "tuning must visibly cut the loss: {} -> {}",
            report.loss_before,
            report.loss_best
        );
        // The pawn value must have been pushed toward its 200 bound.
        assert!(
            report.best_params[0] >= 150.0,
            "material.pawn must rise: {}",
            report.best_params[0]
        );
        // Only the pawn slot should have moved in a material-only mask.
        assert!(
            report.updated_count >= 1,
            "at least the pawn value must be updated"
        );
        assert!(report.relative_improvement() > 0.0);
        // The best params are still within bounds and reproduce loss_best.
        let defs = EvalParams::bounds();
        for (i, x) in report.best_params.iter().enumerate() {
            assert!(
                *x >= defs[i].min && *x <= defs[i].max,
                "param {i} out of bounds"
            );
        }
        let best_ev = params_of(&base, &report.best_params);
        assert!(
            (ds.texel_loss(&best_ev) - report.loss_best).abs() < 1e-12,
            "best params must reproduce loss_best"
        );
    }

    #[test]
    fn spsa_is_deterministic_per_seed() {
        let base = EvalParams::default();
        let ds = pawn_up_dataset();
        let cfg = SpsaConfig {
            iterations: 60,
            ..SpsaConfig::default()
        };
        let a = tune(&base, &ds, &ParamSubset::all(), &cfg, &mut SplitMix64(3));
        let b = tune(&base, &ds, &ParamSubset::all(), &cfg, &mut SplitMix64(3));
        assert_eq!(a.loss_best, b.loss_best, "loss must replay exactly");
        assert_eq!(a.best_params, b.best_params, "params must replay exactly");
        assert_eq!(a.samples, b.samples, "convergence curve must replay");
        // A different seed must explore a different trajectory.
        let c = tune(&base, &ds, &ParamSubset::all(), &cfg, &mut SplitMix64(4));
        assert_ne!(a.best_params, c.best_params, "different seeds diverge");
    }

    #[test]
    fn callback_observes_monotone_best_loss() {
        let base = EvalParams::default();
        let ds = pawn_up_dataset();
        let cfg = SpsaConfig {
            iterations: 100,
            ..SpsaConfig::default()
        };
        let mut seen = Vec::new();
        let report = tune_with_callback(
            &base,
            &ds,
            &ParamSubset::from_prefixes(&["material"]),
            &cfg,
            &mut SplitMix64(9),
            |k, loss, best| {
                if k % 5 == 0 {
                    seen.push((k, loss, best.piece_values[0]));
                }
            },
        );
        assert!(seen.len() >= 5, "callback must fire at sampled iterations");
        for w in seen.windows(2) {
            assert!(
                w[1].1 <= w[0].1 + 1e-12,
                "best-so-far loss must never worsen ({} -> {})",
                w[0].1,
                w[1].1
            );
        }
        assert_eq!(seen.last().unwrap().1, report.loss_best);
        // On the clean pawn-up dataset the best answer improves over time.
        assert!(seen.last().unwrap().2 > seen.first().unwrap().2);
    }

    #[test]
    fn step_sizes_decay_with_iteration() {
        let cfg = SpsaConfig::default();
        let s0 = cfg.step(0);
        let s100 = cfg.step(100);
        let s1000 = cfg.step(1000);
        assert!(s0 > s100 && s100 > s1000, "steps must decay");
        let c0 = cfg.perturb(0);
        let c100 = cfg.perturb(100);
        assert!(c0 > c100, "perturbation must decay");
        assert!(c0 > 0.0 && s0 > 0.0);
    }
}

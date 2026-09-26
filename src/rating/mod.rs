//! Win/draw/loss statistics: strength from match results.
//!
//! Two tools are provided, both working purely from counts of wins, draws and
//! losses (never from node counts, depth or NPS — the only valid strength
//! signal is what happens on the board):
//!
//! * [`Wdl::elo`] — the logistic Elo difference implied by a score rate
//!   `(wins + draws / 2) / games`, with [`Wdl::elo_ci`] giving a 95%
//!   confidence interval via a Wilson interval on the score rate.
//! * [`Sprt`] — the trinomial sequential probability ratio test: the
//!   candidate is *accepted* as stronger than the baseline (H1) or *rejected*
//!   (H0) when the accumulated log-likelihood ratio crosses the error-rate
//!   bounds, with draws modelled by a `draw_elo` logistic.
//!
//! # Draw model
//!
//! The SPRT uses the standard three-outcome logistic model: for a player with
//! Elo advantage `g` over the opponent and draw threshold `D`
//! ([`SprtConfig::draw_elo`]),
//!
//! ```text
//! L(x)      = 1 / (1 + 10^(-x/400))
//! p_win(g)  = L(g - D)
//! p_loss(g) = L(-g - D)
//! p_draw(g) = 1 - p_win(g) - p_loss(g)
//! ```
//!
//! which is symmetric (`p_win(0) == p_loss(0)`, draw rate peaks at `g = 0`)
//! and reproduces the usual "score rate = 1/(1 + 10^(-g/400))" marginal when
//! draws are marginalized out.

use std::fmt;

/// Raw win/draw/loss counts of one engine versus another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Wdl {
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
}

impl Wdl {
    /// Total finished games.
    pub fn games(&self) -> u64 {
        self.wins + self.draws + self.losses
    }

    /// Score rate `(wins + draws/2) / games`; `0.5` when nothing is recorded.
    pub fn score_rate(&self) -> f64 {
        let g = self.games();
        if g == 0 {
            0.5
        } else {
            (self.wins as f64 + 0.5 * self.draws as f64) / g as f64
        }
    }

    /// The draw ratio `draws / games`.
    pub fn draw_ratio(&self) -> f64 {
        let g = self.games();
        if g == 0 {
            0.0
        } else {
            self.draws as f64 / g as f64
        }
    }

    /// Logistic Elo difference of the first player over the second implied by
    /// the score rate: `-400 * log10(1/s - 1)`. `+∞`/`-∞` for perfect / blank
    /// records.
    pub fn elo(&self) -> f64 {
        elo_from_score(self.score_rate())
    }

    /// Standard error of the *score rate* under the binomial model
    /// `sqrt(s(1-s)/games)`.
    pub fn score_se(&self) -> f64 {
        let g = self.games();
        if g == 0 {
            return 0.0;
        }
        let s = self.score_rate();
        (s * (1.0 - s) / g as f64).sqrt()
    }

    /// 95% confidence interval of the Elo difference: a Wilson interval on
    /// the score rate mapped through the Elo transform. `z = 1.96` for the
    /// two-sided 95% level. Returns `(low, high)`.
    pub fn elo_ci(&self) -> (f64, f64) {
        let (lo, hi) = self.score_ci(1.96);
        (elo_from_score(lo), elo_from_score(hi))
    }

    /// A Wilson score interval for the score rate at the given normal
    /// quantile `z` (`1.96` ≈ 95%).
    pub fn score_ci(&self, z: f64) -> (f64, f64) {
        let g = self.games();
        if g == 0 {
            return (0.5, 0.5);
        }
        let s = self.score_rate();
        let n = g as f64;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let centre = (s + z2 / (2.0 * n)) / denom;
        let half = z * (s * (1.0 - s) / n + z2 / (4.0 * n * n)).sqrt() / denom;
        // Wilson bounds are probabilities: clamp the tiny-sample overshoot
        // (e.g. 1.0000000000000002 with one decisive game) into [0, 1] before
        // the Elo transform, which requires a score in range.
        (
            (centre - half).clamp(0.0, 1.0),
            (centre + half).clamp(0.0, 1.0),
        )
    }
}

impl fmt::Display for Wdl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "W{} D{} L{} ({} games)",
            self.wins,
            self.draws,
            self.losses,
            self.games()
        )
    }
}

/// Score rate → logistic Elo difference. `s` must be in `[0, 1]`; boundary
/// scores map to `±∞`.
pub fn elo_from_score(s: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&s));
    if s <= 0.0 {
        f64::NEG_INFINITY
    } else if s >= 1.0 {
        f64::INFINITY
    } else {
        -400.0 * (1.0 / s - 1.0).log10()
    }
}

/// The three outcome probabilities under the logistic draw model.
///
/// `g` is the player's Elo advantage over the opponent, `draw_elo` the draw
/// threshold `D ≥ 0`. The three values always sum to 1.
pub fn outcome_probs(g: f64, draw_elo: f64) -> (f64, f64, f64) {
    let p_win = logistic(g - draw_elo);
    let p_loss = logistic(-g - draw_elo);
    let p_draw = 1.0 - p_win - p_loss;
    (p_win, p_draw, p_loss)
}

/// `1 / (1 + 10^(-x/400))`: the Elo → score logistic.
#[inline]
pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-x / 400.0))
}

/// Result of one match game from the *first* engine's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Win,
    Draw,
    Loss,
}

impl GameResult {
    /// The white-relative symbolic result (for PGN tags and reporting).
    pub fn from_white(outcome: crate::selfplay::Outcome) -> GameResult {
        match outcome {
            crate::selfplay::Outcome::WhiteWin => GameResult::Win,
            crate::selfplay::Outcome::Draw => GameResult::Draw,
            crate::selfplay::Outcome::BlackWin => GameResult::Loss,
        }
    }
}

/// Error rates and hypotheses of a sequential probability ratio test.
#[derive(Debug, Clone, Copy)]
pub struct SprtConfig {
    /// H0: the candidate is **not** stronger than this advantage (`elo0`
    /// usually negative, e.g. -2.0).
    pub elo0: f64,
    /// H1: the candidate is at least this advantage stronger (usually small
    /// and positive, e.g. +3.0).
    pub elo1: f64,
    /// Draw threshold of the outcome model (see module docs).
    pub draw_elo: f64,
    /// False-positive rate: accepting H1 when H0 is true.
    pub alpha: f64,
    /// False-negative rate: accepting H0 when H1 is true.
    pub beta: f64,
    /// Hard cap on games before the test is declared inconclusive.
    pub max_games: u64,
}

impl Default for SprtConfig {
    fn default() -> Self {
        SprtConfig {
            elo0: -2.0,
            elo1: 3.0,
            draw_elo: 100.0,
            alpha: 0.05,
            beta: 0.05,
            max_games: 20_000,
        }
    }
}

impl SprtConfig {
    /// H0 probability bounds: `A = ln((1-β)/α)`, `B = ln(β/(1-α))`.
    fn bounds(&self) -> (f64, f64) {
        let a = ((1.0 - self.beta) / self.alpha).ln();
        let b = (self.beta / (1.0 - self.alpha)).ln();
        (a, b)
    }

    /// The log-likelihood ratio contributed by a single game whose outcome is
    /// `r` (from the candidate's point of view): `ln(p1/p0)` where `p1` uses
    /// `elo1` and `p0` uses `elo0`.
    fn llr_one(&self, r: GameResult) -> f64 {
        let (w1, d1, l1) = outcome_probs(self.elo1, self.draw_elo);
        let (w0, d0, l0) = outcome_probs(self.elo0, self.draw_elo);
        let p1 = match r {
            GameResult::Win => w1,
            GameResult::Draw => d1,
            GameResult::Loss => l1,
        };
        let p0 = match r {
            GameResult::Win => w0,
            GameResult::Draw => d0,
            GameResult::Loss => l0,
        };
        (p1 / p0).ln()
    }
}

/// The accumulated state of an SPRT run.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sprt {
    pub config: SprtConfig,
    pub games: u64,
    pub llr: f64,
}

/// A live SPRT decision: keep playing, or accept/reject the candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtDecision {
    /// Neither bound crossed yet — keep playing games.
    Running,
    /// `llr >= A`: the evidence favours the candidate being stronger.
    AcceptH1,
    /// `llr <= B`: the evidence favours the candidate being *not* stronger.
    AcceptH0,
    /// The game cap was reached without a decision; report inconclusively.
    MaxGames,
}

impl Sprt {
    pub fn new(config: SprtConfig) -> Sprt {
        Sprt {
            config,
            games: 0,
            llr: 0.0,
        }
    }

    /// Feeds one game result and returns the updated decision.
    pub fn update(&mut self, r: GameResult) -> SprtDecision {
        self.games += 1;
        self.llr += self.config.llr_one(r);
        self.decision()
    }

    /// The decision implied by the current accumulated `llr`.
    pub fn decision(&self) -> SprtDecision {
        let (a, b) = self.config.bounds();
        if self.llr >= a {
            SprtDecision::AcceptH1
        } else if self.llr <= b {
            SprtDecision::AcceptH0
        } else if self.games >= self.config.max_games {
            SprtDecision::MaxGames
        } else {
            SprtDecision::Running
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_rate_and_elo_round_trip() {
        // 5W 3D 2L: score 6.5/10 = 0.65 → a moderate advantage.
        let w = Wdl {
            wins: 5,
            draws: 3,
            losses: 2,
        };
        assert!((w.score_rate() - 0.65).abs() < 1e-12);
        let elo = w.elo();
        assert!((elo - 400.0 * (0.65f64 / 0.35f64).log10()).abs() < 1e-9);
        assert!(elo > 90.0 && elo < 120.0, "got {elo}"); // 10*log10(13/7)*... ~= 107
    }

    #[test]
    fn elo_flips_with_side_swap() {
        let a = Wdl {
            wins: 10,
            draws: 0,
            losses: 2,
        };
        let b = Wdl {
            wins: 2,
            draws: 0,
            losses: 10,
        };
        assert!(
            (a.elo() + b.elo()).abs() < 1e-9,
            "{} vs {}",
            a.elo(),
            b.elo()
        );
    }

    #[test]
    fn perfect_record_is_infinite_elo() {
        let w = Wdl {
            wins: 8,
            draws: 0,
            losses: 0,
        };
        assert_eq!(w.elo(), f64::INFINITY);
        assert!(w.elo_ci().1.is_infinite(), "upper bound must be +∞");
    }

    #[test]
    fn confidence_interval_narrows_with_games() {
        let few = Wdl {
            wins: 5,
            draws: 0,
            losses: 5,
        };
        let many = Wdl {
            wins: 50,
            draws: 0,
            losses: 50,
        };
        let (flo, fhi) = few.elo_ci();
        let (mlo, mhi) = many.elo_ci();
        // Same point estimate (score 0.5 → elo 0) but the wider sample has a
        // strictly tighter interval.
        assert!(
            (mhi - mlo) < (fhi - flo),
            "wider sample must tighten the CI"
        );
        assert!((few.elo()).abs() < 1e-9 && (many.elo()).abs() < 1e-9);
        assert!(flo < 0.0 && fhi > 0.0);
    }

    #[test]
    fn outcome_probabilities_are_consistent() {
        for g in [-150.0, -50.0, 0.0, 50.0, 150.0] {
            let (w, d, l) = outcome_probs(g, 100.0);
            assert!((w + d + l - 1.0).abs() < 1e-12, "probs must sum to 1");
            assert!(w >= 0.0 && d >= 0.0 && l >= 0.0);
        }
        // Symmetry: a +g advantage gives the mirror of a −g disadvantage.
        let (w1, d1, l1) = outcome_probs(60.0, 100.0);
        let (w2, d2, l2) = outcome_probs(-60.0, 100.0);
        assert!((w1 - l2).abs() < 1e-12 && (l1 - w2).abs() < 1e-12);
        assert!((d1 - d2).abs() < 1e-12, "draw rate is even in g");
        // At g = 0 wins and losses are equal, draws in between.
        let (w0, d0, l0) = outcome_probs(0.0, 100.0);
        assert!((w0 - l0).abs() < 1e-12);
        assert!(d0 > 0.0, "a draw rate must exist at parity");
    }

    #[test]
    fn wins_and_losses_push_llr_in_opposite_directions() {
        let cfg = SprtConfig::default();
        let mut s = Sprt::new(cfg);
        s.update(GameResult::Win);
        assert!(s.llr > 0.0, "a win must favour the candidate: {}", s.llr);
        let mut s = Sprt::new(cfg);
        s.update(GameResult::Loss);
        assert!(
            s.llr < 0.0,
            "a loss must disfavour the candidate: {}",
            s.llr
        );
        // Draws carry a (small) sign too — under elo1 wins are more likely
        // than under elo0, so draws slightly favour the weaker hypothesis
        // only when the draw rates differ; assert the magnitude is tiny.
        let mut s = Sprt::new(cfg);
        s.update(GameResult::Draw);
        assert!(s.llr.abs() < 0.5, "draws should be near-indifferent");
    }

    #[test]
    fn strong_sequence_accepts_h1_within_budget() {
        // A candidate that genuinely outclasses the baseline (say +300 Elo
        // equivalent: 8+ wins per 10, some draws) must hit the upper bound.
        let cfg = SprtConfig {
            max_games: 1_000_000,
            ..SprtConfig::default()
        };
        let (a, _) = cfg.bounds();
        let mut s = Sprt::new(cfg);
        // Simulate a strong player: biased results.
        let mut rng = crate::book::SplitMix64(0xDEAD_BEEF);
        let mut wins = 0;
        let mut losses = 0;
        let mut draws = 0;
        for _ in 0..10_000 {
            // ~65% win, 25% draw, 10% loss through a coarse LCG-free draw.
            let roll = rng.next() % 100;
            let r = if roll < 65 {
                wins += 1;
                GameResult::Win
            } else if roll < 75 {
                draws += 1;
                GameResult::Draw
            } else {
                losses += 1;
                GameResult::Loss
            };
            if s.update(r) == SprtDecision::AcceptH1 {
                return; // accepted: good
            }
        }
        panic!(
            "must accept a genuinely strong candidate within 10k games (llr {}), results W{wins} D{draws} L{losses}, bound {a}",
            s.llr
        );
    }

    #[test]
    fn weak_sequence_accepts_h0() {
        let cfg = SprtConfig::default();
        let _ = cfg.bounds();
        let mut s = Sprt::new(cfg);
        for _ in 0..10_000 {
            // ~65% losses of the candidate.
            let r = if (crate::book::SplitMix64(s.games.wrapping_mul(31) + 7)).next() % 100 < 65 {
                GameResult::Loss
            } else {
                GameResult::Draw
            };
            if s.update(r) == SprtDecision::AcceptH0 {
                return;
            }
        }
        panic!(
            "must reject a weak candidate within 10k games (llr {})",
            s.llr
        );
    }

    #[test]
    fn balanced_play_stays_running() {
        // A 200-game split that is genuinely even must not cross either
        // bound — the test is designed to need ~many games to resolve small
        // deltas.
        let cfg = SprtConfig::default();
        let mut s = Sprt::new(cfg);
        let mut rng = crate::book::SplitMix64(99);
        for _ in 0..200 {
            let roll = rng.next() % 100;
            let r = match roll {
                0..=34 => GameResult::Win,
                35..=64 => GameResult::Draw,
                _ => GameResult::Loss,
            };
            let d = s.update(r);
            assert!(
                d == SprtDecision::Running,
                "200 even games must not resolve: {d:?}"
            );
        }
    }
}

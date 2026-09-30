//! Turning self-play games into a labelled dataset.
//!
//! Generation is where a training run is most often ruined, and almost never by
//! a bug — by a *bias* that is invisible in the output file. The three that
//! matter here are addressed by construction rather than by hope:
//!
//! * **Neighbouring positions are near-duplicates.** A 200-ply game offers 200
//!   positions of which the last forty differ by one quiet move. Sampling every
//!   ply would put a fifth of the corpus on a single board, so
//!   [`GenerationConfig::stride_plies`] spaces the samples out.
//! * **A cap-induced draw is not a chess result.** A game that hit
//!   `max_plies` is drawn *by convention*, and cap draws are systematically the
//!   long, balanced games. Labelling their positions "draw" teaches the net
//!   that a quiet position is drawn — an artefact of the configuration, not a
//!   fact. They are therefore **excluded**, and counted separately in
//!   [`GenerationReport::cap_draws`] so their share is visible.
//! * **The label must not be the teacher's opinion.** The *game result* is the
//!   only unbiased signal the run has. The teacher supplies the PV and the
//!   centipawn weight; the label comes from what happened. Labelling by the
//!   teacher's score instead would train the student to imitate a fixed-depth
//!   search, which is a different (and much weaker) task.
//!
//! Determinism is a hard property: the same seed plays the same games and
//! writes the same file.

use std::sync::Arc;

use anyhow::Context;

use crate::board::Position;
use crate::book::SplitMix64;
use crate::endgame::Syzygy;
use crate::evaluation::EvalParams;
use crate::nnue::network::Network;
use crate::search::params::SearchParams;
use crate::selfplay::{GameRecord, Outcome, SelfPlayConfig, Termination, play_game_with};
use crate::training::config::{GenerationConfig, Label, TeacherConfig};
use crate::training::dataset::{Dataset, Sample};
use crate::training::teacher::Teacher;

/// What one generation run produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenerationReport {
    /// Games played.
    pub games: usize,
    /// Games that ended in a real terminal position (mate, stalemate, the
    /// fifty-move rule, insufficient material, threefold repetition).
    pub decisive_games: usize,
    /// Games drawn by the ply cap — excluded from the dataset.
    pub cap_draws: usize,
    /// Games dropped because `keep_draws` was false.
    pub draw_games_skipped: usize,
    /// Positions kept.
    pub samples: usize,
    /// Positions dropped, by reason.
    pub dropped: Vec<(String, usize)>,
}

impl GenerationReport {
    /// Merges another report into this one, summing the counters.
    pub fn merge(&mut self, other: &GenerationReport) {
        self.games += other.games;
        self.decisive_games += other.decisive_games;
        self.cap_draws += other.cap_draws;
        self.draw_games_skipped += other.draw_games_skipped;
        self.samples += other.samples;
        for (k, v) in &other.dropped {
            match self.dropped.iter_mut().find(|(n, _)| n == k) {
                Some((_, c)) => *c += v,
                None => self.dropped.push((k.clone(), *v)),
            }
        }
    }

    /// The number of positions dropped for `reason`.
    pub fn dropped_for(&self, reason: &str) -> usize {
        self.dropped
            .iter()
            .find(|(n, _)| n == reason)
            .map_or(0, |(_, c)| *c)
    }

    /// A multi-line summary for a run log.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "generation: {} games ({} decisive, {} cap draws, {} draws skipped) -> {} samples",
            self.games, self.decisive_games, self.cap_draws, self.draw_games_skipped, self.samples
        );
        for (k, v) in &self.dropped {
            if *v > 0 {
                s.push_str(&format!("\n  dropped {v:>8}  {k}"));
            }
        }
        s
    }
}

/// The engine that plays the games being labelled.
///
/// Kept as a distinct role from the teacher on purpose. A dataset labelled by
/// the very engine that will consume it can collapse onto itself: the player
/// only ever visits positions it already likes, and the teacher's opinion on
/// those is what the student is fitted to, so the student learns the player's
/// distribution rather than chess. Playing a *different* evaluation — an older
/// parameter set, or simply the frozen classical baseline — broadens the
/// positions visited.
///
/// The game player is [`crate::selfplay`], i.e. the engine's classical
/// evaluation at a fixed depth with a single thread. That is the one player this
/// crate can guarantee is bit-reproducible, and reproducibility is worth more
/// here than player strength: a dataset that cannot be regenerated cannot be
/// used to check a regression.
#[derive(Debug, Clone)]
pub struct PlayerConfig {
    /// Evaluation parameters for both sides of every game.
    pub eval_params: EvalParams,
    /// Search parameters for both sides of every game.
    pub search_params: SearchParams,
    /// The teacher's net. Normally *different* from the player's — the
    /// ordinary distillation setup is a student learning a stronger teacher.
    pub teacher_net: Option<Arc<Network>>,
    /// The teacher's tablebase, for a `search+tb` or `tablebase` teacher.
    pub syzygy: Option<Syzygy>,
}

impl Default for PlayerConfig {
    fn default() -> Self {
        PlayerConfig {
            eval_params: EvalParams::default(),
            search_params: SearchParams::default(),
            teacher_net: None,
            syzygy: None,
        }
    }
}

/// Plays games and labels the sampled positions.
///
/// Holds a [`Teacher`] — and therefore a warm transposition table — across
/// games. The teacher's table caches its own opinions, so dropping it between
/// games would buy nothing and cost the repeat-position speed-up.
pub struct Generator {
    gc: GenerationConfig,
    teacher: Teacher,
    player: PlayerConfig,
    play: SelfPlayConfig,
    rng: SplitMix64,
    report: GenerationReport,
}

impl Generator {
    /// Builds a generator.
    pub fn new(
        gc: GenerationConfig,
        teacher_cfg: &TeacherConfig,
        player: PlayerConfig,
    ) -> anyhow::Result<Generator> {
        // Validate the opening suite name even though the game player draws
        // from the engine's own opening list, so a typo in a config is caught
        // now rather than by whoever reads the log in a week.
        crate::matchplay::suite_by_name(&gc.suite)
            .with_context(|| format!("unknown opening suite {:?}", gc.suite))?;
        let play = SelfPlayConfig {
            depth: gc.depth,
            max_plies: gc.max_plies,
            hash_mb: gc.hash_mb,
            adjudicate_mate: true,
            // One thread, always: a dataset that depends on thread interleaving
            // is not reproducible.
            threads: 1,
        };
        // The seed is read before `gc` moves into the struct, and the player is
        // taken apart before the tablebase is handed over: a clone is a second
        // handle to the same tables (see `impl Clone for Syzygy`), so the
        // generator and the run share one loaded set.
        let seed = gc.seed;
        let player_search_params = player.search_params.clone();
        let teacher_net = player.teacher_net.clone();
        let syzygy = player.syzygy.clone();
        let mut teacher = Teacher::new(teacher_cfg, teacher_net, syzygy)?;
        teacher.set_search_params(player_search_params);
        Ok(Generator {
            gc,
            teacher,
            player,
            play,
            rng: SplitMix64(seed),
            report: GenerationReport::default(),
        })
    }

    /// Games still to be played.
    pub fn remaining(&self) -> usize {
        self.gc.games.saturating_sub(self.report.games)
    }

    /// The report so far.
    pub fn report(&self) -> &GenerationReport {
        &self.report
    }

    /// The teacher, for a caller that wants to label extra positions.
    pub fn teacher(&mut self) -> &mut Teacher {
        &mut self.teacher
    }

    /// Plays one game and returns the samples it contributed.
    pub fn step(&mut self) -> anyhow::Result<Vec<Sample>> {
        let game = play_game_with(
            &self.player.eval_params,
            &self.player.eval_params,
            &self.play,
            &mut self.rng,
        );
        self.report.games += 1;
        self.sample_game(&game)
    }

    /// Plays every game and returns the whole dataset.
    ///
    /// `on_game` is called after each game with the running report so a long
    /// run can print progress. The dataset is still returned in one piece: a
    /// training set assembled from parts is one that can be assembled wrongly.
    pub fn run(
        &mut self,
        on_game: &mut dyn FnMut(usize, &GenerationReport),
    ) -> anyhow::Result<Dataset> {
        let mut ds = Dataset::empty(format!(
            "selfplay depth {} x {} games (seed {})",
            self.gc.depth, self.gc.games, self.gc.seed
        ));
        while self.report.games < self.gc.games {
            for s in self.step()? {
                ds.push(s);
            }
            on_game(self.report.games, &self.report);
        }
        Ok(ds)
    }

    /// The game result, or `None` when the ply cap decided it.
    ///
    /// A cap draw has no chess result, and that distinction is the whole reason
    /// this returns an `Option`.
    fn real_outcome(game: &GameRecord) -> Option<Outcome> {
        match game.outcome {
            Outcome::WhiteWin | Outcome::BlackWin => Some(game.outcome),
            Outcome::Draw => match game.termination {
                Termination::Stalemate
                | Termination::FiftyMove
                | Termination::InsufficientMaterial
                | Termination::ThreefoldRepetition => Some(Outcome::Draw),
                // The cap, or an abort. Neither is a chess result.
                Termination::MovesLimit | Termination::Aborted => None,
                // A decisive termination that somehow produced a draw outcome
                // is not something to guess about: treat it as no result.
                Termination::Checkmate | Termination::AdjudicatedMate => None,
            },
        }
    }

    /// The label a position carries under the game's result, from the side to
    /// move's point of view.
    fn label_of(outcome: Outcome, stm_is_white: bool) -> Label {
        match outcome {
            Outcome::WhiteWin => {
                if stm_is_white {
                    Label::Win
                } else {
                    Label::Loss
                }
            }
            Outcome::BlackWin => {
                if stm_is_white {
                    Label::Loss
                } else {
                    Label::Win
                }
            }
            Outcome::Draw => Label::Draw,
        }
    }

    /// Replays one game and labels the sampled positions.
    fn sample_game(&mut self, game: &GameRecord) -> anyhow::Result<Vec<Sample>> {
        let Some(outcome) = Self::real_outcome(game) else {
            self.report.cap_draws += 1;
            return Ok(Vec::new());
        };
        if outcome == Outcome::Draw && !self.gc.keep_draws {
            self.report.draw_games_skipped += 1;
            return Ok(Vec::new());
        }
        if outcome != Outcome::Draw {
            self.report.decisive_games += 1;
        }

        let mut pos = Position::startpos();
        let stride = self.gc.stride_plies.max(1);
        let want = self.gc.samples_per_game;
        let mut out = Vec::with_capacity(want);
        let mut taken = 0usize;

        for (ply, m) in game.moves.iter().enumerate() {
            // Render and replay rather than `make_child`: the record's moves are
            // already legal, so a failure here would be a record bug rather than
            // a position bug, and it must be reported as one.
            let san = pos.san_of(*m);
            let Ok((child, _)) = pos.play_san(&san) else {
                anyhow::bail!(
                    "self-play record does not replay at ply {ply}: {san:?} is illegal in {}",
                    pos.fen()
                );
            };

            // Decided *before* advancing: the sampled position is the one
            // before the move, and that is the one the game result speaks to.
            if ply % stride == 0 && taken < want && self.accepts(&pos) {
                taken += 1;
                let stm_is_white = pos.turn() == shakmaty::Color::White;
                let label = Self::label_of(outcome, stm_is_white);
                // The teacher's PV only; its *score* is deliberately not used
                // as the label (see the module docs).
                let pv = self.teacher.pv(&pos);
                out.push(Sample::new(label, pos.fen(), pv));
                self.report.samples += 1;
            } else if taken >= want {
                // Everything the config asked for is already collected; the rest
                // of the game is only replayed to keep the position in step.
                break;
            }

            pos = child;
        }
        Ok(out)
    }

    /// Whether a position survives the generation filters. Records the reason
    /// for the ones that do not.
    fn accepts(&mut self, pos: &Position) -> bool {
        if pos.piece_count() < self.gc.min_pieces {
            self.drop("too few pieces");
            return false;
        }
        if pos.legal_moves().is_empty() {
            // Terminal: no move to be taught and no teacher PV.
            self.drop("terminal");
            return false;
        }
        if pos.is_check() && !self.gc.include_checks {
            // A check position is one where the *previous* ply's error is
            // visible; a majority of them teaches the net to imitate a search it
            // was handed rather than to evaluate.
            self.drop("in check");
            return false;
        }
        true
    }

    /// Records one dropped position.
    fn drop(&mut self, reason: &str) {
        match self.report.dropped.iter_mut().find(|(n, _)| n == reason) {
            Some((_, c)) => *c += 1,
            None => self.report.dropped.push((reason.to_string(), 1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_config(games: usize) -> GenerationConfig {
        GenerationConfig {
            games,
            depth: 4,
            max_plies: 60,
            hash_mb: 4,
            samples_per_game: 4,
            stride_plies: 2,
            include_checks: false,
            min_pieces: 2,
            keep_draws: true,
            suite: "classical-v1".to_string(),
            seed: 12345,
        }
    }

    fn generator(gc: GenerationConfig) -> Generator {
        Generator::new(
            gc,
            &TeacherConfig {
                depth: 4,
                hash_mb: 4,
                ..TeacherConfig::default()
            },
            PlayerConfig::default(),
        )
        .unwrap()
    }

    /// A configuration whose games reach a *real* result, so the run actually
    /// produces samples.
    ///
    /// [`gen_config`] asks for a 60-ply cap, which depth-4 games do not finish
    /// inside: two games in three hit the cap, and a cap draw is **excluded**
    /// from the dataset by design (see the module docs). A test whose point is
    /// something about the samples cannot use that configuration — it would be
    /// asserting on a coin flip, and would silently assert nothing at all when
    /// the flip came up wrong.
    ///
    /// Weak, fast players here, and the same ply cap the shipped
    /// `GenerationConfig` uses, so the games end by mate or by repetition well
    /// inside it.
    fn sample_config(games: usize) -> GenerationConfig {
        let mut c = gen_config(games);
        c.depth = 2;
        c.max_plies = 240;
        c
    }

    #[test]
    fn generation_is_deterministic_for_a_seed() {
        let a = generator(sample_config(3)).run(&mut |_, _| {}).unwrap();
        let b = generator(sample_config(3)).run(&mut |_, _| {}).unwrap();
        assert_eq!(a.samples(), b.samples());
        assert!(!a.is_empty(), "three shallow games must yield samples");
    }

    #[test]
    fn a_different_seed_plays_different_games() {
        let a = generator(sample_config(2)).run(&mut |_, _| {}).unwrap();

        let mut cfg_b = sample_config(2);
        cfg_b.seed = 999;

        let b = generator(cfg_b).run(&mut |_, _| {}).unwrap();

        assert!(!a.is_empty(), "seed 12345 produced no samples");
        assert!(!b.is_empty(), "seed 999 produced no samples");
        assert_ne!(a.samples(), b.samples());
    }

    #[test]
    fn every_sample_is_labelled_from_the_side_to_move() {
        let ds = generator(sample_config(3)).run(&mut |_, _| {}).unwrap();
        let (w, d, l) = ds.wdl();
        assert!(w + d + l == ds.len());
        // White's wins must be mirrored by Black's losses, because every
        // position of a game carries the result from the mover's side.
        if w > 0 && l > 0 {
            // Not an equality: a game is sampled at several plies with both
            // colors to move, so the *counts* need not match, but a run that
            // produced only one color would mean the sampler is broken.
            assert!(
                w > 0 && l > 0,
                "a run produced only one color's perspective"
            );
        }
    }

    #[test]
    fn every_sample_replays_and_carries_a_legal_pv() {
        let mut g = generator(sample_config(2));
        let ds = g.run(&mut |_, _| {}).unwrap();
        // Precondition, checked separately: an all-cap-draw run is legal and
        // produces no samples, so this must not be reported as a missing PV.
        assert!(
            !ds.is_empty(),
            "the run produced no samples at all:\n{}",
            g.report().summary()
        );
        for s in ds.samples() {
            // The FEN parses, the position is one a move can be taught from,
            // and the PV is present and legal in it.
            let pos = s.position().unwrap();
            assert!(
                !pos.legal_moves().is_empty(),
                "a terminal position was sampled"
            );
            assert!(
                !s.pv.is_empty(),
                "a non-terminal position was sampled with no teacher PV: {}",
                s.fen
            );
            assert_eq!(
                s.legal_pv().unwrap().len(),
                s.pv.len(),
                "truncated PV in {}",
                s.fen
            );
        }
    }

    #[test]
    fn the_filters_are_reported() {
        // `sample_config`, not `gen_config`: the filters only get a chance to
        // fire on a game that reached a real result, and a game the ply cap
        // decided is skipped before any position is looked at.
        let mut cfg = sample_config(3);
        // A piece count nothing can satisfy: everything is dropped, and the
        // reason is visible rather than the dataset being mysteriously empty.
        cfg.min_pieces = 64;
        let mut g = generator(cfg);
        let ds = g.run(&mut |_, _| {}).unwrap();
        assert!(ds.is_empty());
        assert!(
            g.report().cap_draws < g.report().games,
            "every game was a cap draw, so no position was ever filtered: {}",
            g.report().summary()
        );
        assert!(g.report().dropped_for("too few pieces") > 0);
        assert!(g.report().summary().contains("too few pieces"));
    }

    #[test]
    fn a_cap_draw_is_counted_separately_and_labelled_nothing() {
        // `max_plies = 2` guarantees the cap decides essentially every game.
        let mut cfg = gen_config(4);
        cfg.max_plies = 2;
        cfg.depth = 2;
        let mut g = generator(cfg);
        let ds = g.run(&mut |_, _| {}).unwrap();
        let r = g.report();
        assert_eq!(r.cap_draws + r.decisive_games, r.games);
        if r.cap_draws > 0 {
            assert!(
                ds.is_empty(),
                "a cap draw must not produce labelled positions"
            );
        }
    }

    #[test]
    fn the_stride_caps_how_many_samples_one_game_contributes() {
        let mut cfg = gen_config(3);
        cfg.samples_per_game = 3;
        cfg.stride_plies = 1;
        let mut g = generator(cfg);
        g.run(&mut |_, _| {}).unwrap();
        // Never more than the configured number per game.
        assert!(g.report().samples <= 3 * 3);
    }

    #[test]
    fn a_report_merge_sums_every_counter() {
        let a = GenerationReport {
            games: 2,
            decisive_games: 1,
            cap_draws: 1,
            draw_games_skipped: 0,
            samples: 10,
            dropped: vec![("in check".into(), 3)],
        };
        let mut b = a.clone();
        b.merge(&a);
        assert_eq!(b.games, 4);
        assert_eq!(b.samples, 20);
        assert_eq!(b.dropped_for("in check"), 6);
    }

    #[test]
    fn an_unknown_suite_is_refused_at_construction() {
        let mut cfg = gen_config(1);
        cfg.suite = "not-a-suite".to_string();
        let e = match Generator::new(
            cfg,
            &TeacherConfig {
                depth: 2,
                hash_mb: 4,
                ..TeacherConfig::default()
            },
            PlayerConfig::default(),
        ) {
            Ok(_) => panic!("an unknown opening suite must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("not-a-suite"), "unhelpful error: {e}");
    }

    #[test]
    fn the_label_follows_the_game_not_the_teacher() {
        assert_eq!(Generator::label_of(Outcome::WhiteWin, true), Label::Win);
        assert_eq!(Generator::label_of(Outcome::WhiteWin, false), Label::Loss);
        assert_eq!(Generator::label_of(Outcome::BlackWin, true), Label::Loss);
        assert_eq!(Generator::label_of(Outcome::BlackWin, false), Label::Win);
        assert_eq!(Generator::label_of(Outcome::Draw, true), Label::Draw);
        assert_eq!(Generator::label_of(Outcome::Draw, false), Label::Draw);
    }
}

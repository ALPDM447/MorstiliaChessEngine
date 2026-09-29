//! NNUE training infrastructure.
//!
//! This module is the engine-side half of a training run: it decides **what
//! the data is**, **who labels it**, and **how a candidate net is judged** —
//! and it deliberately does *not* decide how the weights move. There is no
//! optimizer here, no learning rate, no loss. Those belong to the trainer that
//! consumes this configuration, and a half-implemented optimizer living in the
//! engine would produce a config whose fields are read and then ignored, which
//! is the worst kind of infrastructure: it looks like it works.
//!
//! # The pipeline
//!
//! ```text
//!   GenerationConfig  ──►  Generator  ──►  Dataset  ──►  split(train, test)
//!   (self-play)          (filter+label)   (line format)
//!                                                          │
//!                          TeacherConfig  ──►  Teacher  ────┤   labels a position
//!                                       (search / tablebase)│
//!                                                          ▼
//!                                             ValidationConfig ──► a pass/fail
//!                                             (candidate vs reference)
//! ```
//!
//! # The two labels, kept apart on purpose
//!
//! A position in the dataset carries a **label** ([`config::Label`]) and, beside
//! it, a **score** ([`teacher::Labelled`]). They are not the same quantity and
//! neither is a derivative of the other:
//!
//! * the *label* says what happened in the game — the only unbiased signal the
//!   run has;
//! * the *score* is a fixed-depth search's centipawn opinion, kept as a
//!   **weight** ([`teacher::Labelled::confidence`]) and never as a regression
//!   target.
//!
//! Fusing them — regressing on the teacher's centipawn value — is the single
//! most common way an NNUE run wastes its compute, and it is the reason the
//! types here are separate.
//!
//! # What a run guarantees
//!
//! * **Determinism.** Same config + same seed ⇒ same games, same samples, same
//!   file, byte for byte. Every control is a fixed depth; nothing is
//!   wall-clock dependent.
//! * **A readable, replayable artefact.** The dataset is line-oriented text
//!   (FEN + SAN), not a binary blob, so it diffs, greps and survives this code.
//! * **A refusal over a guess.** A configuration that cannot do what it claims
//!   (a tablebase teacher with no tables, a policy objective with a teacher that
//!   supplies no PV) is an error at construction, not a silent degradation.

pub mod config;
pub mod dataset;
pub mod generate;
pub mod teacher;

pub use config::{
    GenerationConfig, Label, SplitConfig, TeacherConfig, TeacherKind, TrainingConfig,
    ValidationConfig,
};
pub use dataset::{DATA_FORMAT, Dataset, Health, Sample};
pub use generate::{GenerationReport, Generator, PlayerConfig};
pub use teacher::{LabelSource, Labelled, Teacher};

use std::sync::Arc;

use crate::endgame::Syzygy;
use crate::nnue::network::Network;

/// Everything a run needs, loaded from one config file.
#[derive(Debug, Clone)]
pub struct Run {
    pub config: TrainingConfig,
    /// The teacher's net, decoded once.
    pub teacher_net: Option<Arc<Network>>,
    /// The tablebase, inert when the path is empty.
    pub syzygy: Syzygy,
}

impl Run {
    /// Loads a config file and prepares its teacher.
    ///
    /// An empty `syzygy_path` yields an inert tablebase, which is what a
    /// search-only teacher wants; a `tablebase` teacher with an inert tablebase
    /// is a *configuration error* and is reported as one rather than being
    /// allowed to label every endgame by search.
    pub fn load(path: impl AsRef<std::path::Path>) -> anyhow::Result<Run> {
        let config = TrainingConfig::load(path)?;
        Run::from_config(config)
    }

    /// Prepares a run from an in-memory config.
    pub fn from_config(config: TrainingConfig) -> anyhow::Result<Run> {
        config.validate()?;
        let teacher_net = match &config.teacher.net {
            Some(p) => {
                let resolved = crate::nnue::resolve_net_path(&p.to_string_lossy());
                Some(Arc::new(crate::nnue::load_network(&resolved).map_err(
                    |e| anyhow::anyhow!("cannot load the teacher net {}: {e}", resolved.display()),
                )?))
            }
            None => match config.teacher.kind {
                // A search teacher with no net uses the classical evaluation,
                // which is a legitimate (much weaker) teacher — but it should
                // not be a silent default for a distillation run.
                TeacherKind::Search | TeacherKind::SearchPlusTablebase => {
                    eprintln!(
                        "training: the teacher has no net; it will label with the classical evaluation"
                    );
                    None
                }
                // A tablebase teacher never needs a net; asking for one would
                // be a wasted 100 MB decode.
                TeacherKind::Tablebase => None,
            },
        };
        let syzygy = if config.syzygy_path.as_os_str().is_empty() {
            Syzygy::none()
        } else {
            let (tb, report) = Syzygy::load(&config.syzygy_path.to_string_lossy());
            if !tb.is_loaded() {
                eprintln!("training: WARNING: no Syzygy tables loaded ({report:?})");
            }
            tb
        };
        Ok(Run {
            config,
            teacher_net,
            syzygy,
        })
    }

    /// Builds the generator this config describes.
    pub fn generator(&self) -> anyhow::Result<Generator> {
        Generator::new(
            self.config.generation.clone(),
            &self.config.teacher,
            PlayerConfig {
                teacher_net: self.teacher_net.clone(),
                // A second handle to the same tables, not a reload: see
                // `impl Clone for Syzygy`.
                syzygy: Some(self.syzygy.clone()),
                ..PlayerConfig::default()
            },
        )
    }

    /// The dataset split the config asks for, from an already-built dataset.
    pub fn split(&self, ds: &Dataset) -> (Dataset, Dataset) {
        ds.split(self.config.split.test_fraction, self.config.split.seed)
    }

    /// Builds a dataset, labels it, writes it, and reports the health audit.
    ///
    /// The health audit is *returned*, not merely printed: a run that produced a
    /// dataset full of duplicates or truncated PVs has produced something the
    /// caller must be able to see programmatically before it is trained on.
    pub fn generate(
        &self,
        on_game: &mut dyn FnMut(usize, &GenerationReport),
    ) -> anyhow::Result<(Dataset, GenerationReport, Health)> {
        let mut gc = self.generator()?;
        let ds = gc.run(on_game)?;
        let report = gc.report().clone();
        let health = ds.validate();
        Ok((ds, report, health))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_config_builds_a_generator() {
        let mut cfg = TrainingConfig::default();
        cfg.generation.games = 1;
        cfg.generation.depth = 3;
        cfg.generation.hash_mb = 4;
        cfg.teacher.depth = 3;
        cfg.teacher.hash_mb = 4;
        let run = Run::from_config(cfg).unwrap();
        let gc = run.generator().unwrap();
        assert_eq!(gc.remaining(), 1);
    }

    #[test]
    fn an_invalid_config_is_refused_before_any_game_is_played() {
        let mut cfg = TrainingConfig::default();
        cfg.generation.games = 0;
        assert!(Run::from_config(cfg).is_err());
    }

    #[test]
    fn a_split_follows_the_config() {
        let run = Run::from_config(TrainingConfig::default()).unwrap();
        let mut ds = Dataset::empty("t");
        for _ in 0..50 {
            ds.push(Sample::without_pv(
                Label::Win,
                "r1bq1rk1/ppp2ppp/2n5/3p4/3P4/2P5/PP1B1PPP/RNBQ1RK1 w - - 0 8",
            ));
        }
        let (train, test) = run.split(&ds);
        assert_eq!(train.len() + test.len(), ds.len());
        assert!(test.len() > 0);
    }

    #[test]
    fn a_tablebase_teacher_without_a_path_is_refused() {
        let mut cfg = TrainingConfig::default();
        cfg.teacher.kind = TeacherKind::Tablebase;
        cfg.syzygy_path = std::path::PathBuf::new();
        let e = Run::from_config(cfg).unwrap_err().to_string();
        assert!(e.contains("SyzygyPath"), "unhelpful error: {e}");
    }
}

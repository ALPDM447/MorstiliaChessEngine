//! Configuration for a training run.
//!
//! Everything a training run needs that is *not* data: how much data, how it
//! is shaped, where the teacher comes from, and how a candidate net is
//! validated. The file is TOML and round-trips, so a run is reproducible from
//! the config plus the data plus the seed.
//!
//! # What is deliberately absent
//!
//! There is no optimizer, no learning rate and no loss weight here. Those
//! belong to the trainer that consumes this configuration, and inventing a
//! half-implemented one in the engine would produce a config file whose fields
//! are read and then ignored — the worst kind of infrastructure, because it
//! looks like it works. What this module owns is the *dataset policy* and the
//! *validation policy*, both of which are engine decisions.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The outcome label of one training position, from the side to move's point of
/// view.
///
/// Three states, not a regression target: the information a game actually
/// carries about a position is its result, and any regression scalar built from
/// it is a lossy re-encoding. Keeping the label discrete means a trainer can
/// choose a distribution over it (a WDL head, a policy over the moves) without
/// the config having to pre-decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Label {
    /// The side to move wins.
    #[default]
    Win,
    /// The game is drawn.
    Draw,
    /// The side to move loses.
    Loss,
}

impl Label {
    /// The `W`/`D`/`L` character used in the text serialisation.
    pub const fn as_char(self) -> char {
        match self {
            Label::Win => 'W',
            Label::Draw => 'D',
            Label::Loss => 'L',
        }
    }

    /// Parses `W`/`D`/`L`, case-insensitive, plus the spelled-out forms.
    pub fn parse(s: &str) -> Option<Label> {
        match s.trim().to_ascii_uppercase().as_str() {
            "W" | "WIN" | "1" => Some(Label::Win),
            "D" | "DRAW" | "0" | "0.5" => Some(Label::Draw),
            "L" | "LOSS" | "-1" => Some(Label::Loss),
            _ => None,
        }
    }

    /// The regression target a WDL head would use. Centipawns, from the side to
    /// move's point of view.
    pub const fn cp_target(self) -> i32 {
        match self {
            Label::Win => 10_000,
            Label::Draw => 0,
            Label::Loss => -10_000,
        }
    }
}

/// How the dataset is split.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitConfig {
    /// Fraction held out for validation, in `(0, 1)`. The remainder trains.
    pub test_fraction: f64,
    /// The shuffle seed. Zero means "do not shuffle", which is only correct
    /// when the input is already in random order — a dataset that arrives
    /// sorted by game outcome would otherwise put every win in the test half.
    pub seed: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        SplitConfig {
            test_fraction: 0.1,
            seed: 1,
        }
    }
}

/// How much data to build and how to shape it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    /// Games of self-play to play.
    pub games: usize,
    /// Fixed search depth per move. Depth control is the only *deterministic*
    /// control: a `movetime` control would make the dataset depend on machine
    /// load, and a dataset that is not reproducible is a dataset that cannot be
    /// regenerated to check a regression.
    pub depth: i32,
    /// Maximum plies per game before it is adjudicated a draw.
    pub max_plies: usize,
    /// Transposition table, MB, per side.
    pub hash_mb: usize,
    /// Positions sampled per game. A 200-ply game offers far more positions
    /// than a dataset of a useful size needs, and sampling *every* ply would
    /// make neighbouring positions near-duplicates.
    pub samples_per_game: usize,
    /// Sample at most one position per this many plies. `0` disables the
    /// stride and relies on `samples_per_game` alone.
    pub stride_plies: usize,
    /// Include positions where the side to move is in check.
    ///
    /// Off by default and worth keeping off unless a run says otherwise: a
    /// check position is one where the *previous* ply's error is visible, and
    /// a majority of them teaches the net to imitate a search it was handed
    /// rather than to evaluate.
    pub include_checks: bool,
    /// Drop positions with fewer than this many pieces on the board. A
    /// three-piece endgame has almost no sparse features left to learn from and
    /// is usually better served by a tablebase.
    pub min_pieces: usize,
    /// Draw games contribute `0` labels by default. With this set they
    /// contribute draws at the usual rate, which is the right choice for a WDL
    /// head and the wrong one for a pure policy objective.
    pub keep_draws: bool,
    /// The opening-suite name for the self-play games.
    pub suite: String,
    /// The self-play seed. The same seed replays the identical dataset.
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            games: 1_000,
            depth: 9,
            max_plies: 240,
            hash_mb: 64,
            samples_per_game: 16,
            stride_plies: 4,
            include_checks: false,
            min_pieces: 4,
            keep_draws: true,
            suite: "classical-v1".to_string(),
            seed: 1,
        }
    }
}

/// Where the teacher's opinions come from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherConfig {
    /// The searcher that supplies scores and PVs.
    pub kind: TeacherKind,
    /// Search depth for the teacher's own searches.
    pub depth: i32,
    /// Transposition table, MB, for the teacher.
    pub hash_mb: usize,
    /// The teacher may use a different net than the student is training, which
    /// is the normal distillation setup. `None` means the embedded net.
    pub net: Option<PathBuf>,
    /// Maximum centipawns a teacher's score may claim. A teacher that is
    /// certain about a position it is not certain about is worse than a weaker
    /// teacher, and this is the only bound on it.
    pub max_score_cp: i32,
    /// A score beyond `±max_score_cp` is treated as a proved mate/loss and
    /// relabelled to the exact WDL. Never silently: a dataset of positions all
    /// labelled "mate found" teaches the net nothing.
    pub fold_decisive: bool,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        TeacherConfig {
            kind: TeacherKind::Search,
            depth: 12,
            hash_mb: 64,
            net: None,
            max_score_cp: 1_500,
            fold_decisive: true,
        }
    }
}

/// Which engine supplies the labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum TeacherKind {
    /// The engine's own search. The default, and the only one that needs no
    /// external component.
    #[default]
    Search,
    /// A Syzygy tablebase probe, used for the endgame tail where it is exact
    /// and the search is guessing.
    Tablebase,
    /// The engine's own search, but every position inside a tablebase's range
    /// takes the tablebase's verdict instead.
    #[serde(rename = "search+tb")]
    SearchPlusTablebase,
}

impl TeacherKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            TeacherKind::Search => "search",
            TeacherKind::Tablebase => "tablebase",
            TeacherKind::SearchPlusTablebase => "search+tb",
        }
    }

    /// Whether this teacher can produce a principal variation.
    ///
    /// A tablebase cannot: it answers an outcome, not a move. A policy head
    /// trained on PVs needs a teacher that can supply one, and a config that
    /// asked for both would otherwise fail deep inside a data file.
    pub const fn supplies_pv(self) -> bool {
        matches!(self, TeacherKind::Search | TeacherKind::SearchPlusTablebase)
    }
}

/// How a candidate net is judged before it is allowed to replace the shipped
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationConfig {
    /// Positions per acceptance test, drawn from the held-out split.
    pub tactical_positions: usize,
    /// The largest centipawn disagreement between the candidate and the
    /// reference on the *reference's* good moves that still counts as a pass.
    /// A tighter bound rejects nets for differences nobody can perceive; a
    /// looser one lets a broken net through.
    pub max_score_delta_cp: i32,
    /// The largest fraction of test positions on which the candidate may pick a
    /// different best move before it is rejected.
    pub max_best_move_churn: f64,
    /// Evaluate on the side to move, or from White's point of view. The
    /// dataset labels are side-to-move, so this must be `true` for a comparison
    /// to mean anything.
    pub stm_evaluation: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        ValidationConfig {
            tactical_positions: 512,
            max_score_delta_cp: 8,
            max_best_move_churn: 0.02,
            stm_evaluation: true,
        }
    }
}

/// Everything one training run needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// Where the dataset is written and read.
    pub data_path: PathBuf,
    /// Where the candidate net is written.
    pub out_net: PathBuf,
    /// A short description embedded in the net header, so the artefact names
    /// its own run.
    pub description: String,
    pub generation: GenerationConfig,
    pub teacher: TeacherConfig,
    pub split: SplitConfig,
    pub validation: ValidationConfig,
    /// Where a Syzygy tablebase lives, for the `tablebase` teachers. Empty
    /// keeps the tablebase inert rather than guessing a path.
    pub syzygy_path: PathBuf,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            data_path: PathBuf::from("training/data.txt"),
            out_net: PathBuf::from("training/candidate.nnue"),
            description: "morstilia training candidate".to_string(),
            generation: GenerationConfig::default(),
            teacher: TeacherConfig::default(),
            split: SplitConfig::default(),
            validation: ValidationConfig::default(),
            syzygy_path: PathBuf::new(),
        }
    }
}

impl TrainingConfig {
    /// Reads a TOML file.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<TrainingConfig> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read training config {}: {e}", path.display()))?;
        TrainingConfig::from_toml_str(&text)
            .map_err(|e| anyhow::anyhow!("invalid training config {}: {e}", path.display()))
    }

    /// Parses TOML. Unknown fields are an error, so a stale config that names a
    /// knob this engine no longer has fails loudly instead of being read as if
    /// it had been honoured.
    pub fn from_toml_str(text: &str) -> anyhow::Result<TrainingConfig> {
        toml::from_str(text).map_err(Into::into)
    }

    /// Serializes to TOML.
    pub fn to_toml_string(&self) -> anyhow::Result<String> {
        toml::to_string_pretty(self).map_err(Into::into)
    }

    /// Writes the config as TOML.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, self.to_toml_string()?)
            .map_err(|e| anyhow::anyhow!("cannot write {}: {e}", path.display()))
    }

    /// Fails unless the configuration is internally consistent.
    ///
    /// Checked up front rather than discovered halfway through a run: every
    /// condition here is a mistake that would otherwise cost hours to notice.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.generation.games == 0 {
            anyhow::bail!("generation.games must be >= 1");
        }
        if self.generation.depth < 1 {
            anyhow::bail!("generation.depth must be >= 1");
        }
        if self.generation.samples_per_game == 0 {
            anyhow::bail!("generation.samples_per_game must be >= 1");
        }
        if self.generation.max_plies < 2 {
            anyhow::bail!("generation.max_plies must be >= 2");
        }
        if !(0.0..1.0).contains(&self.split.test_fraction) {
            anyhow::bail!(
                "split.test_fraction must be in [0, 1) — 1.0 would leave no training data"
            );
        }
        if self.teacher.depth < 1 {
            anyhow::bail!("teacher.depth must be >= 1");
        }
        if self.teacher.max_score_cp < 1 {
            anyhow::bail!("teacher.max_score_cp must be >= 1");
        }
        if !self.teacher.kind.supplies_pv() && self.validation.tactical_positions > 0 {
            // Not fatal: a tablebase-only teacher can still validate a net on
            // *scores*. It is worth saying out loud, though, because the
            // obvious mistake here is to expect PVs that cannot exist.
            eprintln!(
                "training: the {} teacher supplies no PV; policy validation will be skipped",
                self.teacher.kind.as_str()
            );
        }
        Ok(())
    }

    /// The data file's format version this config writes.
    pub const fn data_format(&self) -> u32 {
        crate::training::dataset::DATA_FORMAT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip_through_their_text_form() {
        for l in [Label::Win, Label::Draw, Label::Loss] {
            assert_eq!(Label::parse(&l.as_char().to_string()), Some(l));
        }
        assert_eq!(Label::parse(" win "), Some(Label::Win));
        assert_eq!(Label::parse("DRAW"), Some(Label::Draw));
        assert_eq!(Label::parse("0.5"), Some(Label::Draw));
        assert_eq!(Label::parse("maybe"), None);
    }

    #[test]
    fn cp_targets_are_antisymmetric_about_a_draw() {
        assert_eq!(Label::Win.cp_target(), -Label::Loss.cp_target());
        assert_eq!(Label::Draw.cp_target(), 0);
    }

    #[test]
    fn default_config_is_valid_and_round_trips() {
        let c = TrainingConfig::default();
        c.validate().expect("the default config must be valid");
        let text = c.to_toml_string().unwrap();
        let back = TrainingConfig::from_toml_str(&text).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn a_tampered_config_is_rejected_rather_than_ignored() {
        let text = TrainingConfig::default().to_toml_string().unwrap();
        // Append a knob this engine does not have.
        let tampered = format!("{text}\nnot_a_real_knob = 3\n");
        assert!(TrainingConfig::from_toml_str(&tampered).is_err());
    }

    #[test]
    fn validate_rejects_the_degenerate_settings() {
        let mut c = TrainingConfig::default();
        c.generation.games = 0;
        assert!(c.validate().is_err());

        let mut c = TrainingConfig::default();
        c.split.test_fraction = 1.0;
        assert!(c.validate().is_err(), "a 100% test split trains on nothing");

        let mut c = TrainingConfig::default();
        c.teacher.max_score_cp = 0;
        assert!(c.validate().is_err());

        let mut c = TrainingConfig::default();
        c.generation.samples_per_game = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn only_search_teachers_supply_a_pv() {
        assert!(TeacherKind::Search.supplies_pv());
        assert!(TeacherKind::SearchPlusTablebase.supplies_pv());
        assert!(!TeacherKind::Tablebase.supplies_pv());
    }

    #[test]
    fn a_tablebase_teacher_config_is_still_valid() {
        // A tablebase-only run cannot do policy validation, but it must not be
        // *rejected* for that: it is a legitimate way to train the endgame.
        let mut c = TrainingConfig::default();
        c.teacher.kind = TeacherKind::Tablebase;
        c.validate()
            .expect("a tablebase teacher is a legal configuration");
    }

    #[test]
    fn a_saved_config_reloads_identically() {
        let dir = std::env::temp_dir().join(format!("morstilia-train-{}", std::process::id()));
        let path = dir.join("train.toml");
        let mut c = TrainingConfig::default();
        c.generation.games = 7;
        c.description = "round trip".to_string();
        c.save(&path).unwrap();
        assert_eq!(TrainingConfig::load(&path).unwrap(), c);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

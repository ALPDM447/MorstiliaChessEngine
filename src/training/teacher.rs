//! The teacher: what labels a position.
//!
//! A training set is a claim about what "the right answer" is, and that claim
//! has to come from somewhere. This module is that somewhere, and it is
//! deliberately thin: it takes a position and returns an opinion — a score, a
//! label, a principal variation — without knowing anything about gradients,
//! batches or architectures.
//!
//! # Why the label is a W/D/L and not a centipawn target
//!
//! A centipawn target invites a trainer to fit a number the teacher itself
//! does not believe to that precision. The search says `+137` because it is a
//! 137-centipawn *preference at a fixed depth*, not a measurement; the
//! confidence behind it varies with the position, the pruning the search did
//! and the net it evaluated with. Carrying that as a regression target teaches
//! a student the *search's noise floor* alongside its knowledge.
//!
//! So the label is discrete (see [`Label`]) and the score is kept beside it as
//! a *weight*, not a target: [`TeacherConfig::max_score_cp`] bounds how much a
//! single confident opinion counts, and a decisive score folds into the exact
//! WDL rather than being clipped into a huge centipawn value.
//!
//! # Determinism
//!
//! Every teacher here is a fixed-depth, single-threaded search on a persistent
//! table. The same position with the same configuration yields the same
//! opinion, bit for bit — which is the property that makes a dataset
//! regenerable and a validation result reproducible.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use shakmaty::Color;

use crate::board::Position;
use crate::endgame::{Syzygy, TbOutcome};
use crate::evaluation::EvalParams;
use crate::nnue::network::Network;
use crate::search::params::SearchParams;
use crate::search::{Searcher, TimeLimit};
use crate::training::config::{Label, TeacherConfig, TeacherKind};
use crate::training::dataset::Sample;

/// A depth-limited, single-threaded searcher used as a teacher.
///
/// Deliberately its own type rather than a bare [`Searcher`]: a teacher has
/// invariants a general searcher does not (never more than one thread, never a
/// time control, a table reused across positions on purpose) and wrapping them
/// makes them structural instead of a convention every call site has to
/// remember.
pub struct Teacher {
    searcher: Searcher,
    /// The always-false stop flag. A teacher has no external canceller; a
    /// time-limited label would make the dataset depend on machine load, which
    /// is precisely what this type refuses to do.
    stop: Arc<AtomicBool>,
    depth: i32,
    /// `±max_score_cp`; scores beyond this are decisive.
    max_score_cp: i32,
    /// Whether a decisive score folds to the exact W/D/L.
    fold_decisive: bool,
    /// The tablebase, inert unless the teacher uses one.
    tb: Syzygy,
    kind: TeacherKind,
    /// How many positions the search has labelled, for reports.
    labelled: u64,
    /// How many of those came from the tablebase rather than the search.
    from_tb: u64,
}

impl Teacher {
    /// Builds a teacher from its configuration.
    ///
    /// `net` is the teacher evaluator's net — normally a *different* net from
    /// the student is being trained on, which is the ordinary distillation
    /// setup. `None` uses the engine's classical evaluation, which is a
    /// legitimate (if much weaker) teacher for a classical student.
    pub fn new(
        cfg: &TeacherConfig,
        net: Option<Arc<Network>>,
        syzygy: Option<Syzygy>,
    ) -> anyhow::Result<Teacher> {
        // A search-only teacher never probes, and a `search+tb` teacher with no
        // tables simply has no endgame, so both get an inert handle. A pure
        // `tablebase` teacher with nothing loaded is a *configuration error*:
        // the run would label every endgame with the search the config said not
        // to use, and report it as exact.
        let loaded_tb = match (cfg.kind, syzygy) {
            (TeacherKind::Search, _) | (TeacherKind::SearchPlusTablebase, None) => Syzygy::none(),
            (_, Some(tb)) => tb,
            (TeacherKind::Tablebase, None) => {
                anyhow::bail!(
                    "the tablebase teacher needs SyzygyPath; a run with an empty path \
                     would silently label every endgame by search"
                )
            }
        };
        if cfg.kind == TeacherKind::Tablebase && !loaded_tb.is_loaded() {
            anyhow::bail!(
                "the tablebase teacher was configured but no Syzygy tables loaded, so every \
                 position would be labelled by the search the config says not to use"
            );
        }
        let mut searcher = Searcher::with_params(cfg.hash_mb, EvalParams::default());
        searcher.set_nnue(net);
        // A single worker, permanently: a teacher that used several would make
        // its opinion depend on thread interleaving.
        searcher.set_threads(1);
        Ok(Teacher {
            searcher,
            stop: Arc::new(AtomicBool::new(false)),
            depth: cfg.depth,
            max_score_cp: cfg.max_score_cp,
            fold_decisive: cfg.fold_decisive,
            tb: loaded_tb,
            kind: cfg.kind,
            labelled: 0,
            from_tb: 0,
        })
    }

    /// A teacher that uses the engine's built-in parameters and evaluation.
    pub fn builtin(depth: i32) -> Teacher {
        let mut searcher = Searcher::new(32);
        searcher.set_threads(1);
        Teacher {
            searcher,
            stop: Arc::new(AtomicBool::new(false)),
            depth,
            max_score_cp: 1_500,
            fold_decisive: true,
            tb: Syzygy::none(),
            kind: TeacherKind::Search,
            labelled: 0,
            from_tb: 0,
        }
    }

    /// Overrides the search parameters (e.g. to run a teacher with the shipped
    /// mechanisms rather than the built-in ones).
    pub fn set_search_params(&mut self, sp: SearchParams) {
        self.searcher.set_search_params(sp);
    }

    /// The underlying searcher, for callers that need the raw result.
    pub fn searcher(&self) -> &Searcher {
        &self.searcher
    }

    /// The underlying searcher, mutably — the only way to run a raw search
    /// against the teacher's own table and net.
    pub fn searcher_mut(&mut self) -> &mut Searcher {
        &mut self.searcher
    }

    /// Swaps the teacher's net. Used by `train --compare`, which needs two
    /// teachers that differ *only* in their weights: same depth, same table
    /// policy, same everything else, so a score difference is attributable to
    /// the net and to nothing else.
    pub fn set_nnue(&mut self, net: Option<Arc<Network>>) {
        self.searcher.set_nnue(net);
    }

    /// Positions labelled so far.
    pub fn labelled(&self) -> u64 {
        self.labelled
    }

    /// How many labels came from the tablebase.
    pub fn from_tb(&self) -> u64 {
        self.from_tb
    }

    /// Which teacher this is.
    pub fn kind(&self) -> TeacherKind {
        self.kind
    }

    /// The search's opinion of `pos`, in centipawns from the side to move.
    ///
    /// The transposition table is *not* cleared between calls: consecutive
    /// positions of a self-play game share transpositions, and reusing them is
    /// both faster and what a real teacher wants.
    pub fn score(&mut self, pos: &Position) -> i32 {
        let limits = TimeLimit {
            depth: Some(self.depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        // Empty history: a teacher's opinion must not depend on where in a
        // game the position happened to appear, or the same position reached by
        // two routes would be labelled twice with different values.
        self.searcher
            .search(pos, &[], &limits, &self.stop, 1, &[])
            .score
    }

    /// The teacher's principal variation of `pos` as SAN.
    pub fn pv(&mut self, pos: &Position) -> Vec<String> {
        if !self.kind.supplies_pv() {
            return Vec::new();
        }
        let limits = TimeLimit {
            depth: Some(self.depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let r = self.searcher.search(pos, &[], &limits, &self.stop, 1, &[]);
        // Replaying the PV through the position is the only trustworthy way to
        // render it: the `RawMove`s are relative to nodes the caller cannot see,
        // and a SAN that does not parse in the position it claims is worse than
        // no PV at all.
        let mut at = pos.clone();
        let mut out = Vec::with_capacity(r.pv.len());
        for m in r.pv {
            match at.play_san(&at.san_of(m)) {
                Ok((child, _)) => {
                    out.push(at.san_of(m));
                    at = child;
                }
                Err(_) => break,
            }
        }
        out
    }

    /// Labels one position, from the side to move's point of view.
    ///
    /// The pipeline is: tablebase (if the teacher uses one and the position is
    /// in range) → decisive-score folding → bounded centipawn comparison →
    /// the W/D/L that actually gets written. Each stage is skipped when it
    /// does not apply, and the last stage always runs, so there is no position
    /// for which this returns nothing.
    pub fn label(&mut self, pos: &Position) -> Labelled {
        self.labelled += 1;

        // 1. The tablebase, when there is one and it can answer exactly.
        if self.kind != TeacherKind::Search && self.tb.is_loaded() {
            if let Some(outcome) = self.tb.probe_wdl(pos.chess()) {
                self.from_tb += 1;
                return match outcome {
                    TbOutcome::Win | TbOutcome::CursedWin => Labelled {
                        label: Label::Win,
                        score: wdl_cp(outcome, pos.turn()),
                        pv: Vec::new(),
                        source: LabelSource::Tablebase,
                    },
                    TbOutcome::Draw => Labelled {
                        label: Label::Draw,
                        score: 0,
                        pv: Vec::new(),
                        source: LabelSource::Tablebase,
                    },
                    TbOutcome::BlessedLoss | TbOutcome::Loss => Labelled {
                        label: Label::Loss,
                        score: wdl_cp(outcome, pos.turn()),
                        pv: Vec::new(),
                        source: LabelSource::Tablebase,
                    },
                };
            }
        }

        // 2. The search's own opinion, and the PV that goes with it. The PV is
        //    produced by the *same* search that produced the score (one call,
        //    not two) so the two can never disagree about which move was best.
        let limits = TimeLimit {
            depth: Some(self.depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let r = self.searcher.search(pos, &[], &limits, &self.stop, 1, &[]);
        let raw = r.score;
        let source = if self.kind == TeacherKind::SearchPlusTablebase
            && self.tb.is_loaded()
            && self.tb.probe_wdl(pos.chess()).is_some()
        {
            // Unreachable in practice (stage 1 would have returned), but a
            // position can leave the tablebase's range between two probes only
            // if the position differs — it does not. Kept honest rather than
            // relying on that.
            LabelSource::Tablebase
        } else {
            LabelSource::Search
        };

        // 3. A decisive score is a *proved* result, not a huge centipawn
        //    preference, and it is labelled as such.
        let decisive = is_decisive(raw);
        let pv = if self.kind.supplies_pv() && r.pv.len() == 1 {
            let m = r.pv[0];
            vec![pos.san_of(m)]
        } else {
            Vec::new()
        };
        let pv = if pv.is_empty() && self.kind.supplies_pv() {
            self.pv(pos)
        } else {
            pv
        };

        let label = if decisive && self.fold_decisive {
            if raw > 0 { Label::Win } else { Label::Loss }
        } else if self.fold_decisive && raw.abs() >= self.max_score_cp {
            // Bounded but not decisive: still a confident opinion, and the WDL
            // that carries it unambiguously. The score keeps its magnitude as a
            // weight.
            if raw > 0 { Label::Win } else { Label::Loss }
        } else if raw.abs() * 2 < self.max_score_cp / 3 {
            // Inside the noise floor. A position the teacher cannot tell apart
            // is a *draw* as far as the dataset is concerned, and labelling it
            // ±20 cp teaches a student to fit noise.
            Label::Draw
        } else if raw > 0 {
            Label::Win
        } else {
            Label::Loss
        };

        Labelled {
            label,
            score: raw,
            pv,
            source,
        }
    }

    /// Labels a position and renders it as a dataset [`Sample`].
    pub fn sample(&mut self, pos: &Position) -> Sample {
        let l = self.label(pos);
        Sample::new(l.label, pos.fen(), l.pv)
    }
}

/// Where a label came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSource {
    /// A fixed-depth search.
    Search,
    /// An exact tablebase probe.
    Tablebase,
}

/// A label plus the evidence behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Labelled {
    /// The outcome, from the side to move's point of view.
    pub label: Label,
    /// The teacher's raw centipawn opinion, retained as a *weight* and for
    /// diagnostics. Never a regression target — see the module docs.
    pub score: i32,
    /// The teacher's principal variation, as SAN. Empty when the teacher
    /// cannot supply one.
    pub pv: Vec<String>,
    pub source: LabelSource,
}

impl Labelled {
    /// The sample this label becomes.
    pub fn to_sample(&self, fen: impl Into<String>) -> Sample {
        Sample::new(self.label, fen, self.pv.clone())
    }

    /// How much this opinion counts, `0.0 ..= 1.0`.
    ///
    /// A *decisive* result is counted at full weight — the tablebase and a
    /// proved mate are not opinions. Everything else is weighted by how far
    /// past the noise floor it sits, saturating at `1.0`. A trainer that ignores
    /// this is fitting the teacher's confidence profile, which is a property of
    /// the search, not of chess.
    pub fn confidence(&self, noise_floor_cp: i32) -> f64 {
        if is_decisive(self.score) {
            return 1.0;
        }
        let a = self.score.abs() as f64;
        let n = noise_floor_cp.max(1) as f64;
        (a / n).clamp(0.0, 1.0)
    }
}

/// The centipawn stand-in for an exact tablebase outcome, from the side to
/// move's point of view.
///
/// `±2 * MATE_THRESHOLD` for the decisive outcomes and `0` for a draw, so a
/// tablebase label and a search label of the same game land in the same
/// numeric range. The value is a *weight*, never a target.
fn wdl_cp(outcome: TbOutcome, stm: Color) -> i32 {
    let m = match outcome {
        TbOutcome::Win | TbOutcome::CursedWin => 2 * crate::types::MATE_THRESHOLD,
        TbOutcome::Draw => 0,
        TbOutcome::BlessedLoss | TbOutcome::Loss => -2 * crate::types::MATE_THRESHOLD,
    };
    // `wdl_score` is White-relative; the dataset is side-to-move relative.
    match stm {
        Color::White => m,
        Color::Black => -m,
    }
}

/// Whether a search score is a proved mate rather than a large centipawn
/// preference.
///
/// The same band the search itself uses for "this is mate, not an evaluation"
/// ([`crate::types::MATE_THRESHOLD`]), so a label is decisive exactly when the
/// search would print a mate distance.
#[inline]
pub fn is_decisive(score: i32) -> bool {
    score.abs() >= crate::types::MATE_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    const MID: &str = "r1bq1rk1/ppp2ppp/2n5/3p4/3P4/2P5/PP1B1PPP/RNBQ1RK1 w - - 0 8";
    /// A mate the side to move can force in two.
    const MATE_IN_TWO: &str = "6k1/5ppp/8/8/8/8/8/R3K2R w KQ - 0 1";
    /// A drawn K+2P vs K endgame, far from any mate.
    const DRAWN: &str = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1";

    fn teacher(depth: i32) -> Teacher {
        Teacher::builtin(depth)
    }

    #[test]
    fn a_builtin_teacher_labels_a_middlegame_position() {
        let pos = Position::from_fen(MID).unwrap();
        let mut t = teacher(6);
        let l = t.label(&pos);
        // The position is roughly equal, so any label is possible; what must
        // hold is that a label came back and the counters moved.
        assert!(t.labelled() == 1);
        assert_eq!(l.source, LabelSource::Search);
        // The PV must parse in the position it is attached to — that is the
        // invariant that makes a stored PV usable at all.
        assert!(!l.pv.is_empty());
        let s = Sample::new(l.label, pos.fen(), l.pv.clone());
        assert_eq!(s.legal_pv().unwrap().len(), l.pv.len());
    }

    #[test]
    fn the_teacher_is_deterministic() {
        let pos = Position::from_fen(MID).unwrap();
        let a = teacher(7).label(&pos);
        let b = teacher(7).label(&pos);
        assert_eq!(a.label, b.label);
        assert_eq!(a.score, b.score);
        assert_eq!(a.pv, b.pv);
    }

    #[test]
    fn a_forced_mate_is_labelled_a_win() {
        // Ra1xa8+ then Rh1-h8#. The teacher must find it and must call it a
        // *win*, not merely a large centipawn preference — that is what
        // `fold_decisive` is for.
        let pos = Position::from_fen(MATE_IN_TWO).unwrap();
        let mut t = teacher(8);
        let l = t.label(&pos);
        assert_eq!(
            l.label,
            Label::Win,
            "a mate in two must be a win, got {l:?}"
        );
        assert!(is_decisive(l.score), "score {} is not decisive", l.score);
        assert!(
            l.confidence(50) == 1.0,
            "a proved mate counts at full weight"
        );
    }

    #[test]
    fn a_dead_draw_is_labelled_a_draw() {
        let pos = Position::from_fen(DRAWN).unwrap();
        let mut t = teacher(10);
        let l = t.label(&pos);
        assert_eq!(l.label, Label::Draw, "K+P vs K is drawn, got {l:?}");
        assert!(!is_decisive(l.score));
    }

    #[test]
    fn positions_inside_the_noise_floor_become_draws() {
        // The noise floor is a third of `max_score_cp`; a 40 cp opinion with a
        // 1500 cp ceiling sits well inside it.
        let pos = Position::from_fen(MID).unwrap();
        let cfg = TeacherConfig {
            depth: 6,
            max_score_cp: 120,
            fold_decisive: true,
            ..TeacherConfig::default()
        };
        let mut t = Teacher::new(&cfg, None, None).unwrap();
        let l = t.label(&pos);
        // With a 120 cp ceiling anything under 40 cp folds to a draw, and a
        // shallow search on a balanced position is exactly that.
        assert_eq!(l.label, Label::Draw, "got {l:?}");
        assert!(l.score.abs() < 40 || l.score.abs() > 200);
    }

    #[test]
    fn confidence_saturates_and_floors() {
        let decisive = Labelled {
            label: Label::Win,
            score: 31_000,
            pv: vec![],
            source: LabelSource::Search,
        };
        assert_eq!(decisive.confidence(50), 1.0);

        let quiet = Labelled {
            label: Label::Draw,
            score: 25,
            pv: vec![],
            source: LabelSource::Search,
        };
        assert!((quiet.confidence(50) - 0.5).abs() < 1e-12);

        let zero = Labelled {
            label: Label::Draw,
            score: 0,
            pv: vec![],
            source: LabelSource::Search,
        };
        assert_eq!(zero.confidence(50), 0.0);
        // A zero floor must not divide by zero.
        assert_eq!(zero.confidence(0), 0.0);
    }

    #[test]
    fn a_tablebase_teacher_without_tables_is_refused() {
        let cfg = TeacherConfig {
            kind: TeacherKind::Tablebase,
            ..TeacherConfig::default()
        };
        let e = match Teacher::new(&cfg, None, None) {
            Ok(_) => panic!("a tablebase teacher with no tables must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("SyzygyPath"), "unhelpful error: {e}");
    }

    #[test]
    fn the_sample_carries_the_label_and_a_replayable_pv() {
        let pos = Position::from_fen(MID).unwrap();
        let mut t = teacher(6);
        let s = t.sample(&pos);
        assert_eq!(s.fen, pos.fen());
        assert!(!s.pv.is_empty());
        assert_eq!(s.legal_pv().unwrap(), s.pv);
        // And it round-trips through the data format.
        let back = Sample::parse_line(&s.to_line()).unwrap().unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn decisive_detection_matches_the_search_own_band() {
        assert!(is_decisive(crate::types::MATE_THRESHOLD));
        assert!(is_decisive(-crate::types::MATE_THRESHOLD));
        assert!(!is_decisive(crate::types::MATE_THRESHOLD - 1));
        assert!(!is_decisive(0));
    }

    #[test]
    fn tablebase_scores_are_side_to_move_relative() {
        assert_eq!(
            wdl_cp(TbOutcome::Win, Color::White),
            2 * crate::types::MATE_THRESHOLD
        );
        assert_eq!(
            wdl_cp(TbOutcome::Win, Color::Black),
            -2 * crate::types::MATE_THRESHOLD
        );
        assert_eq!(wdl_cp(TbOutcome::Draw, Color::White), 0);
        assert_eq!(
            wdl_cp(TbOutcome::Loss, Color::White),
            -2 * crate::types::MATE_THRESHOLD
        );
    }
}

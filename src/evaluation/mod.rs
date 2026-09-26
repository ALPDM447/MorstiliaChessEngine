//! Classical (pre-NNUE) tapered evaluation.
//!
//! # Architecture
//!
//! The public entry point is [`Evaluator::evaluate`]. Internally the
//! evaluation is a sum of independent, individually testable terms, each in a
//! dedicated submodule:
//!
//! * [`material`] — piece values, bishop pair
//! * [`pieces`] — piece-square tables (PST)
//! * [`activity`] — piece activity and coordination: outposts, bad bishop,
//!   development, connected rooks, rook behind a passed pawn
//! * [`pawns`] — pawn structure: doubled / isolated / backward / connected /
//!   islands
//! * [`passed_pawns`] — passed pawns (with protected/connected bonuses)
//! * [`mobility`]
//! * [`king`] — king safety: pawn shield, attack units, open files near the
//!   king, pawn holes
//! * [`threats`] — threats, rook files/seventh rank, space
//! * [`phase`] — game-phase blend for tapered scores
//!
//! Every term returns a [`Score`] `{ mg, eg }`; the final number is a
//! phase-weighted blend expressed from the side-to-move's point of view.
//! The interface is deliberately narrow (`Evaluator` with no parameters but
//! the position) so that an NNUE can later be swapped in behind the same
//! `evaluate` signature.
//!
//! # Instrumentation
//!
//! [`Evaluator::evaluate_parts`] returns the per-component breakdown
//! ([`EvalParts`]) the search's `--stats` and UCI-debug paths print without
//! recomputing anything: thirteen numbers (material, pst, pawn, mobility,
//! king, pieces, passed, space, threats, phase, final) per evaluated
//! position. The breakdown is white-relative (`final` is already
//! side-to-move-relative).

pub mod activity;
pub mod king;
pub mod material;
pub mod mobility;
pub mod params;
pub mod passed_pawns;
pub mod pawns;
pub mod phase;
pub mod pieces;
pub mod threats;

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use std::sync::OnceLock;

use shakmaty::Position as _;

pub use material::PIECE_VALUES;
pub use params::EvalParams;

/// The lazily-initialized baseline parameter set backing the parameterless
/// [`Evaluator::evaluate`] / [`Evaluator::evaluate_parts`] entry points. The
/// search hot path avoids this entirely: it passes the searcher's own
/// `&EvalParams` through [`Evaluator::evaluate_with`].
pub(crate) fn default_params() -> &'static EvalParams {
    static PARAMS: OnceLock<EvalParams> = OnceLock::new();
    PARAMS.get_or_init(EvalParams::default)
}

/// Midgame/endgame pair used by the tapered evaluation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Score {
    pub mg: i32,
    pub eg: i32,
}

impl Score {
    #[inline]
    pub const fn new(mg: i32, eg: i32) -> Score {
        Score { mg, eg }
    }

    #[inline]
    pub const fn mg_only(mg: i32) -> Score {
        Score { mg, eg: 0 }
    }

    #[inline]
    pub const fn eg_only(eg: i32) -> Score {
        Score { mg: 0, eg }
    }

    #[inline]
    pub const fn zero() -> Score {
        Score { mg: 0, eg: 0 }
    }

    /// Doubles the value (used for small binary contributions).
    #[inline]
    pub fn double(self) -> Score {
        Score::new(self.mg * 2, self.eg * 2)
    }
}

impl Add for Score {
    type Output = Score;
    #[inline]
    fn add(self, rhs: Score) -> Score {
        Score::new(self.mg + rhs.mg, self.eg + rhs.eg)
    }
}

impl Sub for Score {
    type Output = Score;
    #[inline]
    fn sub(self, rhs: Score) -> Score {
        Score::new(self.mg - rhs.mg, self.eg - rhs.eg)
    }
}

impl Neg for Score {
    type Output = Score;
    #[inline]
    fn neg(self) -> Score {
        Score::new(-self.mg, -self.eg)
    }
}

impl Mul<i32> for Score {
    type Output = Score;
    #[inline]
    fn mul(self, rhs: i32) -> Score {
        Score::new(self.mg * rhs, self.eg * rhs)
    }
}

impl AddAssign for Score {
    #[inline]
    fn add_assign(&mut self, rhs: Score) {
        self.mg += rhs.mg;
        self.eg += rhs.eg;
    }
}

impl SubAssign for Score {
    #[inline]
    fn sub_assign(&mut self, rhs: Score) {
        self.mg -= rhs.mg;
        self.eg -= rhs.eg;
    }
}

/// Blends a score by the game phase (`phase` goes 0 → endgame … 24 → opening).
///
/// The 24 half-point divisor is the classic Stockfish-style constant; the
/// parameterized phase is fed through [`tapered_at`] so a tuned `phase_max`
/// stays in sync with the taper.
#[inline]
pub fn tapered(score: Score, phase: i32) -> i32 {
    tapered_at(score, phase, 24)
}

/// [`tapered`] with an explicit phase maximum — used by the parameterized
/// evaluation so a tuned `phase_max` and the taper divisor can never drift.
#[inline]
pub fn tapered_at(score: Score, phase: i32, phase_max: i32) -> i32 {
    let max = phase_max.max(1);
    (score.mg * phase + score.eg * (max - phase)) / max
}

/// The per-component breakdown of one evaluation, produced by
/// [`Evaluator::evaluate_parts`] without recomputing any term.
///
/// All `Score` components are white-relative (`mg`/`eg` halves of the
/// tapered blend); [`EvalParts::final_score`] is the final blended value,
/// already expressed from the side to move's point of view and already
/// zeroed on insufficient material.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvalParts {
    /// Material + bishop pair.
    pub material: Score,
    /// Piece-square tables.
    pub pst: Score,
    /// Pawn structure (doubled/isolated/backward/connected/islands).
    pub pawn: Score,
    /// Piece mobility.
    pub mobility: Score,
    /// King safety (shield, attack units, open files, holes).
    pub king: Score,
    /// Piece activity/coordination (outposts, bad bishop, development, rook terms).
    pub pieces: Score,
    /// Passed pawns (with protected/connected increments).
    pub passed: Score,
    /// Pawn-space advantage.
    pub space: Score,
    /// Threats + rook open/semi-open/seventh-rank bonuses.
    pub threats: Score,
    /// Game phase, 0 (endgame) .. 24 (opening).
    pub phase: i32,
    /// Raw (white-relative) midgame total: the sum of every `*_mg` component
    /// before blending.
    pub total_mg: i32,
    /// Raw (white-relative) endgame total before blending.
    pub total_eg: i32,
    /// The final blended, side-to-move-relative score (`Evaluator::evaluate`).
    pub final_score: i32,
}

impl EvalParts {
    /// The phase-weighted (blended) value of a single component, white-relative.
    #[inline]
    pub fn blended(&self, s: Score) -> i32 {
        tapered(s, self.phase)
    }

    /// Sum of every positional/material component (white-relative), before the
    /// side-to-move sign and the insufficient-material zero.
    pub fn total(&self) -> Score {
        self.material
            + self.pst
            + self.pawn
            + self.mobility
            + self.king
            + self.pieces
            + self.passed
            + self.space
            + self.threats
    }

    /// One-line numerical breakdown used by `--bench --stats` and the UCI
    /// debug path (`info string estat ...`). Not hot-path code.
    pub fn stats_row(&self) -> String {
        format!(
            "mg {} eg {} material {} pst {} pawn {} mobility {} king {} pieces {} passed {} space {} threats {} phase {} final {}",
            self.total_mg,
            self.total_eg,
            self.blended(self.material),
            self.blended(self.pst),
            self.blended(self.pawn),
            self.blended(self.mobility),
            self.blended(self.king),
            self.blended(self.pieces),
            self.blended(self.passed),
            self.blended(self.space),
            self.blended(self.threats),
            self.phase,
            self.final_score,
        )
    }
}

/// The engine's static evaluation of a position, from the point of view of
/// the side to move.
#[derive(Default)]
pub struct Evaluator;

impl Evaluator {
    /// Evaluates `pos` in centipawns, positive when good for the side to
    /// move, using the baseline parameter set.
    #[inline]
    pub fn evaluate(&self, pos: &crate::board::Position) -> i32 {
        self.evaluate_with(pos, default_params())
    }

    /// Evaluates `pos` with a specific tunable parameter set (the search hot
    /// path; `p` is a shared reference owned by the search, so nothing is
    /// copied or rebuilt per node).
    #[inline]
    pub fn evaluate_with(&self, pos: &crate::board::Position, p: &EvalParams) -> i32 {
        self.evaluate_parts_with(pos, p).final_score
    }

    /// Evaluates `pos` from **White's** point of view — the convention Texel
    /// tuning and dataset scoring use (the ordinary [`Evaluator::evaluate`]
    /// returns the value from the side to move's perspective, which flips
    /// every ply). Positional terms are already white-relative, so this is
    /// just the pre-sign tapered total; it costs nothing beyond
    /// [`Evaluator::evaluate_parts_with`].
    #[inline]
    pub fn evaluate_white_with(&self, pos: &crate::board::Position, p: &EvalParams) -> i32 {
        let parts = self.evaluate_parts_with(pos, p);
        tapered_at(
            Score::new(parts.total_mg, parts.total_eg),
            parts.phase,
            p.phase_max,
        )
    }

    /// Computes the full evaluation and its per-component breakdown with the
    /// baseline parameter set.
    pub fn evaluate_parts(&self, pos: &crate::board::Position) -> EvalParts {
        self.evaluate_parts_with(pos, default_params())
    }

    /// Computes the full evaluation and its per-component breakdown for an
    /// explicit parameter set.
    ///
    /// Cheap enough for the hot path (stack-only, no allocation: one
    /// [`pawns::PawnInfo`] scan and one [`passed_pawns::PassedInfo`] scan are
    /// shared by every term); the breakdown is what the instrumentation
    /// prints, so the search never evaluates twice.
    pub fn evaluate_parts_with(&self, pos: &crate::board::Position, p: &EvalParams) -> EvalParts {
        let board = pos.board();
        let phase = phase::game_phase(board, p);
        let pawn_info = pawns::PawnInfo::scan(board);
        let passed_info = passed_pawns::PassedInfo::scan(&pawn_info);

        let material = material::evaluate_material(board, p);
        let pst = pieces::evaluate_pst(board, p);
        let pawn = pawns::evaluate_pawns(&pawn_info, p);
        let mobility = mobility::evaluate_mobility(board, &pawn_info, p);
        let king = king::evaluate_king_safety(board, &pawn_info, p);
        let pieces = activity::evaluate_pieces(board, &pawn_info, &passed_info, phase, p);
        let passed = passed_pawns::evaluate_passed(board, &pawn_info, &passed_info, p);
        let space = threats::evaluate_space(board, &pawn_info, p);
        let threats = threats::evaluate_threats(board, &pawn_info, p)
            + threats::evaluate_rooks(board, &pawn_info, p);

        let total = material + pst + pawn + mobility + king + pieces + passed + space + threats;
        let mut value = tapered_at(total, phase, p.phase_max);

        // Draw detection used directly by the search (material-only draws).
        if pos.chess().is_insufficient_material() {
            value = 0;
        }
        let final_score = if pos.turn() == shakmaty::Color::Black {
            -value
        } else {
            value
        };

        EvalParts {
            material,
            pst,
            pawn,
            mobility,
            king,
            pieces,
            passed,
            space,
            threats,
            phase,
            total_mg: total.mg,
            total_eg: total.eg,
            final_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;

    #[test]
    fn startpos_eval_is_symmetric_and_small() {
        let ev = Evaluator;
        let v = ev.evaluate(&Position::startpos());
        assert_eq!(v, 0, "startpos must be exactly equal");
    }

    #[test]
    fn startpos_components_are_balanced_to_zero() {
        // Every term is color-symmetric by construction at the initial
        // position: each side contributes the same, so each component is 0.
        let parts = Evaluator.evaluate_parts(&Position::startpos());
        for (name, s) in [
            ("material", parts.material),
            ("pst", parts.pst),
            ("pawn", parts.pawn),
            ("mobility", parts.mobility),
            ("king", parts.king),
            ("pieces", parts.pieces),
            ("passed", parts.passed),
            ("space", parts.space),
            ("threats", parts.threats),
        ] {
            assert_eq!(
                s.mg, s.eg,
                "{name} component is balanced mg==eg at startpos: {s:?}"
            );
            assert_eq!(s.mg, 0, "{name} must vanish at startpos: {s:?}");
        }
        assert_eq!(parts.phase, 24);
        assert_eq!(parts.final_score, 0);
    }

    #[test]
    fn swapped_colors_eval_negates() {
        // Mirror the position (flip colors) — a mirrored eval must negate.
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 4 4";
        let pos = Position::from_fen(fen).unwrap();
        let fen2 = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 4 4";
        let pos2 = Position::from_fen(fen2).unwrap();
        let v1 = Evaluator.evaluate(&pos);
        let v2 = Evaluator.evaluate(&pos2);
        assert_eq!(v1, -v2, "color mirror must negate eval");
    }

    #[test]
    fn vertical_color_mirror_is_consistent() {
        // A genuine mirror: fold the board vertically (rank 1 ↔ 8) and swap
        // colors, with the *same* side to move afterwards. Every new term
        // (bad bishop, outposts, development, islands, holes, open files near
        // the king) must flip sign exactly.
        let original = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP2PPP/R2Q1RK1 w - - 0 1";
        let mirrored = "r2q1rk1/ppp2ppp/2nbpn2/3p1b2/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1";
        let v_w = Evaluator.evaluate(&Position::from_fen(original).unwrap());
        let v_m = Evaluator.evaluate(&Position::from_fen(mirrored).unwrap());
        assert_eq!(
            v_w, -v_m,
            "vertical colour mirror must negate: {v_w} vs {v_m}"
        );
    }

    #[test]
    fn eval_parts_are_consistent_with_evaluate() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 4 4",
            "6k1/8/8/8/3P4/8/8/4K3 w - - 0 1",
            "6k1/5Q2/8/8/8/8/8/4K3 b - - 0 1",
            "r2q1rk1/ppp2ppp/2nbpn2/3p1b2/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",
        ] {
            let pos = Position::from_fen(fen).expect("valid test FEN");
            let parts = Evaluator.evaluate_parts(&pos);
            assert_eq!(
                parts.final_score,
                Evaluator.evaluate(&pos),
                "parts.final must equal evaluate for {fen}"
            );
            // The blended component sum (before the STM sign) must agree with
            // the white-relative value used by the taper. Integer division in
            // `tapered` truncates each component independently, so allow the
            // per-component rounding error (at most ~1cp per component).
            let mut sum = 0i32;
            for s in [
                parts.material,
                parts.pst,
                parts.pawn,
                parts.mobility,
                parts.king,
                parts.pieces,
                parts.passed,
                parts.space,
                parts.threats,
            ] {
                sum += parts.blended(s);
            }
            let stm_adjusted = if pos.turn() == shakmaty::Color::Black {
                -sum
            } else {
                sum
            };
            assert!(
                (stm_adjusted - parts.final_score).abs() <= 9,
                "component blend must reproduce final (within truncation) for {fen}: {stm_adjusted} vs {}",
                parts.final_score
            );
        }
    }

    #[test]
    fn material_dominates() {
        // White has three rooks against one black rook (plus both kings).
        let pos = Position::from_fen("7k/8/8/8/8/8/r7/RR1RK3 w - - 0 1").unwrap();
        let v = Evaluator.evaluate(&pos);
        assert!(
            v > 800,
            "white up material must be strongly positive, got {v}"
        );
    }

    #[test]
    fn up_a_queen_is_positive() {
        let pos = Position::from_fen("6k1/8/8/8/8/8/5Q2/6K1 w - - 0 1").unwrap();
        assert!(Evaluator.evaluate(&pos) > 700);
        let pos2 = Position::from_fen("6k1/8/8/8/8/8/5q2/6K1 w - - 0 1").unwrap();
        assert!(Evaluator.evaluate(&pos2) < -700);
    }

    #[test]
    fn tapered_blend_interpolates_linearly() {
        // A pure-mg value must fade to its pure-eg value as phase drops.
        let s = Score::new(24, 0);
        assert_eq!(tapered(s, 24), 24); // full opening
        assert_eq!(tapered(s, 12), 12); // mid-game
        assert_eq!(tapered(s, 0), 0); // full endgame
        let s2 = Score::new(0, 48);
        assert_eq!(tapered(s2, 12), 24);
        assert_eq!(tapered(s2, 0), 48);
        // Even phases tap into both halves: (24*6 + 48*18)/24 = (144+864)/24 = 42.
        assert_eq!(tapered(Score::new(24, 48), 6), 42);
    }

    #[test]
    fn passed_pawn_on_seventh_rank_is_winning() {
        // The single strongest classical term: a passer on the 7th ranks high
        // in both phases and must dominate the position.
        let pos = Position::from_fen("6k1/3P4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let v = Evaluator.evaluate(&pos);
        assert!(v > 400, "seventh-rank passer must be winning: {v}");
    }

    #[test]
    fn centralized_king_beats_cornered_king_in_endgame() {
        // Same material (K+R vs K), the only difference is white king and rook
        // centralization. The endgame-weighted PST/activity terms must see it.
        let corner = Position::from_fen("7k/8/8/8/8/8/8/KR6 w - - 0 1").unwrap();
        let central = Position::from_fen("7k/8/8/3R4/3K4/8/8/8 w - - 0 1").unwrap();
        let v_corner = Evaluator.evaluate(&corner);
        let v_central = Evaluator.evaluate(&central);
        assert!(
            v_central > v_corner,
            "centralized pieces must beat the corner: {v_central} vs {v_corner}"
        );
    }

    #[test]
    fn stats_row_carries_every_component() {
        let parts = Evaluator.evaluate_parts(
            &Position::from_fen(
                "r2q1rk1/ppp2ppp/2nbpn2/3p1b2/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",
            )
            .unwrap(),
        );
        let row = parts.stats_row();
        for token in [
            "material", "pst", "pawn", "mobility", "king", "pieces", "passed", "space", "threats",
            "phase", "final",
        ] {
            assert!(row.contains(token), "stats_row missing {token}: {row}");
        }
    }

    #[test]
    fn params_baseline_reproduces_legacy_eval() {
        // The shipped baseline `EvalParams` reproduces the parameterless
        // `Evaluator::evaluate` exactly, both from the in-code default and
        // after a TOML round-trip (the `config/baseline_eval.toml` path).
        let p = EvalParams::default();
        let toml_p = EvalParams::from_toml_str(&p.to_toml_string().unwrap()).unwrap();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 4 4",
            "r2q1rk1/ppp2ppp/2nbpn2/3p1b2/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",
            "6k1/8/8/8/3P4/8/8/4K3 w - - 0 1",
            "7k/1R6/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let v = Evaluator.evaluate(&pos);
            assert_eq!(
                Evaluator.evaluate_with(&pos, &p),
                v,
                "default params must reproduce evaluate for {fen}"
            );
            assert_eq!(
                Evaluator.evaluate_with(&pos, &toml_p),
                v,
                "TOML round-tripped params must reproduce evaluate for {fen}"
            );
        }
    }
}

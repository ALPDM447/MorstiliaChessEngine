//! Pruning margins and depth gates for the main search.
//!
//! The rules themselves live in `alphabeta.rs`; this module keeps the tuning
//! constants and their pure helpers together so they are easy to adjust and
//! unit-test. All margins are in centipawns and grow with depth, because a
//! deeper search accumulates more (uncertain) positional value.
//!
//! Every rule has **two** forms:
//!
//! * a `const fn` with the formula spelled out as literals, and
//! * a `*_sp` function that reads the same formula from
//!   [`SearchParams`](crate::search::params::SearchParams).
//!
//! The two are the *same* function at the default parameter values, and a unit
//! test (`default_parameters_reproduce_the_compiled_formulas`) asserts that
//! exhaustively over the whole depth range. That is the mechanism behind the
//! "moving a threshold into the parameters must not change behaviour" contract:
//! the literals are the reference, the test is the proof, and no threshold can
//! be moved into the parameter set without the test noticing a drift.
//!
//! The `const fn` forms are retained because the depth-gate relationships
//! between them are checked at compile time (`depths_are_sane`), which a
//! runtime parameter set cannot express.

use crate::search::params::SearchParams;
use crate::types::Depth;

/// Shallow-depth gate for per-move futility pruning.
pub const FUTILITY_DEPTH: Depth = 4;

/// Shallow-depth gate for razoring.
pub const RAZOR_DEPTH: Depth = 3;

/// Gate for reverse futility (static null) pruning.
pub const RFP_DEPTH: Depth = 6;

/// Null move requires at least this remaining depth (and non-PV nodes).
pub const NULL_MOVE_MIN_DEPTH: Depth = 2;

/// Null move (probing) requires at least this remaining depth. ProbCut's
/// probe runs at `depth - PROBCUT_DEPTH`, so the gate keeps the probe ≥ 1.
pub const PROBCUT_DEPTH: Depth = 5;

/// Extra margin used by delta pruning in quiescence (optimism: a capture
/// might gain up to this much beyond the victim value, e.g. removing a
/// defender).
pub const DELTA_MARGIN: i32 = 200;

/// Margin for skipping a quiet move: the static eval plus this must still be
/// below `alpha` for the move to be futile.
#[inline]
pub const fn futility_margin(depth: Depth, improving: bool) -> i32 {
    let base = 100 + 75 * depth;
    if improving { base - 25 } else { base }
}

/// Margin for razoring: below `alpha - margin` the node is not worth the full
/// search at all; drop straight into quiescence.
#[inline]
pub const fn razor_margin(depth: Depth) -> i32 {
    230 + 90 * depth
}

/// Margin for reverse futility pruning: if the static eval already beats
/// `beta` by this much, the node cannot fail high enough to matter.
#[inline]
pub const fn reverse_futility_margin(depth: Depth, improving: bool) -> i32 {
    let base = 80 + 70 * depth;
    if improving { base - 25 } else { base }
}

/// Margin for ProbCut: the static eval must beat `beta` by this much before
/// the shallow proof-search is worth running. Grows with depth like the other
/// margins, because a deeper search leaves more room for the opponent to
/// claw a static advantage back.
#[inline]
pub const fn probcut_margin(depth: Depth) -> i32 {
    110 + 85 * depth
}

/// Null move reduction: plies skipped by the null move; grows with depth and
/// is deeper by one in "stagnant" positions (static eval did not improve
/// since two plies ago), where a pass is more likely to hold — the classic
/// Stockfish-style `r += improving ? 0 : 1`.
#[inline]
pub const fn null_move_reduction(depth: Depth, improving: bool) -> Depth {
    3 + depth / 4 + if improving { 0 } else { 1 }
}

/// SEE threshold below which a losing capture is pruned in the main search
/// (more tolerant at depth, so deeper searches still notice spectacular
/// hanging pieces).
#[inline]
pub const fn see_prune_threshold(depth: Depth) -> i32 {
    -80 * depth
}

/// A quiet move that fails this hard in history and is not among the first
/// few moves is pruned outright (history-based move-count pruning).
pub const HISTORY_PRUNE_THRESHOLD: i32 = -8_000;

/// After this many searched quiet moves, later quiets become cheap to prune
/// at shallow depths.
pub const QUIET_PRUNE_LIMIT: usize = 5;

/// [`futility_margin`] with the coefficients read from `sp`.
#[inline]
pub fn futility_margin_sp(sp: &SearchParams, depth: Depth, improving: bool) -> i32 {
    let base = sp.futility_margin_base + sp.futility_margin_depth * depth;
    if improving {
        base - sp.futility_margin_improving
    } else {
        base
    }
}

/// [`razor_margin`] with the coefficients read from `sp`.
#[inline]
pub fn razor_margin_sp(sp: &SearchParams, depth: Depth) -> i32 {
    sp.razor_margin_base + sp.razor_margin_depth * depth
}

/// [`reverse_futility_margin`] with the coefficients read from `sp`.
#[inline]
pub fn reverse_futility_margin_sp(sp: &SearchParams, depth: Depth, improving: bool) -> i32 {
    let base = sp.rfp_margin_base + sp.rfp_margin_depth * depth;
    if improving {
        base - sp.rfp_margin_improving
    } else {
        base
    }
}

/// [`probcut_margin`] with the coefficients read from `sp`.
#[inline]
pub fn probcut_margin_sp(sp: &SearchParams, depth: Depth) -> i32 {
    sp.probcut_margin_base + sp.probcut_margin_depth * depth
}

/// [`null_move_reduction`] with the coefficients read from `sp`.
#[inline]
pub fn null_move_reduction_sp(sp: &SearchParams, depth: Depth, improving: bool) -> Depth {
    sp.null_move_reduction_base
        + depth / sp.null_move_reduction_div
        + if improving {
            0
        } else {
            sp.null_move_reduction_stagnant
        }
}

/// [`see_prune_threshold`] with the coefficients read from `sp`.
#[inline]
pub fn see_prune_threshold_sp(sp: &SearchParams, depth: Depth) -> i32 {
    -(sp.see_prune_base + sp.see_prune_depth * depth)
}

/// Stockfish 19 Step 8 — razoring **at the child** (`search.cpp:991-992`).
///
/// ```text
/// if (!PvNode && eval < alpha - 482 * depth * depth)
///     return qsearch<NonPV>(pos, ss, alpha, beta);
/// ```
///
/// Evaluated in the parent immediately after playing the move, so the child is
/// dropped into quiescence without ever entering the main search. The quadratic
/// depth term is what makes this safe near the horizon: at `depth == 1` the
/// margin is 482 and at `depth == 8` it is 30_848, so a deep child needs an
/// enormous deficit before it can be abandoned.
///
/// Distinct from the node-level razor in this module, which *probes* quiescence
/// and only returns if the probe confirms the fail-low. The child form is
/// cheaper (no probe branch back into the main search) and slightly more
/// aggressive; both may be active, in which case the parent's runs first and
/// the child's never sees the position.
#[inline]
pub fn sf19_child_razor_margin(sp: &SearchParams, depth: Depth) -> i32 {
    sp.child_razor_margin_base * depth * depth
}

/// Stockfish 19 Step 9 — reverse futility **at the child**
/// (`search.cpp:996-1007`).
///
/// ```text
/// futilityMult  = min(45 + depth * 4, 85) - 20 * !ttHit
/// futilityMargin = futilityMult * depth
///                - (2789 * improving + 335 * opponentWorsening) * mult / 1024
///                + |correctionValue| / 198435
/// if (eval - futilityMargin >= beta)
///     return (661 * beta + 363 * eval) / 1024;
/// ```
///
/// The blend `(661 * beta + 363 * eval) / 1024` is Stockfish's *smoothed* fail
/// high: it returns a score strictly between `beta` and `eval` rather than
/// `eval` itself, so the value stays a valid bound with slack. Returning a bare
/// `eval` would be a stronger claim than a reduced child search can support.
///
/// `opponent_worsening` is the LMR hindsight term
/// (`ss->reducedDepth` / `opponentWorsening` at `search.cpp:867`); Morstilia
/// passes it in from [`SearchThread::prior_reduction`], and it is zero when the
/// `lmr_hindsight` gate is off.
///
/// The depth gate is `< 19` (`seekMate ? 6 : 19`) and is *not* tuned in
/// Stockfish — the comment says so — so the bound is exposed only so the
/// constant is visible in one place, not so it can be moved casually.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn sf19_child_rfp_margin(
    sp: &SearchParams,
    depth: Depth,
    improving: bool,
    opponent_worsening: bool,
    tt_hit: bool,
    correction_value: i32,
) -> i32 {
    let mult = (sp.child_rfp_mult_base + depth * sp.child_rfp_mult_depth)
        .min(sp.child_rfp_mult_cap)
        - sp.child_rfp_no_tt * i32::from(!tt_hit);
    let hindsight = sp.child_rfp_improving * i32::from(improving)
        + sp.child_rfp_worsening * i32::from(opponent_worsening);
    mult * depth - hindsight * mult / sp.child_rfp_hindsight_div
        + correction_value.abs() / sp.child_rfp_correction_div
}

/// Stockfish 19's smoothed fail-high value for the child RFP
/// (`search.cpp:1007`).
#[inline]
pub fn sf19_child_rfp_value(sp: &SearchParams, beta: i32, eval: i32) -> i32 {
    (sp.child_rfp_value_beta * beta + sp.child_rfp_value_eval * eval) / sp.child_rfp_value_den
}

/// True when `side` still has at least one attacking piece (queen, rook,
/// bishop or knight) on `board`. The null-move gate refuses to pass in pure
/// king-and-pawn positions: there passing is a real move and zugzwang can
/// flip the result.
pub fn side_has_attacking_pieces(board: &shakmaty::Board, side: shakmaty::Color) -> bool {
    use shakmaty::Role;
    !board.by_piece(Role::Queen.of(side)).is_empty()
        || !board.by_piece(Role::Rook.of(side)).is_empty()
        || !board.by_piece(Role::Bishop.of(side)).is_empty()
        || !board.by_piece(Role::Knight.of(side)).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margins_grow_with_depth() {
        for d in 1..=12 {
            assert!(futility_margin(d, true) > futility_margin(d - 1, true));
            assert!(razor_margin(d) > razor_margin(d - 1));
            assert!(reverse_futility_margin(d, true) > reverse_futility_margin(d - 1, true));
        }
    }

    #[test]
    fn improving_variant_is_smaller() {
        for d in 1..=8 {
            assert!(futility_margin(d, true) < futility_margin(d, false));
            assert!(reverse_futility_margin(d, true) < reverse_futility_margin(d, false));
        }
    }

    #[test]
    fn depths_are_sane() {
        // Compile-time constants, so the relationships are checked by the
        // compiler rather than at run time.
        const {
            assert!(FUTILITY_DEPTH >= RAZOR_DEPTH);
            assert!(RFP_DEPTH > FUTILITY_DEPTH);
            assert!(NULL_MOVE_MIN_DEPTH >= 2);
            assert!(PROBCUT_DEPTH >= 5, "probcut needs a deep enough probe");
            assert!(RAZOR_DEPTH < PROBCUT_DEPTH);
        }
    }

    #[test]
    fn null_reduction_is_positive_and_bounded() {
        for d in 1..=24 {
            for improving in [true, false] {
                let r = null_move_reduction(d, improving);
                assert!(r >= 3, "null move always skips at least 3 plies");
                assert!(r <= 11, "adaptive +1 must stay bounded: got {r}");
            }
        }
    }

    #[test]
    fn null_reduction_is_deeper_when_not_improving() {
        for d in 1..=24 {
            assert!(
                null_move_reduction(d, false) > null_move_reduction(d, true),
                "a stagnant position must null-reduce one ply more at depth {d}"
            );
        }
    }

    #[test]
    fn probcut_margin_grows_with_depth() {
        for d in PROBCUT_DEPTH..=16 {
            assert!(probcut_margin(d) > probcut_margin(d - 1));
            assert!(probcut_margin(d) >= 300, "must be a meaningful margin");
        }
    }

    #[test]
    fn side_has_attacking_pieces_reflects_material() {
        use shakmaty::{Color, Position as _};
        // The opening: both sides have attacking pieces.
        let opening =
            crate::testutil::chess("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(side_has_attacking_pieces(opening.board(), Color::White));
        assert!(side_has_attacking_pieces(opening.board(), Color::Black));

        // A pure king-and-pawn endgame: neither side may pass (zugzwang).
        let pawns = crate::testutil::chess("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
        assert!(!side_has_attacking_pieces(pawns.board(), Color::White));
        assert!(!side_has_attacking_pieces(pawns.board(), Color::Black));

        // One white knight changes only white's gate.
        let knight = crate::testutil::chess("4k3/8/8/8/8/8/2N5/4K3 w - - 0 1");
        assert!(side_has_attacking_pieces(knight.board(), Color::White));
        assert!(!side_has_attacking_pieces(knight.board(), Color::Black));
        // Queens, rooks and bishops each count too.
        let big = crate::testutil::chess("3q1rk1/p4ppp/8/8/8/8/P4PPP/3R1RK1 w - - 0 1");
        for side in [Color::White, Color::Black] {
            assert!(side_has_attacking_pieces(big.board(), side));
        }
    }

    #[test]
    fn null_probe_always_searches_less_than_the_move_loop() {
        // The nulled child must be searched strictly shallower than a real
        // move (`d - 1 - R < d - 1`), so a failed probe never costs more than
        // the first real move it replaces — even at the minimum depth where
        // the probe drops straight into quiescence.
        for d in NULL_MOVE_MIN_DEPTH..=24 {
            for improving in [true, false] {
                let reduced = d - 1 - null_move_reduction(d, improving);
                assert!(
                    reduced < d - 1,
                    "null probe at depth {d} (improving={improving}) must reduce: got {reduced}"
                );
            }
        }
    }

    #[test]
    fn probcut_probe_depth_is_never_negative() {
        // The probe runs at `depth - PROBCUT_DEPTH`: at the gate depth that is
        // a qsearch (0), deeper it is a real reduced search. Either way it
        // must be strictly less than `depth` (the probe is cheap) and never
        // negative.
        for d in PROBCUT_DEPTH..=PROBCUT_DEPTH + 20 {
            let probe = d - PROBCUT_DEPTH;
            assert!(
                (0..d).contains(&probe),
                "probe at depth {d} must be in [0, {d}): got {probe}"
            );
        }
    }

    /// The contract that makes it safe to move a threshold into
    /// [`SearchParams`]: at the default values the parameterised formulas are
    /// *identically* the compiled literal ones. Checked over the full depth
    /// range and both `improving` states, for every margin the search uses.
    #[test]
    fn default_parameters_reproduce_the_compiled_formulas() {
        let sp = SearchParams::default();
        for d in -4..=64 {
            for improving in [true, false] {
                assert_eq!(
                    futility_margin_sp(&sp, d, improving),
                    futility_margin(d, improving),
                    "futility_margin at depth {d}, improving={improving}"
                );
                assert_eq!(
                    reverse_futility_margin_sp(&sp, d, improving),
                    reverse_futility_margin(d, improving),
                    "reverse_futility_margin at depth {d}, improving={improving}"
                );
                assert_eq!(
                    null_move_reduction_sp(&sp, d, improving),
                    null_move_reduction(d, improving),
                    "null_move_reduction at depth {d}, improving={improving}"
                );
            }
            assert_eq!(
                razor_margin_sp(&sp, d),
                razor_margin(d),
                "razor_margin at {d}"
            );
            assert_eq!(
                probcut_margin_sp(&sp, d),
                probcut_margin(d),
                "probcut_margin at {d}"
            );
            assert_eq!(
                see_prune_threshold_sp(&sp, d),
                see_prune_threshold(d),
                "see_prune_threshold at {d}"
            );
        }
        assert_eq!(sp.futility_depth, FUTILITY_DEPTH);
        assert_eq!(sp.razor_depth, RAZOR_DEPTH);
        assert_eq!(sp.rfp_depth, RFP_DEPTH);
        assert_eq!(sp.null_move_min_depth, NULL_MOVE_MIN_DEPTH);
        assert_eq!(sp.probcut_depth, PROBCUT_DEPTH);
        assert_eq!(sp.history_prune_threshold, HISTORY_PRUNE_THRESHOLD);
        assert_eq!(sp.quiet_prune_limit as usize, QUIET_PRUNE_LIMIT);
    }

    /// The Stockfish 19 child formulas must reproduce `sf_19`'s constants at
    /// the default parameter values. The numbers are written out here
    /// independently of [`SearchParams`] on purpose: a test that simply called
    /// the same accessor would agree with any drift.
    #[test]
    fn child_pruning_reproduces_the_stockfish_constants() {
        let sp = SearchParams::default();
        for d in 0..=24 {
            assert_eq!(
                sf19_child_razor_margin(&sp, d),
                482 * d * d,
                "child razor at depth {d}"
            );
        }
        for d in 0..=24 {
            for improving in [true, false] {
                for tt_hit in [true, false] {
                    for worsening in [true, false] {
                        let mult = (45 + 4 * d).min(85) - 20 * i32::from(!tt_hit);
                        let hint = 2789 * i32::from(improving) + 335 * i32::from(worsening);
                        let want = mult * d - hint * mult / 1024;
                        assert_eq!(
                            sf19_child_rfp_margin(&sp, d, improving, worsening, tt_hit, 0),
                            want,
                            "child RFP at depth {d} improving={improving} tt_hit={tt_hit} worse={worsening}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            sf19_child_rfp_margin(&sp, 6, true, false, true, 1_984_350),
            {
                let mult = (45 + 24).min(85) - 0;
                mult * 6 - 2789 * mult / 1024 + 1_984_350 / 198435
            }
        );
        assert_eq!(
            sf19_child_rfp_value(&sp, 200, 400),
            (661 * 200 + 363 * 400) / 1024
        );
    }
}

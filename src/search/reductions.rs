//! Late move reductions (LMR).
//!
//! Quiet moves searched late in the move list are usually refuted cheaply, so
//! they are first searched with a reduced depth; if the reduced search still
//! beats `alpha` the move is re-searched at full depth. The reduction amount
//! follows the classic `ln(depth) * ln(move_index)` curve, tempered by node
//! type (PV nodes reduce less) and whether the position is "improving"
//! (the static eval increased since the previous ply).
//!
//! A second, *modern* model is layered on top and ports the Stockfish 19
//! `sf_19` reduction formula (`search.cpp:1157-1403`, `1885-1888`): a pure
//! logarithmic ramp scaled by the window, the node type, the TT entry, the
//! cutoff history, the move's own history score, and an all-node multiplier.
//! The classic curve and [`allows_lmr`] are both kept — the modern model
//! supplies an *amount*, and never the *permission* to reduce.

use std::sync::LazyLock;

use crate::search::params::SearchParams;
use crate::search::singular::is_decisive;
use crate::types::{Depth, MAX_DEPTH};

/// Row length of the LMR table (indexed by 1-based move index).
const TABLE_MOVES: usize = 256;

/// `reduction[depth][move_index]`; computed once at first use with no heap in
/// the hot path (a plain table read afterwards).
static LMR_TABLE: LazyLock<[[u8; TABLE_MOVES]; MAX_DEPTH as usize + 1]> = LazyLock::new(|| {
    let mut t = [[0u8; TABLE_MOVES]; MAX_DEPTH as usize + 1];
    for (d, row) in t.iter_mut().enumerate().skip(1) {
        for (m, cell) in row.iter_mut().enumerate().skip(1) {
            let r = 0.77 + (d as f64).ln() * (m as f64).ln() / 2.25;
            *cell = r.clamp(0.0, 3.5) as u8;
        }
    }
    t
});

/// Reduce a reduction by one ply, but never below zero.
///
/// A reduction is a non-negative number of plies. In particular,
/// `0.saturating_sub(1)` is still `-1` for signed integers because `-1` is
/// representable; explicit flooring is therefore required here.
#[inline]
const fn less_ply(r: Depth) -> Depth {
    if r > 0 { r - 1 } else { 0 }
}

/// The LMR reduction in plies for a move with 1-based index `moved` at
/// remaining depth `depth`.
///
/// * depths ≤ 1 and the first move are never reduced,
/// * a non-improving position reduces one ply more,
/// * PV nodes reduce one ply less (their re-search safety valve is cheaper),
/// * the *history* of the move (main quiet history, `i32` score) bends the
///   reduction: a move history has proven good for is searched one ply deeper
///   (`-1`), a move that keeps failing one ply shallower (`+1`),
/// * the result never exceeds `depth - 1` (searching at depth 1 at worst).
///
/// Whether the move is *eligible* for a reduction at all is
/// [`allows_lmr`]; in particular being in check never reaches this function,
/// because a reduced evasion can hide the refutation of the check.
#[inline]
pub fn lmr_reduction(
    depth: Depth,
    moved: usize,
    pv_node: bool,
    improving: bool,
    history: i32,
) -> Depth {
    if depth <= 1 || moved <= 1 {
        return 0;
    }

    let mut r = Depth::from(LMR_TABLE[depth.min(MAX_DEPTH) as usize][moved.min(TABLE_MOVES - 1)]);

    if !improving {
        r += 1;
    }

    if pv_node {
        r = less_ply(r);
    }

    if history < 0 {
        r += 1;
    } else if history > 0 {
        r = less_ply(r);
    }

    r.min(depth - 1)
}

/// Whether the move with 1-based index `moved` may be searched at a reduced
/// depth in a node with `depth` plies left, whose child would be searched at
/// `child_depth`.
///
/// A move is **not** eligible when
///
/// * `child_depth != depth - 1`: the child carries a check extension, and the
///   extra ply *is* the point of the extension — a check is never reduced;
/// * the move is a capture or a promotion: the reduction curve is calibrated
///   for quiets, and every tactical move is searched in full;
/// * **the side to move is in check**: every legal move is then the answer to
///   a forced threat, and there is no "late, probably bad" move to discount.
///   A reduced evasion can drop the only refutation of the check;
/// * `moved` is still within the first moves (a PV node reserves one more):
///   the principal variation must not be lost to an over-reduction;
/// * `depth < 3`: there is nothing to gain from cutting a nearly-shallow node.
#[inline]
pub const fn allows_lmr(
    child_depth: Depth,
    depth: Depth,
    is_tactical: bool,
    in_check: bool,
    moved: usize,
    pv_node: bool,
) -> bool {
    child_depth == depth - 1
        && !is_tactical
        && !in_check
        && depth >= 3
        && moved >= 3 + pv_node as usize
}

// --- Stockfish 19 reduction formula (Step 18) ------------------------------
//
// Everything below ports the `sf_19` late-move-reduction model
// (`search.cpp:1157-1403`, `1885-1888`). Two things are kept deliberately
// separate from the classic curve above:
//
// * **Eligibility** stays [`allows_lmr`]. Stockfish 19 gates LMR only on
//   `depth >= 2 && moveCount > 1` and reduces captures and evasions too; this
//   engine's Stage-02 contract — and its `in_check_nodes_never_reduce` and
//   `tactical_moves_and_check_extensions_never_reduce` tests — require the
//   stronger gate. That gate is a documented deviation and is not negotiable
//   here, so this module supplies the *amount*, not the *permission*.
// * **Amount** comes from [`LmrSignals`] and [`lmr_reduction_modern`].

/// Row length of the Stockfish reduction lookup table, indexed by depth or by
/// 1-based move number. Stockfish sizes it `MAX_MOVES`; the same bound is
/// ample for Morstilia's `MAX_DEPTH` and 270-move move lists.
const MODERN_TABLE: usize = 256;

/// `reductions[i] = int(2872 / 128 * ln i)` (`search.cpp:713`).
///
/// A natural-logarithmic ramp with **no** per-ply slope, which is what makes
/// the modern curve nearly linear in move number at small indices instead of
/// the classic `0.77 + ln(d)ln(m)/2.25`.
static MODERN_REDUCTIONS: LazyLock<[i32; MODERN_TABLE]> = LazyLock::new(|| {
    let mut t = [0i32; MODERN_TABLE];
    for (i, cell) in t.iter_mut().enumerate().skip(1) {
        *cell = (2872.0 / 128.0 * (i as f64).ln()) as i32;
    }
    t
});

/// `reductions[index]`, saturating at the table bound.
#[inline]
fn reductions_at(index: usize) -> i32 {
    MODERN_REDUCTIONS[index.min(MODERN_TABLE - 1)]
}

/// Everything the modern reduction formula reads, snapshotted at the move.
///
/// Taking a snapshot keeps the formula a pure function of *named* signals,
/// which is what makes it unit-testable without a search, and it pins the
/// `tt_data`/history reads before any child search can learn into them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LmrSignals {
    /// Remaining depth at this node.
    pub depth: Depth,
    /// 1-based index of this move in the node's ordering.
    pub move_number: usize,
    /// The node's own `newDepth`, i.e. `depth - 1` plus any extension.
    pub new_depth: Depth,
    /// Full-window node whose line the PV follows.
    pub pv_node: bool,
    /// Zero-window node expected to fail high (Stockfish's `cutNode`).
    pub cut_node: bool,
    /// The static evaluation has improved over two plies ago.
    pub improving: bool,
    /// This node is *not* on the previous iteration's PV (`ss->followPV`).
    ///
    /// Kept separate from `pv_node` on purpose: Stockfish's quiet-move pruning
    /// gate is `else if (!ss->followPV || !PvNode)`, which is a genuine
    /// two-variable condition, and the reduction's own history term needs the
    /// distinction too. Collapsing the two would silently change which
    /// branches are reachable.
    pub follow_pv: bool,
    /// The node has no TT move.
    pub no_tt_move: bool,
    /// The stored TT move is a capture or a promotion.
    pub tt_capture: bool,
    /// `ttData.value > alpha`.
    pub tt_value_over_alpha: bool,
    /// `ttData.depth >= depth`.
    pub tt_depth_sufficient: bool,
    /// The child's beta cutoff count (`(ss + 1)->cutoffCnt`).
    pub child_cutoffs: i32,
    /// This move is the node's TT move.
    pub is_tt_move: bool,
    /// This move is a capture or a promotion.
    pub capture: bool,
    /// `ss->statScore` for this move: history folded into one number.
    pub stat_score: i32,
    /// The node's `alpha`, for the `alpha - eval` term.
    pub alpha: i32,
    /// The node's uncorrected `staticEval`, for the `alpha - eval` term.
    pub static_eval: i32,
    /// The raw correction-history value.
    pub correction_value: i32,
    /// `beta - alpha` at this node.
    pub delta: i32,
    /// `rootDelta`: the width of the window the root was last searched with.
    ///
    /// Stockfish assigns this only inside its aspiration loop
    /// (`search.cpp:394`) and leaves it otherwise uninitialised; Morstilia
    /// seeds it and refreshes it before every root search, so the division in
    /// the formula is always defined.
    pub root_delta: i32,
}

impl LmrSignals {
    /// Stockfish's `allNode = !(PvNode || cutNode)` (`search.cpp:726`).
    #[inline]
    pub const fn all_node(&self) -> bool {
        !(self.pv_node || self.cut_node)
    }
}

/// The base reduction for a node: `reduction(improving, depth, moveNumber,
/// delta)` (`search.cpp:1885-1888`).
///
/// ```text
/// reductionScale = reductions[depth] * reductions[moveNumber]
/// return reductionScale - delta * 577 / rootDelta
///      + !improving * reductionScale * 197 / 512 + 982
/// ```
///
/// The `delta / rootDelta` term is what makes the reduction window-aware: a
/// move searched under a narrow window is already close to the truth, so it is
/// reduced less. It is also the only division in the formula, which is why
/// `root_delta` is seeded defensively.
#[inline]
pub fn reduction_base(sp: &SearchParams, s: &LmrSignals) -> i32 {
    let scale = reductions_at(s.depth.max(0) as usize) * reductions_at(s.move_number);
    let root_delta = s.root_delta.max(1);
    scale - s.delta * sp.lmr_root_delta_num / root_delta
        + i32::from(!s.improving) * scale * sp.lmr_non_improving_num / sp.lmr_non_improving_den
        + sp.lmr_base
}

/// The complete modern reduction `r`, in units of [`SearchParams::lmr_unit`]
/// (`1/1024` of a ply, as in Stockfish).
///
/// Every branch below is a `sf_19` term; none of them is invented.
///
/// ```text
/// r  = reduction(improving, depth, moveCount, delta)
/// r += 929 * ttPv
/// r -= 3023 + PvNode*1004 + (ttValue > alpha)*885
///        + (ttDepth >= depth) * (816 + cutNode*940)
/// r += 697 - 65*moveCount - |correctionValue| / 26310
/// r += (4026 + 933*!ttMove) * cutNode
/// r += 1079 * ttCapture
/// r += 264 + 1095*(childCutoffs > 2) + 1138*allNode   (or -2179 for the TT move)
/// r -= statScore * 439 / 4096
/// r += 3 * clamp(alpha - eval, -64, 96)              (quiet, non-decisive alpha)
/// r += r * 276 / (256*depth + 268)                    (all nodes)
/// ```
#[allow(clippy::too_many_lines)]
pub fn lmr_reduction_modern(sp: &SearchParams, s: &LmrSignals) -> i32 {
    let mut r = reduction_base(sp, s);

    // "Decrease reduction for ttPv nodes" (search.cpp:1161-1162, 1316-1318).
    //
    // The first block is subtracted, not added: a node the table believes is a
    // PV node was searched with a full window, so the moves under it are far
    // better ordered and need far less discounting.
    if s.follow_pv {
        r += sp.lmr_tt_pv;
        r -= sp.lmr_ttpv_adjust_base
            + sp.lmr_ttpv_adjust_pv * i32::from(s.pv_node)
            + sp.lmr_ttpv_adjust_value * i32::from(s.tt_value_over_alpha)
            + (sp.lmr_ttpv_adjust_depth + sp.lmr_ttpv_adjust_depth_cut * i32::from(s.cut_node))
                * i32::from(s.tt_depth_sufficient);
    }

    // "Base reduction offset to compensate for other tweaks" and the two
    // per-move terms (search.cpp:1321-1324).
    r += sp.lmr_base_offset;
    r -= sp.lmr_per_move * s.move_number as i32;
    r -= s.correction_value.abs() / sp.lmr_correction_div;

    // "Increase reduction for cut nodes" (search.cpp:1327-1328).
    if s.cut_node {
        r += sp.lmr_cut_node + sp.lmr_cut_node_no_tt * i32::from(s.no_tt_move);
    }

    // "Increase reduction if ttMove is a capture" (search.cpp:1331-1332).
    if s.tt_capture {
        r += sp.lmr_tt_capture;
    }

    // "Increase reduction if next ply has a lot of fail high" (search.cpp:1335),
    // and its alternative branch for the move that *is* the TT move
    // (search.cpp:1339-1340) — mutually exclusive, in that order.
    if s.child_cutoffs > 1 {
        r += sp.lmr_cutoff_one
            + sp.lmr_cutoff_two * i32::from(s.child_cutoffs > 2)
            + sp.lmr_all_node_cutoff * i32::from(s.all_node());
    } else if s.is_tt_move {
        r -= sp.lmr_tt_move;
    }

    // "Decrease/increase reduction for moves with a good/bad history".
    r -= s.stat_score * sp.lmr_stat_score_num / sp.lmr_stat_score_den;

    // Quiet moves only, and only while alpha is not already decisive: once
    // alpha is a mate score the window is pinned and a static-eval term would
    // be meaningless.
    if !s.capture && !is_decisive(s.alpha) {
        let gap = (s.alpha - s.static_eval).clamp(sp.lmr_alpha_eval_min, sp.lmr_alpha_eval_max);
        r += sp.lmr_alpha_eval * gap;
    }

    // "Scale up reductions for expected ALL nodes" (search.cpp:1358-1359).
    if s.all_node() {
        r += r * sp.lmr_all_node_num
            / (sp.lmr_all_node_depth * s.depth.max(1) + sp.lmr_all_node_offset);
    }

    r
}

/// Stockfish's `ss->statScore` (`search.cpp:1342-1349`): the move's own
/// history folded into the single number the reduction reads.
///
/// ```text
/// if (capture)
///     statScore = 873 * PieceValue[victim] / 128
///                + captureHistory[movedPiece][to][victimType];
/// else
///     statScore = (2252 * mainHistory[us][move]
///                + 1126 * contHist[0][movedPiece][to]
///                + 1093 * contHist[1][movedPiece][to]) / 1024;
/// ```
///
/// This is *move-ordering* history, never correction history. The two live in
/// different tables with different units (`±32767` here, centipawns there) and
/// different signs of usefulness, so conflating them is the classic silent bug
/// in a port of this formula.
///
/// Morstilia's main history and both continuation planes are `i16` tables
/// clamped to `±HIST_MAX = ±32767`, and the capture history is stored pre-scaled
/// by `CAP_HIST_DIV`, so [`History::capture_adjustment`] is the right reader for
/// it.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn lmr_stat_score(
    sp: &SearchParams,
    pos: &crate::board::Position,
    m: crate::types::RawMove,
    tables: &crate::move_ordering::OrderingTables,
    prev: Option<crate::move_ordering::history::MoveCtx>,
    ant: Option<crate::move_ordering::history::MoveCtx>,
    ep: &crate::evaluation::EvalParams,
) -> i32 {
    let board = pos.board();
    if crate::move_ordering::is_capture_or_promotion(board, m) {
        let mvv = crate::move_ordering::capture_score(board, m, ep);
        return sp.stat_score_capture_weight * mvv / sp.stat_score_capture_div
            + tables.history.capture_adjustment(board, m);
    }
    let side = pos.turn();
    let cont = board
        .piece_at(m.from())
        .map(|p| {
            tables
                .history
                .continuation_score(side, p.role as usize, m.to(), prev, ant)
        })
        .unwrap_or(0);
    (sp.stat_score_main * tables.history.history_score(side, m) + sp.stat_score_cont1 * cont)
        / sp.stat_score_div
}

/// Applies the modern reduction to produce the depth the move is searched at.
///
/// ```text
/// d = max(1, min(newDepth - r / 1024, newDepth + 2)) + PvNode
/// ```
///
/// A *negative* `r` is what makes a negative singular extension compound: the
/// `newDepth + lmr_max_bonus` arm is the "limited search extension beyond the
/// first move depth" of the Stockfish comment at `search.cpp:1365-1366`, and it
/// is why the min/max pair is written this way rather than as a plain clamp.
#[inline]
pub fn lmr_search_depth(sp: &SearchParams, s: &LmrSignals, r: i32) -> Depth {
    let reduced = s.new_depth - r / sp.lmr_unit;
    let bounded = reduced.max(1).min(s.new_depth + sp.lmr_max_bonus);
    bounded + sp.lmr_pv_bonus * i32::from(s.pv_node)
}

/// Whether this node may be entered one ply shallower because the table has
/// nothing to order it with — Stockfish 19 Step 11 (`search.cpp:1048-1052`).
///
/// ```text
/// if (!ss->followPV && !allNode && depth >= 6 && !ttData.move) depth--;
/// ```
///
/// `!allNode` is spelled `pv_node || cut_node` because `allNode` is defined as
/// `!(PvNode || cutNode)` (`search.cpp:726`); keeping the disjunction explicit
/// is what stops the two from contradicting each other.
///
/// `follow_pv` is a distinct piece of state — "this node lies on the *previous*
/// iteration's principal variation" — and is deliberately **not** the same as
/// `pv_node`. A PV node in a *new* line has no TT move precisely because the
/// table has not seen the variation yet, which is exactly the case IIR exists
/// to accelerate.
#[inline]
pub const fn allows_iir(
    follow_pv: bool,
    pv_node: bool,
    cut_node: bool,
    depth: Depth,
    no_tt_move: bool,
    min_depth: Depth,
) -> bool {
    !follow_pv && (pv_node || cut_node) && depth >= min_depth && no_tt_move
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_move_is_never_reduced() {
        for d in 0..=12 {
            assert_eq!(lmr_reduction(d, 0, false, true, 0), 0);
            assert_eq!(lmr_reduction(d, 1, false, true, 0), 0);
        }
    }

    #[test]
    fn reduction_grows_with_depth_and_move_index() {
        let base = lmr_reduction(6, 8, false, true, 0);
        let deeper = lmr_reduction(12, 8, false, true, 0);
        let later = lmr_reduction(6, 24, false, true, 0);

        assert!(deeper >= base);
        assert!(later >= base);
        assert_eq!(base, 2, "classic curve value at (6,8): {base}");
    }

    #[test]
    fn non_improving_positions_reduce_more() {
        let improving = lmr_reduction(8, 6, false, true, 0);
        let stagnant = lmr_reduction(8, 6, false, false, 0);

        assert!(stagnant > improving);
    }

    #[test]
    fn pv_nodes_reduce_less() {
        let non_pv = lmr_reduction(8, 10, false, true, 0);
        let pv = lmr_reduction(8, 10, true, true, 0);

        assert!(pv <= non_pv);
    }

    #[test]
    fn history_bends_the_reduction() {
        // A move proven good by history is searched deeper (-1), a move that
        // has kept failing one ply shallower (+1).
        for d in 3..=12 {
            for m in 2..=32 {
                let good = lmr_reduction(d, m, false, true, 4_000);
                let neutral = lmr_reduction(d, m, false, true, 0);
                let bad = lmr_reduction(d, m, false, true, -4_000);

                assert!(good <= neutral, "positive history must reduce less");
                assert!(bad >= neutral, "negative history must reduce more");
                assert!(bad - good <= 2, "history may bend by at most 2 plies");
            }
        }
    }

    #[test]
    fn history_never_lifts_first_move_out_of_zero() {
        for d in 0..=12 {
            assert_eq!(lmr_reduction(d, 1, false, true, -99_999), 0);
            assert_eq!(lmr_reduction(d, 1, false, true, 99_999), 0);
        }
    }

    #[test]
    fn the_reduction_amount_is_never_negative() {
        // A reduction is a number of plies, so it can never be negative.
        // Sweep the whole relevant input space.
        for d in 0..=16i32 {
            for m in 0..=40usize {
                for pv_node in [false, true] {
                    for improving in [false, true] {
                        for h in [-8_000i32, -1, 0, 1, 8_000] {
                            let r = lmr_reduction(d, m, pv_node, improving, h);
                            let where_ = format!(
                                "depth {d}, move {m}, pv={pv_node}, improving={improving}, history={h}"
                            );

                            assert!(r >= 0, "{where_} reduced {r}");
                            assert!(r <= (d - 1).max(0), "{where_} reduced {r}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn never_reduces_below_one_ply_left() {
        for d in 2..=12 {
            for m in 2..=64 {
                for h in [-8_000, 0, 8_000] {
                    let r = lmr_reduction(d, m, true, false, h);

                    assert!(r < d, "depth {d} move {m} reduced {r}");
                    assert!(r >= 0, "depth {d} move {m} history {h} reduced {r}");
                }
            }
        }
    }

    #[test]
    fn table_is_bounded() {
        // The largest entry the curve can produce.
        let max_reduction = lmr_reduction(MAX_DEPTH, TABLE_MOVES - 1, false, false, -99_999);

        assert!((1..=6).contains(&max_reduction), "got {max_reduction}");
    }

    // --- The LMR eligibility gate ------------------------------------------

    #[test]
    fn a_late_quiet_move_in_a_calm_node_is_reducible() {
        // Baseline: depth >= 3, quiet, not in check, third move or later.
        assert!(allows_lmr(7, 8, false, false, 3, false));
        assert!(allows_lmr(7, 8, false, false, 4, false));
    }

    #[test]
    fn no_evasion_in_check_is_ever_reduced() {
        // No legal evasion may be LMR-reduced while the side to move is in
        // check, regardless of move order or node type.
        for d in 1..=24 {
            for moved in 1..=40 {
                for pv_node in [false, true] {
                    assert!(
                        !allows_lmr(d - 1, d, false, true, moved, pv_node),
                        "in-check evasion at depth {d}, move {moved}, pv={pv_node} \
                         must not be reduced"
                    );
                }
            }
        }
    }

    #[test]
    fn the_gate_only_excludes_the_documented_reasons() {
        // Sweep the boolean/input space and compare the predicate to its
        // explicit definition.
        for depth in 1..=8i32 {
            for is_tactical in [false, true] {
                for in_check in [false, true] {
                    for moved in 1..=6usize {
                        for pv_node in [false, true] {
                            for extend in [false, true] {
                                let child_depth = if extend { depth } else { depth - 1 };

                                let expected = !extend
                                    && !is_tactical
                                    && !in_check
                                    && depth >= 3
                                    && moved >= 3 + usize::from(pv_node);

                                assert_eq!(
                                    allows_lmr(
                                        child_depth,
                                        depth,
                                        is_tactical,
                                        in_check,
                                        moved,
                                        pv_node
                                    ),
                                    expected,
                                    "gate mismatch at depth={depth} \
                                     tactical={is_tactical} \
                                     check={in_check} \
                                     moved={moved} \
                                     pv={pv_node} \
                                     extend={extend}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tactical_moves_and_check_extensions_never_reduce() {
        // A capture, promotion, or check-extended child always stays at full
        // depth even when ordered late.
        for d in 3..=12i32 {
            for moved in 3..=40usize {
                assert!(!allows_lmr(d - 1, d, true, false, moved, false), "tactical");
                assert!(
                    !allows_lmr(d, d, false, false, moved, false),
                    "check extension must keep the extra ply"
                );
            }
        }
    }

    #[test]
    fn the_first_moves_are_never_reducible() {
        // First two moves at a cut node, first three at a PV node: the PV
        // cannot be lost to an over-reduction.
        for d in 3..=12i32 {
            assert!(!allows_lmr(d - 1, d, false, false, 1, false));
            assert!(!allows_lmr(d - 1, d, false, false, 2, false));
            assert!(allows_lmr(d - 1, d, false, false, 3, false));

            assert!(!allows_lmr(d - 1, d, false, false, 1, true));
            assert!(!allows_lmr(d - 1, d, false, false, 2, true));
            assert!(!allows_lmr(d - 1, d, false, false, 3, true));
            assert!(allows_lmr(d - 1, d, false, false, 4, true));
        }
    }
}

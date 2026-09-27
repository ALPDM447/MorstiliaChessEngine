//! Late move reductions (LMR).
//!
//! Quiet moves searched late in the move list are usually refuted cheaply, so
//! they are first searched with a reduced depth; if the reduced search still
//! beats `alpha` the move is re-searched at full depth. The reduction amount
//! follows the classic `ln(depth) * ln(move_index)` curve, tempered by node
//! type (PV nodes reduce less) and whether the position is "improving"
//! (the static eval increased since the previous ply).

use std::sync::LazyLock;

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

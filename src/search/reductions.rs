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
    for d in 1..=MAX_DEPTH as usize {
        for m in 1..TABLE_MOVES {
            let r = 0.77 + (d as f64).ln() * (m as f64).ln() / 2.25;
            t[d][m] = r.clamp(0.0, 3.5) as u8;
        }
    }
    t
});

/// The LMR reduction in plies for a move with 1-based index `moved` at
/// remaining depth `depth`.
///
/// * depths ≤ 1 and the first move are never reduced,
/// * a non-improving position and being in check reduce one ply more,
/// * PV nodes reduce one ply less (their re-search safety valve is cheaper),
/// * the *history* of the move (main quiet history, `i32` score) bends the
///   reduction: a move history has proven good for is searched one ply deeper
///   (`-1`), a move that keeps failing one ply shallower (`+1`),
/// * the result never exceeds `depth - 1` (searching at depth 1 at worst).
#[inline]
pub fn lmr_reduction(
    depth: Depth,
    moved: usize,
    pv_node: bool,
    improving: bool,
    in_check: bool,
    history: i32,
) -> Depth {
    if depth <= 1 || moved <= 1 {
        return 0;
    }
    let mut r = Depth::from(LMR_TABLE[depth.min(MAX_DEPTH) as usize][moved.min(TABLE_MOVES - 1)]);
    if !improving {
        r += 1;
    }
    if in_check {
        r += 1;
    }
    if pv_node {
        r = r.saturating_sub(1);
    }
    if history < 0 {
        r += 1;
    } else if history > 0 {
        r = r.saturating_sub(1);
    }
    r.min(depth - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_move_is_never_reduced() {
        for d in 0..=12 {
            assert_eq!(lmr_reduction(d, 0, false, true, false, 0), 0);
            assert_eq!(lmr_reduction(d, 1, false, true, false, 0), 0);
        }
    }

    #[test]
    fn reduction_grows_with_depth_and_move_index() {
        let base = lmr_reduction(6, 8, false, true, false, 0);
        let deeper = lmr_reduction(12, 8, false, true, false, 0);
        let later = lmr_reduction(6, 24, false, true, false, 0);
        assert!(deeper >= base);
        assert!(later >= base);
        assert_eq!(base, 2, "classic curve value at (6,8): {base}");
    }

    #[test]
    fn non_improving_and_check_reduce_more() {
        let improving = lmr_reduction(8, 6, false, true, false, 0);
        let stagnant = lmr_reduction(8, 6, false, false, false, 0);
        let evading = lmr_reduction(8, 6, false, true, true, 0);
        assert!(stagnant > improving);
        assert!(evading >= improving);
    }

    #[test]
    fn pv_nodes_reduce_less() {
        let non_pv = lmr_reduction(8, 10, false, true, false, 0);
        let pv = lmr_reduction(8, 10, true, true, false, 0);
        assert!(pv <= non_pv);
    }

    #[test]
    fn history_bends_the_reduction() {
        // A move proven good by history is searched deeper (-1), a move that
        // has kept failing one ply shallower (+1) — the depth/move-number/history
        // triangle the reduction is built on.
        for d in 3..=12 {
            for m in 2..=32 {
                let good = lmr_reduction(d, m, false, true, false, 4_000);
                let neutral = lmr_reduction(d, m, false, true, false, 0);
                let bad = lmr_reduction(d, m, false, true, false, -4_000);
                assert!(good <= neutral, "positive history must reduce less");
                assert!(bad >= neutral, "negative history must reduce more");
                assert!(bad - good <= 2, "history may bend by at most 2 plies");
            }
        }
    }

    #[test]
    fn history_never_lifts_first_move_out_of_zero() {
        // The first move is never reduced regardless of history.
        for d in 0..=12 {
            assert_eq!(lmr_reduction(d, 1, false, true, false, -99_999), 0);
            assert_eq!(lmr_reduction(d, 1, false, true, false, 99_999), 0);
        }
    }

    #[test]
    fn never_reduces_below_one_ply_left() {
        for d in 2..=12 {
            for m in 2..=64 {
                for h in [-8_000, 0, 8_000] {
                    let r = lmr_reduction(d, m, true, false, true, h);
                    assert!(r <= d - 1, "depth {d} move {m} reduced {r}");
                    assert!(r >= 0);
                }
            }
        }
    }

    #[test]
    fn table_is_bounded() {
        // The largest entry the curve can produce.
        let max_reduction = lmr_reduction(MAX_DEPTH, TABLE_MOVES - 1, false, false, false, -99_999);
        assert!((1..=6).contains(&max_reduction), "got {max_reduction}");
    }
}

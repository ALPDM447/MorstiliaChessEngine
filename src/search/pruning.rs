//! Pruning margins and depth gates for the main search.
//!
//! The rules themselves live in `alphabeta.rs`; this module keeps the tuning
//! constants and their pure helpers together so they are easy to adjust and
//! unit-test. All margins are in centipawns and grow with depth, because a
//! deeper search accumulates more (uncertain) positional value.

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
        assert!(FUTILITY_DEPTH >= RAZOR_DEPTH);
        assert!(RFP_DEPTH > FUTILITY_DEPTH);
        assert!(NULL_MOVE_MIN_DEPTH >= 2);
        assert!(PROBCUT_DEPTH >= 5, "probcut needs a deep enough probe");
        assert!(RAZOR_DEPTH < PROBCUT_DEPTH);
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
}

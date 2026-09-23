//! Time management.
//!
//! Converts a UCI `go ...` command into a [`TimeLimit`]: a real per-move
//! budget allocated from the clock, following the classic "share of the
//! remaining clock (+ increment), capped at half the clock" heuristic. When
//! `movestogo` is absent the number of remaining moves is estimated from the
//! fullmove number (`30 - fullmove / 2`), so a 10-minute clock yields roughly
//! 12–25 s per move in a middlegame.
//!
//! The search treats the limit as two numbers: a *soft* budget (stop between
//! iterative-deepening iterations, so an in-flight iteration is always
//! finished cleanly) and a *hard* deadline (checked inside the search, so we
//! never blow the clock).

use shakmaty::{Color, Position as _};

use crate::board::Position;
use crate::uci::parser::GoParams;

/// A fully resolved search limit. All times in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeLimit {
    /// Absolute depth ceiling (from `go depth N`).
    pub depth: Option<i32>,
    /// Absolute node cap (from `go nodes N`).
    pub nodes: Option<u64>,
    /// Fixed per-move budget (from `go movetime N`).
    pub movetime_ms: Option<u64>,
    /// Stop between iterations once this many ms have elapsed (0 = never).
    pub soft_ms: u64,
    /// Absolute deadline inside the search (0 = none).
    pub hard_ms: u64,
    /// `go infinite` / no budget at all: ignore time entirely.
    pub infinite: bool,
}

/// Fallback budget when the GUI provides no clock information.
pub const DEFAULT_BUDGET_MS: u64 = 5_000;

/// Minimum number of moves we ever plan for (avoids absurd single-move eats).
pub const MIN_PLANNED_MOVES: u32 = 8;
/// Maximum number of moves we ever plan for.
pub const MAX_PLANNED_MOVES: u32 = 40;

impl TimeLimit {
    /// No time pressure at all (`go infinite`, internal calls, tests).
    pub const fn unlimited() -> TimeLimit {
        TimeLimit {
            depth: None,
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        }
    }

    /// The soft budget, interpreted as a deadline or "none".
    #[inline]
    pub fn soft_deadline(&self) -> Option<std::time::Instant> {
        if self.soft_ms == 0 {
            None
        } else {
            Some(std::time::Instant::now() + std::time::Duration::from_millis(self.soft_ms))
        }
    }

    /// The hard deadline, interpreted as a deadline or "none".
    #[inline]
    pub fn hard_deadline(&self) -> Option<std::time::Instant> {
        if self.hard_ms == 0 {
            None
        } else {
            Some(std::time::Instant::now() + std::time::Duration::from_millis(self.hard_ms))
        }
    }
}

/// Builds the [`TimeLimit`] for a parsed `go` command on `pos`.
///
/// Precedence: `movetime` > clock allocation (`wtime/btime`) > a fixed
/// default. `depth`, `nodes` and `infinite` are passed through.
pub fn time_limit_from_go(go: &GoParams, pos: &Position) -> TimeLimit {
    let depth = go.depth.filter(|&d| d > 0).map(|d| d as i32);
    let nodes = go.nodes;

    let infty = go.infinite
        && go.movetime.is_none()
        && go.wtime.is_none()
        && go.btime.is_none()
        && depth.is_none()
        && nodes.is_none();

    let (soft_ms, hard_ms, movetime_ms) = if infty {
        // `go infinite` / no limits at all: ignore time entirely.
        (0, 0, None)
    } else if let Some(mt) = go.movetime {
        (mt, mt, Some(mt))
    } else if let Some(budget) = clock_budget(go, pos) {
        (budget * 3 / 4, budget, None)
    } else {
        (DEFAULT_BUDGET_MS * 3 / 4, DEFAULT_BUDGET_MS, None)
    };

    TimeLimit {
        depth,
        nodes,
        movetime_ms,
        soft_ms,
        hard_ms,
        infinite: infty,
    }
}

/// The per-move budget in ms from the clocks, or `None` when no clock is set.
///
/// Model:
///
/// ```text
/// moves   = movestogo, or estimate 30 - fullmove/2 (clamped to [8, 40])
/// budget  = clock / moves + increment
/// budget  = min(budget, clock / 2)
/// ```
fn clock_budget(go: &GoParams, pos: &Position) -> Option<u64> {
    let (clock, inc) = match pos.turn() {
        Color::White => (go.wtime?, go.winc.unwrap_or(0)),
        Color::Black => (go.btime?, go.binc.unwrap_or(0)),
    };
    if clock == 0 {
        return None;
    }

    let moves = match go.movestogo {
        Some(m) => m.max(1).min(MAX_PLANNED_MOVES),
        None => estimate_moves(pos.chess().fullmoves().get()),
    };

    let per_move = clock / u64::from(moves) + inc;
    Some(per_move.min(clock / 2).max(1))
}

/// Estimates the number of moves remaining from the fullmove counter
/// (roughly: 30 moves minus half the plies already played).
#[inline]
fn estimate_moves(fullmove: u32) -> u32 {
    let m = 30i64 - i64::from(fullmove) / 2;
    m.clamp(i64::from(MIN_PLANNED_MOVES), i64::from(MAX_PLANNED_MOVES)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;

    #[test]
    fn movetime_wins_over_clocks_and_becomes_hard() {
        let go = GoParams {
            movetime: Some(1234),
            wtime: Some(60_000),
            ..GoParams::default()
        };
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert_eq!(tl.movetime_ms, Some(1234));
        assert_eq!(tl.hard_ms, 1234);
        assert_eq!(tl.soft_ms, 1234);
        assert!(!tl.infinite);
    }

    #[test]
    fn clock_budget_uses_movestogo() {
        // 10-minute clock, 20 moves to go: 600s/20 = 30s, capped at half (5min).
        // movestogo wins over the estimate.
        let go = GoParams {
            wtime: Some(600_000),
            btime: Some(600_000),
            movestogo: Some(20),
            ..GoParams::default()
        };
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert!(!tl.infinite);
        assert_eq!(tl.hard_ms, 30_000);
        assert_eq!(tl.soft_ms, 22_500);
    }

    #[test]
    fn clock_budget_estimates_moves_from_fullmove() {
        // No movestogo: startpos (fullmove 1) estimates 30 - 0 = 30 moves.
        // 10 minutes / 30 = 20s. Soft budget 15s.
        let go = GoParams {
            wtime: Some(600_000),
            btime: Some(600_000),
            winc: Some(0),
            binc: Some(0),
            ..GoParams::default()
        };
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert_eq!(tl.hard_ms, 20_000);
        assert_eq!(tl.soft_ms, 15_000);
    }

    #[test]
    fn clock_budget_includes_increment() {
        // 60s clock, 30 moves, +1s increment: 60/30 + 1 = 3s per move.
        let go = GoParams {
            wtime: Some(60_000),
            winc: Some(1000),
            ..GoParams::default()
        };
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert_eq!(tl.hard_ms, 3_000);
    }

    #[test]
    fn clock_budget_never_exceeds_half_the_clock() {
        // movestogo 1: full clock would go to the next move, but we cap at half.
        let go = GoParams {
            wtime: Some(10_000),
            winc: Some(0),
            movestogo: Some(1),
            ..GoParams::default()
        };
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert_eq!(tl.hard_ms, 5_000);
    }

    #[test]
    fn go_infinite_ignores_time() {
        let go = GoParams {
            infinite: true,
            ..GoParams::default()
        };
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert!(tl.infinite);
        assert_eq!(tl.hard_ms, 0);
        assert_eq!(tl.soft_ms, 0);
    }

    #[test]
    fn no_clock_falls_back_to_default_budget() {
        let go = GoParams::default();
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert_eq!(tl.hard_ms, DEFAULT_BUDGET_MS);
        assert_eq!(tl.soft_ms, DEFAULT_BUDGET_MS * 3 / 4);
        assert_eq!(tl.depth, None);
    }

    #[test]
    fn depth_passes_through_and_zero_is_ignored() {
        let go = GoParams {
            depth: Some(7),
            ..GoParams::default()
        };
        let tl = time_limit_from_go(&go, &Position::startpos());
        assert_eq!(tl.depth, Some(7));
        let go0 = GoParams {
            depth: Some(0),
            ..GoParams::default()
        };
        assert_eq!(time_limit_from_go(&go0, &Position::startpos()).depth, None);
    }
}

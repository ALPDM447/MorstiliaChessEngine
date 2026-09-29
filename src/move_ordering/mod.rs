//! Move ordering.
//!
//! A move list is sorted by a descending [`score_move`] value before every
//! node is searched. The tiers, in decreasing priority:
//!
//! 1. the TT move (if any),
//! 2. captures & promotions, ordered by MVV-LVA ([`tt_move::capture_score`]),
//! 3. killer moves for the current ply,
//! 4. quiet moves, scored from the history tables ([`History`]; countermove
//!    and continuation bonuses included),
//! 5. *losing* captures — captures whose static exchange evaluation is
//!    negative. Stockfish searches these last (`sf_19/src/move.cpp`: the
//!    `BAD_CAPTURE` stage follows `QUIET`), because a capture that gives back
//!    more than it takes is refuted far more often than any quiet move.
//!
//! All ordering state lives in [`OrderingTables`], owned by the search — the
//! engine keeps no global mutable state, and a `Threads = 1` search is
//! therefore fully deterministic.

pub mod history;
pub mod killers;
pub mod see;
pub mod tt_move;

use shakmaty::Role;

use crate::board::Position;
use crate::evaluation::params::EvalParams;
use crate::types::{MAX_MOVES, MoveList, RawMove};

pub use history::{History, MoveCtx};
pub use killers::Killers;
pub use tt_move::{capture_score, is_capture_or_promotion, victim_value};

/// All move-ordering state used by one search thread.
#[derive(Clone, Debug)]
pub struct OrderingTables {
    pub killers: Killers,
    pub history: History,
}

impl Default for OrderingTables {
    fn default() -> Self {
        OrderingTables::new()
    }
}

impl OrderingTables {
    pub fn new() -> OrderingTables {
        OrderingTables {
            killers: Killers::new(),
            history: History::new(),
        }
    }

    /// Clears everything (`ucinewgame`).
    pub fn reset(&mut self) {
        self.killers.reset();
        self.history.clear();
    }
}

/// Scores `m` for ordering. `tt_move` may be [`RawMove::NULL`]; `ply`, `prev`
/// and `ant` are the current ply and the previous two move contexts (only
/// used for killer and continuation lookups). `p` carries the tunable material
/// values that drive the MVV-LVA tiers (the searcher's shared parameter set).
// The scoring context (position, TT move, tables, ply, the two previous move
// contexts, the eval parameters) is exactly the state a move-ordering decision
// depends on; a struct here would only forward all of it unchanged.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn score_move(
    pos: &Position,
    m: RawMove,
    tt_move: RawMove,
    tables: &OrderingTables,
    ply: usize,
    prev: Option<MoveCtx>,
    ant: Option<MoveCtx>,
    p: &EvalParams,
) -> i32 {
    if tt_move != RawMove::NULL && m == tt_move {
        return tt_move::TT_TIER;
    }

    let board = pos.board();
    if is_capture_or_promotion(board, m) {
        // MVV-LVA dominates; the capture-history tiebreak stays inside the
        // victim tier (see `History::capture_adjustment`).
        return tt_move::CAPTURE_TIER
            + capture_score(board, m, p)
            + tables.history.capture_adjustment(board, m);
    }

    let (k1, k2) = tables.killers.get(ply);
    if m == k1 {
        return tt_move::KILLER1_TIER;
    }
    if m == k2 {
        return tt_move::KILLER2_TIER;
    }

    // Quiet move: blend main history, the countermove bonus and the
    // continuation history in the context of the line just played.
    let side = pos.turn();
    let mut s = tables.history.history_score(side, m);
    if m == tables.history.counter_move(side, prev) {
        s += history::HIST_MAX;
    }
    if let Some(piece) = board.piece_at(m.from()) {
        s += tables
            .history
            .continuation_score(side, piece.role as usize, m.to(), prev, ant);
    }
    s
}

/// Sorts `moves` in place by [`score_move`] (descending). Uses no continuation
/// context — suitable for independent call sites such as position helpers;
/// the search thread uses [`order_moves_ctx`] to pass the line context.
pub fn order_moves(
    moves: &mut MoveList,
    pos: &Position,
    tables: &OrderingTables,
    tt_move: RawMove,
    p: &EvalParams,
) {
    order_moves_ctx(moves, pos, tables, tt_move, 0, None, None, p);
}

/// [`order_moves`] with full search context: killer lookup at `ply` and the
/// previous two move contexts for continuation history.
///
/// Keys are computed **once per move** into a stack array, which is then sorted
/// with the key carried alongside the move, and finally copied back. Calling
/// [`score_move`] from inside a comparator — which is what a plain
/// `sort_unstable_by_key(|m| -score_move(m))` does — evaluates the key
/// `O(n log n)` times, and one evaluation is several random reads into the
/// history tables. The resulting permutation is unchanged up to the
/// tie-breaking of an unstable sort, which was never load-bearing: two moves
/// with identical ordering scores are interchangeable to the search.
///
/// The `BAD_CAPTURE` split is Stockfish's `sf_19/src/move.cpp` stage order: a
/// capture whose static exchange evaluation is negative is searched *after* every
/// quiet move, not merely lower within the capture tier. A capture that gives
/// back more than it takes is refuted more often than any quiet move, and both
/// the per-move SEE pruning and the singular/multi-cut machinery want a correct
/// capture in front of them.
///
/// `BAD_CAPTURE_TIER` sits below every quiet score, so the split needs no second
/// sort pass: the demotion is expressed purely as a lower key.
#[allow(clippy::too_many_arguments)]
pub fn order_moves_ctx(
    moves: &mut MoveList,
    pos: &Position,
    tables: &OrderingTables,
    tt_move: RawMove,
    ply: usize,
    prev: Option<MoveCtx>,
    ant: Option<MoveCtx>,
    p: &EvalParams,
) {
    let board = pos.board();
    // `(negated key, move)` so an ascending sort yields a descending key order,
    // matching the previous `-score_move(m)` comparator exactly.
    let mut keyed: [(i32, u16, RawMove); MAX_MOVES] = [(0, 0, RawMove::NULL); MAX_MOVES];
    let n = moves.len().min(MAX_MOVES);
    for (i, slot) in keyed.iter_mut().enumerate().take(n) {
        let m = moves.get(i);
        let mut key = score_move(pos, m, tt_move, tables, ply, prev, ant, p);
        if key >= tt_move::CAPTURE_TIER && key < tt_move::KILLER2_TIER && see::see(board, m, p) < 0
        {
            key = tt_move::BAD_CAPTURE_TIER + capture_score(board, m, p);
        }
        // The second field is the move's index, so the sort stays *stable* in
        // the original order for equal keys without needing `RawMove: Ord`.
        // That is strictly more deterministic than the previous unstable
        // comparator, which could permute equal-keyed moves arbitrarily.
        *slot = (-key, i as u16, m);
    }
    // Sorting compares the key and then the original index, so moves with
    // identical ordering scores keep the order the generator produced. That is
    // *more* deterministic than the previous unstable comparator, which was
    // free to permute equal-keyed moves arbitrarily.
    keyed[..n].sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    for (i, &(_, _, m)) in keyed[..n].iter().enumerate() {
        moves.set(i, m);
    }
}

/// Convenience: the mover's role for a quiet-move history update at `pos`
/// (the position *before* the move was played).
#[inline]
pub fn moving_role(pos: &Position, m: RawMove) -> Option<Role> {
    pos.board().piece_at(m.from()).map(|p| p.role)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use shakmaty::Color;

    fn pos(fen: &str) -> Position {
        Position::from_fen(fen).unwrap()
    }

    fn uci(p: &Position, s: &str) -> RawMove {
        p.raw_move_from_uci(s).unwrap()
    }

    fn params() -> crate::evaluation::EvalParams {
        crate::evaluation::EvalParams::default()
    }

    #[test]
    fn tt_move_is_first() {
        let p = pos("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3");
        let mut moves = p.legal_moves();
        let tables = OrderingTables::new();
        let tt = uci(&p, "d2d4");
        order_moves(&mut moves, &p, &tables, tt, &params());
        assert_eq!(moves.get(0), tt);
    }

    #[test]
    fn captures_precede_killers_precede_quiets() {
        // White Bxe5 is the only capture; the killer (king a1a2) must follow
        // it, and everything after the killer is quiet.
        let p = pos("k7/8/8/4p3/3B4/8/8/K7 w - - 0 1");
        let mut moves = p.legal_moves();
        let mut tables = OrderingTables::new();
        let capture = uci(&p, "d4e5"); // Bxe5
        let killer = uci(&p, "a1a2");
        tables.killers.store(2, killer);
        order_moves_ctx(
            &mut moves,
            &p,
            &tables,
            RawMove::NULL,
            2,
            None,
            None,
            &params(),
        );

        let cap_pos = moves.iter().position(|m| m == capture).unwrap();
        let killer_pos = moves.iter().position(|m| m == killer).unwrap();
        assert!(cap_pos < killer_pos, "capture must precede the killer");
        for m in moves.iter().skip(killer_pos + 1) {
            assert!(
                !is_capture_or_promotion(p.board(), m),
                "non-killer capture after killer: {m:?}"
            );
        }
    }

    #[test]
    fn ordered_list_contains_all_legal_moves() {
        let p = Position::startpos();
        let mut moves = p.legal_moves();
        let tables = OrderingTables::new();
        order_moves(&mut moves, &p, &tables, uci(&p, "e2e4"), &params());
        assert_eq!(moves.len(), 20);
        let mut sorted: Vec<_> = moves.iter().collect();
        sorted.sort_by_key(|m| m.raw());
        let mut plain: Vec<_> = p.legal_moves().iter().collect();
        plain.sort_by_key(|m| m.raw());
        assert_eq!(sorted, plain, "ordering must be a permutation");
    }

    #[test]
    fn history_influences_quiet_order() {
        let p = Position::startpos();
        let mut tables = OrderingTables::new();
        let good: RawMove = uci(&p, "g1f3");
        tables
            .history
            .update_history(p.turn(), good, history::bonus(12));
        let mut moves = p.legal_moves();
        order_moves(&mut moves, &p, &tables, RawMove::NULL, &params());
        // g1f3 must precede a move we never rewarded (e.g. a2a3).
        let good_pos = moves.iter().position(|m| m == good).unwrap();
        let bad_pos = moves.iter().position(|m| m == uci(&p, "a2a3")).unwrap();
        assert!(good_pos < bad_pos);
    }

    #[test]
    fn countermove_tops_quiet_ties() {
        // Play e2e4 first, then order BLACK's position with the previous move
        // context: the recorded countermove d7d5 must outrank a neutral quiet
        // (score_move reads the countermove via the `prev` argument).
        let p = Position::startpos();
        let prev_move = uci(&p, "e2e4");
        let prev = MoveCtx::of(&p, prev_move).unwrap();
        let (after_e4, _) = p.play_uci("e2e4").unwrap();
        let cm = uci(&after_e4, "d7d5");
        let mut tables = OrderingTables::new();
        tables
            .history
            .set_counter_move(Color::Black, Some(prev), cm);
        let mut moves = after_e4.legal_moves();
        order_moves_ctx(
            &mut moves,
            &after_e4,
            &tables,
            RawMove::NULL,
            0,
            Some(prev),
            None,
            &params(),
        );
        let cm_pos = moves
            .iter()
            .position(|m| m == cm && !is_capture_or_promotion(after_e4.board(), m))
            .unwrap();
        let quiet_pos = moves
            .iter()
            .position(|m| m != cm && !is_capture_or_promotion(after_e4.board(), m))
            .unwrap();
        assert!(cm_pos < quiet_pos);
    }

    #[test]
    fn capture_history_cannot_outrank_wide_victim_tiers_in_score() {
        // Saturate a pawn-victim capture's history strongly and push a
        // queen-victim capture to its floor: `score_move` must still order the
        // queen capture first (see `History::capture_adjustment`).
        let mut tables = OrderingTables::new();
        let pawn_takes = pos("6k1/8/8/8/8/4n3/3P4/6K1 w - - 0 1");
        let pm = uci(&pawn_takes, "d2e3"); // pawn takes knight
        let queen_takes = pos("6k1/8/8/8/8/8/2q5/2R3K1 w - - 0 1");
        let qm = uci(&queen_takes, "c1c2"); // rook takes queen
        for _ in 0..1000 {
            tables.history.update_capture(pawn_takes.board(), pm, 100);
            tables.history.update_capture(queen_takes.board(), qm, -100);
        }
        let sq = score_move(
            &pawn_takes,
            pm,
            RawMove::NULL,
            &tables,
            0,
            None,
            None,
            &params(),
        );
        let sk = score_move(
            &queen_takes,
            qm,
            RawMove::NULL,
            &tables,
            0,
            None,
            None,
            &params(),
        );
        assert!(
            sk > sq,
            "capture history must never lift a pawn-victim capture above a \
             queen-victim one in score_move: {sq} vs {sk}"
        );
    }

    #[test]
    fn capture_history_orders_same_victim_by_learning() {
        // Rc1xc2 and Nb4xc2 both capture the c2 queen; MVV-LVA prefers the
        // knight attacker, but a saturated history reward for the rook
        // exchange must flip the ordering inside the victim tier.
        let mut tables = OrderingTables::new();
        let p = pos("6k1/8/8/8/1N6/8/2q5/2R3K1 w - - 0 1");
        let rook_cap = uci(&p, "c1c2");
        let knight_cap = uci(&p, "b4c2");
        for _ in 0..1000 {
            tables.history.update_capture(p.board(), rook_cap, 100);
        }
        let s_rook = score_move(
            &p,
            rook_cap,
            RawMove::NULL,
            &tables,
            0,
            None,
            None,
            &params(),
        );
        let s_knight = score_move(
            &p,
            knight_cap,
            RawMove::NULL,
            &tables,
            0,
            None,
            None,
            &params(),
        );
        assert!(
            s_rook > s_knight,
            "capture history must decide same-victim ties in score_move: \
             {s_rook} vs {s_knight}"
        );
    }
}

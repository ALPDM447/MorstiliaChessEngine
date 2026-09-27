//! Quiescence search.
//!
//! After the full-width search reaches depth 0, only tactically relevant
//! moves are explored: captures, promotions and — when the side to move is in
//! check — every legal evasion. The node "stands pat" at its static
//! evaluation, and only captures that can actually improve the standing are
//! searched:
//!
//! * **delta pruning** skips captures whose touched material (+
//!   [`pruning::DELTA_MARGIN`] optimism) cannot raise the stand-pat score to
//!   `alpha`;
//! * **SEE pruning** skips captures that lose material outright (negative
//!   static exchange) when not in check.
//!
//! Neither prune ever skips a capture that *gives check*: the node it leads to
//! is a node where the opponent is in check and quiescence is already at full
//! width, so the check is a tactical resource the static estimates above cannot
//! price. (Quiet checks are still never generated here — that would be a
//! horizon change, not a safety fix.)
//!
//! Quiescence never writes the transposition table.

use shakmaty::Role;
use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::evaluation::EvalParams;
use crate::move_ordering::history::MoveCtx;
use crate::move_ordering::{order_moves_ctx, tt_move::victim_value};
use crate::search::pruning::DELTA_MARGIN;
use crate::search::{SearchShared, SearchThread, is_draw};
use crate::types::{MAX_PLY, RawMove, mated_in};

/// The value a capture "touches" for delta pruning: the captured piece (a
/// pawn for en passant) plus any promotion surplus. Reads the *tunable*
/// material values (`p`) so delta pruning stays consistent with SEE and the
/// evaluation under any parameter set.
#[inline]
fn touched_value(pos: &Position, m: RawMove, p: &EvalParams) -> i32 {
    let victim = victim_value(pos.board(), m, p);
    match m.promotion() {
        Some(role) => victim + p.piece_value(role) - p.piece_value(Role::Pawn),
        None => victim,
    }
}

/// Orders a set of quiescence moves (captures only unless in check).
pub fn qsearch(
    pos: &Position,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    shared: &SearchShared,
    thread: &mut SearchThread,
    history: &[Zobrist64],
) -> i32 {
    if thread.mark_node(shared) {
        return 0;
    }
    thread.stats.qsearch_nodes += 1;
    if ply >= MAX_PLY - 1 {
        return thread.evaluate_at(pos, shared, ply);
    }

    let in_check = pos.is_check();
    let static_eval = thread.evaluate_at(pos, shared, ply);
    let mut moves = if in_check {
        pos.legal_moves()
    } else {
        pos.capture_moves()
    };
    if moves.is_empty() && in_check {
        // Checkmate: keep the real mate distance — a tablebase verdict here
        // would flatten it to a mere "loss".
        return mated_in(ply as i32);
    }
    if moves.is_empty() {
        // Nothing worth capturing: a quiet leaf. A definitive tablebase
        // verdict supersedes the stand-pat evaluation (for every non-TB
        // position the probe is a no-op and this path returns exactly what
        // it did before); otherwise stand pat at the static evaluation
        // instead of pretending the position is drawn.
        if let Some(score) = shared.probe_tb(pos, thread) {
            return score;
        }
        return static_eval;
    }

    if is_draw(pos, thread, history, ply) {
        return 0;
    }

    // Syzygy: an exact outcome ends the leaf. Runs after the draw checks
    // (the 50-move rule and insufficient material already resolved the
    // position) and before the stand-pat, so a tablebase win/loss is
    // preferred over the static evaluation. Checkmate was handled above.
    if let Some(score) = shared.probe_tb(pos, thread) {
        return score;
    }

    // Stand pat: the position's static value alone caps the downside. Not in
    // check: passing is illegal there, so the score is the best evasion's and
    // starts from being mated, never from the static evaluation.
    let mut best = if in_check {
        mated_in(ply as i32)
    } else {
        if static_eval >= beta {
            return static_eval;
        }
        if static_eval > alpha {
            alpha = static_eval;
        }
        static_eval
    };

    let prev = thread.ctx[ply];
    let ant = thread.ctx[ply.saturating_sub(1)];
    // The stored TT move (if any) is the strongest cue even here — it only
    // ever reorders the generated plays, which are captures only when not in
    // check, so a quiet TT move is simply absent from the list.
    let tt_move = shared
        .tt
        .probe_move(pos.hash.into())
        .unwrap_or(RawMove::NULL);
    order_moves_ctx(
        &mut moves,
        pos,
        &thread.tables,
        tt_move,
        ply,
        prev,
        ant,
        &shared.params,
    );

    for i in 0..moves.len() {
        if thread.stopped {
            return best;
        }
        let m = moves.get(i);

        // A capture that gives check is not an ordinary capture: the node it
        // leads to has the *opponent* to move in check, where quiescence itself
        // is at full width, so it can resolve a line no static exchange
        // evaluation can see (a fork behind the capture, a discovered attack
        // the "defender" cannot answer). Both prunes below therefore skip it,
        // exactly as the main search's move loop already does. The test plays
        // the move on a scratch copy, so it only runs once a prune condition
        // has already fired.
        let mut gives_check = None;
        let mut gives_check = || *gives_check.get_or_insert_with(|| pos.make_child(m).is_check());

        // Delta pruning: the capture cannot possibly reach near alpha.
        if !in_check
            && best + touched_value(pos, m, &shared.params) + DELTA_MARGIN <= alpha
            && !gives_check()
        {
            thread.stats.delta_pruned += 1;
            continue;
        }
        // SEE pruning: a capture that loses material is a horizon blunder,
        // not a way out of the quiet-move tail. SEE is consulted first because
        // it is far cheaper than building the child position.
        if !in_check {
            thread.stats.see_calls += 1;
            if crate::move_ordering::see::see(pos.board(), m, &shared.params) < 0 && !gives_check()
            {
                thread.stats.see_pruned += 1;
                continue;
            }
        }

        let child = thread.make_child(pos, m, shared, ply + 1);
        thread.ctx[ply + 1] = MoveCtx::of(pos, m);
        let score = -qsearch(&child, -beta, -alpha, ply + 1, shared, thread, history);
        if score > best {
            best = score;
            if score >= beta {
                break;
            }
            if score > alpha {
                alpha = score;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::evaluation::Evaluator;
    use crate::search::{SearchShared, SearchThread};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64};

    fn q(fen: &str) -> i32 {
        let pos = Position::from_fen(fen).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let empty_tt = Arc::new(crate::tt::TranspositionTable::new(1));
        let default_params = Arc::new(crate::evaluation::EvalParams::default());
        let shared = SearchShared {
            tt: empty_tt,
            params: default_params,
            stop,
            nodes: Arc::new(AtomicU64::new(0)),
            node_cap: None,
            tb: Arc::new(crate::endgame::Syzygy::none()),
            nnue: None,
        };
        let mut thread = SearchThread::new();
        qsearch(&pos, -32_001, 32_001, 0, &shared, &mut thread, &[])
    }

    /// Runs qsearch with a custom (narrow) window, returning the score and the
    /// delta-prune counter so delta boundaries can be tested deterministically.
    fn q_window(fen: &str, alpha: i32, beta: i32) -> (i32, u64) {
        let pos = Position::from_fen(fen).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let empty_tt = Arc::new(crate::tt::TranspositionTable::new(1));
        let default_params = Arc::new(crate::evaluation::EvalParams::default());
        let shared = SearchShared {
            tt: empty_tt,
            params: default_params,
            stop,
            nodes: Arc::new(AtomicU64::new(0)),
            node_cap: None,
            tb: Arc::new(crate::endgame::Syzygy::none()),
            nnue: None,
        };
        let mut thread = SearchThread::new();
        let v = qsearch(&pos, alpha, beta, 0, &shared, &mut thread, &[]);
        (v, thread.stats.delta_pruned)
    }

    #[test]
    fn finds_free_queen() {
        // White to move: Rxc7 wins the undefended queen; quiescence must
        // leave white clearly ahead (a full rook/queen swing from -343).
        let v = q("6k1/2q5/8/8/8/8/8/2R3K1 w - - 0 1");
        assert!(v >= 500, "hanging queen must be found: got {v}");
    }

    #[test]
    fn avoids_losing_capture() {
        // Bxe5 (d4xe5) is a bishop-for-knight exchange defended by the f6 pawn
        // (SEE < 0); quiescence must keep the stand-pat instead of capturing.
        let fen = "6k1/8/5p2/4n3/3B4/8/8/4K3 w - - 0 1";
        let v = q(fen);
        let stand_pat = Evaluator.evaluate(&Position::from_fen(fen).unwrap());
        assert_eq!(v, stand_pat, "only capture is losing, must stand pat");
    }

    #[test]
    fn check_evenings_are_searched() {
        // Black to play, in check from the white rook on h8 along the 8th
        // rank; the king must step out (it cannot capture h8) because a
        // stand-pat would leave black down a rook.
        let v = q("4k2R/8/8/8/8/8/8/4K3 b - - 0 1");
        assert!(v < 0, "white is up a rook: got {v}");
    }

    #[test]
    fn in_check_never_stands_pat() {
        // The white knight on c7 checks the black king and forks the queen on
        // a8. Black is statically a queen up, but every evasion drops the
        // queen, so the score must come from the evasions, not the stand-pat.
        let fen = "q3k3/2N5/8/8/8/8/7P/6K1 b - - 0 1";
        let stand_pat = Evaluator.evaluate(&Position::from_fen(fen).unwrap());
        let v = q(fen);
        assert!(stand_pat > 0, "black is statically ahead: got {stand_pat}");
        assert!(v < 0, "the fork wins the queen: got {v}");
    }

    #[test]
    fn pinned_pieces_cannot_capture() {
        // Black's knight on c6 is pinned to its king on c8 by the white rook
        // on c2: Nxd4 would open the c-file and expose the king, so it is
        // illegal. That is black's *only* capture candidate, so quiescence
        // must stand pat instead of grabbing the "free" d4 pawn.
        let fen = "2k5/8/2n5/8/3P4/8/2R5/4K3 b - - 0 1";
        let v = q(fen);
        let stand_pat = Evaluator.evaluate(&Position::from_fen(fen).unwrap());
        assert_eq!(v, stand_pat, "the pinned capture is illegal: got {v}");
    }

    #[test]
    fn en_passant_capture_is_searched() {
        // White e5 pawn can take the black d5 pawn en passant (black just
        // played d7-d5, EP square d6): quiescence must pick up the free pawn.
        let fen = "6k1/8/8/3pP3/8/8/8/6K1 w - d6 0 1";
        let v = q(fen);
        let stand_pat = Evaluator.evaluate(&Position::from_fen(fen).unwrap());
        assert!(
            v >= stand_pat + 50,
            "e.p. capture must win the pawn: {v} vs stand-pat {stand_pat}"
        );
    }

    #[test]
    fn delta_pruning_skips_captures_far_below_alpha() {
        // Black is up a bishop (≈ +330 vs white's d4 pawn) and its only
        // capture is Bxd4 (a pawn). With a window demanding ~+550, the pawn
        // capture cannot possibly bridge `best(+330) + 100 + 200` to alpha,
        // so delta pruning must skip it and stand pat. The counter proves the
        // prune happened; the returned score must equal the stand-pat.
        let (v, delta_pruned) = q_window("4k3/8/8/8/3P4/8/8/4K1b1 b - - 0 1", 550, 900);
        let stand_pat =
            Evaluator.evaluate(&Position::from_fen("4k3/8/8/8/3P4/8/8/4K1b1 b - - 0 1").unwrap());
        assert_eq!(delta_pruned, 1, "the only capture must be delta-pruned");
        assert_eq!(v, stand_pat, "stand pat: the capture was skipped");
    }

    // --- Captures that give check -------------------------------------------

    /// Runs qsearch with the given window, returning the score together with
    /// the two prune counters the checking-capture rule is about:
    /// `(score, delta_pruned, see_calls, see_pruned)`.
    fn q_prunes(fen: &str, alpha: i32, beta: i32) -> (i32, u64, u64, u64) {
        let pos = Position::from_fen(fen).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let empty_tt = Arc::new(crate::tt::TranspositionTable::new(1));
        let default_params = Arc::new(crate::evaluation::EvalParams::default());
        let shared = SearchShared {
            tt: empty_tt,
            params: default_params,
            stop,
            nodes: Arc::new(AtomicU64::new(0)),
            node_cap: None,
            tb: Arc::new(crate::endgame::Syzygy::none()),
            nnue: None,
        };
        let mut thread = SearchThread::new();
        let v = qsearch(&pos, alpha, beta, 0, &shared, &mut thread, &[]);
        (
            v,
            thread.stats.delta_pruned,
            thread.stats.see_calls,
            thread.stats.see_pruned,
        )
    }

    #[test]
    fn a_checking_capture_survives_delta_pruning() {
        // White's stand pat is -241, and both available captures are worth
        // 100, so `best + 100 + 200 = 59` cannot reach a window demanding 400:
        // the delta margin rules out *both* captures on material alone. One of
        // them, Nxc6+, is not an ordinary capture — it forks the king on e7 and
        // the rook on a5, and the node it leads to has black in check, at full
        // width, where seven replies are resolved — so it must be searched.
        let fen = "8/4k3/2p5/rP6/1N6/8/6K1/8 w - - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        let m = pos.raw_move_from_uci("b4c6").unwrap();
        assert!(pos.raw_move_legal(m));
        assert!(pos.make_child(m).is_check(), "Nxc6+ must give check");
        let stand_pat = Evaluator.evaluate(&pos);

        let (v, delta_pruned, see_calls, see_pruned) = q_prunes(fen, 400, 32_001);
        assert_eq!(
            delta_pruned, 1,
            "only the quiet bxc6 may be delta-pruned; Nxc6+ gives check"
        );
        assert_eq!(see_pruned, 0, "no capture is SEE-pruned here");
        assert!(see_calls >= 1, "the SEE test must still be consulted");
        assert!(
            v > stand_pat + 400,
            "the fork must be found: stand pat {stand_pat}, qsearch {v}"
        );
    }

    #[test]
    fn a_checking_capture_survives_see_pruning() {
        // Rxb8+ looks like rook-for-knight: the bishop on e5 does guard b8, so
        // SEE is -180 and the exchange is "lost". It cannot actually recapture
        // — it is pinned against the king on e8 by the rook on e1, so b8 is
        // free and the check is resolved in white's favour. Pruning the move
        // because of a material-only static exchange throws away 600-odd
        // centipawns, so a capture that gives check skips the SEE test.
        let fen = "1n2k3/8/8/1R2b3/8/8/8/4R1K1 w - - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        let params = crate::evaluation::EvalParams::default();
        let m = pos.raw_move_from_uci("b5b8").unwrap();
        assert!(pos.raw_move_legal(m), "Rxb8+ must be legal");
        assert!(pos.make_child(m).is_check(), "Rxb8+ must give check");
        let see_value = crate::move_ordering::see::see(pos.board(), m, &params);
        assert!(
            see_value < 0,
            "the fixture needs a SEE-negative capture, got {see_value}"
        );
        // The pin really holds: black's only answers to the check are king moves.
        let child = pos.make_child(m);
        assert!(child.is_check());
        for reply in child.legal_moves().iter() {
            assert_ne!(
                reply.to_uci().as_str(),
                "e5b8",
                "Bxb8 must be illegal — the bishop is pinned"
            );
        }

        let stand_pat = Evaluator.evaluate(&pos);
        let (v, _, see_calls, see_pruned) = q_prunes(fen, -32_001, 32_001);
        assert!(
            v > stand_pat + 400,
            "the free knight must be won: stand pat {stand_pat}, qsearch {v}"
        );
        assert!(see_calls >= 1, "the SEE test must still be consulted");
        assert_eq!(see_pruned, 0, "no capture may be SEE-pruned");
    }

    #[test]
    fn no_checking_capture_is_see_pruned() {
        // A battery over positions in which *every* capture gives check, and at
        // least one of them loses material on paper. Each fixture is validated
        // first, so the counter assertion below is a statement about the rule
        // and not about the position: quiescence still asks about every one of
        // these captures, and prunes none of them. A capture may still be
        // *rejected on its merits* inside the child node, but never skipped
        // blind.
        for fen in [
            // Qf7xd7+ (the only capture) is a queen for a knight that is
            // defended twice over.
            "2bk4/2nr1Q2/p6B/7P/4P1P1/N6B/7K/R7 w - - 6 55",
            // Rxd7+ walks into the bishop on d7 and the king on b7.
            "5n1r/1k1b4/5p1p/7p/P1K5/RP1R4/8/8 w - - 2 55",
            // Nxb4+ is a knight for a pawn defended by two bishops.
            "8/r7/1k2P3/b1p4N/1n4R1/6Pp/b7/3K4 w - - 4 55",
            // Rxe5+ hangs the rook to the bishop on b4.
            "4kr2/8/r2p4/b3b1RP/p2pB3/2nK4/P7/8 w - - 4 53",
            // Rxf4+ is a rook for a pawn that is defended.
            "5k2/8/3R4/p3p2P/P2Npb2/7P/5RK1/1r6 w - - 3 52",
            // And the pinned bishop from the score test above.
            "1n2k3/8/8/1R2b3/8/8/8/4R1K1 w - - 0 1",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let params = crate::evaluation::EvalParams::default();
            let captures = pos.capture_moves();
            assert!(!captures.is_empty(), "{fen}: the fixture needs a capture");
            let mut losing = 0;
            for m in captures.iter() {
                assert!(
                    pos.make_child(m).is_check(),
                    "{fen} {}: the fixture needs every capture to give check",
                    m.to_uci()
                );
                if crate::move_ordering::see::see(pos.board(), m, &params) < 0 {
                    losing += 1;
                }
            }
            assert!(
                losing > 0,
                "{fen}: the fixture needs a capture that loses material"
            );

            let (_, _, see_calls, see_pruned) = q_prunes(fen, -32_001, 32_001);
            assert!(see_calls >= 1, "{fen}: the SEE test was never consulted");
            assert_eq!(
                see_pruned, 0,
                "{fen}: a capture that gives check must never be SEE-pruned"
            );
        }
    }

    // --- Syzygy -------------------------------------------------------------

    /// Like `q` but with the real 3/4-piece tables from `tests/data/syzygy`
    /// loaded, returning `(score, &thread.stats)` so TB counters can be
    /// asserted too.
    fn q_tb(fen: &str) -> (i32, crate::search::stats::SearchStats) {
        let pos = Position::from_fen(fen).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let empty_tt = Arc::new(crate::tt::TranspositionTable::new(1));
        let default_params = Arc::new(crate::evaluation::EvalParams::default());
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/syzygy");
        let (tb, report) = crate::endgame::Syzygy::load(dir.to_str().unwrap());
        assert_eq!(report.max_pieces, 4, "tables must be present for this test");
        let shared = SearchShared {
            tt: empty_tt,
            params: default_params,
            stop,
            nodes: Arc::new(AtomicU64::new(0)),
            node_cap: None,
            tb: Arc::new(tb),
            nnue: None,
        };
        let mut thread = SearchThread::new();
        let v = qsearch(&pos, -32_001, 32_001, 0, &shared, &mut thread, &[]);
        (v, thread.stats)
    }

    #[test]
    fn tablebase_win_outranks_the_quiet_stand_pat() {
        // A *quiet* leaf — no captures at all — so the old code would have
        // stood pat at the static evaluation; the Syzygy verdict must
        // override it with the exact win band. The quiet path is the one
        // that demands the probe before the stand-pat return.
        let (v, stats) = q_tb("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
        assert_eq!(v, 19_999, "unconditional TB win band");
        assert!(stats.tb_probes >= 1);
        assert!(stats.tb_hits >= 1);
        assert!(stats.tb_wins >= 1);
    }

    #[test]
    fn tablebase_loss_resolves_the_leaf() {
        let (v, stats) = q_tb("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
        assert_eq!(v, -19_999, "unconditional TB loss band (win-symmetric)");
        assert!(stats.tb_losses >= 1, "loss counter must increment");
    }

    #[test]
    fn tablebase_checkmate_still_returns_a_mate_score() {
        // Black is checkmated (Kg6 + Qg7 vs Kh8): qsearch must return the
        // real mate distance, never a mere tablebase "loss" (-20000).
        let (v, _) = q_tb("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
        assert!(
            v < -20_000 && v > -32_001,
            "checkmate must keep its mate score: got {v}"
        );
    }

    #[test]
    fn tablebase_draw_leaf_scores_zero() {
        let (v, stats) = q_tb("4k2r/8/8/8/8/8/8/3RK3 w - - 0 1");
        assert_eq!(v, 0, "KRvKR draw");
        assert!(stats.tb_draws >= 1, "draw counter must increment");
    }
}

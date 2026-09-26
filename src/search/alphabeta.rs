//! PVS alpha-beta search: the heart of the engine.
//!
//! One node:
//!
//! 1. node bookkeeping (count, stop) and search-ply guards,
//! 2. mate-distance pruning,
//! 3. terminal/draw detection,
//! 4. transposition-table probe (with mate-score re-basing),
//! 5. node-level pruning: reverse futility, razoring, null-move probing and
//!    ProbCut (guarded against check, PV nodes, mate bounds and pawn-only
//!    zugzwang positions),
//! 6. move generation, ordering (TT move, MVV-LVA, killers, history),
//! 7. the principal-variation loop: futility pruning, SEE pruning,
//!    history pruning, late-move reductions (depth/move-number/history
//!    driven, with full-depth re-search on fail-high), PVS zero-window
//!    re-searches,
//! 8. history/killer updates on quiet beta cutoffs,
//! 9. TT store (skipped once the search is aborted).
//!
//! The board is never mutated: every child is a fresh `Position` clone
//! ([`crate::board::Position::make_child`]), so push/pop bugs are impossible
//! and sibling searches never interfere.

use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::move_ordering::history::MoveCtx;
use crate::move_ordering::{is_capture_or_promotion, moving_role, order_moves_ctx, see};
use crate::search::pruning;
use crate::search::reductions;
use crate::search::{
    SearchShared, SearchThread, is_draw, node_score_to_tt, qsearch, tt_score_to_node,
};
use crate::tt::Bound;
use crate::types::{Depth, INFINITE, MAX_MOVES, MAX_PLY, RawMove, mate_in, mated_in};

/// Scores a node with the PVS alpha-beta algorithm.
///
/// `depth` is the remaining depth (≥ 1); `ply` is the distance from the root
/// (drives mate scoring, killers, and the search stacks); `alpha`/`beta` form
/// the search window, from the point of view of the side to move; `history`
/// carries the pre-root position hashes for repetition detection; `allow_null`
/// permits a null-move probe at this node (false directly under a null move,
/// so two passes can never be played in a row).
pub fn alphabeta(
    pos: &Position,
    mut alpha: i32,
    beta: i32,
    depth: Depth,
    ply: usize,
    shared: &SearchShared,
    thread: &mut SearchThread,
    history: &[Zobrist64],
    allow_null: bool,
) -> i32 {
    if thread.mark_node(shared) {
        return 0;
    }

    // Every non-terminal node starts with an empty PV row; terminal nodes
    // (mate/stalemate, qsearch, TT cutoffs) never write one, so parents must
    // not read stale child lines.
    thread.pv_len[ply] = 0;

    let pv_node = beta - alpha > 1;
    let in_check = pos.is_check();

    // Mate distance pruning: keep the window inside what this ply can achieve.
    alpha = alpha.max(mated_in(ply as i32));
    let beta = beta.min(mate_in(ply as i32 + 1));
    if alpha >= beta {
        return alpha;
    }

    if ply >= MAX_PLY - 1 {
        return thread.evaluate_at(pos, shared, ply);
    }
    if depth <= 0 {
        return qsearch::qsearch(pos, alpha, beta, ply, shared, thread, history);
    }

    // Transposition-table probe.
    thread.stats.tt_probes += 1;
    let tt_entry = shared.tt.probe(pos.hash.into());
    let tt_move = tt_entry.map_or(RawMove::NULL, |e| e.mv);
    if let Some(tt) = tt_entry {
        thread.stats.tt_hits += 1;
        if tt.depth >= depth && !pv_node {
            let score = tt_score_to_node(tt.score, ply);
            match tt.bound {
                Bound::Exact => {
                    thread.stats.tt_cutoffs += 1;
                    return score;
                }
                Bound::Lower if score >= beta => {
                    thread.stats.tt_cutoffs += 1;
                    return score;
                }
                Bound::Upper if score <= alpha => {
                    thread.stats.tt_cutoffs += 1;
                    return score;
                }
                _ => {}
            }
        }
    }

    let mut moves = pos.legal_moves();
    if moves.is_empty() {
        return if in_check { mated_in(ply as i32) } else { 0 };
    }

    if is_draw(pos, thread, history, ply) {
        return 0;
    }

    let static_eval = thread.evaluate_at(pos, shared, ply);
    thread.evals[ply] = static_eval;
    thread.hashes[ply] = pos.hash;

    let improving = !in_check && ply >= 2 && static_eval > thread.evals[ply - 2];

    // Reverse futility pruning: the position is already so good that no
    // opponent reply can bring it back below beta.
    if !pv_node
        && !in_check
        && depth <= pruning::RFP_DEPTH
        && static_eval - pruning::reverse_futility_margin(depth, improving) >= beta
    {
        thread.stats.rfp_pruned += 1;
        return static_eval;
    }

    // Razoring: shallow and hopeless — drop straight into quiescence; no
    // quiet move can recover a deficit this large. A "successful" razor is
    // one whose qsearch confirms the fail-low (`score <= alpha`); the node
    // was resolved without the quiet-move subtree.
    if !pv_node
        && !in_check
        && depth <= pruning::RAZOR_DEPTH
        && static_eval + pruning::razor_margin(depth) < alpha
    {
        thread.stats.razor_attempts += 1;
        // Same position, same ply: the accumulator slot for `ply` is still
        // valid, so quiescence reuses it.
        let razor_score = qsearch::qsearch(pos, alpha, beta, ply, shared, thread, history);
        if razor_score <= alpha {
            thread.stats.razor_cutoffs += 1;
        }
        return razor_score;
    }

    // Null-move pruning: pass, and if the opponent still cannot beat beta
    // the move loop is skipped entirely. Never in PV nodes (they must return
    // exact scores), in check (passing is illegal), twice in a row
    // (`allow_null`), or when the side to move has only pawns (passing is a
    // real move there and zugzwang can flip the result). The reduction is
    // adaptive: a position whose eval has not improved ("stagnant") gets one
    // extra ply of reduction, since the pass is more likely to hold.
    if !pv_node
        && !in_check
        && allow_null
        && depth >= pruning::NULL_MOVE_MIN_DEPTH
        && static_eval >= beta
        && pruning::side_has_attacking_pieces(pos.board(), pos.turn())
    {
        if pos.null_move().is_some() {
            thread.stats.null_probes += 1;
            thread.ctx[ply + 1] = None;
            // Passing changes nothing on the board, so the child's accumulator
            // is the parent's — a copy, not a refresh.
            let nulled = thread.make_child_null(pos, shared, ply + 1);
            let null_score = -alphabeta(
                &nulled,
                -beta,
                -beta + 1,
                depth - 1 - pruning::null_move_reduction(depth, improving),
                ply + 1,
                shared,
                thread,
                history,
                false,
            );
            if thread.stopped {
                return 0;
            }
            if null_score >= beta {
                thread.stats.null_cutoffs += 1;
                return null_score;
            }
        }
    }

    // ProbCut: at depth, at a cut node, a static advantage over `beta` by a
    // healthy margin makes a fail-high overwhelming; a shallow null-window
    // proof search decides it cheaply and the whole subtree is skipped.
    //
    // Guards mirror null move's: never in PV nodes (exact scores required),
    // in check (tactics stay full-width), near mate bounds (returning `beta`
    // there would distort mate distances), or with only pawns left (zugzwang
    // can make "the opponent cannot improve" false). The proof search is a
    // normal `alphabeta` at `depth - PROBCUT_DEPTH`, so its TT entries are
    // valid for *its* depth; the cut node itself stores nothing (no move was
    // established).
    if !pv_node
        && !in_check
        && allow_null
        && depth >= pruning::PROBCUT_DEPTH
        && beta.abs() < crate::search::MATE_ZONE
        && static_eval + pruning::probcut_margin(depth) >= beta
        && pruning::side_has_attacking_pieces(pos.board(), pos.turn())
    {
        thread.stats.probcut_attempts += 1;
        // Re-entering at the *same* position and ply: slot `ply` still describes
        // `pos`, so the proof search needs no accumulator update at all.
        let prob_score = alphabeta(
            pos,
            beta - 1,
            beta,
            depth - pruning::PROBCUT_DEPTH,
            ply,
            shared,
            thread,
            history,
            true,
        );
        if thread.stopped {
            return 0;
        }
        if prob_score >= beta {
            thread.stats.probcut_cutoffs += 1;
            return beta;
        }
    }

    let prev = thread.ctx[ply];
    let ant = thread.ctx[ply.saturating_sub(1)];
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

    let mut best = -INFINITE;
    let mut best_move = RawMove::NULL;
    let original_alpha = alpha;
    let mut searched = 0usize;
    let mut quiets_failed: [RawMove; MAX_MOVES] = [RawMove::NULL; MAX_MOVES];
    let mut num_failed = 0usize;
    let mut caps_failed: [RawMove; MAX_MOVES] = [RawMove::NULL; MAX_MOVES];
    let mut num_caps_failed = 0usize;

    for i in 0..moves.len() {
        if thread.stopped {
            return 0;
        }
        let m = moves.get(i);
        let is_tactical = is_capture_or_promotion(pos.board(), m);

        // --- Pruning (only below the root of the local window) ---

        // Futility: a quiet move cannot lift the static position to alpha.
        if !pv_node
            && !in_check
            && !is_tactical
            && depth <= pruning::FUTILITY_DEPTH
            && searched > 0
            && static_eval + pruning::futility_margin(depth, improving) <= alpha
        {
            thread.stats.futility_pruned += 1;
            continue;
        }

        // SEE pruning: hopeless captures are skipped outright.
        if !pv_node && !in_check && is_tactical {
            thread.stats.see_calls += 1;
            let see_v = see::see(pos.board(), m, &shared.params);
            if see_v < pruning::see_prune_threshold(depth) {
                thread.stats.see_pruned += 1;
                continue;
            }
        }

        // History-based quiet pruning at shallow depths.
        if !pv_node
            && !in_check
            && !is_tactical
            && depth <= 6
            && searched >= pruning::QUIET_PRUNE_LIMIT
            && thread.tables.history.history_score(pos.turn(), m) < pruning::HISTORY_PRUNE_THRESHOLD
        {
            thread.stats.history_pruned += 1;
            continue;
        }

        // --- Search the move ---
        let child = thread.make_child(pos, m, shared, ply + 1);
        thread.ctx[ply + 1] = MoveCtx::of(pos, m);

        // Check extension: a move that gives check is searched one ply deeper,
        // so forced tactics at the horizon are never truncated. The child node
        // would compute `in_check` for itself anyway; this shares that result.
        let child_depth = depth - 1 + i32::from(child.is_check());

        // --- Late move reductions (quiet, late, non-checking moves) ---
        // The reduction is a pure function of depth, move number (how late in
        // the ordering), node type, "improving" status, check status and the
        // move's learned history — a move that has proven good is searched
        // deeper, one that keeps failing is cut more. Only the first move (or
        // the first two at PV nodes) is never reduced, so the PV cannot be
        // lost to an over-reduction.
        let mut new_depth = child_depth;
        let mut reduced = false;
        if child_depth == depth - 1
            && !is_tactical
            && depth >= 3
            && searched >= 2 + usize::from(pv_node)
            && !(in_check && searched < 4)
        {
            let history = thread.tables.history.history_score(pos.turn(), m);
            let r = reductions::lmr_reduction(
                depth,
                searched + 1,
                pv_node,
                improving,
                in_check,
                history,
            );
            if r > 0 {
                let d = (child_depth - r).max(1);
                reduced = d < child_depth;
                new_depth = d;
            }
        }
        if reduced {
            thread.stats.lmr_reduced += 1;
        }

        let score = if searched == 0 {
            -alphabeta(
                &child,
                -beta,
                -alpha,
                child_depth,
                ply + 1,
                shared,
                thread,
                history,
                true,
            )
        } else {
            let mut s = -alphabeta(
                &child,
                -alpha - 1,
                -alpha,
                new_depth,
                ply + 1,
                shared,
                thread,
                history,
                true,
            );
            if s > alpha && reduced {
                thread.stats.lmr_researched += 1;
                s = -alphabeta(
                    &child,
                    -alpha - 1,
                    -alpha,
                    child_depth,
                    ply + 1,
                    shared,
                    thread,
                    history,
                    true,
                );
            }
            if pv_node && s > alpha && s < beta {
                s = -alphabeta(
                    &child,
                    -beta,
                    -alpha,
                    child_depth,
                    ply + 1,
                    shared,
                    thread,
                    history,
                    true,
                );
            }
            s
        };

        if thread.stopped {
            return 0;
        }

        searched += 1;
        if score <= original_alpha {
            if is_tactical {
                caps_failed[num_caps_failed] = m;
                num_caps_failed += 1;
            } else {
                quiets_failed[num_failed] = m;
                num_failed += 1;
            }
        }

        if score > best {
            best = score;
            best_move = m;
            thread.pv_push(ply, m);
            if score > alpha {
                alpha = score;
                if score >= beta {
                    // Beta cutoff: honor the cutoff move in the ordering
                    // tables and punish everything that failed to raise the
                    // score along the way (quiets and captures alike).
                    if !is_tactical {
                        update_quiet_tables(thread, pos, m, prev, ant, ply, depth);
                    } else {
                        thread
                            .tables
                            .history
                            .update_capture(pos.board(), m, history_bonus(depth));
                    }
                    let fail_bonus = -history_bonus(depth);
                    for &fm in &quiets_failed[..num_failed] {
                        if fm != RawMove::NULL {
                            thread
                                .tables
                                .history
                                .update_history(pos.turn(), fm, fail_bonus);
                            if let Some(ctx) = MoveCtx::of(pos, fm) {
                                thread.tables.history.update_continuation(
                                    pos.turn(),
                                    ctx.piece,
                                    ctx.to,
                                    prev,
                                    ant,
                                    fail_bonus,
                                );
                            }
                        }
                    }
                    for &fm in &caps_failed[..num_caps_failed] {
                        if fm != RawMove::NULL {
                            thread
                                .tables
                                .history
                                .update_capture(pos.board(), fm, fail_bonus);
                        }
                    }
                    thread.stats.beta_cutoffs += 1;
                    thread.stats.moves_until_cutoff += searched as u64;
                    if searched == 1 {
                        thread.stats.first_move_cutoffs += 1;
                    }
                    break;
                }
            }
        }
    }

    if !thread.stopped {
        let bound = if best >= beta {
            Bound::Lower
        } else if pv_node && best > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        shared.tt.store(
            pos.hash.into(),
            best_move,
            node_score_to_tt(best, ply),
            depth,
            bound,
        );
    }
    best
}

/// History/killer updates for a quiet move that caused a beta cutoff.
#[inline]
fn update_quiet_tables(
    thread: &mut SearchThread,
    pos: &Position,
    m: RawMove,
    prev: Option<MoveCtx>,
    ant: Option<MoveCtx>,
    ply: usize,
    depth: Depth,
) {
    thread.tables.killers.store(ply, m);
    let bonus = history_bonus(depth);
    thread.tables.history.update_history(pos.turn(), m, bonus);
    thread.tables.history.set_counter_move(pos.turn(), prev, m);
    if let Some(role) = moving_role(pos, m) {
        thread.tables.history.update_continuation(
            pos.turn(),
            role as usize,
            m.to(),
            prev,
            ant,
            bonus,
        );
    }
}

/// The history bonus for a node searched with `depth` plies remaining.
#[inline]
fn history_bonus(depth: Depth) -> i32 {
    use crate::move_ordering::history::bonus;
    bonus(depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    /// Runs a single fixed-depth search from a FEN.
    fn search_depth(fen: &str, depth: Depth) -> crate::search::SearchResult {
        let pos = Position::from_fen(fen).unwrap();
        let mut searcher = crate::search::Searcher::new(4);
        let stop = AtomicBool::new(false);
        let limits = crate::search::TimeLimit {
            depth: Some(depth),
            nodes: Some(10_000_000),
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        searcher.search(&pos, &[], &limits, &Arc::new(stop), 1, &[])
    }

    #[test]
    fn finds_forced_mate_in_two() {
        // 1.Rd8+! Rxd8 2.Rxd8# — a three-ply forced mate. The rook on d1
        // sits behind the checker on d2 so the recapture is possible; black's
        // only reply to the check is Rxd8 (verified with shakmaty).
        let fen = "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1";
        let r = search_depth(fen, 4);
        assert!(r.best != RawMove::NULL, "must find a move");
        assert!(
            crate::types::is_mate(r.score),
            "mate in two: got {}",
            r.score
        );
        assert!(
            (3..=4).contains(&r.pv.len()),
            "PV must show the mating line, got {:?}",
            r.pv
        );
        assert_eq!(
            r.pv.first().map(|m| m.to_uci()).unwrap(),
            "d2d8",
            "the mating key must be Rd8+"
        );
    }

    #[test]
    fn wins_hanging_queen() {
        // White is a free capture of the undefended black queen on f2.
        let fen = "6k1/8/8/8/8/8/5q2/5Q1K w - - 0 1";
        let r = search_depth(fen, 3);
        assert_eq!(r.best.to_uci().as_str(), "f1f2", "must take the queen");
        assert!(r.score > 700, "white wins the queen: got {}", r.score);
    }

    #[test]
    fn returns_legal_best_move() {
        let r = search_depth(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            4,
        );
        assert!(r.best != RawMove::NULL);
        let pos = Position::startpos();
        assert!(
            pos.raw_move_legal(r.best),
            "best move must be legal, got {}",
            r.best.to_uci()
        );
    }

    /// Searches `fen` at `depth` on a *shared* searcher (warm TT between
    /// calls). Returns the result.
    fn search_on(
        searcher: &mut crate::search::Searcher,
        fen: &str,
        depth: Depth,
    ) -> crate::search::SearchResult {
        let pos = Position::from_fen(fen).unwrap();
        let stop = AtomicBool::new(false);
        let limits = crate::search::TimeLimit {
            depth: Some(depth),
            nodes: Some(10_000_000),
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        searcher.search(&pos, &[], &limits, &Arc::new(stop), 1, &[])
    }

    #[test]
    fn aspiration_research_keeps_scores_stable() {
        // Aspiration narrows the window from depth 4 on; adjacent depths must
        // still agree on the position's value within the widening budget.
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ] {
            let r4 = search_depth(fen, 4);
            let r5 = search_depth(fen, 5);
            assert_eq!(r4.depth, 4);
            assert_eq!(r5.depth, 5, "search must reach the requested depth");
            assert!(r4.best != RawMove::NULL && r5.best != RawMove::NULL);
            assert!(
                (r5.score - r4.score).abs() < 300,
                "aspiration re-search drifted: d4 {} vs d5 {}",
                r4.score,
                r5.score
            );
        }
    }

    #[test]
    fn aspiration_keeps_mate_scores_intact() {
        // 1.Rd8+! Rxd8 2.Rxd8# forces repeated fail-highs that widen the
        // window; the mate must survive the widen-and-research dance.
        let fen = "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1";
        let r = search_depth(fen, 6);
        assert!(crate::types::is_mate(r.score), "got {}", r.score);
        assert_eq!(
            r.pv.first().map(|m| m.to_uci()).unwrap(),
            "d2d8",
            "the mating key must be Rd8+"
        );
    }

    #[test]
    fn check_chain_mate_survives_minimal_depth() {
        // The mating line is a checking quiet (Rd8+), a forced capture reply,
        // and a recapture mate. At depth 3 the full-width search (with the
        // check extension keeping the reply out of the horizon) must still
        // see the mate in all three plies.
        let fen = "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1";
        let r = search_depth(fen, 3);
        assert!(
            crate::types::is_mate(r.score),
            "mate must be visible at depth 3: got {}",
            r.score
        );
    }

    #[test]
    fn pv_is_a_legal_line_to_play() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let r = search_depth(fen, 5);
        assert!(r.best != RawMove::NULL);
        assert!(
            r.pv.first().copied() == Some(r.best),
            "PV must start with the best move"
        );
        let mut pos = Position::from_fen(fen).unwrap();
        for (i, &m) in r.pv.iter().enumerate() {
            assert!(m != RawMove::NULL);
            assert!(
                pos.raw_move_legal(m),
                "PV move {} at index {i} is illegal",
                m.to_uci()
            );
            pos = pos.make_child(m);
        }
    }

    #[test]
    fn single_thread_stats_are_deterministic() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let a = search_depth(fen, 4);
        let b = search_depth(fen, 4);
        assert_eq!(a.best, b.best);
        assert_eq!(a.score, b.score);
        assert_eq!(
            a.nodes, b.nodes,
            "node count must be a pure function of the search"
        );
        assert_eq!(
            a.stats, b.stats,
            "instrumentation must also be deterministic"
        );
    }

    #[test]
    fn pre_aborted_search_writes_no_tt_entries() {
        // The stop flag is set *before* the search starts. The search may
        // still enter up to one 1024-node sampling window before noticing
        // (mark_node refreshes the stop signal every 1024 nodes) — but from
        // the moment it notices, every node is aborted and no TT store,
        // probe result or cutoff may be recorded. The store counter proves
        // the "aborted searches must not write TT" invariant.
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        let mut searcher = crate::search::Searcher::new(4);
        let stop = AtomicBool::new(true); // external stop already requested
        let limits = crate::search::TimeLimit {
            depth: Some(16),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let r = searcher.search(&pos, &[], &limits, &Arc::new(stop), 1, &[]);
        assert!(r.stopped, "search must report the pre-requested abort");
        let stores = searcher.tt.stores();
        assert_eq!(stores, r.tt_stores, "harvested counter must match the TT");
        assert!(
            stores <= 1024 && stores <= r.nodes,
            "only the pre-notice sampling window may store: {stores} stores, {} nodes",
            r.nodes
        );
        assert!(
            r.stats.tt_probes <= 1024,
            "a pre-aborted search may probe at most one sampling window"
        );
        // The pruning stack must not skirt the abort: every counter is at
        // most the number of nodes actually entered, and cutoffs never exceed
        // their attempts.
        let s = &r.stats;
        for (name, v) in [
            ("null_probes", s.null_probes),
            ("probcut_attempts", s.probcut_attempts),
            ("razor_attempts", s.razor_attempts),
        ] {
            assert!(v <= r.nodes, "{name} {v} must not exceed nodes {}", r.nodes);
        }
        assert!(s.null_cutoffs <= s.null_probes);
        assert!(s.probcut_cutoffs <= s.probcut_attempts);
        assert!(s.lmr_researched <= s.lmr_reduced);
        assert!(s.total_pruned() <= r.nodes, "pruned can never exceed nodes");
    }

    #[test]
    fn node_capped_search_aborts_deterministically() {
        // A small node cap stops the search mid-iteration (the cap is checked
        // on the 1024-node sampling cadence): the abort must be reported and
        // the whole trajectory — nodes, counters, result — reproducible.
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        let run = |cap: u64| {
            let mut searcher = crate::search::Searcher::new(4);
            let stop = AtomicBool::new(false);
            let limits = crate::search::TimeLimit {
                depth: Some(12),
                nodes: Some(cap),
                movetime_ms: None,
                soft_ms: 0,
                hard_ms: 0,
                infinite: true,
            };
            searcher.search(&pos, &[], &limits, &Arc::new(stop), 1, &[])
        };
        let a = run(50_000);
        assert!(a.stopped, "the node cap must abort the search");
        assert!(
            a.best != RawMove::NULL,
            "the completed part still yields a move"
        );
        let b = run(50_000);
        assert_eq!(a.nodes, b.nodes, "aborted search must be deterministic");
        assert_eq!(a.stats, b.stats);
        assert_eq!(a.best, b.best);
    }

    #[test]
    fn mate_scores_rebase_correctly_through_the_tt() {
        // The same position searched again on a warm table must reproduce the
        // mate: stored scores are ply-relative and `tt_score_to_node` must
        // restore them exactly at ply 0.
        let fen = "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1";
        let mut searcher = crate::search::Searcher::new(4);
        let deep = search_on(&mut searcher, fen, 4);
        assert!(
            crate::types::is_mate(deep.score) && crate::types::mate_plies(deep.score) <= 3,
            "case must be mate in <= 3 plies: got {}",
            deep.score
        );
        // Shallow re-search on the same table: the stored mate must survive
        // the probe (and re-base to the same "mate in N plies" family).
        let shallow = search_on(&mut searcher, fen, 2);
        assert!(
            crate::types::is_mate(shallow.score) && crate::types::mate_plies(shallow.score) <= 3,
            "TT mate score must survive re-basing: got {}",
            shallow.score
        );
    }

    // --- Stage 3: modern pruning & reductions ------------------------------

    #[test]
    fn null_move_never_fires_in_pawn_endgames() {
        // White is clearly winning (bishop-less, rook-less): a pure
        // king-and-pawn endgame. Passing is a *real* move here and zugzwang
        // can flip the result, so the null-move gate
        // (`side_has_attacking_pieces`) must keep the probe off everywhere.
        // Without the gate this search would probe null moves (white's static
        // eval beats beta at cut nodes) and could score the position wrong.
        let fen = "8/8/8/8/8/4P3/4K3/7k w - - 0 1";
        let r = search_depth(fen, 6);
        assert_eq!(r.depth, 6);
        assert!(r.best != RawMove::NULL);
        assert_eq!(
            r.stats.null_probes, 0,
            "no attacking pieces: null move must never fire (score {})",
            r.score
        );
        assert!(r.score > 0, "white is a pawn up: got {}", r.score);
    }

    #[test]
    fn null_move_fires_and_cuts_in_middlegames() {
        // With pieces on the board the probe is both attempted and — along
        // the many stable, favorable cut nodes of a quiet middlegame — lands.
        let r = search_depth(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            6,
        );
        assert!(
            r.stats.null_probes > 0,
            "middlegame cut nodes must probe null moves"
        );
        assert!(
            r.stats.null_cutoffs > 0,
            "some probes must actually cut ({} probes)",
            r.stats.null_probes
        );
        assert!(r.stats.null_cutoffs <= r.stats.null_probes);
    }

    #[test]
    fn lmr_reduces_and_re_searches_quiet_moves() {
        // A balanced middlegame searched deep enough for both halves of LMR
        // to show up: many late quiet moves are searched reduced, and at least
        // one reduced move beats alpha and is re-searched at full depth — the
        // fail-low safety valve. (The re-search is genuinely rare with good
        // ordering, so this needs depth, not just a quiet position.)
        let r = search_depth(
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
            10,
        );
        assert!(r.stats.lmr_reduced > 0, "late quiet moves must be reduced");
        assert!(
            r.stats.lmr_researched > 0,
            "a reduced move must beat alpha and be re-searched ({} reduced)",
            r.stats.lmr_reduced
        );
        assert!(r.stats.lmr_researched <= r.stats.lmr_reduced);
    }

    #[test]
    fn futility_pruning_fires_at_the_horizon() {
        // Kiwipete at depth 4: lines where one side's static eval gets
        // materially worse (a piece lost in that line) cannot recover to the
        // already-established bound — their quiet moves are futility-pruned.
        let r = search_depth(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            4,
        );
        assert!(
            r.stats.futility_pruned > 0,
            "hopeless quiet moves at the horizon must be futility-pruned"
        );
        assert!(r.best != RawMove::NULL);
    }

    #[test]
    fn razoring_fires_when_hopeless() {
        // Shallow kiwipete: some nodes are so far below the bound already
        // established by their siblings that the razor margin dwarfs any
        // possible recovery — those nodes drop straight into quiescence.
        let r = search_depth(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            4,
        );
        assert!(
            r.stats.razor_attempts > 0,
            "hopeless shallow nodes must be razored"
        );
        assert!(r.stats.razor_cutoffs <= r.stats.razor_attempts);
    }

    #[test]
    fn probcut_fires_and_keeps_the_win() {
        // White is up a whole queen. Deep cut nodes with a huge static
        // advantage over beta must be resolved by the shallow ProbCut proof
        // search — both attempted and cutting — while the move itself (taking
        // the queen) stays the best line.
        let fen = "6k1/8/8/8/8/8/5q2/5Q1K w - - 0 1";
        let r = search_depth(fen, 8);
        assert!(
            r.stats.probcut_attempts > 0,
            "a winning position at depth 8 must probe ProbCut"
        );
        assert!(
            r.stats.probcut_cutoffs > 0,
            "some probes must cut ({} attempts)",
            r.stats.probcut_attempts
        );
        assert_eq!(
            r.best.to_uci().as_str(),
            "f1f2",
            "the winning capture must survive ProbCut"
        );
        assert!(r.score > 700, "white wins the queen: got {}", r.score);
    }

    #[test]
    fn mate_survives_the_full_prune_stack() {
        // Stage 2's mate battery with the whole pruning stack live: ProbCut
        // may only ever *skip cut nodes that fail high*, which can never turn
        // a real mate into a non-mate; the forced Rd8+! Rxd8 Rxd8# must still
        // be found and its key must not change.
        let fen = "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1";
        let r = search_depth(fen, 8);
        assert!(
            crate::types::is_mate(r.score),
            "mate must survive the prunes: got {}",
            r.score
        );
        assert_eq!(
            r.pv.first().map(|m| m.to_uci()).unwrap(),
            "d2d8",
            "the mating key must be Rd8+"
        );
    }

    #[test]
    fn reverse_futility_fires_at_shallow_depth() {
        // White up a queen is so far above beta (minus the RFP margin) at
        // shallow nodes that the whole search is skipped at the static value.
        let r = search_depth("6k1/8/8/8/8/8/5q2/5Q1K w - - 0 1", 4);
        assert!(
            r.stats.rfp_pruned > 0,
            "overwhelming static advantage must trigger reverse futility"
        );
    }

    #[test]
    fn pruning_stack_does_not_bias_aspiration_stability() {
        // Re-run the aspiration stability check with the pruning stack live:
        // adjacent depths must still agree on the value within the widening
        // budget (aspiration re-searches must not be fooled by reduced scores).
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ] {
            let r4 = search_depth(fen, 4);
            let r5 = search_depth(fen, 5);
            assert_eq!(r4.depth, 4);
            assert_eq!(r5.depth, 5);
            assert!(
                (r5.score - r4.score).abs() < 300,
                "prunes drifted d4 {} vs d5 {}",
                r4.score,
                r5.score
            );
        }
    }
}

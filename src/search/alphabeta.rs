//! PVS alpha-beta search: the heart of the engine.
//!
//! This version keeps the Stage-02 correctness contract from the previous
//! search while restoring the Stockfish-19 search mechanisms that sit on top
//! of it: singular extensions, multi-cut / negative extensions, cut-node
//! propagation, IIR, followPV, the modern LMR model, SF19 child pruning,
//! SF19 null move / ProbCut, TT-ProbCut, SF19 late move pruning and the
//! stricter TT-cutoff / GHI path.

use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::move_ordering::history::MoveCtx;
use crate::move_ordering::{
    is_capture_or_promotion, moving_role, order_moves_ctx, see, victim_value,
};
use crate::search::correction;
use crate::search::pruning;
use crate::search::reductions::{self, LmrSignals};
use crate::search::singular::{self, SingularCtx, SingularOutcome};
use crate::search::{
    SearchShared, SearchThread, is_draw, node_score_to_tt, qsearch, tt_score_to_node,
};
use crate::tt::Bound;
use crate::types::{Depth, INFINITE, MAX_MOVES, MAX_PLY, RawMove, mate_in, mated_in};

/// Scores a node with PVS alpha-beta.
#[allow(clippy::too_many_arguments)]
pub fn alphabeta(
    pos: &Position,
    mut alpha: i32,
    beta: i32,
    mut depth: Depth,
    ply: usize,
    shared: &SearchShared,
    thread: &mut SearchThread,
    history: &[Zobrist64],
    allow_null: bool,
    cut_node: bool,
) -> i32 {
    if thread.mark_node(shared) {
        return 0;
    }

    thread.pv_len[ply] = 0;
    thread.cutoff_count[ply] = 0;

    let sp = &shared.sp;
    let gates = &sp.gates;
    let cut_node = cut_node && gates.cut_nodes;
    let pv_node = beta - alpha > 1;
    // `allNode` has no binding here on purpose: its only two consumers are the
    // modern reduction formula and the IIR gate, and both read it off the
    // `LmrSignals` snapshot (`LmrSignals::all_node`, `reductions::allows_iir`),
    // which is a pure function of `pv_node` / `cut_node` and is unit-tested
    // there. A third, unused copy at the node level could only drift from it.
    let root_node = ply == 0;
    let excluded_move = thread.excluded_move[ply];
    let excluded = excluded_move.is_some();

    // Follow the *previous iteration's* PV, not the current node's PV type.
    // The root itself is outside alphabeta in Morstilia, so the virtual root
    // is represented by the condition `ply == 1` below.
    let follow_pv = if gates.follow_pv {
        let on = if root_node {
            true
        } else if ply == 1 {
            if thread.prev_pv_len == 0 {
                false
            } else {
                matches_previous_move(thread.ctx[1], thread.prev_pv[0])
            }
        } else {
            ply - 1 < MAX_PLY
                && thread.follow_pv[ply - 1]
                && ply - 1 < thread.prev_pv_len
                && matches_previous_move(thread.ctx[ply], thread.prev_pv[ply - 1])
        };
        thread.follow_pv[ply] = on;
        on
    } else {
        thread.follow_pv[ply] = false;
        false
    };

    // Mate-distance window clamp.
    alpha = alpha.max(mated_in(ply as i32));
    let beta = beta.min(mate_in(ply as i32 + 1));
    if alpha >= beta {
        return alpha;
    }

    // Hard search-ply guard. Checkmate still wins over the leaf evaluation,
    // and the raw evaluator is used for non-mates (no correction context is
    // assumed at the final leaf).
    let in_check = pos.is_check();
    if ply >= MAX_PLY - 1 {
        if in_check && pos.is_mated() {
            return mated_in(ply as i32);
        }
        return correction::clamp_non_mate(thread.raw_evaluate_at(pos, shared, ply));
    }

    // Quiescence is the depth-zero boundary, before the main-search terminal
    // machinery just as in the previous Stage-02 implementation.
    if depth <= 0 {
        return qsearch::qsearch(pos, alpha, beta, ply, shared, thread, history);
    }

    // The 50-move rule / repetition are path properties and must beat TT data.
    if pos.halfmoves() >= 100 {
        if !pos.is_mated() {
            return 0;
        }
    } else if is_draw(pos, thread, history, ply) {
        return 0;
    }

    // ---------------------------------------------------------------------
    // TT probe and cutoff
    // ---------------------------------------------------------------------
    thread.stats.tt_probes += 1;
    let tt_entry = shared.tt.probe(pos.hash.into());
    let tt_move = tt_entry.map_or(RawMove::NULL, |e| e.mv);
    let tt_value = tt_entry.map(|e| tt_score_to_node(e.score, ply));
    if tt_entry.is_some() {
        thread.stats.tt_hits += 1;
    }

    if let Some(tt) = tt_entry {
        let score = tt_score_to_node(tt.score, ply);

        let usable = if gates.sf19_tt_cutoff {
            !pv_node
                && !excluded
                && tt.depth > depth - i32::from(score <= beta)
                && singular::is_valid(score)
                && if score >= beta {
                    singular::has_lower_bound(tt.bound)
                } else {
                    !matches!(tt.bound, Bound::Lower)
                }
                && (cut_node == (score >= beta) || depth > 4)
                && pos.halfmoves() < sp.tt_cutoff_rule50_limit as u32
        } else {
            !pv_node
                && !excluded
                && tt.depth >= depth
                && match tt.bound {
                    Bound::Exact => true,
                    Bound::Lower => score >= beta,
                    Bound::Upper => score <= alpha,
                }
        };

        if usable {
            if gates.tt_ghi_probe
                && depth >= sp.tt_ghi_min_depth
                && pos.halfmoves() < sp.tt_cutoff_rule50_limit as u32
                && tt_move != RawMove::NULL
                && !singular::is_decisive(score)
                && pos.raw_move_legal(tt_move)
            {
                let child = pos.make_child(tt_move);
                if let Some(next) = shared.tt.probe(child.hash.into()) {
                    let next_score = tt_score_to_node(next.score, ply + 1);
                    if (score >= beta) == (-next_score >= beta) {
                        thread.stats.tt_cutoffs += 1;
                        return score;
                    }
                    // Mismatch: the GHI hint is inconclusive, fall through to
                    // the ordinary TT cutoff and the rest of the search.
                } else {
                    thread.stats.tt_cutoffs += 1;
                    return score;
                }
            } else {
                thread.stats.tt_cutoffs += 1;
                return score;
            }
        }
    }

    // Small TT ProbCut idea (SF19 Step 13). This is independent from the
    // stricter TT-cutoff gate and intentionally disabled when its own gate is
    // off. Exact / decisive values are not returned as a shortcut.
    if gates.tt_probcut
        && !pv_node
        && !excluded
        && let Some(score) = tt_value
        && !singular::is_decisive(score)
        && tt_entry.is_some_and(|e| e.depth >= depth - sp.tt_probcut_depth)
        && score >= beta + sp.tt_probcut_beta
    {
        return score;
    }

    // A direct node with no legal move is settled before static evaluation.
    let mut moves = pos.legal_moves();
    if moves.is_empty() {
        return if in_check { mated_in(ply as i32) } else { 0 };
    }

    // ---------------------------------------------------------------------
    // Static evaluation / node-level pruning
    // ---------------------------------------------------------------------
    let static_eval = thread.evaluate_at(pos, shared, ply, in_check);
    thread.evals[ply] = static_eval;
    thread.hashes[ply] = pos.hash;

    let mut improving = !in_check && ply >= 2 && static_eval > thread.evals[ply - 2];

    // Evaluation-difference history bonus: evidence about the move that
    // reached this node, not this node's side to move. Promotions/captures,
    // check-forced replies and null moves do not teach the main history here.
    if gates.sf19_late_pruning && !in_check && ply > 0 {
        let entered = thread.entered[ply];
        if entered.real_move && !entered.parent_in_check && !entered.capture {
            if let Some(ctx) = thread.ctx[ply] {
                let diff = (-(thread.evals[ply - 1] + static_eval))
                    .clamp(sp.eval_diff_min, sp.eval_diff_max)
                    + sp.eval_diff_offset;
                let bonus = diff * sp.eval_diff_main_weight;
                // This move was played by the parent, i.e. by the opposite
                // colour from the current node.
                let side = !pos.turn();
                let previous = RawMove::new(ctx.from, ctx.to, RawMove::NORMAL);
                thread.tables.history.update_history(side, previous, bonus);
            }
        }
    }

    // Legacy node-level reverse futility remains the baseline mechanism.
    if !pv_node
        && !in_check
        && depth <= sp.rfp_depth
        && static_eval - pruning::reverse_futility_margin_sp(sp, depth, improving) >= beta
    {
        thread.stats.rfp_pruned += 1;
        return static_eval;
    }

    // Legacy razoring remains the baseline mechanism and uses qsearch to
    // verify the fail-low.
    if !pv_node
        && !in_check
        && depth <= sp.razor_depth
        && static_eval + pruning::razor_margin_sp(sp, depth) < alpha
    {
        thread.stats.razor_attempts += 1;
        let razor_score = qsearch::qsearch(pos, alpha, beta, ply, shared, thread, history);
        if razor_score <= alpha {
            thread.stats.razor_cutoffs += 1;
            return razor_score;
        }
    }

    // ---------------------------------------------------------------------
    // Null move: SF19 path or the pre-Phase-02 path when its gate is off.
    //
    // The margin is *subtracted* from `beta`, as `nmp_margin_base`'s
    // documentation states: the wider the margin, the further below `beta` a
    // static evaluation may be and still justify a pass. Adding the base
    // instead inverts that, and turns the gate into "static eval must exceed
    // beta", which almost never fires.
    // ---------------------------------------------------------------------
    if gates.sf19_null_move {
        if cut_node
            && !in_check
            && allow_null
            && !excluded
            && depth >= sp.null_move_min_depth
            && ply as i32 >= thread.nmp_min_ply
            && static_eval
                >= beta
                    - sp.nmp_margin_base
                    - sp.nmp_margin_depth * depth
                    - sp.nmp_margin_improving * i32::from(improving)
            && pruning::side_has_attacking_pieces(pos.board(), pos.turn())
            && beta >= sp.nmp_beta_floor
        {
            if let Some(nulled) = pos.null_move() {
                thread.stats.null_probes += 1;
                thread.ctx[ply + 1] = None;
                thread.entered[ply + 1] = crate::search::EnteredPly::NONE;
                let r = sp.nmp_reduction_base
                    + depth / sp.nmp_reduction_div
                    + ((static_eval - beta) / sp.nmp_reduction_divisor).max(0);
                let null_depth = depth - 1 - r;
                let nulled = thread.make_child_null(nulled, shared, ply + 1);
                let null_score = -alphabeta(
                    &nulled,
                    -beta,
                    -beta + 1,
                    null_depth,
                    ply + 1,
                    shared,
                    thread,
                    history,
                    false,
                    false,
                );
                if thread.stopped {
                    return 0;
                }
                if null_score >= beta && !singular::is_decisive(null_score) {
                    if thread.nmp_min_ply != 0 || depth < sp.nmp_verification_min_depth {
                        thread.stats.null_cutoffs += 1;
                        return null_score;
                    }

                    let old_min = thread.nmp_min_ply;
                    thread.nmp_min_ply = ply as i32 + 3 * null_depth.max(0) / 4;
                    let verify = alphabeta(
                        pos,
                        beta - 1,
                        beta,
                        null_depth,
                        ply,
                        shared,
                        thread,
                        history,
                        false,
                        false,
                    );
                    thread.nmp_min_ply = old_min;
                    if thread.stopped {
                        return 0;
                    }
                    if verify >= beta {
                        thread.stats.null_cutoffs += 1;
                        return null_score;
                    }
                }
            }
        }
    } else if !pv_node
        && !in_check
        && allow_null
        && !excluded
        && depth >= sp.null_move_min_depth
        && static_eval >= beta
        && pruning::side_has_attacking_pieces(pos.board(), pos.turn())
    {
        if let Some(nulled) = pos.null_move() {
            thread.stats.null_probes += 1;
            thread.ctx[ply + 1] = None;
            thread.entered[ply + 1] = crate::search::EnteredPly::NONE;
            let nulled = thread.make_child_null(nulled, shared, ply + 1);
            let null_score = -alphabeta(
                &nulled,
                -beta,
                -beta + 1,
                depth - 1 - pruning::null_move_reduction_sp(sp, depth, improving),
                ply + 1,
                shared,
                thread,
                history,
                false,
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

    // The IIR gate uses the SF19 adaptation: followPV is a separate concept
    // from pv_node, and the reduction applies when the node is PV or cut.
    if gates.iir
        && reductions::allows_iir(
            follow_pv,
            pv_node,
            cut_node,
            depth,
            tt_move == RawMove::NULL,
            sp.iir_min_depth,
        )
        && !excluded
    {
        depth -= sp.iir_reduction;
        thread.stats.iir_reductions += 1;
    }

    if depth <= 0 {
        return qsearch::qsearch(pos, alpha, beta, ply, shared, thread, history);
    }

    // The post-null improving update is part of SF19's null/IIR sequence.
    improving |= !in_check && static_eval >= beta;

    let correction_value = thread.correction_at(pos, ply);

    // ---------------------------------------------------------------------
    // SF19 capture-driven ProbCut, falling back to the old static ProbCut when
    // its gate is disabled.
    //
    // Faithful to SF19's Step 12 in three places that matter:
    //
    // * The candidate set is `MoveList<CAPTURES>` filtered by
    //   `see_ge(m, probCutBeta - ss->staticEval)`. The probe therefore only
    //   looks at captures that actually gain enough to reach the raised beta.
    // * The TT term skips a node whose stored value is already *below*
    //   `probCutBeta` — such a node cannot fail high, so probing it is waste.
    //   (The previous `tt_value >= prob_cut_beta` test had this inverted: it
    //   skipped exactly the nodes where a probe pays off.)
    // * The reduced search only runs when `probCutDepth > 0`; otherwise the
    //   quiescence value alone decides, instead of re-entering alpha-beta with
    //   a non-positive depth.
    //
    // `!in_check` matches SF19 too: its ProbCut block carries no `inCheck`
    // term because control has already jumped past it — `if (ss->inCheck)
    // goto moves_loop;` precedes Step 12, and the ProbCut move picker asserts
    // `!pos.checkers()`.
    // ---------------------------------------------------------------------
    if gates.sf19_probcut {
        let prob_beta =
            beta + sp.probcut_beta_base - sp.probcut_beta_improving * i32::from(improving);
        let prob_depth = depth
            - if improving {
                sp.probcut_probe_depth_improving
            } else {
                sp.probcut_probe_depth_stagnant
            };

        let tt_refutes = tt_value.is_some_and(|v| v < prob_beta);

        if !pv_node
            && !in_check
            && allow_null
            && depth >= sp.probcut_min_depth
            && !singular::is_decisive(beta)
            && prob_beta < crate::search::MATE_ZONE
            && !tt_refutes
            && pruning::side_has_attacking_pieces(pos.board(), pos.turn())
        {
            thread.stats.probcut_attempts += 1;
            let mut captures = pos.capture_moves();
            order_moves_ctx(
                &mut captures,
                pos,
                &thread.tables,
                tt_move,
                ply,
                thread.ctx[ply],
                thread.ctx[ply.saturating_sub(1)],
                &shared.params,
            );

            let see_threshold = prob_beta - static_eval;

            for i in 0..captures.len() {
                let m = captures.get(i);
                if excluded_move == Some(m) {
                    continue;
                }
                if see::see(pos.board(), m, &shared.params) < see_threshold {
                    continue;
                }
                let child = thread.make_child(pos, m, shared, ply + 1);
                thread.ctx[ply + 1] = MoveCtx::of(pos, m);
                thread.entered[ply + 1] = crate::search::EnteredPly {
                    parent_in_check: in_check,
                    capture: true,
                    real_move: true,
                };

                let qv = -qsearch::qsearch(
                    &child,
                    -prob_beta,
                    -prob_beta + 1,
                    ply + 1,
                    shared,
                    thread,
                    history,
                );
                if thread.stopped {
                    return 0;
                }
                if qv < prob_beta {
                    continue;
                }

                let value = if prob_depth > 0 {
                    thread.prior_reduction[ply + 1] = 0;
                    -alphabeta(
                        &child,
                        -prob_beta,
                        -prob_beta + 1,
                        prob_depth,
                        ply + 1,
                        shared,
                        thread,
                        history,
                        !cut_node,
                        !cut_node,
                    )
                } else {
                    qv
                };
                if thread.stopped {
                    return 0;
                }
                if value >= prob_beta {
                    shared.tt.store(
                        pos.hash.into(),
                        m,
                        node_score_to_tt(value, ply),
                        prob_depth + 1,
                        Bound::Lower,
                    );
                    if !singular::is_decisive(value) {
                        thread.stats.probcut_cutoffs += 1;
                        return value - (prob_beta - beta);
                    }
                }
            }
        }
    } else if !pv_node
        && !in_check
        && allow_null
        && !excluded
        && depth >= sp.probcut_depth
        && beta.abs() < crate::search::MATE_ZONE
        && static_eval + pruning::probcut_margin_sp(sp, depth) >= beta
        && pruning::side_has_attacking_pieces(pos.board(), pos.turn())
    {
        thread.stats.probcut_attempts += 1;
        let prob_score = alphabeta(
            pos,
            beta - 1,
            beta,
            depth - sp.probcut_depth,
            ply,
            shared,
            thread,
            history,
            true,
            !cut_node,
        );
        if thread.stopped {
            return 0;
        }
        if prob_score >= beta {
            thread.stats.probcut_cutoffs += 1;
            return beta;
        }
    }

    // ---------------------------------------------------------------------
    // Move ordering
    // ---------------------------------------------------------------------
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
    let mut quiets_failed = [RawMove::NULL; MAX_MOVES];
    let mut num_failed = 0usize;
    let mut caps_failed = [RawMove::NULL; MAX_MOVES];
    let mut num_caps_failed = 0usize;

    for i in 0..moves.len() {
        if thread.stopped {
            return 0;
        }

        let m = moves.get(i);
        if excluded_move == Some(m) {
            continue;
        }

        let is_tactical = is_capture_or_promotion(pos.board(), m);
        let forced = searched == 0;
        let move_number = searched + 1;

        // Cheap old pruning first. The checking-move safety test allocates a
        // child only when a prune condition would otherwise reject the move.
        let mut check_child: Option<Position> = None;
        let mut ensure_check = |thread: &mut SearchThread| {
            if check_child.is_none() {
                check_child = Some(thread.make_child(pos, m, shared, ply + 1));
            }
            check_child.as_ref().is_some_and(|child| child.is_check())
        };

        let legacy_futility = !pv_node
            && !in_check
            && !is_tactical
            && depth <= sp.futility_depth
            && searched > 0
            && static_eval + pruning::futility_margin_sp(sp, depth, improving) <= alpha;
        if legacy_futility {
            if !ensure_check(thread) {
                thread.stats.futility_pruned += 1;
                continue;
            }
        }

        let mut see_value: Option<i32> = None;
        let legacy_see = !pv_node && !in_check && is_tactical;
        if legacy_see {
            thread.stats.see_calls += 1;
            let sv = see::see(pos.board(), m, &shared.params);
            see_value = Some(sv);
            if sv < pruning::see_prune_threshold_sp(sp, depth) && !forced {
                if !ensure_check(thread) {
                    thread.stats.see_pruned += 1;
                    continue;
                }
            }
        }

        let history_score = if !is_tactical {
            thread.tables.history.history_score(pos.turn(), m)
        } else {
            0
        };

        if !pv_node
            && !in_check
            && !is_tactical
            && depth <= 6
            && searched >= sp.quiet_prune_limit as usize
            && history_score < sp.history_prune_threshold
        {
            if !ensure_check(thread) {
                thread.stats.history_pruned += 1;
                continue;
            }
        }

        // SF19 late quiet/capture pruning is intentionally additive to the old
        // Stage-02 rules. Its own gate therefore cannot alter the old baseline
        // when disabled.
        if gates.sf19_late_pruning && !in_check && searched > 0 && (!follow_pv || !pv_node) {
            if is_tactical && !pv_node {
                let lmr_depth = (depth - 1).max(1);
                let cap_hist = thread.tables.history.capture_adjustment(pos.board(), m);
                let gain = victim_value(pos.board(), m, &shared.params)
                    + m.promotion().map_or(0, |r| {
                        shared.params.piece_value(r)
                            - shared.params.piece_value(shakmaty::Role::Pawn)
                    });
                let cap_futility = depth < sp.capture_futility_max_depth
                    && static_eval
                        + sp.capture_futility_base
                        + sp.capture_futility_depth * lmr_depth
                        + sp.capture_futility_history * cap_hist / sp.see_capture_history_div
                        + gain
                        <= alpha;
                if cap_futility && !ensure_check(thread) {
                    thread.stats.futility_pruned += 1;
                    continue;
                }

                // The cache only ever serves the legacy SEE prune above (which
                // runs on the same move, earlier in this iteration). When that
                // prune did not run — a PV node or an in-check node — the value
                // is computed here and consumed immediately, so it is not worth
                // caching.
                let sv = if let Some(v) = see_value {
                    v
                } else {
                    thread.stats.see_calls += 1;
                    see::see(pos.board(), m, &shared.params)
                };
                let see_margin = sp.see_capture_base * lmr_depth
                    + sp.see_capture_history * cap_hist / sp.see_capture_history_div;
                if sv < -see_margin && !forced && !ensure_check(thread) {
                    thread.stats.see_pruned += 1;
                    continue;
                }
            } else {
                let quiet_threshold_div = if improving {
                    sp.late_quiet_div_improving
                } else {
                    sp.late_quiet_div
                };
                let late_limit = (sp.late_quiet_base + depth * depth) / quiet_threshold_div.max(1);
                let cont = pos
                    .board()
                    .piece_at(m.from())
                    .map(|piece| {
                        thread.tables.history.continuation_score(
                            pos.turn(),
                            piece.role as usize,
                            m.to(),
                            prev,
                            ant,
                        )
                    })
                    .unwrap_or(0);
                let cont_score =
                    cont + sp.cont_history_main * history_score / sp.cont_history_main_div.max(1);

                if searched >= 1
                    && cont_score < sp.cont_history_prune * (depth - 1).max(1)
                    && !ensure_check(thread)
                {
                    thread.stats.history_pruned += 1;
                    continue;
                }

                let quiet_futility = depth - 1 < sp.quiet_futility_max_depth
                    && static_eval
                        + sp.quiet_futility_depth * (depth - 1).max(1)
                        + if static_eval > alpha {
                            sp.quiet_futility_alpha
                        } else {
                            0
                        }
                        + sp.quiet_futility_base
                        <= alpha;
                if quiet_futility && !ensure_check(thread) {
                    thread.stats.futility_pruned += 1;
                    continue;
                }

                if move_number >= late_limit.max(1) as usize
                    && (!follow_pv || !pv_node)
                    && !ensure_check(thread)
                {
                    thread.stats.history_pruned += 1;
                    continue;
                }

                if depth > 1 {
                    let sv = {
                        thread.stats.see_calls += 1;
                        see::see(pos.board(), m, &shared.params)
                    };
                    let threshold = sp.see_quiet * (depth - 1).max(1) * (depth - 1).max(1);
                    if sv < -threshold && !ensure_check(thread) {
                        thread.stats.see_pruned += 1;
                        continue;
                    }
                }
            }
        }

        let child = check_child
            .take()
            .unwrap_or_else(|| thread.make_child(pos, m, shared, ply + 1));
        let child_in_check = child.is_check();
        thread.ctx[ply + 1] = MoveCtx::of(pos, m);
        thread.entered[ply + 1] = crate::search::EnteredPly {
            parent_in_check: in_check,
            capture: is_tactical,
            real_move: true,
        };

        let check_extension = i32::from(child_in_check);
        let child_depth = depth - 1 + check_extension;
        let mut extension = 0;

        // -----------------------------------------------------------------
        // Singular extension chain
        // -----------------------------------------------------------------
        if gates.singular && m == tt_move {
            let tt = tt_entry;
            if let Some(tt) = tt {
                let seek_mate = thread.root_depth >= 16 && thread.root_score.abs() >= 2000;
                let sctx = SingularCtx {
                    sp,
                    root_node,
                    pv_node,
                    cut_node,
                    tt_pv: pv_node,
                    in_check,
                    seek_mate,
                    ply_beyond_root: ply as Depth > thread.root_depth,
                    tt_move,
                    is_tt_candidate: m == tt_move,
                    tt_value: tt_score_to_node(tt.score, ply),
                    tt_depth: tt.depth,
                    tt_bound: tt.bound,
                    tt_capture: is_capture_or_promotion(pos.board(), tt_move),
                    depth,
                    new_depth: child_depth,
                    alpha,
                    beta,
                    static_eval,
                    correction_value,
                    tt_move_history: thread.tt_move_history,
                    shuffling: thread.is_shuffling(pos, m, ply),
                    excluded_move: excluded,
                };

                if singular::singular_candidate(&sctx) {
                    thread.stats.singular_candidates += 1;
                    thread.stats.singular_tests += 1;

                    let singular_beta = singular::singular_beta(&sctx);
                    let singular_depth = singular::singular_depth(&sctx);
                    let old_excluded = thread.excluded_move[ply];
                    thread.excluded_move[ply] = Some(m);
                    let value = alphabeta(
                        pos,
                        singular_beta - 1,
                        singular_beta,
                        singular_depth,
                        ply,
                        shared,
                        thread,
                        history,
                        true,
                        cut_node,
                    );
                    thread.excluded_move[ply] = old_excluded;

                    if thread.stopped {
                        return 0;
                    }

                    match singular::classify(&sctx, value) {
                        SingularOutcome::Extend { extension: ext, .. } => {
                            thread.stats.singular_successes += 1;
                            thread.stats.singular_extensions += 1;
                            match ext {
                                1 => thread.stats.singular_ext_1 += 1,
                                2 => thread.stats.singular_ext_2 += 1,
                                _ => thread.stats.singular_ext_3 += 1,
                            }
                            extension = ext;
                        }
                        SingularOutcome::MultiCut {
                            value,
                            correction_bonus,
                            ttmh_shift,
                        } => {
                            thread.stats.singular_multicut += 1;
                            thread.stats.singular_failures += 1;
                            if correction_bonus != 0 {
                                let ctx = correction::CorrectionCtx::from_stack(&thread.ctx, ply);
                                thread.corrections.update(pos, ctx, correction_bonus);
                            }
                            if gates.tt_move_history {
                                thread.shift_tt_move_history(ttmh_shift, sp.ttmh_limit);
                            }
                            return value;
                        }
                        SingularOutcome::NegativeExt { extension: ext } => {
                            thread.stats.singular_neg_extensions += 1;
                            thread.stats.singular_failures += 1;
                            extension = ext;
                        }
                        SingularOutcome::None => {
                            thread.stats.singular_failures += 1;
                        }
                    }
                }
            }
        }

        let new_depth = child_depth + extension;
        // The full-width child depth, unclamped: at `new_depth <= 0` the child
        // is handed to quiescence, which is the only way the main search ever
        // reaches its horizon. A floor of 1 here would make the depth counter
        // non-decreasing (1 - 1 + 0 -> max(1, 0) = 1), so the search would
        // never return to quiescence and would only ever be bounded by the TT,
        // repetition detection and `MAX_PLY`.
        let mut search_depth = new_depth;
        let mut reduced = false;

        if reductions::allows_lmr(
            child_depth,
            depth,
            is_tactical,
            in_check,
            move_number,
            pv_node,
        ) {
            if gates.modern_lmr {
                let stat_score = reductions::lmr_stat_score(
                    sp,
                    pos,
                    m,
                    &thread.tables,
                    prev,
                    ant,
                    &shared.params,
                );
                let signals = LmrSignals {
                    depth,
                    move_number,
                    new_depth,
                    pv_node,
                    cut_node,
                    improving,
                    follow_pv,
                    no_tt_move: tt_move == RawMove::NULL,
                    tt_capture: tt_move != RawMove::NULL
                        && is_capture_or_promotion(pos.board(), tt_move),
                    tt_value_over_alpha: tt_value.is_some_and(|v| v > alpha),
                    tt_depth_sufficient: tt_entry.is_some_and(|e| e.depth >= depth),
                    child_cutoffs: thread.cutoff_count[ply + 1],
                    is_tt_move: m == tt_move,
                    capture: is_tactical,
                    stat_score,
                    alpha,
                    static_eval,
                    correction_value,
                    delta: beta - alpha,
                    root_delta: thread.root_delta,
                };
                let r = reductions::lmr_reduction_modern(sp, &signals);
                search_depth = reductions::lmr_search_depth(sp, &signals, r);
                reduced = search_depth < new_depth;
            } else {
                let history_score = if is_tactical {
                    0
                } else {
                    thread.tables.history.history_score(pos.turn(), m)
                };
                let r = reductions::lmr_reduction(
                    depth,
                    move_number,
                    pv_node,
                    improving,
                    history_score,
                );
                search_depth = (new_depth - r).max(1);
                reduced = search_depth < new_depth;
            }
        }

        if reduced {
            thread.stats.lmr_reduced += 1;
        }

        // SF19 child-node pruning is evaluated in the parent after make_move.
        // It is additive and only active through its dedicated gates.
        let mut child_pruned_score = None;
        if gates.sf19_razor || gates.sf19_rfp {
            let child_tt = if !pv_node && !child_in_check {
                let e = shared.tt.probe(child.hash.into());
                thread.stats.tt_probes += 1;
                if e.is_some() {
                    thread.stats.tt_hits += 1;
                }
                e
            } else {
                None
            };

            if !pv_node && !child_in_check {
                let child_eval = thread.evaluate_at(&child, shared, ply + 1, false);
                thread.evals[ply + 1] = child_eval;

                if gates.sf19_razor
                    && child_eval < alpha - pruning::sf19_child_razor_margin(sp, new_depth.max(1))
                {
                    let qv =
                        -qsearch::qsearch(&child, -beta, -alpha, ply + 1, shared, thread, history);
                    if thread.stopped {
                        return 0;
                    }
                    child_pruned_score = Some(qv);
                }

                if child_pruned_score.is_none()
                    && gates.sf19_rfp
                    && !excluded
                    && new_depth
                        < if thread.root_depth >= 16 && thread.root_score.abs() >= 2000 {
                            sp.rfp_child_seek_mate_depth
                        } else {
                            sp.child_rfp_max_depth
                        }
                {
                    let prior = thread.prior_reduction[ply + 1];
                    let opponent_worsening = gates.lmr_hindsight
                        && (prior >= sp.lmr_hindsight_min_reduction
                            || (prior >= sp.lmr_hindsight_min_reduction_2
                                && child_eval + static_eval > sp.lmr_hindsight_depth));
                    let tt_hit = child_tt.is_some();
                    let margin = pruning::sf19_child_rfp_margin(
                        sp,
                        new_depth.max(1),
                        child_eval > static_eval,
                        opponent_worsening,
                        tt_hit,
                        thread.correction_at(&child, ply + 1),
                    );
                    let child_beta = if searched == 0 { -alpha } else { -alpha };
                    if child_eval - margin >= child_beta {
                        child_pruned_score =
                            Some(-pruning::sf19_child_rfp_value(sp, child_beta, child_eval));
                    }
                }
            }
        }

        let score = if let Some(v) = child_pruned_score {
            v
        } else if searched == 0 {
            thread.prior_reduction[ply + 1] = 0;
            -alphabeta(
                &child,
                -beta,
                -alpha,
                new_depth,
                ply + 1,
                shared,
                thread,
                history,
                true,
                false,
            )
        } else {
            thread.prior_reduction[ply + 1] = if reduced {
                (new_depth - search_depth).max(0)
            } else {
                0
            };
            let mut s = -alphabeta(
                &child,
                -alpha - 1,
                -alpha,
                search_depth,
                ply + 1,
                shared,
                thread,
                history,
                true,
                true,
            );
            if thread.stopped {
                return 0;
            }

            if s > alpha && reduced {
                thread.stats.lmr_researched += 1;
                thread.prior_reduction[ply + 1] = 0;
                s = -alphabeta(
                    &child,
                    -alpha - 1,
                    -alpha,
                    new_depth,
                    ply + 1,
                    shared,
                    thread,
                    history,
                    true,
                    false,
                );
            }
            if pv_node && s > alpha && s < beta {
                thread.prior_reduction[ply + 1] = 0;
                s = -alphabeta(
                    &child,
                    -beta,
                    -alpha,
                    new_depth,
                    ply + 1,
                    shared,
                    thread,
                    history,
                    true,
                    false,
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
                if num_caps_failed < MAX_MOVES {
                    caps_failed[num_caps_failed] = m;
                    num_caps_failed += 1;
                }
            } else if num_failed < MAX_MOVES {
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
                        if fm == RawMove::NULL {
                            continue;
                        }
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
                    thread.cutoff_count[ply] += 1;
                    break;
                }
            }
        }
    }

    // If every move was pruned, return a real fail-low value rather than the
    // sentinel. In ordinary play the first move is forced to survive; this is
    // the explicit guard that keeps the invariant true even if future pruning
    // changes make an all-pruned node reachable.
    if best_move == RawMove::NULL {
        best = alpha;
    }

    if !thread.stopped {
        let bonus = correction::learn_bonus(
            in_check,
            best_move != RawMove::NULL && is_capture_or_promotion(pos.board(), best_move),
            best_move != RawMove::NULL,
            best,
            static_eval,
            depth,
        );
        if bonus != 0 {
            let ctx = correction::CorrectionCtx::from_stack(&thread.ctx, ply);
            thread.corrections.update(pos, ctx, bonus);
        }

        if gates.tt_move_history && !pv_node && !excluded && tt_move != RawMove::NULL {
            let hit = best_move == tt_move;
            let shift = if hit {
                sp.ttmh_hit_bonus
            } else {
                -sp.ttmh_miss_penalty
            };
            thread.shift_tt_move_history(shift, sp.ttmh_limit);
        }

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

#[inline]
fn matches_previous_move(ctx: Option<MoveCtx>, want: RawMove) -> bool {
    if want == RawMove::NULL {
        return false;
    }
    match ctx {
        Some(c) => c.from == want.from() && c.to == want.to(),
        None => false,
    }
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

#[inline]
fn history_bonus(depth: Depth) -> i32 {
    crate::move_ordering::history::bonus(depth)
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

    /// Searches `fen` at `depth` on a shared searcher (warm TT between calls).
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

    /// A minimal `SearchShared` with its own table, for single-node tests.
    fn shared() -> SearchShared {
        SearchShared {
            tt: Arc::new(crate::tt::TranspositionTable::new(1)),
            params: Arc::new(crate::evaluation::EvalParams::default()),
            sp: Arc::new(crate::search::params::SearchParams::default()),
            stop: Arc::new(AtomicBool::new(false)),
            nodes: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            node_cap: None,
            tb: Arc::new(crate::endgame::Syzygy::none()),
            nnue: None,
        }
    }

    #[test]
    fn finds_forced_mate_in_two() {
        // 1.Rd8+! Rxd8 2.Rxd8#
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
    fn repetition_outranks_a_winning_tt_entry() {
        let pos = Position::from_fen("6k1/8/8/8/8/8/5Q2/6K1 b - - 0 1").unwrap();

        let shared = shared();

        shared
            .tt
            .store(pos.hash.into(), RawMove::NULL, 500, 20, Bound::Exact);

        let mut thread = SearchThread::new();
        let history = [pos.hash, pos.hash];

        let score = alphabeta(
            &pos,
            -1,
            0,
            4,
            1,
            &shared,
            &mut thread,
            &history,
            true,
            false,
        );

        assert_eq!(score, 0, "a repeated position is a draw, not the TT score");
    }

    #[test]
    fn wins_hanging_queen() {
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

    #[test]
    fn aspiration_research_keeps_scores_stable() {
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
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

        let pos = Position::from_fen(fen).unwrap();
        let mut searcher = crate::search::Searcher::new(4);
        let stop = AtomicBool::new(true);

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

        assert_eq!(stores, r.tt_stores);
        assert!(
            stores <= 1024 && stores <= r.nodes,
            "only the pre-notice sampling window may store: {stores} stores, {} nodes",
            r.nodes
        );
        assert!(
            r.stats.tt_probes <= 1024,
            "a pre-aborted search may probe at most one sampling window"
        );

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
        let fen = "1r4k1/5ppp/8/8/8/8/3R1PPP/3R3K w - - 0 1";

        let mut searcher = crate::search::Searcher::new(4);

        let deep = search_on(&mut searcher, fen, 4);

        assert!(
            crate::types::is_mate(deep.score) && crate::types::mate_plies(deep.score) <= 3,
            "case must be mate in <= 3 plies: got {}",
            deep.score
        );

        let shallow = search_on(&mut searcher, fen, 2);

        assert!(
            crate::types::is_mate(shallow.score) && crate::types::mate_plies(shallow.score) <= 3,
            "TT mate score must survive re-basing: got {}",
            shallow.score
        );
    }

    // --- Modern pruning & reductions ---------------------------------------

    #[test]
    fn null_move_never_fires_in_pawn_endgames() {
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
        // Black's queen on f2 hangs to the white queen, and the black knight on
        // f3 gives the tree the captures that SF19's capture-driven ProbCut
        // needs: with bare `KQ vs K` after Qxf2 the position has no capture at
        // all, so a faithful ProbCut can never cut there.
        let fen = "6k1/8/8/8/8/5n2/5q2/5Q1K w - - 0 1";
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
        let r = search_depth("6k1/8/8/8/8/8/5q2/5Q1K w - - 0 1", 4);

        assert!(
            r.stats.rfp_pruned > 0,
            "overwhelming static advantage must trigger reverse futility"
        );
    }

    #[test]
    fn pruning_stack_does_not_bias_aspiration_stability() {
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

    // --- Stage 02 correctness ------------------------------------------------

    #[test]
    fn fifty_move_draw_beats_a_winning_tt_entry() {
        let fen = "6k1/8/8/8/8/8/5Q2/6K1 b - - 100 60";
        let pos = Position::from_fen(fen).unwrap();

        assert_eq!(pos.halfmoves(), 100);
        assert!(!pos.is_mated());

        let shared = shared();

        shared
            .tt
            .store(pos.hash.into(), RawMove::NULL, 500, 40, Bound::Exact);

        let mut thread = SearchThread::new();
        let history = [pos.hash];

        let score = alphabeta(
            &pos,
            0,
            1,
            4,
            0,
            &shared,
            &mut thread,
            &history,
            true,
            false,
        );

        assert_eq!(score, 0, "a fifty-move draw is a draw, not the TT score");
    }

    #[test]
    fn checkmate_on_the_hundredth_halfmove_is_not_a_draw() {
        let fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 100 80";
        let pos = Position::from_fen(fen).unwrap();

        assert_eq!(pos.halfmoves(), 100);
        assert!(pos.is_check());
        assert!(pos.is_mated());

        let shared = shared();

        shared
            .tt
            .store(pos.hash.into(), RawMove::NULL, 0, 40, Bound::Exact);

        let mut thread = SearchThread::new();
        let history = [pos.hash];

        let score = alphabeta(
            &pos,
            -1000,
            1000,
            4,
            0,
            &shared,
            &mut thread,
            &history,
            true,
            false,
        );

        assert!(
            score < -crate::search::MATE_ZONE,
            "checkmate must outrank the fifty-move draw, got {}",
            score
        );
    }

    #[test]
    fn max_ply_leaf_reports_checkmate() {
        let pos = Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();

        let shared = shared();
        let mut thread = SearchThread::new();
        let history = [pos.hash];
        let ply = MAX_PLY - 1;

        let score = alphabeta(
            &pos,
            -1000,
            1000,
            1,
            ply,
            &shared,
            &mut thread,
            &history,
            true,
            false,
        );

        assert_eq!(
            score,
            mated_in(ply as i32),
            "the leaf must report mate at exactly the leaf's distance"
        );
    }

    #[test]
    fn max_ply_leaf_is_an_uncorrected_non_mate_evaluation() {
        let pos = Position::from_fen("6k1/8/8/8/8/8/4Q3/6K1 w - - 0 1").unwrap();

        let shared = shared();
        let mut thread = SearchThread::new();
        let history = [pos.hash];
        let ply = MAX_PLY - 1;

        let uncorrected = alphabeta(
            &pos,
            -1000,
            1000,
            1,
            ply,
            &shared,
            &mut thread,
            &history,
            true,
            false,
        );

        let raw = correction::clamp_non_mate(thread.raw_evaluate_at(&pos, &shared, ply));

        assert_eq!(
            uncorrected, raw,
            "the leaf must be the raw evaluation, not a corrected one"
        );
        assert!(
            uncorrected.abs() < crate::search::MATE_ZONE,
            "a static evaluation is never in the mate zone, got {}",
            uncorrected
        );
    }

    #[test]
    fn in_check_nodes_never_reduce() {
        let fen = "8/4R3/8/8/4k3/8/8/K7 b - - 0 1";
        let pos = Position::from_fen(fen).unwrap();

        assert!(pos.is_check(), "the root must be a check node");

        let evasions = pos.legal_moves().len();
        assert_eq!(
            evasions, 6,
            "the fixture must have exactly six legal quiet evasions"
        );

        let shared = shared();
        let mut thread = SearchThread::new();
        let history = [pos.hash];

        let score = alphabeta(
            &pos,
            -1000,
            1000,
            3,
            0,
            &shared,
            &mut thread,
            &history,
            true,
            false,
        );

        assert!(
            score > -INFINITE && score < INFINITE,
            "search must return a finite score: {score}"
        );

        assert_eq!(
            thread.stats.lmr_reduced, 0,
            "an in-check node must never perform LMR, got {} reductions across {} evasions",
            thread.stats.lmr_reduced, evasions
        );

        // `thread.nodes` is the worker's exact per-node count; the *shared*
        // counter is only committed in 1024-node batches (see
        // `SearchThread::mark_node`), so it reads 0 for any search shorter than
        // a batch and cannot answer "did we descend?" at this size.
        assert!(
            thread.nodes > 1,
            "the search must actually descend past the root, got {} nodes",
            thread.nodes
        );
    }

    #[test]
    fn a_search_never_returns_the_negative_infinite_sentinel() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "6k1/8/8/8/8/8/5Q2/6K1 b - - 0 1",
            "8/2k5/8/8/8/8/5PPP/6K1 w - - 0 1",
        ];

        for fen in fens {
            for depth in 1..=6 {
                for (alpha, beta) in [(-1, 0), (0, 1), (-100, 100), (-2000, 2000)] {
                    let pos = Position::from_fen(fen).unwrap();
                    let shared = shared();
                    let mut thread = SearchThread::new();
                    let history = [pos.hash];

                    let score = alphabeta(
                        &pos,
                        alpha,
                        beta,
                        depth,
                        0,
                        &shared,
                        &mut thread,
                        &history,
                        true,
                        false,
                    );

                    assert_ne!(
                        score, -INFINITE,
                        "{} d{} window [{},{}] returned the sentinel",
                        fen, depth, alpha, beta
                    );

                    if let Some(e) = shared.tt.probe(pos.hash.into()) {
                        let stored = tt_score_to_node(e.score, 0);

                        assert_ne!(stored, -INFINITE, "{} d{} stored the sentinel", fen, depth);
                    }
                }
            }
        }
    }
}

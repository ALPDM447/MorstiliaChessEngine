//! Singular extensions, multi-cut pruning and negative extensions: the pure
//! decision layer.
//!
//! Stockfish 19 (`sf_19/src/search.cpp:1246-1303`) runs **one** excluded-move
//! verification search and then a single, strictly ordered, mutually exclusive
//! three-arm chain:
//!
//! ```text
//! if (value < singularBeta)          -> positive singular extension  (1/2/3 plies)
//! else if (value >= beta && !is_decisive(value)) -> multi-cut: return `value`
//! else if (ttValue >= beta || cutNode)         -> negative extension (-3)
//! ```
//!
//! A move can take **at most one** of these three outcomes: the chain is an
//! `if / else if / else if`, and this module models it as a
//! [`SingularOutcome`] enum precisely so the mutual exclusion is a type-level
//! property rather than a convention.
//!
//! This module is deliberately free of engine state — it takes a
//! [`SingularCtx`] snapshot and returns a decision. The *control flow* that
//! consumes the decision (probe placement, the `excluded_move` set/restore, the
//! correction-history update and the TT store) stays in
//! [`crate::search::alphabeta`], so nothing here can hide the chain.
//!
//! # Provenance
//!
//! Every formula below is a transcription of a `sf_19` literal. Where
//! Morstilia cannot represent a Stockfish concept exactly, the deviation is
//! named in the doc comment on the item that needs it.

use crate::endgame::{TB_CURSED, TB_WIN};
use crate::search::MATE_ZONE;
use crate::search::params::SearchParams;
use crate::tt::Bound;
use crate::types::{Depth, RawMove};

/// The tablebase-win band: `TB_WIN - 1` (19 999) is the strongest score
/// [`crate::endgame::wdl_score`] can return for a won position.
///
/// Stockfish's analogue is `VALUE_TB_WIN_IN_MAX_PLY`; `sf_19/src/types.h`
/// defines `is_win(v) = v >= VALUE_TB_WIN_IN_MAX_PLY`, so the two bands a
/// "decisive" score can live in are the tablebase band and the mate band.
pub const TB_WIN_IN_MAX_PLY: i32 = TB_WIN - 1;

/// A score is *valid* when it is a real score rather than "no value".
///
/// Stockfish's `is_valid` rejects `VALUE_NONE`, which its TT writes for a
/// provisional entry. Morstilia's 12-byte entry has no such sentinel — a probed
/// entry always carries a searched score — so this is vacuously true. It is
/// kept as a named predicate so the candidate gate reads one-to-one with
/// `search.cpp:1247` and so the reason is documented where the gate lives.
#[inline]
pub const fn is_valid(_value: i32) -> bool {
    true
}

/// A mate score for the side to move.
#[inline]
pub const fn is_win(value: i32) -> bool {
    value >= MATE_ZONE
}

/// A mate score for the side to move's opponent.
#[inline]
pub const fn is_loss(value: i32) -> bool {
    value <= -MATE_ZONE
}

/// Stockfish's `is_decisive`: a mate **or** a tablebase score.
///
/// Morstilia scores tablebase wins and cursed wins at ±19 999 / ±10 000, which
/// is *inside* the mate zone (31 872) and therefore invisible to a mate-only
/// test. Both bands are covered here, and since `TB_WIN_IN_MAX_PLY <
/// MATE_ZONE` the two terms collapse to one comparison.
///
/// No legitimate non-mate evaluation can reach this band: `clamp_non_mate`
/// bounds a static evaluation by ±24 000, and a corrected evaluation is the raw
/// evaluation plus a correction of at most ±`crate::search::correction::BONUS_LIMIT`
/// (2 048), so the realistic ceiling is a few thousand centipawns.
///
/// The multi-cut path must never return an unproven decisive value, so this is
/// the guard that enforces that.
#[inline]
pub const fn is_decisive(value: i32) -> bool {
    let v = if value < 0 { -value } else { value };
    // `TB_CURSED` is the weakest tablebase score, so this covers both the
    // honest and the cursed tablebase bands as well as mates.
    v >= MATE_ZONE || v >= TB_CURSED
}

/// Every fact the singular decision needs, snapshotted at the candidate move.
///
/// Taking a snapshot (rather than passing the engine state) is what keeps the
/// three-arm chain testable in isolation, and it also pins the values *before*
/// the verification search runs — which matters, because that search can learn
/// into the correction history and so change what `correction_value` would read
/// afterwards.
#[derive(Debug, Clone, Copy)]
pub struct SingularCtx<'a> {
    /// The active parameter set.
    pub sp: &'a SearchParams,

    // --- Node classification --------------------------------------------
    /// The root node (singular extensions never fire there).
    pub root_node: bool,
    /// Full-window node whose line the PV follows.
    pub pv_node: bool,
    /// Zero-window node expected to fail high (Stockfish's `cutNode`).
    pub cut_node: bool,
    /// Stockfish's `ss->ttPv`. **Adaptation:** Morstilia's TT entry has no
    /// `is_pv` flag, so this is the node's own `pv_node`. The visible effect
    /// is that `depth >= 6 + ttPv` becomes `>= 7` at PV nodes.
    pub tt_pv: bool,
    /// The side to move is in check.
    pub in_check: bool,
    /// `rootDepth >= 16 && |rootScore| >= 2000` — the engine is chasing a mate
    /// and refuses to let an extension shorten the search.
    pub seek_mate: bool,
    /// `ss->ply > rootDepth`.
    pub ply_beyond_root: bool,

    // --- Transposition table --------------------------------------------
    /// The stored move.
    pub tt_move: RawMove,
    /// The move currently being tested is *this* stored move
    /// (`move == ttMove`, `search.cpp:1236`).
    ///
    /// Carried as a snapshot field rather than tested at the call site so the
    /// gate below stays the single definition of what makes a singular
    /// candidate. It is a distinct question from [`SingularCtx::tt_move`]:
    /// that says the table has a move, this says the loop is on it. Without
    /// this the verification search would exclude a move the *table* never
    /// suggested while `singularBeta` was derived from the table's value for a
    /// *different* move — the exclusion and the beta would describe two
    /// different refutations.
    pub is_tt_candidate: bool,
    /// The stored score, already re-based to this ply by `tt_score_to_node`.
    pub tt_value: i32,
    /// The stored depth.
    pub tt_depth: Depth,
    /// The stored bound.
    pub tt_bound: Bound,
    /// `pos.capture_stage(ttMove)`.
    pub tt_capture: bool,

    // --- This node's own data -------------------------------------------
    /// Remaining depth.
    pub depth: Depth,
    /// `newDepth` at the point of the singular block, i.e. `depth - 1` plus any
    /// extension already applied by the check extension.
    pub new_depth: Depth,
    /// The node's current window.
    pub alpha: i32,
    pub beta: i32,
    /// The node's (corrected) static evaluation.
    pub static_eval: i32,
    /// The raw correction-history value read *before* the verification search.
    pub correction_value: i32,
    /// The worker's `ttMoveHistory` scalar.
    pub tt_move_history: i32,
    /// `is_shuffling(candidate, ss, pos)`.
    pub shuffling: bool,
    /// A move is already excluded at this node.
    pub excluded_move: bool,
}

/// The transposition-table bound test that stands in for
/// `ttData.bound & BOUND_LOWER`.
///
/// Stockfish encodes bounds as **flags**: `BOUND_NONE = 0`, `BOUND_UPPER = 1`,
/// `BOUND_LOWER = 2`, and crucially
/// `BOUND_EXACT = BOUND_UPPER | BOUND_LOWER = 3` (`sf_19/src/types.h:143-144`),
/// so `bound & BOUND_LOWER` is true for an exact entry *and* for a lower bound,
/// and false only for an upper bound.
///
/// Morstilia's `Bound` (`src/tt/table.rs`) is `#[repr(u8)] { Exact = 0, Lower = 1,
/// Upper = 2 }`, packed as a single discriminant at bits 40..48 with no bit
/// flags anywhere (`From<u8>` maps `1 -> Lower`, `2 -> Upper`, everything else
/// to `Exact`). `Bound::Exact` therefore has *no* Lower bit to test.
///
/// The semantic port is exact rather than approximate: an exact entry asserts
/// `score == best`, which is in particular `score >= beta` — the same claim the
/// Lower flag makes. So the correct adaptation is "everything except an upper
/// bound", which is also precisely the predicate the existing cutoff at
/// `alphabeta.rs` uses (`Bound::Lower if score >= beta`).
#[inline]
pub const fn has_lower_bound(bound: Bound) -> bool {
    !matches!(bound, Bound::Upper)
}

/// Whether the current move is a singular candidate (`search.cpp:1234-1248`).
///
/// Every term is the `sf_19` condition, in Stockfish's order — the
/// `move == ttMove` test first (`search.cpp:1236`), then
/// `depth >= 6 + ttPv && !excludedMove` (`:1246`), then
/// `is_valid(ttValue) && !isWin(ttValue) && (ttValue >= beta || cutNode)` from
/// the branch arms (`:1255`, `:1290`). The bound test is
/// [`has_lower_bound`], and [`is_valid`] is vacuous here (see its docs).
///
/// The two branch preconditions are deliberately *not* folded in here: they
/// select between the three arms, and [`classify`] is what reads them. This
/// function answers only "is the verification search worth running at all".
#[inline]
pub fn singular_candidate(ctx: &SingularCtx<'_>) -> bool {
    ctx.is_tt_candidate
        && !ctx.root_node
        && ctx.tt_move != RawMove::NULL
        && !ctx.excluded_move
        && ctx.depth >= ctx.sp.singular_min_depth + i32::from(ctx.tt_pv)
        && is_valid(ctx.tt_value)
        && !is_decisive(ctx.tt_value)
        && has_lower_bound(ctx.tt_bound)
        && ctx.tt_depth >= ctx.depth - ctx.sp.singular_depth_margin
        && !ctx.shuffling
        && !ctx.seek_mate
}

/// The verification search's beta (`search.cpp:1250`).
///
/// `ttValue - (59 + 66 * (ttPv && !PvNode)) * depth / 63`.
///
/// Note there is **no** cut-node term in `sf_19`; the `+ 78 * depth / 63` that
/// older Stockfish releases applied to cut nodes is not present and is not
/// ported. The TT-PV factor is large on purpose: a node the table already
/// believes is a PV node, searched with a non-PV window, needs a much closer
/// beta before the remaining moves are judged "singular".
#[inline]
pub fn singular_beta(ctx: &SingularCtx<'_>) -> i32 {
    let sp = ctx.sp;
    let pv_factor = i32::from(ctx.tt_pv && !ctx.pv_node);
    ctx.tt_value
        - (sp.singular_beta_base + sp.singular_beta_pv_factor * pv_factor) * ctx.depth
            / sp.singular_beta_depth_div
}

/// The depth of the verification search (`search.cpp:1251`).
///
/// `newDepth / 2`. In `sf_19` the `newDepth` at this point is exactly
/// `depth - 1`, because `sf_19` grants no check extension. Morstilia's
/// `newDepth` also carries the check extension, so the probe can be one ply
/// deeper when the candidate move gives check.
#[inline]
pub fn singular_depth(ctx: &SingularCtx<'_>) -> Depth {
    ctx.new_depth / ctx.sp.singular_depth_div
}

/// The window the verification search runs in: `[singularBeta - 1,
/// singularBeta]`, negated into the child's point of view by the caller.
#[inline]
pub fn verification_window(ctx: &SingularCtx<'_>) -> (i32, i32) {
    let b = singular_beta(ctx);
    (b - 1, b)
}

/// The margin below which the singular extension is worth two plies
/// (`search.cpp:1260-1261`).
///
/// ```text
/// doubleMargin = -2 + 204 * PvNode - 152 * !ttCapture - correctionAdjustment
///              - 1175 * ttMoveHistory / 114178 - (ply > rootDepth) * 38
/// ```
///
/// The comment "generally, higher singularBeta and *lower* extension margins
/// scale well" in `sf_19` explains the shape: a *small* margin makes it easier
/// to qualify for the deeper extension, so a genuinely singular move is not
/// settled with a single extra ply.
#[inline]
pub fn double_margin(ctx: &SingularCtx<'_>) -> i32 {
    let sp = ctx.sp;
    let corr = correction_adjustment(ctx);
    let not_capture = i32::from(!ctx.tt_capture);
    sp.singular_double_base + sp.singular_double_pv * i32::from(ctx.pv_node)
        - sp.singular_double_not_capture * not_capture
        - corr
        - sp.singular_ttmh_num * ctx.tt_move_history / sp.singular_ttmh_div
        - sp.singular_double_beyond_root * i32::from(ctx.ply_beyond_root)
}

/// The margin below which the singular extension is worth three plies
/// (`search.cpp:1262-1263`).
///
/// ```text
/// tripleMargin = 70 + 279 * PvNode - 188 * !ttCapture + 81 * ttPv
///              - correctionAdjustment - (ply > rootDepth) * 43
/// ```
#[inline]
pub fn triple_margin(ctx: &SingularCtx<'_>) -> i32 {
    let sp = ctx.sp;
    let corr = correction_adjustment(ctx);
    let not_capture = i32::from(!ctx.tt_capture);
    sp.singular_triple_base + sp.singular_triple_pv * i32::from(ctx.pv_node)
        - sp.singular_triple_not_capture * not_capture
        + sp.singular_triple_tt_pv * i32::from(ctx.tt_pv)
        - corr
        - sp.singular_triple_beyond_root * i32::from(ctx.ply_beyond_root)
}

/// `|correctionValue| / 198368` (`search.cpp:1259`).
///
/// Morstilia's raw correction uses the same `CV_TO_CP = 131_072` scale
/// Stockfish applies in `to_corrected_static_eval` (`search.cpp:105-107`), so
/// the ratio is transferable unchanged.
#[inline]
pub fn correction_adjustment(ctx: &SingularCtx<'_>) -> i32 {
    ctx.correction_value.abs() / ctx.sp.singular_correction_div
}

/// What the single verification search proved.
///
/// The three variants are the three arms of the chain and are mutually
/// exclusive by construction: [`classify`] returns exactly one, and the
/// caller `match`es on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingularOutcome {
    /// Arm 1 — the candidate really is the only move that fails high. Extend
    /// it by `1..=3` plies and add the extra ply to the node's depth.
    Extend {
        /// `1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin)`.
        extension: Depth,
        /// The verification score, kept for the caller's statistics.
        value: i32,
    },
    /// Arm 2 — the position fails high *without* the candidate move, so it is
    /// not singular at all: several moves fail high, and the subtree can be
    /// pruned by returning the verification score as a soft bound.
    MultiCut {
        /// The value to return from the node.
        value: i32,
        /// The correction-history bonus to apply first. Zero when the guard
        /// `!inCheck && value > staticEval` did not hold, or when the bonus
        /// clamped to zero.
        correction_bonus: i32,
        /// The `ttMoveHistory` shift to apply first.
        ttmh_shift: i32,
    },
    /// Arm 3 — the candidate is not singular, but the node cannot multi-cut
    /// either (the verification failed high only over `ttValue - margin`, not
    /// over `beta`). Shorten the candidate instead.
    NegativeExt {
        /// Always the parameterised negative extension (`-3` in `sf_19`).
        extension: Depth,
    },
    /// No arm applies: the verification failed high but neither decisively nor
    /// in a way that justifies shortening the candidate.
    None,
}

/// The single mutually exclusive decision chain of `search.cpp:1257-1303`.
///
/// `value` is the score the verification search returned.
pub fn classify(ctx: &SingularCtx<'_>, value: i32) -> SingularOutcome {
    let sp = ctx.sp;
    let sbeta = singular_beta(ctx);

    // Arm 1: the candidate move is the only one that reaches `singularBeta`.
    if value < sbeta {
        let d = double_margin(ctx);
        let t = triple_margin(ctx);
        let extension = 1 + i32::from(value < sbeta - d) + i32::from(value < sbeta - t);
        return SingularOutcome::Extend { extension, value };
    }

    // Arm 2: multi-cut. The whole subtree can be pruned because the position
    // fails high over the node's *own* beta even with the candidate removed.
    // `!is_decisive` is what keeps a mate or a tablebase score from being
    // returned on the strength of a reduced search.
    if value >= ctx.beta && !is_decisive(value) {
        return SingularOutcome::MultiCut {
            value,
            correction_bonus: multicut_correction_bonus(ctx, value),
            ttmh_shift: -(sp.multicut_ttmh_penalty + sp.multicut_ttmh_depth * ctx.depth),
        };
    }

    // Arm 3: negative extension. Reached when the verification failed high but
    // not over `beta`, so no multi-cut is possible; if the TT move is
    // expected to fail high, or this is a cut node, the candidate is
    // discounted in favour of the other moves.
    if ctx.tt_value >= ctx.beta || ctx.cut_node {
        return SingularOutcome::NegativeExt {
            extension: sp.singular_negative_extension,
        };
    }

    SingularOutcome::None
}

/// The correction-history update performed on the multi-cut return path
/// (`search.cpp:1281-1287`).
///
/// Guarded on `!inCheck && value > staticEval`, because a check node has no
/// evaluation of its own to correct (see
/// [`crate::search::SearchThread::evaluate_at`]) and a value at or below the
/// static evaluation is not evidence of under-evaluation. The magnitude is
/// scaled by the verification depth and clamped to
/// `±CORRECTION_HISTORY_LIMIT / 4`, which is
/// [`crate::search::correction::BONUS_LIMIT`].
///
/// This is deliberately the *opposite* sign convention from the normal
/// end-of-node update: the node failed high without its best move, which is
/// evidence that the static evaluation was too low.
#[inline]
pub fn multicut_correction_bonus(ctx: &SingularCtx<'_>, value: i32) -> i32 {
    if ctx.in_check || value <= ctx.static_eval {
        return 0;
    }
    let sp = ctx.sp;
    let raw = (value - ctx.static_eval) * singular_depth(ctx) * sp.multicut_correction_scale
        / sp.multicut_correction_div;
    raw.clamp(-sp.multicut_correction_limit, sp.multicut_correction_limit)
}

/// Stockfish's `StatsEntry::operator<<` (`sf_19/src/history.h:69-78`):
/// a damped, saturating update towards `+limit` / `-limit`.
///
/// `val = val + bonus - val * |bonus| / limit`, with `bonus` clamped to
/// `±limit`. The damping term is what keeps a small bonus from moving a large
/// magnitude proportionally, and it guarantees the result can never leave
/// `(-limit, limit)`.
#[inline]
pub fn tt_move_history_shift(current: i32, bonus: i32, limit: i32) -> i32 {
    let limit = limit.max(1);
    let bonus = bonus.clamp(-limit, limit);
    // An out-of-band `current` (e.g. after the limit was lowered) would
    // otherwise be pushed further out instead of back inside the band.
    let current = current.clamp(-limit, limit);
    current + bonus - current * bonus.abs() / limit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endgame::TbOutcome;
    use crate::endgame::wdl_score;
    use crate::search::correction::BONUS_LIMIT;
    use crate::types::{MATE, MAX_PLY};
    use shakmaty::Square;

    fn sp() -> SearchParams {
        SearchParams::default()
    }

    /// Any non-null move; the candidate gate only compares it for nullness.
    fn any_move() -> RawMove {
        RawMove::new(Square::new(0), Square::new(1), RawMove::NORMAL)
    }

    /// A non-singular-eligible baseline context, so each test changes exactly
    /// one field.
    fn ctx(sp: &SearchParams) -> SingularCtx<'_> {
        SingularCtx {
            sp,
            root_node: false,
            pv_node: false,
            cut_node: false,
            tt_pv: false,
            in_check: false,
            seek_mate: false,
            ply_beyond_root: false,
            tt_move: any_move(),
            // The baseline context is the TT move itself, which is the only
            // move the gate ever accepts.
            is_tt_candidate: true,
            tt_value: 100,
            tt_depth: 10,
            tt_bound: Bound::Lower,
            tt_capture: false,
            depth: 8,
            new_depth: 7,
            alpha: 0,
            beta: 200,
            static_eval: 100,
            correction_value: 0,
            tt_move_history: 0,
            shuffling: false,
            excluded_move: false,
        }
    }

    // --- bound semantics --------------------------------------------------

    #[test]
    fn lower_bound_test_admits_exact_like_stockfish() {
        // Stockfish: BOUND_EXACT = BOUND_UPPER | BOUND_LOWER, so
        // `bound & BOUND_LOWER` is true for Exact and Lower, false for Upper.
        assert!(has_lower_bound(Bound::Lower));
        assert!(has_lower_bound(Bound::Exact));
        assert!(!has_lower_bound(Bound::Upper));
    }

    // --- decisive ---------------------------------------------------------

    #[test]
    fn decisive_covers_mate_and_tablebase() {
        assert!(is_decisive(MATE - 1));
        assert!(is_decisive(MATE_ZONE));
        assert!(is_decisive(-MATE_ZONE));
        assert!(is_decisive(TB_WIN_IN_MAX_PLY));
        assert!(is_decisive(-TB_WIN_IN_MAX_PLY));
        assert!(is_decisive(wdl_score(TbOutcome::Win)));
        assert!(is_decisive(wdl_score(TbOutcome::Loss)));
        assert!(is_decisive(wdl_score(TbOutcome::CursedWin)));
        assert!(!is_decisive(0));
        assert!(!is_decisive(1500));
        assert!(!is_decisive(-1500));
    }

    #[test]
    fn decisive_band_is_strictly_below_any_evaluation() {
        // Every band `is_decisive` covers is far above a real evaluation, so
        // the guard can never fire on a normal score.
        assert!(TB_WIN_IN_MAX_PLY < MATE_ZONE);
        assert_eq!(MATE_ZONE, MATE - MAX_PLY as i32);
        assert_eq!(TB_WIN_IN_MAX_PLY, TB_WIN - 1);
    }

    // --- singular beta ----------------------------------------------------

    #[test]
    fn singular_beta_is_the_sf19_formula() {
        let sp = sp();
        for (tt_value, depth) in [(100, 8), (250, 12), (900, 20), (-150, 6)] {
            for (tt_pv, pv_node) in [(false, false), (true, true), (true, false), (false, true)] {
                let mut c = ctx(&sp);
                c.tt_value = tt_value;
                c.depth = depth;
                c.tt_pv = tt_pv;
                c.pv_node = pv_node;
                let want = tt_value - (59 + 66 * i32::from(tt_pv && !pv_node)) * depth / 63;
                assert_eq!(singular_beta(&c), want, "ttv={tt_value} d={depth}");
            }
        }
    }

    #[test]
    fn singular_beta_is_always_below_the_tt_value() {
        // The verification window sits *below* the TT score: the point is to
        // ask "do the other moves get anywhere near what the table says this
        // move is worth", and the answer must be a weaker bar.
        let sp = sp();
        for depth in 6..=40i32 {
            for tt_pv in [false, true] {
                let mut c = ctx(&sp);
                c.depth = depth;
                c.tt_pv = tt_pv;
                c.tt_value = 1000;
                assert!(singular_beta(&c) < c.tt_value, "depth {depth}");
            }
        }
    }

    #[test]
    fn tt_pv_non_pv_raises_the_bar() {
        let sp = sp();
        let mut a = ctx(&sp);
        let mut b = ctx(&sp);
        a.tt_pv = false;
        b.tt_pv = true;
        assert!(singular_beta(&b) < singular_beta(&a));
    }

    // --- the candidate gate ----------------------------------------------

    #[test]
    fn a_qualified_entry_is_a_candidate() {
        let sp = sp();
        assert!(singular_candidate(&ctx(&sp)));
    }

    #[test]
    fn gate_excludes_each_documented_reason() {
        let sp = sp();
        let cases: Vec<(&str, Box<dyn Fn(&mut SingularCtx<'_>)>)> = vec![
            (
                "a move the table never suggested",
                Box::new(|c: &mut SingularCtx<'_>| c.is_tt_candidate = false),
            ),
            (
                "root",
                Box::new(|c: &mut SingularCtx<'_>| c.root_node = true),
            ),
            (
                "no tt move",
                Box::new(|c: &mut SingularCtx<'_>| c.tt_move = RawMove::NULL),
            ),
            (
                "excluded",
                Box::new(|c: &mut SingularCtx<'_>| c.excluded_move = true),
            ),
            ("shallow", Box::new(|c: &mut SingularCtx<'_>| c.depth = 5)),
            (
                "decisive tt value",
                Box::new(|c: &mut SingularCtx<'_>| c.tt_value = MATE_ZONE),
            ),
            (
                "upper bound",
                Box::new(|c: &mut SingularCtx<'_>| c.tt_bound = Bound::Upper),
            ),
            (
                "shallow tt entry",
                Box::new(|c: &mut SingularCtx<'_>| c.tt_depth = 4),
            ),
            (
                "shuffling",
                Box::new(|c: &mut SingularCtx<'_>| c.shuffling = true),
            ),
            (
                "seeking mate",
                Box::new(|c: &mut SingularCtx<'_>| c.seek_mate = true),
            ),
        ];
        for (name, f) in cases {
            let mut c = ctx(&sp);
            f(&mut c);
            assert!(!singular_candidate(&c), "{name} must disqualify");
        }
    }

    /// `move == ttMove` is the *first* term of `sf_19`'s gate
    /// (`search.cpp:1236`). It is what makes the exclusion and the beta
    /// describe the same refutation: the verification search removes this move,
    /// and `singularBeta` is derived from the table's value for this move.
    #[test]
    fn only_the_table_move_is_ever_a_singular_candidate() {
        let sp = sp();
        let mut c = ctx(&sp);
        c.is_tt_candidate = false;
        assert!(!singular_candidate(&c));
        c.is_tt_candidate = true;
        assert!(singular_candidate(&c));
    }

    #[test]
    fn minimum_depth_is_six_plus_tt_pv() {
        let sp = sp();
        for tt_pv in [false, true] {
            let mut c = ctx(&sp);
            c.tt_pv = tt_pv;
            let floor = 6 + i32::from(tt_pv);
            c.depth = floor - 1;
            assert!(!singular_candidate(&c), "depth {floor} - 1 must fail");
            c.depth = floor;
            assert!(singular_candidate(&c), "depth {floor} must pass");
        }
    }

    #[test]
    fn tt_depth_margin_is_depth_minus_three() {
        let sp = sp();
        let mut c = ctx(&sp);

        c.depth = 10;
        c.tt_depth = 7;
        assert!(singular_candidate(&c));

        c.tt_depth = 6;
        assert!(!singular_candidate(&c));
    }

    // --- extension ladder -------------------------------------------------

    #[test]
    fn extension_ladder_is_one_two_or_three() {
        let sp = sp();
        let sbeta = singular_beta(&ctx(&sp));
        for v in [sbeta - 1, sbeta - 50, sbeta - 200, sbeta - 1000, 0, -5000] {
            let c = ctx(&sp);
            match classify(&c, v) {
                SingularOutcome::Extend { extension, .. } => {
                    assert!((1..=3).contains(&extension), "value {v} gave {extension}");
                }
                other => panic!("value {v} should have extended, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_barely_singular_move_gets_one_ply() {
        let sp = sp();
        let mut c = ctx(&sp);
        c.pv_node = true;

        let sbeta = singular_beta(&c);
        let v = sbeta - 1;

        match classify(&c, v) {
            SingularOutcome::Extend { extension, .. } => assert_eq!(extension, 1),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn margins_move_in_the_documented_directions() {
        let sp = sp();
        // A PV node raises both margins: a PV move has to be much better
        // before it earns the deeper extension.
        let a = ctx(&sp);
        let mut b = ctx(&sp);
        b.pv_node = true;
        assert!(double_margin(&b) > double_margin(&a));
        assert!(triple_margin(&b) > triple_margin(&a));
        // A capturing TT move raises both margins.
        let mut c = ctx(&sp);
        c.tt_capture = true;
        assert!(double_margin(&c) > double_margin(&a));
        assert!(triple_margin(&c) > triple_margin(&a));
        // A strong correction lowers both margins.
        let mut d = ctx(&sp);
        d.correction_value = 198_368 * 3;
        assert!(double_margin(&d) < double_margin(&a));
        assert!(triple_margin(&d) < triple_margin(&a));
        // A positive ttMoveHistory lowers the double margin.
        let mut e = ctx(&sp);
        e.tt_move_history = 1_000;
        assert!(double_margin(&e) < double_margin(&a));
        // ttPv raises the triple margin.
        let mut f = ctx(&sp);
        f.tt_pv = true;
        assert!(triple_margin(&f) > triple_margin(&a));
        // Past the root depth both margins shrink.
        let mut g = ctx(&sp);
        g.ply_beyond_root = true;
        assert!(double_margin(&g) < double_margin(&a));
        assert!(triple_margin(&g) < triple_margin(&a));
    }

    // --- the three-arm chain ---------------------------------------------

    #[test]
    fn arm_one_wins_when_the_verification_fails_low() {
        let sp = sp();
        let mut c = ctx(&sp);
        c.cut_node = true;
        c.tt_value = 500;
        let v = singular_beta(&c) - 1;
        assert!(matches!(classify(&c, v), SingularOutcome::Extend { .. }));
    }

    #[test]
    fn arm_two_wins_when_the_verification_fails_high_over_beta() {
        let sp = sp();
        let mut c = ctx(&sp);
        c.beta = 150;
        c.tt_value = 100;
        let v = 200; // >= beta, non-decisive
        match classify(&c, v) {
            SingularOutcome::MultiCut { value, .. } => assert_eq!(value, 200),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn arm_two_never_fires_on_a_decisive_value() {
        // The point of `!is_decisive(value)`: a mate or tablebase score
        // produced by a reduced search must not be returned as the node's
        // score.
        let sp = sp();
        let mut c = ctx(&sp);
        c.beta = 150;
        for v in [
            MATE_ZONE,
            -MATE_ZONE,
            TB_WIN_IN_MAX_PLY,
            wdl_score(TbOutcome::Win),
        ] {
            assert!(
                !matches!(classify(&c, v), SingularOutcome::MultiCut { .. }),
                "{v} must not multi-cut"
            );
        }
    }

    #[test]
    fn arm_three_wins_at_a_cut_node() {
        // A cut node with a verification that failed high over the *reduced*
        // beta only: no multi-cut, but the candidate is discounted.
        let sp = sp();
        let mut c = ctx(&sp);
        c.cut_node = true;
        c.beta = 100_000;
        c.tt_value = 100;
        let v = singular_beta(&c) + 1; // >= singularBeta, < beta
        match classify(&c, v) {
            SingularOutcome::NegativeExt { extension } => assert_eq!(extension, -3),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn arm_three_wins_when_the_tt_value_already_fails_high() {
        let sp = sp();
        let mut c = ctx(&sp);
        c.cut_node = false;
        c.beta = 500;
        c.tt_value = 500;

        let v = singular_beta(&c) + 1;

        assert!(v < c.beta);

        assert!(matches!(
            classify(&c, v),
            SingularOutcome::NegativeExt { extension: -3 }
        ));
    }

    #[test]
    fn no_arm_applies_at_an_ordinary_node() {
        // Not a cut node and the TT value does not fail high: the verification
        // succeeded (>= singularBeta) but the node learns nothing from it.
        let sp = sp();
        let c = ctx(&sp);
        let v = singular_beta(&c) + 1;
        assert_eq!(classify(&c, v), SingularOutcome::None);
    }

    #[test]
    fn the_three_arms_are_mutually_exclusive() {
        // Sweep the whole interesting space and prove that no input ever
        // reaches two arms: a value that fails low can never satisfy
        // `value >= beta` (the beta is at or above the singular beta here), and
        // a value that fails high over beta can never be below singularBeta.
        let sp = sp();
        for depth in 6..=20i32 {
            for tt_value in [-1000, 0, 100, 800, 3000] {
                for cut_node in [false, true] {
                    for tt_pv in [false, true] {
                        for beta in [tt_value - 500, tt_value, tt_value + 900] {
                            let mut c = ctx(&sp);
                            c.depth = depth;
                            c.new_depth = depth - 1;
                            c.tt_value = tt_value;
                            c.cut_node = cut_node;
                            c.tt_pv = tt_pv;
                            c.beta = beta;
                            let sbeta = singular_beta(&c);
                            for v in [sbeta - 1, sbeta, sbeta + 1, beta - 1, beta, beta + 1] {
                                let o = classify(&c, v);
                                if let SingularOutcome::Extend { .. } = o {
                                    assert!(
                                        v < sbeta,
                                        "extend at v={v} sbeta={sbeta} is not < sbeta"
                                    );
                                }
                                if let SingularOutcome::MultiCut { value, .. } = o {
                                    assert!(value >= c.beta && !is_decisive(value));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- multi-cut correction bonus ---------------------------------------

    #[test]
    fn multicut_bonus_is_guarded_and_bounded() {
        let sp = sp();
        let mut c = ctx(&sp);
        c.beta = 100;
        c.static_eval = 100;
        c.tt_value = 100;
        // value == staticEval: no evidence of under-evaluation.
        assert_eq!(multicut_correction_bonus(&c, 100), 0);
        // A check node has no evaluation of its own to correct.
        c.in_check = true;
        assert_eq!(multicut_correction_bonus(&c, 1000), 0);
        c.in_check = false;
        // Otherwise: positive, and clamped to the correction-history limit.
        let b = multicut_correction_bonus(&c, 10_000);
        assert!(b > 0);
        assert!(b <= sp.multicut_correction_limit);
        assert!(b <= BONUS_LIMIT + sp.multicut_correction_limit);
        // Monotone in the gap.
        assert!(multicut_correction_bonus(&c, 500) > multicut_correction_bonus(&c, 200));
    }

    // --- ttMoveHistory ----------------------------------------------------

    #[test]
    fn tt_move_history_shift_is_damped_and_saturating() {
        let limit = 8192;
        assert_eq!(tt_move_history_shift(0, 918, limit), 918);
        assert_eq!(tt_move_history_shift(0, -747, limit), -747);
        // Never leaves the band, for any input.
        for cur in [-9000i32, -8192, -100, 0, 100, 8192, 9000] {
            for bonus in [-9000i32, -747, -1, 0, 1, 918, 9000] {
                let v = tt_move_history_shift(cur, bonus, limit);
                assert!(
                    (-limit..=limit).contains(&v),
                    "cur {cur} bonus {bonus} gave {v}"
                );
            }
        }
        // Monotone: more bonus, higher value.
        for cur in [-4000i32, 0, 4000] {
            assert!(tt_move_history_shift(cur, 500, limit) > tt_move_history_shift(cur, 0, limit));
        }
    }

    // --- verification window ---------------------------------------------

    #[test]
    fn verification_window_is_a_zero_window_at_the_singular_beta() {
        let sp = sp();
        let c = ctx(&sp);
        let (a, b) = verification_window(&c);
        assert_eq!(b - a, 1, "the probe is a null-window search");
        assert_eq!(b, singular_beta(&c));
    }

    #[test]
    fn singular_depth_is_half_of_new_depth() {
        let sp = sp();
        for new_depth in 1..=40i32 {
            let mut c = ctx(&sp);
            c.new_depth = new_depth;
            assert_eq!(singular_depth(&c), new_depth / 2);
        }
    }
}

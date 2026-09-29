//! Tunable search parameters (singular extensions, multi-cut, negative
//! extensions, IIR, late move reductions and the pruning thresholds).
//!
//! Every constant in this module has a documented provenance: either an
//! existing Stage-02 pruning threshold (moved here **verbatim**, with a
//! bit-identical default and formula, so this is a pure refactor), or a
//! Stockfish 19 `sf_19` `search.cpp` literal.
//!
//! # Layout
//!
//! * [`SearchParams`] holds the *numeric* tunables. They are declared once in
//!   a single macro invocation, which generates the struct, `Default`,
//!   [`SearchParams::to_vec`], [`SearchParams::from_vec`] and
//!   [`SearchParams::bounds`] from that one list. The three flat-vector
//!   methods therefore cannot drift apart, and a round-trip unit test
//!   re-checks it anyway.
//! * [`SearchGates`] holds the *boolean* on/off switches for the mechanisms
//!   that did not exist before (singular extensions, multi-cut, negative
//!   extensions, IIR, modern LMR, `followPV`, the `ttMoveHistory` table, ...).
//!   Gates are addressed **by name**, never by index, and are deliberately
//!   *not* part of the flat vector: an SPSA step has no meaningful gradient
//!   across a boolean.
//!
//! # Default / baseline contract
//!
//! Two rules hold simultaneously, and they are the whole point of this file:
//!
//! 1. **No pre-existing behaviour changed by becoming configurable.** Every
//!    Stage-02 constant below has the value it had before this module
//!    existed, and its formula is untouched. `params_baseline_reproduces_legacy_pruning`
//!    in [`crate::search::pruning`] asserts exactly that.
//! 2. **Every new mechanism is explicitly gated** and its gate defaults to
//!    `true`, which is the Stockfish-19-faithful setting. Flipping the gates
//!    back to `false` reproduces the pre-Phase-02 search. That makes the
//!    A/B and SPRT infrastructure in [`crate::matchplay`] able to measure a
//!    single mechanism at a time.
//!
//! TOML round-trip follows [`crate::evaluation::params::EvalParams`]: a
//! `SearchParamsPath` UCI option or a `--search-params` CLI flag loads a file;
//! an unknown field is an error rather than a silent no-op.

use serde::{Deserialize, Serialize};

use crate::evaluation::params::ParamDef;

/// Generates the numeric half of [`SearchParams`]: the struct, its
/// `Default`, and the three flat-tuner views — all from one declaration
/// list, so they are structurally incapable of disagreeing.
macro_rules! search_tunables {
    ($(
        $(#[$m:meta])*
        $ident:ident : $ty:ty = $def:expr, $lo:expr, $hi:expr;
    )*) => {
        /// Numeric search tunables. See the module docs for the provenance
        /// rule and the default/baseline contract.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct SearchParams {
            $(
                $(#[$m])*
                pub $ident: $ty,
            )*

            /// Boolean mechanism switches. Serialized as a `[gates]` table.
            pub gates: SearchGates,
        }

        impl Default for SearchParams {
            /// The Stockfish-19-faithful baseline: Stage-02 pruning defaults
            /// preserved, every new mechanism enabled.
            fn default() -> Self {
                Self { $( $ident: $def, )* gates: SearchGates::default() }
            }
        }

        impl SearchParams {
            /// Number of numeric tunables (gates excluded).
            pub fn param_count(&self) -> usize {
                Self::bounds().len()
            }

            /// The flat vector of numeric tunables, in declaration order.
            pub fn to_vec(&self) -> Vec<f64> {
                vec![$( self.$ident as f64, )*]
            }

            /// Writes `v` (declaration order) back into this struct. Out of
            /// range values are stored as given; the tuner owns clamping, and
            /// keeping them verbatim makes a parameter sweep diagnosable.
            pub fn from_vec(&mut self, v: &[f64]) {
                let mut i = 0usize;
                $(
                    if i < v.len() {
                        self.$ident = v[i] as $ty;
                    }
                    i += 1;
                )*
                let _ = i;
            }

            /// Names and bounds of every numeric tunable, in
            /// [`SearchParams::to_vec`] order.
            pub fn bounds() -> Vec<ParamDef> {
                vec![$(
                    ParamDef {
                        name: stringify!($ident),
                        min: $lo as f64,
                        max: $hi as f64,
                    },
                )*]
            }
        }
    };
}

search_tunables! {
    // ---------------------------------------------------------------------
    // Singular extensions — sf_19 search.cpp:1246-1269
    //
    // The candidate gate, the singular beta and the 1/2/3-ply extension
    // ladder. Defaults are the `sf_19` literals.
    // ---------------------------------------------------------------------

    /// `depth >= 6 + tt_pv` (search.cpp:1246). Morstilia's TT entry has no
    /// `is_pv` flag, so `tt_pv` is the node's own `pv_node`; at a PV node
    /// this makes the effective minimum depth 7.
    singular_min_depth: i32 = 6, 2, 24;
    /// `tt_data.depth >= depth - 3` (search.cpp:1248).
    singular_depth_margin: i32 = 3, 0, 12;
    /// Divisor in `singular_beta`.
    singular_beta_depth_div: i32 = 63, 1, 255;
    /// The `59` in `ttValue - (59 + 66 * (ttPv && !PvNode)) * depth / 63`.
    singular_beta_base: i32 = 59, 0, 255;
    /// The `66`, applied when the node is a TT-PV node but not a PV node.
    singular_beta_pv_factor: i32 = 66, 0, 255;
    /// `singularDepth = newDepth / 2` uses integer division (search.cpp:1251).
    singular_depth_div: i32 = 2, 1, 8;

    /// `doubleMargin` base, `-2` (search.cpp:1260).
    singular_double_base: i32 = -2, -512, 512;
    /// `204 * PvNode` (search.cpp:1260).
    singular_double_pv: i32 = 204, 0, 1024;
    /// `-152 * !ttCapture` (search.cpp:1260).
    singular_double_not_capture: i32 = 152, 0, 1024;
    /// `correctionValue / 198368` (search.cpp:1259). Morstilia's raw
    /// correction uses the same `CV_TO_CP = 131072` scale as Stockfish, so the
    /// ratio transfers verbatim.
    singular_correction_div: i32 = 198368, 1, 4000000;
    /// `-1175 * ttMoveHistory / 114178` (search.cpp:1261).
    singular_ttmh_num: i32 = 1175, 0, 8192;
    /// Denominator of the `ttMoveHistory` term.
    singular_ttmh_div: i32 = 114178, 1, 4000000;
    /// `-(ss->ply > rootDepth) * 38` (search.cpp:1261).
    singular_double_beyond_root: i32 = 38, 0, 512;

    /// `tripleMargin` base, `70` (search.cpp:1262).
    singular_triple_base: i32 = 70, -512, 1024;
    /// `279 * PvNode` (search.cpp:1262).
    singular_triple_pv: i32 = 279, 0, 1024;
    /// `-188 * !ttCapture` (search.cpp:1262).
    singular_triple_not_capture: i32 = 188, 0, 1024;
    /// `+81 * ss->ttPv` (search.cpp:1262).
    singular_triple_tt_pv: i32 = 81, 0, 1024;
    /// `-(ss->ply > rootDepth) * 43` (search.cpp:1263).
    singular_triple_beyond_root: i32 = 43, 0, 512;

    /// `extension = -3` (search.cpp:1302). Negative by construction: a
    /// negative extension is a reduction of the candidate move.
    singular_negative_extension: i32 = -3, -8, 0;

    // ---------------------------------------------------------------------
    // Multi-cut — sf_19 search.cpp:1277-1289
    // ---------------------------------------------------------------------

    /// `ttMoveHistory << -421 - 110 * depth` (search.cpp:1279).
    multicut_ttmh_penalty: i32 = 421, 0, 8192;
    /// The `110 * depth` in the same line.
    multicut_ttmh_depth: i32 = 110, 0, 8192;
    /// `(value - staticEval) * singularDepth * 177 / 1024` (search.cpp:1284).
    multicut_correction_scale: i32 = 177, 0, 4096;
    /// Denominator of the multi-cut correction bonus.
    multicut_correction_div: i32 = 1024, 1, 65536;
    /// The clamp bound is `CORRECTION_HISTORY_LIMIT / 4`, which is
    /// [`crate::search::correction::BONUS_LIMIT`]. Exposed so a tuned file can
    /// tighten it; the default is the built-in value.
    multicut_correction_limit: i32 = 2048, 64, 8192;

    // ---------------------------------------------------------------------
    // `ttMoveHistory` — sf_19 history.h:196 (`StatsEntry<i16, 8192>`, a
    // single scalar per worker, not a table) and search.cpp:706/1575.
    // ---------------------------------------------------------------------

    /// Update limit `D` in `StatsEntry::operator<<`.
    ttmh_limit: i32 = 8192, 256, 32767;
    /// `918` for "the TT move turned out to be the best move" (search.cpp:1575).
    ttmh_hit_bonus: i32 = 918, 0, 8192;
    /// `-747` otherwise (search.cpp:1575).
    ttmh_miss_penalty: i32 = 747, 0, 8192;

    // ---------------------------------------------------------------------
    // Internal iterative reductions — sf_19 search.cpp:1051
    // ---------------------------------------------------------------------

    /// `depth >= iirMinDepth`.
    iir_min_depth: i32 = 6, 1, 32;
    /// `depth--`. Stockfish 19 removes exactly **one** ply here; the module
    /// docs of `sf_19` note "making IIR more aggressive scales poorly", so
    /// the default is 1 and this exists only so the SPSA tuner can explore
    /// the neighbourhood.
    iir_reduction: i32 = 1, 1, 8;

    // ---------------------------------------------------------------------
    // Late move reductions — sf_19 search.cpp:1157-1423
    //
    // `r` is an integer in units of [`lmr_unit`] (1024) plies. These are the
    // `sf_19` weights, transcribed.
    // ---------------------------------------------------------------------

    /// `depth--`. One unit of `r` is `1 / lmr_unit` plies.
    lmr_unit: i32 = 1024, 1, 8192;
    /// `reductions[i] = int(lmrTableScaleNum / lmrTableScaleDen * ln i)`.
    lmr_table_scale_num: i32 = 2872, 100, 16384;
    /// Denominator of the `ln` reduction table.
    lmr_table_scale_den: i32 = 128, 1, 4096;
    /// `delta * lmrRootDeltaNum / rootDelta` (search.cpp:1887).
    lmr_root_delta_num: i32 = 577, 0, 8192;
    /// `!improving * reductionScale * lmrNonImprovingNum / lmrNonImprovingDen`.
    lmr_non_improving_num: i32 = 197, 0, 4096;
    /// Denominator of the non-improving term.
    lmr_non_improving_den: i32 = 512, 1, 8192;
    /// The `+ 982` base offset (search.cpp:1887).
    lmr_base: i32 = 982, -4096, 8192;
    /// `r += lmrTtPvBase * ttPv` (search.cpp:1162).
    lmr_tt_pv: i32 = 929, -4096, 8192;
    /// The leading `3023` of the ttPv adjustment, which is *subtracted*
    /// (search.cpp:1317).
    lmr_ttpv_adjust_base: i32 = 3023, -8192, 16384;
    /// `+ PvNode * 1004` (search.cpp:1317).
    lmr_ttpv_adjust_pv: i32 = 1004, -8192, 16384;
    /// `+ (ttData.value > alpha) * 885` (search.cpp:1317).
    lmr_ttpv_adjust_value: i32 = 885, -8192, 16384;
    /// The `816` in `(ttData.depth >= depth) * (816 + cutNode * 940)`.
    lmr_ttpv_adjust_depth: i32 = 816, -8192, 16384;
    /// The `940 * cutNode` inside the same term.
    lmr_ttpv_adjust_depth_cut: i32 = 940, -8192, 16384;
    /// The `+ 697` base offset (search.cpp:1321).
    lmr_base_offset: i32 = 697, -4096, 8192;
    /// `r -= lmrPerMove * moveCount` (search.cpp:1323).
    lmr_per_move: i32 = 65, 0, 1024;
    /// `r -= |correctionValue| / lmrCorrectionDiv` (search.cpp:1324).
    lmr_correction_div: i32 = 26310, 1, 4000000;
    /// `r += lmrCutNode` at a cut node (search.cpp:1328).
    lmr_cut_node: i32 = 4026, -8192, 16384;
    /// The extra `933` when the node has no TT move (search.cpp:1328).
    lmr_cut_node_no_tt: i32 = 933, -8192, 16384;
    /// `r += lmrTtCapture * ttCapture` (search.cpp:1332).
    lmr_tt_capture: i32 = 1079, -8192, 16384;
    /// `r += lmrCutoffOne` when the next ply already had a cutoff.
    lmr_cutoff_one: i32 = 264, -8192, 16384;
    /// The extra `1095` for more than one cutoff (search.cpp:1336).
    lmr_cutoff_two: i32 = 1095, -8192, 16384;
    /// The extra `1138` at an all-node (search.cpp:1336).
    lmr_all_node_cutoff: i32 = 1138, -8192, 16384;
    /// `r -= 2179` for the move that *is* the TT move (search.cpp:1340).
    lmr_tt_move: i32 = 2179, 0, 16384;
    /// `r -= statScore * lmrStatScoreNum / lmrStatScoreDen` (search.cpp:1352).
    lmr_stat_score_num: i32 = 439, 0, 16384;
    /// Denominator of the stat-score term.
    lmr_stat_score_den: i32 = 4096, 1, 65536;
    // --- `ss->statScore` itself, search.cpp:1342-1349 --------------------
    /// `873 * PieceValue[victim] / statScoreCaptureDiv` (search.cpp:1343).
    stat_score_capture_weight: i32 = 873, 0, 8192;
    /// The `128` in the capture branch of `statScore`.
    stat_score_capture_div: i32 = 128, 1, 8192;
    /// The main-history weight `2252` in the quiet branch of `statScore`.
    stat_score_main: i32 = 2252, 0, 16384;
    /// The first continuation-history weight `1126`.
    stat_score_cont1: i32 = 1126, 0, 16384;
    /// The second continuation-history weight `1093`.
    stat_score_cont2: i32 = 1093, 0, 16384;
    /// The `/ 1024` that normalises the quiet branch back to history units.
    stat_score_div: i32 = 1024, 1, 65536;
    /// `r += lmrAlphaEval * clamp(alpha - eval, lo, hi)` (search.cpp:1355).
    lmr_alpha_eval: i32 = 3, 0, 64;
    /// Lower clamp of the `alpha - eval` term.
    lmr_alpha_eval_min: i32 = -64, -1024, 0;
    /// Upper clamp of the `alpha - eval` term.
    lmr_alpha_eval_max: i32 = 96, 0, 1024;
    /// `r += r * lmrAllNode / (lmrAllNodeDepth * depth + lmrAllNodeOffset)`
    /// (search.cpp:1359).
    lmr_all_node_num: i32 = 276, 0, 4096;
    /// Depth coefficient of the all-node scale.
    lmr_all_node_depth: i32 = 256, 1, 4096;
    /// Constant term of the all-node scale.
    lmr_all_node_offset: i32 = 268, 1, 4096;
    /// `d = max(1, min(newDepth - r / lmrUnit, newDepth + lmrMaxBonus))`.
    lmr_max_bonus: i32 = 2, 0, 16;
    /// `depth >= lmrMinDepth` (search.cpp:1362).
    lmr_min_depth: i32 = 2, 1, 16;
    /// `+ PvNode` on the searched depth (search.cpp:1369).
    lmr_pv_bonus: i32 = 1, 0, 8;
    /// `value > bestValue + lmrDeeperThreshold` triggers a deeper re-search.
    lmr_deeper_threshold: i32 = 53, 0, 4096;
    /// `value < bestValue + lmrShallowerThreshold` triggers a shallower one.
    lmr_shallower_threshold: i32 = 8, -4096, 4096;
    /// `r += lmrNoTtMoveFullDepth` in the skip-LMR path (search.cpp:1399).
    lmr_no_tt_move: i32 = 1127, 0, 16384;
    /// `r > lmrFullDepthHigh` in the skip-LMR path (search.cpp:1403).
    lmr_full_depth_high: i32 = 5234, 0, 32768;
    /// `r > lmrFullDepthHigher` in the skip-LMR path (search.cpp:1403).
    lmr_full_depth_higher: i32 = 5487, 0, 32768;
    /// `newDepth > lmrFullDepthHigherMinDepth` for the second step above.
    lmr_full_depth_higher_min_depth: i32 = 2, 1, 16;
    /// Hindsight: `priorReduction >= lmrHindsightMinReduction` (search.cpp:867).
    lmr_hindsight_min_reduction: i32 = 3, 0, 16;
    /// Hindsight second condition (search.cpp:869).
    lmr_hindsight_min_reduction_2: i32 = 2, 0, 16;
    /// `ss->staticEval + (ss-1)->staticEval > lmrHindsightDepth` (search.cpp:869).
    lmr_hindsight_depth: i32 = 166, -4096, 4096;
    /// Initial value of `rootDelta`. Stockfish 19 assigns it only inside its
    /// aspiration loop; Morstilia seeds it so the `reduction()` division is
    /// always defined, and `iterative_search` overwrites it with the live
    /// root window before every root search.
    lmr_root_delta_default: i32 = 32, 1, 4096;

    // ---------------------------------------------------------------------
    // Pruning thresholds that already existed (Stage 02). Moved verbatim.
    //
    // Defaults are the *previous* hardcoded constants, and the margin
    // formulas in `pruning.rs` are unchanged: making them configurable must
    // not by itself alter a single node.
    // ---------------------------------------------------------------------

    /// Was `pruning::FUTILITY_DEPTH`.
    futility_depth: i32 = 4, 0, 32;
    /// Was `pruning::RAZOR_DEPTH`.
    razor_depth: i32 = 3, 0, 32;
    /// Was `pruning::RFP_DEPTH`.
    rfp_depth: i32 = 6, 0, 32;
    /// Was `pruning::NULL_MOVE_MIN_DEPTH`.
    null_move_min_depth: i32 = 2, 0, 32;
    /// Was `pruning::PROBCUT_DEPTH`.
    probcut_depth: i32 = 5, 0, 32;
    /// Was `pruning::DELTA_MARGIN`.
    delta_margin: i32 = 200, 0, 2000;
    /// Was `pruning::HISTORY_PRUNE_THRESHOLD`.
    history_prune_threshold: i32 = -8000, -32768, 0;
    /// Was `pruning::QUIET_PRUNE_LIMIT`.
    quiet_prune_limit: i32 = 5, 0, 256;
    /// The *additive* base of `see_prune_threshold`, which is `-80 * depth`
    /// and therefore has no base at all.
    ///
    /// Zero is the only value that reproduces Stage 02. A base of 80 would make
    /// the threshold `-160` at depth 1 — twice as strict as the search it is
    /// supposed to be a parameterisation of — and no test would necessarily
    /// notice, because the search still prunes and still plays legal moves.
    /// It is kept as a knob because a tuner legitimately wants to shift the
    /// whole curve; it is kept at zero because that is what the engine was.
    see_prune_base: i32 = 0, 0, 2000;
    /// Was the depth coefficient of `see_prune_threshold` (`-80 * depth`).
    see_prune_depth: i32 = 80, 0, 2000;
    /// Was `futility_margin`'s `100 + 75 * depth` base.
    futility_margin_base: i32 = 100, 0, 2000;
    /// Was `futility_margin`'s depth coefficient.
    futility_margin_depth: i32 = 75, 0, 2000;
    /// Was `futility_margin`'s improving bonus (subtracted when improving).
    futility_margin_improving: i32 = 25, 0, 1000;
    /// Was `razor_margin`'s `230 + 90 * depth` base.
    razor_margin_base: i32 = 230, 0, 2000;
    /// Was `razor_margin`'s depth coefficient.
    razor_margin_depth: i32 = 90, 0, 2000;
    /// Was `reverse_futility_margin`'s `80 + 70 * depth` base.
    rfp_margin_base: i32 = 80, 0, 2000;
    /// Was `reverse_futility_margin`'s depth coefficient.
    rfp_margin_depth: i32 = 70, 0, 2000;
    /// Was `reverse_futility_margin`'s improving bonus.
    rfp_margin_improving: i32 = 25, 0, 1000;
    /// Was `probcut_margin`'s `110 + 85 * depth` base.
    probcut_margin_base: i32 = 110, 0, 2000;
    /// Was `probcut_margin`'s depth coefficient.
    probcut_margin_depth: i32 = 85, 0, 2000;
    /// Was `null_move_reduction`'s constant `3`.
    null_move_reduction_base: i32 = 3, 1, 32;
    /// Was `null_move_reduction`'s `depth / 4` divisor.
    null_move_reduction_div: i32 = 4, 1, 32;
    /// Was `null_move_reduction`'s stagnant bonus.
    null_move_reduction_stagnant: i32 = 1, 0, 8;
    /// Max reduction plies `null_move_reduction` may reach; the previous
    /// implementation's arithmetic already bounded it at 11 for `depth <= 24`.
    null_move_reduction_max: i32 = 11, 1, 32;

    // ---------------------------------------------------------------------
    // Stockfish 19 child-node pruning — search.cpp:991-1007. Evaluated in the
    // parent right after `do_move`, so the child never enters the main search.
    // Additive to the node-level razor/RFP above, which remain the authority.
    // ---------------------------------------------------------------------

    /// `482` in `eval < alpha - 482 * depth * depth` (search.cpp:991).
    child_razor_margin_base: i32 = 482, 0, 4096;
    /// `45 + childRfpMultDepth * depth` (search.cpp:999).
    child_rfp_mult_base: i32 = 45, 0, 4096;
    /// The `4` in `45 + 4 * depth` (search.cpp:999).
    child_rfp_mult_depth: i32 = 4, 0, 4096;
    /// The `85` cap on the multiplier (search.cpp:999).
    child_rfp_mult_cap: i32 = 85, 1, 4096;
    /// `- 20 * !ttHit` (search.cpp:1000).
    child_rfp_no_tt: i32 = 20, 0, 4096;
    /// The `2789 * improving` hindsight weight (search.cpp:1003).
    child_rfp_improving: i32 = 2789, 0, 32768;
    /// The `335 * opponentWorsening` hindsight weight (search.cpp:1003).
    child_rfp_worsening: i32 = 335, 0, 32768;
    /// The `/ 1024` that keeps the hindsight term in margin units.
    child_rfp_hindsight_div: i32 = 1024, 1, 65536;
    /// `|correctionValue| / childRfpCorrectionDiv` (search.cpp:1004).
    child_rfp_correction_div: i32 = 198435, 1, 4000000;
    /// The `661` in the smoothed fail-high `(661 * beta + 363 * eval) / 1024`.
    child_rfp_value_beta: i32 = 661, 0, 32768;
    /// The `363` in the same expression.
    child_rfp_value_eval: i32 = 363, 0, 32768;
    /// The `/ 1024` in the same expression.
    child_rfp_value_den: i32 = 1024, 1, 65536;
    /// The `depth < (seekMate ? 6 : 19)` gate. Stockfish marks this "SHOULD NOT
    /// be tuned"; it is a parameter only so a tuned file can express the
    /// constraint rather than hard-code it.
    child_rfp_max_depth: i32 = 19, 1, 64;

    // ---------------------------------------------------------------------
    // Evaluation-difference history bonus — sf_19 search.cpp:978-986.
    // `clamp(-(parentStaticEval + staticEval), -189, 194) + 60`, weighted by
    // 11 into the main history of the side that played the previous move.
    // ---------------------------------------------------------------------

    /// The `-189` lower clamp.
    eval_diff_min: i32 = -189, -4096, 0;
    /// The `194` upper clamp.
    eval_diff_max: i32 = 194, 0, 4096;
    /// The `+ 60` applied after the clamp.
    eval_diff_offset: i32 = 60, -4096, 4096;
    /// The `* 11` bonus weight.
    eval_diff_main_weight: i32 = 11, 0, 4096;

    // ---------------------------------------------------------------------
    // SF19 pruning mechanisms. Each is behind its own gate, so enabling them
    // is the only thing that changes the search.
    // ---------------------------------------------------------------------

    /// `staticEval >= beta - nmpMarginBase - nmpMarginDepth * depth
    ///  - nmpMarginImproving * improving + nmpMarginOffset` (search.cpp:1011).
    nmp_margin_base: i32 = 365, -4096, 8192;
    /// Depth coefficient of the null-move margin.
    nmp_margin_depth: i32 = 13, 0, 512;
    /// Improving coefficient of the null-move margin.
    nmp_margin_improving: i32 = 47, 0, 512;
    /// `R = nmpReductionBase + depth / nmpReductionDiv + max((staticEval - beta) / nmpReductionDivisor, 0)`.
    nmp_reduction_base: i32 = 7, 0, 32;
    /// Depth divisor of the null-move reduction.
    nmp_reduction_div: i32 = 3, 1, 32;
    /// Divisor of the `(staticEval - beta)` term.
    nmp_reduction_divisor: i32 = 256, 1, 8192;
    /// `beta >= nmpBetaFloor` (search.cpp:1012).
    nmp_beta_floor: i32 = -2000, -8000, 0;
    /// `depth < nmpVerificationMinDepth` skips the verification search
    /// (search.cpp:1027).
    nmp_verification_min_depth: i32 = 16, 2, 64;
    /// `probCutBeta = beta + probcutBetaBase - probcutBetaImproving * improving`.
    probcut_beta_base: i32 = 241, -2048, 8192;
    /// Improving coefficient of the ProbCut beta.
    probcut_beta_improving: i32 = 64, 0, 1024;
    /// `depth >= probcutMinDepth` (search.cpp:1058).
    probcut_min_depth: i32 = 3, 0, 32;
    /// `probCutDepth = depth - probcutProbeDepthImproving` (search.cpp:1063).
    probcut_probe_depth_improving: i32 = 5, 0, 32;
    /// `probCutDepth = depth - probcutProbeDepthStagnant` (search.cpp:1063).
    probcut_probe_depth_stagnant: i32 = 3, 0, 32;
    /// The "small ProbCut idea": `beta + ttProbcutBeta` (search.cpp:1101).
    tt_probcut_beta: i32 = 428, -2048, 8192;
    /// `ttData.depth >= depth - ttProbcutDepth` (search.cpp:1102).
    tt_probcut_depth: i32 = 4, 0, 32;
    /// Reverse futility at the *child*: `depth < rfpChildMaxDepth`
    /// (search.cpp:996).
    rfp_child_max_depth: i32 = 19, 1, 64;
    /// The same bound while seeking mate.
    rfp_child_seek_mate_depth: i32 = 6, 1, 32;
    /// `futilityMult = min(rfpChildMultBase + rfpChildMultDepth * depth, rfpChildMultCap)`.
    rfp_child_mult_base: i32 = 45, -1024, 1024;
    /// Depth coefficient of the child futility multiplier.
    rfp_child_mult_depth: i32 = 4, 0, 256;
    /// Cap of the child futility multiplier.
    rfp_child_mult_cap: i32 = 85, 1, 1024;
    /// `futilityMult -= rfpChildMultNoTt` on a TT miss (search.cpp:1000).
    rfp_child_mult_no_tt: i32 = 20, 0, 512;
    /// Improving coefficient inside the child RFP margin.
    rfp_child_margin_improving: i32 = 2789, 0, 16384;
    /// Opponent-worsening coefficient inside the child RFP margin.
    rfp_child_margin_worsening: i32 = 335, 0, 16384;
    /// Divisor of the correction term inside the child RFP margin.
    rfp_child_correction_div: i32 = 198435, 1, 4000000;
    /// The child RFP return is a blend of beta and eval.
    rfp_child_return_beta: i32 = 661, 0, 4096;
    /// The other half of that blend.
    rfp_child_return_eval: i32 = 363, 0, 4096;
    /// Razoring at the child: `eval < alpha - razorChildMargin * depth * depth`.
    razor_child_margin: i32 = 482, 0, 8192;
    /// Late quiet-move pruning: `moveCount >= (base + depth^2) / div`.
    late_quiet_base: i32 = 3, 0, 64;
    /// Divisor of the late-move threshold (`2 - improving` in `sf_19`).
    late_quiet_div: i32 = 2, 1, 16;
    late_quiet_div_improving: i32 = 1, 1, 16;
    /// Capture futility: `staticEval + base + perDepth * lmrDepth + piece value`.
    capture_futility_base: i32 = 234, -2048, 4096;
    /// Depth coefficient of capture futility.
    capture_futility_depth: i32 = 247, -2048, 4096;
    /// Capture-history term of capture futility.
    capture_futility_history: i32 = 134, -2048, 4096;
    /// `lmrDepth < captureFutilityMaxDepth` (search.cpp:1181).
    capture_futility_max_depth: i32 = 8, 0, 64;
    /// SEE margin for captures and checks: `base * depth + history * h / div`.
    see_capture_base: i32 = 177, 0, 4096;
    /// Capture-history term of the capture SEE margin.
    see_capture_history: i32 = 34, 0, 4096;
    /// Divisor of the capture-history SEE term.
    see_capture_history_div: i32 = 1024, 1, 65536;
    /// Quiet SEE: `seeQuiet * lmrDepth^2`.
    see_quiet: i32 = 23, 0, 4096;
    /// Continuation-history pruning threshold is `contHistory * depth`.
    cont_history_prune: i32 = -4136, -65536, 0;
    /// Main-history contribution: `contMain * history / contMainDiv`.
    cont_history_main: i32 = 69, -4096, 4096;
    /// Divisor of the main-history contribution.
    cont_history_main_div: i32 = 32, 1, 4096;
    /// Quiet futility: `staticEval + perDepth * lmrDepth + alphaBonus + base`.
    quiet_futility_depth: i32 = 119, -2048, 4096;
    /// `alpha_bonus` when `staticEval > alpha`.
    quiet_futility_alpha: i32 = 90, 0, 4096;
    /// Constant term of quiet futility.
    quiet_futility_base: i32 = 164, -2048, 4096;
    /// `lmrDepth < quietFutilityMaxDepth` (search.cpp:1218).
    quiet_futility_max_depth: i32 = 12, 0, 64;
    /// Reverse-TT-cutoff depth for the "mismatch makes the entry useless"
    /// penalty (search.cpp:916).
    tt_penalize_min_depth: i32 = 6, 1, 32;
    /// The graph-history-interaction two-ply probe (search.cpp:895).
    tt_ghi_min_depth: i32 = 7, 1, 32;
    /// High rule50 counts suppress transposition cutoffs (search.cpp:893).
    tt_cutoff_rule50_limit: i32 = 96, 0, 200;
}

/// Boolean mechanism switches.
///
/// Each one turns a Stockfish-19 mechanism on or off independently, so the
/// A/B and SPRT infrastructure can measure them one at a time. Every default
/// is `true`, i.e. Stockfish-19-faithful. Setting all of them to `false`
/// reproduces the pre-Phase-02 search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchGates {
    /// Step 16 singular extensions, multi-cut **and** negative extensions
    /// (they are one mutually exclusive chain, so they share a switch).
    pub singular: bool,
    /// Step 11 internal iterative reductions.
    pub iir: bool,
    /// The Stockfish 19 `reduction()` formula in place of the classic
    /// `ln(depth) * ln(move)` curve.
    pub modern_lmr: bool,
    /// Per-ply `followPV` bookkeeping and the decisions that read it (IIR
    /// and the quiet-move pruning branch).
    pub follow_pv: bool,
    /// The `ttMoveHistory` scalar and its three update points.
    pub tt_move_history: bool,
    /// The hindsight depth adjustment from `priorReduction` (search.cpp:867).
    pub lmr_hindsight: bool,
    /// Step 8 razoring at the child (`eval < alpha - 482 * depth^2`).
    pub sf19_razor: bool,
    /// Step 9 reverse futility at the child.
    pub sf19_rfp: bool,
    /// Step 10 SF19 null move: cut-node gated, adaptive reduction,
    /// high-depth verification search.
    pub sf19_null_move: bool,
    /// Step 12 SF19 ProbCut (capture list driven, `probCutBeta`).
    pub sf19_probcut: bool,
    /// Step 13 "a small ProbCut idea" (TT-based shortcut).
    pub tt_probcut: bool,
    /// Step 15 late-move pruning, capture futility/SEE and the continuation
    /// history based quiet pruning.
    pub sf19_late_pruning: bool,
    /// `cutNode` bookkeeping and the LMR/negative-extension terms that read
    /// it. Inert without the other gates, but turning it off alone does
    /// change the reduction.
    pub cut_nodes: bool,
    /// The stricter SF19 TT cutoff predicate
    /// (`cutNode == (ttValue >= beta) || depth > 4`).
    pub sf19_tt_cutoff: bool,
    /// The graph-history-interaction two-ply probe before a TT cutoff.
    pub tt_ghi_probe: bool,
}

impl Default for SearchGates {
    fn default() -> Self {
        SearchGates {
            singular: true,
            iir: true,
            modern_lmr: true,
            follow_pv: true,
            tt_move_history: true,
            lmr_hindsight: true,
            sf19_razor: true,
            sf19_rfp: true,
            sf19_null_move: true,
            sf19_probcut: true,
            tt_probcut: true,
            sf19_late_pruning: true,
            cut_nodes: true,
            sf19_tt_cutoff: true,
            tt_ghi_probe: true,
        }
    }
}

impl SearchGates {
    /// All mechanisms off: the pre-Phase-02 search.
    pub fn all_off() -> SearchGates {
        SearchGates {
            singular: false,
            iir: false,
            modern_lmr: false,
            follow_pv: false,
            tt_move_history: false,
            lmr_hindsight: false,
            sf19_razor: false,
            sf19_rfp: false,
            sf19_null_move: false,
            sf19_probcut: false,
            tt_probcut: false,
            sf19_late_pruning: false,
            cut_nodes: false,
            sf19_tt_cutoff: false,
            tt_ghi_probe: false,
        }
    }

    /// Gate names in a stable order, for diagnostics and the tuner log.
    pub const NAMES: &'static [&'static str] = &[
        "singular",
        "iir",
        "modern_lmr",
        "follow_pv",
        "tt_move_history",
        "lmr_hindsight",
        "sf19_razor",
        "sf19_rfp",
        "sf19_null_move",
        "sf19_probcut",
        "tt_probcut",
        "sf19_late_pruning",
        "cut_nodes",
        "sf19_tt_cutoff",
        "tt_ghi_probe",
    ];

    /// Reads a gate by name (case-insensitive). Used by the UCI
    /// `setoption` path and the tuning CLI.
    pub fn get(&self, name: &str) -> Option<bool> {
        let n = name.trim().to_ascii_lowercase();
        Some(match n.as_str() {
            "singular" => self.singular,
            "iir" => self.iir,
            "modern_lmr" => self.modern_lmr,
            "follow_pv" => self.follow_pv,
            "tt_move_history" => self.tt_move_history,
            "lmr_hindsight" => self.lmr_hindsight,
            "sf19_razor" => self.sf19_razor,
            "sf19_rfp" => self.sf19_rfp,
            "sf19_null_move" => self.sf19_null_move,
            "sf19_probcut" => self.sf19_probcut,
            "tt_probcut" => self.tt_probcut,
            "sf19_late_pruning" => self.sf19_late_pruning,
            "cut_nodes" => self.cut_nodes,
            "sf19_tt_cutoff" => self.sf19_tt_cutoff,
            "tt_ghi_probe" => self.tt_ghi_probe,
            _ => return None,
        })
    }

    /// Writes a gate by name. Returns `false` for an unknown name.
    pub fn set(&mut self, name: &str, on: bool) -> bool {
        let n = name.trim().to_ascii_lowercase();
        match n.as_str() {
            "singular" => self.singular = on,
            "iir" => self.iir = on,
            "modern_lmr" => self.modern_lmr = on,
            "follow_pv" => self.follow_pv = on,
            "tt_move_history" => self.tt_move_history = on,
            "lmr_hindsight" => self.lmr_hindsight = on,
            "sf19_razor" => self.sf19_razor = on,
            "sf19_rfp" => self.sf19_rfp = on,
            "sf19_null_move" => self.sf19_null_move = on,
            "sf19_probcut" => self.sf19_probcut = on,
            "tt_probcut" => self.tt_probcut = on,
            "sf19_late_pruning" => self.sf19_late_pruning = on,
            "cut_nodes" => self.cut_nodes = on,
            "sf19_tt_cutoff" => self.sf19_tt_cutoff = on,
            "tt_ghi_probe" => self.tt_ghi_probe = on,
            _ => return false,
        }
        true
    }
}

impl SearchGates {
    /// Parses the accepted spellings of a boolean gate value.
    ///
    /// `true`/`1`/`on`/`yes` and `false`/`0`/`off`/`no`, case-insensitive and
    /// trimmed. Anything else is `None` so a `setoption` with a typo is
    /// rejected instead of silently leaving the gate at its default.
    pub fn parse_gate_value(v: &str) -> Option<bool> {
        SearchParams::parse_gate_value(v)
    }
}

impl SearchParams {
    /// Parses the accepted spellings of a boolean gate value.
    pub fn parse_gate_value(v: &str) -> Option<bool> {
        match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        }
    }

    /// Serializes to a TOML string.
    pub fn to_toml_string(&self) -> anyhow::Result<String> {
        toml::to_string(self).map_err(Into::into)
    }

    /// Parses a TOML string. Unknown fields are an error, so a malformed or
    /// stale tuned file fails loudly instead of being silently ignored.
    pub fn from_toml_str(s: &str) -> anyhow::Result<SearchParams> {
        toml::from_str(s).map_err(Into::into)
    }

    /// Loads parameters from a TOML file.
    pub fn load(path: &str) -> anyhow::Result<SearchParams> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read search params {path:?}: {e}"))?;
        SearchParams::from_toml_str(&s)
            .map_err(|e| anyhow::anyhow!("invalid search params {path:?}: {e}"))
    }

    /// Writes parameters to a TOML file.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let s = self.to_toml_string()?;
        std::fs::write(path, s)
            .map_err(|e| anyhow::anyhow!("cannot write search params {path:?}: {e}"))
    }

    /// A short fingerprint of the active configuration, for match reports and
    /// the tuning log. Two engines may only be compared when these match.
    pub fn fingerprint(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: i64| {
            h ^= v as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for x in self.to_vec() {
            mix((x * 1024.0).round() as i64);
        }
        for (i, name) in SearchGates::NAMES.iter().enumerate() {
            let _ = i;
            mix(i64::from(self.gates.get(name).unwrap_or(false)));
        }
        format!("{h:016x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_vector_round_trips() {
        let p = SearchParams::default();
        let v = p.to_vec();
        assert_eq!(v.len(), SearchParams::param_count(&p));
        assert_eq!(v.len(), SearchParams::bounds().len());
        let mut q = SearchParams::default();
        q.from_vec(&v);
        assert_eq!(p, q, "to_vec/from_vec must be exact inverses");
    }

    #[test]
    fn every_bound_brackets_its_default() {
        for (d, p) in SearchParams::bounds()
            .iter()
            .zip(SearchParams::default().to_vec())
        {
            assert!(
                d.min <= p && p <= d.max,
                "{} default {p} outside [{}, {}]",
                d.name,
                d.min,
                d.max
            );
            assert!(d.min < d.max, "{} has an empty range", d.name);
        }
    }

    #[test]
    fn from_vec_clamps_nothing_and_tolerates_short_input() {
        let mut p = SearchParams::default();
        p.from_vec(&[]);
        assert_eq!(p, SearchParams::default(), "an empty vector is a no-op");
        p.from_vec(&[7.0, 7.0]);
        assert_eq!(p.singular_min_depth, 7);
    }

    #[test]
    fn gates_have_unique_names_and_round_trip() {
        let mut names: Vec<&str> = SearchGates::NAMES.to_vec();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "gate names must be unique");

        let mut g = SearchGates::default();
        for name in SearchGates::NAMES {
            assert!(g.get(name).is_some(), "{name} must be readable");
            g.set(name, false);
            assert_eq!(g.get(name), Some(false), "{name} must be writable");
        }
        assert_eq!(g, SearchGates::all_off());
    }

    #[test]
    fn all_off_disables_every_gate() {
        let g = SearchGates::all_off();
        for name in SearchGates::NAMES {
            assert_eq!(g.get(name), Some(false), "{name} must be off");
        }
    }

    #[test]
    fn default_gates_are_all_on() {
        let g = SearchGates::default();
        for name in SearchGates::NAMES {
            assert_eq!(g.get(name), Some(true), "{name} must be on by default");
        }
    }

    #[test]
    fn toml_round_trips() {
        let p = SearchParams::default();
        let s = p.to_toml_string().unwrap();
        let q = SearchParams::from_toml_str(&s).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn unknown_toml_field_is_rejected() {
        assert!(SearchParams::from_toml_str("not_a_real_param = 1\n").is_err());
        assert!(SearchParams::from_toml_str("[gates]\nnope = true\n").is_err());
    }

    #[test]
    fn fingerprint_separates_configurations() {
        let a = SearchParams::default();
        let mut b = SearchParams::default();
        assert_eq!(a.fingerprint(), b.fingerprint());
        b.singular_min_depth += 1;
        assert_ne!(a.fingerprint(), b.fingerprint());
        let mut c = SearchParams::default();
        c.gates.singular = false;
        assert_ne!(a.fingerprint(), c.fingerprint());
    }
}

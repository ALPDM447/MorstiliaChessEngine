//! Search instrumentation: per-search counters and derived quality metrics.
//!
//! Every counter is a plain `u64` updated with a single increment in the hot
//! path — no atomics, no allocation. Each worker thread owns a
//! [`SearchStats`] inside its [`crate::search::SearchThread`]; the result
//! aggregation sums every worker's counters (with `Threads = 1` the sum is
//! trivially exact and fully deterministic). The one counter that lives on
//! the shared table instead — TT stores — is harvested separately into
//! [`crate::search::SearchResult::tt_stores`] so it is never double-counted.
//!
//! The derived percentages answer the classic questions:
//!
//! * `tt_hit_pct` — how often the table is useful at all;
//! * `tt_cutoff_pct` — how often a stored bound ends a node immediately;
//! * `first_move_cutoff_pct` — how often the *first searched* move caused a
//!   beta cutoff: the single best measure of move-ordering quality
//!   (Stockfish's "move-count" / first-move success proxy);
//! * `avg_moves_until_cutoff` — how many moves were searched before the
//!   cutoff move, averaged over all beta cutoffs;
//! * `see_prune_pct` — what share of SEE evaluations actually pruned a move;
//! * `total_pruned` — how many children were skipped by a pruning decision
//!   (the sum of every per-move and node-level prune counter, so "how much
//!   work did the prunes save me").

/// All per-thread counters for one search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Quiescence nodes entered (a subset of the node total).
    pub qsearch_nodes: u64,
    /// Transposition-table probes issued by the main search.
    pub tt_probes: u64,
    /// Probes that matched a stored entry.
    pub tt_hits: u64,
    /// Nodes whose search ended immediately on a usable TT bound.
    pub tt_cutoffs: u64,
    /// Fail highs (`score >= beta`) produced by the move loop.
    pub beta_cutoffs: u64,
    /// Beta cutoffs caused by the very first searched move — the purest
    /// ordering signal (TT move / best capture / top killer).
    pub first_move_cutoffs: u64,
    /// Sum of `moves searched so far` over every beta cutoff; dividing by
    /// `beta_cutoffs` yields the average moves needed to refute a node.
    pub moves_until_cutoff: u64,
    /// SEE evaluations run in the main search and quiescence.
    pub see_calls: u64,
    /// Moves skipped because SEE found them hopeless.
    pub see_pruned: u64,
    /// Null-move probes attempted (all of them, failed and successful).
    pub null_probes: u64,
    /// Null-move probes that failed high (`null_score >= beta`): the move
    /// loop was skipped entirely.
    pub null_cutoffs: u64,
    /// Quiet moves searched at a reduced depth (LMR).
    pub lmr_reduced: u64,
    /// Reduced quiet moves that beat `alpha` and were re-searched at full
    /// depth — the LMR fail-low safety valve firing.
    pub lmr_researched: u64,
    /// Quiet moves pruned by per-move futility.
    pub futility_pruned: u64,
    /// Quiet moves pruned by the (shallow, late) history-based rule.
    pub history_pruned: u64,
    /// Nodes pruned by reverse futility (static null move).
    pub rfp_pruned: u64,
    /// Razoring attempts (nodes dropped straight into quiescence).
    pub razor_attempts: u64,
    /// Razoring successes: the razor qsearch resolved the node to a fail-low
    /// (`score <= alpha`), confirming the position really was hopeless.
    pub razor_cutoffs: u64,
    /// ProbCut probes attempted (shallow null-window proof searches).
    pub probcut_attempts: u64,
    /// ProbCut probes that failed high and cut the whole subtree.
    pub probcut_cutoffs: u64,
    /// Captures skipped by delta pruning in quiescence.
    pub delta_pruned: u64,
}

impl SearchStats {
    /// Adds another thread's counters into this one (result aggregation).
    #[inline]
    pub fn add(&mut self, other: &SearchStats) {
        self.qsearch_nodes += other.qsearch_nodes;
        self.tt_probes += other.tt_probes;
        self.tt_hits += other.tt_hits;
        self.tt_cutoffs += other.tt_cutoffs;
        self.beta_cutoffs += other.beta_cutoffs;
        self.first_move_cutoffs += other.first_move_cutoffs;
        self.moves_until_cutoff += other.moves_until_cutoff;
        self.see_calls += other.see_calls;
        self.see_pruned += other.see_pruned;
        self.null_probes += other.null_probes;
        self.null_cutoffs += other.null_cutoffs;
        self.lmr_reduced += other.lmr_reduced;
        self.lmr_researched += other.lmr_researched;
        self.futility_pruned += other.futility_pruned;
        self.history_pruned += other.history_pruned;
        self.rfp_pruned += other.rfp_pruned;
        self.razor_attempts += other.razor_attempts;
        self.razor_cutoffs += other.razor_cutoffs;
        self.probcut_attempts += other.probcut_attempts;
        self.probcut_cutoffs += other.probcut_cutoffs;
        self.delta_pruned += other.delta_pruned;
    }

    /// Share of probes that found an entry, `0..=100`.
    pub fn tt_hit_pct(&self) -> f64 {
        pct(self.tt_hits, self.tt_probes)
    }

    /// Share of probes whose stored bound ended the node outright.
    pub fn tt_cutoff_pct(&self) -> f64 {
        pct(self.tt_cutoffs, self.tt_probes)
    }

    /// Share of beta cutoffs caused by the first searched move.
    pub fn first_move_cutoff_pct(&self) -> f64 {
        pct(self.first_move_cutoffs, self.beta_cutoffs)
    }

    /// Average number of moves searched before a beta cutoff landed.
    pub fn avg_moves_until_cutoff(&self) -> f64 {
        if self.beta_cutoffs == 0 {
            0.0
        } else {
            self.moves_until_cutoff as f64 / self.beta_cutoffs as f64
        }
    }

    /// Share of SEE evaluations that ended in a move being pruned.
    pub fn see_prune_pct(&self) -> f64 {
        pct(self.see_pruned, self.see_calls)
    }

    /// Share of reduced LMR moves that needed the full-depth re-search
    /// (the re-search rate is a healthy 0..=100% — near 0 means the model is
    /// over-confident, near 100 means the reductions are pointless).
    pub fn lmr_research_pct(&self) -> f64 {
        pct(self.lmr_researched, self.lmr_reduced)
    }

    /// Share of null-move probes that actually cut the node.
    pub fn null_cutoff_pct(&self) -> f64 {
        pct(self.null_cutoffs, self.null_probes)
    }

    /// Share of ProbCut probes that cut the node.
    pub fn probcut_cutoff_pct(&self) -> f64 {
        pct(self.probcut_cutoffs, self.probcut_attempts)
    }

    /// Share of razoring attempts whose node really did fail low.
    pub fn razor_cutoff_pct(&self) -> f64 {
        pct(self.razor_cutoffs, self.razor_attempts)
    }

    /// Total nodes whose full search was skipped by a pruning decision: every
    /// per-move prune (futility, history, SEE in the main search, delta and
    /// SEE in quiescence) plus every node-level prune (reverse futility,
    /// razoring, null-move cutoffs, ProbCut cutoffs). Null and ProbCut
    /// *attempts* that failed still searched (the probe itself ran), so only
    /// the successful cutoffs are counted.
    pub fn total_pruned(&self) -> u64 {
        self.futility_pruned
            + self.history_pruned
            + self.see_pruned
            + self.delta_pruned
            + self.rfp_pruned
            + self.razor_attempts
            + self.null_cutoffs
            + self.probcut_cutoffs
    }
}

/// `part / whole * 100`, guarding the divide-by-zero.
fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additions_merge_and_never_double_count_tt_stores() {
        let mut a = SearchStats {
            qsearch_nodes: 10,
            tt_probes: 100,
            tt_hits: 40,
            tt_cutoffs: 30,
            beta_cutoffs: 80,
            first_move_cutoffs: 60,
            moves_until_cutoff: 200,
            see_calls: 50,
            see_pruned: 20,
            null_probes: 9,
            null_cutoffs: 7,
            lmr_reduced: 31,
            lmr_researched: 6,
            futility_pruned: 12,
            history_pruned: 3,
            rfp_pruned: 4,
            razor_attempts: 8,
            razor_cutoffs: 8,
            probcut_attempts: 5,
            probcut_cutoffs: 2,
            delta_pruned: 1,
        };
        let b = SearchStats {
            qsearch_nodes: 5,
            tt_probes: 100,
            tt_hits: 30,
            tt_cutoffs: 20,
            beta_cutoffs: 20,
            first_move_cutoffs: 10,
            moves_until_cutoff: 90,
            see_calls: 50,
            see_pruned: 5,
            null_probes: 2,
            null_cutoffs: 1,
            lmr_reduced: 13,
            lmr_researched: 2,
            futility_pruned: 5,
            history_pruned: 1,
            rfp_pruned: 1,
            razor_attempts: 2,
            razor_cutoffs: 1,
            probcut_attempts: 3,
            probcut_cutoffs: 1,
            delta_pruned: 0,
        };
        a.add(&b);
        assert_eq!(a.qsearch_nodes, 15);
        assert_eq!(a.tt_probes, 200);
        assert_eq!(a.beta_cutoffs, 100);
        assert_eq!(a.moves_until_cutoff, 290);
        assert_eq!(a.null_probes, 11);
        assert_eq!(a.lmr_researched, 8);
        assert_eq!(a.probcut_cutoffs, 3);
    }

    #[test]
    fn derived_metrics_are_sane() {
        let s = SearchStats {
            tt_probes: 200,
            tt_hits: 100,
            tt_cutoffs: 50,
            beta_cutoffs: 100,
            first_move_cutoffs: 75,
            moves_until_cutoff: 250,
            see_calls: 100,
            see_pruned: 25,
            null_probes: 20,
            null_cutoffs: 8,
            lmr_reduced: 40,
            lmr_researched: 10,
            futility_pruned: 30,
            history_pruned: 5,
            rfp_pruned: 4,
            razor_attempts: 10,
            razor_cutoffs: 6,
            probcut_attempts: 6,
            probcut_cutoffs: 2,
            delta_pruned: 7,
            ..SearchStats::default()
        };
        assert!((s.tt_hit_pct() - 50.0).abs() < 1e-9);
        assert!((s.tt_cutoff_pct() - 25.0).abs() < 1e-9);
        assert_eq!(s.first_move_cutoff_pct(), 75.0);
        assert_eq!(s.avg_moves_until_cutoff(), 2.5);
        assert_eq!(s.see_prune_pct(), 25.0);
        assert_eq!(s.lmr_research_pct(), 25.0);
        assert_eq!(s.null_cutoff_pct(), 40.0);
        assert_eq!(s.razor_cutoff_pct(), 60.0);
        assert_eq!(s.probcut_cutoff_pct(), 100.0 / 3.0);
        assert_eq!(
            s.total_pruned(),
            30 + 5 + 25 + 7 + 4 + 10 + 8 + 2,
            "every prune family must be included exactly once"
        );
    }

    #[test]
    fn empty_stats_report_zero_not_nan() {
        let s = SearchStats::default();
        assert_eq!(s.tt_hit_pct(), 0.0);
        assert_eq!(s.first_move_cutoff_pct(), 0.0);
        assert_eq!(s.avg_moves_until_cutoff(), 0.0);
        assert_eq!(s.total_pruned(), 0);
    }
}

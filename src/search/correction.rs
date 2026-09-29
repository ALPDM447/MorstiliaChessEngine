//! Correction history: a learned adjustment on top of the raw static
//! evaluation.
//!
//! A static evaluation is an estimate, and the estimate is *systematically*
//! biased in a given kind of position — a pawn structure, a set of minor pieces,
//! a material balance. When the search later finds out how that kind of
//! position really goes, the error is a property of the position's material
//! shape, not of the specific line, so it can be learned and fed back into the
//! next occurrence. This is the same principle as history heuristics, applied to
//! evaluation rather than to move ordering.
//!
//! # What is learned
//!
//! Four families, each a small `i16` cell updated with a gravity rule, exactly
//! like [`crate::move_ordering::history`] so the storage and update conventions
//! stay consistent across the codebase:
//!
//! 1. **Pawn structure** — keyed by the pawn placement alone.
//! 2. **Minor pieces** — keyed by the knight/bishop placement alone.
//! 3. **Non-pawn material** — two cells per bucket, one keyed by white's
//!    non-pawn placement and one by black's, *summed* under a single weight.
//! 4. **Continuation** — keyed by the `(piece, to)` of the move that reached
//!    this node, read through the table selected by the move two or four plies
//!    ago.
//!
//! The three material keys are maintained incrementally on
//! [`crate::board::Position`], so reading a family is one array index and an
//! update is a handful of XORs — no per-node hashing.
//!
//! Stockfish addresses the two continuation families through a *chain* of
//! tables: `ss->continuationCorrectionHistory` is the `table[piece][to]` of the
//! move that reached the node two (or four) plies back, and each such table is
//! itself indexed by the `(piece, to)` of the move that reached the current
//! node. Both halves of the key are exact, and the chain is 448 × 448 `i16` —
//! small enough to reproduce directly, which is what this module does (see
//! [`CONT_CELLS`]). Both `(piece, to)` pairs are read *after* the move, so a
//! promoting pawn contributes as the queen it became.
//!
//! # The value
//!
//! The four families are read into a single integer `cv` with fixed weights
//! (Stockfish's), and the static evaluation becomes `eval + cv / 131072`. The
//! weights are deliberately unbalanced: the continuation family dominates
//! because it is the most specific signal, and the material families modulate
//! it.
//!
//! The saturated ceiling is 540 centipawns, computed from the weights in
//! `correction_magnitude_is_bounded_by_construction`. Real searches stay well
//! below it — 83 to 246 centipawns in the positions measured — and reach it only
//! in a bare KQ-vs-K endgame, where the absent pawns and minor pieces collapse
//! two of the families into a single global cell. That is a meaningful position
//! change but never a mate score, and [`correct`] keeps it out of the mate zone
//! regardless.
//!
//! # Where it is applied
//!
//! Exactly one place per node, and only outside check: Stockfish's
//! `to_corrected_static_eval` is called from the "static evaluation of the
//! position" step, which a check node skips in favour of the inherited
//! `(ss - 2)->staticEval`. A check node therefore has no opinion of its own to
//! correct, and reading a cell there would be learning from a fallback value.
//!
//! # Bounds
//!
//! A correction is *always* clamped away from the mate zone by [`correct`], so
//! no pruning gate that tests `beta.abs() < MATE_ZONE` can be fooled by a
//! runaway table. Individual cells are bounded by [`CORRECTION_LIMIT`], so no
//! cell can overflow and the weighted sum is bounded by construction.
//!
//! # Learning rule
//!
//! A node updates the tables when the search's verdict disagrees with the static
//! evaluation *in the direction the node's outcome actually went*, and the
//! best move is quiet (a capture's outcome is already explained by material, so
//! there is nothing structural to learn from it). The bonus is the observed
//! error scaled by depth — a deeper search knows the position better, so its
//! opinion is worth more — and points in the direction the evaluation was
//! wrong.

use crate::board::Color;
use crate::board::Position;
use crate::board::Square;
use crate::move_ordering::history::MoveCtx;
use crate::types::MAX_NON_MATE;

/// Largest magnitude a single correction cell may hold.
///
/// `i16::MAX` would allow the weighted sum to overflow `i32`, so the limit is
/// Stockfish's `CORRECTION_HISTORY_LIMIT`.
pub const CORRECTION_LIMIT: i32 = 1024;

/// Index of each colour inside a [`crate::board::Position::keys`] array.
const WHITE: usize = Color::White as usize;
const BLACK: usize = Color::Black as usize;

/// Largest magnitude the *observed error* may have before it is taught.
///
/// Stockfish clamps the signed error to a quarter of the cell limit and only
/// then scales it into the per-family bonuses. Keeping the clamp here is what
/// bounds how fast a cell can saturate, so the correction stays a nudge rather
/// than a claim.
pub const BONUS_LIMIT: i32 = CORRECTION_LIMIT / 4;

/// Stockfish's `1061 * bonus / 1024`: one global trim applied to the error
/// before the per-family weights divide it up again.
const BONUS_TRIM: i32 = 1061;

/// Divisor turning the weighted correction sum into centipawns.
///
/// Stockfish's `131072`, chosen so a fully saturated set of families (every
/// cell at `±CORRECTION_LIMIT`) yields a few hundred centipawns — enough to
/// reorder near-equal evaluations, never enough to invent a tactical verdict.
pub const CV_TO_CP: i32 = 131_072;

/// Weight of each family in the correction sum.
///
/// Stockfish's five constants, unchanged: they encode how specific (and so how
/// trustworthy) each family is. Note the non-pawn weight applies to the **sum**
/// of the two non-pawn cells, so the family as a whole is worth up to twice
/// `W_NON_PAWN`, and each continuation horizon carries the full continuation
/// weight rather than a share of it.
const W_PAWN: i32 = 15_341;
const W_MINOR: i32 = 10_569;
const W_NON_PAWN: i32 = 12_906;

/// The continuation family reads two context slots — the moves played two and
/// four plies ago — and Stockfish multiplies their **sum** by this one constant:
///
/// ```cpp
/// 8761 * ((*(ss - 2)->continuationCorrectionHistory)[pc][sq]
///        + (*(ss - 4)->continuationCorrectionHistory)[pc][sq])
/// ```
///
/// Distributing a single weight over a sum gives the *same* weight to each
/// addend, so both families carry `8761` in full. Splitting the constant into
/// two halves that happen to add up would halve the family's whole contribution
/// — the continuation family is the most specific signal in the correction, and
/// quietly halving it is exactly the kind of deviation that shows up later as
/// "the correction does not seem to do anything".
const W_CONT_2PLY: i32 = 8_761;
const W_CONT_4PLY: i32 = 8_761;

/// Stockfish's `nonPawnWeight` from `update_correction_history`, applied to
/// *each* of the white and black non-pawn cells.
const NON_PAWN_WEIGHT: i32 = 186;

/// Rows of the continuation family: `piece * 64 + to`. Seven role slots (indexed
/// by `Role as usize`, slot 0 unused) times 64 target squares.
const PIECE_TO: usize = 7 * 64;

/// Buckets for the shared, colour-independent base table.
///
/// A power of two so the index is a mask. The table holds a `(colour, key)`
/// cell per bucket; collisions between unrelated material shapes are absorbed by
/// the fact that a correction only ever *nudges* an evaluation.
const BASE_BUCKETS: usize = 1 << 16;

/// Cells in one continuation family: the full `(selecting, incoming)` cross
/// product.
///
/// This is Stockfish's `MultiArray<Stats<i16, 1024, PIECE_NB, SQUARE_NB>,
/// PIECE_NB, SQUARE_NB>` flattened — 448 × 448 = 200,704 cells, 384 KiB of
/// `i16` per family, 768 KiB for both.
///
/// An earlier revision of this module hashed the selecting move into 64 buckets
/// to avoid a table of tables, on the grounds that 448 × 448 looked expensive.
/// It is not: 768 KiB per thread is under half the 1.75 MiB the quiet
/// continuation history in [`crate::move_ordering::history`] already spends per
/// thread, and the hashed form was not free — it collapsed the selecting
/// context, which is the half of the key that makes the family *continuation*
/// rather than a generic piece-to-square table. Two horizons of a collapsed key
/// are not the same as two horizons of a real key, and the approximation would
/// have been the one place in this module that quietly meant something other
/// than what its name said.
const CONT_CELLS: usize = PIECE_TO * PIECE_TO;

/// Stockfish's stand-in when the node has no incoming move.
///
/// `correction_value` returns this constant instead of reading a cell, because
/// at the root there is no move whose `(piece, to)` could address one:
///
/// ```cpp
/// const int cntcv = m.is_ok() ? 8761 * (...) : 64049;
/// ```
///
/// It is carried rather than replaced by 0 for two reasons. The first is that it
/// is what Stockfish does, so any later comparison against it holds. The second
/// is that a non-zero constant is *visible* where a zero would be silent: it
/// shows up in the tests as a number that has to be accounted for, so "the
/// continuation families contribute nothing here" stays a claim somebody is
/// making. At 64049 / 131072 = 0.49 it rounds away in [`correct`], so the search
/// cannot tell the two choices apart — which is the honest thing to record, and
/// the reason this is a fidelity detail rather than a design decision.
pub const NO_INCOMING: i32 = 64_049;

/// Stored value of one correction family.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Cell(i16);

impl Cell {
    /// Gravity update: `entry += bonus - entry * |bonus| / LIMIT`, clamped.
    ///
    /// Identical in form to [`crate::move_ordering::history`]'s update, so both
    /// tables decay toward the newest evidence at the same rate.
    #[inline]
    fn update(&mut self, bonus: i32) {
        let cur = i32::from(self.0);
        let next = cur + bonus - cur * bonus.abs() / CORRECTION_LIMIT;
        self.0 = next.clamp(-CORRECTION_LIMIT, CORRECTION_LIMIT) as i16;
    }
}

/// The four material families for one colour of one bucket.
///
/// Stockfish's `CorrHist` entry keeps `pawn`, `minor` and *two* non-pawn
/// cells — `nonPawnWhite` and `nonPawnBlack` — and `correction_value` sums the
/// pair under one weight. They are genuinely two different cells and not one
/// cell read twice: each is addressed by the *other* colour's non-pawn key, so
/// "how well do positions with this much white non-pawn material get judged"
/// and the same question for black are learned and read independently.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct Bundle {
    pawn: Cell,
    minor: Cell,
    /// Addressed by white's non-pawn key, read by whoever is to move.
    non_pawn_white: Cell,
    /// Addressed by black's non-pawn key, read by whoever is to move.
    non_pawn_black: Cell,
}

/// Per-thread correction history.
///
/// One instance per [`crate::search::SearchThread`], like every other learned
/// table in the engine: the engine keeps no global mutable search state, so a
/// `Threads = 1` search stays fully deterministic.
///
/// Allocation: `BASE_BUCKETS * 2` bundles (colour × bucket) plus two
/// continuation tables, about 1.3 MB of `i16` — the same order as the existing
/// history tables.
#[derive(Clone, Debug)]
pub struct CorrectionHistory {
    /// `colour * BASE_BUCKETS + bucket`, holding the material families.
    base: Box<[Bundle]>,
    /// Two-ply continuation, indexed by
    /// `selecting * PIECE_TO + incoming` (see [`CONT_CELLS`]).
    cont_2ply: Box<[Cell]>,
    /// Four-ply continuation, same layout, read through the move played four
    /// plies ago rather than two.
    cont_4ply: Box<[Cell]>,
}

/// The three move contexts a node's correction is read against.
///
/// Stockfish reaches all three through its stack: `(ss - 1)->currentMove` is the
/// move that reached this node and names the cell, while
/// `ss->continuationCorrectionHistory` for plies two and four back names the
/// *table*. Bundling the three here keeps that shape visible at the call site
/// instead of hiding it behind a positional list, and makes it a compile error to
/// pass the two-ply context where the four-ply one belongs — the mistake two
/// positional `Option<MoveCtx>` arguments invited.
#[derive(Copy, Clone, Debug, Default)]
pub struct CorrectionCtx {
    /// The move that reached this node. `None` at the root.
    pub incoming: Option<MoveCtx>,
    /// The move played two plies ago, which selects the table the two-ply family
    /// reads from. `None` at the first two plies, or just after a null move.
    pub two_plies_ago: Option<MoveCtx>,
    /// The move played four plies ago, selecting the four-ply family's table.
    pub four_plies_ago: Option<MoveCtx>,
}

impl CorrectionCtx {
    /// The context for the node at search ply `ply`.
    ///
    /// `ctx[ply]` is the move that *reached* the node, so the two-ply and
    /// four-ply selecting moves are `ctx[ply - 1]` and `ctx[ply - 3]`. Missing
    /// slots near the root are `None` by construction, which is what makes the
    /// root's correction fall back to the material families alone.
    #[inline]
    pub fn from_stack(ctx: &[Option<MoveCtx>; crate::types::MAX_PLY], ply: usize) -> CorrectionCtx {
        CorrectionCtx {
            incoming: ctx[ply],
            two_plies_ago: ctx[ply.saturating_sub(1)],
            four_plies_ago: ctx[ply.saturating_sub(3)],
        }
    }
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        CorrectionHistory::new()
    }
}

impl CorrectionHistory {
    pub fn new() -> CorrectionHistory {
        CorrectionHistory {
            base: vec![Bundle::default(); 2 * BASE_BUCKETS].into_boxed_slice(),
            cont_2ply: vec![Cell::default(); CONT_CELLS].into_boxed_slice(),
            cont_4ply: vec![Cell::default(); CONT_CELLS].into_boxed_slice(),
        }
    }

    /// Zeroes every table (`ucinewgame`).
    pub fn clear(&mut self) {
        self.base.fill(Bundle::default());
        self.cont_2ply.fill(Cell::default());
        self.cont_4ply.fill(Cell::default());
    }

    /// The correction for `pos` in the search context `ctx`.
    ///
    /// The colour always comes from the position, never from the caller, so the
    /// same position read in the same context always yields the same value.
    #[inline]
    pub fn correction(&self, pos: &Position, ctx: CorrectionCtx) -> i32 {
        let us = pos.turn() as usize;

        let mut cv = W_PAWN * i32::from(self.base[base_index(us, pos.keys.pawn)].pawn.0)
            + W_MINOR * i32::from(self.base[base_index(us, pos.keys.minor)].minor.0)
            + W_NON_PAWN
                * (i32::from(
                    self.base[base_index(us, pos.keys.non_pawn[WHITE])]
                        .non_pawn_white
                        .0,
                ) + i32::from(
                    self.base[base_index(us, pos.keys.non_pawn[BLACK])]
                        .non_pawn_black
                        .0,
                ));

        // Both continuation families name their cell by the move that *reached*
        // this node — the role standing on its destination after the move, and
        // that square — and differ only in which earlier move selected the table.
        // Both roles are post-move: Stockfish reads `pos.piece_on(m.to_sq())` and
        // its selecting table is indexed by `dirtyPiece.pc`, the same promotion-
        // aware role, so a promoting pawn contributes as the queen it became.
        match ctx.incoming {
            Some(incoming) => {
                let in_idx = piece_to(incoming.landed, incoming.to);
                cv += W_CONT_2PLY
                    * i32::from(self.cont_2ply[cont_index(ctx.two_plies_ago, in_idx)].0);
                cv += W_CONT_4PLY
                    * i32::from(self.cont_4ply[cont_index(ctx.four_plies_ago, in_idx)].0);
            }
            None => cv += NO_INCOMING,
        }
        cv
    }

    /// Learns from the search's verdict at `pos`.
    ///
    /// `bonus` is the signed evaluation error in centipawns: positive when the
    /// search found the position *better* than the static evaluation said (so
    /// the cell should grow), negative when worse. The material families and the
    /// two continuation families are updated with different weights, matching
    /// Stockfish: the continuation families are the most specific signal and
    /// learn fastest.
    pub fn update(&mut self, pos: &Position, ctx: CorrectionCtx, bonus: i32) {
        let us = pos.turn() as usize;
        let bonus = BONUS_TRIM * bonus.clamp(-BONUS_LIMIT, BONUS_LIMIT) / 1024;

        let pawn = base_index(us, pos.keys.pawn);
        let minor = base_index(us, pos.keys.minor);
        let wnp = base_index(us, pos.keys.non_pawn[WHITE]);
        let bnp = base_index(us, pos.keys.non_pawn[BLACK]);

        self.base[pawn].pawn.update(bonus);
        self.base[minor].minor.update(bonus * 150 / 128);
        // Both non-pawn cells learn from the same bonus, and both are read by
        // the sum above: a position's non-pawn evaluation error is evidence
        // about both sides' non-pawn material, not only about the mover's.
        self.base[wnp]
            .non_pawn_white
            .update(bonus * NON_PAWN_WEIGHT / 128);
        self.base[bnp]
            .non_pawn_black
            .update(bonus * NON_PAWN_WEIGHT / 128);

        if let Some(incoming) = ctx.incoming {
            let in_idx = piece_to(incoming.landed, incoming.to);
            self.cont_2ply[cont_index(ctx.two_plies_ago, in_idx)].update(bonus * 130 / 128);
            self.cont_4ply[cont_index(ctx.four_plies_ago, in_idx)].update(bonus * 70 / 128);
        }
    }
}

/// Index of the material bundle for `us` in the bucket named by `key`.
#[inline]
fn base_index(us: usize, key: u64) -> usize {
    us * BASE_BUCKETS + (key as usize & (BASE_BUCKETS - 1))
}

/// `role * 64 + to`, the row of a continuation table.
#[inline]
fn piece_to(role: usize, to: Square) -> usize {
    role * 64 + to.to_usize()
}

/// Index of the continuation cell a selecting context and an incoming move meet
/// at.
///
/// `selecting` is `None` within the first two plies of the search, where no
/// earlier move picked a table; index 0 then names the same "no move has
/// happened yet" cell Stockfish's `&continuationCorrectionHistory[NO_PIECE][0]`
/// does, so root nodes still have a real, if uninformative, address to read and
/// write rather than being special-cased at every use site.
#[inline]
fn cont_index(selecting: Option<MoveCtx>, incoming: usize) -> usize {
    let sel = selecting.map_or(0, |c| piece_to(c.landed, c.to));
    sel * PIECE_TO + incoming
}

/// Clamps a static evaluation out of the mate zone.
///
/// The clamp is the safety property that makes the whole feature safe to feed
/// into pruning: whatever the tables hold, a corrected evaluation stays inside
/// the non-mate range, so no score this module produces is ever read as a forced
/// mate by [`crate::types::is_mate`] and a learned value can never masquerade as
/// a mate. This is the one hard invariant the whole feature rests on.
///
/// The bound is [`MAX_NON_MATE`] itself, matching Stockfish's
/// `VALUE_TB_WIN_IN_MAX_PLY - 1` / `VALUE_TB_LOSS_IN_MAX_PLY + 1`, which are
/// the magnitudes immediately outside its tablebase range. `is_mate` is a strict
/// `>`, so exactly ±24000 is already rejected and no extra margin is needed: an
/// earlier revision clamped to ±23999 "for safety", which bought nothing and
/// cost the boundary being the same number Stockfish uses.
#[inline]
pub fn clamp_non_mate(eval: i32) -> i32 {
    eval.clamp(-MAX_NON_MATE, MAX_NON_MATE)
}

/// Applies a correction to a raw static evaluation: `eval + cv / CV_TO_CP`,
/// clamped by [`clamp_non_mate`].
#[inline]
pub fn correct(eval: i32, cv: i32) -> i32 {
    clamp_non_mate(eval + cv / CV_TO_CP)
}

/// Whether a node's outcome is worth learning from, and by how much.
///
/// Learning is gated on three conditions, each of which exists to stop the
/// tables from learning noise:
///
/// * **not in check** — in check the static evaluation is a fallback, not a
///   claim about the position, so its "error" says nothing about the material;
/// * **the best move is not a capture** — a capture's outcome is explained by
///   the exchange itself, and the material families would absorb an error that
///   is really about one specific line;
/// * **the error points the same way as the node's outcome** — a node that found
///   nothing better than `alpha` (no best move) should have been judged *worse*
///   than the evaluation said, and a node that found a move above `beta` should
///   have been judged *better*. When the two disagree, the evaluation was
///   roughly right and nothing is learned.
///
/// The magnitude is the signed error scaled by depth — a deeper search knows
/// the position better, so its opinion counts for more — and by a larger factor
/// when a move was found, since those nodes are the ones that actually decided
/// something. The result is clamped to `±BONUS_LIMIT` before it is trimmed into
/// the per-family bonuses in [`CorrectionHistory::update`].
#[inline]
pub fn learn_bonus(
    in_check: bool,
    best_is_capture: bool,
    has_best_move: bool,
    score: i32,
    eval: i32,
    depth: i32,
) -> i32 {
    if in_check || best_is_capture {
        return 0;
    }
    if (score > eval) != has_best_move {
        return 0;
    }
    let factor = if has_best_move { 12 } else { 18 };
    ((score - eval) * depth * factor / 128).clamp(-BONUS_LIMIT, BONUS_LIMIT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::board::Role;

    fn fen(s: &str) -> Position {
        Position::from_fen(s).unwrap()
    }

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    /// The context of a node with no incoming move and no selecting moves.
    const NONE: CorrectionCtx = CorrectionCtx {
        incoming: None,
        two_plies_ago: None,
        four_plies_ago: None,
    };

    /// A context whose only populated slot is the incoming move.
    const fn ctx_only(incoming: MoveCtx) -> CorrectionCtx {
        CorrectionCtx {
            incoming: Some(incoming),
            two_plies_ago: None,
            four_plies_ago: None,
        }
    }

    #[test]
    fn a_fresh_table_is_neutral() {
        // Every cell is zero, so the weighted sum is zero and the correction
        // cannot move the evaluation. This is the property the search depends on
        // for its very first nodes.
        //
        // The one non-zero term is Stockfish's `NO_INCOMING` stand-in, which is
        // returned instead of reading a cell when the node has no incoming move.
        // 64049 / 131072 is 0.49, so it rounds away in `correct` and is invisible
        // to the search; it is carried here only because it is what Stockfish
        // returns and pretending otherwise would make the parity claim false.
        let h = CorrectionHistory::new();
        let pos = fen(START);
        assert_eq!(
            h.correction(&pos, NONE),
            NO_INCOMING,
            "only the no-incoming stand-in may contribute"
        );
        let cv = h.correction(&pos, ctx_only(ctx_at(Role::Pawn, Square::A1)));
        assert_eq!(cv, 0, "a node with an incoming move reads only empty cells");
        assert_eq!(correct(37, cv), 37, "and must not move the evaluation");
        assert_eq!(
            correct(37, NO_INCOMING),
            37,
            "the stand-in is sub-centipawn"
        );
    }

    #[test]
    fn indexing_stays_inside_the_tables() {
        // Sweep many positions, including mirrored ones and extreme material,
        // checking that both reads and writes stay in bounds. A masked index
        // cannot overflow, so the real risk is arithmetic on the *value*, which
        // the next tests cover.
        let mut h = CorrectionHistory::new();
        for f in [
            START,
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let pos = fen(f);
            h.update(&pos, NONE, 500);
            let _ = h.correction(&pos, NONE);
        }
    }

    #[test]
    fn colour_handling_separates_the_sides() {
        // Two mirrored positions: the same material, opposite side to move.
        // After teaching white's cells, white's correction must differ from
        // black's even though the material keys are identical.
        let white = fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
        let black = fen("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1");
        let mut h = CorrectionHistory::new();

        let before = h.correction(&white, NONE);
        h.update(&white, NONE, 800);

        assert!(
            h.correction(&white, NONE) > before,
            "the updated side must learn"
        );
        assert_eq!(
            h.correction(&black, NONE),
            NO_INCOMING,
            "the mirrored side must be untouched"
        );
    }

    #[test]
    fn bounded_updates_never_exceed_the_limit() {
        // Hammer one cell from both directions and prove the gravity rule
        // converges to (and never passes) the limit.
        let pos = fen(START);
        let mut h = CorrectionHistory::new();
        for _ in 0..100_000 {
            h.update(&pos, NONE, CORRECTION_LIMIT);
        }
        let saturated = h.correction(&pos, NONE);
        assert!(saturated > 0, "the cell must have learned a positive bias");
        for _ in 0..200_000 {
            h.update(&pos, NONE, -CORRECTION_LIMIT);
        }
        let reversed = h.correction(&pos, NONE);
        assert!(
            reversed < 0,
            "the cell must be able to learn a negative bias too"
        );

        // The extreme: a correction must never turn a sane evaluation into
        // something the search reads as a mate, whatever the tables hold.
        for cv in [i32::MIN / 4, -1_000_000, 0, 1_000_000, i32::MAX / 4] {
            for eval in [0, 100, -100, 5_000, -5_000, 30_000, -30_000] {
                let c = correct(eval, cv);
                assert!(
                    !crate::types::is_mate(c),
                    "correct({eval}, {cv}) = {c} reads as a mate"
                );
            }
        }
    }

    #[test]
    fn a_correction_never_produces_a_mate_score() {
        // The specific invariant pruning depends on: with any table content, a
        // correction is not decisive.
        let pos = fen(START);
        let mut h = CorrectionHistory::new();
        for i in 0..500 {
            h.update(&pos, NONE, if i % 2 == 0 { 1024 } else { -1024 });
            let cv = h.correction(&pos, NONE);
            for eval in [0i32, 1, -1, 1000, -1000, 25_000, -25_000] {
                let c = correct(eval, cv);
                assert!(
                    !crate::types::is_mate(c),
                    "correct({eval}, {cv}) = {c} looked like a mate"
                );
            }
        }

        // And the converse: the clamp must only ever bite at the mate boundary,
        // never near a real evaluation. `MAX_NON_MATE` is exactly the first
        // magnitude `is_mate` rejects (`is_mate` is a strict `>`), so it is itself
        // a legal value and the clamp may not move it.
        assert_eq!(MAX_NON_MATE, 24_000, "the documented boundary");
        assert!(
            !crate::types::is_mate(MAX_NON_MATE),
            "strict `>` at the bound"
        );
        assert!(crate::types::is_mate(MAX_NON_MATE + 1), "and one past it");
        assert_eq!(
            correct(MAX_NON_MATE, 0),
            MAX_NON_MATE,
            "the bound is not moved"
        );
        assert_eq!(correct(-MAX_NON_MATE, 0), -MAX_NON_MATE);
        assert_eq!(correct(MAX_NON_MATE + 1, 0), MAX_NON_MATE, "but past it is");
        assert_eq!(correct(-MAX_NON_MATE - 1, 0), -MAX_NON_MATE);
        // A deep but legitimate evaluation is passed through untouched: the
        // clamp is a mate-zone guard, not a compression of the score range.
        assert_eq!(correct(23_000, 0), 23_000, "a deep real eval must survive");
        assert_eq!(correct(-23_000, 0), -23_000);
        assert_eq!(correct(1_000, 0), 1_000, "and an ordinary one");
    }

    #[test]
    fn updates_are_deterministic_for_the_same_history_state() {
        // Two independently built tables fed the same sequence must agree
        // exactly, node for node. This is what makes a `Threads = 1` search
        // reproducible.
        let positions: Vec<Position> = [
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        .iter()
        .map(|f| Position::from_fen(f).unwrap())
        .collect();

        let mut a = CorrectionHistory::new();
        let mut b = CorrectionHistory::new();
        for round in 0..50i32 {
            for pos in &positions {
                let bonus = (round * 37 + (pos.hash.0 as i32 & 0xffff)) % 900 - 450;
                a.update(pos, NONE, bonus);
                b.update(pos, NONE, bonus);
                assert_eq!(a.correction(pos, NONE), b.correction(pos, NONE));
            }
        }
    }

    #[test]
    fn the_continuation_family_learns_its_own_context() {
        // Teach the two-ply family through a real move context and prove the
        // correction for a position reached by *that* move differs from the
        // same material reached without it.
        let root = fen("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1");
        let m = root.raw_move_from_uci("f3g5").unwrap();
        let ctx = MoveCtx::of(&root, m).unwrap();
        let child = root.make_child(m);

        let mut h = CorrectionHistory::new();
        assert_eq!(
            h.correction(&child, ctx_only(ctx)),
            0,
            "nothing learned yet"
        );
        h.update(&child, ctx_only(ctx), 700);
        let with_ctx = h.correction(&child, ctx_only(ctx));
        assert!(
            with_ctx > 0,
            "the context cell must have learned, got {with_ctx}"
        );

        // The update also touched the material families, so the total is not
        // zero without a context either. What matters is that dropping the
        // context strictly *loses* the continuation contribution: reading a
        // different family sum is not the same as reading the same sum.
        let without_ctx = h.correction(&child, NONE);
        assert!(
            without_ctx < with_ctx,
            "the continuation family must contribute: with {with_ctx}, without {without_ctx}"
        );

        // The cell is addressed by the role that moved and the square it
        // reached, so a different role on that same square is a different cell
        // and cannot pick up this evidence. (No promotion here, so `piece` and
        // `landed` are the same knight — the contrasting case is the promotion
        // test below.)
        let other = ctx_at(Role::Bishop, ctx.to);
        assert_ne!(
            h.correction(&child, ctx_only(other)),
            with_ctx,
            "a different role must address a different cell"
        );
    }

    /// A hand-built context for a move of `role` onto `to`, for tests that need a
    /// move that was never actually played.
    fn ctx_at(role: Role, to: Square) -> MoveCtx {
        MoveCtx {
            piece: role as usize,
            landed: role as usize,
            from: to,
            to,
        }
    }

    /// A context with all three slots filled, so a test can name one slot wrongly
    /// and observe the difference.
    fn full(incoming: MoveCtx, two: MoveCtx, four: MoveCtx) -> CorrectionCtx {
        CorrectionCtx {
            incoming: Some(incoming),
            two_plies_ago: Some(two),
            four_plies_ago: Some(four),
        }
    }

    #[test]
    fn the_two_continuation_horizons_select_their_own_tables() {
        // Stockfish reads two *different* chains — the one the move two plies ago
        // installed and the one the move four plies ago installed. An earlier
        // revision of this module addressed both with the same hashed older
        // context, which made the two horizons one horizon stored twice. The
        // property to protect is that evidence learned through the four-ply
        // selecting move is *not* reachable by naming the two-ply move instead.
        let root = fen("4k3/8/8/8/4N3/8/5N2/4K3 w - - 0 1");
        let incoming = MoveCtx::of(&root, root.raw_move_from_uci("f2g4").unwrap()).unwrap();
        let two = MoveCtx::of(&root, root.raw_move_from_uci("e4d6").unwrap()).unwrap();
        let four = MoveCtx::of(&root, root.raw_move_from_uci("f2h3").unwrap()).unwrap();
        let child = root.make_child(root.raw_move_from_uci("f2g4").unwrap());

        let mut h = CorrectionHistory::new();
        h.update(&child, full(incoming, two, four), 700);

        // Each horizon, read alone, must be answered by its own selecting move.
        let two_ply = h.correction(
            &child,
            CorrectionCtx {
                incoming: Some(incoming),
                two_plies_ago: Some(two),
                four_plies_ago: None,
            },
        );
        let four_ply_right = h.correction(
            &child,
            CorrectionCtx {
                incoming: Some(incoming),
                two_plies_ago: None,
                four_plies_ago: Some(four),
            },
        );
        // Naming the *two-ply* move in the four-ply slot must find nothing there.
        let four_ply_wrong = h.correction(
            &child,
            CorrectionCtx {
                incoming: Some(incoming),
                two_plies_ago: None,
                four_plies_ago: Some(two),
            },
        );
        assert!(
            two_ply > 0 && four_ply_right > 0,
            "both horizons must learn: {two_ply}, {four_ply_right}"
        );
        assert!(
            four_ply_wrong < four_ply_right,
            "the four-ply family must be selected by the four-ply move: \
             right {four_ply_right}, wrong {four_ply_wrong}"
        );

        // And symmetrically: the two-ply slot must not answer for the four-ply
        // move. Reading both horizons with the *same* selecting move would give a
        // number no single Stockfish read produces.
        let collapsed = h.correction(
            &child,
            CorrectionCtx {
                incoming: Some(incoming),
                two_plies_ago: Some(four),
                four_plies_ago: Some(four),
            },
        );
        assert_ne!(
            collapsed,
            h.correction(&child, full(incoming, two, four)),
            "one selecting move must not be able to address both horizons"
        );
    }

    #[test]
    fn a_promotion_is_keyed_by_the_promoted_role() {
        // Stockfish's correction key is `pos.piece_on(m.to_sq())` — the role
        // standing on the destination *after* the move — so b7b8q must be filed
        // under the queen's row, not the pawn's. Reading it back under the pawn
        // row must therefore find nothing.
        let root = fen("1n5k/P7/8/8/8/8/8/6K1 w - - 0 1");
        let m = root.raw_move_from_uci("a7a8q").unwrap();
        let ctx = MoveCtx::of(&root, m).unwrap();
        assert_eq!(ctx.piece, Role::Pawn as usize, "a pawn is what moved");
        assert_eq!(ctx.landed, Role::Queen as usize, "a queen is what arrived");

        let child = root.make_child(m);
        let mut h = CorrectionHistory::new();
        h.update(&child, ctx_only(ctx), 700);
        let as_queen = h.correction(&child, ctx_only(ctx));
        assert!(as_queen > 0, "the promotion cell must have learned");

        // The same move described with the pre-move role addresses the pawn row.
        let as_pawn = ctx_only(MoveCtx {
            piece: ctx.landed,
            landed: ctx.piece,
            from: ctx.from,
            to: ctx.to,
        });
        assert!(
            h.correction(&child, as_pawn) < as_queen,
            "the promotion must be filed under the promoted role"
        );
    }

    #[test]
    fn both_non_pawn_colours_are_learned_and_read() {
        // Stockfish keeps `nonPawnWhite` and `nonPawnBlack` as separate cells and
        // *sums* them under one weight. An earlier revision read only the mover's
        // own, halving the family. Both halves are checked here: the update must
        // reach both cells, and the read must consume both.
        let pos = fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let us = pos.turn() as usize;
        let mut h = CorrectionHistory::new();
        h.update(&pos, NONE, BONUS_LIMIT);

        let wnp = h.base[base_index(us, pos.keys.non_pawn[WHITE])].non_pawn_white;
        let bnp = h.base[base_index(us, pos.keys.non_pawn[BLACK])].non_pawn_black;
        assert_ne!(wnp, Cell::default(), "the white-keyed cell must learn");
        assert_ne!(bnp, Cell::default(), "the black-keyed cell must learn");

        let full = h.correction(&pos, NONE);
        // Zeroing either cell must remove exactly its own weight. If only one
        // colour were read, zeroing the other would change nothing.
        h.base[base_index(us, pos.keys.non_pawn[BLACK])].non_pawn_black = Cell::default();
        let without_black = h.correction(&pos, NONE);
        assert_eq!(
            full - without_black,
            W_NON_PAWN * i32::from(bnp.0),
            "the black-keyed cell must be read"
        );

        h.base[base_index(us, pos.keys.non_pawn[WHITE])].non_pawn_white = Cell::default();
        let without_white = h.correction(&pos, NONE);
        assert_eq!(
            without_black - without_white,
            W_NON_PAWN * i32::from(wnp.0),
            "the white-keyed cell must be read"
        );
    }

    #[test]
    fn correction_magnitude_is_bounded_by_construction() {
        // With every cell saturated, the sum must stay inside the documented
        // envelope: a few hundred centipawns.
        let pos = fen(START);
        let mut h = CorrectionHistory::new();
        for _ in 0..50_000 {
            h.update(&pos, NONE, CORRECTION_LIMIT);
        }
        let cv = h.correction(&pos, NONE);
        let cp = cv / CV_TO_CP;
        // The theoretical maximum, computed from the weights rather than
        // remembered: every cell saturates at ±CORRECTION_LIMIT, and a family
        // contributes its weight times its cell count. The pawn and minor
        // families are one cell each; the non-pawn family is two (white-keyed
        // and black-keyed, summed); the continuation family is two (two-ply and
        // four-ply), each carrying the full 8761.
        let ceiling = CORRECTION_LIMIT
            * (W_PAWN + W_MINOR + 2 * W_NON_PAWN + W_CONT_2PLY + W_CONT_4PLY)
            / CV_TO_CP;
        assert_eq!(ceiling, 540, "the documented saturated envelope");
        assert!(
            cp <= ceiling,
            "a real correction must stay under the envelope: {cp} > {ceiling}"
        );

        // The ceiling is reachable, but only where the material keys alias into
        // one or two cells. Measured over a real single-thread search, the
        // largest |cv| actually read is 83 cp (kiwipete depth 11), 103 cp (the
        // start position at depth 13), 216 cp (kiwipete depth 14) and 246 cp (a
        // pawn endgame at depth 14) — and 540 cp, the ceiling exactly, in a bare
        // KQ-vs-K endgame, which has no pawns and no minor pieces and therefore
        // funnels its pawn and minor families into a single global cell each. So
        // the binding case for the bound is a sparse endgame, not a middlegame,
        // and 540 cp is the number the clamp has to tolerate.
    }

    #[test]
    fn clearing_returns_to_neutral() {
        let pos = fen(START);
        let mut h = CorrectionHistory::new();
        for _ in 0..1000 {
            h.update(&pos, NONE, 500);
        }
        assert!(h.correction(&pos, NONE) > 0);
        h.clear();
        assert_eq!(h.correction(&pos, NONE), NO_INCOMING);
    }

    #[test]
    fn the_learning_bonus_is_the_observed_error() {
        // A node that found a best move well above what the evaluation claimed
        // teaches a positive correction, scaled by depth.
        assert_eq!(learn_bonus(false, false, true, 100, 100, 6), 0, "no error");
        assert!(
            learn_bonus(false, false, true, 400, 100, 6) > 0,
            "found a move"
        );
        assert!(
            learn_bonus(false, false, false, -400, 100, 6) < 0,
            "found nothing"
        );

        // A deeper node is worth more: the same error teaches more at depth 12
        // than at depth 4.
        let shallow = learn_bonus(false, false, true, 200, 100, 4);
        let deep = learn_bonus(false, false, true, 200, 100, 12);
        assert!(deep > shallow, "deeper searches know the position better");

        // The gates: in check, a capture, or an error pointing the wrong way all
        // mean "learn nothing".
        assert_eq!(learn_bonus(true, false, true, 400, 100, 8), 0, "in check");
        assert_eq!(learn_bonus(false, true, true, 400, 100, 8), 0, "capture");
        // Found a move but the score is *below* the evaluation: contradictory.
        assert_eq!(learn_bonus(false, false, true, 50, 100, 8), 0, "wrong way");
        // Found nothing but the score is *above* the evaluation: contradictory.
        assert_eq!(
            learn_bonus(false, false, false, 400, 100, 8),
            0,
            "wrong way"
        );

        // Never beyond the cell limit, however large the error.
        for (score, eval) in [
            (30_000i32, -30_000i32),
            (-30_000, 30_000),
            (0, i32::MIN / 2),
        ] {
            for d in 1..=40 {
                let b = learn_bonus(false, false, true, score, eval, d);
                assert!(
                    (-CORRECTION_LIMIT..=CORRECTION_LIMIT).contains(&b),
                    "bonus {b} escaped the limit"
                );
            }
        }
    }
}

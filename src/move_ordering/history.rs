//! History heuristics: how quiet moves are ordered and learned from.
//!
//! Three complementary tables, all fed from the same signal (a quiet move
//! that improves/fails around `beta` on the search path), following the
//! classic Stockfish-style gravity update:
//!
//! ```text
//! entry += bonus - entry * |bonus| / 16384
//! ```
//!
//! * **Main history** — per side to move, keyed by the move's `(from, to)`.
//! * **Countermoves** — per side to move, keyed by the *previous* move's
//!   `(from, to)`: the best response found so far to that move.
//! * **Continuation history** — per `(piece, to)` of the move, keyed by a
//!   hash of the previous two moves' `(piece, to)` contexts, so a quiet move
//!   is judged in the context of the line just played.
//! * **Capture history** — per side, keyed by the `(victim, attacker)` pair:
//!   captures that produce beta cutoffs (or fail) teach the ordering which
//!   exchanges are (not) profitable, breaking MVV-LVA ties between captures
//!   of the same victim.
//!
//! Values are stored as `i16` and clamped to `±HIST_MAX`; reads widen to
//! `i32`. This keeps a full table set around 2 MB, cheap to copy per thread.

use shakmaty::{Board, Color, Role, Square};

use crate::types::RawMove;

/// Symmetric clamp for stored history values.
pub const HIST_MAX: i32 = 32_767;

/// `from * 64 + to` row length: every ordered (from, to) pair.
const FROM_TO: usize = 64 * 64;

/// `piece * 64 + to` row length: 7 role slots (indexed by `Role as usize`,
/// slot 0 unused) times 64 target squares.
const PIECE_TO: usize = 7 * 64;

/// Power-of-two bucket count for the continuation-history context hash.
const CMH_BUCKETS: usize = 1024;

/// Capture-history row length: an `8 * 8` (victim, attacker) grid per side —
/// victim 0 means a pure promotion (no captured piece), victims/attackers are
/// `Role as usize` (1..=6). `2 * 64` i16 entries is 256 bytes.
const CAPTURE_GRID: usize = 8 * 8;

/// Capture-history stored values are scaled down by this divisor on read so a
/// modest handful of rewards cannot drown the MVV-LVA tier (see
/// [`History::capture_adjustment`]).
pub const CAP_HIST_DIV: i32 = 4;

/// Hard clamp on the capture-history ordering adjustment. The value bounds
/// history to a tiebreak role: it can decide between otherwise-equal
/// captures of the same victim, and it can never lift a pawn-victim capture
/// above a rook- or queen-victim capture even when both are at their
/// extreme adjustment values. (Adjacent minor tiers — knight vs bishop —
/// already overlap in MVV-LVA terms because the attacker-value spread
/// exceeds the victim gap, so no clamp can make that pair a total order;
/// history is free to decide there, which is the intended learning signal.)
pub const CAP_HIST_MAX_ADJ: i32 = 2047;

/// Context of a played move on the search path: the moving piece's role
/// (indexed as `Role as usize`) plus its origin and destination squares.
///
/// The search records one of these per ply as it walks down a line; the
/// previous move's `(from, to)` keys the countermove table and the previous
/// two `(piece, to)` contexts key the continuation table.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MoveCtx {
    pub piece: usize,
    pub from: Square,
    pub to: Square,
}

impl MoveCtx {
    /// Builds the context for a move played from `pos` (the position *before*
    /// the move, so the moving piece can be read).
    #[inline]
    pub fn of(pos: &crate::board::Position, m: RawMove) -> Option<MoveCtx> {
        pos.board().piece_at(m.from()).map(|p| MoveCtx {
            piece: p.role as usize,
            from: m.from(),
            to: m.to(),
        })
    }
}

/// The sign-carrying bonus for a remaining depth: positive rewards use
/// `+bonus(d)`, failed moves use `-bonus(d)`.
#[inline]
pub const fn bonus(depth_remaining: i32) -> i32 {
    depth_remaining * depth_remaining
}

/// Gravity update: `entry += bonus - entry * |bonus| / 16384`, clamped.
#[inline]
fn gravity(entry: &mut i16, bonus: i32) {
    let cur = i32::from(*entry);
    let next = cur + bonus - cur * bonus.abs() / 16_384;
    *entry = next.clamp(-HIST_MAX, HIST_MAX) as i16;
}

/// Moves ordered by history heuristics. Owned by the search (never global);
/// one set per thread keeps `Threads = 1` fully deterministic.
///
/// The tables are heap-allocated flat slices: `History` is a handful of words
/// on the stack, which matters because search threads own one each. Indexing:
/// `main[side * FROM_TO + from_to]`,
/// `counter[side * FROM_TO + from_to]`,
/// `continuation[side * PIECE_TO * CMH_BUCKETS + piece_to * CMH_BUCKETS + bucket]`.
#[derive(Clone, Debug)]
pub struct History {
    /// `[side][from * 64 + to]` quiet-move scores.
    main: Box<[i16]>,
    /// `[side][prev_from * 64 + prev_to]` best response to the previous move.
    counter: Box<[RawMove]>,
    /// `[side][piece * 64 + to][context_bucket]` continuation scores.
    continuation: Box<[i16]>,
    /// `[side][victim * 8 + attacker]` capture scores.
    capture: Box<[i16]>,
}

impl Default for History {
    fn default() -> Self {
        History::new()
    }
}

impl History {
    pub fn new() -> History {
        History {
            main: vec![0i16; 2 * FROM_TO].into_boxed_slice(),
            counter: vec![RawMove::NULL; 2 * FROM_TO].into_boxed_slice(),
            continuation: vec![0i16; 2 * PIECE_TO * CMH_BUCKETS].into_boxed_slice(),
            capture: vec![0i16; 2 * CAPTURE_GRID].into_boxed_slice(),
        }
    }

    /// Zeroes every table (`ucinewgame`, or a fresh search on a new position).
    pub fn clear(&mut self) {
        self.main.fill(0);
        self.counter.fill(RawMove::NULL);
        self.continuation.fill(0);
        self.capture.fill(0);
    }

    /// Clears only the countermove table (kept cheap for `setoption`-driven
    /// resets that should not lose long-term history).
    pub fn clear_counter(&mut self) {
        self.counter.fill(RawMove::NULL);
    }

    // --- reads -------------------------------------------------------------

    /// Main-history value of a quiet move for the side to move.
    #[inline]
    pub fn history_score(&self, side: Color, m: RawMove) -> i32 {
        i32::from(self.main[side as usize * FROM_TO + from_to(m.from(), m.to())])
    }

    /// The countermove recorded for `prev` (the move just played by the
    /// opponent). `NULL` when none is known.
    #[inline]
    pub fn counter_move(&self, side: Color, prev: Option<MoveCtx>) -> RawMove {
        match prev {
            Some(p) => self.counter[side as usize * FROM_TO + from_to(p.from, p.to)],
            None => RawMove::NULL,
        }
    }

    /// Continuation-history value of a quiet move `(piece -> to)` given the
    /// previous two move contexts.
    #[inline]
    pub fn continuation_score(
        &self,
        side: Color,
        piece: usize,
        to: Square,
        prev: Option<MoveCtx>,
        ant: Option<MoveCtx>,
    ) -> i32 {
        let bucket = cont_bucket(prev, ant);
        i32::from(self.continuation[cont_index(side, piece, to, bucket)])
    }

    // --- updates -----------------------------------------------------------

    /// Rewards/penalizes a quiet move for the side to move (`bonus` as given
    /// by [`bonus`], possibly negated for failed moves).
    #[inline]
    pub fn update_history(&mut self, side: Color, m: RawMove, bonus_pts: i32) {
        let idx = side as usize * FROM_TO + from_to(m.from(), m.to());
        gravity(&mut self.main[idx], bonus_pts);
    }

    /// Records the best response found to `prev` (may be any move).
    #[inline]
    pub fn set_counter_move(&mut self, side: Color, prev: Option<MoveCtx>, m: RawMove) {
        if let Some(p) = prev {
            self.counter[side as usize * FROM_TO + from_to(p.from, p.to)] = m;
        }
    }

    /// Updates the continuation entry for `(piece -> to)` in the context of
    /// the previous two moves.
    #[inline]
    pub fn update_continuation(
        &mut self,
        side: Color,
        piece: usize,
        to: Square,
        prev: Option<MoveCtx>,
        ant: Option<MoveCtx>,
        bonus_pts: i32,
    ) {
        let bucket = cont_bucket(prev, ant);
        let idx = cont_index(side, piece, to, bucket);
        gravity(&mut self.continuation[idx], bonus_pts);
    }

    /// Ordering adjustment for a capture/promotion, from the capture-history
    /// table. The stored value is scaled by [`CAP_HIST_DIV`] and clamped to
    /// `±[CAP_HIST_MAX_ADJ]` so MVV-LVA remains the dominant signal: history
    /// breaks ties between exchanges of the same victim and can only nudge
    /// captures of nearby victim tiers. The mover's color selects the side
    /// slot (matching [`History::update_capture`]).
    #[inline]
    pub fn capture_adjustment(&self, board: &Board, m: RawMove) -> i32 {
        let side = match board.piece_at(m.from()) {
            Some(p) => p.color as usize,
            None => 0,
        };
        let v =
            i32::from(self.capture[side * CAPTURE_GRID + capture_index(board, m)]) / CAP_HIST_DIV;
        v.clamp(-CAP_HIST_MAX_ADJ, CAP_HIST_MAX_ADJ)
    }

    /// Rewards/penalizes a capture or promotion for the side to move
    /// (`bonus_pts` as given by [`bonus`], possibly negated for failed moves).
    #[inline]
    pub fn update_capture(&mut self, board: &Board, m: RawMove, bonus_pts: i32) {
        // Victim 0 (pure promotion) slots exist but are never rewarded: a
        // promotion's value already dominates its ordering tier.
        if victim_role(board, m) == 0 {
            return;
        }
        let side = match board.piece_at(m.from()) {
            Some(p) => p.color,
            None => return,
        };
        let idx = side as usize * CAPTURE_GRID + capture_index(board, m);
        gravity(&mut self.capture[idx], bonus_pts);
    }
}

/// The victim role as a capture-history grid index: a pawn for en passant,
/// the piece on `to` for a capture, or 0 for a pure promotion (no victim).
#[inline]
fn victim_role(board: &Board, m: RawMove) -> usize {
    if m.is_en_passant() {
        Role::Pawn as usize
    } else {
        match board.role_at(m.to()) {
            Some(role) => role as usize,
            None => 0,
        }
    }
}

/// `victim * 8 + attacker` grid index for a capture/promotion move.
#[inline]
fn capture_index(board: &Board, m: RawMove) -> usize {
    let victim = victim_role(board, m);
    let attacker = board.piece_at(m.from()).map_or(0, |p| p.role as usize);
    victim * 8 + attacker
}

#[inline]
fn cont_index(side: Color, piece: usize, to: Square, bucket: usize) -> usize {
    side as usize * (PIECE_TO * CMH_BUCKETS) + (piece * 64 + to.to_usize()) * CMH_BUCKETS + bucket
}

#[inline]
fn from_to(from: Square, to: Square) -> usize {
    from.to_usize() * 64 + to.to_usize()
}

/// Hashes the previous two move contexts into a `CMH_BUCKETS`-sized bucket.
#[inline]
fn cont_bucket(prev: Option<MoveCtx>, ant: Option<MoveCtx>) -> usize {
    let a = prev.map_or(0, ctx_key);
    let b = ant.map_or(0, ctx_key);
    let h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    ((h >> 32) as usize) & (CMH_BUCKETS - 1)
}

#[inline]
fn ctx_key(ctx: MoveCtx) -> u64 {
    (ctx.piece as u64) * 64 + ctx.to.to_usize() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use shakmaty::Role;

    fn mv(from: Square, to: Square) -> RawMove {
        RawMove::new(from, to, 0)
    }

    fn ctx(fen: &str, uci: &str) -> Option<MoveCtx> {
        let p = Position::from_fen(fen).unwrap();
        MoveCtx::of(&p, p.raw_move_from_uci(uci).unwrap())
    }

    #[test]
    fn history_grows_on_reward_and_clamps() {
        let mut h = History::new();
        let m = mv(Square::E2, Square::E4);
        for _ in 0..100 {
            h.update_history(Color::White, m, bonus(16)); // 256
        }
        let v = h.history_score(Color::White, m);
        assert!(v > 0 && v <= HIST_MAX, "got {v}");
        assert!(v < 15_000, "gravity converges, not saturates: {v}");
    }

    #[test]
    fn penalty_is_symmetric_in_sign() {
        let mut h = History::new();
        let m_bad = mv(Square::A2, Square::A3);
        let m_good = mv(Square::B2, Square::B3);
        h.update_history(Color::Black, m_good, bonus(10));
        h.update_history(Color::Black, m_bad, -bonus(10));
        assert!(h.history_score(Color::Black, m_good) > 0);
        assert!(h.history_score(Color::Black, m_bad) < 0);
    }

    #[test]
    fn history_is_per_side() {
        let mut h = History::new();
        let m = mv(Square::E2, Square::E4);
        h.update_history(Color::White, m, bonus(8));
        assert_eq!(h.history_score(Color::Black, m), 0);
    }

    #[test]
    fn countermove_round_trip() {
        let mut h = History::new();
        let prev = ctx(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4",
        )
        .unwrap();
        let resp = mv(Square::E7, Square::E5);
        assert_eq!(h.counter_move(Color::Black, Some(prev)), RawMove::NULL);
        h.set_counter_move(Color::Black, Some(prev), resp);
        assert_eq!(h.counter_move(Color::Black, Some(prev)), resp);
        // A different previous move has no countermove.
        let other = ctx(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "d2d4",
        )
        .unwrap();
        assert_eq!(h.counter_move(Color::Black, Some(other)), RawMove::NULL);
        // Black's counter table and White's are separate.
        assert_eq!(h.counter_move(Color::White, Some(prev)), RawMove::NULL);
    }

    #[test]
    fn continuation_context_bucketing_is_deterministic() {
        let mut h = History::new();
        let prev = ctx(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "e7e5",
        )
        .unwrap();
        let ant = ctx(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4",
        )
        .unwrap();
        let to = Square::G1;
        let piece = Role::Knight as usize;

        h.update_continuation(Color::Black, piece, to, Some(prev), Some(ant), bonus(12));
        let v1 = h.continuation_score(Color::Black, piece, to, Some(prev), Some(ant));
        let v2 = h.continuation_score(Color::Black, piece, to, Some(prev), Some(ant));
        assert_eq!(v1, v2, "same context must hit the same bucket");
        assert!(v1 > 0);

        // A different previous move usually lands elsewhere; at minimum the
        // lookup must not panic and stay in range.
        let other = ctx(
            "rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            "e7e6",
        )
        .unwrap();
        let _ = h.continuation_score(Color::Black, piece, to, Some(other), Some(ant));
    }

    #[test]
    fn clear_zeroes_everything() {
        let mut h = History::new();
        let m = mv(Square::E2, Square::E4);
        let prev = ctx(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "d2d4",
        )
        .unwrap();
        h.update_history(Color::White, m, bonus(10));
        h.set_counter_move(Color::Black, Some(prev), m);
        h.clear();
        assert_eq!(h.history_score(Color::White, m), 0);
        assert_eq!(h.counter_move(Color::Black, Some(prev)), RawMove::NULL);
        assert_eq!(
            h.continuation_score(Color::White, 2, Square::F3, None, None),
            0
        );
    }

    // --- capture history ---------------------------------------------------

    fn cap_move(fen: &str, uci: &str) -> (Position, RawMove) {
        let p = Position::from_fen(fen).unwrap();
        let m = p.raw_move_from_uci(uci).unwrap();
        (p, m)
    }

    #[test]
    fn capture_history_rewards_and_penalizes() {
        let mut h = History::new();
        // Rxc2: white rook takes the black queen on c2.
        let (pos, m) = cap_move("6k1/8/8/8/8/8/2q5/2R3K1 w - - 0 1", "c1c2");
        assert_eq!(h.capture_adjustment(pos.board(), m), 0);
        h.update_capture(pos.board(), m, bonus(8));
        let rewarded = h.capture_adjustment(pos.board(), m);
        assert!(rewarded > 0, "reward must lift the adjustment: {rewarded}");
        h.update_capture(pos.board(), m, -bonus(8));
        assert!(
            h.capture_adjustment(pos.board(), m) < rewarded,
            "penalty must lower the adjustment"
        );
    }

    #[test]
    fn capture_history_is_per_side() {
        let mut h = History::new();
        // Both moves are "rook at a1 captures queen on a2", one per side —
        // the same (victim, attacker) grid index, different side slots.
        let (w, wm) = cap_move("6k1/8/8/8/8/8/r7/Q3K3 w - - 0 1", "a1a2");
        let (b, bm) = cap_move("6k1/8/8/8/8/8/q7/R3K3 b - - 0 1", "a2a1");
        h.update_capture(w.board(), wm, bonus(12));
        assert!(h.capture_adjustment(w.board(), wm) > 0, "white must learn");
        assert_eq!(
            h.capture_adjustment(b.board(), bm),
            0,
            "black's table must be untouched"
        );
    }

    #[test]
    fn capture_history_can_decide_same_victim_ties() {
        // Two captures of the same queen with different attackers: MVV-LVA
        // prefers the knight, but a saturated reward for the rook exchange
        // must let history override that preference. (Identical attacker
        // pairs share the (victim, attacker) slot by design.)
        let (pos, rook) = cap_move("6k1/8/8/8/1N6/8/2q5/2R3K1 w - - 0 1", "c1c2");
        let knight = pos.raw_move_from_uci("b4c2").unwrap();
        let mut h = History::new();
        for _ in 0..1000 {
            h.update_capture(pos.board(), rook, bonus(10));
        }
        assert_eq!(
            h.capture_adjustment(pos.board(), rook),
            CAP_HIST_MAX_ADJ,
            "saturation must hit the clamp"
        );
        let rook_score = crate::move_ordering::tt_move::capture_score(pos.board(), rook)
            + h.capture_adjustment(pos.board(), rook);
        let knight_score = crate::move_ordering::tt_move::capture_score(pos.board(), knight);
        assert!(
            rook_score > knight_score,
            "history must be able to decide same-victim ties: {rook_score} vs {knight_score}"
        );
    }

    #[test]
    fn capture_history_cannot_cross_wide_victim_tiers() {
        // Saturate a pawn-victim capture's history: its score with the maximum
        // bonus still ranks below a queen-victim capture with the maximum
        // penalty, so history can never flip widely separated victim tiers.
        let (pawn_pos, pm) = cap_move("6k1/8/8/8/8/4n3/3P4/6K1 w - - 0 1", "d2e3");
        let (queen_pos, qm) = cap_move("6k1/8/8/8/8/8/2q5/2R3K1 w - - 0 1", "c1c2");
        let mut h = History::new();
        for _ in 0..1000 {
            h.update_capture(pawn_pos.board(), pm, bonus(10));
            h.update_capture(queen_pos.board(), qm, -bonus(10));
        }
        let pawn_score = crate::move_ordering::tt_move::capture_score(pawn_pos.board(), pm)
            + h.capture_adjustment(pawn_pos.board(), pm);
        let queen_score = crate::move_ordering::tt_move::capture_score(queen_pos.board(), qm)
            + h.capture_adjustment(queen_pos.board(), qm);
        // Sanity: the reward really is at its bound by now.
        assert_eq!(h.capture_adjustment(pawn_pos.board(), pm), CAP_HIST_MAX_ADJ);
        assert_eq!(
            h.capture_adjustment(queen_pos.board(), qm),
            -CAP_HIST_MAX_ADJ
        );
        assert!(
            queen_score > pawn_score,
            "pawn-victim capture can never outrank a queen-victim capture: \
             {pawn_score} vs {queen_score}"
        );
    }

    #[test]
    fn capture_history_en_passant_counts_as_a_pawn_victim() {
        // e5xd6 e.p. captures the d5 pawn; it must reward the same grid slot
        // as any pawn capture even though role_at(to) is empty.
        let (pos, m) = cap_move("6k1/8/8/3pP3/8/8/8/6K1 w - d6 0 1", "e5d6");
        let mut h = History::new();
        h.update_capture(pos.board(), m, bonus(10));
        assert!(h.capture_adjustment(pos.board(), m) > 0);
    }

    #[test]
    fn capture_history_ignores_pure_promotions() {
        // A promotion without a capture has no victim: it must not write a
        // slot (its value already dominates the ordering tier). The white
        // pawn on d7 promotes by moving to the empty d8 square.
        let (pos, m) = cap_move("6k1/3P4/8/8/8/8/8/6K1 w - - 0 1", "d7d8q");
        let mut h = History::new();
        h.update_capture(pos.board(), m, bonus(10));
        assert_eq!(h.capture_adjustment(pos.board(), m), 0);
    }
}

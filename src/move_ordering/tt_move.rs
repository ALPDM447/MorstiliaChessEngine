//! Transposition-table move handling and move-class scoring tiers.
//!
//! The TT move is the strongest ordering cue the engine has: a move that
//! produced a cutoff on a previous visit to the same position is tried first
//! at every re-visit. Captures/promotions come next, ordered among themselves
//! by MVV-LVA (most valuable victim, least valuable attacker); only then are
//! quiet moves considered (see [`crate::move_ordering::mod`] for their
//! history-based scoring).
//!
//! The scoring tiers live here so both the ordering code and the search
//! (which decides what to store into the TT) agree on the same constants.

use shakmaty::{Board, Role};

use crate::evaluation::params::EvalParams;
use crate::types::RawMove;

/// Ordering tier for the TT move — always first, well above everything else.
pub const TT_TIER: i32 = 2_000_000;

/// Ordering tier for captures and promotions; the MVV-LVA score sits in
/// `[0, ~21_000]` on top of this base.
pub const CAPTURE_TIER: i32 = 1_000_000;

/// Ordering tier for the first killer move.
pub const KILLER1_TIER: i32 = 800_000;

/// Ordering tier for the second killer move.
pub const KILLER2_TIER: i32 = 799_999;

/// The captured piece's value in the MVV-LVA sense: the piece on `to`, or a
/// pawn for en passant, or 0 for a pure promotion. Reads the tunable material
/// values from `p` (kept consistent with SEE and the evaluation).
#[inline]
pub fn victim_value(board: &Board, m: RawMove, p: &EvalParams) -> i32 {
    if m.is_en_passant() {
        p.piece_value(Role::Pawn)
    } else {
        match board.role_at(m.to()) {
            Some(role) => p.piece_value(role),
            None => 0,
        }
    }
}

/// MVV-LVA score for any capture or promotion. Victim value dominates
/// (scaled by 16); among equal victims the cheapest attacker is preferred.
/// A promotion adds `(promoted − pawn) * 16` so an underpromotion that
/// captures is still ranked sensibly against a lone queening move.
#[inline]
pub fn capture_score(board: &Board, m: RawMove, p: &EvalParams) -> i32 {
    let victim = victim_value(board, m, p);
    let attacker = board
        .piece_at(m.from())
        .map_or(0, |piece| p.piece_value(piece.role));
    let mut score = victim * 16 - attacker;
    if let Some(promo) = m.promotion() {
        score += (p.piece_value(promo) - p.piece_value(Role::Pawn)) * 16;
    }
    score
}

/// True when `m` is a capture (including en passant) or a promotion — the
/// moves that are ordered before quiet moves and whose MVV-LVA score applies.
#[inline]
pub fn is_capture_or_promotion(board: &Board, m: RawMove) -> bool {
    m.is_promotion() || m.is_en_passant() || board.role_at(m.to()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;

    fn pos(fen: &str) -> Position {
        Position::from_fen(fen).unwrap()
    }

    fn params() -> EvalParams {
        EvalParams::default()
    }

    fn score(fen: &str, uci: &str) -> i32 {
        let p = pos(fen);
        let m = p.raw_move_from_uci(uci).unwrap();
        capture_score(p.board(), m, &params())
    }

    #[test]
    fn pawn_takes_queen_outranks_queen_takes_pawn() {
        // White pawn d2 captures the queen on e3 (diagonal capture);
        // white queen c1 captures the pawn on d2.
        let pawn_takes = score("6k1/8/8/8/8/4q3/3P4/2K5 w - - 0 1", "d2e3");
        let queen_takes = score("6k1/8/8/8/8/8/3p4/2Q3K1 w - - 0 1", "c1d2");
        assert!(pawn_takes > queen_takes, "{pawn_takes} vs {queen_takes}");
    }

    #[test]
    fn cheapest_attacker_wins_tiebreak() {
        // Both a rook (e1e3) and a pawn (d2e3) can capture the same knight in
        // otherwise identical positions: the pawn ranks higher (same victim,
        // cheaper attacker).
        let rook = score("6k1/8/8/8/8/4n3/8/4R1K1 w - - 0 1", "e1e3");
        let pawn = score("6k1/8/8/8/8/4n3/3P4/6K1 w - - 0 1", "d2e3");
        assert!(pawn > rook, "{pawn} vs {rook}");
    }

    #[test]
    fn en_passant_victim_is_a_pawn() {
        let p = pos("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        let m = p.raw_move_from_uci("e5f6").unwrap();
        assert!(m.is_en_passant());
        assert_eq!(victim_value(p.board(), m, &params()), 100);
        assert!(is_capture_or_promotion(p.board(), m));
    }

    #[test]
    fn promotion_advertises_promoted_value() {
        let promo = score("7k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q");
        let promo_capture = score("3r2k1/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8q");
        assert!(
            promo_capture >= promo,
            "capturing promo ranks at least as high"
        );
        assert!(promo > 10_000, "queen promo dominates small captures");
    }

    #[test]
    fn tiers_are_ordered() {
        assert!(TT_TIER > CAPTURE_TIER);
        assert!(CAPTURE_TIER > KILLER1_TIER);
        assert!(KILLER1_TIER > KILLER2_TIER);
    }
}

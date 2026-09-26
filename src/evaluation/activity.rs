//! Piece activity, coordination and positional piece features.
//!
//! Everything here is *in addition to* the piece-square tables in `pieces`:
//!
//! * **Outposts** — a knight/bishop on a square an own pawn defends, that the
//!   enemy cannot attack with a pawn, in the enemy camp. Such pieces are
//!   immune to pawn harassment and dominate the sector.
//! * **Bad bishop** — a bishop blocked by its own pawns on the same colour
//!   complex is a standing liability.
//! * **Development** — in the opening, a minor still on its home square is
//!   undeveloped; the penalty pushes the engine to bring the pieces out.
//! * **Connected rooks** — both rooks on the same clearly-open rank: they
//!   co-operate down the files and up the following ranks.
//! * **Rook behind a passed pawn** — a rook backing its own passer from
//!   behind, or hunting an enemy passer from the far side, is the classic
//!   endgame force multiplier.

use shakmaty::{Bitboard, Board, Color, Role, Square};

use crate::evaluation::Score;
use crate::evaluation::params::EvalParams;
use crate::evaluation::passed_pawns::PassedInfo;
use crate::evaluation::pawns::{PawnInfo, file_mask, rank_mask};

/// Light squares of the board (a1 is dark, so bit 0 is clear).
const LIGHT_SQUARES: u64 = 0xAA55_AA55_AA55_AA55;

/// Home squares of both knights and both bishops (white back rank).
const WHITE_MINOR_HOME: u64 = (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6); // b1 c1 f1 g1
/// ...and the black back rank (vertical mirror).
const BLACK_MINOR_HOME: u64 = WHITE_MINOR_HOME << 56;

/// Evaluates piece activity/coordination for both colors (white minus black).
/// All weights come from `p`; `phase` gates the opening-only development term.
pub fn evaluate_pieces(
    board: &Board,
    info: &PawnInfo,
    passed: &PassedInfo,
    phase: i32,
    p: &EvalParams,
) -> Score {
    let mut score = Score::zero();
    let occupied = board.occupied();

    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let mut color_score = Score::zero();
        let enemy_pawn_attacks = info.pawn_attacks_of(!color);

        // --- Outposts -----------------------------------------------------
        // Enemy camp ranks for `color` (the opponent's half of the board).
        let camp = if color == Color::White {
            rank_mask(4) | rank_mask(5) // ranks 5-6
        } else {
            rank_mask(1) | rank_mask(2) // ranks 2-3
        };
        for (role, bonus_arr) in [
            (Role::Knight, &p.knight_outpost),
            (Role::Bishop, &p.bishop_outpost),
        ] {
            board.by_piece(role.of(color)).for_each(|sq| {
                if (info.pawn_attacks_of(color) & Bitboard::from_square(sq)).any()
                    && (enemy_pawn_attacks & Bitboard::from_square(sq)).is_empty()
                    && (Bitboard::from_square(sq) & Bitboard(camp)).any()
                {
                    color_score.mg += bonus_arr[0];
                    color_score.eg += bonus_arr[1];
                }
            });
        }

        // --- Bad bishop ----------------------------------------------------
        board.by_piece(Role::Bishop.of(color)).for_each(|sq| {
            let complex = if is_light(sq) {
                LIGHT_SQUARES
            } else {
                !LIGHT_SQUARES
            };
            let same_colour_pawns = (info.pawns_of(color) & Bitboard(complex)).count() as i32;
            let excess = (same_colour_pawns - 1).max(0).min(p.bad_bishop_cap);
            color_score.mg += p.bad_bishop_pawn[0] * excess;
            color_score.eg += p.bad_bishop_pawn[1] * excess;
        });

        // --- Development (opening only) -------------------------------------
        if phase >= p.development_phase {
            let home = if color == Color::White {
                WHITE_MINOR_HOME
            } else {
                BLACK_MINOR_HOME
            };
            let minors_home = (board.by_piece(Role::Knight.of(color))
                | board.by_piece(Role::Bishop.of(color)))
                & Bitboard(home);
            color_score.mg += p.undeveloped[0] * minors_home.count() as i32;
            color_score.eg += p.undeveloped[1] * minors_home.count() as i32;
        }

        // --- Rooks connected ------------------------------------------------
        let rooks = board.by_piece(Role::Rook.of(color));
        if rooks.count() == 2 {
            let r1 = rooks.first().expect("two rooks");
            let r2 = rooks.last().expect("two rooks");
            if r1.rank() == r2.rank() {
                let (a, b) = if u8::from(r1.file()) < u8::from(r2.file()) {
                    (r1, r2)
                } else {
                    (r2, r1)
                };
                let between = between_on_rank(a, b);
                if (Bitboard(between) & occupied).is_empty() {
                    color_score.mg += p.rooks_connected[0];
                    color_score.eg += p.rooks_connected[1];
                }
            }
        }

        // --- Rook behind a passed pawn ---------------------------------------
        rooks.for_each(|sq| {
            let f = u8::from(sq.file());
            let rr = u8::from(sq.rank());
            let file_bb = Bitboard(file_mask(f));
            // Ranks strictly above / below the rook's own rank.
            let above = if rr == 7 { 0 } else { !0u64 << (8 * (rr + 1)) };
            let below = if rr == 0 { 0 } else { (1u64 << (8 * rr)) - 1 };
            // A rook supports its own passers that run *ahead* of it, and
            // chases enemy passers from the far side (`ahead` of the rook).
            let (own_ahead, enemy_ahead) = if color == Color::White {
                (above, below)
            } else {
                (below, above)
            };
            let behind_own = !(passed.of(color) & file_bb & Bitboard(own_ahead)).is_empty();
            let behind_enemy = !(passed.of(!color) & file_bb & Bitboard(enemy_ahead)).is_empty();
            if behind_own || behind_enemy {
                color_score.mg += p.rook_behind_passed[0];
                color_score.eg += p.rook_behind_passed[1];
            }
        });

        // --- Queen behind a passed pawn / under pawn fire ---------------------
        board.by_piece(Role::Queen.of(color)).for_each(|sq| {
            let f = u8::from(sq.file());
            let file_bb = Bitboard(file_mask(f));
            let rr = u8::from(sq.rank());
            let above = if rr == 7 { 0 } else { !0u64 << (8 * (rr + 1)) };
            let below = if rr == 0 { 0 } else { (1u64 << (8 * rr)) - 1 };
            let (own_ahead, _enemy_ahead) = if color == Color::White {
                (above, below)
            } else {
                (below, above)
            };
            if !(passed.of(color) & file_bb & Bitboard(own_ahead)).is_empty() {
                color_score.mg += p.queen_behind_passed[0];
                color_score.eg += p.queen_behind_passed[1];
            }
            if (enemy_pawn_attacks & Bitboard::from_square(sq)).any() {
                color_score.mg += p.queen_under_pawn_attack[0];
                color_score.eg += p.queen_under_pawn_attack[1];
            }
        });

        // --- Minor coordination ----------------------------------------------
        // A bishop and knight developed to squares that co-operate with each
        // other (one on a long diagonal, the other controlling central
        // outposts) is a stronger pair than two isolated minors.
        let knights = board.by_piece(Role::Knight.of(color));
        let bishops = board.by_piece(Role::Bishop.of(color));
        if knights.any() && bishops.any() {
            let knight_home = if color == Color::White {
                (1 << 1) | (1 << 6) // b1, g1
            } else {
                (1 << 57) | (1 << 62) // b8, g8
            };
            let bishop_home = if color == Color::White {
                (1 << 2) | (1 << 5) // c1, f1
            } else {
                (1 << 58) | (1 << 61) // c8, f8
            };
            let developed_knights = knights & !Bitboard(knight_home);
            let developed_bishops = bishops & !Bitboard(bishop_home);
            if developed_knights.any() && developed_bishops.any() {
                color_score.mg += p.minor_coordination[0];
                color_score.eg += p.minor_coordination[1];
            }
        }

        score.mg += sign * color_score.mg;
        score.eg += sign * color_score.eg;
    }
    score
}

/// Squares strictly between `a` and `b` (same rank, `a` left of `b`).
fn between_on_rank(a: Square, b: Square) -> u64 {
    let af = u8::from(a.file());
    let bf = u8::from(b.file());
    let mut m = 0u64;
    for f in af + 1..bf {
        m |= file_mask(f);
    }
    m & rank_mask(u8::from(a.rank()))
}

#[inline]
fn is_light(sq: Square) -> bool {
    (u8::from(sq.file()) + u8::from(sq.rank())) % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Position as _;

    fn parts(fen: &str) -> (Board, PawnInfo, PassedInfo) {
        let board = crate::testutil::chess(fen).board().clone();
        let info = PawnInfo::scan(&board);
        let passed = PassedInfo::scan(&info);
        (board, info, passed)
    }

    fn score(fen: &str) -> Score {
        let (b, i, p) = parts(fen);
        evaluate_pieces(&b, &i, &p, 24, &EvalParams::default())
    }

    #[test]
    fn supported_central_outpost_is_bonused() {
        // White knight on e5, defended by the d4 pawn, no black pawn can reach
        // e5 (black pawns are on b4, f7, g7, h7). White must clearly prefer
        // this to a knight on e1.
        let good = score("r1bqkbnr/pppp1ppp/8/4N3/1P6/8/P1PPPPPP/R1BQKB1R w KQkq - 0 1");
        let plain = score("r1bqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(good.mg > plain.mg, "{good:?} vs {plain:?}");
    }

    #[test]
    fn pawn_attacked_square_is_no_outpost() {
        // Same e5 knight but black has a d6 pawn attacking e5: the square is
        // not a safe outpost any more.
        let safe = parts("6k1/8/8/4N3/3P4/8/8/4K3 w - - 0 1");
        let hounded = parts("6k1/8/3p4/4N3/3P4/8/8/4K3 w - - 0 1");
        let s_safe = evaluate_pieces(&safe.0, &safe.1, &safe.2, 24, &EvalParams::default());
        let s_hound = evaluate_pieces(
            &hounded.0,
            &hounded.1,
            &hounded.2,
            24,
            &EvalParams::default(),
        );
        assert!(s_safe.mg > s_hound.mg, "{s_safe:?} vs {s_hound:?}");
    }

    #[test]
    fn own_pawn_support_is_required_for_outpost() {
        // d4 pawn supports e5? No — d4 attacks c5/e5, so the knight on e5 is
        // *not* supported. Compare against the supported case: the pieces
        // term must drop.
        let supported = parts("6k1/8/8/4N3/3P4/8/8/4K3 w - - 0 1");
        let unsupported = parts("6k1/8/8/4N3/8/8/8/4K3 w - - 0 1");
        let s_sup = evaluate_pieces(
            &supported.0,
            &supported.1,
            &supported.2,
            24,
            &EvalParams::default(),
        );
        let s_unsup = evaluate_pieces(
            &unsupported.0,
            &unsupported.1,
            &unsupported.2,
            24,
            &EvalParams::default(),
        );
        assert!(s_sup.mg > s_unsup.mg, "{s_sup:?} vs {s_unsup:?}");
    }

    #[test]
    fn bad_bishop_is_penalized() {
        // White bishop on f1 (light squares). With white pawns on light
        // squares (a2, c2, e2, g2) the bishop is genuinely bad; with the same
        // pawns on dark squares it is not. Everything else (the bishop still
        // sits on its home square, the development term) cancels, so the
        // difference is exactly the bad-bishop penalty.
        let bad = parts("6k1/8/8/8/8/8/P1P1P1P1/4K1B1 w - - 0 1"); // a2 c2 e2 g2 (light)
        let ok = parts("6k1/8/8/8/8/8/1P1P1P1P/4K1B1 w - - 0 1"); // b2 d2 f2 h2 (dark)
        let s_bad = evaluate_pieces(&bad.0, &bad.1, &bad.2, 24, &EvalParams::default());
        let s_good = evaluate_pieces(&ok.0, &ok.1, &ok.2, 24, &EvalParams::default());
        assert!(
            s_bad.mg < s_good.mg,
            "light-square pawns hurt the f1 bishop: {s_bad:?} vs {s_good:?}"
        );
        assert_eq!(
            s_bad.mg - s_good.mg,
            EvalParams::default().bad_bishop_pawn[0] * 3
        );
    }

    #[test]
    fn development_penalizes_home_minors_in_the_opening() {
        // Same material, but one FEN has the queenside knight still on b1,
        // the other has it on c3 (developed).
        let lazy = parts("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let dev = parts("rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 1");
        let s_lazy = evaluate_pieces(&lazy.0, &lazy.1, &lazy.2, 24, &EvalParams::default());
        let s_dev = evaluate_pieces(&dev.0, &dev.1, &dev.2, 24, &EvalParams::default());
        assert!(
            s_dev.mg > s_lazy.mg,
            "developing the knight must improve the piece term: {} vs {}",
            s_dev.mg,
            s_lazy.mg
        );
        // In a pure endgame (phase 0) the term must be silent.
        let eg = parts("6k1/8/8/8/8/8/8/4KN2 w - - 0 1");
        let s_eg = evaluate_pieces(&eg.0, &eg.1, &eg.2, 0, &EvalParams::default());
        assert_eq!(s_eg.mg, 0, "development is an opening term only");
    }

    #[test]
    fn rooks_connected_on_open_rank() {
        // Ra1 + Rb1 share rank 1 with nothing between (the e-/g- knights have
        // moved away); a1 + h1 have pieces between by default.
        let connected = parts("6k1/8/8/8/8/8/8/RR2K3 w - - 0 1");
        let separated = parts("6k1/8/8/8/8/8/8/R3K2R w - - 0 1");
        let s_con = evaluate_pieces(
            &connected.0,
            &connected.1,
            &connected.2,
            24,
            &EvalParams::default(),
        );
        let s_sep = evaluate_pieces(
            &separated.0,
            &separated.1,
            &separated.2,
            24,
            &EvalParams::default(),
        );
        assert!(s_con.mg > s_sep.mg, "{s_con:?} vs {s_sep:?}");
        assert_eq!(
            s_con.mg - s_sep.mg,
            EvalParams::default().rooks_connected[0],
            "exactly the connected-rook bonus"
        );
    }

    #[test]
    fn rook_behind_passed_pawn_is_bonused() {
        // White rook d1 behind its own d5 passer vs a rook on a1.
        let behind = parts("6k1/8/8/3P4/8/8/8/3R2K1 w - - 0 1");
        let idle = parts("6k1/8/8/3P4/8/8/8/R3K3 w - - 0 1");
        let s_behind = evaluate_pieces(&behind.0, &behind.1, &behind.2, 24, &EvalParams::default());
        let s_idle = evaluate_pieces(&idle.0, &idle.1, &idle.2, 24, &EvalParams::default());
        assert!(s_behind.mg > s_idle.mg, "{s_behind:?} vs {s_idle:?}");
        assert_eq!(
            s_behind.mg - s_idle.mg,
            EvalParams::default().rook_behind_passed[0],
            "exactly the behind-passer bonus"
        );

        // And behind an *enemy* passer (white rook on d8 chasing black's d5).
        let chase = parts("6k1/3R4/8/3p4/8/8/8/4K3 w - - 0 1");
        let s_chase = evaluate_pieces(&chase.0, &chase.1, &chase.2, 24, &EvalParams::default());
        assert!(
            s_chase.mg > s_idle.mg,
            "a rook behind the enemy passer is also worth a bonus"
        );
    }

    #[test]
    fn bare_queen_endgame_has_no_piece_features() {
        // Only kings and a queen: no knights/bishops/rooks/pawns, so no
        // outpost, bad-bishop, development, connected-rook or behind-passer
        // term — the component must be exactly zero.
        let pos = parts("7k/5Q2/8/8/8/8/8/4K3 w - - 0 1");
        let s = evaluate_pieces(&pos.0, &pos.1, &pos.2, 24, &EvalParams::default());
        assert_eq!(s.mg, 0);
        assert_eq!(s.eg, 0);
    }

    #[test]
    fn queen_behind_passed_pawn_is_bonused() {
        // White queen on d1 behind its own d5 passer vs a queen on a1.
        let behind = parts("6k1/8/8/3P4/8/8/8/3Q2K1 w - - 0 1");
        let idle = parts("6k1/8/8/3P4/8/8/8/Q3K3 w - - 0 1");
        let s_behind = evaluate_pieces(&behind.0, &behind.1, &behind.2, 24, &EvalParams::default());
        let s_idle = evaluate_pieces(&idle.0, &idle.1, &idle.2, 24, &EvalParams::default());
        assert!(s_behind.mg > s_idle.mg, "{s_behind:?} vs {s_idle:?}");
        assert_eq!(
            s_behind.mg - s_idle.mg,
            EvalParams::default().queen_behind_passed[0],
            "exactly the behind-passer bonus"
        );
    }

    #[test]
    fn queen_under_enemy_pawn_attack_is_penalized() {
        // White queen on e5 attacked by a black pawn on d6 vs the same queen
        // on e5 with no attacking pawn.
        let safe = parts("6k1/8/8/4Q3/8/8/8/4K3 w - - 0 1");
        let hounded = parts("6k1/8/3p4/4Q3/8/8/8/4K3 w - - 0 1");
        let s_safe = evaluate_pieces(&safe.0, &safe.1, &safe.2, 24, &EvalParams::default());
        let s_hound = evaluate_pieces(
            &hounded.0,
            &hounded.1,
            &hounded.2,
            24,
            &EvalParams::default(),
        );
        assert!(s_safe.mg > s_hound.mg, "{s_safe:?} vs {s_hound:?}");
        assert_eq!(
            s_safe.mg - s_hound.mg,
            -EvalParams::default().queen_under_pawn_attack[0],
            "exactly the queen-under-pawn penalty"
        );
    }

    #[test]
    fn developed_minor_pair_is_bonused() {
        // White knight on f3 + bishop on c3 (both developed) vs the same
        // pieces still on their home squares (startpos).
        let developed = parts("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2B2N2/PPPP1PPP/R2Q1RK1 w kq - 0 1");
        let home = parts("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let s_dev = evaluate_pieces(
            &developed.0,
            &developed.1,
            &developed.2,
            24,
            &EvalParams::default(),
        );
        let s_home = evaluate_pieces(&home.0, &home.1, &home.2, 24, &EvalParams::default());
        assert!(s_dev.mg > s_home.mg, "{s_dev:?} vs {s_home:?}");
    }
}

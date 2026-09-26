//! Piece-square tables (tapered).
//!
//! Tables are given for White on square indices `A1 == 0 … H8 == 63` and
//! mirrored (vertically) for Black. Positive values favour White.
//!
//! Since Stage 5 the concrete tables live in [`crate::evaluation::EvalParams`]
//! (tunable, serialized to `config/baseline_eval.toml`); this module only
//! supplies the legacy default tables ([`LEGACY_PST`]) that the default
//! parameter set is built from.

use shakmaty::{Board, Color, Role, Square};

use crate::evaluation::Score;
use crate::evaluation::params::EvalParams;

type Table = [i16; 64];

/// The pre-Stage-5 PST constants, exposed so [`EvalParams::default`] can build
/// a byte-identical baseline. `pawn_mg`/`pawn_eg` and `king_mg`/`king_eg` are
/// phase-specific; knights/bishops/rooks/queens share one table (legacy
/// convention, now tunable independently per phase).
pub(crate) struct LegacyPst {
    pub pawn_mg: Table,
    pub pawn_eg: Table,
    pub knight: Table,
    pub bishop: Table,
    pub rook: Table,
    pub queen: Table,
    pub king_mg: Table,
    pub king_eg: Table,
}

pub(crate) const LEGACY_PST: LegacyPst = LegacyPst {
    pawn_mg: PAWN_MG,
    pawn_eg: PAWN_EG,
    knight: KNIGHT_MG,
    bishop: BISHOP_MG,
    rook: ROOK_MG,
    queen: QUEEN_MG,
    king_mg: KING_MG,
    king_eg: KING_EG,
};

/// Pawn PSTs. Values are in centipawns; index 0 = rank 1 file a … 63 = rank 8.
const PAWN_MG: Table = [
    0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5,
    10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20,
    -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
];

const PAWN_EG: Table = [
    0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 50, 50, 50, 50, 50, 50, 50, 50, 30, 30,
    30, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_MG: Table = [
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10,
    0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10,
    5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
];

const BISHOP_MG: Table = [
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10,
    -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
];

const ROOK_MG: Table = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
    0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0,
    0, 5, 5, 0, 0, 0,
];

const QUEEN_MG: Table = [
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0,
    0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
];

const KING_MG: Table = [
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40,
    -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30,
    -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0,
    10, 30, 20,
];

const KING_EG: Table = [
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20, 30,
    30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30,
    -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30,
    -50, -50,
];

/// Returns the PST index for `sq` seen from `color`'s point of view.
#[inline]
fn table_index(sq: Square, color: Color) -> usize {
    if color == Color::White {
        sq.to_usize()
    } else {
        sq.flip_vertical().to_usize()
    }
}

/// Piece-square score for the whole board (white minus black), reading the
/// tunable tables from `p` (roles 1..=6, phase 0 = mg / 1 = eg).
pub fn evaluate_pst(board: &Board, p: &EvalParams) -> Score {
    let mut score = Score::zero();
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        for role in [
            Role::Pawn,
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
            Role::King,
        ] {
            board.by_piece(role.of(color)).for_each(|sq| {
                let idx = table_index(sq, color);
                let r = role as usize;
                score.mg += sign * p.pst.tables[r][0][idx];
                score.eg += sign * p.pst.tables[r][1][idx];
            });
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::params::pst_from;
    use shakmaty::Position as _;

    fn pst_of(fen: &str) -> Score {
        let board = crate::testutil::chess(fen).board().clone();
        evaluate_pst(&board, &EvalParams::default())
    }

    #[test]
    fn centered_knight_beats_corner_knight() {
        let s_center = pst_of("6k1/8/8/8/3N4/8/8/4K3 w - - 0 1");
        let s_corner = pst_of("6k1/8/8/8/8/8/8/N3K3 w - - 0 1");
        let s_none = pst_of("6k1/8/8/8/8/8/8/4K3 w - - 0 1");
        assert!(s_center.mg > s_corner.mg);
        assert!(s_center.mg > s_none.mg);
    }

    #[test]
    fn pst_black_mirror_negates() {
        // A position and its true color mirror (colors swapped + board
        // flipped vertically) must produce opposite PST values.
        let s_w = pst_of("6k1/8/8/8/4P3/8/4N3/4K3 w - - 0 1");
        let s_m = pst_of("4k3/4n3/8/4p3/8/8/8/6K1 b - - 0 1");
        assert_eq!(s_w.mg, -s_m.mg);
        assert_eq!(s_w.eg, -s_m.eg);
    }

    /// The default PST tables must reproduce the legacy const tables exactly
    /// (the whole point of the `LEGACY_PST` derivation).
    #[test]
    fn default_pst_match_legacy_tables() {
        let legacy = LEGACY_PST;
        let p = EvalParams::default();
        assert_eq!(
            p.pst.tables[Role::Pawn as usize][0],
            pst_from(&legacy.pawn_mg)
        );
        assert_eq!(
            p.pst.tables[Role::Pawn as usize][1],
            pst_from(&legacy.pawn_eg)
        );
        for (role, table) in [
            (Role::Knight, legacy.knight),
            (Role::Bishop, legacy.bishop),
            (Role::Rook, legacy.rook),
            (Role::Queen, legacy.queen),
        ] {
            assert_eq!(p.pst.tables[role as usize][0], pst_from(&table));
            assert_eq!(p.pst.tables[role as usize][1], pst_from(&table));
        }
        assert_eq!(
            p.pst.tables[Role::King as usize][0],
            pst_from(&legacy.king_mg)
        );
        assert_eq!(
            p.pst.tables[Role::King as usize][1],
            pst_from(&legacy.king_eg)
        );
    }
}

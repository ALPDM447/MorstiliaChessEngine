//! Real NNUE evaluation: the `nn-1a298aa575a0` net that Stockfish 19 ships
//! with, wired into this engine's search.
//!
//! # Layout
//!
//! | module | what it owns |
//! |---|---|
//! | [`format`] | the `.nnue` container: version, per-section hashes, LEB128 |
//! | [`attacks`] | the const attack tables (knight/king/pawn/ray geometry) |
//! | [`types`] | colours, piece codes, the dirty-change records |
//! | [`board`] | a Stockfish-shaped board plus the threat bookkeeping |
//! | [`features`] | the three feature sets and their index formulas |
//! | [`accumulator`] | the incrementally maintained input sums |
//! | [`network`] | the weights and the forward pass |
//!
//! # Why it is built this way
//!
//! The search has no make/unmake: every node is a fresh [`crate::board::Position`]
//! clone. Rather than keeping a long-lived NNUE board and hoping it stays in
//! sync with the position, [`board::Board::from_position`] rebuilds it from the
//! position and [`board::Board::apply_move`] derives the child board and the
//! dirty records in one pass. That is a little more work per node than
//! Stockfish's make/unmake, and it makes an entire category of bug
//! impossible.
//!
//! Accumulators live in a per-ply stack ([`AccumulatorStack`]) on the search
//! thread, not in the `Position` — the position is cloned, so an accumulator
//! stored inside it would be cloned too.
//!
//! The forward pass is plain scalar integer arithmetic. There is no SIMD here;
//! see the module docs of [`network`].

pub mod accumulator;
pub mod attacks;
pub mod board;
pub mod embedded;
pub mod features;
pub mod format;
pub mod network;
pub mod types;

use std::path::{Path, PathBuf};

use crate::board::Position;
use crate::nnue::accumulator::Accumulator;
use crate::nnue::board::Board;
use crate::nnue::network::Network;
use crate::nnue::types::Color;
use crate::types::MATE;

pub use accumulator::{AccumulatorStack, L1, LAYER_STACKS, PSQT_BUCKETS};
pub use board::MoveDirties;
pub use embedded::{embedded_network_size, load_embedded_network};
pub use network::{LoadError, NETWORK_HASH, NetworkOutput, load_from, load_network};
pub use types::{PIECE_VALUE, SQ_NONE};

/// The bundled net's file name, relative to the `nnue/` directory.
pub const DEFAULT_NET_FILE: &str = "nn-1a298aa575a0.nnue";

/// The bound [`blend`] clamps to, so the network can never report a score the
/// search would confuse with a mate or a tablebase result.
///
/// Stockfish derives it from its 246-ply `MAX_PLY`
/// (`VALUE_TB_WIN_IN_MAX_PLY - 1`); this engine searches 128 plies, so the same
/// formula with its own `MAX_PLY` gives the analogous, tighter bound.
pub const EVAL_LIMIT: i32 = MATE - 2 * crate::types::MAX_PLY as i32 - 2;

/// The position facts the evaluation needs that the accumulator does not carry.
///
/// Kept separate from the accumulator because they are recomputed from the
/// position in a handful of instructions, while the accumulator costs a thousand
/// multiply-adds to build.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EvalMeta {
    /// Side to move, in Stockfish's order.
    pub stm: Color,
    /// Total number of pieces on the board.
    pub piece_count: u32,
    /// Number of pawns **of either colour** on the board (`pos.count<PAWN>()`).
    ///
    /// Worth spelling out: the bare `Position::count<Pt>()` template in
    /// `position.h` is `count<Pt>(WHITE) + count<Pt>(BLACK)`, so the material
    /// term that scales the network output is symmetric in the two sides'
    /// pawn counts, unlike the network's own output, which is side-relative.
    pub pawns: i32,
    /// Non-pawn material of *both* colours (`pos.non_pawn_material()`).
    pub non_pawn_material: i32,
    /// The 50-move counter (`pos.rule50_count()`).
    pub rule50: u32,
}

impl EvalMeta {
    /// Reads the facts out of `pos` and an already-built [`Board`].
    pub fn of(pos: &Position, board: &Board) -> EvalMeta {
        let stm = Color::from_shakmaty(pos.turn());
        EvalMeta {
            stm,
            piece_count: board.piece_count(),
            pawns: board.all_pawns().count_ones() as i32,
            non_pawn_material: board.non_pawn_material(),
            rule50: pos.halfmoves(),
        }
    }

    /// Reads the same facts straight off the position, without needing a
    /// [`Board`].
    ///
    /// This is what the search uses: by the time a node is scored the board it
    /// was built from is a temporary again, and these are five bitboard
    /// popcounts rather than a board rebuild.
    pub fn of_position(pos: &Position) -> EvalMeta {
        let stm = Color::from_shakmaty(pos.turn());
        let board = pos.board();
        EvalMeta {
            stm,
            piece_count: board.occupied().count() as u32,
            pawns: board.pawns().count() as i32,
            non_pawn_material: non_pawn_material_of(board),
            rule50: pos.halfmoves(),
        }
    }

    /// The evaluation bucket `(piece_count - 1) / 4`.
    #[inline]
    pub fn bucket(&self) -> usize {
        ((self.piece_count.max(1) - 1) / 4) as usize
    }
}

/// Stockfish's final blend from the network's two raw outputs to a
/// centipawn score, from the side to move's point of view.
///
/// Transcribed from `Eval::evaluate` with `optimism = 0`, which is what the
/// search always passes. The two blending steps are:
///
/// * **complexity damping** — a position where the PSQT and positional terms
///   disagree is one where the net is least reliable, so the total is pulled
///   towards zero proportionally to `|psqt - positional|`;
/// * **rule-50 damping** — a position approaching the 50-move draw is scaled
///   down linearly.
///
/// The material scaling in between uses `534 * <all pawns> +
/// non_pawn_material`, both of which count *both* colours, so the factor grows
/// with the amount of material on the board regardless of who owns it.
#[inline]
pub fn blend(out: NetworkOutput, meta: &EvalMeta) -> i32 {
    const COMPLEXITY_SHIFT: i64 = 18_236;
    const MATERIAL_SCALE: i64 = 91_000;
    const PAWN_MATERIAL: i32 = 534;

    let nnue = i64::from(out.raw());
    let complexity = i64::from((out.psqt - out.positional).abs());
    // `optimism` is zero throughout the search, so its own update and its
    // contribution below are both no-ops — kept explicit for the record.
    debug_assert_eq!(0i64 + 0i64 * complexity / 476, 0);
    let nnue = nnue - nnue * complexity / COMPLEXITY_SHIFT;

    let material = PAWN_MATERIAL * meta.pawns + meta.non_pawn_material;
    let mut v = (nnue + (nnue * i64::from(material)) / MATERIAL_SCALE) as i32;

    v -= v * meta.rule50 as i32 / 199;
    v.clamp(-EVAL_LIMIT, EVAL_LIMIT)
}

/// Evaluates `pos` from its accumulator, returning a side-to-move-relative
/// centipawn score.
///
/// `acc` must hold a computed accumulator for `pos`; use
/// [`Accumulator::refresh`] to produce one.
pub fn evaluate(net: &Network, pos: &Position, acc: &Accumulator) -> i32 {
    let meta = EvalMeta::of_position(pos);
    let out = net.evaluate(acc, meta.stm, meta.bucket());
    blend(out, &meta)
}

/// Stockfish's `pos.non_pawn_material()`: every non-pawn piece of both colours,
/// in the net's own centipawn units (king 0).
fn non_pawn_material_of(board: &shakmaty::Board) -> i32 {
    let mat = board.material();
    let of = |role: shakmaty::Role| {
        PIECE_VALUE[u8::from(role) as usize] * (mat.white.get(role) + mat.black.get(role)) as i32
    };
    of(shakmaty::Role::Knight)
        + of(shakmaty::Role::Bishop)
        + of(shakmaty::Role::Rook)
        + of(shakmaty::Role::Queen)
}

/// Loads the bundled net once per process and hands out a shared reference.
///
/// Only the crate's own tests use this; production code goes through
/// [`load_network`] so a missing or corrupt file is reported rather than
/// silently worked around.
#[cfg(test)]
pub fn test_net() -> &'static Network {
    use std::sync::OnceLock;
    static NET: OnceLock<Network> = OnceLock::new();
    NET.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("nnue")
            .join(DEFAULT_NET_FILE);
        load_network(&path).unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()))
    })
}

/// A [`Board`] plus the accumulated search facts, for callers that need both.
pub fn board_and_meta(pos: &Position) -> (Board, EvalMeta) {
    let board = Board::from_position(pos);
    let meta = EvalMeta::of(pos, &board);
    (board, meta)
}

/// The `.nnue` file to load for a configured path.
///
/// An empty path means "the bundled net". The search is tried in this order:
///
/// 1. `path` itself, as given (absolute or relative to the *working
///    directory*, which is what a UCI GUI's relative paths mean);
/// 2. `path` relative to the directory holding the executable — where an
///    installed engine's `nnue/` directory lives;
/// 3. the bundled `nnue/<DEFAULT_NET_FILE>` next to the executable, when
///    `path` is empty.
///
/// Exposed as a single function so the UCI layer, the CLI and the tests all
/// resolve the same file.
pub fn resolve_net_path(path: &str) -> PathBuf {
    if !path.is_empty() {
        let direct = PathBuf::from(path);
        if direct.is_file() {
            return direct;
        }
        if let Some(found) = next_to_exe(&direct) {
            return found;
        }
        // Fall through to the direct path so the error message names what the
        // user actually typed rather than a path we invented.
        return direct;
    }
    let bundled = PathBuf::from("nnue").join(DEFAULT_NET_FILE);
    if bundled.is_file() {
        return bundled;
    }
    next_to_exe(&bundled).unwrap_or(bundled)
}

/// `p` resolved against the executable's directory, if it exists there.
fn next_to_exe(p: &Path) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join(p);
    candidate.is_file().then_some(candidate)
}

/// The position suite the NNUE tests sweep over: Stockfish's own evaluation
/// regression set (opening, middlegame, tactical, promotion, en passant,
/// castling, and one position per piece-count bucket), plus a few endgames.
/// Ground truth from Stockfish 19 exists for exactly these.
#[cfg(test)]
pub const TEST_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "6k1/8/8/3pP3/8/8/8/6K1 w - d6 0 1",
    "6k1/8/8/3Pp3/8/8/8/6K1 w - e6 0 1",
    "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
    "4k3/8/8/8/8/8/8/3QK3 b - - 0 1",
    "4k2r/8/8/8/8/8/8/3RK3 w - - 0 1",
    "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
    "8/PPPk4/8/8/8/8/4Kppp/8 b - - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "8/5ppp/8/8/8/8/PPP5/4K1k1 w - - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 1",
    "4k3/1p1p4/8/8/8/8/1P1P4/4K3 w - - 0 1",
    "8/1p1k4/8/8/8/8/1P6/1K6 w - - 0 1",
    "8/8/8/2k5/8/8/8/4KQ2 w - - 0 1",
    "4k2r/6K1/8/8/8/8/8/8 w k - 0 1",
    "2r3k1/pp3ppp/8/8/8/8/PPP3PP/2R3K1 w - - 0 1",
    "8/8/1k6/8/2p5/8/1K6/8 w - - 0 1",
    "r3k2r/8/8/4b3/8/8/8/R3K2R b KQkq - 0 1",
    "8/5k2/8/8/8/8/2K5/8 w - - 0 1",
    "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
    "8/8/8/8/8/5k2/8/4K2R w K - 0 1",
    "r3k3/8/8/8/8/8/8/R3K2R b KQ - 0 1",
    "8/8/8/8/1k6/8/1K6/8 b - - 0 1",
    "2b1k3/pppp1ppp/8/8/8/8/PPPP1PPP/2B1K3 w - - 0 1",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::types::{KNIGHT, QUEEN, ROOK};

    /// A full refresh is all the evaluation needs; the incremental path is
    /// checked against it in `accumulator`'s own tests.
    fn eval_of(fen: &str) -> (NetworkOutput, i32) {
        let net = test_net();
        let pos = Position::from_fen(fen).unwrap();
        let board = Board::from_position(&pos);
        let mut acc = Accumulator::new();
        acc.refresh(Color::White, &board, net);
        acc.refresh(Color::Black, &board, net);
        let meta = EvalMeta::of(&pos, &board);
        let out = net.evaluate(&acc, meta.stm, meta.bucket());
        let score = evaluate(net, &pos, &acc);
        (out, score)
    }

    #[test]
    fn non_pawn_material_counts_both_colours() {
        // White knight and rook on rank 1, Black queen and king on rank 8.
        // Kings are worth zero, so the sum is the three real pieces.
        let pos = Position::from_fen("3q3k/8/8/8/8/8/8/1NR1K3 w - - 0 1").unwrap();
        let expect =
            PIECE_VALUE[ROOK as usize] + PIECE_VALUE[KNIGHT as usize] + PIECE_VALUE[QUEEN as usize];
        assert_eq!(non_pawn_material_of(pos.board()), expect);
        // A colour-relative count would see only the rook and the knight, so
        // the queen is exactly the piece that distinguishes the two readings.
        assert_eq!(PIECE_VALUE[QUEEN as usize], 2538);
    }

    /// The material term of the blend counts *both* sides' pawns — Stockfish's
    /// `pos.count<PAWN>()` is `count<PAWN>(WHITE) + count<PAWN>(BLACK)`, not the
    /// side to move's count. Getting this backwards is silent: it only shows up
    /// in positions with more than a queen's worth of material.
    #[test]
    fn the_pawn_count_covers_both_colours() {
        // 3+3 pawns, so a colour-relative count would say 3 and this says 6.
        let pos = Position::from_fen("8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1").unwrap();
        let meta = EvalMeta::of_position(&pos);
        assert_eq!(meta.pawns, 6);
        assert_eq!(EvalMeta::of(&pos, &Board::from_position(&pos)), meta);
    }

    /// The two ways of reading the position facts must agree, or the score
    /// would depend on which one the caller happened to use.
    #[test]
    fn both_readings_of_the_position_facts_agree() {
        for fen in crate::nnue::TEST_FENS {
            let pos = Position::from_fen(fen).unwrap();
            let board = Board::from_position(&pos);
            assert_eq!(
                EvalMeta::of(&pos, &board),
                EvalMeta::of_position(&pos),
                "in {fen}"
            );
        }
    }

    #[test]
    fn the_bucket_follows_the_piece_count() {
        let meta = |n: u32| EvalMeta {
            stm: Color::White,
            piece_count: n,
            pawns: 0,
            non_pawn_material: 0,
            rule50: 0,
        };
        assert_eq!(meta(1).bucket(), 0);
        assert_eq!(meta(4).bucket(), 0);
        assert_eq!(meta(5).bucket(), 1);
        assert_eq!(meta(32).bucket(), 7);
    }

    /// Reproduces the blend by hand, term by term, so the constants
    /// (`18236`, `534`, `91000`, `199`) and the truncation of every division
    /// are pinned rather than merely observed.
    #[test]
    fn blend_damps_complexity_and_the_rule50_counter() {
        let base = EvalMeta {
            stm: Color::White,
            piece_count: 32,
            pawns: 8,
            non_pawn_material: 6000,
            rule50: 0,
        };
        // Agreed terms: the output passes through (the material term only
        // scales the score, it does not add material).
        let agreed = NetworkOutput {
            psqt: 20,
            positional: 20,
        };
        // material = 534 * 8 pawns + 6000 non-pawn material = 10272
        // complexity = |20 - 20| = 0, so the complexity damping is a no-op
        // v = 40 + 40 * 10272 / 91000 = 40 + 4 (truncating) = 44
        let v = blend(agreed, &base);
        assert_eq!(v, 44, "the hand-computed blend");

        // Disagreeing terms pay the complexity tax. The magnitudes have to be
        // large enough for `nnue * complexity / 18236` to be non-zero.
        //
        // Both outputs below have the *same* raw total (psqt + positional =
        // 400), so the only difference between them is the complexity tax:
        // splitting the two terms gives complexity 400, which removes
        // 400 * 400 / 18236 = 8 from a net score of 400.
        let agreed_big = NetworkOutput {
            psqt: 200,
            positional: 200,
        };
        let split = NetworkOutput {
            psqt: 400,
            positional: 0,
        };
        let damped_by_complexity = 400 - 400 * 400 / 18236;
        assert_eq!(damped_by_complexity, 392);
        let s = blend(split, &base);
        assert_eq!(
            s,
            damped_by_complexity + damped_by_complexity * 10272 / 91000,
            "complexity damping"
        );
        assert_eq!(blend(agreed_big, &base), 400 + 400 * 10272 / 91000);
        assert!(
            s < blend(agreed_big, &base),
            "a less agreed output must score lower: {s} vs {}",
            blend(agreed_big, &base)
        );

        // A 50-move counter damps the score linearly and truncates.
        let after_100 = blend(
            agreed,
            &EvalMeta {
                rule50: 100,
                ..base
            },
        );
        assert_eq!(after_100, 44 - 44 * 100 / 199);
        assert!(after_100 < v, "rule50 must damp: {after_100} vs {v}");
        let after_99 = blend(agreed, &EvalMeta { rule50: 99, ..base });
        assert_eq!(after_99, 44 - 44 * 99 / 199);
        assert!(after_99 > after_100, "the damping is monotone in rule50");
    }

    #[test]
    fn evaluation_is_a_pure_function_of_the_position() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ] {
            let a = eval_of(fen);
            let b = eval_of(fen);
            assert_eq!(a, b, "evaluation must be deterministic in {fen}");
        }
    }

    /// The start position is balanced: the PSQT term must vanish and the
    /// positional term must be small, from either side to move.
    #[test]
    fn the_start_position_is_symmetric() {
        let (out, _) = eval_of("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(out.psqt, 0, "a symmetric position has no PSQT term");
        assert!(out.positional.abs() < 100, "got {}", out.positional);
    }

    /// Mirrors a FEN the way `Board::flip_vertical` + colour swap does: ranks in
    /// reverse order, piece case flipped, side to move swapped, castling rights
    /// relabelled (`K`↔`Q`) and re-sorted, and the en passant square dropped
    /// (a mirrored double push still has one, but `Position::from_fen` does not
    /// need it and the ground truth is taken without one).
    fn mirror_fen(fen: &str) -> String {
        let mut parts = fen.split(' ');
        let board = parts.next().unwrap();
        let stm = parts.next().unwrap();
        let castling = parts.next().unwrap();
        let _ep = parts.next().unwrap();
        let halfmoves = parts.next().unwrap_or("0");
        let fullmoves = parts.next().unwrap_or("1");

        let ranks: Vec<String> = board.split('/').map(|r| r.to_string()).collect();
        let board: Vec<String> = ranks
            .iter()
            .rev()
            .map(|rank| {
                rank.chars()
                    .map(|c| {
                        if c.is_ascii_uppercase() {
                            c.to_ascii_lowercase()
                        } else {
                            c.to_ascii_uppercase()
                        }
                    })
                    .collect::<String>()
            })
            .collect();

        let rights: String = castling
            .chars()
            .map(|c| match c {
                'K' => 'k',
                'Q' => 'q',
                'k' => 'K',
                'q' => 'Q',
                c => c,
            })
            .collect::<Vec<char>>()
            .iter()
            .collect();
        // Keep the canonical KQkq order the FEN grammar expects.
        let ordered: String = ['K', 'Q', 'k', 'q']
            .iter()
            .filter(|c| rights.contains(**c))
            .collect();
        let rights = if ordered.is_empty() {
            "-".into()
        } else {
            ordered
        };

        format!(
            "{} {} {} - {} {}",
            board.join("/"),
            if stm == "w" { "b" } else { "w" },
            rights,
            halfmoves,
            fullmoves
        )
    }

    /// Flipping the colours of a position is the exact operation the two NNUE
    /// perspectives are related by, which pins the colour handling of the whole
    /// pipeline: every feature set satisfies
    /// `make_index(Black, s, pc, ksq) == make_index(White, s ^ 56, pc ^ 8, ksq ^ 56)`,
    /// so the White-perspective accumulator of the flipped board *is* the
    /// Black-perspective accumulator of the original and vice versa.
    ///
    /// The evaluator is side-to-move relative, and the flip swaps the side to
    /// move as well, so the two *roles* line up: `psqt` and `positional` both
    /// come from the same accumulator bucket in both positions and must come
    /// out equal. Verified above on Kiwipete, where the mirrored White
    /// perspective reproduces the original Black one bucket for bucket.
    #[test]
    fn a_colour_flip_preserves_the_side_to_move_relative_score() {
        for fen in crate::nnue::TEST_FENS {
            let mirrored = mirror_fen(fen);
            let a = eval_of(fen);
            let b = eval_of(&mirrored);
            assert_eq!(a.0.psqt, b.0.psqt, "psqt in {fen} -> {mirrored}");
            assert_eq!(
                a.0.positional, b.0.positional,
                "positional in {fen} -> {mirrored}"
            );
            assert_eq!(a.1, b.1, "final score in {fen} -> {mirrored}");
        }
    }

    /// The same board with only the side to move swapped. `psqt` is
    /// `(psqtAcc[stm] - psqtAcc[them]) / 2`, so it negates exactly;
    /// `positional` reads the side-to-move perspective, and the two perspectives
    /// are *separate weight rows*, so it is a different number. This is the test
    /// that would catch the net being wrongly assumed colour-symmetric — the
    /// Kiwipete position propagates to −421 for White and +902 for Black.
    #[test]
    fn swapping_the_side_to_move_negates_psqt_but_not_positional() {
        let mut saw_a_difference = false;
        for fen in crate::nnue::TEST_FENS {
            let flipped = flip_side_to_move(fen);
            let a = eval_of(fen);
            let b = eval_of(&flipped);
            assert_eq!(
                a.0.psqt, -b.0.psqt,
                "psqt is stm-relative in {fen} / {flipped}"
            );
            if a.0.positional != b.0.positional {
                saw_a_difference = true;
            }
        }
        assert!(
            saw_a_difference,
            "the two perspectives should not be weight-identical"
        );
    }

    /// Swaps the side-to-move field of a FEN, keeping everything else.
    ///
    /// The en passant square is dropped: an ep square that is legal for White to
    /// move on is by definition not legal for Black to move on, so keeping it
    /// would produce a FEN `Position::from_fen` rejects.
    fn flip_side_to_move(fen: &str) -> String {
        let mut parts = fen.split(' ').map(str::to_owned).collect::<Vec<_>>();
        parts[1] = if parts[1] == "w" {
            "b".into()
        } else {
            "w".into()
        };
        parts[3] = "-".into();
        parts.join(" ")
    }

    /// The accumulator the evaluation reads must be the one a full refresh
    /// produces — a cheap end-to-end guard on the whole chain.
    #[test]
    fn evaluation_uses_the_refreshed_accumulator() {
        let net = test_net();
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let board = Board::from_position(&pos);
        let mut acc = Accumulator::new();
        for p in Color::ALL {
            assert!(!acc.computed[p.idx()]);
            acc.refresh(p, &board, net);
            assert!(acc.computed[p.idx()]);
        }
        // A non-zero accumulator is a prerequisite for a meaningful score.
        assert!(acc.accumulation[Color::White.idx()].iter().any(|&v| v != 0));
        let score = evaluate(net, &pos, &acc);
        assert!(score.abs() < 3000, "implausible score {score}");
    }
}

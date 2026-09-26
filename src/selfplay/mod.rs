//! Deterministic engine self-play: two searchers (each with its own parameter
//! set) play out a game under fixed-depth search, producing a W/D/L outcome
//! and a full PGN game record.
//!
//! Everything is a pure function of the inputs:
//!
//! ```text
//! (white params, black params, seed, config) → GameRecord
//! ```
//!
//! The only randomness is a seeded [`crate::book::SplitMix64`] used to pick
//! the opening; from the opening moves onward the game is a deterministic
//! `Threads = 1` search at a fixed depth (a higher [`SelfPlayConfig::threads`]
//! value runs a Lazy-SMP search on both sides — used for `Threads = 1` vs
//! `Threads = N` strength comparisons, not for tuning datasets, because
//! parallel searches are non-deterministic). The same seed therefore replays
//! the exact same game — the basis for reproducible tuning datasets and for
//! honest W/D/L matches (the self-play binary pipes these records straight
//! into [`crate::rating`]).
//!
//! Games end naturally (mate, stalemate, 50-move rule, insufficient
//! material, threefold repetition) or by score adjudication: when the search
//! *proves* a forced mate within its depth, the game is decided immediately
//! instead of grinding out the final forced-move sequence. Every record
//! carries the [`Termination`] that ended it, so match reports can say *how*
//! games were decided, not just W/D/L.

use shakmaty::Position as _;
use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::book::SplitMix64;
use crate::evaluation::EvalParams;
use crate::search::{Searcher, TimeLimit};
use crate::types::{RawMove, is_mate};

/// Why a self-play game ended. Every variant maps to an
/// [`Outcome`] (White's point of view); matches record the reason so a match
/// report can show *how* games were decided, not just W/D/L.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// The side to move is mated.
    Checkmate,
    /// The side to move has no legal move and is not in check.
    Stalemate,
    /// Fifty moves (100 plies) without a pawn move or capture.
    FiftyMove,
    /// Insufficient material to mate.
    InsufficientMaterial,
    /// The current position occurred for the third time.
    ThreefoldRepetition,
    /// The game hit the configured ply cap (a draw by convention).
    MovesLimit,
    /// The search proved a forced mate and the game was cut short.
    AdjudicatedMate,
    /// The search produced no move; the game was declared a draw.
    Aborted,
}

impl Termination {
    /// A short, stable label for reports and PGN tags.
    pub fn label(&self) -> &'static str {
        match self {
            Termination::Checkmate => "checkmate",
            Termination::Stalemate => "stalemate",
            Termination::FiftyMove => "fifty-move",
            Termination::InsufficientMaterial => "insufficient-material",
            Termination::ThreefoldRepetition => "threefold-repetition",
            Termination::MovesLimit => "move-limit",
            Termination::AdjudicatedMate => "adjudicated-mate",
            Termination::Aborted => "aborted",
        }
    }

    /// True when the result is a draw (everything except decisive terminations).
    pub fn is_draw(&self) -> bool {
        !matches!(self, Termination::Checkmate | Termination::AdjudicatedMate)
    }

    /// The outcome implied by this termination at `pos` (the position that
    /// triggered it): `Checkmate` gives the win to the opponent of the side to
    /// move; `AdjudicatedMate` is handled by the caller; everything else is a
    /// draw.
    pub fn outcome(&self, pos: &Position) -> Outcome {
        match self {
            Termination::Checkmate => match pos.turn() {
                shakmaty::Color::White => Outcome::BlackWin,
                shakmaty::Color::Black => Outcome::WhiteWin,
            },
            _ => Outcome::Draw,
        }
    }
}

/// The reason a position is already over, without any search: the 50-move
/// rule, insufficient material, threefold repetition (three occurrences of
/// the current position within the zeroing window), or checkmate/stalemate.
pub fn terminal_state(pos: &Position, history: &[Zobrist64]) -> Option<Termination> {
    if pos.halfmoves() >= 100 {
        return Some(Termination::FiftyMove);
    }
    if pos.chess.is_insufficient_material() {
        return Some(Termination::InsufficientMaterial);
    }
    let h = pos.hash;
    let seen = history.iter().filter(|&&x| x == h).count();
    if seen >= 2 {
        return Some(Termination::ThreefoldRepetition);
    }
    if pos.legal_moves().is_empty() {
        return Some(if pos.is_check() {
            Termination::Checkmate
        } else {
            Termination::Stalemate
        });
    }
    None
}

/// The result of a game, always from **White's** point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    WhiteWin,
    Draw,
    BlackWin,
}

impl Outcome {
    /// The PGN `Result` tag value.
    pub fn to_pgn(self) -> &'static str {
        match self {
            Outcome::WhiteWin => "1-0",
            Outcome::Draw => "1/2-1/2",
            Outcome::BlackWin => "0-1",
        }
    }

    /// The white-relative result as a real score used by Texel tuning
    /// targets: `1.0`, `0.5`, `0.0`.
    pub fn to_score(self) -> f64 {
        match self {
            Outcome::WhiteWin => 1.0,
            Outcome::Draw => 0.5,
            Outcome::BlackWin => 0.0,
        }
    }
}

/// A deterministic opening line applied before the engines take over.
#[derive(Debug, Clone, Copy)]
pub struct Opening {
    pub name: &'static str,
    /// UCI moves from the start position (must each be legal in sequence).
    pub plies: &'static [&'static str],
}

/// A short, standard repertoire; all lines verify as legal sequences from the
/// start position (the self-play smoke test asserts this).
pub const OPENINGS: &[Opening] = &[
    Opening {
        name: "italian",
        plies: &["e2e4", "e7e5", "g1f3", "b8c6"],
    },
    Opening {
        name: "spanish",
        plies: &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
    },
    Opening {
        name: "sicilian",
        plies: &["e2e4", "c7c5", "g1f3", "d7d6"],
    },
    Opening {
        name: "french",
        plies: &["e2e4", "e7e6", "d2d4", "d7d5"],
    },
    Opening {
        name: "caro-kann",
        plies: &["e2e4", "c7c6", "d2d4", "d7d5"],
    },
    Opening {
        name: "queens-gambit",
        plies: &["d2d4", "d7d5", "c2c4", "e7e6"],
    },
    Opening {
        name: "nimsowitsch",
        plies: &["d2d4", "g8f6", "c2c4", "e7e6"],
    },
    Opening {
        name: "english",
        plies: &["c2c4", "e7e5", "g1f3", "g8f6"],
    },
    Opening {
        name: "kings-indian",
        plies: &["d2d4", "g8f6", "c2c4", "g7g6"],
    },
    Opening {
        name: "bishops-opening",
        plies: &["e2e4", "e7e5", "f1c4", "g8f6"],
    },
];

/// Self-play knobs. Defaults target quick smoke tests; `depth` is the fixed
/// search depth per move. `threads = 1` (the default) makes a game a pure
/// function of `(white params, black params, seed, config)`; higher values
/// use the Lazy-SMP search for strength comparisons (`Threads = 1` vs
/// `Threads = N`) and are **not** deterministic — do not use them to build
/// tuning datasets.
#[derive(Debug, Clone, Copy)]
pub struct SelfPlayConfig {
    /// Fixed search depth per move.
    pub depth: i32,
    /// Hard cap on plies before the game is declared a draw.
    pub max_plies: usize,
    /// TT size for each side's searcher.
    pub hash_mb: usize,
    /// End the game immediately when the search proves a forced mate.
    pub adjudicate_mate: bool,
    /// Worker threads per side (1 = deterministic fast path).
    pub threads: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            depth: 6,
            max_plies: 240,
            hash_mb: 16,
            adjudicate_mate: true,
            threads: 1,
        }
    }
}

/// One self-play game: the moves played (opening + search), the outcome, the
/// reason it ended and the opening that was selected.
#[derive(Debug, Clone)]
pub struct GameRecord {
    pub moves: Vec<RawMove>,
    pub outcome: Outcome,
    pub termination: Termination,
    pub opening: String,
}

impl GameRecord {
    /// Replays the game and returns a PGN document (SAN move text, result
    /// tags). `white/black` name the two parameter sets for the header.
    pub fn to_pgn(&self, white: &str, black: &str, event: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("[Event \"{event}\"]\n"));
        out.push_str("[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n");
        out.push_str(&format!("[White \"{white}\"]\n"));
        out.push_str(&format!("[Black \"{black}\"]\n"));
        out.push_str(&format!("[Opening \"{}\"]\n", self.opening));
        out.push_str(&format!("[Termination \"{}\"]\n", self.termination.label()));
        out.push_str(&format!("[Result \"{}\"]\n\n", self.outcome.to_pgn()));

        let mut pos = Position::startpos();
        let mut san: Vec<String> = Vec::with_capacity(self.moves.len());
        for &m in &self.moves {
            san.push(san_of(&pos, m));
            pos = pos.make_child(m);
        }
        let mut tokens: Vec<String> = Vec::with_capacity(self.moves.len() + 8);
        for (i, s) in san.iter().enumerate() {
            if i % 2 == 0 {
                tokens.push(format!("{}.", i / 2 + 1));
            }
            tokens.push(s.clone());
        }
        tokens.push(self.outcome.to_pgn().to_string());

        // Wrap at ~72 columns on token boundaries.
        let mut line = String::new();
        for t in tokens {
            if !line.is_empty() && line.len() + t.len() + 1 > 72 {
                out.push_str(&line);
                out.push('\n');
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&t);
        }
        out.push_str(&line);
        out.push('\n');
        out
    }
}

/// Plays one game with `white`/`black` parameter sets, seeded by `rng`.
pub fn play_game(white: &EvalParams, black: &EvalParams, rng: &mut SplitMix64) -> GameRecord {
    play_game_with(white, black, &SelfPlayConfig::default(), rng)
}

/// [`play_game`] with an explicit configuration.
pub fn play_game_with(
    white: &EvalParams,
    black: &EvalParams,
    cfg: &SelfPlayConfig,
    rng: &mut SplitMix64,
) -> GameRecord {
    let opening = OPENINGS[(rng.next() % OPENINGS.len() as u64) as usize];

    let mut pos = Position::startpos();
    let mut moves: Vec<RawMove> = Vec::new();
    // Hashes of all positions before the current root since the last zeroing
    // move (the search's repetition convention).
    let mut history: Vec<Zobrist64> = Vec::new();

    for uci in opening.plies {
        match pos.play_uci(uci) {
            Ok((child, raw)) => {
                record(&mut history, &pos, &child);
                moves.push(raw);
                pos = child;
            }
            Err(_) => break, // defensive: OPENINGS are validated in tests
        }
    }

    let mut white_searcher = Searcher::with_params(cfg.hash_mb, white.clone());
    let mut black_searcher = Searcher::with_params(cfg.hash_mb, black.clone());
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let (outcome, termination) = loop {
        // --- natural terminations (no search needed) ----------------------
        if let Some(t) = terminal_state(&pos, &history) {
            break (t.outcome(&pos), t);
        }
        if moves.len() >= cfg.max_plies {
            break (Outcome::Draw, Termination::MovesLimit);
        }

        // --- search & play the side to move's move ------------------------
        let searcher = if pos.turn() == shakmaty::Color::White {
            &mut white_searcher
        } else {
            &mut black_searcher
        };
        let limits = TimeLimit {
            depth: Some(cfg.depth),
            nodes: None,
            movetime_ms: None,
            soft_ms: 0,
            hard_ms: 0,
            infinite: true,
        };
        let r = searcher.search(&pos, &history, &limits, &stop, cfg.threads, &[]);
        let m = r.best;

        // Score adjudication: a proven forced mate decides the game without
        // playing out the final forced sequence.
        if cfg.adjudicate_mate && is_mate(r.score) {
            let side_to_move_wins = r.score > 0;
            break (
                if side_to_move_wins {
                    match pos.turn() {
                        shakmaty::Color::White => Outcome::WhiteWin,
                        shakmaty::Color::Black => Outcome::BlackWin,
                    }
                } else {
                    match pos.turn() {
                        shakmaty::Color::White => Outcome::BlackWin,
                        shakmaty::Color::Black => Outcome::WhiteWin,
                    }
                },
                Termination::AdjudicatedMate,
            );
        }

        if m == RawMove::NULL {
            // Search returned nothing (should not happen with unlimited
            // limits); treat as a stalemate-ish draw.
            break (Outcome::Draw, Termination::Aborted);
        }
        let child = pos.make_child(m);
        record(&mut history, &pos, &child);
        moves.push(m);
        pos = child;
    };

    GameRecord {
        moves,
        outcome,
        termination,
        opening: opening.name.to_string(),
    }
}

/// Appends the parent hash to the repetition history, resetting after zeroing
/// moves (captures and pawn pushes) exactly like the game-loop convention.
pub fn record(history: &mut Vec<Zobrist64>, parent: &Position, child: &Position) {
    if child.halfmoves() == 0 {
        history.clear();
    } else {
        history.push(parent.hash);
    }
}

/// Standard algebraic notation for `m` at `pos` (the position before the
/// move), via shakmaty's SAN renderer.
pub fn san_of(pos: &Position, m: RawMove) -> String {
    let m = m.to_shakmaty(pos.board());
    shakmaty::san::San::from_move(pos.chess(), m).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    #[test]
    fn openings_are_legal_sequences() {
        for o in OPENINGS {
            let mut pos = Position::startpos();
            for uci in o.plies {
                let child = pos
                    .play_uci(uci)
                    .unwrap_or_else(|_| panic!("opening {} move {uci} is illegal", o.name));
                pos = child.0;
            }
        }
    }

    #[test]
    fn outcome_tags_are_pgn_shaped() {
        assert_eq!(Outcome::WhiteWin.to_pgn(), "1-0");
        assert_eq!(Outcome::Draw.to_pgn(), "1/2-1/2");
        assert_eq!(Outcome::BlackWin.to_pgn(), "0-1");
        assert_eq!(Outcome::WhiteWin.to_score(), 1.0);
        assert_eq!(Outcome::Draw.to_score(), 0.5);
        assert_eq!(Outcome::BlackWin.to_score(), 0.0);
    }

    /// A fast symmetric smoke game (depth 2): both sides use the baseline
    /// defaults, so the expected result is a draw — but the structural
    /// invariants must hold whichever way it ends.
    #[test]
    fn symmetric_smoke_game_is_well_formed() {
        let params = EvalParams::default();
        let cfg = SelfPlayConfig {
            depth: 2,
            max_plies: 120,
            hash_mb: 4,
            adjudicate_mate: false,
            threads: 1,
        };
        let mut rng = rng(42);
        let g = play_game_with(&params, &params, &cfg, &mut rng);
        assert!(!g.moves.is_empty(), "a game must play some moves");
        assert!(g.moves.len() <= 120);
        // Every move must be legal in sequence.
        let mut pos = Position::startpos();
        for &m in &g.moves {
            assert!(pos.raw_move_legal(m), "illegal move {}", m.to_uci());
            pos = pos.make_child(m);
        }
        // The record's outcome is one of the three states.
        assert!(matches!(
            g.outcome,
            Outcome::WhiteWin | Outcome::Draw | Outcome::BlackWin
        ));
        // PGN must be parseable-ish: tags present, result matches.
        let pgn = g.to_pgn("base", "base", "smoke");
        assert!(
            pgn.contains("[Result \"1/2-1/2\"]")
                || pgn.contains("[Result \"1-0\"]")
                || pgn.contains("[Result \"0-1\"]")
        );
        assert!(pgn.contains(&format!("[Opening \"{}\"]", g.opening)));
        assert!(pgn.trim_end().ends_with(g.outcome.to_pgn()));
    }

    #[test]
    fn smoke_game_records_a_termination() {
        let params = EvalParams::default();
        let cfg = SelfPlayConfig {
            depth: 2,
            max_plies: 60,
            hash_mb: 4,
            adjudicate_mate: false,
            threads: 1,
        };
        let mut rng = rng(42);
        let g = play_game_with(&params, &params, &cfg, &mut rng);
        // Draw-ness of the termination must agree with the outcome.
        assert_eq!(
            g.termination.is_draw(),
            g.outcome == Outcome::Draw,
            "termination {:?} disagrees with outcome {:?}",
            g.termination,
            g.outcome
        );
        // A recorded checkmate must genuinely be a mated position.
        let mut pos = Position::startpos();
        for &m in &g.moves {
            pos = pos.make_child(m);
        }
        if g.termination == Termination::Checkmate {
            assert!(pos.legal_moves().is_empty() && pos.is_check());
        }
        // The PGN carries the termination tag.
        let pgn = g.to_pgn("base", "base", "smoke");
        assert!(pgn.contains(&format!("[Termination \"{}\"]", g.termination.label())));
    }

    #[test]
    fn same_seed_replays_the_same_game() {
        let params = EvalParams::default();
        let mut a = rng(7);
        let mut b = rng(7);
        let g1 = play_game(&params, &params, &mut a);
        let g2 = play_game(&params, &params, &mut b);
        assert_eq!(g1.outcome, g2.outcome);
        assert_eq!(
            g1.moves.len(),
            g2.moves.len(),
            "same seed must replay the same ply count"
        );
        for (x, y) in g1.moves.iter().zip(g2.moves.iter()) {
            assert_eq!(x, y, "same seed must replay identical moves");
        }
    }

    #[test]
    fn different_seeds_diverge_somewhere() {
        let params = EvalParams::default();
        let mut a = rng(1);
        let mut b = rng(99);
        let g1 = play_game(&params, &params, &mut a);
        let g2 = play_game(&params, &params, &mut b);
        // Different openings are guaranteed by the seed → opening selection;
        // if the two random games happened to be one move apart overall, the
        // outcome or move count must still differ somewhere. (This is a
        // heuristic check, not a proof — play the seeds known to differ.)
        let differs = g1.opening != g2.opening
            || g1.outcome != g2.outcome
            || g1.moves.len() != g2.moves.len()
            || g1.moves.iter().zip(g2.moves.iter()).any(|(a, b)| a != b);
        assert!(differs, "different seeds must produce different games");
    }
}

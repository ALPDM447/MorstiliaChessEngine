//! Evaluation datasets for Texel-style tuning.
//!
//! A dataset is an ordered list of `(position, result)` entries where
//! `result` is the white-relative game result: `1.0` (white wins), `0.5`
//! (draw), `0.0` (black wins). Two sources exist, both fully deterministic:
//!
//! * [`DataSet::from_selfplay_games`] — every position reached in a set of
//!   seeded self-play games, scored by the game's outcome (the real signal
//!   the tuner learns from);
//! * [`DataSet::anchors`] — a fixed, hand-picked set of coarse calibration
//!   positions (symmetric → `0.5`, clear material wins → `1.0`/`0.0`) that
//!   pin the sign and monotonicity of the loss surface.
//!
//! The Texel loss — `Σ (σ(K·eval) − y)²` with `K = ln(10)/400` (so an eval
//! of +400 centipawns maps to a 10:1 score ratio) — is computed by
//! [`DataSet::texel_loss`] over the white-relative evaluation.

use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{Context, bail};
use shakmaty::san::San;
use shakmaty::{Color, Role};

use crate::board::Position;
use crate::book::SplitMix64;
use crate::evaluation::phase::game_phase;
use crate::evaluation::{EvalParams, Evaluator};
use crate::selfplay::GameRecord;
use crate::types::RawMove;

/// The Texel sigmoid constant: `K = ln(10) / 400`. An evaluation of `+400`
/// centipawns from White's point of view maps to a win probability of
/// `1/(1 + 10⁻¹) = 10/11 ≈ 0.91`.
pub const TEXEL_K: f64 = std::f64::consts::LN_10 / 400.0;

/// One dataset entry: a position and its white-relative game result.
#[derive(Debug, Clone)]
pub struct EvalEntry {
    pub pos: Position,
    /// `1.0` white wins, `0.5` draw, `0.0` black wins.
    pub result: f64,
}

/// A deterministic training set.
#[derive(Debug, Clone)]
pub struct DataSet {
    pub entries: Vec<EvalEntry>,
    /// Input seed (for provenance / reproducibility reporting).
    pub seed: u64,
    /// Short description of where the entries came from.
    pub source: String,
}

/// Summary statistics of a [`DataSet`] (dataset-pipeline reporting; never
/// touched by the engine hot path).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatasetStats {
    /// Number of entries.
    pub entries: usize,
    /// Entries whose target result favours White (`result > 0.5`).
    pub white_wins: usize,
    /// Entries with an exactly balanced target (`result == 0.5`).
    pub draws: usize,
    /// Entries whose target result favours Black (`result < 0.5`).
    pub black_wins: usize,
    /// Mean target result over all entries (`0.0` when empty).
    pub avg_result: f64,
    /// Mean game phase over all entries, `0.0` (endgame) .. `24.0` (opening).
    pub avg_phase: f64,
    /// Mean absolute material imbalance in centipawns (queen..pawn, no
    /// bishop-pair bonus), i.e. how decisive the positions are on average.
    pub avg_material_cp: f64,
}

impl fmt::Display for DatasetStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} entries · targets W/D/L {}/{}/{} · avg result {:.3} · avg phase {:.1} · avg |material| {:.0} cp",
            self.entries,
            self.white_wins,
            self.draws,
            self.black_wins,
            self.avg_result,
            self.avg_phase,
            self.avg_material_cp
        )
    }
}

impl DataSet {
    /// Collects every position of every game, scored by the game outcome.
    /// Positions are walked from the start position, so the dataset covers
    /// openings as well as middlegames/endgames; `seed` is recorded for
    /// provenance.
    pub fn from_selfplay_games(games: &[GameRecord]) -> DataSet {
        let mut entries = Vec::new();
        for g in games {
            let mut pos = Position::startpos();
            entries.push(EvalEntry {
                pos: pos.clone(),
                result: g.outcome.to_score(),
            });
            for &m in &g.moves {
                pos = pos.make_child(m);
                entries.push(EvalEntry {
                    pos: pos.clone(),
                    result: g.outcome.to_score(),
                });
            }
        }
        DataSet {
            entries,
            seed: 0,
            source: format!("selfplay {} game(s)", games.len()),
        }
    }

    /// Marks the dataset with its provenance seed.
    pub fn with_seed(mut self, seed: u64) -> DataSet {
        self.seed = seed;
        self
    }

    /// A fixed set of coarse calibration anchors. The targets are deliberately
    /// simple (equal material → 0.5; an upper hand → 1.0/0.0) so the *sign*
    /// structure of the loss is always well-defined even with a tiny dataset.
    pub fn anchors() -> DataSet {
        const A: &[(&str, f64)] = &[
            // Perfectly symmetric / equal: score 0.5.
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                0.5,
            ),
            ("4k3/8/8/8/8/8/4K3/8 w - - 0 1", 0.5),
            // Black is up a full rook (KR vs K): white loses.
            ("8/8/8/8/8/2k5/2r5/2K5 w - - 0 1", 0.0),
            // White is up a full rook: white wins.
            ("8/8/8/8/8/2k5/3R4/2K5 w - - 0 1", 1.0),
            // White up a queen for the exchange-ish gap: white wins.
            ("5k2/8/8/8/8/8/4q3/5K2 w - - 0 1", 0.0),
            ("5k2/8/8/8/8/8/4Q3/5K2 w - - 0 1", 1.0),
            // K+P vs K with the central c-pawn: a theoretical win — a strong
            // advantage, though not forced instantly.
            ("8/8/8/8/8/2k5/2P5/2K5 w - - 0 1", 0.8),
            // Bare-kings equality at a mid-piece count: parity.
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                0.5,
            ),
        ];
        let entries = A
            .iter()
            .map(|(fen, result)| EvalEntry {
                pos: Position::from_fen(fen).unwrap_or_else(|_| panic!("bad anchor FEN {fen:?}")),
                result: *result,
            })
            .collect();
        DataSet {
            entries,
            seed: 0,
            source: "anchors".to_string(),
        }
    }

    /// Fisher–Yates shuffle with `rng` (in place, deterministic for a seed).
    pub fn shuffle_with(&mut self, rng: &mut SplitMix64) {
        let n = self.entries.len();
        for i in (1..n).rev() {
            let j = (rng.next() as usize) % (i + 1);
            self.entries.swap(i, j);
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The Texel loss `Σ (σ(K·eval) − y)² / N` over this dataset for `params`.
    pub fn texel_loss(&self, params: &EvalParams) -> f64 {
        let ev = Evaluator;
        let mut sum = 0.0;
        for e in &self.entries {
            let eval = ev.evaluate_white_with(&e.pos, params) as f64;
            let sigma = 1.0 / (1.0 + (-TEXEL_K * eval).exp());
            let d = sigma - e.result;
            sum += d * d;
        }
        sum / self.entries.len().max(1) as f64
    }

    // ------------------------------------------------------------------
    // File-based dataset sources (tooling: I/O and allocation are fine
    // here; none of this runs in the engine hot path).
    // ------------------------------------------------------------------

    /// An empty dataset tagged with `source`.
    pub fn empty(source: &str) -> DataSet {
        DataSet {
            entries: Vec::new(),
            seed: 0,
            source: source.to_string(),
        }
    }

    /// Appends every entry of `other`, merging the source descriptions.
    pub fn extend(&mut self, other: DataSet) {
        self.entries.extend(other.entries);
        if !other.source.is_empty() {
            if !self.source.is_empty() {
                self.source.push_str(", ");
            }
            self.source.push_str(&other.source);
        }
    }

    /// Loads a text file of evaluation positions. Each non-empty line that
    /// does not start with `#` holds one target result and one FEN, in
    /// either order:
    ///
    /// ```text
    /// 0.5 rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1   # result first
    /// 8/8/8/8/8/2k5/3R4/2K5 w - - 0 1 1.0                             # FEN first
    /// ```
    ///
    /// Results may be any target in `[0, 1]` (`0`, `0.5`, `1`, `0.8`, …) or
    /// PGN notation (`1-0`, `0-1`, `1/2-1/2`). The two orders are
    /// disambiguated structurally: the first token of a FEN is the board
    /// part, which never parses as a result, and a result-first line puts
    /// the result exactly at the front.
    pub fn from_fen_file(path: impl AsRef<Path>) -> anyhow::Result<DataSet> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).with_context(|| format!("cannot read {:?}", path))?;
        let mut entries = Vec::new();
        for (idx, raw) in text.lines().enumerate() {
            let lineno = idx + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let first = line.split_whitespace().next().unwrap_or("");
            let (result, fen) = if parse_result(first).is_some() {
                // Result first: everything after the first token is the FEN.
                let fen = line[first.len()..].trim();
                if fen.is_empty() {
                    bail!("{path:?}:{lineno}: result with no FEN");
                }
                (parse_result(first).expect("checked above"), fen.to_string())
            } else {
                // FEN first: the result is the final token.
                let cut = line.rfind(char::is_whitespace).with_context(|| {
                    format!("{path:?}:{lineno}: expected `result FEN` or `FEN result`")
                })?;
                let (fen_part, res_part) = line.split_at(cut);
                let result = parse_result(res_part.trim()).with_context(|| {
                    format!(
                        "{path:?}:{lineno}: last token {:?} is not a result in [0, 1]",
                        res_part.trim()
                    )
                })?;
                (result, fen_part.trim().to_string())
            };
            let pos = Position::from_fen(&fen)
                .with_context(|| format!("{path:?}:{lineno}: invalid FEN {fen:?}"))?;
            entries.push(EvalEntry { pos, result });
        }
        if entries.is_empty() {
            bail!("{path:?}: no positions found");
        }
        Ok(DataSet {
            entries,
            seed: 0,
            source: format!("fen {}", path.display()),
        })
    }

    /// Loads every *finished* game of a PGN file (`1-0`, `0-1`,
    /// `1/2-1/2`) and records all of its positions under the game's result.
    /// Games tagged `*` (unfinished) are skipped; a game whose movetext
    /// cannot be replayed is truncated at that point and kept. An optional
    /// `[FEN "..."]` header sets up composed positions. Comments (`{...}`,
    /// `;`), NAGs, move numbers, annotations and `(...)` variations are
    /// stripped before the SAN plies are replayed.
    pub fn from_pgn_file(path: impl AsRef<Path>) -> anyhow::Result<DataSet> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).with_context(|| format!("cannot read {:?}", path))?;
        // Frame the games: every line-initial `[Event` starts a new game.
        let bytes = text.as_bytes();
        let mut starts: Vec<usize> = Vec::new();
        for i in 0..bytes.len() {
            if bytes[i..].starts_with(b"[Event") && (i == 0 || bytes[i - 1] == b'\n') {
                starts.push(i);
            }
        }
        if starts.is_empty() {
            bail!("{path:?}: no PGN games found (no [Event ...] tag)");
        }
        let mut entries: Vec<EvalEntry> = Vec::new();
        let mut games = 0usize;
        let mut truncated = 0usize;
        let mut skipped = 0usize;
        for (k, &s) in starts.iter().enumerate() {
            let e = starts.get(k + 1).copied().unwrap_or(bytes.len());
            let game = parse_pgn_game(&text[s..e])
                .with_context(|| format!("{path:?}: game {}", games + skipped + 1))?;
            let Some(result) = game.result else {
                skipped += 1;
                continue;
            };
            let mut pos = match &game.fen {
                Some(fen) => Position::from_fen(fen)
                    .with_context(|| format!("{path:?}: bad [FEN {fen:?}] header"))?,
                None => Position::startpos(),
            };
            let mut played = 0usize;
            let mut game_entries = vec![EvalEntry {
                pos: pos.clone(),
                result,
            }];
            for san_tok in &game.sans {
                let Ok(san) = san_tok.parse::<San>() else {
                    break;
                };
                let Ok(m) = san.to_move(pos.chess()) else {
                    break;
                };
                pos = pos.make_child(RawMove::from_shakmaty(m));
                game_entries.push(EvalEntry {
                    pos: pos.clone(),
                    result,
                });
                played += 1;
            }
            if played != game.sans.len() {
                truncated += 1;
            }
            entries.append(&mut game_entries);
            games += 1;
        }
        if games == 0 {
            bail!("{path:?}: no finished games (all {skipped} tagged \"*\")");
        }
        if entries.is_empty() {
            bail!("{path:?}: games produced no positions");
        }
        Ok(DataSet {
            entries,
            seed: 0,
            source: format!(
                "pgn {} ({games} games, {skipped} skipped, {truncated} truncated)",
                path.display()
            ),
        })
    }

    /// Deterministic train/test split: the entries are shuffled with
    /// `SplitMix64(seed)` (Fisher–Yates) and cut so that a `fraction` share
    /// (in `(0, 1)`) ends up in the training set. The same seed always
    /// yields the identical partition; both halves keep their entries'
    /// original order *within* the partition.
    pub fn train_test_split(&self, fraction: f64, seed: u64) -> (DataSet, DataSet) {
        assert!(
            (0.0..1.0).contains(&fraction) && fraction > 0.0,
            "split fraction must be in (0, 1), got {fraction}"
        );
        let mut all = self.clone();
        all.shuffle_with(&mut SplitMix64(seed));
        let n = all.entries.len();
        if n == 0 {
            let test = DataSet::empty(&format!("{} (test)", self.source));
            let mut train = DataSet::empty(&format!("{} (train)", self.source));
            train.seed = seed;
            return (train, test);
        }
        let n_train = ((n as f64) * (1.0 - fraction)).round() as usize;
        let n_train = n_train.clamp(1, n.saturating_sub(1).max(1));
        let test_entries = all.entries.split_off(n_train);
        let mut train = all;
        train.seed = seed;
        train.source = format!("{} (train)", self.source);
        let test = DataSet {
            entries: test_entries,
            seed,
            source: format!("{} (test)", self.source),
        };
        (train, test)
    }

    /// Target balance, mean result, mean game phase and mean absolute
    /// material imbalance over this dataset.
    pub fn stats(&self) -> DatasetStats {
        let vals = EvalParams::default();
        let mut white_wins = 0usize;
        let mut draws = 0usize;
        let mut black_wins = 0usize;
        let mut result_sum = 0.0f64;
        let mut phase_sum = 0.0f64;
        let mut material_sum = 0.0f64;
        for e in &self.entries {
            if e.result > 0.5 {
                white_wins += 1;
            } else if e.result < 0.5 {
                black_wins += 1;
            } else {
                draws += 1;
            }
            result_sum += e.result;
            let board = e.pos.board();
            phase_sum += game_phase(board, &vals) as f64;
            let mut mat = 0i32;
            for role in [
                Role::Pawn,
                Role::Knight,
                Role::Bishop,
                Role::Rook,
                Role::Queen,
            ] {
                let w = board.by_piece(role.of(Color::White)).count() as i32;
                let b = board.by_piece(role.of(Color::Black)).count() as i32;
                mat += (w - b) * vals.piece_value(role);
            }
            material_sum += mat.unsigned_abs() as f64;
        }
        let n = self.entries.len().max(1) as f64;
        DatasetStats {
            entries: self.entries.len(),
            white_wins,
            draws,
            black_wins,
            avg_result: result_sum / n,
            avg_phase: phase_sum / n,
            avg_material_cp: material_sum / n,
        }
    }
}

/// Parses one result token: any float in `[0, 1]` (target scores), or PGN
/// notation (`1-0`, `0-1`, `1/2-1/2`, `1/2`, `draw`).
fn parse_result(tok: &str) -> Option<f64> {
    if let Ok(v) = tok.parse::<f64>() {
        return (0.0..=1.0).contains(&v).then_some(v);
    }
    match tok {
        "1-0" => Some(1.0),
        "0-1" => Some(0.0),
        "1/2-1/2" | "1/2" | "draw" => Some(0.5),
        _ => None,
    }
}

/// One parsed PGN game.
struct PgnGame {
    /// White-relative result; `None` when the game is unfinished (`*`).
    result: Option<f64>,
    /// Optional `[FEN "..."]` setup header.
    fen: Option<String>,
    /// SAN plies, annotations and decorations stripped.
    sans: Vec<String>,
}

/// Extracts result/FEN headers and cleaned SAN plies from one PGN game
/// block (headers + movetext).
fn parse_pgn_game(block: &str) -> anyhow::Result<PgnGame> {
    let mut result_tok: Option<String> = None;
    let mut fen: Option<String> = None;
    let mut movetext = String::new();
    let mut in_headers = true;
    for line in block.lines() {
        // `;` starts a comment that runs to the end of the line.
        let line = line.split(';').next().unwrap_or("");
        let t = line.trim();
        if in_headers {
            if t.starts_with('[') {
                if let Some(v) = tag_value(t, "Result") {
                    result_tok = Some(v);
                }
                if let Some(v) = tag_value(t, "FEN") {
                    fen = Some(v);
                }
                continue;
            }
            if t.is_empty() {
                continue;
            }
            in_headers = false;
        }
        if !t.is_empty() {
            movetext.push_str(t);
            movetext.push(' ');
        }
    }
    let movetext = strip_comments(&movetext);
    let movetext = strip_variations(&movetext);
    let mut sans = Vec::new();
    for raw in movetext.split_whitespace() {
        // Result markers (`1-0`, `0-1`, `1/2-1/2`, plain scores) are
        // recognised on the *raw* token: trimming move numbers first would
        // mangle them (`1-0` → `-0`, `0-1` → `-1`).
        if parse_result(raw).is_some() {
            continue;
        }
        if let Some(nag) = raw.strip_prefix('$') {
            if nag.chars().all(|c| c.is_ascii_digit()) {
                continue; // NAG annotation
            }
        }
        let tok = raw.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        if tok.is_empty() || tok == "--" {
            continue; // bare move number / null move
        }
        let tok = tok.strip_suffix("e.p.").unwrap_or(tok);
        let tok = tok.trim_end_matches(['+', '#', '!', '?']);
        if !tok.is_empty() {
            sans.push(tok.to_string());
        }
    }
    Ok(PgnGame {
        result: result_tok.as_deref().and_then(parse_result),
        fen,
        sans,
    })
}

/// Returns the quoted value of a `[Key "Value"]` header line.
fn tag_value(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix('[')?.trim_start();
    let rest = rest.strip_prefix(key)?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start().strip_prefix('"')?;
    let end = rest.rfind('"')?;
    Some(rest[..end].to_string())
}

/// Removes `{ ... }` comments (a dangling `{` drops the rest).
fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut skip = false;
    for c in s.chars() {
        match c {
            '{' => skip = true,
            '}' => skip = false,
            _ if !skip => out.push(c),
            _ => {}
        }
    }
    out
}

/// Removes nested `( ... )` RAV (variation) sections.
fn strip_variations(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_loss_is_bounded_and_finite() {
        let params = EvalParams::default();
        let ds = DataSet::anchors();
        assert!(!ds.is_empty());
        let l = ds.texel_loss(&params);
        assert!(
            l.is_finite() && l > 0.0,
            "anchors must not fit perfectly: {l}"
        );
        // Each term is in [0, 1], so the mean can never exceed 1.
        assert!(l <= 1.0);
    }

    #[test]
    fn selfplay_dataset_covers_the_whole_game() {
        // A tiny synthetic game record: 3 plies, white wins. The moves are
        // collected by actually playing the line, so each UCI string is legal
        // in context.
        let pos = crate::board::Position::startpos();
        let (p2, m1) = pos.play_uci("e2e4").unwrap();
        let (p3, m2) = p2.play_uci("e7e5").unwrap();
        let (_p4, m3) = p3.play_uci("g1f3").unwrap();
        let record = GameRecord {
            moves: vec![m1, m2, m3],
            outcome: crate::selfplay::Outcome::WhiteWin,
            termination: crate::selfplay::Termination::Checkmate,
            opening: "test".to_string(),
        };
        // The record is replayed from startpos in order, so the dataset walks
        // the real line.
        let ds = DataSet::from_selfplay_games(&[record]);
        // startpos + 3 positions after each move.
        assert_eq!(ds.entries.len(), 4);
        for e in &ds.entries {
            assert_eq!(e.result, 1.0, "all positions inherit the game result");
        }
    }

    #[test]
    fn loss_improves_when_material_toward_the_target() {
        // A K+P vs K position is a won game for White: target 1.0. The
        // evaluation is dominated by the (tunable) pawn value, so severely
        // undervaluing the pawn must push the loss up. (The exact structure
        // noise is irrelevant: both losses shift by the same constant.)
        let ds = DataSet {
            entries: vec![EvalEntry {
                pos: Position::from_fen("8/8/8/8/8/2k5/2P5/2K5 w - - 0 1").unwrap(),
                result: 1.0,
            }],
            seed: 0,
            source: "test".to_string(),
        };
        let mut params = EvalParams::default();
        let l_base = ds.texel_loss(&params);
        params.piece_values[0] = 30; // severely undervalue the pawn
        let l_shifted = ds.texel_loss(&params);
        assert!(
            l_shifted > l_base,
            "undervaluing the pawn must worsen the win fit: {l_shifted} vs {l_base}"
        );
    }

    #[test]
    fn shuffle_is_deterministic_per_seed() {
        let ds_a = DataSet::anchors();
        let ds_b = DataSet::anchors();
        let mut r1 = SplitMix64(5);
        let mut r2 = SplitMix64(5);
        let mut a = ds_a.clone();
        let mut b = ds_b.clone();
        a.shuffle_with(&mut r1);
        b.shuffle_with(&mut r2);
        for (x, y) in a.entries.iter().zip(b.entries.iter()) {
            assert_eq!(x.pos.fen(), y.pos.fen(), "same seed → same shuffle");
            assert_eq!(x.result, y.result);
        }
    }

    #[test]
    fn parse_result_accepts_targets_and_pgn_notation() {
        assert_eq!(parse_result("0"), Some(0.0));
        assert_eq!(parse_result("0.5"), Some(0.5));
        assert_eq!(parse_result("0.8"), Some(0.8), "fractional target scores");
        assert_eq!(parse_result("1"), Some(1.0));
        assert_eq!(parse_result("1-0"), Some(1.0));
        assert_eq!(parse_result("0-1"), Some(0.0));
        assert_eq!(parse_result("1/2-1/2"), Some(0.5));
        // Out-of-range floats and everything else are rejected.
        assert_eq!(parse_result("1.5"), None);
        assert_eq!(parse_result("-0.1"), None);
        assert_eq!(parse_result("e4"), None);
        assert_eq!(parse_result("*"), None);
    }

    #[test]
    fn split_partitions_deterministically_per_seed() {
        let ds = DataSet::anchors();
        let (t1, e1) = ds.train_test_split(0.5, 42);
        let (t2, e2) = ds.train_test_split(0.5, 42);
        assert_eq!(t1.len() + e1.len(), ds.len());
        assert_eq!(t1.len(), e1.len(), "50% split of 8 entries");
        // The fraction is the TEST share: a 0.25 split leaves 6 train / 2
        // test (regression for the fraction landing on the wrong side).
        let (tr, te) = ds.train_test_split(0.25, 42);
        assert_eq!(tr.len(), 6, "train keeps 1 - fraction of 8 entries");
        assert_eq!(te.len(), 2, "test receives exactly the fraction");
        // Same seed → identical partition.
        for (a, b) in t1.entries.iter().zip(t2.entries.iter()) {
            assert_eq!(a.pos.fen(), b.pos.fen());
        }
        for (a, b) in e1.entries.iter().zip(e2.entries.iter()) {
            assert_eq!(a.pos.fen(), b.pos.fen());
        }
        // A different seed must produce *some* partition (tiny datasets may
        // collide, so only check the full 8 entries for a real difference
        // over both halves together).
        let (_, e3) = ds.train_test_split(0.5, 43);
        let same = e1
            .entries
            .iter()
            .zip(e3.entries.iter())
            .all(|(a, b)| a.pos.fen() == b.pos.fen())
            && e1.len() == e3.len();
        assert!(!same || ds.len() <= 2, "different seeds must differ here");
    }

    #[test]
    fn stats_count_targets_and_report_ranges() {
        let ds = DataSet::anchors();
        let s = ds.stats();
        assert_eq!(s.entries, 8);
        assert_eq!(s.draws, 3, "three 0.5 targets");
        assert_eq!(s.white_wins, 3, "two 1.0 targets and one 0.8 target");
        assert_eq!(s.black_wins, 2, "two 0.0 targets");
        assert!((0.0..=1.0).contains(&s.avg_result));
        assert!((0.0..=24.0).contains(&s.avg_phase));
        assert!(s.avg_material_cp >= 0.0);
        // Known mean: (0.5+0.5+0+1+0+1+0.8+0.5)/8 = 0.5375.
        assert!((s.avg_result - 0.5375).abs() < 1e-9);
    }

    #[test]
    fn stats_display_is_a_single_line() {
        let ds = DataSet::anchors();
        let text = ds.stats().to_string();
        assert!(!text.contains('\n'));
        assert!(text.contains("8 entries"));
        assert!(text.contains("avg phase"));
    }
}

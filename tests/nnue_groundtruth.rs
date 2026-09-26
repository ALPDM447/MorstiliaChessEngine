//! The authoritative check: this engine's NNUE must agree with Stockfish 19
//! itself, to the last centipawn, on every position in
//! `tests/data/nnue_groundtruth.json`.
//!
//! The data was produced by a patched Stockfish 19 `sf_19` binary, which was
//! asked to print, for each of the 38 positions:
//!
//! * `psqt[bucket]` — the material term, from the side to move's view;
//! * `pos[bucket]` — the positional term, from the side to move's view;
//! * `final_stm` — the fully blended evaluation, side to move;
//! * `final_white` — the same, from White's view.
//!
//! Nothing here is hand-computed. If this test passes, the file layout, the
//! feature sets, the index formulas, the accumulator, the arithmetic and the
//! blend are all exactly right; it is the one test that can prove that.

use std::path::Path;
use std::sync::OnceLock;

use morstilia::board::Position;
use morstilia::nnue::accumulator::Accumulator;
use morstilia::nnue::board::Board;
use morstilia::nnue::network::{Network, load_network};
use morstilia::nnue::types::Color;
use morstilia::nnue::{DEFAULT_NET_FILE, EvalMeta, LAYER_STACKS, blend, evaluate};

/// One bucket of a Stockfish `NNUE network contributions` table row.
#[derive(Debug, serde::Deserialize)]
struct Bucket {
    psqt: i32,
    pos: i32,
    tot: i32,
}

#[derive(Debug, serde::Deserialize)]
struct Truth {
    fen: String,
    buckets: Vec<Bucket>,
    final_stm: i32,
    final_white: i32,
}

fn net() -> &'static Network {
    static NET: OnceLock<Network> = OnceLock::new();
    NET.get_or_init(|| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("nnue")
            .join(DEFAULT_NET_FILE);
        load_network(&path).expect("the bundled net must load")
    })
}

fn truth() -> &'static Vec<Truth> {
    static TRUTH: OnceLock<Vec<Truth>> = OnceLock::new();
    TRUTH.get_or_init(|| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("data")
            .join("nnue_groundtruth.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        serde_json::from_str(&text).expect("ground truth must be valid JSON")
    })
}

/// Refreshes both perspectives and evaluates, exactly as the search does for a
/// node it has no valid incremental state for.
fn full_eval(fen: &str) -> (Vec<(i32, i32)>, i32) {
    let net = net();
    let pos = Position::from_fen(fen).unwrap();
    let board = Board::from_position(&pos);
    let mut acc = Accumulator::new();
    for p in Color::ALL {
        acc.refresh(p, &board, net);
    }
    let meta = EvalMeta::of(&pos, &board);
    let buckets = (0..LAYER_STACKS)
        .map(|b| {
            let out = net.evaluate(&acc, meta.stm, b);
            (out.psqt, out.positional)
        })
        .collect();
    (
        buckets,
        blend(net.evaluate(&acc, meta.stm, meta.bucket()), &meta),
    )
}

#[test]
fn every_bucket_matches_stockfish_19() {
    let data = truth();
    assert_eq!(data.len(), 38, "the suite must not silently shrink");
    let mut bad = Vec::new();
    for t in data {
        let (buckets, _) = full_eval(&t.fen);
        for (b, expect) in t.buckets.iter().enumerate() {
            let (psqt, pos) = buckets[b];
            if psqt != expect.psqt || pos != expect.pos || psqt + pos != expect.tot {
                bad.push(format!(
                    "{} bucket {b}: psqt {psqt} vs {}, pos {pos} vs {}",
                    t.fen, expect.psqt, expect.pos
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} bucket mismatches against Stockfish:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn the_blended_score_matches_stockfish_19() {
    let data = truth();
    let mut bad = Vec::new();
    for t in data {
        let net = net();
        let pos = Position::from_fen(&t.fen).unwrap();
        let board = Board::from_position(&pos);
        let mut acc = Accumulator::new();
        for p in Color::ALL {
            acc.refresh(p, &board, net);
        }
        let stm = evaluate(net, &pos, &acc);
        if stm != t.final_stm {
            bad.push(format!("{}: stm {stm} vs {}", t.fen, t.final_stm));
        }
        let white = if pos.turn() == shakmaty::Color::White {
            stm
        } else {
            -stm
        };
        if white != t.final_white {
            bad.push(format!("{}: white {white} vs {}", t.fen, t.final_white));
        }
    }
    assert!(
        bad.is_empty(),
        "{} blended-score mismatches against Stockfish:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

//! Evaluation-speed benchmark: nanoseconds per classical evaluation over a
//! fixed five-position battery (opening, tactical middlegame, quiet
//! middlegame, KP endgame, KR endgame).
//!
//! Run with `cargo bench --bench eval`. Each criterion iteration evaluates
//! all five positions, so divide the reported time by 5 for per-evaluation
//! numbers:
//!
//! * `evaluate/defaults` — the search hot path (`Evaluator::evaluate`);
//! * `evaluate/params-ref` — the explicit-parameter path the tuner and the
//!   match tooling use (`evaluate_with`);
//! * `evaluate/parts-breakdown` — the instrumentation path behind
//!   `info string estat` / `--evaluate` (all ten components).

use criterion::{Criterion, criterion_group, criterion_main};
use morstilia::board::Position;
use morstilia::evaluation::{EvalParams, Evaluator};

const FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/1p2bppp/p2pbn2/4p3/4P3/1PN1BP2/PQPP2PP/R3KB1R w KQ - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/8/8/8/8/2k5/3R4/2K5 w - - 0 1",
];

fn bench_eval(c: &mut Criterion) {
    let positions: Vec<Position> = FENS
        .iter()
        .map(|f| Position::from_fen(f).expect("battery FEN"))
        .collect();
    let params = EvalParams::default();
    let ev = Evaluator;

    c.bench_function("evaluate/defaults (x5)", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for p in &positions {
                acc += ev.evaluate(p);
            }
            criterion::black_box(acc)
        })
    });

    c.bench_function("evaluate/params-ref (x5)", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for p in &positions {
                acc += ev.evaluate_with(p, &params);
            }
            criterion::black_box(acc)
        })
    });

    c.bench_function("evaluate/parts-breakdown (x5)", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for p in &positions {
                acc += ev.evaluate_parts_with(p, &params).final_score;
            }
            criterion::black_box(acc)
        })
    });
}

criterion_group!(benches, bench_eval);
criterion_main!(benches);

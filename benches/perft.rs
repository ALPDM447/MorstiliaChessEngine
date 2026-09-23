//! Perft micro-benchmark: raw move-generation / make-move throughput on
//! known positions. Run with `cargo bench --bench perft`.

use criterion::{Criterion, criterion_group, criterion_main};
use morstilia::board::Position;

fn perft_startpos(c: &mut Criterion) {
    for depth in 1..=5u32 {
        let pos = Position::startpos();
        c.bench_function(&format!("perft startpos depth {depth}"), move |b| {
            b.iter(|| pos.perft(depth));
        });
    }
}

fn perft_kiwipete(c: &mut Criterion) {
    for depth in 1..=4u32 {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .expect("valid FEN");
        c.bench_function(&format!("perft kiwipete depth {depth}"), move |b| {
            b.iter(|| pos.perft(depth));
        });
    }
}

criterion_group!(benches, perft_startpos, perft_kiwipete);
criterion_main!(benches);

//! Search-speed benchmark: full iterative-deepening searches at fixed
//! depths, reporting nodes per second. Run with `cargo bench --bench nps`.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use criterion::{Criterion, criterion_group, criterion_main};
use morstilia::board::Position;
use morstilia::search::{Searcher, TimeLimit};

fn bench_search(c: &mut Criterion, name: &str, fen: &str, depth: i32) {
    let pos = Position::from_fen(fen).expect("valid FEN");
    let mut searcher = Searcher::new(64);
    let stop = Arc::new(AtomicBool::new(false));
    let limits = TimeLimit {
        depth: Some(depth),
        ..TimeLimit::unlimited()
    };
    c.bench_function(name, |b| {
        b.iter(|| {
            let r = searcher.search(&pos, &[], &limits, &stop, 1, &[]);
            criterion::black_box(r.nodes)
        });
    });
}

fn search_startpos(c: &mut Criterion) {
    bench_search(
        c,
        "search startpos depth 14",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        14,
    );
}

fn search_kiwipete(c: &mut Criterion) {
    bench_search(
        c,
        "search kiwipete depth 14",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        14,
    );
}

fn search_complex_middlegame(c: &mut Criterion) {
    bench_search(
        c,
        "search middlegame depth 16",
        "r2q1rk1/1p2bppp/p2pbn2/4p3/4P3/1PN1BP2/PQPP2PP/R3KB1R w KQ - 0 1",
        16,
    );
}

criterion_group!(
    benches,
    search_startpos,
    search_kiwipete,
    search_complex_middlegame
);
criterion_main!(benches);

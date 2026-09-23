# Morstilia

Morstilia is a classical (pre-NNUE) UCI chess engine written in pure Rust.

It uses [`shakmaty`](https://crates.io/crates/shakmaty) for chess rules and move generation.

## Features

* Alpha-beta search with PVS
* Iterative deepening
* Quiescence search
* Transposition table
* Null-move pruning
* LMR, RFP, razoring and ProbCut
* SEE and history-based move ordering
* Classical tapered evaluation
* Multi-threaded root search
* Polyglot opening book support
* UCI protocol support
* Perft testing

## Requirements

* Stable Rust
* Cargo
* Rust edition 2021

## Build

Clone the repository and run:

```bash
cargo build --release
```

The optimized engine binary will be created at:

```text
target/release/morstilia
```

## Run

Start the engine in UCI mode:

```bash
./target/release/morstilia
```

The engine reads UCI commands from `stdin`.

For example:

```text
uci
isready
position startpos
go depth 10
quit
```

## CLI

Run a fixed-depth search:

```bash
./target/release/morstilia --fen "<FEN>" --depth 10
```

Run perft:

```bash
./target/release/morstilia --perft 5
```

Run the built-in benchmark:

```bash
./target/release/morstilia --bench --depth 13
```

Show engine information:

```bash
./target/release/morstilia --version
```

## Tests

Run the complete test suite:

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## License

See `LICENSE`.

# Morstilia

Morstilia is a UCI chess engine written in Rust. Version 7 evaluates positions
with the NNUE network that ships with Stockfish 19 (`nn-1a298aa575a0`), run by
Morstilia's own integer implementation of the network. The hand-written
classical evaluation from earlier versions is still available as an option.

Chess rules and move generation come from
[`shakmaty`](https://crates.io/crates/shakmaty).

## What is new in 7.0.0

* **NNUE evaluation.** The Stockfish 19 net is embedded in the binary and is
  the default evaluator. It matches Stockfish's own scalar arithmetic exactly
  (checked by the `tests/nnue*.rs` ground-truth tests).
* **Check handling in search fixed.** Quiescence no longer stands pat while
  in check, and futility, SEE and history pruning no longer skip moves that
  give check. Razoring now returns only when quiescence confirms the fail-low,
  so quiet checks at shallow depth are still searched. Before the fix the
  engine could miss forced lines and hang pieces after a check.
* **License changed to GPL-3.0**, to match the embedded Stockfish net.

Informal check against Morstilia 6.0.0: 20 games at 10+0.1, one thread,
16 MB hash, books off, ten openings with colours swapped. Morstilia 7 with
NNUE scored 14 wins, 2 losses and 4 draws (16/20). Twenty games are a rough
signal, not a precise Elo measurement.

The NNUE forward pass is plain scalar code without SIMD, so the engine
searches about 20 times fewer nodes per second with NNUE than with the
classical evaluation. It is still clearly stronger.

## Features

* Principal variation search with iterative deepening and aspiration windows
* Quiescence search with full check evasions
* Lock-free shared transposition table
* Null-move pruning, late move reductions, reverse futility pruning,
  razoring, ProbCut, futility, SEE and history pruning
* Move ordering by TT move, MVV-LVA, SEE, killers and history
* NNUE evaluation (Stockfish 19 net) or a tunable classical evaluation
* Lazy SMP multi-threading (`Threads` 1 to 1024)
* Polyglot opening books
* Syzygy endgame tablebases (WDL and DTZ, through
  [`shakmaty-syzygy`](https://crates.io/crates/shakmaty-syzygy))
* MultiPV and pondering
* Perft, a benchmark, evaluation tuning and a match driver

## Building

You need stable Rust (edition 2024) and Cargo.

The NNUE network is compiled into the binary but is too large to keep in git.
Before building, place it at `nnue/nn-1a298aa575a0.nnue` in the repository
root. The optional opening book goes at `book/book.bin`. Both files are
distributed with the release.

```text
MorstiliaChessEngine/
├── nnue/
│   └── nn-1a298aa575a0.nnue
└── book/
    └── book.bin
```

Then build:

```bash
cargo build --release
```

This produces `target/release/morstilia` (the engine), `target/release/tune`
(evaluation tuning) and `target/release/selfplay` (engine-vs-engine matches;
`morstilia-selfplay` is the same program).

## Running

Start the engine and talk to it over UCI, or load it into any UCI GUI:

```bash
./target/release/morstilia
```

```text
uci
isready
position startpos
go depth 12
quit
```

### UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 64 | Transposition table size in MB |
| `Threads` | 1 | Search threads (Lazy SMP) |
| `Eval` | `nnue` | `nnue` or `classical` |
| `NNUEFile` | empty | Load a different `.nnue` file instead of the embedded net |
| `EvalParamsPath` | empty | TOML parameter file for the classical evaluation |
| `BookEnabled` | true | Use the opening book |
| `BookPath` | empty | Book file or directory; empty means auto-detect |
| `SyzygyPath` | empty | Directory holding Syzygy `.rtbw`/`.rtbz` files |
| `SearchDepth` | 0 | Fixed search depth; 0 means no limit |
| `MultiPV` | 1 | Number of principal variations to report (1 to 4) |
| `Ponder` | false | Allow pondering |
| `Debug` | false | Extra diagnostic output |

If `NNUEFile` cannot be loaded, the engine reports the error and falls back
to the classical evaluation.

### Opening book

The engine reads Polyglot `.bin` books. With `BookPath` empty it looks for
`book/book.bin`, then `book/AllOpeningsMorstilia.bin`, first next to the
executable and then in the working directory. Missing or corrupt books are
reported on stderr and never stop the engine. The book is used only for
ordinary `go` commands, not for `go infinite`, pondering or `searchmoves`.

### Syzygy tablebases

```text
setoption name SyzygyPath value /path/to/syzygy
```

Tables are probed at quiescence leaves when the material fits the loaded set,
and the DTZ-best move is tried first at the root. An empty or unreadable path
simply disables probing.

## Command line

```bash
./target/release/morstilia --fen "<FEN>" --depth 12 --eval nnue   # one-shot search
./target/release/morstilia --perft 5                               # perft from startpos
./target/release/morstilia --bench --depth 13                      # benchmark
./target/release/morstilia --evaluate "<FEN>"                      # classical eval breakdown
./target/release/morstilia --version
```

Searches from the command line accept `--threads`, `--hash`, `--syzygy`,
`--eval classical|nnue`, `--nnue <file>` and `--eval-params <file>`. They use
the classical evaluation unless `--eval nnue` is given, so bench node counts
stay comparable with earlier versions. With one thread the search is fully
deterministic.

## Tuning and matches

The classical evaluation has 864 parameters in `config/baseline_eval.toml`.
`tune` fits them with SPSA against self-play or PGN data, and `selfplay` plays
seeded matches between parameter sets or search settings, reporting
wins/draws/losses, Elo with a 95% confidence interval and an SPRT decision.

```bash
./target/release/tune --games 8 --depth 4 --iterations 500 \
    --subset material,mobility --out tuned.toml
./target/release/selfplay --games 40 --depth 6 \
    --white tuned.toml --black config/baseline_eval.toml
```

Run either program with `--help` for all options. Results from the earlier
tuning experiments are in [`docs/STAGE5_RESULTS.md`](docs/STAGE5_RESULTS.md).

## Tests

```bash
cargo test --release
cargo bench
```

## License

Copyright (C) 2026 Alp Dumlupınar

Morstilia is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See `LICENSE` for the full text.

Versions up to 6.0.0 were released under the MIT license.

The NNUE network `nn-1a298aa575a0.nnue` was trained by the
[Stockfish](https://github.com/official-stockfish/Stockfish) developers and
is used under the GNU General Public License v3.0.

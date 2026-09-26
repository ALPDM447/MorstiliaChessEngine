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
* Lazy-SMP shared multi-threading (`Threads` 1–1024, lock-free shared TT)
* Polyglot opening book support
* **Syzygy endgame tablebases** (WDL + DTZ, 3–7 pieces via
  [`shakmaty-syzygy`](https://crates.io/crates/shakmaty-syzygy)), probed at
  quiescence leaves and used to order/verify root moves
* UCI protocol support
* Perft testing
* **Tunable evaluation** — every classical constant lives in one TOML file
  (`config/baseline_eval.toml`, 864 parameters), swappable via UCI
  `EvalParamsPath` or `--eval-params`
* **SPSA tuning** (`tune`) — deterministic, seeded, checkpointed
  Texel-style tuning over self-play data, with regression protection
* **Strength measurement** (`selfplay`) — W/D/L, Elo with 95%
  confidence interval and a trinomial SPRT; never NPS/depth node counts

## Requirements

* Stable Rust
* Cargo
* Rust edition 2024

## Build

Clone the repository and run:

```bash
cargo build --release
```

The optimized engine binary will be created at:

```text
target/release/morstilia
```

The Stage-5 tools add two more binaries:

```text
target/release/tune      # SPSA evaluation tuning
target/release/selfplay  # seeded self-play + W/D/L/Elo/SPRT
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

### Loading a tuned parameter set

Point UCI at a tuned TOML file:

```text
setoption name EvalParamsPath value tuned.toml
position startpos
go depth 10
```

or pass it on the command line for one-shot searches / the bench:

```bash
./target/release/morstilia --fen "<FEN>" --depth 10 --eval-params tuned.toml
./target/release/morstilia --bench --depth 11 --eval-params tuned.toml
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

The engine-vs-engine match driver is a separate binary:

```bash
./target/release/morstilia-selfplay --help
```

(`selfplay` and `morstilia-selfplay` are the same program; see
"Engine-vs-engine matches (Stage 8)" below for the full command reference.)

## Multi-threaded search (Lazy SMP)

The search engine uses **Lazy SMP**: every worker reads and searches the
same full, deterministically-ordered root move list, and a shared lock-free
transposition table is the *only* coordination channel. The calling thread's
result (best move, score, PV, depth) is authoritative; helper threads
contribute nodes and TT fill only. There is no root-split assignment, no
result merging, and no per-node locking or allocation.

* **`Threads`** (UCI `setoption`, CLI `--threads`, default `1`) selects the
  total search threads. The pool is **persistent**: workers are spawned once
  and parked between searches, keeping their history/killer tables warm.
  `setoption name Threads value N` (or `--bench` with `--threads`) is
  honored between searches; shrinking joins and retires workers.
* **`Hash`** (UCI `setoption`, CLI `--hash`, default `64` MB) sizes the
  shared TT; resizing is atomic (the table is swapped behind an `Arc`) and
  safe even between searches.
* `Threads = 1` is a fully inline fast path with **zero** parallel overhead
  and is byte-for-byte deterministic — identical node counts and best moves
  run after run (the regression tool and `--bench` rely on this).
* Finished multi-threaded searches append `info string smp ...` lines to the
  UCI stream: worker count, root-task total, shared-TT hit/cutoff rates, and
  per-worker nodes/time/depth (plus a best-effort idle estimate). `Threads =
  1` output is unchanged, so single-thread transcripts stay byte-identical.

Scaling is measured with the bench, never interpreted as Elo:

```bash
./target/release/morstilia --bench --depth 11 --threads 1 --hash 64
./target/release/morstilia --bench --depth 11 --threads 2 --hash 64
./target/release/morstilia --bench --depth 11 --threads 4 --hash 64
./target/release/morstilia --bench --depth 11 --threads 8 --hash 64
```

Report per-thread count the aggregate `total nodes … time … nps …` line;
with `--stats` you also get shared-TT hit/cutoff and per-worker balances.
Gains are NPS/speedup numbers only — see the self-play section for honest
strength measurement.

Representative scaling (`--bench --depth 11 --hash 64`, release build, on a
16-core machine; the search self-limits at depth 11 so wall times are the
time-to-depth cost per position):

```text
threads   total nodes     wall    NPS        speedup (NPS)   efficiency
1         623,535         985ms   633k       1.00x           100%
2         1,236,375       1093ms  1,131k     1.79x           89%
4         2,239,960       962ms   2,328k     3.68x           92%
8         3,493,659       713ms   4,900k     7.74x           97%
```

Read these carefully: Lazy SMP *grows* total nodes with worker count (all
workers search the full tree and share TT work), so NPS here is aggregated
nodes per wall second and **overstates** the strength-relevant gain. The
honest signal is time-to-depth for a single position (kiwipete, depth 11):
`0.78s → 0.78s → 0.69s → 0.56s` at `1/2/4/8` threads — typical Lazy-SMP
scaling. For real strength, run a `selfplay` match (`Threads=1` vs
`Threads=N` with `--threads`).

Print the per-component evaluation breakdown of a position (clean single
line, no search):

```bash
./target/release/morstilia --evaluate "<FEN>"
```

Export the baseline evaluation parameters (the seed material for tuning):

```bash
./target/release/morstilia --export-params config/baseline_eval.toml
```

Show engine information:

```bash
./target/release/morstilia --version
```

## Opening book

The engine reads standard Polyglot `.bin` books. By default it auto-detects a
book next to the executable or in the working directory
(`book/book.bin`, then `book/AllOpeningsMorstilia.bin`); `BookPath` points at
a specific file or directory, and `BookEnabled` (default `true`) switches the
book off entirely. Book errors (missing/corrupt files) are reported on
stderr and are never fatal.

Book moves are legal-filtered and chosen by weight (deterministic
highest-weight selection for `best_move`, seeded weighted selection for
variety). The book is consulted only for ordinary `go` commands — never for
`go infinite`, ponder, or `searchmoves`, and never by the CLI one-shot
searches. When a position has no book entry the engine simply searches.

## Syzygy endgame tablebases

Point the engine at a directory of Syzygy tables (`.rtbw` + `.rtbz`, both
WDL and DTZ, from a provider such as
[lichess](https://tablebase.lichess.ovh/tables/standard/3-4-5/) or
[tablebase.sesse.net](https://tablebase.sesse.net)):

```text
setoption name SyzygyPath value /path/to/syzygy
```

* Positions are probed only when the material fits the loaded set (the
  largest table piece count is auto-detected), so midgame trees never touch
  the tablebase at all.
* Probes run at **quiescence** leaves; a definitive result ends the node —
  an unconditional win/loss scores a fixed band (`±19,999`), a 50-move-fragile
  "cursed" win / "blessed" loss scores `±10,000`, a draw `0`. Mate scores
  (`is_mate`) always come from the search's own mate detection, never from a
  tablebase win claim, and `Threads = 1` stays byte-for-byte deterministic
  (a sealed probe at a leaf is a pure function of the position).
* At the **root**, the DTZ-optimal move of a decisive position is searched
  first (fastest win / longest resistance under the 50-move rule), winning
  moves precede draws, and losing moves come last. Root selection still
  confirms every move with the search.
* The 50-move counter is respected end to end: the search's own
  draw detection runs before leaf probing, and WDL resolution never
  overstates a result (ambiguous "maybe" edges are read conservatively).
* An empty, missing or unreadable `SyzygyPath` degrades to an inert
  tablebase — the engine simply stops probing; load warnings and a summary
  (`loaded N table file(s) (max M pieces)`) go to stderr.
* CLI one-shot searches and the bench accept `--syzygy <dir>`:
  `--fen "<FEN>" --depth N --syzygy <dir>`. Without `--syzygy` the search is
  byte-for-byte identical to a run with no tables (bench parity is preserved).
  With `--bench --stats --syzygy`, a `tb probe / hit / win / draw / loss /
  cursed` line is added per position once probing actually happens.

Example — a KQvK win reported as a centipawn band (never a mate claim):

```text
position fen 4k3/8/8/8/8/8/8/3QK3 w - - 0 1
go depth 4
    info depth 4 score cp 19999 nodes 272 nps 34000 time 8 pv d1d5 e8e7 d5d1 e7e6
    bestmove d1d5
```

## Evaluation parameters

All tunable constants — material values, the 12 piece-square tables,
mobility weights, pawn structure, passed-pawn bonuses, king safety, piece
activity, threats and the phase weights — live in a serializable
`EvalParams` (864 parameters). `EvalParams::default()` reproduces the
pre-Stage-5 hardcoded evaluation *exactly*; the shipped baseline is
`config/baseline_eval.toml`. The search hot path reads the parameters from a
shared reference owned by the searcher (no per-node copying), so
`Threads=1` stays fully deterministic for any parameter set.

The tuning tools read and write the same TOML format, so a tuned export can
be dropped straight into the engine via `EvalParamsPath` / `--eval-params`.

## Self-play and strength measurement

`selfplay` plays a seeded, fixed-depth match between two parameter sets (by
default the baseline against itself) and reports the strength signal:

```bash
./target/release/selfplay --games 20 --depth 5 \
    --white tuned.toml --sprt-elo0 -2 --sprt-elo1 3 --out match.pgn
```

```text
match: white tuned.toml vs black baseline (white is the candidate)
games: 20  seed: 1  depth: 5  tt: 16 MB
W11 D8 L1 (20 games)
score 0.750  elo +190.8  95% CI [+21.8, +359.9]
sprt: elo0 -2 elo1 3  llr +0.184  games 20  decision Running
dataset: 2848 entries from 20 games (seed 1, depth 5) -> pipe into tune
```

* Strength comes only from what happens on the board: **W/D/L → score rate →
  logistic Elo**, a **95% Wilson confidence interval** on that Elo, and a
  **trinomial SPRT** (`--candidate white|black`) with the classic
  `elo0`/`elo1` error bounds. NPS/depth/node totals are never reported as
  strength.
* Every game is a pure function of `(white params, black params, seed,
  config)` — only the opening selection uses randomness — so the same seed
  replays the identical match. The final line shows how many `EvalEntry`
  rows the match feeds into the tuner.
* `--threads N` searches each move with N threads per side (default `1`).
  This is the tool for comparing `Threads = 1` vs `Threads = N` strength;
  note that `N > 1` makes games **non-deterministic**, so never use it to
  build tuning datasets (those must stay at the default).

The Stage-5 style run above (`selfplay --games 20 --depth 5`) is an
**intermediate signal only**: it cannot separate a real improvement from
noise. Use the Stage-8 match driver below for anything that goes in a
changelog or decides a merge.

## Engine-vs-engine matches (Stage 8)

`selfplay` (alias `morstilia-selfplay`) is the reproducible match driver: two
engine configurations — different eval files, thread counts, hash sizes,
search settings or (one day) NNUE weights — play a deterministic match
head-to-head. It generalizes the Stage-5 fixed-depth self-play: seeded
opening shuffle becomes a seeded **opening suite**, depth-only control
becomes full **time controls**, and spot checks become honest match runs with
statistics.

```bash
./target/release/morstilia-selfplay --games 1000 --depth 10 --seed 42 \
    --white tuned.toml --black config/baseline_eval.toml \
    --parallel 8 --report results/a10.json
```

### What a run reports

```text
match: tuned vs baseline  (candidate: tuned)
suite: classical-v1 (48 positions, seeded order from seed 42)
time control: depth 10  threads 1 / 1  tt 16 / 16 MB
games: 1000 completed of 1000 requested  colors alternate: true
W412 D399 L189 (1000 games)
score 0.612  elo +78.8  95% CI [+46.1, +111.5]
avg game length 92.4 plies
terminations: checkmate: 288, threefold-repetition: 401, move-limit: 311
baseline: frozen as Stage8-Classical-SMP-Baseline
sprt: elo0 -2 elo1 3  llr +3.4  games 214  decision AcceptH1
```

* Strength is **W/D/L → score rate → logistic Elo** with a **95% Wilson
  confidence interval** and a **trinomial SPRT** — never NPS/depth/node
  totals, and never quoted from tiny samples (a 20-game run stays an
  intermediate signal).
* Every game is a pure function of the configuration and the seed: one seed
  → one 48-position order → byte-identical games at `Threads = 1` depth
  control (enforced by tests, and reproducible across resume).
* The JSON `--report` carries the same stats plus the full per-game detail
  (every UCI move, outcome, termination), both sides' engine params and
  `Threads`/`Hash`/Syzygy settings, and `source.source_fingerprint` — the
  exact source revision — so a report is self-describing forever.

### Time controls and search settings

`--tc` accepts `depth=N`, `movetime=S` (fixed seconds per move) or classic
`M/B+I` (e.g. `40/10+0.1`). `--depth N` is shorthand for `depth=N`. Per side:
`--white-threads`/`--black-threads` (default `--threads 1`) and
`--white-hash`/`--black-hash` (default `--hash 16` MB). `--parallel N` plays
up to N games concurrently; at `Threads = 1` depth control the games come out
**identical** to sequential (per-game seeds are scheduling-independent).

### The opening suite

`--suite classical-v1` (the only built-in) is a fixed, duplicate-free set of
48 legal positions — 24 White-to-move and 24 Black-to-move — spanning open,
semi-open, closed and gambit lines as well as transposition-safe setups. It
has **nothing to do with the Polyglot opening book** (`--book`/`book/*.bin`),
which only affects UCI `go`. Games consume the seeded Fisher–Yates order and
cycle when `--games` exceeds 48, and `--alternate on` (default) swaps colors
every game so both sides play both colors.

### SPRT auto-stop

`--sprt --sprt-elo0 -2 --sprt-elo1 3` (defaults `-2`/`3`, `alpha`/`beta`
0.05, `draw_elo` 100) runs a trinomial sequential test: as soon as the
log-likelihood ratio crosses a bound the match stops with `AcceptH1`,
`AcceptH0` or the `max_games` cap. Set `--sprt-elo0 -100 --sprt-elo1 100`
for a decisive-gap gate; the decision is recorded in the report.

### Reproducibility, resume, baseline freeze

* `--seed N` replays the identical match; save it with `--report`.
* `--resume match.json` continues an aborted match from its completed games
  byte-identically. A mismatched seed/suite/time control/engine/SPRT
  configuration is **refused** rather than silently mixed.
* `--record-baseline match.json` stamps the report
  `baseline: Stage8-Classical-SMP-Baseline` — the frozen classical+SMP
  baseline: `config/baseline_eval.toml`, `Threads 1`, `Hash 16 MB`, suite
  `classical-v1`, documented game count, result and Elo+CI below. The
  baseline makes no absolute human-Elo claim; it is the reference for future
  A-vs-B work.

### A-vs-B regression (eval / search / Threads / params / NNUE)

Run each candidate against the **same** shared baseline with the **same**
seed/suite/time control, then compare the two reports:

```bash
./target/release/morstilia-selfplay --games 1000 --depth 10 --seed 42 \
    --white tuned.toml --candidate white --report results/new.json
./target/release/morstilia-selfplay --games 1000 --depth 10 --seed 42 \
    --white config/baseline_eval.toml --candidate white --report results/old.json
./target/release/morstilia-selfplay --compare results/new.json results/old.json
```

`--compare` prints a machine-readable JSON verdict: score-rate delta with an
independent-binomial 95% CI and `improved` / `similar` / `regressed` —
`incomparable` when the two runs did not share seed/suite/time control, and
`insufficient` when a record has no games. The same workflow compares
`Threads = 1` vs `Threads = N`, search changes, parameter sets and future
NNUE variants.

### The frozen baseline reference

```bash
./target/release/morstilia-selfplay --games 400 --depth 10 --seed 1 \
    --white config/baseline_eval.toml --black config/baseline_eval.toml \
    --report baseline-stage8.json --record-baseline baseline-stage8.json
```

Interpretation: `W/D/L`, `score 0.5`, `elo ≈ 0`, an Elo CI that must contain
0 — a same-engine self-match is a calibration run, not a strength number.

## Evaluation tuning

`tune` builds a dataset by deterministic self-play (baseline vs baseline at
a fixed depth), runs SPSA over the flat parameter vector with two-sided
perturbation and clamped projection, writes checkpoints at fixed iterations,
exports the best set found, and finishes with a regression report:

```bash
./target/release/tune --smoke --seed 42 --iterations 300 \
    --subset material,mobility,pst --checkpoint-every 50 --out tuned.toml
```

```text
dataset: 1041 entries from selfplay 8 game(s) (seed 42)
stats:   1041 entries · targets W/D/L 739/162/140 · avg result 0.788 · avg phase 13.8 · avg |material| 253 cp
subset: material,mobility,pst (783 params selected)
checkpoint     50: best loss 0.117617 -> checkpoints/tuned_it000050.toml
...
spsa: 300 iterations  loss 0.130315 -> 0.107414 (best, 17.6% improvement)  final 0.107991
tuning: updated 670 of 783 params  exported -> tuned.toml
regression mate-in-two    best d2d8  tactic ok deterministic ok nodes (131, 131)  eval  534 ->  562
regression hanging-queen  best f1f2  tactic ok deterministic ok nodes (231, 231)  eval  -37 ->  -63
regression summary: tactics ok determinism ok max eval delta 28 cp avg 27.0 cp
loss drop: 0.0229 over the training set  (1041 entries, seed 42)
```

* The objective is the Texel loss `Σ(σ(K·eval) − y)²` with
  `K = ln(10)/400` over every position of every self-play game, scored by
  the game outcome (white-relative, `1.0/0.5/0.0`).
* Fully deterministic: one seed → one dataset → one SPSA trajectory → byte-
  identical checkpoints and export files.
* `--subset` restricts the active parameters by dotted prefix (`material`,
  `mobility`, `pst`, `pawns`, `passed`, `king`, `activity`, `threats`,
  `space`, `phase`, `bishop_pair`); the default tunes all 864. A first real
  run with a few thousand positions should use tens of thousands of
  iterations (`--iterations 50000`).
* **Regression protection**: after tuning, the tool re-plays the pinned
  tactic suite (mate-in-two at depth 4, hanging queen at depth 3) with the
  tuned set, verifies `Threads = 1` determinism (identical nodes across two
  runs) and prints the per-position baseline-vs-tuned evaluation deltas.
  Validate strength with `selfplay tuned.toml vs baseline`.

### Dataset sources: files, PGN, train/test split

Without `--data`/`--pgn` the dataset comes from seeded baseline-vs-baseline
self-play (fully reproducible per seed). With them the self-play is skipped
and positions are loaded from files instead — equally deterministic:

```bash
# `<result> <FEN>` or `<FEN> <FEN-result>` lines, `#` comments; targets may be
# any value in [0, 1] (0.8 = "clear advantage") or PGN (1-0, 0-1, 1/2-1/2)
./target/release/tune --data tests/data/eval_positions.txt --iterations 200

# every finished game of a PGN file: SAN plies replayed from the initial
# position or the [FEN] setup header; `*`-games are skipped
./target/release/tune --pgn games.pgn --shuffle --test-split 0.2 --seed 1 \
    --iterations 500 --out tuned.toml
```

* Sources may be combined: `--data a.txt --data b.txt --pgn c.pgn`.
* `--shuffle` Fisher–Yates shuffles the loaded entries with a seed derived
  from `--seed` (same seed → same order).
* `--test-split F` (in `(0,1)`) holds out a fraction of the entries and the
  run reports **train and test loss, before and after tuning** — the test
  half is never trained on, so it is the honest generalisation signal
  (a *worse* test loss is flagged as `NO — overfit`).
* Every dataset prints `stats`: entry count, target W/D/L balance, mean
  result, mean game phase (`0..24`) and mean absolute material imbalance —
  a quick sanity check that the data matches the intended distribution.

### Comparing parameter files

```bash
./target/release/tune --compare config/baseline_eval.toml tuned.toml
```

Prints a per-group summary (changed count, max and sum `|Δ|` per parameter
group) followed by every differing parameter as
`name  A -> B (+Δ)`, or `identical: 864 / 864 parameters match` when the
files are equal — the original-vs-tuned review step for every export.

### Tuning recipe

```bash
# 1. export the seed material (already shipped as config/baseline_eval.toml)
./target/release/morstilia --export-params config/baseline_eval.toml

# 2. generate a dataset and tune a small subset (quick)
./target/release/tune --games 8 --depth 4 --iterations 500 \
    --subset material,mobility --out tuned.toml --smoke

# 3. measure honestly against the baseline
./target/release/selfplay --games 40 --depth 6 \
    --white tuned.toml --black config/baseline_eval.toml

# 4. load the tuned set into the engine
./target/release/morstilia --bench --depth 11 --eval-params tuned.toml
```

### Measured results (Stage-5 smoke experiment)

Full tables, exact repro commands and an honest reading of the numbers live in
[`docs/STAGE5_RESULTS.md`](docs/STAGE5_RESULTS.md). Headlines from the
recorded runs (Intel Core Ultra 7 255H, rustc 1.98.1, release):

* **Tuning**: `tune --smoke --seed 42 --iterations 300` on 1041 self-play
  positions dropped the train loss `0.130315 → 0.107414` (−17.6%); the
  held-out test half of the same corpus dropped `0.116436 → 0.092465`
  (no overfit), with the pinned tactics suite and `Threads = 1` determinism
  still green after export.
* **Eval speed**: ≈ **362 ns** per classical evaluation (search path,
  5-position criterion battery ÷ 5); ≈ 329 ns via the explicit-parameter
  path. Hot paths remain allocation-free; parameter values do not change
  code paths.
* **Before/after `--bench --depth 10`** (baseline vs `tuned.toml`): total
  555,787 nodes / 982 ms → 438,381 nodes / 713 ms (−21% nodes, position-
  dependent — startpos actually searched 4.8× more nodes).
* **Strength**: 150-game SPRT match (tuned candidate vs baseline, depth 5,
  seed 7, alternate colours) → W69 D24 L57, score 0.540, **Elo +27.9
  [−27.7, +83.4]**, LLR +0.218, decision `MaxGames`. **Inconclusive**: the
  CI contains zero and the SPRT never approached its bound — no strength gain
  is claimed. A 60-game A/A sanity run (baseline vs baseline) landed at
  −23.2 Elo [−110.4, +64.0], consistent with equal strength.

Stage 5 also found and fixed four real bugs, each now covered by a
regression test: the passed-pawn king-proximity colour asymmetry
(`src/evaluation/passed_pawns.rs`), a wrong-sided `--test-split`, PGN result
markers mis-tokenised in the dataset loader, and bare boolean flags in the
match CLI swallowing the following option (`--sprt --report …` used to fail;
`--tc depth=N`/`movetime=S` now parse as documented). Details in
`docs/STAGE5_RESULTS.md` §1.

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
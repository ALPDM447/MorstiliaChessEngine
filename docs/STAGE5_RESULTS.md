# Stage 5 — Evaluation tuning & strength testing: results

Everything below was produced by the commands shown, on the machine described,
with no manual edits to any number. Re-running the repro block reproduces the
dataset, the tuning trajectory and both match reports byte-for-byte (all
seeding is SplitMix64-derived; `Threads = 1`).

* **Machine**: Intel Core Ultra 7 255H, 16 logical CPUs, Linux
* **Toolchain**: rustc 1.98.1, cargo 1.98.1, release profile
* **Binaries**: `target/release/{morstilia,tune,selfplay}`

---

## 1. Bugs found and fixed during Stage 5

All four were surfaced by the new Stage-5 tests/commands; none rewrote working
search or evaluation architecture.

### 1.1 Colour-mirror asymmetry in the passed-pawn king-proximity term

`tests/stage5.rs::stm_eval_is_colour_mirror_antisymmetric` (new) caught a
genuine pre-existing (Stage-4) asymmetry: ranks are 0-based in shakmaty, and
`evaluate_passed` computed the promotion square as `8 * (8 - rank)` for White
(rank index 8 → always out of bounds → silently fell back to the *pawn's own
square*) and `-8 * (rank - 1)` for Black (one rank short of the promotion
square). The king-proximity scaling therefore saw nonsense distances and
applied asymmetrically.

Probe on `8/8/8/8/8/2k5/2P5/2K5 w - - 0 1` and its colour mirror:

| component | before fix (pos / mirror) | after fix (pos / mirror) |
|---|---|---|
| `passed` | **+10 / −7** ✗ | **+7 / −7** ✓ |
| stm eval | 102 / 99 ✗ | 99 / 99 ✓ |

All other components (`material`, `pst`, `pawn`, `mobility`, `king`, `space`,
`threats`) already negated correctly. Fix: `8 * (7 - rank)` (White) and
`-8 * rank` (Black). This directly matters for tuning: the tuner would
otherwise optimise a term that scores mirrored positions differently.

### 1.2 `--test-split` put the fraction on the wrong side

`train_test_split(f)` produced `train = n·f` instead of `train = n·(1−f)`, so
`--test-split 0.2` trained on 20% and held out 80%. A 0.5-based unit test
masked it (symmetric). Fixed, with asymmetric-fraction assertions added to
`src/tuning/dataset.rs` (0.25 → 6/2 of 8) and `tests/stage5.rs`.

### 1.3 PGN movetext result markers mangled by move-number trimming

The tokenizer trimmed move numbers *before* recognising results: `1-0` → `-0`
(accidentally parsed as the float −0.0 → skipped) but `0-1` → `-1` (not
parsed → pushed as a "SAN" → game flagged `truncated`). Result recognition now
runs on the raw token; `tests/data/sample.pgn` loads as
`(3 games, 0 skipped, 0 truncated)` and the integration test asserts exactly
that string.

### 1.4 Match CLI: bare boolean flags swallowed the next option; `--tc depth=N` rejected

`flag_value` unconditionally consumes the next token, and `--sprt`,
`--no-alternate`, `--no-adjudicate` are valueless — so `--sprt --report x`
failed with `unknown argument: x` (the tool's own `--help` examples were
unrunnable). Also `parse_tc` only accepted `depth 5`/`depth5` although the
usage text documents `depth=N | movetime=S`. Fixed in `src/matchplay/cli.rs`
(inline `name=value` normalisation, boolean pre-arms, separator-trimming
`parse_tc`), guarded by the new end-to-end test
`bare_sprt_flag_does_not_swallow_the_next_option` and extended `tc_parsing`
unit assertions. No existing match behaviour changed.

---

## 2. Datasets (repro: §6, commands A–C)

| run | source | entries | targets W/D/L | avg result | avg phase | avg abs(material) |
|---|---|---|---|---|---|---|
| A | self-play, 8 games, seed 42 | 1041 | 739/162/140 | 0.788 | 13.8 | 253 cp |
| B | run A re-loaded via `--data`, shuffled (seed 7) | 1041 | 739/162/140 | 0.788 | 13.8 | 253 cp |
| C | `tests/data/sample.pgn`, 3 games | 22 | 8/5/9 | 0.477 | 18.5 | 32 cp |

Run B split (`--test-split 0.2 --seed 7`): **train 833 / test 208**, and the
per-half stats prove the halves differ (train avg result 0.792, test 0.772).
The dataset produced by run A round-trips through `--dump-data` → `--data`
losslessly (1042 lines = 1 header + 1041 entries).

---

## 3. Tuning runs (SPSA, Texel loss, K = ln(10)/400)

### Run A — documented smoke recipe (self-play dataset)

```text
dataset: 1041 entries from selfplay 8 game(s) (seed 42)
subset: material,mobility,pst (783 params selected)
checkpoint     50: best loss 0.117617 -> checkpoints/tuned_it000050.toml
checkpoint    100: best loss 0.113583 -> checkpoints/tuned_it000100.toml
checkpoint    150: best loss 0.111198 -> checkpoints/tuned_it000150.toml
checkpoint    200: best loss 0.109686 -> checkpoints/tuned_it000200.toml
checkpoint    250: best loss 0.108919 -> checkpoints/tuned_it000250.toml
checkpoint    300: best loss 0.107414 -> checkpoints/tuned_it000300.toml
spsa: 300 iterations  loss 0.130315 -> 0.107414 (best, 17.6% improvement)  final 0.107991
tuning: updated 670 of 783 params  exported -> tuned.toml
regression mate-in-two    best d2d8  tactic ok deterministic ok nodes (131, 131)  eval  534 ->  562
regression hanging-queen  best f1f2  tactic ok deterministic ok nodes (231, 231)  eval  -37 ->  -63
regression summary: tactics ok determinism ok max eval delta 28 cp avg 27.0 cp
loss drop: 0.0229 over the training set  (1041 entries, seed 42)
```

### Run B — file dataset + shuffle + train/test split (all 864 params)

```text
dataset: shuffled with seed 104373592874730
split:   train 833 / test 208 (fraction 0.2, seed 7)
spsa: 200 iterations  loss 0.133781 -> 0.106518 (best, 20.4% improvement)  final 0.106518
holdout: test loss 0.116436 -> 0.092465 (208 entries, yes better)
tuning: updated 721 of 864 params  exported -> /tmp/opencode/tuned_split.toml
regression summary: tactics ok determinism ok max eval delta 31 cp avg 26.0 cp
```

The holdout line is the honest generalisation signal: the test half was never
trained on, and its loss improved alongside the train loss (no overfit flag).

### Run C — PGN dataset end-to-end

```text
dataset: 22 entries from pgn tests/data/sample.pgn (3 games, 0 skipped, 0 truncated) (seed 1)
spsa: 30 iterations  loss 0.230565 -> 0.173043 (best, 24.9% improvement)  final 0.173043
regression summary: tactics ok determinism ok max eval delta 19 cp avg 16.0 cp
```

### Original-vs-tuned review (`tune --compare`)

```text
compare: config/baseline_eval.toml vs tuned.toml
group summary:
  material       changed    5  max |d|    7.0  sum |d|    21.0
  mobility       changed    9  max |d|   11.0  sum |d|    54.0
  pst            changed  656  max |d|   11.0  sum |d|  1996.0
  material.pawn              100 -> 93 (-7)
  material.knight            320 -> 314 (-6)
  ...
```

and `tune --compare config/baseline_eval.toml config/baseline_eval.toml` →
`identical: 864 / 864 parameters match`.

---

## 4. Benchmarks

### 4.1 Evaluation speed — `cargo bench --bench eval`

Criterion, 5-position battery per iteration (divide by 5 for per-evaluation):

| bench | time (95% CI) | per evaluation |
|---|---|---|
| `evaluate/defaults` (search hot path) | 1.811 µs [1.758, 1.857] | **≈ 362 ns** |
| `evaluate/params-ref` (tuner/match path) | 1.646 µs [1.645, 1.648] | **≈ 329 ns** |
| `evaluate/parts-breakdown` (diagnostics) | 1.642 µs [1.641, 1.643] | **≈ 328 ns** |

The parameter *values* do not change code paths (no value-dependent
branching), so baseline and tuned parameter files evaluate at identical speed;
the three variants differ only by which wrapper is used. Hot paths remain
allocation-free.

### 4.2 Search throughput — `cargo bench --bench nps`

| bench | median | spread |
|---|---|---|
| startpos depth 14 | 4.42 ms | 4.415–4.424 |
| kiwipete depth 14 | 61.9 ms | 56.5–70.8 |
| middlegame depth 16 | 15.8 ms | 11.5–22.4 |

### 4.3 Move generation — `cargo bench --bench perft`

startpos: d3 70.0 µs, d4 1.62 ms, d5 40.0 ms (≈121M nodes/s);
kiwipete: d3 671 µs, d4 31.2 ms (≈131M nodes/s).

### 4.4 Fixed benchmark before/after tuning — `morstilia --bench --stats`

depth 10, threads 1, hash 64 MB; "before" = `config/baseline_eval.toml`,
"after" = `tuned.toml` (run A export):

| position | baseline nodes / ms | tuned nodes / ms | NPS before → after | best before → after |
|---|---|---|---|---|
| startpos | 11,919 / 39 | 57,832 / 134 | 306K → 432K | b1c3 → **e2e3** |
| kiwipete | 434,399 / 779 | 182,361 / 321 | 558K → 568K | e2a6 → e2a6 |
| middlegame | 71,734 / 134 | 147,997 / 229 | 535K → 646K | c1g5 → **c3a4** |
| endgame | 37,735 / 30 | 50,191 / 29 | 1.26M → 1.73M | e2e3 → **e2f3** |
| **total** | **555,787 / 982** | **438,381 / 713** | 566K → 615K | |

Total nodes **−21%**, total time **−27%** with the tuned set, but the effect
is position-dependent (startpos searched 4.8× more nodes — its TT hit rate
dropped 48.9% → 34.2% under the changed PST values). NPS varies with the
qsearch share of the tree and is never used as a strength claim.

---

## 5. Strength testing (self-play, Threads = 1, suite classical-v1)

### 5.1 Sanity A/A — baseline vs baseline (identical parameter files)

```text
games: 60  seed 11  depth 5  parallel 8
W20 D16 L24 (60 games)
score 0.467  elo -23.2  95% CI [-110.4, +64.0]
```

Both sides use `config/baseline_eval.toml`; the CI comfortably contains 0, so
the harness shows no significant side bias at this sample size.

### 5.2 A/B — tuned (candidate, both colours) vs baseline, SPRT

```text
games: 150  seed 7  depth 5  parallel 8  colors alternate: true
W69 D24 L57 (150 games)
score 0.540  elo +27.9  95% CI [-27.7, +83.4]
sprt: elo0 -2 elo1 3 alpha 0.05 beta 0.05  llr +0.218  games 150  decision MaxGames
avg game length 136.9 plies
terminations: adjudicated-mate: 126, move-limit: 6, threefold-repetition: 18
```

**Honest verdict: inconclusive.** The point estimate favours the tuned set
(+27.9 Elo) but the 95% confidence interval [−27.7, +83.4] contains zero, and
the SPRT stopped on the game cap with LLR **+0.218 far below the +2.945
acceptance bound** (α = β = 0.05). No strength gain is *proven*; detecting a
true ~28-Elo difference against the −2/+3 bounds would need on the order of
2000+ depth-5 games, not 150. Report as: smoke experiment, dataset small
(1041 positions, white-biased targets), signal positive but **not
statistically significant**.

Neither Elo figure is derived from NPS/depth/nodes — they come exclusively
from game outcomes.

Machine-readable artifacts: `results/baseline_aa.{json,pgn}`,
`results/tuned_vs_baseline.{json,pgn}`.

---

## 6. Exact reproduction commands

```bash
cargo build --release

# --- A. dataset from seeded self-play, dumped for reuse -------------------
./target/release/tune --smoke --seed 42 --iterations 300 \
    --subset material,mobility,pst --checkpoint-every 50 \
    --out tuned.toml --dump-data /tmp/ds.txt

# --- B. same dataset through the file pipeline: shuffle + split -----------
./target/release/tune --data /tmp/ds.txt --shuffle --test-split 0.2 \
    --seed 7 --iterations 200 --out /tmp/tuned_split.toml

# --- C. PGN pipeline ------------------------------------------------------
./target/release/tune --pgn tests/data/sample.pgn --iterations 30 \
    --out /tmp/tuned_pgn.toml

# --- D. original-vs-tuned review -----------------------------------------
./target/release/tune --compare config/baseline_eval.toml tuned.toml
./target/release/tune --compare config/baseline_eval.toml \
    config/baseline_eval.toml        # -> identical: 864 / 864

# --- E. benchmarks --------------------------------------------------------
cargo bench --bench eval -- --warm-up-time 1 --measurement-time 3
cargo bench --bench nps -- --warm-up-time 1 --measurement-time 3 --sample-size 10
cargo bench --bench perft -- --warm-up-time 1 --measurement-time 3
./target/release/morstilia --bench --stats
./target/release/morstilia --bench --stats --eval-params tuned.toml

# --- F. strength tests ----------------------------------------------------
mkdir -p results
./target/release/selfplay --games 60 --depth 5 --seed 11 --parallel 8 \
    --white config/baseline_eval.toml --black config/baseline_eval.toml \
    --candidate white --report results/baseline_aa.json \
    --out results/baseline_aa.pgn
./target/release/selfplay --games 150 --depth 5 --seed 7 --parallel 8 \
    --white tuned.toml --black config/baseline_eval.toml \
    --candidate white --sprt --report results/tuned_vs_baseline.json \
    --out results/tuned_vs_baseline.pgn

# --- G. tests -------------------------------------------------------------
cargo fmt --check
cargo test
cargo build --release
```

---

## 7. Regression protection

* `tests/stage5.rs` (9 tests): baseline-parameter parity with the shipped
  `config/baseline_eval.toml`, colour-mirror antisymmetry battery (11 FENs
  incl. passed-pawn endgames — this is the test that found §1.1), FEN file
  both-orders/garbage rejection, PGN replay counts + zero truncation,
  split determinism **and** fraction direction, shuffle reproducibility,
  dataset stats, `tune --compare` CLI (identical + differing), candidate
  parameter loading via `MORSTILIA_EVAL_PARAMS`.
* `tests/match_cli.rs` (7 tests): deterministic replay, resume, baseline
  freeze, compare verdict, **and** the bare-`--sprt` guard (§1.4).
* `src/tuning/dataset.rs` unit tests (14 with the `tuning::` filter): result
  parsing (floats + PGN notation), split direction, stats, stats display.
* Built-in tactics suite + `Threads = 1` determinism run after every tune
  (`regression ... deterministic ok` lines in §3); Stage-4 suite (293 tests)
  unchanged and green.

## 8. Known limitations (stated honestly)

* Smoke scale: 8 self-play games / 1041 positions and 300 SPSA iterations on
  a 783-param subset — real tuning needs tens of thousands of iterations over
  thousands of games (documented in the README).
* Dataset targets are white-biased (5 wins / 1 draw / 2 losses → avg result
  0.788); a larger corpus with balanced results is the obvious next step.
* Depth-5 matches have a low draw rate (16%) and heavy mate adjudication
  (126/150); results at real time controls may differ.
* The A/B experiment is statistically inconclusive (§5.2) — no strength gain
  is claimed.
* Criterion nps spreads for kiwipete/middlegame are wide (concurrent
  background load on the machine); medians reported.
* A concurrent session is developing NNUE (Stage 6) in `src/nnue/`; its
  in-flight tests may fail independently of this stage (verified: NNUE code
  references no classical-evaluation internals; `SearchShared.nnue` defaults
  to `None`, so every number above is pure classical evaluation).

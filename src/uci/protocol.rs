//! UCI protocol: [`UciEngine`] — the stateful driver that owns the board,
//! transposition table, opening books and (during `go`) the search thread.
//!
//! # Threading model
//!
//! `go` hands the [`Searcher`] to a worker thread along with the current
//! board and repetition history; the worker runs the search, prints the
//! `info`/`bestmove` lines through a single shared writer (`Arc<Mutex<...>>`)
//! and hands the searcher back through a second shared slot. Mutating
//! commands (`position`, `setoption`, `ucinewgame`, `go`, `quit`) flip the
//! stop flag and join the worker first, so they never race the search. The
//! UCI `stop` command only flips the flag — the worker notices on its next
//! periodic check.
//!
//! # Stdout contract
//!
//! The writer only ever emits `uciok`, `readyok`, `bestmove` and `info ...`
//! lines. Diagnostics (book-load errors, debug notes) go to stderr.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use shakmaty::zobrist::Zobrist64;

use crate::board::Position;
use crate::book::{PolyglotBook, SplitMix64};
use crate::config::{EngineConfig, EvalMode};
use crate::endgame::{LoadReport, Syzygy};
use crate::evaluation::Evaluator;
use crate::nnue::network::Network;
use crate::search::{SearchResult, Searcher};
use crate::types::{RawMove, is_mate, mate_plies};
use crate::uci::parser::{Command, GoParams};

/// Books are only consulted while the game is still in the opening
/// (roughly the first 24 plies). Deeper than that, positions are far past
/// any real book coverage and probing wastes time.
const BOOK_PLIES_LIMIT: usize = 24;

/// The stateful UCI engine driver.
pub struct UciEngine {
    config: EngineConfig,
    board: Position,
    /// Zobrist hashes of all positions *before* the current root since the
    /// last zeroing move (used for repetition detection during the search).
    /// `history[i]` is the position after the `i`-th non-zeroing move.
    history: Vec<Zobrist64>,
    /// Owned by the engine; moved into the worker thread during `go` and
    /// returned when the search finishes.
    searcher: Option<Searcher>,
    books: Vec<PolyglotBook>,
    /// The stop flag of the *current* search (replaced on every `go`).
    stop: Arc<AtomicBool>,
    /// The running search worker, if any.
    worker: Option<JoinHandle<()>>,
    /// Slot through which the worker returns the [`Searcher`].
    searcher_back: Option<Arc<Mutex<Option<Searcher>>>>,
    /// Slot through which the worker publishes the finished [`SearchResult`]
    /// (for CLI diagnostics; never written into stdout).
    last_result: Arc<Mutex<Option<SearchResult>>>,
    /// The single writer for all UCI output on stdout.
    out: Arc<Mutex<Box<dyn Write + Send>>>,
}

impl Default for UciEngine {
    fn default() -> Self {
        UciEngine::new()
    }
}

impl UciEngine {
    /// Creates an engine writing to real stdout.
    pub fn new() -> UciEngine {
        UciEngine::with_writer(Box::new(std::io::stdout()))
    }

    /// Creates an engine writing to an arbitrary sink (tests, pipes).
    pub fn with_writer(writer: Box<dyn Write + Send>) -> UciEngine {
        let mut config = EngineConfig::default();
        let books = load_books(&config);
        let params = load_params(&config);
        // Syzygy: load whatever the configured path holds (an empty path
        // keeps the tablebase inert). Warnings + the loaded-file summary go
        // to stderr — never into the UCI stdout stream.
        let (syzygy, tb_report) = load_syzygy(&config.syzygy_path);
        write_syzygy_report(&tb_report);
        // NNUE: the default config asks for the classical evaluator, so this is
        // a no-op for a stock engine. Decoding the 100 MB of LEB128 into
        // resident weights costs ~0.25 s (release) / ~2 s (debug), so it
        // happens once here and never per search.
        let nnue = load_nnue(&mut config);
        let mut searcher = Searcher::with_params(crate::config::DEFAULT_HASH_MB, params);
        searcher.set_syzygy(syzygy);
        searcher.set_nnue(nnue);
        UciEngine {
            config,
            board: Position::startpos(),
            history: Vec::new(),
            searcher: Some(searcher),
            books,
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            searcher_back: None,
            last_result: Arc::new(Mutex::new(None)),
            out: Arc::new(Mutex::new(writer)),
        }
    }

    /// Dispatches one parsed command. Returns `false` after `quit`.
    pub fn handle(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Uci => self.write_uci(),
            Command::IsReady => self.output("readyok\n"),
            Command::UciNewGame => self.uci_new_game(),
            Command::Position { fen, moves } => {
                self.set_position(fen.as_deref(), &moves);
            }
            Command::Go(params) => self.go(&params),
            Command::Stop => self.stop(),
            Command::Quit => {
                self.stop_and_join();
                return false;
            }
            Command::SetOption { name, value } => self.set_option(&name, value.as_deref()),
            Command::PonderHit => {} // accepted; the engine does not ponder.
            Command::Unknown(_) => {}
        }
        true
    }

    /// Parses a raw line and dispatches it (the `main.rs` loop).
    pub fn handle_line(&mut self, line: &str) -> bool {
        self.handle(crate::uci::parser::parse(line))
    }

    // --- Command handlers -------------------------------------------------

    fn write_uci(&self) {
        let mut out = self.out.lock().unwrap();
        let _ = writeln!(out, "id name {}", env!("MORSTILIA_NAME"));
        let _ = writeln!(out, "id author {}", env!("MORSTILIA_ID_AUTHOR"));
        for line in self.config.option_log() {
            let _ = writeln!(out, "{line}");
        }
        let _ = writeln!(out, "uciok");
        let _ = out.flush();
    }

    fn uci_new_game(&mut self) {
        self.stop_and_join();
        if let Some(s) = &mut self.searcher {
            s.clear_tt();
        }
        self.board = Position::startpos();
        self.history.clear();
    }

    /// Applies `position [fen ... | startpos] moves ...` atomically: on any
    /// invalid FEN or move the current board is left untouched.
    fn set_position(&mut self, fen: Option<&str>, moves: &[String]) {
        self.stop_and_join();
        let base = match fen {
            Some(f) => match Position::from_fen(f) {
                Ok(p) => p,
                Err(_) => {
                    eprintln!("info string ignoring invalid FEN: {f}");
                    return;
                }
            },
            None => Position::startpos(),
        };

        let mut board = base;
        let mut history: Vec<Zobrist64> = Vec::new();
        for m in moves {
            match board.play_uci(m) {
                Ok((child, _)) => {
                    // A zeroing move (capture/pawn push) resets the 50-move
                    // clock and with it the repetition window.
                    if child.halfmoves() == 0 {
                        history.clear();
                    } else {
                        history.push(board.hash);
                    }
                    board = child;
                }
                Err(_) => {
                    eprintln!("info string ignoring invalid move: {m}");
                    return;
                }
            }
        }
        self.board = board;
        self.history = history;
    }

    fn set_option(&mut self, name: &str, value: Option<&str>) {
        self.stop_and_join();
        let changed = self.config.set_option(name, value);
        if !changed {
            eprintln!("info string ignoring unknown option: {name}");
            return;
        }
        match name {
            "Hash" => {
                if let Some(s) = &mut self.searcher {
                    s.resize(self.config.hash_mb);
                }
            }
            "Threads" => {
                // Pre-size the persistent worker pool (growing/shrinking it
                // joins retired workers so no thread leaks across searches).
                if let Some(s) = &mut self.searcher {
                    s.set_threads(self.config.threads);
                }
            }
            "EvalParamsPath" => {
                // Swap the parameter set in place: the TT (and its game state)
                // stays untouched, only the evaluation lens changes.
                let params = load_params(&self.config);
                if let Some(s) = &mut self.searcher {
                    s.params = Arc::new(params);
                }
            }
            "Eval" => {
                // Change evaluation mode. If switching to NNUE, try to load the net.
                // On load failure, fall back to classical and report error.
                let nnue = if self.config.eval == EvalMode::Nnue {
                    load_nnue(&mut self.config)
                } else {
                    None
                };
                // If we requested NNUE but couldn't load a net, fall back to classical.
                if self.config.eval == EvalMode::Nnue && nnue.is_none() {
                    eprintln!("info string nnue: falling back to the classical evaluator");
                    self.config.eval = EvalMode::Classical;
                }
                if let Some(s) = &mut self.searcher {
                    s.set_nnue(nnue);
                }
            }
            "NNUEFile" => {
                // Change the NNUE file path. If currently in NNUE mode, try to load
                // the new net. On failure, keep the current evaluator (don't change
                // config.eval) and report error — the user may send a corrected path.
                let nnue = if self.config.eval == EvalMode::Nnue {
                    load_nnue(&mut self.config)
                } else {
                    None
                };
                if let Some(s) = &mut self.searcher {
                    s.set_nnue(nnue);
                }
            }
            "BookEnabled" | "BookPath" => {
                self.books = load_books(&self.config);
            }
            "SyzygyPath" => {
                // Reload the tablebase from the (possibly changed) path. An
                // empty or invalid path degrades to an inert tablebase — the
                // engine simply stops probing.
                let (tb, report) = load_syzygy(&self.config.syzygy_path);
                write_syzygy_report(&report);
                if let Some(s) = &mut self.searcher {
                    s.set_syzygy(tb);
                }
            }
            _ => {}
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Starts (or re-starts) a search on a worker thread. If the opening
    /// book applies, the move is printed directly and no thread is spawned.
    fn go(&mut self, params: &GoParams) {
        self.stop_and_join();

        // Apply the configured absolute depth ceiling.
        let mut limits = time_limit_from_go(params, &self.board);
        if self.config.search_depth > 0 {
            let d = limits.depth.unwrap_or(self.config.search_depth);
            limits.depth = Some(d.min(self.config.search_depth));
        }

        // Opening book (not consulted for infinite/ponder/searchmoves).
        if let Some(m) = self.book_move(params) {
            if self.config.debug {
                eprintln!("info string book hit: selected {}", m.to_uci());
            }
            self.output(&format!("bestmove {}\n", m.to_uci()));
            return;
        }

        // Move the searcher into the worker thread and take it back on join.
        let root = self.board.clone();
        let history = self.history.clone();
        let threads = self.config.threads.max(1);
        let searchmoves: Vec<RawMove> = params
            .searchmoves
            .iter()
            .filter_map(|s| self.board.raw_move_from_uci(s))
            .collect();

        let stop = Arc::new(AtomicBool::new(false));
        self.stop = stop.clone();
        let mut searcher = self.searcher.take().unwrap_or_else(|| {
            Searcher::with_params(self.config.hash_mb, load_params(&self.config))
        });
        let back: Arc<Mutex<Option<Searcher>>> = Arc::new(Mutex::new(None));
        let back2 = back.clone();
        let result_slot = self.last_result.clone();
        let debug = self.config.debug;
        let out = self.out.clone();

        let handle = std::thread::spawn(move || {
            let result = searcher.search(&root, &history, &limits, &stop, threads, &searchmoves);
            // Snapshot the params before the searcher goes back home (the
            // debug eval breakdown below still needs them).
            let params = searcher.params.clone();
            *result_slot.lock().unwrap() = Some(result.clone());
            *back2.lock().unwrap() = Some(searcher);
            let mut sink = out.lock().unwrap();
            let _ = write_info(&mut *sink, &result);
            let best = if result.is_none() {
                "0000".to_string()
            } else {
                result.best.to_uci()
            };
            let _ = writeln!(sink, "bestmove {best}");
            if debug {
                let _ = write_debug_stats(&mut *sink, &result);
                let _ = write_eval_breakdown(&mut *sink, &root, &params);
            }
            let _ = sink.flush();
        });

        self.worker = Some(handle);
        self.searcher_back = Some(back);
    }

    // --- Helpers ------------------------------------------------------------

    /// A weighted-random legal book move for the current position, if any
    /// book is loaded and the position is still in the opening.
    fn book_move(&mut self, params: &GoParams) -> Option<RawMove> {
        if !self.config.book_enabled || self.books.is_empty() {
            return None;
        }
        if params.infinite || params.ponder || !params.searchmoves.is_empty() {
            return None;
        }
        if self.history.len() >= BOOK_PLIES_LIMIT {
            return None;
        }
        // Seed from the position so book play is stable for a given line.
        let seed: u64 = self.board.hash.into();
        let mut rng = SplitMix64(seed);
        for book in &self.books {
            if let Some(m) = book.weighted_move(&self.board, &mut || rng.next()) {
                return Some(m);
            }
        }
        None
    }

    /// Flips the stop flag, joins the worker (if any) and reclaims the
    /// [`Searcher`]. Safe to call when no search is running.
    pub fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.take_worker();
    }

    /// Joins the worker (if any) and reclaims the [`Searcher`] **without**
    /// flipping the stop flag. Use this after a *bounded* `go` (e.g. the
    /// CLI's `go depth N`) where the search is expected to finish on its own;
    /// `stop_and_join` would abort it early.
    pub fn join(&mut self) {
        self.take_worker();
    }

    /// Internal: waits for the current worker and takes the searcher back.
    fn take_worker(&mut self) {
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
        if let Some(back) = self.searcher_back.take() {
            if let Ok(mut guard) = back.lock() {
                if let Some(s) = guard.take() {
                    self.searcher = Some(s);
                }
            }
        }
    }

    /// Writes a raw line to the single output sink (locking it first).
    pub fn output(&self, s: &str) {
        let mut out = self.out.lock().unwrap();
        let _ = out.write_all(s.as_bytes());
        let _ = out.flush();
    }

    /// The most recent finished search's full [`SearchResult`], including its
    /// instrumentation counters. `None` until the first `go` completes. This
    /// is diagnostic plumbing — it is never written to stdout on its own.
    pub fn last_result(&self) -> Option<SearchResult> {
        self.last_result.lock().unwrap().clone()
    }
}

/// Builds the `info depth ... score ... nodes ... nps ... time ... pv ...`
/// line for a finished search. Returns without writing when the search never
/// produced a result (aborted before the first iteration). Multi-threaded
/// searches append one `info string smp ...` line — worker count, root-task
/// total, shared-TT hit/cutoff rate plus a per-worker nodes/time/depth (and a
/// best-effort idle estimate: wall time minus the worker's own busy period).
/// `Threads = 1` output is unchanged, keeping single-thread bench transcripts
/// byte-identical.
fn write_info(out: &mut dyn Write, r: &SearchResult) -> std::io::Result<()> {
    if r.is_none() {
        return Ok(());
    }
    let score = if is_mate(r.score) {
        format!("mate {}", mate_plies(r.score))
    } else {
        format!("cp {}", r.score)
    };
    let pv: Vec<String> = r.pv.iter().map(|m| m.to_uci()).collect();
    writeln!(
        out,
        "info depth {} score {} nodes {} nps {} time {} pv {}",
        r.depth,
        score,
        r.nodes,
        r.nps(),
        r.time_ms,
        pv.join(" ")
    )?;
    if r.threads > 1 {
        writeln!(
            out,
            "info string smp threads {} workers {} rootmoves {} nps {} tt_hit {:.1}% tt_cut {:.1}%",
            r.threads,
            r.workers.len(),
            r.root_moves,
            r.nps(),
            r.stats.tt_hit_pct(),
            r.stats.tt_cutoff_pct(),
        )?;
        let wall = r.time_ms.max(1);
        for (i, w) in r.workers.iter().enumerate() {
            writeln!(
                out,
                "info string smp worker {}{} nodes {} time {}ms idle {}ms depth {}",
                if w.main { "*" } else { "" },
                i,
                w.nodes,
                w.time_ms,
                wall.saturating_sub(w.time_ms),
                w.depth,
            )?;
        }
    }
    Ok(())
}

/// Writes the instrumentation counters as an `info string` line (only emitted
/// in `Debug` mode, so the regular stdout contract is untouched).
fn write_debug_stats(out: &mut dyn Write, r: &SearchResult) -> std::io::Result<()> {
    let s = &r.stats;
    writeln!(
        out,
        "info string stats qnodes {} tt_probe {} tt_hit {} ({:.1}%) tt_cut {} ({:.1}%) \
         beta_cut {} first_cut {} ({:.1}%) avg_moves {:.2} see {} prune {} ({:.1}%) tt_stores {}",
        s.qsearch_nodes,
        s.tt_probes,
        s.tt_hits,
        s.tt_hit_pct(),
        s.tt_cutoffs,
        s.tt_cutoff_pct(),
        s.beta_cutoffs,
        s.first_move_cutoffs,
        s.first_move_cutoff_pct(),
        s.avg_moves_until_cutoff(),
        s.see_calls,
        s.see_pruned,
        s.see_prune_pct(),
        r.tt_stores,
    )?;
    writeln!(
        out,
        "info string stats2 null {} ({}) lmr {} ({}) fut {} rfp {} \
         razor {} ({}) probcut {} ({}) pruned {} ebf {:.2}",
        s.null_probes,
        s.null_cutoffs,
        s.lmr_researched,
        s.lmr_reduced,
        s.futility_pruned,
        s.rfp_pruned,
        s.razor_cutoffs,
        s.razor_attempts,
        s.probcut_cutoffs,
        s.probcut_attempts,
        s.total_pruned(),
        r.ebf(),
    )?;
    if s.tb_probes > 0 {
        writeln!(
            out,
            "info string stats3 tb_probe {} tb_hit {} ({:.1}%) win {} draw {} loss {} cursed {}",
            s.tb_probes,
            s.tb_hits,
            if s.tb_probes > 0 {
                100.0 * s.tb_hits as f64 / s.tb_probes as f64
            } else {
                0.0
            },
            s.tb_wins,
            s.tb_draws,
            s.tb_losses,
            s.tb_cursed,
        )
    } else {
        Ok(())
    }
}

/// Writes the evaluation component breakdown for `pos` (only in Debug mode),
/// using the search's own parameter set so tuned evals are diagnosed
/// consistently.
fn write_eval_breakdown(
    out: &mut dyn Write,
    pos: &Position,
    params: &crate::evaluation::EvalParams,
) -> std::io::Result<()> {
    let parts = Evaluator.evaluate_parts_with(pos, params);
    writeln!(out, "info string estat {}", parts.stats_row())
}

// --- Opening-book discovery -------------------------------------------------

/// Candidate book paths tried in order when `BookPath` is empty: next to the
/// executable first (`book/book.bin`, then `book/AllOpeningsMorstilia.bin`),
/// falling back to the working directory.
fn auto_book_paths() -> Vec<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if !roots.iter().any(|r| r == &cwd) {
            roots.push(cwd);
        }
    }
    for root in roots {
        let book_dir = root.join("book");
        cands.push(book_dir.join("book.bin"));
        cands.push(book_dir.join("AllOpeningsMorstilia.bin"));
    }
    cands
}

/// Loads a book from a file or, when the path is a directory, from every
/// `*.bin` inside it. Errors are reported on stderr and never fatal.
fn load_books(config: &EngineConfig) -> Vec<PolyglotBook> {
    let mut books = Vec::new();
    if !config.book_enabled {
        return books;
    }
    let paths: Vec<PathBuf> = if config.book_path.is_empty() {
        auto_book_paths()
    } else {
        vec![PathBuf::from(&config.book_path)]
    };
    for p in paths {
        if p.is_dir() {
            match std::fs::read_dir(&p) {
                Ok(rd) => {
                    let mut files: Vec<PathBuf> = rd
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|f| f.extension().is_some_and(|x| x == "bin"))
                        .collect();
                    files.sort();
                    for f in files {
                        match PolyglotBook::load(&f) {
                            Ok(b) if !b.is_empty() => books.push(b),
                            Ok(_) => {}
                            Err(e) => eprintln!("info string book error: {e}"),
                        }
                    }
                }
                Err(e) => eprintln!("info string book error: cannot read {}: {e}", p.display()),
            }
        } else if p.exists() {
            match PolyglotBook::load(&p) {
                Ok(b) if !b.is_empty() => books.push(b),
                Ok(_) => {}
                Err(e) => eprintln!("info string book error: {e}"),
            }
        }
    }
    if !books.is_empty() {
        eprintln!(
            "info string loaded {} opening book(s) ({} entries)",
            books.len(),
            books.iter().map(PolyglotBook::len).sum::<usize>()
        );
    }
    books
}

/// Loads the Syzygy tablebase for an engine configuration. See
/// [`crate::endgame::Syzygy::load`]; the UCI stdout sink never sees
/// tablebase traffic.
fn load_syzygy(path: &str) -> (Syzygy, LoadReport) {
    Syzygy::load(path)
}

/// Prints the tablebase load summary and any warnings to stderr (never into
/// the UCI stdout stream). Mirrors the book-load reporting conventions.
fn write_syzygy_report(report: &LoadReport) {
    for w in &report.warnings {
        eprintln!("info string syzygy: {w}");
    }
    if report.files > 0 {
        eprintln!(
            "info string syzygy: loaded {} table file(s) (max {} pieces) from {}",
            report.files, report.max_pieces, report.path
        );
    }
}

/// Loads the evaluation parameters for an engine configuration. An empty
/// `EvalParamsPath` yields the baseline defaults; a configured but unreadable
/// or malformed file falls back to the defaults with a stderr note (never
/// fatal — the engine must still start).
fn load_params(config: &EngineConfig) -> crate::evaluation::EvalParams {
    use crate::evaluation::EvalParams;
    if config.eval_params_path.is_empty() {
        return EvalParams::default();
    }
    match EvalParams::load(&config.eval_params_path) {
        Ok(p) => {
            eprintln!(
                "info string loaded eval params from {}",
                config.eval_params_path
            );
            p
        }
        Err(e) => {
            eprintln!(
                "info string eval params {}: {e:#} — using baseline defaults",
                config.eval_params_path
            );
            EvalParams::default()
        }
    }
}

/// Loads the NNUE net for an engine configuration, if one is selected.
///
/// Returns `None` with a stderr note if the net cannot be used. The caller
/// is responsible for handling the fallback (e.g., reverting `config.eval`
/// when `Eval nnue` was explicitly requested but the net failed to load).
/// This function does NOT modify `config.eval`.
///
/// The stdout contract is absolute: this only ever writes to stderr.
fn load_nnue(config: &mut EngineConfig) -> Option<Arc<Network>> {
    use crate::config::EvalMode;
    use crate::nnue;

    if config.eval != EvalMode::Nnue {
        return None;
    }

    // If an explicit NNUEFile is set, try to load from that file.
    // Otherwise, use the embedded default network.
    let net = if !config.nnue_path.is_empty() {
        let path = nnue::resolve_net_path(&config.nnue_path);
        match nnue::load_network(&path) {
            Ok(net) => {
                eprintln!(
                    "info string nnue: loaded {} (network hash {:#010x})",
                    path.display(),
                    nnue::NETWORK_HASH
                );
                Some(net)
            }
            Err(e) => {
                eprintln!("info string nnue: cannot use {}: {e}", path.display());
                None
            }
        }
    } else {
        // No explicit file: use the embedded Stockfish 19 network
        match nnue::load_embedded_network() {
            Some(net) => {
                eprintln!(
                    "info string nnue: loaded embedded default (network hash {:#010x})",
                    nnue::NETWORK_HASH
                );
                Some(net)
            }
            None => {
                eprintln!("info string nnue: embedded default network failed validation");
                None
            }
        }
    };
    net.map(Arc::new)
}

/// Re-exports the converter from the time module for the `go` command.
pub(crate) use crate::search::time::time_limit_from_go;

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory writer that keeps a clonable handle to its buffer, so tests
    /// can read the accumulated output without downcasting the `Box<dyn Write>`.
    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl SharedBuffer {
        fn read(&self) -> String {
            let bytes = self.0.lock().unwrap().clone();
            String::from_utf8(bytes).expect("output is utf-8")
        }
    }

    /// Drives an engine with an in-memory writer and returns its output.
    fn drive(lines: &[&str]) -> (String, UciEngine) {
        let buf = SharedBuffer::default();
        let out = buf.clone();
        let mut engine = UciEngine::with_writer(Box::new(buf));
        for line in lines {
            if !engine.handle_line(line) {
                break;
            }
        }
        // Retire any worker so its output is visible. Bounded `go`s finish on
        // their own; `go infinite` runs in `drive` are always coupled with an
        // explicit `stop`, so waiting without flipping the flag is enough.
        engine.join();
        let s = out.read();
        (s, engine)
    }

    #[test]
    fn uci_handshake_is_clean() {
        let (s, _) = drive(&["uci", "isready"]);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], format!("id name {}", env!("MORSTILIA_NAME")));
        assert!(lines.iter().any(|l| *l == "uciok"));
        assert!(lines.iter().any(|l| l.starts_with("option name Hash")));
        // stdout contract: only the four allowed line kinds.
        for l in s.lines() {
            assert!(
                l.starts_with("id ")
                    || l.starts_with("option ")
                    || l == "uciok"
                    || l == "readyok"
                    || l.starts_with("info ")
                    || l.starts_with("bestmove "),
                "leaked line: {l:?}"
            );
        }
    }

    /// The `bestmove <m>` value from a transcript.
    fn bestmove(out: &str) -> String {
        out.lines()
            .find_map(|l| l.strip_prefix("bestmove "))
            .map(str::to_string)
            .unwrap_or_else(|| panic!("no bestmove line in: {out}"))
    }

    #[test]
    fn book_serves_opening_moves_then_search_takes_over() {
        // A scratch book containing *only* a startpos entry: the first `go`
        // answers from the book (no `info` lines), the second — after the
        // position left the book (e2e4 played) — must run a real search.
        use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "morstilia_book_switch_{}_{}",
            std::process::id(),
            N.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mini.bin");
        let mut bytes = Vec::new();
        // startpos key; e2(12)->e4(28); raw = from<<6|to = 796.
        let key = 0x463b96181691fc9cu64;
        bytes.extend_from_slice(&key.to_be_bytes());
        bytes.extend_from_slice(&796u16.to_be_bytes());
        bytes.extend_from_slice(&10u16.to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let (s, _) = drive(&[
            "setoption name BookEnabled value true",
            &format!("setoption name BookPath value {}", path.display()),
            "position startpos",
            "go depth 1",
        ]);
        assert_eq!(
            bestmove(&s),
            "e2e4",
            "the book must answer at startpos: {s}"
        );
        assert!(
            !s.contains("info depth"),
            "a book move must come without search info: {s}"
        );

        let (s, _) = drive(&[
            "setoption name BookEnabled value true",
            &format!("setoption name BookPath value {}", path.display()),
            "position startpos moves e2e4",
            "go depth 2",
        ]);
        assert!(
            s.lines().any(|l| l.starts_with("info depth ")),
            "after the book ends the search must run: {s}"
        );
        assert!(bestmove(&s).len() >= 4, "bestmove present: {s}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn go_depth_produces_bestmove_line() {
        // Disable the auto-discovered opening book explicitly so the bounded
        // `go` exercises the search itself (the repo ships a real default book
        // at `book/book.bin`, which would otherwise answer `e2e4` directly).
        let (s, _) = drive(&[
            "setoption name BookEnabled value false",
            "position startpos",
            "go depth 3",
        ]);
        assert!(
            s.lines().any(|l| l.starts_with("bestmove ")),
            "missing bestmove in: {s}"
        );
        assert!(
            s.lines().any(|l| l.starts_with("info depth ")),
            "missing info line in: {s}"
        );
        let best = s
            .lines()
            .find_map(|l| l.strip_prefix("bestmove "))
            .expect("bestmove present");
        assert!(
            best.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()),
            "bad bestmove {best}"
        );
    }

    #[test]
    fn invalid_position_is_ignored_atomically() {
        let (s, _) = drive(&["position startpos moves e2e5", "go depth 2"]);
        // e2e5 is illegal: the board must still be the untouched startpos, so
        // the search succeeds and emits a normal bestmove.
        assert!(s.lines().any(|l| l.starts_with("bestmove ")), "got: {s}");
    }

    #[test]
    fn ucinewgame_clears_tt_and_restarts() {
        let (s, _) = drive(&[
            "ucinewgame",
            "position startpos",
            "go depth 2",
            "ucinewgame",
            "isready",
        ]);
        assert!(s.lines().any(|l| l == "readyok"));
        assert!(s.lines().any(|l| l.starts_with("bestmove ")));
    }

    #[test]
    fn go_infinite_then_stop_reports_a_move() {
        let (s, _) = drive(&["position startpos", "go infinite", "stop"]);
        assert!(
            s.lines().any(|l| l.starts_with("bestmove ")),
            "stop must yield a bestmove: {s}"
        );
    }

    #[test]
    fn setoption_hash_resizes() {
        let (_, engine) = drive(&["setoption name Hash value 256"]);
        assert_eq!(engine.config.hash_mb, 256);
        let (_, engine) = drive(&["setoption name Threads value 4"]);
        assert_eq!(engine.config.threads, 4);
        let (s, _) = drive(&["setoption name Hash value notanumber", "isready"]);
        assert!(s.lines().any(|l| l == "readyok"));
    }

    #[test]
    fn no_book_found_is_not_fatal() {
        // A missing/disabled book is never fatal: with the book turned off
        // (and, by extension, any unreadable BookPath), the engine must still
        // search normally. (The repo ships a default book at `book/book.bin`,
        // which is why the book is disabled explicitly here.)
        let (s, _) = drive(&[
            "setoption name BookEnabled value false",
            "position startpos",
            "go depth 2",
        ]);
        assert!(s.lines().any(|l| l.starts_with("bestmove ")));
    }

    #[test]
    fn quit_stops_search_and_returns_false() {
        let mut engine = UciEngine::with_writer(Box::new(Vec::<u8>::new()));
        assert!(engine.handle(Command::Go(GoParams {
            infinite: true,
            ..GoParams::default()
        })));
        assert!(!engine.handle(Command::Quit));
        // The worker has been joined: the searcher is back home.
        assert!(engine.searcher.is_some());
        assert!(engine.worker.is_none());
    }
}

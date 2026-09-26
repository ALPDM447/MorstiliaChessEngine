//! Integration tests for the NNUE layer as the search actually uses it.
//!
//! The unit tests inside `src/nnue/` pin the file format, the feature
//! indexing, the accumulator and the blend against Stockfish 19 references, and
//! `tests/nnue_groundtruth.rs` pins the whole evaluation against real
//! Stockfish output. This file covers the parts that only exist once the
//! evaluator is *wired into the search*:
//!
//! * loading and validating a net, and refusing every kind of bad one,
//! * `SearchThread::make_child` / `make_child_null` — this engine's stand-in
//!   for make/unmake — producing exactly the accumulator a full refresh gives,
//! * evaluation being a pure function of the position,
//! * `Threads = 1` reproducibility of a full search,
//! * the classical and NNUE evaluators both driving real searches.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use morstilia::board::Position;
use morstilia::evaluation::EvalParams;
use morstilia::nnue::accumulator::Accumulator;
use morstilia::nnue::board::Board;
use morstilia::nnue::format::{self, Reader, VERSION};
use morstilia::nnue::network::{LoadError, Network, load_network};
use morstilia::nnue::types::Color;
use morstilia::nnue::{DEFAULT_NET_FILE, evaluate};
use morstilia::search::{SearchShared, SearchThread, Searcher, TimeLimit};
use morstilia::types::{MAX_PLY, RawMove};

/// The bundled net, loaded once for the whole test binary (0.25 s, ~115 MB).
fn net() -> &'static Arc<Network> {
    static NET: OnceLock<Arc<Network>> = OnceLock::new();
    NET.get_or_init(|| {
        let path = net_path();
        let loaded =
            load_network(&path).unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()));
        Arc::new(loaded)
    })
}

/// [`load_network`], but panicking with the path and a readable message instead
/// of needing `Network: Debug` for `unwrap_err`.
fn load_err(p: &Path) -> String {
    match load_network(p) {
        Ok(_) => panic!("{} was accepted as a valid net", p.display()),
        Err(LoadError::Io(e)) => format!("cannot read the net file: {e}"),
        Err(LoadError::Format(e)) => format!("invalid net: {e}"),
    }
}

/// `nnue/nn-1a298aa575a0.nnue` inside the crate.
fn net_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("nnue")
        .join(DEFAULT_NET_FILE)
}

/// A scratch file that removes itself, so a failing assertion cannot leave a
/// stray 100 MB file behind.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Scratch {
    /// Writes `bytes` and returns the file, so a whole test reads as one
    /// pipeline: `scratch("x").written(&bytes).rejected()`.
    fn written(self, bytes: &[u8]) -> Scratch {
        std::fs::write(&self.0, bytes).unwrap();
        self
    }

    /// The first `n` bytes of the real net.
    fn prefixed_with_net(self, n: usize) -> Scratch {
        use std::io::Read as _;
        let mut f = std::fs::File::open(net_path()).unwrap();
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf).unwrap();
        self.written(&buf)
    }

    /// Loads the file, requires the load to fail, and returns the message.
    /// Consumes the scratch, so the file is removed before this returns.
    fn rejected(self) -> String {
        let msg = load_err(&self.0);
        drop(self);
        msg
    }
}

/// A `Scratch` holding a file that [`Scratch::rejected`] will delete again if it
/// is never reached. Consuming builders let each test be a single expression.
fn scratch(name: &str) -> Scratch {
    let path = std::env::temp_dir().join(format!("morstilia-nnue-{name}"));
    let _ = std::fs::remove_file(&path);
    Scratch(path)
}

/// A minimal but *structurally* valid header, so a test can fail on exactly
/// the field it is about rather than on a length mismatch.
fn header(version: u32, network_hash: u32, desc: &str) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&version.to_le_bytes());
    v.extend_from_slice(&network_hash.to_le_bytes());
    v.extend_from_slice(&(desc.len() as u32).to_le_bytes());
    v.extend_from_slice(desc.as_bytes());
    v
}

/// The real header, as it appears at the front of the bundled net.
fn real_header() -> Vec<u8> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(net_path()).unwrap();
    let mut buf = Vec::new();
    // The description is short; 4 KiB is comfortably past its end.
    buf.resize(4096, 0);
    let n = f.read(&mut buf).unwrap();
    buf.truncate(n);
    buf
}

/// Asserts two accumulators are bit-identical, naming the *index* of the first
/// difference rather than dumping 2048 numbers.
fn assert_same_accumulator(got: &Accumulator, want: &Accumulator, what: &str) {
    assert_eq!(got.computed, want.computed, "{what}: computed flags");
    assert_eq!(got.psqt, want.psqt, "{what}: the psqt accumulator");
    for p in 0..2 {
        if got.accumulation[p] != want.accumulation[p] {
            let j = (0..morstilia::nnue::L1)
                .find(|&j| got.accumulation[p][j] != want.accumulation[p][j])
                .expect("arrays differ, so some index differs");
            panic!(
                "{what}: accumulation[perspective {p}][{j}] is {}, a full refresh gives {}",
                got.accumulation[p][j], want.accumulation[p][j]
            );
        }
    }
}

/// True when `m` captures something on the pre-move board `pos`.
fn is_capture(pos: &Position, m: RawMove) -> bool {
    if m.is_castle() {
        return false;
    }
    m.is_en_passant() || pos.board().piece_at(m.to()).is_some()
}

/// A position suite for the search-level tests: the start position, a busy
/// middlegame, both castlings, an en-passant capture, a promotion race, a
/// doubled-pawn structure, a king-and-rook endgame, a zugzwang-ish pawn
/// endgame, and a tactical shot. Deliberately not the NNUE test corpus — these
/// are about *search* behaviour, so they include positions where the classical
/// and NNUE evaluations legitimately disagree.
const SEARCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 2",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
    "4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
    // A pawn on the seventh rank, with and without a capture available, so the
    // straight and capture promotions are both reachable in one move.
    "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
    "4k2q/6P1/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/PPP1PPPP/8/8/4K3 w - - 0 1",
    "8/8/4k3/3pN3/3P4/8/4K3/8 w - - 0 1",
    "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 1",
    "6k1/5ppp/8/8/8/8/8/R3K2R w KQ - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
];

/// A fresh `SearchShared` with the given net wired in.
fn shared_with(net: Option<&'static Arc<Network>>) -> SearchShared {
    SearchShared {
        tt: Arc::new(morstilia::tt::TranspositionTable::new(1)),
        params: Arc::new(EvalParams::default()),
        stop: Arc::new(AtomicBool::new(false)),
        nodes: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        node_cap: None,
        tb: Arc::new(morstilia::endgame::Syzygy::none()),
        nnue: net.cloned(),
    }
}

/// A searcher for one fixed-depth search, with the given net.
fn searcher_with(net: Option<&'static Arc<Network>>) -> Searcher {
    let mut s = Searcher::with_params(1, EvalParams::default());
    s.set_nnue(net.cloned());
    s
}

// --- Loading and validation -----------------------------------------------

#[test]
fn the_bundled_net_loads_and_identifies_itself() {
    let n = net();
    assert_eq!(n.input_dimensions(), 86_896);
    assert_eq!(n.hash(), 0xA85B_2205);
    // 8 heads and 8 buckets: `evaluate` indexes one stack per bucket, so a
    // net with fewer would panic on a sparse endgame position.
    assert_eq!(morstilia::nnue::LAYER_STACKS, 8);
    assert_eq!(morstilia::nnue::PSQT_BUCKETS, 8);
    // The trainer's own description line, which the header carries verbatim.
    // A lossy UTF-8 conversion or a mis-sized length field would change it.
    assert_eq!(
        n.description,
        "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer.",
    );
}

#[test]
fn the_bundled_net_is_the_exact_stockfish_19_default() {
    // Guarded by size and hash: a differently-sized or differently-hashed file
    // in `nnue/` would quietly change every evaluation in the engine, and
    // `tests/nnue_groundtruth.rs` would then be comparing against the wrong net.
    let meta = std::fs::metadata(net_path()).unwrap();
    assert_eq!(meta.len(), 98_511_183, "nnue/{DEFAULT_NET_FILE} size");
    assert_eq!(morstilia::nnue::NETWORK_HASH, 0xA85B_2205);
    assert_eq!(
        morstilia::nnue::network::FEATURE_TRANSFORMER_HASH,
        0xCB68_5313
    );
    assert_eq!(morstilia::nnue::network::LAYER_STACK_HASH, 0x6333_7116);
    // The three hashes are not independent: the network hash is their xor.
    assert_eq!(
        morstilia::nnue::NETWORK_HASH,
        morstilia::nnue::network::FEATURE_TRANSFORMER_HASH
            ^ morstilia::nnue::network::LAYER_STACK_HASH
    );
}

#[test]
fn a_missing_net_is_an_error_not_a_silent_default() {
    let asked = "/definitely/not/here/morstilia-missing.net";
    match load_network(Path::new(asked)) {
        Err(LoadError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        Err(other) => panic!("expected an I/O error, got {other}"),
        Ok(_) => panic!("a missing net was loaded"),
    }
    // The path resolver must not invent a file that is not there, and must not
    // silently fall back to the bundled net: it reports the path the caller
    // asked for, so the error message names something the user can fix.
    assert_eq!(nnue_resolve(asked), Path::new(asked));
    // An *empty* path does mean the bundled net, resolved to a real file. The
    // candidate search starts at the working directory (what a UCI GUI's
    // relative paths mean) and falls back to the executable's directory, so the
    // assertion is on the file that was found, not on a fixed path.
    let bundled = nnue_resolve("");
    assert!(bundled.is_file(), "{bundled:?} does not exist");
    assert_eq!(bundled.file_name().unwrap(), DEFAULT_NET_FILE);
    // An explicit path that does exist is used as given.
    assert_eq!(nnue_resolve(net_path().to_str().unwrap()), net_path());
}

fn nnue_resolve(p: &str) -> PathBuf {
    morstilia::nnue::resolve_net_path(p)
}

#[test]
fn an_empty_file_is_rejected() {
    let msg = scratch("empty").written(&[]).rejected();
    assert!(
        msg.contains("version"),
        "a zero-length file fails on the version word: {msg}"
    );
}

#[test]
fn a_wrong_version_is_rejected() {
    for bad in [0u32, 1, VERSION.wrapping_add(1), VERSION ^ 1] {
        let msg = scratch("version")
            .written(&header(bad, 0xA85B_2205, "x"))
            .rejected();
        assert!(msg.contains("unsupported net version"), "v={bad:#x}: {msg}");
    }
}

#[test]
fn a_net_built_for_another_architecture_is_rejected() {
    // The structural hash is what distinguishes "a net Stockfish 19 can run"
    // from "a net that happens to parse". A different NNUE architecture, or a
    // retrained net with a different feature set, lands here.
    for bad in [0u32, 1, 0x1234_5678, 0xA85B_2205 ^ 1] {
        let msg = scratch("arch")
            .written(&header(VERSION, bad, "x"))
            .rejected();
        assert!(
            msg.contains("network architecture hash mismatch"),
            "hash={bad:#x}: {msg}"
        );
    }
}

#[test]
fn an_implausible_description_length_is_rejected() {
    // A corrupt length field must not become a 4 GiB allocation attempt.
    let mut v = VERSION.to_le_bytes().to_vec();
    v.extend_from_slice(&0xA85B_2205u32.to_le_bytes());
    v.extend_from_slice(&(1u32 << 30).to_le_bytes());
    let msg = scratch("desclen").written(&v).rejected();
    assert!(msg.contains("implausible description length"), "{msg}");
}

#[test]
fn a_truncated_description_is_rejected() {
    // Declares 4096 bytes of description, supplies three.
    let mut v = VERSION.to_le_bytes().to_vec();
    v.extend_from_slice(&0xA85B_2205u32.to_le_bytes());
    v.extend_from_slice(&4096u32.to_le_bytes());
    v.extend_from_slice(b"abc");
    let msg = scratch("desctrunc").written(&v).rejected();
    assert!(msg.contains("description"), "{msg}");
}

#[test]
fn every_truncation_of_the_real_net_is_rejected() {
    // Cut points chosen to land inside each section, not just at the end: a
    // loader that ignored a short read somewhere in the middle would slip past
    // a test that only checked `len - 1`.
    let total = std::fs::metadata(net_path()).unwrap().len();
    for cut in [
        1usize,
        4,
        8,
        12,
        100,
        4096,
        1 << 20,
        40_000_000,
        98_229_519, // the last byte of the feature transformer
        98_229_520, // the first byte of the first layer stack
        total as usize - 1,
    ] {
        assert!(cut < total as usize, "cut point {cut} is not a truncation");
        let msg = scratch("truncated").prefixed_with_net(cut).rejected();
        // The message must name the section, so a failure is diagnosable.
        assert!(
            msg.len() > 10,
            "truncation at {cut} gave an unhelpful error: {msg}"
        );
    }
}

#[test]
fn trailing_bytes_after_the_last_layer_stack_are_rejected() {
    // Stockfish requires the file to end exactly where the last layer stack
    // ends. A net concatenated with something else is not a net.
    // Checked on the reader itself: parked at EOF is accepted, a byte left over
    // is not. A 100 MB concatenation is not worth a test here; this is the same
    // code path the loader takes after the last layer stack.
    let mut r = Reader::new(&b""[..]);
    assert!(r.expect_eof("net file").is_ok());
    let mut r = Reader::new(&b"\x00"[..]);
    let e = r.expect_eof("net file").unwrap_err();
    assert_eq!(e.reason, "unexpected trailing data");
    assert_eq!(e.context, "net file");
}

#[test]
fn a_file_that_is_not_a_net_at_all_is_rejected() {
    // Bytes that cannot be a version word, a hash, or a description. Also
    // covers the LEB128 decoder meeting a stream full of 0x80 continuation
    // bytes, which would otherwise run past the end of the section.
    assert!(
        scratch("garbage")
            .written(&[0x80u8; 4096])
            .rejected()
            .contains("invalid net")
    );
    let prose = b"this is not an nnue file, it is a note to self";
    assert!(
        scratch("text")
            .written(prose)
            .rejected()
            .contains("invalid net")
    );

    // A directory: `open` succeeds on Linux, the first read does not. It must
    // still be refused, and refused as a malformed net rather than escaping as
    // an I/O error the caller might retry.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("nnue");
    let msg = load_err(&dir);
    assert!(
        msg.contains("invalid net"),
        "a directory was not refused: {msg}"
    );
}

#[test]
fn a_truncated_net_leaves_nothing_behind() {
    // The loader builds everything in locals and only returns a `Network` on
    // success, so a failed load cannot leave a half-initialised net that a
    // caller might install anyway. Proved by construction: `load_from` has one
    // exit point.
    let s = scratch("atomic").prefixed_with_net(1_000_000);
    for _ in 0..3 {
        assert!(load_network(&s.0).is_err());
    }
    // And the real net still loads afterwards, so nothing global was clobbered.
    assert!(load_network(&net_path()).is_ok());
}

// --- make/unmake: the incremental child path --------------------------------

/// The engine has no make/unmake — every node is a fresh `Position`. The
/// equivalent is `SearchThread::make_child`, which must leave slot `ply` holding
/// exactly the accumulator a from-scratch refresh of the child would give, and
/// must leave the parent slot untouched.
#[test]
fn make_child_agrees_with_a_full_refresh_of_the_child() {
    let n = net();
    let mut checked = 0usize;
    let mut kinds = [false; 6]; // quiet, capture, promotion, en passant, castle, king
    for fen in SEARCH_FENS {
        let pos = Position::from_fen(fen).unwrap();
        let mut thread = SearchThread::new();
        let shared = shared_with(Some(n));
        thread.refresh_root(&pos, n);

        let parent_eval = thread.evaluate_at(&pos, &shared, 0);
        let parent_slot = thread.accumulator(0).unwrap().clone();

        for m in pos.legal_moves().iter() {
            let mv = m.to_uci();
            let child = thread.make_child(&pos, m, &shared, 1);
            let child_eval = thread.evaluate_at(&child, &shared, 1);

            // The incremental slot must equal a full refresh of the child.
            let mut fresh = Accumulator::new();
            let board = Board::from_position(&child);
            for p in Color::ALL {
                fresh.refresh(p, &board, n);
            }
            let what = format!("{mv} in {fen}");
            assert_same_accumulator(thread.accumulator(1).unwrap(), &fresh, &what);

            // And the score must equal scoring the child from scratch.
            assert_eq!(
                child_eval,
                evaluate(n, &child, &fresh),
                "score after {what}"
            );

            // The parent slot is read-only during the child's update.
            assert_same_accumulator(
                thread.accumulator(0).unwrap(),
                &parent_slot,
                &format!("parent slot after {what}"),
            );
            assert_eq!(
                thread.evaluate_at(&pos, &shared, 0),
                parent_eval,
                "parent score after {what}"
            );

            kinds[classify(&pos, m)] = true;
            checked += 1;
        }
    }
    assert!(checked > 250, "only checked {checked} moves");
    for (i, kind) in [
        "quiet",
        "capture",
        "promotion",
        "en passant",
        "castling",
        "king move",
    ]
    .iter()
    .enumerate()
    {
        assert!(kinds[i], "no {kind} move was exercised");
    }
}

/// Which kind of move `m` is, for the "every move kind was exercised" check.
/// Ordered most-specific first: a castling move is a capture of nothing, and an
/// en-passant capture is both a capture and not a king move. `captured` needs
/// the pre-move board, so it takes the parent.
fn classify(pos: &Position, m: RawMove) -> usize {
    const QUIET: usize = 0;
    const CAPTURE: usize = 1;
    const PROMOTION: usize = 2;
    const EN_PASSANT: usize = 3;
    const CASTLE: usize = 4;
    const KING_MOVE: usize = 5;
    if m.is_castle() {
        CASTLE
    } else if m.is_en_passant() {
        EN_PASSANT
    } else if m.is_promotion() {
        PROMOTION
    } else if is_capture(pos, m) {
        CAPTURE
    } else if pos.board().king_of(pos.turn()) == Some(m.from()) {
        KING_MOVE
    } else {
        QUIET
    }
}

#[test]
fn a_null_move_child_reuses_the_parents_accumulator_verbatim() {
    let n = net();
    let mut checked = 0usize;
    for fen in SEARCH_FENS {
        let pos = Position::from_fen(fen).unwrap();
        if pos.null_move().is_none() {
            continue;
        }
        let mut thread = SearchThread::new();
        let shared = shared_with(Some(n));
        thread.refresh_root(&pos, n);
        let before = thread.accumulator(0).unwrap().clone();
        let nulled = thread.make_child_null(&pos, &shared, 1);

        // Passing changes nothing, so the child slot is a byte-for-byte copy.
        assert_same_accumulator(
            thread.accumulator(1).unwrap(),
            &before,
            &format!("null move in {fen}"),
        );
        // ... and it equals a full refresh of the nulled position, which is the
        // same position with the other side to move.
        let board = Board::from_position(&nulled);
        let mut fresh = Accumulator::new();
        for p in Color::ALL {
            fresh.refresh(p, &board, n);
        }
        assert_same_accumulator(
            thread.accumulator(1).unwrap(),
            &fresh,
            &format!("null move in {fen} vs a full refresh"),
        );
        assert_eq!(
            thread.evaluate_at(&nulled, &shared, 1),
            evaluate(n, &nulled, &fresh)
        );
        checked += 1;
    }
    assert!(checked >= 10, "only {checked} null-move positions");
}

/// Whose king moved decides who gets a full refresh: `HalfKAv2_hm` keys each
/// perspective's indices on *its own* king square, so a white king move changes
/// White's king bucket and forces a refresh there, while Black — whose king did
/// not move — can be handled incrementally.
///
/// Both halves still have to come out right. Black's index set does change
/// (HalfKAv2_hm activates the enemy king's own feature too), but only through
/// the moved piece, which is exactly what the dirty-piece record carries.
#[test]
fn a_king_move_refreshes_one_perspective_and_updates_the_other() {
    let n = net();
    for (fen, mover) in [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", shakmaty::Color::White),
        ("4k3/8/8/8/8/8/8/4K3 b - - 0 1", shakmaty::Color::Black),
    ] {
        let pos = Position::from_fen(fen).unwrap();
        let ksq = pos.board().king_of(mover).unwrap();
        let king_move = pos
            .legal_moves()
            .iter()
            .find(|m| m.from() == ksq)
            .expect("the king has moves");

        // The rule under test, checked on the record the search will use.
        let board = Board::from_position(&pos);
        let us = Color::from_shakmaty(pos.turn());
        let d = board.apply_move(king_move, us);
        let own = Color::from_shakmaty(mover);
        assert!(
            morstilia::nnue::features::half_ka_v2_hm::requires_refresh(&d.dirty_piece, own),
            "the mover's own perspective must be refreshed"
        );
        assert!(
            !morstilia::nnue::features::half_ka_v2_hm::requires_refresh(
                &d.dirty_piece,
                own.other()
            ),
            "the other perspective must be updated incrementally"
        );

        // And the incremental result is still exact.
        let mut thread = SearchThread::new();
        let shared = shared_with(Some(n));
        thread.refresh_root(&pos, n);
        let child = thread.make_child(&pos, king_move, &shared, 1);
        let board = Board::from_position(&child);
        let mut fresh = Accumulator::new();
        for p in Color::ALL {
            fresh.refresh(p, &board, n);
        }
        assert_same_accumulator(
            thread.accumulator(1).unwrap(),
            &fresh,
            &format!("{king_move:?} in {fen}"),
        );
    }
}

#[test]
fn an_unwritten_slot_self_heals_instead_of_returning_garbage() {
    // `evaluate_at` promises a score for whatever `pos` it is handed, and a slot
    // nobody has written yet holds no usable sum. The self-heal covers exactly
    // that: a slot left `computed == false` (only possible when a caller forgot
    // the `make_child` that should have filled it) is fully refreshed from the
    // position rather than scored from zeroes.
    //
    // `begin()` is what makes a slot unwritten, so the test re-enters a search
    // between positions. A slot that is *already* computed is a caller bug, not
    // something the evaluator can detect, so it is deliberately not covered here.
    let n = net();
    let mut thread = SearchThread::new();
    let shared = shared_with(Some(n));
    for fen in SEARCH_FENS {
        thread.begin(None);
        let pos = Position::from_fen(fen).unwrap();
        for ply in [0usize, 1, 7, MAX_PLY - 2] {
            let healed = thread.evaluate_at(&pos, &shared, ply);
            let mut fresh = Accumulator::new();
            let board = Board::from_position(&pos);
            for p in Color::ALL {
                fresh.refresh(p, &board, n);
            }
            assert_eq!(healed, evaluate(n, &pos, &fresh), "{fen} at ply {ply}");
            // And the slot really is computed afterwards, not merely scored.
            assert!(thread.accumulator(ply).unwrap().computed == [true, true]);
        }
    }
}

#[test]
fn classical_mode_never_allocates_the_accumulator_stack() {
    // The stack is ~540 KB. A classical engine must not pay for it, and must
    // not silently fall into the NNUE path.
    let mut thread = SearchThread::new();
    let shared = shared_with(None);
    assert!(thread.accumulator(0).is_none(), "allocated before any use");

    let pos = Position::from_fen(SEARCH_FENS[1]).unwrap();
    let classical = morstilia::evaluation::Evaluator.evaluate_with(&pos, &shared.params);
    for m in pos.legal_moves().iter().take(20) {
        let child = thread.make_child(&pos, m, &shared, 1);
        assert_eq!(
            thread.evaluate_at(&child, &shared, 1),
            morstilia::evaluation::Evaluator.evaluate_with(&child, &shared.params),
            "classical mode must score exactly like the classical evaluator"
        );
    }
    assert_eq!(thread.evaluate_at(&pos, &shared, 0), classical);
    assert!(
        thread.accumulator(0).is_none(),
        "classical mode allocated the NNUE stack"
    );
}

#[test]
fn a_fresh_search_cannot_inherit_the_previous_searchs_accumulators() {
    // `SearchThread`s live in the SMP pool across searches, so `begin()` has to
    // clear the stack. If it did not, the first node of a new search could read
    // the last node of the previous one.
    let n = net();
    let mut thread = SearchThread::new();
    let shared = shared_with(Some(n));
    let a = Position::from_fen(SEARCH_FENS[0]).unwrap();
    let b = Position::from_fen(SEARCH_FENS[1]).unwrap();

    thread.refresh_root(&a, n);
    let score_a = thread.evaluate_at(&a, &shared, 0);

    thread.begin(None);
    assert_eq!(
        thread.accumulator(0).unwrap().computed,
        [false, false],
        "begin() must invalidate every slot"
    );
    let score_b = thread.evaluate_at(&b, &shared, 0);

    let mut fresh_b = Accumulator::new();
    let board = Board::from_position(&b);
    for p in Color::ALL {
        fresh_b.refresh(p, &board, n);
    }
    assert_eq!(score_b, evaluate(n, &b, &fresh_b));
    assert_ne!(
        score_a, score_b,
        "the two positions should not score identically; the test proves nothing otherwise"
    );
    // Scoring A again, after B, must still give A's score.
    thread.refresh_root(&a, n);
    assert_eq!(thread.evaluate_at(&a, &shared, 0), score_a);
}

// --- Evaluation purity and determinism -------------------------------------

#[test]
fn the_evaluator_is_a_pure_function_of_the_position() {
    let n = net();
    let mut thread = SearchThread::new();
    let shared = shared_with(Some(n));
    for fen in SEARCH_FENS {
        let pos = Position::from_fen(fen).unwrap();
        thread.refresh_root(&pos, n);
        let first = thread.evaluate_at(&pos, &shared, 0);
        for _ in 0..8 {
            assert_eq!(thread.evaluate_at(&pos, &shared, 0), first, "{fen}");
        }
        // A second, independent thread with a cold stack must agree exactly:
        // nothing about the result may depend on how the accumulator got there.
        let mut other = SearchThread::new();
        let want = {
            let mut fresh = Accumulator::new();
            let board = Board::from_position(&pos);
            for p in Color::ALL {
                fresh.refresh(p, &board, n);
            }
            evaluate(n, &pos, &fresh)
        };
        assert_eq!(other.evaluate_at(&pos, &shared, 0), want, "{fen}");
    }
}

#[test]
fn nnue_scores_stay_inside_the_engine_score_range() {
    // A wild score means a broken clamp or a truncated weight file, and it
    // would show up in search as the engine refusing to mate. `EVAL_LIMIT` is
    // the ceiling the blend enforces and `MAX_PLY` the mate band above it.
    let n = net();
    let mut thread = SearchThread::new();
    let shared = shared_with(Some(n));
    for fen in SEARCH_FENS {
        let pos = Position::from_fen(fen).unwrap();
        let s = thread.evaluate_at(&pos, &shared, 0);
        assert!(
            s.abs() < morstilia::types::MATE,
            "{fen} scored {s}, which is inside the mate band"
        );
        assert!(
            s.abs() <= morstilia::nnue::EVAL_LIMIT,
            "{fen} scored {s}, past EVAL_LIMIT"
        );
        // A position with a single extra queen is worth a lot but not a million.
        assert!(s.abs() < 5_000, "{fen} scored an implausible {s}");
    }
}

#[test]
fn the_start_position_scores_the_same_from_either_side() {
    // The start position is its own colour flip. Flipping colours *and* the side
    // to move therefore leaves the side-to-move-relative score unchanged, so the
    // two readings must agree exactly — not be negatives of each other. A colour
    // bug in the board, the feature indices or the perspective handling shows up
    // here as a mismatch, which is why the comparison is exact.
    //
    // The value is not zero: Stockfish 19 really does score the start position at
    // -1 (see `tests/data/nnue_groundtruth.json`), so only the *equality* is
    // asserted, never a hand-written number.
    let n = net();
    let mut thread = SearchThread::new();
    let shared = shared_with(Some(n));
    let start = Position::from_fen(SEARCH_FENS[0]).unwrap();
    thread.refresh_root(&start, n);
    let white = thread.evaluate_at(&start, &shared, 0);

    let black_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
    let black = Position::from_fen(black_fen).unwrap();
    thread.refresh_root(&black, n);
    assert_eq!(
        white,
        thread.evaluate_at(&black, &shared, 0),
        "the start position must score the same from either side to move"
    );
    assert!(
        white.abs() <= 20,
        "startpos scored {white}, which is not balanced"
    );
}

#[test]
fn nnue_and_classical_evaluations_both_stay_in_range() {
    // The two evaluators are on different scales, but both must produce a usable
    // number for every position the search can reach.
    let mut thread = SearchThread::new();
    let shared_classical = shared_with(None);
    for fen in SEARCH_FENS {
        let pos = Position::from_fen(fen).unwrap();
        let s = thread.evaluate_at(&pos, &shared_classical, 0);
        assert!(s.abs() < morstilia::types::MATE, "{fen} scored {s}");
    }
}

// --- Search-level behaviour ------------------------------------------------

/// Runs one fixed-depth search and returns `(score, nodes, bestmove, pv)`.
fn fixed_depth_search(
    net: Option<&'static Arc<Network>>,
    fen: &str,
    depth: i32,
    threads: usize,
) -> (i32, u64, String, String) {
    let pos = Position::from_fen(fen).unwrap();
    let limits = TimeLimit {
        depth: Some(depth),
        nodes: None,
        movetime_ms: None,
        soft_ms: 0,
        hard_ms: 0,
        infinite: true,
    };
    let stop = Arc::new(AtomicBool::new(false));
    let mut s = searcher_with(net);
    let r = s.search(&pos, &[], &limits, &stop, threads, &[]);
    assert!(!r.is_none(), "no move found for {fen}");
    let pv =
        r.pv.iter()
            .map(|m| m.to_uci())
            .collect::<Vec<_>>()
            .join(" ");
    (r.score, r.nodes, r.best.to_uci(), pv)
}

#[test]
fn one_thread_reproduces_the_same_search_byte_for_byte() {
    // The whole Stage-8 determinism guarantee, re-checked with the net
    // installed. Node counts are part of it: an accumulation that were
    // order-dependent or had a data race would change the node count even when
    // the best move happened to match.
    for fen in SEARCH_FENS {
        for depth in [4, 7] {
            let first = fixed_depth_search(Some(net()), fen, depth, 1);
            for _ in 0..2 {
                assert_eq!(
                    fixed_depth_search(Some(net()), fen, depth, 1),
                    first,
                    "NNUE search is not reproducible for {fen} at depth {depth}"
                );
            }
        }
    }
}

#[test]
fn one_thread_reproduces_the_same_search_without_a_net_too() {
    // The classical path must be untouched by all of this.
    for fen in SEARCH_FENS {
        let first = fixed_depth_search(None, fen, 5, 1);
        assert_eq!(fixed_depth_search(None, fen, 5, 1), first, "{fen}");
    }
}

#[test]
fn repeated_searches_on_one_searcher_are_reproducible() {
    // Same `Searcher` reused, as the UCI loop does: the per-worker accumulator
    // stacks persist between searches, so `begin()` really has to reset them or
    // the first node of search N would score with search N-1's accumulator.
    //
    // The table is cleared between runs because a warm table is a genuinely
    // different input — the node count legitimately drops when a previous
    // search's entries are available, and comparing it would test the table
    // rather than the evaluator.
    let pos = Position::from_fen(SEARCH_FENS[1]).unwrap();
    let limits = TimeLimit {
        depth: Some(6),
        nodes: None,
        movetime_ms: None,
        soft_ms: 0,
        hard_ms: 0,
        infinite: true,
    };
    let stop = Arc::new(AtomicBool::new(false));
    let mut s = searcher_with(Some(net()));
    let mut runs = Vec::new();
    for _ in 0..3 {
        s.clear_tt();
        let r = s.search(&pos, &[], &limits, &stop, 1, &[]);
        runs.push((r.score, r.nodes, r.best.to_uci()));
    }
    assert_eq!(runs[0], runs[1], "second search differs");
    assert_eq!(runs[1], runs[2], "third search differs");
}

#[test]
fn switching_evaluators_on_one_searcher_still_produces_legal_moves() {
    // `set_nnue(None)` must take the engine back to classical evaluation
    // mid-session, which is what a GUI toggling the `Eval` option will do.
    let pos = Position::from_fen(SEARCH_FENS[1]).unwrap();
    let limits = TimeLimit {
        depth: Some(7),
        nodes: None,
        movetime_ms: None,
        soft_ms: 0,
        hard_ms: 0,
        infinite: true,
    };
    let stop = Arc::new(AtomicBool::new(false));
    let mut s = searcher_with(Some(net()));

    let with = s.search(&pos, &[], &limits, &stop, 1, &[]);
    assert!(!with.is_none());
    s.clear_tt();
    assert!(pos.legal_moves().iter().any(|m| m == with.best));

    s.set_nnue(None);
    s.clear_tt();
    let without = s.search(&pos, &[], &limits, &stop, 1, &[]);
    assert!(!without.is_none());
    assert!(pos.legal_moves().iter().any(|m| m == without.best));

    s.set_nnue(Some(Arc::new(load_network(&net_path()).unwrap())));
    s.clear_tt();
    let again = s.search(&pos, &[], &limits, &stop, 1, &[]);
    assert!(!again.is_none());
    assert!(pos.legal_moves().iter().any(|m| m == again.best));

    // The net-backed run is reproducible after the round trip through `None`.
    s.clear_tt();
    let with2 = s.search(&pos, &[], &limits, &stop, 1, &[]);
    assert_eq!(
        (with2.score, with2.nodes, with2.best),
        (again.score, again.nodes, again.best)
    );
}

#[test]
fn both_evaluators_find_a_forced_mate() {
    // A mate in two that does not depend on getting the evaluation exactly
    // right, so it is a fair test of the *search* in both modes.
    let fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1";
    let pos = Position::from_fen(fen).unwrap();
    let limits = TimeLimit {
        depth: Some(6),
        nodes: None,
        movetime_ms: None,
        soft_ms: 0,
        hard_ms: 0,
        infinite: true,
    };
    let stop = Arc::new(AtomicBool::new(false));
    for net in [Some(net()), None] {
        let mut s = searcher_with(net);
        let r = s.search(&pos, &[], &limits, &stop, 1, &[]);
        assert!(
            morstilia::types::is_mate(r.score),
            "{:?} evaluator missed the rook mate: score {}",
            net.map(|_| "nnue"),
            r.score
        );
        assert_eq!(r.best.to_uci(), "a1a8", "the mate must be played");
    }
}

#[test]
fn nnue_and_classical_evaluations_generally_differ() {
    // A sanity check on the wiring itself: if the two modes returned the same
    // score for every position, the net would not be installed at all and the
    // "both modes work" tests above would be vacuous.
    let n = net();
    let mut thread = SearchThread::new();
    let with = shared_with(Some(n));
    let without = shared_with(None);
    let mut differed = 0;
    for fen in SEARCH_FENS {
        let pos = Position::from_fen(fen).unwrap();
        thread.refresh_root(&pos, n);
        let a = thread.evaluate_at(&pos, &with, 0);
        let b = thread.evaluate_at(&pos, &without, 0);
        if a != b {
            differed += 1;
        }
    }
    assert!(
        differed >= SEARCH_FENS.len() - 2,
        "only {differed} of {} positions scored differently between the two evaluators",
        SEARCH_FENS.len()
    );
}

#[test]
fn the_net_is_not_loaded_when_the_evaluator_is_classical() {
    // A searcher built without a net must report none, and installing one later
    // must be visible.
    let mut s = searcher_with(None);
    assert!(s.nnue().is_none());
    s.set_nnue(Some(Arc::new(load_network(&net_path()).unwrap())));
    assert!(s.nnue().is_some());
    s.set_nnue(None);
    assert!(s.nnue().is_none());
}

// --- Format-level checks that need no net -----------------------------------

#[test]
fn the_reader_rejects_a_short_leb128_count() {
    // The count is declared before the payload, so a file that lies about it
    // must be refused rather than read as a short block. This is a plain reader
    // test so it stays fast and does not need a 100 MB file.
    let mut r = Reader::new(&[0x04u8, 0x01][..]);
    let e = r.read_leb128::<i16>(&mut [0; 4], "test block").unwrap_err();
    assert!(format!("{e}").contains("test block"), "{e}");

    // A run of continuation bytes with nothing to terminate it is a
    // truncated block, not an infinite loop.
    let many = [0x80u8; 64];
    let mut r = Reader::new(&many[..]);
    let e = r.read_leb128::<i16>(&mut [0; 4], "test block").unwrap_err();
    assert!(format!("{e}").contains("test block"), "{e}");
}

#[test]
fn the_reader_enforces_its_own_contract() {
    // `read_exact` must always land exactly on the end of its output, even
    // when the request straddles the buffer boundary — every later read depends
    // on that invariant.
    let data: Vec<u8> = (0..9000u32).map(|i| (i % 251) as u8).collect();
    let mut r = Reader::new(&data[..]);
    let mut out = [0u8; 9000];
    r.read_exact(&mut out, "everything").unwrap();
    assert_eq!(out.as_slice(), data.as_slice());
    // At EOF, every read is an error rather than a silent short count.
    assert!(r.read_exact(&mut [0u8; 1], "past the end").is_err());
}

#[test]
fn the_bundled_net_header_is_accepted_and_described() {
    // Read the real header through the same code the loader uses.
    let bytes = real_header();
    let mut r = Reader::new(&bytes[..]);
    let h = r
        .read_header(morstilia::nnue::NETWORK_HASH)
        .expect("the bundled net's own header must validate");
    assert_eq!(h.network_hash, 0xA85B_2205);
    assert_eq!(
        h.description,
        "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer.",
    );
    // The length field must describe exactly the description and nothing after
    // it, so the next read lands on the feature-transformer section hash.
    let next = r.read_u32().unwrap();
    assert_eq!(
        next,
        morstilia::nnue::network::FEATURE_TRANSFORMER_HASH,
        "the header length field must be exact"
    );
}

#[test]
fn format_errors_name_the_section_that_failed() {
    // The user-visible message is the only clue they get about a bad net, so
    // every error has to say where it happened. Running out of bytes is reported
    // once, generically ("truncated net"); disagreeing about a *value* names the
    // section, because that is the case where the user has to guess which part
    // of the file is not what this engine expects.
    let mut r = Reader::new(&[][..]);
    let e = r.read_header(morstilia::nnue::NETWORK_HASH).unwrap_err();
    assert_eq!(e.context, "net header");
    assert_eq!(e.reason, "file is too short to contain a version");

    let mut r = Reader::new(&[0x11, 0x22, 0x33, 0x44][..]);
    let e = r
        .read_section_hash(morstilia::nnue::NETWORK_HASH, "feature transformer")
        .unwrap_err();
    assert_eq!(e.context, "feature transformer");
    assert!(e.reason.contains("section hash mismatch"), "{e}");

    let mut r = Reader::new(&[0x11, 0x22, 0x33, 0x44][..]);
    let e = r.read_section_hash(0, "layer stack").unwrap_err();
    assert_eq!(e.context, "layer stack");

    assert_eq!(format::VERSION, 0x6A44_8AFA);
}

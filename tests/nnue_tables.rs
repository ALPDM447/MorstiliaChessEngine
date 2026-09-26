//! Every NNUE bitboard table is compared against data printed straight out of
//! Stockfish 19.
//!
//! The reference file `tests/data/attacks_reference.txt` was produced by
//! linking a small program against Stockfish's own object files and printing
//! the tables (`Attacks::init()` fills the runtime ones). The unit tests in
//! `src/nnue/attacks.rs` pin the interesting landmarks by hand; this file
//! pins *every* entry, which is what catches a table that is right for the
//! squares somebody thought to check and wrong everywhere else.

use morstilia::nnue::attacks::{
    self, BISHOP_ATTACKS, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, PAWN_PAIR_BB, RAY_PASS_BB,
    ROOK_ATTACKS,
};
use morstilia::nnue::types::{Color, QUEEN};

/// Parses the reference file into `(section, values)`.
fn reference() -> Vec<(String, Vec<u64>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/attacks_reference.txt"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path));
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let mut declared: usize = 0;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            // Close out the previous section before starting a new one: a
            // section may span many lines (ray_pass_bb is 4096 entries).
            if let Some((name, values)) = out.last_mut() {
                assert_eq!(
                    values.len(),
                    declared,
                    "section {name} should hold {declared} entries",
                );
            }
            let (name, count) = rest.split_once(']').expect("section header `[name] count`");
            out.push((name.to_string(), Vec::new()));
            declared = count.trim().parse().expect("section length");
            continue;
        }
        let (_, values) = out.last_mut().expect("data before any section header");
        for token in line.split_whitespace() {
            values.push(u64::from_str_radix(token, 16).unwrap_or_else(|e| panic!("{token}: {e}")));
        }
    }
    if let Some((name, values)) = out.last_mut() {
        assert_eq!(
            values.len(),
            declared,
            "last section {name} should hold {declared} entries",
        );
    }
    out
}

fn section(name: &str) -> Vec<u64> {
    let sections = reference();
    let (_, v) = sections
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("section {name} missing from the reference file"));
    v.clone()
}

#[test]
fn every_reference_section_is_consumed_by_a_test() {
    let expected = [
        "pawn_pair_bb",
        "ray_pass_bb",
        "knight_attacks",
        "king_attacks",
        "bishop_attacks",
        "rook_attacks",
        "white_pawn_attacks",
        "black_pawn_attacks",
    ];
    let got: Vec<String> = reference().into_iter().map(|(n, _)| n).collect();
    assert_eq!(got, expected);
}

#[test]
fn leaper_and_slider_tables_match_stockfish() {
    let knight = section("knight_attacks");
    assert_eq!(knight.len(), 64);
    for s in 0..64 {
        assert_eq!(KNIGHT_ATTACKS[s], knight[s], "knight on {s}");
    }

    let king = section("king_attacks");
    for s in 0..64 {
        assert_eq!(KING_ATTACKS[s], king[s], "king on {s}");
    }

    let bishop = section("bishop_attacks");
    for s in 0..64 {
        assert_eq!(BISHOP_ATTACKS[s], bishop[s], "bishop on {s}");
    }

    let rook = section("rook_attacks");
    for s in 0..64 {
        assert_eq!(ROOK_ATTACKS[s], rook[s], "rook on {s}");
    }

    let white = section("white_pawn_attacks");
    for s in 0..64 {
        assert_eq!(PAWN_ATTACKS[0][s], white[s], "white pawn on {s}");
    }

    let black = section("black_pawn_attacks");
    for s in 0..64 {
        assert_eq!(PAWN_ATTACKS[1][s], black[s], "black pawn on {s}");
    }

    // A queen is the union of its two halves on every square.
    for s in 0..64 {
        assert_eq!(
            attacks::attacks_bb(QUEEN, s, 0),
            bishop[s] | rook[s],
            "queen on {s}",
        );
    }
    // `Color` is Stockfish-ordered, so `White == 0`.
    assert_eq!(Color::White.idx(), 0);
    assert_eq!(attacks::pawn_attacks(Color::White, 28), white[28]);
    assert_eq!(attacks::pawn_attacks(Color::Black, 28), black[28]);
}

#[test]
fn pawn_pair_and_ray_pass_tables_match_stockfish() {
    let pp = section("pawn_pair_bb");
    assert_eq!(pp.len(), 64);
    for s in 0..64 {
        assert_eq!(PAWN_PAIR_BB[s], pp[s], "pawn pair band of {s}");
        assert_eq!(attacks::pawn_pair_bb(s), pp[s], "pawn_pair_bb({s})");
    }

    let rp = section("ray_pass_bb");
    assert_eq!(rp.len(), 64 * 64);
    for s1 in 0..64 {
        for s2 in 0..64 {
            assert_eq!(
                attacks::ray_pass_bb(s1, s2),
                rp[s1 * 64 + s2],
                "ray_pass_bb({s1}, {s2})",
            );
        }
    }
}

/// The tables are `const`, so every *use* re-runs the const evaluator. If the
/// index arithmetic is wrong this is where it shows up as a panic inside
/// `const` evaluation, which is much harder to read than a plain assert.
#[test]
fn const_tables_are_usable_in_a_const_context() {
    const RP: u64 = RAY_PASS_BB[0][24];
    const PP: u64 = PAWN_PAIR_BB[27];
    const KN: u64 = KNIGHT_ATTACKS[27];
    assert_eq!(RP, 0x0101_0101_0101_0100);
    assert_eq!(PP, 0x001c_1c1c_141c_1c00);
    assert_eq!(KN, 0x0000_1422_0022_1400);
}

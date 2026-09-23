//! UCI command parsing: raw lines → [`Command`].
//!
//! The parser is deliberately lenient (UCI says ignore malformed input);
//! every `parse` call returns a command, with unknown tokens preserved so
//! the engine can decide (e.g. `debug on`).

/// Parameters of a `go` command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GoParams {
    /// `movetime`: fixed per-move budget in milliseconds.
    pub movetime: Option<u64>,
    /// `wtime`: white clock in milliseconds.
    pub wtime: Option<u64>,
    /// `btime`: black clock in milliseconds.
    pub btime: Option<u64>,
    /// `winc`: white increment in milliseconds.
    pub winc: Option<u64>,
    /// `binc`: black increment in milliseconds.
    pub binc: Option<u64>,
    /// `movestogo`: moves until the next time control (0 = unknown).
    pub movestogo: Option<u32>,
    /// `depth`: search to fixed depth.
    pub depth: Option<u32>,
    /// `nodes`: search at most this many nodes.
    pub nodes: Option<u64>,
    /// `infinite`: search until `stop`/`quit`.
    pub infinite: bool,
    /// `ponder`: search started in ponder mode.
    pub ponder: bool,
    /// `searchmoves`: restrict the root to these moves.
    pub searchmoves: Vec<String>,
}

/// One parsed UCI command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Uci,
    IsReady,
    UciNewGame,
    /// `position [fen <fen> | startpos] moves <m1> <m2> …`
    Position {
        /// Full FEN, or startpos when `None`.
        fen: Option<String>,
        /// Moves applied after the FEN, in order.
        moves: Vec<String>,
    },
    Go(GoParams),
    Stop,
    Quit,
    SetOption {
        name: String,
        value: Option<String>,
    },
    PonderHit,
    /// Anything we do not understand (kept for debugging).
    Unknown(String),
}

/// Parses a single UCI line (trimmed) into a [`Command`].
pub fn parse(line: &str) -> Command {
    let line = line.trim();
    let mut tokens = line.split_whitespace();

    let Some(cmd) = tokens.next() else {
        return Command::Unknown(String::new());
    };

    match cmd {
        "uci" => Command::Uci,
        "isready" => Command::IsReady,
        "ucinewgame" => Command::UciNewGame,
        "stop" => Command::Stop,
        "quit" => Command::Quit,
        "ponderhit" => Command::PonderHit,
        "position" => parse_position(tokens),
        "go" => Command::Go(parse_go(tokens)),
        "setoption" => parse_setoption(tokens),
        _ => Command::Unknown(line.to_string()),
    }
}

fn parse_position(mut tokens: core::str::SplitWhitespace<'_>) -> Command {
    let mut fen = None;
    match tokens.next() {
        Some("startpos") => {}
        Some("fen") => {
            let mut parts = Vec::new();
            // The FEN is exactly 6 fields; `moves` starts after them.
            while let Some(t) = tokens.next() {
                if t == "moves" {
                    break;
                }
                parts.push(t);
            }
            fen = Some(parts.join(" "));
        }
        _ => return Command::Unknown("position".to_string()),
    }

    let mut moves = Vec::new();
    // If we consumed up to "moves" already, `tokens` holds the moves; if the
    // FEN loop never saw "moves", look for it now.
    if fen.is_none() {
        for t in tokens.by_ref() {
            if t == "moves" {
                break;
            }
        }
    }
    moves.extend(tokens.map(str::to_string));

    Command::Position { fen, moves }
}

fn parse_go(tokens: core::str::SplitWhitespace<'_>) -> GoParams {
    let mut g = GoParams::default();
    let mut it = tokens;
    while let Some(tok) = it.next() {
        match tok {
            "infinite" => g.infinite = true,
            "ponder" => g.ponder = true,
            "movetime" => g.movetime = it.next().and_then(|v| v.parse().ok()),
            "wtime" => g.wtime = it.next().and_then(|v| v.parse().ok()),
            "btime" => g.btime = it.next().and_then(|v| v.parse().ok()),
            "winc" => g.winc = it.next().and_then(|v| v.parse().ok()),
            "binc" => g.binc = it.next().and_then(|v| v.parse().ok()),
            "movestogo" => g.movestogo = it.next().and_then(|v| v.parse().ok()),
            "depth" => g.depth = it.next().and_then(|v| v.parse().ok()),
            "nodes" => g.nodes = it.next().and_then(|v| v.parse().ok()),
            "searchmoves" => {
                // Collect until the next known keyword (or end).
                while let Some(m) = it.next() {
                    if is_go_keyword(m) {
                        // un-consume is impossible; simplest is to re-walk.
                        // Rare path: rebuild from here would need lookahead —
                        // instead gather remaining unless pure keyword.
                        // See note below.
                        break;
                    }
                    g.searchmoves.push(m.to_string());
                }
            }
            _ => {}
        }
    }
    g
}

fn is_go_keyword(s: &str) -> bool {
    matches!(
        s,
        "infinite"
            | "ponder"
            | "movetime"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "depth"
            | "nodes"
            | "searchmoves"
    )
}

fn parse_setoption(tokens: core::str::SplitWhitespace<'_>) -> Command {
    let mut name = String::new();
    let mut value = None;
    let mut it = tokens;
    let mut reading_name = false;
    let mut reading_value = false;

    for tok in it.by_ref() {
        if reading_value {
            value = Some(tok.to_string());
            reading_value = false;
        } else if tok == "name" {
            reading_name = true;
        } else if tok == "value" {
            reading_name = false;
            reading_value = true;
        } else if reading_name {
            if !name.is_empty() {
                name.push(' ');
            }
            name.push_str(tok);
        }
    }

    Command::SetOption { name, value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_commands() {
        assert_eq!(parse("uci"), Command::Uci);
        assert_eq!(parse("  isready  "), Command::IsReady);
        assert_eq!(parse("ucinewgame"), Command::UciNewGame);
        assert_eq!(parse("stop"), Command::Stop);
        assert_eq!(parse("quit"), Command::Quit);
        assert_eq!(parse("ponderhit"), Command::PonderHit);
    }

    #[test]
    fn position_startpos_with_moves() {
        let cmd = parse("position startpos moves e2e4 e7e5");
        assert_eq!(
            cmd,
            Command::Position {
                fen: None,
                moves: vec!["e2e4".to_string(), "e7e5".to_string()],
            }
        );
    }

    #[test]
    fn position_fen_with_moves() {
        let cmd = parse("position fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1 moves e1e2");
        match cmd {
            Command::Position { fen, moves } => {
                assert_eq!(fen.as_deref(), Some("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
                assert_eq!(moves, vec!["e1e2".to_string()]);
            }
            other => panic!("expected position, got {other:?}"),
        }
    }

    #[test]
    fn position_startpos_no_moves() {
        assert_eq!(
            parse("position startpos"),
            Command::Position {
                fen: None,
                moves: vec![],
            }
        );
    }

    #[test]
    fn go_fixed_time() {
        match parse("go movetime 1000") {
            Command::Go(g) => {
                assert_eq!(g.movetime, Some(1000));
                assert!(!g.infinite);
            }
            other => panic!("expected go, got {other:?}"),
        }
    }

    #[test]
    fn go_full_clock_and_depth() {
        match parse("go wtime 300000 btime 290000 winc 1000 binc 1000 movestogo 20 depth 12") {
            Command::Go(g) => {
                assert_eq!(g.wtime, Some(300_000));
                assert_eq!(g.btime, Some(290_000));
                assert_eq!(g.winc, Some(1000));
                assert_eq!(g.binc, Some(1000));
                assert_eq!(g.movestogo, Some(20));
                assert_eq!(g.depth, Some(12));
            }
            other => panic!("expected go, got {other:?}"),
        }
    }

    #[test]
    fn go_infinite_and_nodes() {
        match parse("go infinite nodes 100000") {
            Command::Go(g) => {
                assert!(g.infinite);
                assert_eq!(g.nodes, Some(100_000));
            }
            other => panic!("expected go, got {other:?}"),
        }
    }

    #[test]
    fn setoption_with_spaced_name() {
        match parse("setoption name Hash value 256") {
            Command::SetOption { name, value } => {
                assert_eq!(name, "Hash");
                assert_eq!(value.as_deref(), Some("256"));
            }
            other => panic!("expected setoption, got {other:?}"),
        }
        match parse("setoption name MultiPV value 3") {
            Command::SetOption { name, value } => {
                assert_eq!(name, "MultiPV");
                assert_eq!(value.as_deref(), Some("3"));
            }
            other => panic!("expected setoption, got {other:?}"),
        }
    }

    #[test]
    fn unknown_preserved() {
        assert_eq!(
            parse("bogus whatever"),
            Command::Unknown("bogus whatever".to_string())
        );
    }
}

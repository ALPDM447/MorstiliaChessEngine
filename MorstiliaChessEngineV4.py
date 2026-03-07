import chess
import sys
import chess.polyglot
import os
import glob
import threading
import time
import random
import logging
import argparse
import chess.pgn
import datetime
import io

#nuitka tek dosya için
def getBasePath():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = getBasePath()
BOOK_PATH = os.path.join(BASE_DIR, "AllOpeningsMorstilia.bin")

BOOKS = []

def load_books(path=None):
    global BOOKS
    # close existing
    for r in BOOKS:
        try:
            r.close()
        except Exception:
            pass
    BOOKS = []
    if path is None:
        path = engine_options.get("BookPath", BOOK_PATH)
    # if path is a directory, open all .bin files
    try:
        if os.path.isdir(path):
            for f in glob.glob(os.path.join(path, "*.bin")):
                try:
                    reader = chess.polyglot.open_reader(f)
                    BOOKS.append(reader)
                    print(f"Loaded book: {f}")
                except Exception:
                    print(f"Could not load book: {f}")
        elif os.path.isfile(path):
            try:
                reader = chess.polyglot.open_reader(path)
                BOOKS.append(reader)
                print(f"Loaded book: {path}")
            except Exception:
                print(f"Could not load book: {path}")
        else:
            # try default book folder
            default_folder = os.path.join(BASE_DIR, "book")
            for f in glob.glob(os.path.join(default_folder, "*.bin")):
                try:
                    reader = chess.polyglot.open_reader(f)
                    BOOKS.append(reader)
                    print(f"Loaded book: {f}")
                except Exception:
                    print(f"Could not load book: {f}")
    except Exception:
        pass


def getBookMove(board):
    if not engine_options.get("BookEnabled", True):
        return None
    entries_all = []
    for book in BOOKS:
        try:
            entries = list(book.find_all(board))
            if entries:
                entries_all.extend(entries)
        except Exception:
            continue
    if not entries_all:
        return None
    # weighted random choice by entry.weight
    total = sum(e.weight for e in entries_all)
    if total <= 0:
        return max(entries_all, key=lambda e: e.weight).move
    r = random.uniform(0, total)
    acc = 0
    for e in entries_all:
        acc += e.weight
        if r <= acc:
            return e.move
    return entries_all[-1].move

knightScore = [
-50,-40,-30,-30,-30,-30,-40,-50,
-40,-20,  0,  5,  5,  0,-20,-40,
-30,  5, 10, 15, 15, 10,  5,-30,
-30,  0, 15, 20, 20, 15,  0,-30,
-30,  5, 15, 20, 20, 15,  5,-30,
-30,  0, 10, 15, 15, 10,  0,-30,
-40,-20,  0,  0,  0,  0,-20,-40,
-50,-40,-30,-30,-30,-30,-40,-50,
]
bishopScore = [
-20,-10,-10,-10,-10,-10,-10,-20,
-10,  5,  0,  0,  0,  0,  5,-10,
-10, 10, 10, 10, 10, 10, 10,-10,
-10,  0, 10, 10, 10, 10,  0,-10,
-10,  5,  5, 10, 10,  5,  5,-10,
-10,  0,  5, 10, 10,  5,  0,-10,
-10,  0,  0,  0,  0,  0,  0,-10,
-20,-10,-10,-10,-10,-10,-10,-20,
]
queenScore = [
-20,-10,-10, -5, -5,-10,-10,-20,
-10,  0,  5,  0,  0,  0,  0,-10,
-10,  5,  5,  5,  5,  5,  0,-10,
 -5,  0,  5,  5,  5,  5,  0, -5,
  0,  0,  5,  5,  5,  5,  0, -5,
-10,  0,  5,  5,  5,  5,  0,-10,
-10,  0,  0,  0,  0,  0,  0,-10,
-20,-10,-10, -5, -5,-10,-10,-20,
]
rookScore= [
  0,  0,  5, 10, 10,  5,  0,  0,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
 -5,  0,  0,  0,  0,  0,  0, -5,
  5, 10, 10, 10, 10, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0,
]
pawnScore = [
  0,  0,  0,  0,  0,  0,  0,  0,
 50, 50, 50, 50, 50, 50, 50, 50,
 10, 10, 20, 30, 30, 20, 10, 10,
  5,  5, 10, 25, 25, 10,  5,  5,
  0,  0,  0, 20, 20,  0,  0,  0,
  5, -5,-10,  0,  0,-10, -5,  5,
  5, 10, 10,-20,-20, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0,
]

PieceValues = {
    chess.PAWN : 100,
    chess.KNIGHT: 320,
    chess.BISHOP : 340,
    chess.ROOK : 500,
    chess.QUEEN : 900,
}

PST = {  # Piece-Square Tables
    chess.PAWN: pawnScore,
    chess.KNIGHT: knightScore,
    chess.BISHOP: bishopScore,
    chess.ROOK: rookScore,
    chess.QUEEN: queenScore,
}

def Puanla(board):

    if board.is_checkmate():
        return -9999999  # side to move kaybetti

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    for piece in PieceValues:

        # White taşlar
        for square in board.pieces(piece, chess.WHITE):
            score += PieceValues[piece]
            if piece in PST:
                score += PST[piece][square]

        # Black taşlar
        for square in board.pieces(piece, chess.BLACK):
            score -= PieceValues[piece]
            if piece in PST:
                score -= PST[piece][chess.square_mirror(square)]

    # Negamax uyumu
    # Mobility: legal moves difference
    mobility = (board.legal_moves.count() if board.turn == chess.WHITE else 0)
    # Using simple mobility: difference in move counts for both sides
    try:
        board_turn = board.turn
        board.push(chess.Move.null())
        opp_moves = board.legal_moves.count()
        board.pop()
    except Exception:
        opp_moves = 0

    if board.turn == chess.WHITE:
        mobility = board.legal_moves.count() - opp_moves
    else:
        mobility = opp_moves - board.legal_moves.count()
    score += mobility * 10

    # Bishop pair bonus
    white_bishops = len(list(board.pieces(chess.BISHOP, chess.WHITE)))
    black_bishops = len(list(board.pieces(chess.BISHOP, chess.BLACK)))
    if white_bishops >= 2:
        score += 50
    if black_bishops >= 2:
        score -= 50

    # Pawn structure: doubled and isolated pawns (simple)
    def pawn_structure(side):
        files = {}
        for sq in board.pieces(chess.PAWN, side):
            f = chess.square_file(sq)
            files[f] = files.get(f, 0) + 1
        doubled = sum(v - 1 for v in files.values() if v > 1)
        isolated = 0
        for f in range(8):
            if files.get(f, 0) > 0:
                if files.get(f - 1, 0) == 0 and files.get(f + 1, 0) == 0:
                    isolated += files.get(f, 0)
        return doubled, isolated

    wd, wi = pawn_structure(chess.WHITE)
    bd, bi = pawn_structure(chess.BLACK)
    score -= (wd - bd) * 20  # doubled penalty
    score -= (wi - bi) * 30  # isolated penalty

    # King safety (very simple): pawn shield in front of king
    def king_shield_score(side):
        ksq = list(board.pieces(chess.KING, side))[0] if list(board.pieces(chess.KING, side)) else None
        if ksq is None:
            return 0
        file = chess.square_file(ksq)
        rank = chess.square_rank(ksq)
        shield = 0
        directions = [(-1, 1), (0, 1), (1, 1)] if side == chess.WHITE else [(-1, -1), (0, -1), (1, -1)]
        for dx, dy in directions:
            f = file + dx
            r = rank + dy
            if 0 <= f < 8 and 0 <= r < 8:
                sq = chess.square(f, r)
                p = board.piece_at(sq)
                if p is not None and p.piece_type == chess.PAWN and p.color == side:
                    shield += 1
        return shield * 10

    score += king_shield_score(chess.WHITE)
    score -= king_shield_score(chess.BLACK)

    return score if board.turn == chess.WHITE else -score

def negamax(board, depth, alpha, beta, ply=0):
    # cooperative stop
    if globals().get("STOP_SEARCH", False):
        return 0
    if board.is_game_over():
        return Puanla(board)
    if depth == 0:
        return quiescence(board, alpha, beta, ply)

    orig_alpha = alpha
    key = zobrist_hash(board)
    # TT probe
    tt_hit = tt_probe(key, depth, alpha, beta)
    if tt_hit is not None:
        return tt_hit

    max_eval = -float("inf")
    best_move = None

    moves_iter = order_moves(board, board.legal_moves, ply)
    for move in moves_iter:
        board.push(move)
        eval = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            best_move = move
        alpha = max(alpha, eval)
        if alpha >= beta:
            # store killer if not a capture
            try:
                if not board.is_capture(move):
                    kms = KILLERS.setdefault(ply, [])
                    u = move.uci()
                    if u not in kms:
                        kms.insert(0, u)
                        if len(kms) > 2:
                            kms.pop()
            except Exception:
                pass
            # update history for non-capture
            try:
                if not board.is_capture(move):
                    HISTORY[move.uci()] = HISTORY.get(move.uci(), 0) + (depth * depth)
            except Exception:
                pass
            break

    # store in TT
    if max_eval <= orig_alpha:
        flag = "UPPERBOUND"
    elif max_eval >= beta:
        flag = "LOWERBOUND"
    else:
        flag = "EXACT"
    try:
        tt_store(key, depth, max_eval, flag, best_move)
    except Exception:
        pass

    return max_eval


def generate_captures(board):
    return [m for m in board.legal_moves if board.is_capture(m) or m.promotion is not None]


def quiescence(board, alpha, beta, ply=0):
    # stand pat
    stand_pat = Puanla(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    moves = order_moves(board, generate_captures(board), ply)
    for move in moves:
        if globals().get("STOP_SEARCH", False):
            break
        board.push(move)
        score = -quiescence(board, -beta, -alpha, ply + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def findBestMove(board, depth):
    best_move = None
    alpha = -float("inf")
    beta = float("inf")
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if score > alpha:
            alpha = score
            best_move = move
    return best_move

# Global search control and engine options
STOP_SEARCH = False
engine_options = {
    "Hash": 16,
    "Threads": 1,
    "SearchDepth": 5,
    "BookPath": BOOK_PATH,
    "BookEnabled": True,
}

# Transposition table and Zobrist hashing
TRANSPOSITION_TABLE = {}
TT_MAX_ENTRIES = engine_options.get("Hash", 16) * 1024
ZOBRIST_PIECE = {}
ZOBRIST_BLACK_TO_MOVE = 0

def init_zobrist(seed=0xC0FFEE):
    global ZOBRIST_PIECE, ZOBRIST_BLACK_TO_MOVE, TT_MAX_ENTRIES
    rnd = random.Random(seed)
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    colors = [chess.WHITE, chess.BLACK]
    ZOBRIST_PIECE = {}
    for piece in pieces:
        for color in colors:
            for sq in range(64):
                key = (piece, color, sq)
                ZOBRIST_PIECE[key] = rnd.getrandbits(64)
    ZOBRIST_BLACK_TO_MOVE = rnd.getrandbits(64)
    TT_MAX_ENTRIES = max(1024, engine_options.get("Hash", 16) * 1024)


def zobrist_hash(board):
    h = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None:
            key = (p.piece_type, p.color, sq)
            h ^= ZOBRIST_PIECE.get(key, 0)
    if board.turn == chess.BLACK:
        h ^= ZOBRIST_BLACK_TO_MOVE
    return h


def tt_probe(key, depth, alpha, beta):
    entry = TRANSPOSITION_TABLE.get(key)
    if entry is None:
        return None
    if entry["depth"] < depth:
        return None
    flag = entry["flag"]
    score = entry["score"]
    if flag == "EXACT":
        return score
    if flag == "LOWERBOUND":
        if score > alpha:
            alpha = score
    elif flag == "UPPERBOUND":
        if score < beta:
            beta = score
    if alpha >= beta:
        return score
    return None


def tt_store(key, depth, score, flag, move):
    if len(TRANSPOSITION_TABLE) > TT_MAX_ENTRIES:
        try:
            TRANSPOSITION_TABLE.pop(next(iter(TRANSPOSITION_TABLE)))
        except StopIteration:
            TRANSPOSITION_TABLE.clear()
    TRANSPOSITION_TABLE[key] = {"depth": depth, "score": score, "flag": flag, "move": move}


# Move ordering helpers: MVV-LVA, killers, history
KILLERS = {}  # ply -> list of killer move uci strings
HISTORY = {}  # move_uci -> score

def move_uci_key(move):
    return move.uci()

def mvv_lva_score(board, move):
    # higher is better
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim is None:
            # en passant capture: victim is pawn
            victim_value = PieceValues.get(chess.PAWN, 100)
        else:
            victim_value = PieceValues.get(victim.piece_type, 0)
        attacker_value = PieceValues.get(attacker.piece_type, 0) if attacker is not None else 0
        return victim_value * 1000 - attacker_value
    # promotions get a bonus
    if move.promotion is not None:
        return 90000
    return 0

def order_moves(board, moves, ply):
    # moves: iterable of chess.Move
    mv_list = list(moves)
    tt_key = zobrist_hash(board)
    tt_entry = TRANSPOSITION_TABLE.get(tt_key)
    tt_move_uci = tt_entry["move"].uci() if tt_entry and tt_entry.get("move") else None

    def score(m):
        s = 0
        # TT move highest
        try:
            if tt_move_uci is not None and m.uci() == tt_move_uci:
                s += 1000000
        except Exception:
            pass
        # captures via MVV-LVA
        s += mvv_lva_score(board, m)
        # killer moves
        kms = KILLERS.get(ply, [])
        try:
            if m.uci() in kms:
                s += 50000
        except Exception:
            pass
        # history heuristic
        s += HISTORY.get(m.uci(), 0)
        return -s  # negative for sorting ascending

    mv_list.sort(key=score)
    return mv_list

def _iterative_deepening_search(board, max_depth, time_limit=None):
    global STOP_SEARCH
    best_move = None
    start = time.perf_counter()
    prev_score = 0
    for depth in range(1, max_depth + 1):
        if STOP_SEARCH:
            break
        # time check
        if time_limit is not None and (time.perf_counter() - start) * 1000 >= time_limit:
            break

        # aspiration window logic
        if depth == 1:
            score, mv = search_root(board, depth, -float("inf"), float("inf"))
        else:
            asp = 50  # centipawns
            alpha = prev_score - asp
            beta = prev_score + asp
            score, mv = search_root(board, depth, alpha, beta)
            # if failed low or high, research with full window
            if score <= alpha or score >= beta:
                score, mv = search_root(board, depth, -float("inf"), float("inf"))

        if mv is not None:
            best_move = mv
        prev_score = score

        # cooperative yield / time check after depth
        if STOP_SEARCH:
            break
        if time_limit is not None and (time.perf_counter() - start) * 1000 >= time_limit:
            break

    return best_move


def search_root(board, depth, alpha, beta):
    # Search the root with alpha-beta and return (score, best_move)
    best_move = None
    best_score = -float("inf")
    a = alpha
    b = beta
    moves = order_moves(board, board.legal_moves, ply=0)
    for move in moves:
        if globals().get("STOP_SEARCH", False):
            break
        board.push(move)
        # probe TT for child position
        key = zobrist_hash(board)
        tt_val = tt_probe(key, depth - 1, -b, -a)
        if tt_val is not None:
            score = -tt_val
        else:
            score = -negamax(board, depth - 1, -b, -a, ply=1)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if score > a:
            a = score
        if a >= b:
            break
    return best_score, best_move



def uci_loop():
    board = chess.Board()

    # initialize zobrist and TT
    try:
        init_zobrist()
    except Exception:
        pass
    # load books if available
    try:
        if engine_options.get("BookEnabled", True):
            load_books(engine_options.get("BookPath"))
    except Exception:
        pass

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        command = line.strip()

        if command == "uci":
            print("id name Morstilia V5")
            sys.stdout.flush()
            print("id author Alp")
            sys.stdout.flush()
            # expose some options
            print("option name Hash type spin default 16 min 1 max 1024")
            sys.stdout.flush()
            print("option name Threads type spin default 1 min 1 max 64")
            sys.stdout.flush()
            print("option name BookPath type string default %s" % engine_options.get("BookPath"))
            sys.stdout.flush()
            print("option name BookEnabled type check default true")
            sys.stdout.flush()
            print("option name SearchDepth type spin default %d min 1 max 64" % engine_options.get("SearchDepth", 5))
            sys.stdout.flush()
            print("option name Ponder type check default false")
            sys.stdout.flush()
            print("uciok")
            sys.stdout.flush()

        elif command == "isready":
            print("readyok")
            sys.stdout.flush()

        elif command == "ucinewgame":
            board.reset()
            sys.stdout.flush()
        
        elif command.startswith("position"):
           parts = command.split()
           if "startpos" in parts:
               board.reset()
           if "moves" in parts:
               idx = parts.index("moves")
               for m in parts[idx+1:]:
                   board.push_uci(m)
           sys.stdout.flush()

        elif command.startswith("setoption"):
            # basic parsing: setoption name <name> value <value>
            parts = command.split()
            try:
                if "name" in parts:
                    name_idx = parts.index("name") + 1
                    # collect until 'value' or end
                    if "value" in parts:
                        val_idx = parts.index("value") + 1
                        name = " ".join(parts[name_idx:parts.index("value")])
                        value = " ".join(parts[val_idx:])
                    else:
                        name = " ".join(parts[name_idx:])
                        value = ""
                    name = name.strip()
                    value = value.strip()
                    # parse booleans
                    if value.lower() in ("true", "false"):
                        parsed_value = value.lower() == "true"
                    else:
                        try:
                            parsed_value = int(value)
                        except Exception:
                            parsed_value = value
                    if name in engine_options:
                        engine_options[name] = parsed_value
                        # reload books if needed
                        if name in ("BookPath", "BookEnabled"):
                            try:
                                if engine_options.get("BookEnabled", True):
                                    load_books(engine_options.get("BookPath"))
                                else:
                                    # close books
                                    for r in BOOKS:
                                        try:
                                            r.close()
                                        except Exception:
                                            pass
                                    BOOKS.clear()
                            except Exception:
                                pass
            except Exception:
                pass

        elif command.startswith("go"):
            parts = command.split()
            # parse simple go parameters: depth, movetime, wtime, btime, infinite
            max_depth = engine_options.get("SearchDepth", 5)
            time_limit = None
            infinite = False
            if "depth" in parts:
                try:
                    max_depth = int(parts[parts.index("depth") + 1])
                except Exception:
                    pass
            if "movetime" in parts:
                try:
                    time_limit = int(parts[parts.index("movetime") + 1])
                except Exception:
                    pass
            else:
                # compute a smarter allocation using remaining time / increments / movestogo
                try:
                    tl = compute_time_limit(parts, board)
                    if tl is not None:
                        time_limit = tl
                except Exception:
                    pass
            if "infinite" in parts:
                infinite = True

            # stop any running search
            try:
                globals()["STOP_SEARCH"] = True
            except Exception:
                pass

            # start threaded search so we can accept 'stop' command
            def search_worker(board_copy, max_depth, time_limit, infinite):
                global STOP_SEARCH
                STOP_SEARCH = False
                # prefer book move first
                move = getBookMove(board_copy)
                if move is None:
                    if infinite:
                        # run iterative deepening until stopped
                        mv = _iterative_deepening_search(board_copy, max_depth=64, time_limit=None)
                    else:
                        mv = _iterative_deepening_search(board_copy, max_depth=max_depth, time_limit=time_limit)
                    move = mv
                if move is None:
                    print("bestmove 0000")
                else:
                    print(f"bestmove {move.uci()}")
                sys.stdout.flush()

            board_copy = board.copy()
            t = threading.Thread(target=search_worker, args=(board_copy, max_depth, time_limit, infinite))
            t.daemon = True
            t.start()
        elif command == "stop":
            try:
                globals()["STOP_SEARCH"] = True
            except Exception:
                pass
            sys.stdout.flush()

        elif command == "quit":
            for reader in BOOKS:
                try:
                    reader.close()
                except:
                    pass
            return

def compute_time_limit(parts, board):
    """Compute a reasonable time limit in milliseconds based on UCI 'go' parameters.

    Considers: movetime, wtime/btime, winc/binc, movestogo. Returns an int (ms) or None.
    """
    try:
        # explicit movetime
        if "movetime" in parts:
            try:
                return int(parts[parts.index("movetime") + 1])
            except Exception:
                return None

        inc = 0
        rem = None
        # remaining time and increment for active color
        if board.turn == chess.WHITE:
            if "wtime" in parts:
                try:
                    rem = int(parts[parts.index("wtime") + 1])
                except Exception:
                    rem = None
            if "winc" in parts:
                try:
                    inc = int(parts[parts.index("winc") + 1])
                except Exception:
                    inc = 0
        else:
            if "btime" in parts:
                try:
                    rem = int(parts[parts.index("btime") + 1])
                except Exception:
                    rem = None
            if "binc" in parts:
                try:
                    inc = int(parts[parts.index("binc") + 1])
                except Exception:
                    inc = 0

        movestogo = None
        if "movestogo" in parts:
            try:
                movestogo = int(parts[parts.index("movestogo") + 1])
            except Exception:
                movestogo = None

        if rem is None:
            return None

        # safety reserve to avoid flagging
        reserve = max(50, int(rem * 0.02))

        if movestogo and movestogo > 0:
            time_per_move = rem // movestogo + inc
            alloc = time_per_move - reserve
            return max(20, int(alloc))

        # no movestogo: choose a fraction depending on remaining time
        if rem < 10_000:
            frac = 0.5
        elif rem < 60_000:
            frac = 0.15
        else:
            frac = 0.03

        alloc = int(rem * frac) + inc - reserve
        return max(20, int(alloc))
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morstilia Chess Engine V5")
    parser.add_argument("--mode", choices=["uci", "selfplay"], default="uci")
    parser.add_argument("--games", type=int, default=1, help="Number of selfplay games")
    parser.add_argument("--depth", type=int, default=5, help="Search depth for selfplay")
    parser.add_argument("--log", type=str, default=None, help="Log file path for matches")
    parser.add_argument("--replay", type=str, default=None, help="Replay a PGN file to console")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between moves during replay")
    args = parser.parse_args()

    def setup_logger(logpath=None):
        logger = logging.getLogger("Morstilia")
        logger.setLevel(logging.INFO)
        if logger.handlers:
            return logger
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        if logpath:
            fh = logging.FileHandler(logpath, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger

    def play_one_game(depth, logger=None):
        board = chess.Board()
        moves = []
        while not board.is_game_over():
            move = getBookMove(board)
            if move is None:
                move = findBestMove(board, depth)
            if move is None:
                break
            moves.append(move)
            board.push(move)
        result = board.result(claim_draw=True)
        if logger:
            logger.info(f"Game result: {result}; moves: {' '.join(m.uci() for m in moves)}")
        return board, moves


    def export_game_pgn(board, moves, filename=None, event="Morstilia Selfplay"):
        game = chess.pgn.Game()
        game.headers["Event"] = event
        game.headers["Date"] = datetime.date.today().isoformat()
        node = game
        board0 = chess.Board()
        for m in moves:
            node = node.add_variation(m)
            board0.push(m)
        game.headers["Result"] = board0.result(claim_draw=True)
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(str(game))
        else:
            return str(game)


    def import_pgn_moves(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                game = chess.pgn.read_game(f)
            if game is None:
                return None
            board = game.board()
            moves = []
            for move in game.mainline_moves():
                moves.append(move)
            return moves
        except Exception:
            return None


    def replay_pgn(path, delay=1.0):
        moves = import_pgn_moves(path)
        if moves is None:
            print(f"Could not read PGN from {path}")
            return
        board = chess.Board()
        print(board)
        for m in moves:
            time.sleep(delay)
            board.push(m)
            print("\nMove:", m.uci())
            print(board)

    logger = setup_logger(args.log)

    try:
        if args.mode == "uci":
            uci_loop()
        elif args.mode == "selfplay":
            for i in range(args.games):
                logger.info(f"Starting selfplay game {i+1}/{args.games}")
                board, moves = play_one_game(args.depth, logger)
                # export PGN per game
                pgn_name = f"selfplay_game_{i+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
                try:
                    export_game_pgn(board, moves, filename=pgn_name)
                    logger.info(f"Exported PGN to {pgn_name}")
                except Exception:
                    logger.exception("Failed to export PGN")
                logger.info(f"Finished game {i+1}")
        # replay mode can be called directly
        if args.replay:
            replay_pgn(args.replay, delay=args.delay)
    finally:
        for reader in BOOKS:
            try:
                reader.close()
            except:
                pass
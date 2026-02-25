import chess
import sys
import chess.polyglot
import os

#nuitka tek dosya için
def getBasePath():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = getBasePath()
BOOK_PATH = os.path.join(BASE_DIR, "AllOpeningsMorstilia.bin")

BOOKS = []

try:
    reader = chess.polyglot.open_reader(BOOK_PATH)
    BOOKS.append(reader)
    print(f"Loaded book: {BOOK_PATH}")
except Exception:
    print(f"Could not load book: {BOOK_PATH}")


def getBookMove(board):
    for book in BOOKS:
        try:
            entries = list(book.find_all(board))

            if entries:
                return max(entries, key=lambda e: e.weight).move
        except Exception:
            continue
    return None

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
    return score if board.turn == chess.WHITE else -score

def negamax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return Puanla(board)
    max_eval = -float("inf")

    for move in board.legal_moves:
        board.push(move)
        eval = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        max_eval = max(max_eval, eval)
        alpha = max(alpha, eval)
        if alpha >= beta:
            break
    return max_eval

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



def uci_loop():
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        command = line.strip()

        if command == "uci":
            print("id name Morstilia V4")
            sys.stdout.flush()
            print("id author Alp")
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

        elif command.startswith("go"):
            move = getBookMove(board)
            if move is None:
                move = findBestMove(board,depth=3)
            if move is None:
                print("bestmove 0000")
            else:
                print(f"bestmove {move.uci()}")
            sys.stdout.flush()

        elif command == "quit":
            for reader in BOOKS:
                try:
                    reader.close()
                except:
                    pass
            return

if __name__ == "__main__":
    try:
        uci_loop()
    finally:
        for reader in BOOKS:
            try:
                reader.close()
            except:
                pass
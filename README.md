<h1 align="center">♟️ Morstilia V5</h1>

<p align="center">
  <b>Lightweight UCI-Compatible Chess Engine</b><br>
  Written in Python using <code>python-chess</code>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue">
  <img src="https://img.shields.io/badge/UCI-Compatible-success">
  <img src="https://img.shields.io/badge/Engine-Chess-orange">
  <img src="https://img.shields.io/badge/License-Open--Source-green">
</p>

<hr>

<h2>📌 Overview</h2>

<p>
Morstilia V5 is a lightweight chess engine built in Python. 
It implements the <b>UCI (Universal Chess Interface)</b> protocol and features 
a Polyglot opening book, piece-square table evaluation, and a 
Negamax search algorithm with alpha-beta pruning.
</p>

<hr>

<h2>🚀 Features</h2>

<ul>
  <li>✅ UCI (Universal Chess Interface) compatible</li>
  <li>📖 Polyglot opening book support</li>
  <li>♞ Piece-square table positional evaluation</li>
  <li>🔍 Negamax search algorithm</li>
  <li>⚡ Alpha-beta pruning</li>
  <li>🧠 Material + positional scoring model</li>
  <li>🪶 Lightweight and easy to modify</li>
</ul>

<hr>

<h2>🛠 Requirements</h2>

<ul>
  <li>Python 3.8+</li>
  <li>python-chess</li>
</ul>

<p><b>Install dependencies:</b></p>

<pre><code>pip install chess</code></pre>

<hr>

<h2>📂 Project Structure</h2>

<pre><code>Morstilia/
│
├── Morstilia.py
├── book/AllOpeningsMorstilia.bin
└── README.md
</code></pre>

<hr>

<h2>⚙️ How It Works</h2>

<h3>1️⃣ Opening Phase</h3>

<p>
Morstilia first checks the Polyglot opening book:
</p>

<pre><code>getBookMove(board)</code></pre>

<p>
If a book move is available, it selects the move with the highest weight.
</p>

<h3>2️⃣ Evaluation Function</h3>

<p>The evaluation function (<code>Puanla</code>) is based on:</p>

<ul>
  <li><b>Material values:</b>
    <ul>
      <li>Pawn = 100</li>
      <li>Knight = 320</li>
      <li>Bishop = 340</li>
      <li>Rook = 500</li>
      <li>Queen = 900</li>
    </ul>
  </li>
  <li>Piece-square tables for positional strength</li>
  <li>Checkmate and draw detection</li>
</ul>

<p>
The score is always returned from the perspective of the side to move.
</p>

<h3>3️⃣ Search Algorithm</h3>

<ul>
  <li>Negamax</li>
  <li>Alpha-beta pruning</li>
  <li>Fixed depth search (default: depth = 3)</li>
</ul>

<pre><code>findBestMove(board, depth=3)</code></pre>

<hr>

<h2>🎮 Using the Engine</h2>

<p>
Morstilia supports the UCI protocol and can be used with GUI programs such as:
</p>

<ul>
  <li>Arena</li>
  <li>Cute Chess</li>
  <li>ChessBase</li>
  <li>Lucas Chess</li>
</ul>


<h3>▶️ Option 1: Run as Windows Executable (.exe) (Recommended)</h3>

<p>
If you have compiled the engine as a <code>.exe</code> file (for example using PyInstaller),
you can directly add the executable to your chess GUI as a UCI engine.
</p>

<p>
Steps:
</p>

<ol>
  <li>Open your chess GUI (Arena, Cute Chess, etc.)</li>
  <li>Go to <b>Add New Engine</b></li>
  <li>Select the <code>Morstilia.exe</code> file</li>
  <li>Confirm UCI detection</li>
</ol>

<p>
The engine will then work like any standard UCI chess engine.
</p>

<p><b>Example UCI commands:</b></p>

<pre><code>uci
isready
position startpos moves e2e4 e7e5
go
</code></pre>

<h3>▶️ Option 2: Run with Python</h3>

<p><b>Run from terminal:</b></p>

<pre><code>python Morstilia.py</code></pre>

<hr>

<h2>📖 Opening Book</h2>

<p>
The engine expects the following Polyglot opening book file:
</p>

<pre><code>AllOpeningsMorstilia.bin</code></pre>

<p>
If the file is not found, the engine continues without an opening book.
</p>

<hr>

<h2>⚙️ Customization</h2>

<p>You can improve the engine by:</p>

<ul>
  <li>Increasing search depth</li>
  <li>Adding:
    <ul>
      <li>Iterative deepening</li>
      <li>Quiescence search</li>
      <li>Transposition tables</li>
      <li>Move ordering</li>
      <li>Killer move heuristic</li>
    </ul>
  </li>
  <li>Improving evaluation:
    <ul>
      <li>King safety</li>
      <li>Pawn structure evaluation</li>
      <li>Mobility scoring</li>
      <li>Endgame scaling</li>
    </ul>
  </li>
</ul>

<hr>

<h2>📌 Current Limitations</h2>

<ul>
  <li>Fixed depth search</li>
  <li>No time management</li>
  <li>No transposition table</li>
  <li>No quiescence search</li>
  <li>Basic positional evaluation</li>
</ul>

<h2>📜 License</h2>

<p>
This project is open-source. You are free to modify and improve it.
</p>
<h2>📬 Contact</h2>
<p>
Author: <b>Alp Dumlupınar</b><br>
Email: morstilia@gmail.com<br>
GitHub: https://github.com/ALPDM447
</p>

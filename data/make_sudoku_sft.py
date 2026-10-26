"""Generate the small SFT dataset used by Gemma4_(E2B)_SFT_Sudoku_Board_Parsing.ipynb.

Each example asks the model to turn an 81-character puzzle string (the format
puzzle sites and datasets use) into the `board` list of lists that the GRPO
notebook's `strategy(board, initial)` takes. The answers are computed, not
written by hand, which is what makes this a good fit for SFT.

    python data/make_sudoku_sft.py   # rewrites data/sudoku_sft_{train,test}.jsonl
"""
import json
import os
import random

PROMPTS = [
    "Convert this Sudoku puzzle string into a Python `board` (9x9 list of lists, 0 for empty cells):\n{puzzle}",
    "Here is a Sudoku puzzle as an 81-character string, read row by row. Write it as a Python 9x9 list of lists called `board`, using 0 for blanks.\n{puzzle}",
    "Parse this sudoku into a 9x9 `board` list for Python. Empty cells must be 0.\n\n{puzzle}",
    "{puzzle}\n\nTurn the Sudoku string above into `board = [...]`, nine rows of nine ints, 0 = empty.",
]


def valid(board, r, c, n):
    if n in board[r] or any(board[i][c] == n for i in range(9)):
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return all(board[i][j] != n for i in range(br, br + 3) for j in range(bc, bc + 3))


def solve(board, rng):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                digits = list(range(1, 10))
                rng.shuffle(digits)
                for n in digits:
                    if valid(board, r, c, n):
                        board[r][c] = n
                        if solve(board, rng):
                            return True
                        board[r][c] = 0
                return False
    return True


def make_example(rng):
    board = [[0] * 9 for _ in range(9)]
    solve(board, rng)
    cells = rng.sample(range(81), rng.randint(25, 60))
    for i in cells:
        board[i // 9][i % 9] = 0

    blank = rng.choice(".0")
    puzzle = "".join(str(n) if n else blank for row in board for n in row)
    rows = ",\n".join("    [" + ", ".join(map(str, row)) + "]" for row in board)
    answer = f"```python\nboard = [\n{rows},\n]\n```"
    return {
        "messages": [
            {"role": "user", "content": rng.choice(PROMPTS).format(puzzle = puzzle)},
            {"role": "assistant", "content": answer},
        ],
        "board": board,
    }


if __name__ == "__main__":
    rng = random.Random(3407)
    here = os.path.dirname(os.path.abspath(__file__))
    for split, n in [("train", 1000), ("test", 50)]:
        with open(os.path.join(here, f"sudoku_sft_{split}.jsonl"), "w", newline = "\n") as f:
            for _ in range(n):
                f.write(json.dumps(make_example(rng)) + "\n")

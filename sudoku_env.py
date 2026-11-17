"""Sudoku environment and helpers for the Gemma 4 GRPO Sudoku notebook.

This is the setup the notebook needs but does not teach: the Sudoku game, the
harness that plays a strategy against it, and small helpers for printing and
logging. The parts that define the RL task (prompt, reward functions, trainer)
stay in the notebook.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import copy
import json
import os
import random
import re

from unsloth import check_python_modules, create_locked_down_function, execute_with_time_limit


def _is_valid_placement(board: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if placing num at (row, col) is valid."""
    # Check row
    if num in board[row]:
        return False

    # Check column
    if num in [board[r][col] for r in range(9)]:
        return False

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False

    return True


def _solve_sudoku(board: List[List[int]]) -> bool:
    """Solve sudoku using backtracking (for puzzle generation)."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if _is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if _solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def _generate_complete_board(rng: random.Random) -> List[List[int]]:
    """Generate a complete valid Sudoku board."""
    board = [[0 for _ in range(9)] for _ in range(9)]

    # Fill diagonal 3x3 boxes first (they don't affect each other)
    for box in range(3):
        nums = list(range(1, 10))
        rng.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]

    # Solve the rest
    _solve_sudoku(board)
    return board


@dataclass
class SudokuGame:
    difficulty: int = 40  # Number of cells to remove (20=easy, 40=medium, 50=hard)
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _solution: List[List[int]] = field(init=False, repr=False)
    _initial_board: List[List[int]] = field(init=False, repr=False)
    _moves: int = field(default=0, init=False, repr=False)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

        # Generate complete board
        complete_board = _generate_complete_board(self._rng)
        self._solution = copy.deepcopy(complete_board)

        # Remove cells to create puzzle
        self._board = copy.deepcopy(complete_board)
        cells = [(r, c) for r in range(9) for c in range(9)]
        self._rng.shuffle(cells)

        for r, c in cells[:self.difficulty]:
            self._board[r][c] = 0

        self._initial_board = copy.deepcopy(self._board)
        self._update_state()

    def board(self) -> List[List[int]]:
        """Return current board state."""
        return [row[:] for row in self._board]

    def initial_board(self) -> List[List[int]]:
        """Return initial puzzle state."""
        return [row[:] for row in self._initial_board]

    def state(self) -> str:
        """Return game state: 'ongoing', 'success', or 'failed'."""
        return self._state

    def moves(self) -> int:
        """Return number of moves made."""
        return self._moves

    def place_number(self, row: int, col: int, num: int) -> bool:
        """Place a number on the board. Returns True if valid move."""
        # Validate input
        if not (0 <= row < 9 and 0 <= col < 9):
            self._state = "failed"
            return False

        if not (1 <= num <= 9):
            self._state = "failed"
            return False

        # Can't modify initial cells
        if self._initial_board[row][col] != 0:
            self._state = "failed"
            return False
        if self._board[row][col] != 0:
            self._state = "failed"
            return False
        # Check if placement is valid
        if not _is_valid_placement(self._board, row, col, num):
            self._state = "failed"
            return False

        # Place number
        self._board[row][col] = num
        self._moves += 1
        self._update_state()
        return True

    def _update_state(self) -> None:
        """Update game state based on current board."""
        # Check if puzzle is complete
        if all(self._board[r][c] != 0 for r in range(9) for c in range(9)):
            # Verify solution is correct
            if self._board == self._solution:
                self._state = "success"
            else:
                self._state = "failed"
        else:
            self._state = "ongoing"

    def pretty(self, colors: bool = True) -> str:
        """Pretty print the Sudoku board."""
        RESET = "\x1b[0m"
        INITIAL = "\x1b[38;5;45m"   # Cyan for initial numbers
        PLACED = "\x1b[38;5;196m"    # Red for placed numbers
        EMPTY = "\x1b[38;5;239m"     # Gray for empty cells

        lines = []
        lines.append("┌───────┬───────┬───────┐")

        for row in range(9):
            row_str = "│ "
            for col in range(9):
                num = self._board[row][col]

                if colors:
                    if num == 0:
                        row_str += f"{EMPTY}.{RESET}"
                    elif self._initial_board[row][col] != 0:
                        row_str += f"{INITIAL}{num}{RESET}"
                    else:
                        row_str += f"{PLACED}{num}{RESET}"
                else:
                    row_str += str(num) if num != 0 else "."

                if col % 3 == 2:
                    row_str += " │ "
                else:
                    row_str += " "

            lines.append(row_str.rstrip())

            if row == 8:
                lines.append("└───────┴───────┴───────┘")
            elif row % 3 == 2:
                lines.append("├───────┼───────┼───────┤")

        return "\n".join(lines)


def _execute_strategy(strategy: Callable, game: SudokuGame):
    """Execute a strategy function on a Sudoku game."""
    assert callable(strategy)

    max_moves = 100
    valid_moves = 0  # Track successful moves

    while game.state() == "ongoing" and valid_moves < max_moves:
        try:
            board = game.board()
            initial = game.initial_board()
            result = strategy(board, initial)

            # Validate result format
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                # Invalid format = immediate fail, but return valid moves made
                return valid_moves, "failed"

            row, col, num = result

            # Validate types
            if not all(isinstance(x, int) for x in [row, col, num]):
                return valid_moves, "failed"

            # Try to place number
            success = game.place_number(row, col, num)

            if success:
                valid_moves += 1  # Count this valid move
            else:
                # Invalid move = game fails, but return valid_moves made so far
                return valid_moves, "failed"

        except Exception:
            return valid_moves, "failed"

    if valid_moves >= max_moves and game.state() == "ongoing":
        return valid_moves, "failed"

    return valid_moves, game.state()


# A 5-second limit keeps the session brief; upstream Unsloth uses 10 seconds.
@execute_with_time_limit(5)
def execute_strategy(strategy: Callable, game: SudokuGame):
    """Execute strategy with a 5-second time limit."""
    return _execute_strategy(strategy, game)


def extract_function(text):
    """Extract Python function from markdown code blocks."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board, initial):"):
            return fx
    return None


_ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text):
    """Length of text as displayed, ignoring the color codes SudokuGame.pretty() adds."""
    return len(_ANSI_COLOR.sub("", text))


def print_side_by_side(left, right, gap = "   "):
    """Print two multi-line blocks of text side by side, to save vertical space."""
    left_lines = left.split("\n")
    right_lines = right.split("\n")
    width = max(_visible_len(line) for line in left_lines)
    for i in range(max(len(left_lines), len(right_lines))):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        print(l + " " * (width - _visible_len(l)) + gap + r)


def play_strategy(function, difficulty = 40, seed = None):
    """Sandbox a model-written strategy and play one puzzle with it.

    Returns (outcome, valid_moves, game). outcome is "success" or "failed" when the
    strategy ran; "timeout" or "exception" when running it went wrong; "no_function"
    or "compile_error" when there was nothing to run (game is None then).
    """
    if function is None:
        return "no_function", 0, None
    ok, info = check_python_modules(function)
    if "error" in info:
        return "no_function", 0, None

    try:
        strategy = create_locked_down_function(function)
    except Exception:
        return "compile_error", 0, None

    game = SudokuGame(difficulty = difficulty, seed = seed)
    try:
        valid_moves, game_state = execute_strategy(strategy, game)
    except TimeoutError:
        return "timeout", 0, game
    except Exception:
        return "exception", 0, game

    if valid_moves == difficulty:
        game_state = "success"
    return game_state, valid_moves, game


def check_rewards(solved_reward, reward_per_valid_move, no_valid_moves_reward,
                  timeout_reward, exception_reward, difficulty = 40):
    """Check the scores chosen for Reward 3 before training with them.

    Raises ValueError with a hint when a value is missing or breaks one of the
    design rules; otherwise prints what a few example attempts would score.
    """
    rewards = {
        "solved_reward": solved_reward,
        "reward_per_valid_move": reward_per_valid_move,
        "no_valid_moves_reward": no_valid_moves_reward,
        "timeout_reward": timeout_reward,
        "exception_reward": exception_reward,
    }
    missing = [name for name, value in rewards.items() if not isinstance(value, (int, float))]
    if missing:
        raise ValueError(f"Fill in a number for: {', '.join(missing)}")

    problems = []
    if reward_per_valid_move <= 0:
        problems.append("reward_per_valid_move should be positive, so a strategy that makes more "
                        "legal moves scores higher.")
    best_partial = (difficulty - 1) * reward_per_valid_move
    if solved_reward <= best_partial:
        problems.append(f"solved_reward should beat the best failed attempt ({difficulty - 1} legal "
                        f"moves = {best_partial:g}); otherwise finishing the puzzle is not worth it.")
    for name in ("no_valid_moves_reward", "timeout_reward", "exception_reward"):
        if rewards[name] >= 0:
            problems.append(f"{name} should be negative, so a strategy that never makes a legal "
                            "move ranks below one that makes at least one.")
    if problems:
        raise ValueError("Reward design needs another look:\n- " + "\n- ".join(problems))

    print("Reward design looks good. Example scores:")
    for label, score in [
        ("Solved the puzzle", solved_reward),
        (f"{difficulty - 1} legal moves, then an illegal one", best_partial),
        ("5 legal moves, then an illegal one", 5 * reward_per_valid_move),
        ("Illegal first move", no_valid_moves_reward),
        ("Timed out", timeout_reward),
        ("Raised an exception", exception_reward),
    ]:
        print(f"  {label:<36} {score:g}")


def play_response(response, difficulty = 30, seed = 42):
    """Extract the strategy from a model response, check it for imports, and play one puzzle."""
    function = extract_function(response)
    if function is None:
        print("No `strategy` function found in the output (prose instead of code), nothing to play.")
        return
    if not check_python_modules(function)[0]:
        print("Extracted function uses disallowed imports, nothing to play.")
        print(function)
        return

    outcome, valid_moves, game = play_strategy(function, difficulty, seed)
    print(f"\nValid moves: {valid_moves}, Final state: {outcome}\n")
    if game is None:
        print(function)
    else:
        print_side_by_side(function, game.pretty())


COMPLETIONS_LOG = "outputs/completions_log.jsonl"
_completion_index = 0


def log_completion(function, valid_moves, game_state, reward, game = None):
    """Append a scored completion to COMPLETIONS_LOG and, if a game was played, print it.

    Every completion goes to the JSONL file; on the console, one in five strategies is
    printed in full and the rest are cut to 200 characters.
    """
    global _completion_index
    os.makedirs(os.path.dirname(COMPLETIONS_LOG), exist_ok = True)
    with open(COMPLETIONS_LOG, "a") as f:
        f.write(json.dumps({
            "index": _completion_index,
            "function": function,
            "valid_moves": valid_moves,
            "game_state": game_state,
            "reward": reward,
        }) + "\n")

    if game is not None:
        print(f"\n Valid moves: {valid_moves}, Final state: {game_state}, Reward: {reward}")
        display_function = function if _completion_index % 5 == 0 else (
            function[:200] + "..." if len(function) > 200 else function
        )
        print_side_by_side(display_function, game.pretty())
    _completion_index += 1

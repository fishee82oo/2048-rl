"""A standalone 2048 environment; observations are flattened log2 boards."""
import numpy as np


def merge_line(line: np.ndarray) -> tuple[np.ndarray, int]:
    """Compress toward the start and merge each tile at most once."""
    tiles = [int(value) for value in line if value]
    merged, reward, index = [], 0, 0
    while index < len(tiles):
        value = tiles[index]
        if index + 1 < len(tiles) and value == tiles[index + 1]:
            value *= 2
            reward += value
            index += 1
        merged.append(value)
        index += 1
    return np.array(merged + [0] * (4 - len(merged)), dtype=np.int64), reward


class Game2048:
    ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()
        return self.get_state()

    def _spawn_tile(self) -> None:
        empty = np.argwhere(self.board == 0)
        if len(empty):
            row, col = empty[self.rng.integers(len(empty))]
            self.board[row, col] = 2 if self.rng.random() < 0.9 else 4

    def get_state(self) -> np.ndarray:
        state = np.zeros((4, 4), dtype=np.float32)
        occupied = self.board > 0
        state[occupied] = np.log2(self.board[occupied])
        return state.flatten()

    def simulate_move(self, action: int) -> tuple[np.ndarray, int]:
        """Return the post-slide board and reward, without spawning or mutation."""
        if action not in range(4):
            raise ValueError("Action must be 0 (up), 1 (down), 2 (left), or 3 (right)")
        lines = self.board.T if action in (0, 1) else self.board
        reverse = action in (1, 3)
        result = np.zeros_like(lines)
        reward = 0
        for index, line in enumerate(lines):
            merged, gained = merge_line(line[::-1] if reverse else line)
            result[index] = merged[::-1] if reverse else merged
            reward += gained
        return (result.T.copy() if action in (0, 1) else result), reward

    def get_valid_actions(self) -> list[int]:
        return [action for action in range(4)
                if not np.array_equal(self.simulate_move(action)[0], self.board)]

    def is_game_over(self) -> bool:
        return not self.get_valid_actions()

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        moved, reward = self.simulate_move(action)
        changed = not np.array_equal(moved, self.board)
        if changed:
            self.board = moved
            self.score += reward
            self._spawn_tile()
        return self.get_state(), reward, self.is_game_over(), {
            "score": self.score, "max_tile": int(self.board.max()), "moved": changed,
        }

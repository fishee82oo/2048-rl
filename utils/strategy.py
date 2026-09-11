"""Interpretable top-right snake heuristic; no game rules live here."""
import numpy as np

SNAKE_PATH = [
    (0, 3), (0, 2), (0, 1), (0, 0),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 3), (2, 2), (2, 1), (2, 0),
    (3, 0), (3, 1), (3, 2), (3, 3),
]
SNAKE_WEIGHTS = np.arange(16, 0, -1, dtype=float)
CORNER_WEIGHT = 4.0
SNAKE_WEIGHT = 2.0
EMPTY_WEIGHT = 0.5
STRATEGY_SCALE = 8.0
CORNER_BREAK_PENALTY = -16.0
CORNER_CAPTURE_BONUS = 8.0


def corner_occupied(board: np.ndarray) -> bool:
    return bool(board.max() > 0 and board[0, 3] == board.max())


def snake_quality(board: np.ndarray) -> float:
    """0..1: equally weighted positional quality and descending-pair quality."""
    logs = np.zeros((4, 4), dtype=float)
    occupied = board > 0
    logs[occupied] = np.log2(board[occupied])
    values = np.array([logs[row, col] for row, col in SNAKE_PATH])
    if not values.any():
        return 0.0
    ideal = np.dot(SNAKE_WEIGHTS, np.sort(values)[::-1])
    positional = np.dot(SNAKE_WEIGHTS, values) / ideal
    # Only upward jumps are violations; trailing zeros cost nothing.
    pair_mass = np.maximum(values[:-1], values[1:]).sum()
    violations = np.maximum(values[1:] - values[:-1], 0).sum()
    monotonicity = 1.0 - violations / pair_mass if pair_mass else 1.0
    return float((positional + monotonicity) / 2)


def strategy_score(board: np.ndarray) -> float:
    corner = float(np.log2(board.max())) if corner_occupied(board) else 0.0
    empty = float(np.count_nonzero(board == 0)) / 16
    return CORNER_WEIGHT * corner + SNAKE_WEIGHT * snake_quality(board) + EMPTY_WEIGHT * empty


def strategy_reward(old_board: np.ndarray, new_board: np.ndarray) -> float:
    """Scaled score change plus an explicit corner transition bonus/penalty."""
    reward = STRATEGY_SCALE * (strategy_score(new_board) - strategy_score(old_board))
    before, after = corner_occupied(old_board), corner_occupied(new_board)
    if before and not after:
        reward += CORNER_BREAK_PENALTY
    elif after and not before:
        reward += CORNER_CAPTURE_BONUS
    return float(reward)


def filter_actions(env, valid_actions: list[int]) -> list[int]:
    """Preserve an anchored maximum if possible, including a merge in place."""
    if not corner_occupied(env.board):
        return list(valid_actions)
    maximum = env.board.max()
    preserving = []
    for action in valid_actions:
        board, _, changed = env.simulate_action(action)
        if changed and corner_occupied(board) and board[0, 3] >= maximum:
            preserving.append(action)
    return preserving or list(valid_actions)

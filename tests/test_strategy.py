import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.strategy import (CORNER_BREAK_PENALTY, CORNER_CAPTURE_BONUS, EMPTY_WEIGHT,
                            STRATEGY_SCALE, corner_occupied, filter_actions,
                            snake_quality, strategy_reward, strategy_score)

IDEAL = np.array([[128, 256, 512, 1024], [64, 32, 16, 8],
                  [0, 0, 2, 4], [0, 0, 0, 0]])
SCRAMBLED = np.array([[1024, 2, 64, 256], [8, 128, 4, 32],
                      [512, 0, 16, 0], [0, 0, 0, 0]])


class StrategyTests(unittest.TestCase):
    def test_ideal_beats_same_tiles_scrambled(self):
        np.testing.assert_array_equal(np.sort(IDEAL.ravel()), np.sort(SCRAMBLED.ravel()))
        self.assertGreater(strategy_score(IDEAL), strategy_score(SCRAMBLED) + 20)
        self.assertAlmostEqual(snake_quality(IDEAL), 1)
        self.assertLess(snake_quality(SCRAMBLED), 1)

    def test_snake_order_with_corner_unchanged(self):
        board = IDEAL.copy()
        board[1] = board[1, ::-1]
        self.assertTrue(corner_occupied(board))
        self.assertGreater(snake_quality(IDEAL), snake_quality(board))
        self.assertGreater(strategy_score(IDEAL), strategy_score(board))

    def test_empty_cell_bonus(self):
        board = IDEAL.copy()
        board[2, 2] = 0
        self.assertAlmostEqual(strategy_score(board) - strategy_score(IDEAL), EMPTY_WEIGHT / 16)

    def test_corner_transitions_and_unchanged_board(self):
        broken = IDEAL.copy()
        broken[0, 0], broken[0, 3] = broken[0, 3], broken[0, 0]
        delta = STRATEGY_SCALE * (strategy_score(broken) - strategy_score(IDEAL))
        self.assertAlmostEqual(strategy_reward(IDEAL, broken), delta + CORNER_BREAK_PENALTY)
        self.assertLess(strategy_reward(IDEAL, broken), 0)
        self.assertAlmostEqual(strategy_reward(broken, IDEAL), -delta + CORNER_CAPTURE_BONUS)
        self.assertGreater(strategy_reward(broken, IDEAL), 0)
        self.assertEqual(strategy_reward(IDEAL, IDEAL), 0)
        self.assertFalse(corner_occupied(np.zeros((4, 4), dtype=int)))

    def test_filter_and_simulation_do_not_mutate(self):
        env = Game2048(42)
        env.board[0] = [2, 4, 0, 16]
        board, score = env.board.copy(), env.score
        rng = copy.deepcopy(env.rng.bit_generator.state)
        valid = env.get_valid_actions()
        self.assertEqual(filter_actions(env, valid), [3])
        for action in valid:
            moved, _, changed = env.simulate_action(action)
            self.assertTrue(changed)
            self.assertEqual(moved.sum(), board.sum())
        np.testing.assert_array_equal(board, env.board)
        self.assertEqual(env.score, score)
        self.assertEqual(env.rng.bit_generator.state, rng)

    def test_fallback_unanchored_and_terminal(self):
        env = Game2048()
        env.board[0] = [2, 4, 8, 16]
        self.assertEqual(filter_actions(env, env.get_valid_actions()), [1])
        env.board = np.fliplr(env.board).copy()
        self.assertEqual(filter_actions(env, env.get_valid_actions()), env.get_valid_actions())
        env.board[:] = [[2, 4, 2, 4], [4, 2, 4, 2]] * 2
        self.assertEqual(filter_actions(env, env.get_valid_actions()), [])

    def test_corner_merge_with_tied_maxima(self):
        env = Game2048()
        env.board[0] = [0, 0, 16, 16]
        self.assertEqual(filter_actions(env, env.get_valid_actions()), [3])
        board, reward, _ = env.simulate_action(3)
        self.assertEqual(board[0, 3], 32)
        self.assertEqual(reward, 32)

    def test_exploration_and_exploitation(self):
        env = Game2048()
        env.board[0] = [2, 4, 0, 16]
        agent = DQNAgent(strategy=True)
        for training in (True, False):
            for _ in range(10):
                self.assertEqual(agent.select_action(env.get_state(), env.get_valid_actions(),
                                                    training=training, env=env), 3)
        agent.strategy = False
        self.assertIn(agent.select_action(env.get_state(), env.get_valid_actions()), env.get_valid_actions())

    def test_checkpoint_metadata_and_legacy(self):
        agent = DQNAgent(strategy=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            agent.save(path)
            restored = DQNAgent()
            restored.load(path)
            self.assertTrue(restored.strategy)
            torch.save(agent.online_network.state_dict(), path)
            restored.load(path)
            self.assertFalse(restored.strategy)

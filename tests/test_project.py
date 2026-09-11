import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from agents.baselines import GreedyAgent
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048, merge_line
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer
from utils.seeding import set_seed


class EnvironmentTests(unittest.TestCase):
    def test_merges(self):
        for line, expected, reward in (
            ([2, 2, 0, 0], [4, 0, 0, 0], 4),
            ([2, 2, 2, 2], [4, 4, 0, 0], 8),
            ([4, 4, 8, 8], [8, 16, 0, 0], 24),
            ([2, 2, 4, 0], [4, 4, 0, 0], 4),
            ([0, 2, 0, 2], [4, 0, 0, 0], 4),
        ):
            with self.subTest(line=line):
                actual, gained = merge_line(np.array(line))
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(gained, reward)

    def test_all_directions(self):
        env = Game2048()
        for action in range(4):
            env.board.fill(0)
            if action in (0, 1):
                env.board[:, 0] = [2, 2, 4, 0]
            else:
                env.board[0] = [2, 2, 4, 0]
            result, reward = env.simulate_move(action)
            line = result[:, 0] if action in (0, 1) else result[0]
            np.testing.assert_array_equal(line, [0, 0, 4, 4] if action in (1, 3) else [4, 4, 0, 0])
            self.assertEqual(reward, 4)

    def test_spawning_and_state(self):
        env = Game2048()
        self.assertEqual(env.reset().shape, (16,))
        self.assertEqual(np.count_nonzero(env.board), 2)
        env.board.fill(0)
        env.board[0, :2] = [2, 2]
        before = env.board.copy()
        env.step(0)
        np.testing.assert_array_equal(env.board, before)
        slid, reward = env.simulate_move(2)
        state, gained, done, info = env.step(2)
        delta = env.board - slid
        self.assertEqual(np.count_nonzero(delta), 1)
        self.assertIn(delta[delta != 0].item(), (2, 4))
        self.assertEqual(gained, reward)
        self.assertEqual(info['score'], 4)
        self.assertEqual(state[0], 2)
        self.assertFalse(done)

    def test_game_over(self):
        env = Game2048()
        env.board[:] = [[2, 4, 2, 4], [4, 2, 4, 2]] * 2
        self.assertTrue(env.is_game_over())
        self.assertTrue(env.step(0)[2])
        env.board[0, 0] = 4
        self.assertFalse(env.is_game_over())

    def test_greedy_does_not_mutate_or_consume_spawn_rng(self):
        env, reference = Game2048(12), Game2048(12)
        env.reset()
        reference.reset()
        before = env.board.copy()
        action = GreedyAgent().select_action(env)
        np.testing.assert_array_equal(env.board, before)
        self.assertEqual(env.simulate_move(action)[1], max(env.simulate_move(a)[1] for a in env.get_valid_actions()))
        env.step(action)
        reference.step(action)
        np.testing.assert_array_equal(env.board, reference.board)


class LearningTests(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_network_shape(self):
        self.assertEqual(tuple(DQN()(torch.zeros(7, 16)).shape), (7, 4))

    def test_replay_copies_and_capacity(self):
        replay = ReplayBuffer(2)
        state = np.zeros(16, dtype=np.float32)
        replay.push(state, 0, 4, state, False)
        state[:] = 10
        self.assertEqual(replay.sample(1)[0].sum(), 0)
        for _ in range(3):
            replay.push(state, 0, 0, state, False)
        self.assertEqual(len(replay), 2)

    def test_masking_and_round_trip(self):
        agent = DQNAgent()
        state = np.arange(16, dtype=np.float32)
        for training in (True, False):
            self.assertEqual(agent.select_action(state, [2], training), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            agent.save(path)
            loaded = DQNAgent()
            loaded.load(path)
            torch.testing.assert_close(agent.online_network(torch.tensor(state)),
                                       loaded.online_network(torch.tensor(state)))

    def test_terminal_targets_and_target_sync(self):
        agent = DQNAgent(batch_size=2, target_update_frequency=1)
        for parameter in agent.online_network.parameters():
            parameter.data.zero_()
        for parameter in agent.target_network.parameters():
            parameter.data.zero_()
        agent.target_network.layers[-1].bias.data.fill_(100)
        state = np.zeros(16, dtype=np.float32)
        for _ in range(2):
            agent.store_transition(state, 0, 4, state, True)
        self.assertAlmostEqual(agent.train_step(), 3.5)
        for online, target in zip(agent.online_network.parameters(), agent.target_network.parameters()):
            torch.testing.assert_close(online, target)

    def test_nonterminal_standard_dqn_target(self):
        agent = DQNAgent(batch_size=1, gamma=0.5)
        for network in (agent.online_network, agent.target_network):
            for parameter in network.parameters():
                parameter.data.zero_()
        agent.target_network.layers[-1].bias.data.copy_(torch.tensor([1., 2., 3., 10.]))
        state = np.zeros(16, dtype=np.float32)
        agent.store_transition(state, 0, 4, state, False)
        self.assertAlmostEqual(agent.train_step(), 8.5)


if __name__ == "__main__":
    unittest.main()

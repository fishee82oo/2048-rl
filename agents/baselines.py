"""Valid-action random and immediate-reward greedy policies."""
import random
from env.game_2048 import Game2048


class RandomAgent:
    def select_action(self, env: Game2048) -> int:
        return random.choice(env.get_valid_actions())


class GreedyAgent:
    def select_action(self, env: Game2048) -> int:
        rewards = {action: env.simulate_action(action)[1]
                   for action in env.get_valid_actions()}
        best = max(rewards.values())
        return random.choice([action for action, reward in rewards.items() if reward == best])

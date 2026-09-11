"""Standard DQN with replay, a target network, and masked behavior actions."""
from pathlib import Path
from typing import Optional, Union
import random
import numpy as np
import torch
from torch import nn
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, gamma=0.99, learning_rate=1e-3, batch_size=128,
                 replay_buffer_size=100_000, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay_steps=100_000,
                 target_update_frequency=1000):
        self.online_network = DQN()
        self.target_network = DQN()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma, self.batch_size = gamma, batch_size
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_frequency = target_update_frequency
        self.environment_steps = 0
        self.training_steps = 0
        self.update_target_network()

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, valid_actions: list[int],
                      training: bool = True) -> int:
        if not valid_actions:
            raise ValueError("Cannot select an action on a terminal board")
        epsilon = self.epsilon if training else 0.0
        if training:
            self.environment_steps += 1
        if training and random.random() < epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            q_values = self.online_network(torch.as_tensor(state, dtype=torch.float32))
            masked = torch.full_like(q_values, -torch.inf)
            masked[valid_actions] = q_values[valid_actions]
            return int(masked.argmax().item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.bool)
        predictions = self.online_network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Standard DQN: target network both selects and values the maximum.
            next_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * torch.where(dones, 0.0, next_values)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network()
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_network.state_dict(), path)

    def load(self, path: Union[str, Path]) -> None:
        self.online_network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.update_target_network()

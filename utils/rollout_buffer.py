"""On-policy samples grouped by environment, including final-observation values."""
import numpy as np
import torch


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Arrays [time, ...]; termination kills bootstrap, either boundary kills trace.

    next_values must refer to the pre-reset observation. The last rollout sample
    bootstraps even when neither boundary flag is set.
    """
    rewards, values, next_values = [np.asarray(x, dtype=np.float32)
                                  for x in (rewards, values, next_values)]
    terminated, truncated = np.asarray(terminated, bool), np.asarray(truncated, bool)
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * gae_lambda * (~(terminated[t] | truncated[t])) * carry
        advantages[t] = carry
    return advantages, advantages + values


class RolloutBuffer:
    def __init__(self, num_envs):
        self.trajectories = [[] for _ in range(num_envs)]

    def add(self, env_id, **sample):
        # Store copies detached from both the environment and autograd.
        self.trajectories[env_id].append({k: np.array(v, copy=True) for k, v in sample.items()})

    def tensors(self, gamma, gae_lambda, device):
        combined = {}
        for trajectory in self.trajectories:
            if not trajectory:
                continue
            data = {k: np.stack([s[k] for s in trajectory]) for k in trajectory[0]}
            data['advantages'], data['returns'] = compute_gae(
                data['reward'], data['old_value'], data['next_value'],
                data['terminated'], data['truncated'], gamma, gae_lambda)
            for key, value in data.items():
                combined.setdefault(key, []).append(value)
        return {k: torch.as_tensor(np.concatenate(v), device=device).detach()
                for k, v in combined.items()}

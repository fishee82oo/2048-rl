"""Direct PPO-Clip. No replay, epsilon exploration, or target network."""
from pathlib import Path
import numpy as np
import torch
from models.actor_critic import ActorCritic
from utils.ppo_support import action_mask


def clipped_policy_loss(ratio, advantage, clip_coef):
    return -torch.minimum(ratio * advantage,
                          ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantage).mean()


class PPOAgent:
    def __init__(self, config, device='cpu'):
        self.config = dict(config)
        self.device = torch.device(device)
        self.model = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.strategy = config['corner_filter']

    @torch.no_grad()
    def act(self, states, masks, deterministic=False):
        distribution, values = self.model.distribution(
            torch.as_tensor(states, dtype=torch.float32, device=self.device),
            torch.as_tensor(masks, dtype=torch.bool, device=self.device))
        actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return tuple(x.cpu().numpy() for x in (actions, distribution.log_prob(actions), values))

    def select_action(self, state, valid_actions=None, training=False, env=None, deterministic=True):
        return int(self.act(state, action_mask(env, self.strategy), deterministic)[0])

    def update(self, data):
        cfg = self.config
        advantages = data['advantages'].float()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = len(advantages)
        metrics = []
        with torch.no_grad():
            dist, _ = self.model.distribution(data['state'].float(), data['mask'])
            initial_error = (torch.exp(dist.log_prob(data['action'].long()) - data['old_log_prob']) - 1).abs().max().item()
        stopped = False
        stopping_kl = 0.0
        for epoch in range(cfg['update_epochs']):
            for indices in torch.randperm(n, device=self.device).split(cfg['minibatch_size']):
                dist, value = self.model.distribution(data['state'][indices].float(), data['mask'][indices])
                logratio = dist.log_prob(data['action'][indices].long()) - data['old_log_prob'][indices]
                ratio = logratio.exp()
                kl = ((ratio - 1) - logratio).mean()
                if cfg['target_kl'] is not None and kl.item() > cfg['target_kl']:
                    stopping_kl = kl.item()
                    stopped = True
                    break
                policy = clipped_policy_loss(ratio, advantages[indices], cfg['clip_coef'])
                value_loss = 0.5 * (value - data['returns'][indices]).square().mean()
                entropy = dist.entropy().mean()
                loss = policy + cfg['vf_coef'] * value_loss - cfg['ent_coef'] * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg['max_grad_norm'])
                if not torch.isfinite(loss) or not torch.isfinite(grad):
                    raise FloatingPointError('Non-finite PPO loss/gradient')
                self.optimizer.step()
                metrics.append([x.item() for x in (policy, value_loss, entropy, kl,
                               ((ratio - 1).abs() > cfg['clip_coef']).float().mean(), grad)])
            if stopped:
                break
        keys = ['policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'gradient_norm']
        result = dict(zip(keys, np.mean(metrics, axis=0).tolist())) if metrics else dict.fromkeys(keys, 0.)
        target, prediction = data['returns'], data['old_value']
        variance = target.var(unbiased=False)
        result.update(explained_variance=float(1 - (target - prediction).var(unbiased=False) / variance) if variance > 0 else 0.,
                      initial_ratio_max_error=initial_error, kl_early_stop=stopped,
                      optimizer_updates=len(metrics), stopping_kl=stopping_kl)
        return result

    def save(self, path, **state):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(algorithm='ppo', format_version=1, model=self.model.state_dict(),
                        optimizer=self.optimizer.state_dict(), config=self.config,
                        preprocessing='flatten16_log2_empty0', action_order=['UP','DOWN','LEFT','RIGHT'],
                        **state), path)

    @classmethod
    def load(cls, path, device='cpu'):
        # Full resume checkpoints contain Python/NumPy RNG objects; load only trusted files.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('algorithm') != 'ppo':
            raise ValueError('Expected a PPO checkpoint')
        agent = cls(checkpoint['config'], device)
        agent.model.load_state_dict(checkpoint['model'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        return agent, checkpoint

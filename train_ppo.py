"""Reproducible synchronous multi-environment PPO; checkpoints at update boundaries."""
import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import torch
from agents.ppo_agent import PPOAgent
from env.game_2048 import Game2048
from utils.evaluation import evaluate_policy
from utils.ppo_support import action_mask, choose_device, reward_components, rng_state, restore_rng
from utils.rollout_buffer import RolloutBuffer
from utils.seeding import set_seed

ROOT = Path(__file__).resolve().parent


class Collector:
    def __init__(self, config):
        self.config = config
        self.envs = [Game2048(config['seed'] + i) for i in range(config['num_envs'])]
        for env in self.envs:
            env.reset()
        self.totals = [dict(raw_reward=0., shaping_reward=0., training_reward=0., length=0) for _ in self.envs]
        self.episodes = 0

    def state_dict(self):
        return dict(envs=[dict(board=e.board.copy(), score=e.score, rng=copy.deepcopy(e.rng.bit_generator.state))
                          for e in self.envs], totals=copy.deepcopy(self.totals), episodes=self.episodes)

    def load_state_dict(self, state):
        for env, saved in zip(self.envs, state['envs']):
            env.board, env.score = saved['board'].copy(), saved['score']
            env.rng.bit_generator.state = saved['rng']
        self.totals, self.episodes = copy.deepcopy(state['totals']), state['episodes']

    def collect(self, agent, budget):
        buffer = RolloutBuffer(len(self.envs))
        completed = []
        reward_sums = np.zeros(3)
        used = 0
        while used < budget:
            active = self.envs[:min(len(self.envs), budget-used)]
            states = np.stack([e.get_state() for e in active])
            masks = np.stack([action_mask(e, agent.strategy) for e in active])
            actions, logprobs, values = agent.act(states, masks)
            transitions = []
            for i, env in enumerate(active):
                before = env.board.copy()
                next_state, raw, terminated, info = env.step(int(actions[i]))
                raw, shaping, reward = reward_components(raw, before, env.board, terminated, self.config)
                total = self.totals[i]
                for key, value in zip(('raw_reward','shaping_reward','training_reward'), (raw,shaping,reward)):
                    total[key] += value
                total['length'] += 1
                limit = self.config['max_episode_steps']
                truncated = bool(limit and total['length'] >= limit and not terminated)
                transitions.append((next_state, reward, terminated, truncated))
                reward_sums += (raw,shaping,reward)
                if terminated or truncated:
                    self.episodes += 1
                    completed.append(dict(episode=self.episodes, env_id=i, **total,
                                          score=info['score'], max_tile=info['max_tile'],
                                          terminated=terminated, truncated=truncated))
                    self.totals[i] = dict(raw_reward=0.,shaping_reward=0.,training_reward=0.,length=0)
            # Values of final observations are obtained BEFORE any reset, without action sampling.
            with torch.no_grad():
                next_values = agent.model(torch.as_tensor(np.stack([t[0] for t in transitions]),
                                         dtype=torch.float32, device=agent.device))[1].cpu().numpy()
            for i, (next_state,reward,terminated,truncated) in enumerate(transitions):
                buffer.add(i, state=states[i], mask=masks[i], action=actions[i], old_log_prob=logprobs[i],
                           old_value=values[i], reward=reward, next_value=next_values[i],
                           terminated=terminated, truncated=truncated)
                if terminated or truncated:
                    active[i].reset()
            used += len(active)
        return buffer.tensors(self.config['gamma'], self.config['gae_lambda'], agent.device), completed, reward_sums


def train(config, output, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(config['seed'])
    device = choose_device(config['device'])
    agent = PPOAgent(config, device)
    collector = Collector(config)
    steps, updates, best, elapsed_before = 0, 0, -float('inf'), 0.
    if resume:
        agent, saved = PPOAgent.load(resume, device)
        # Only budget and device can change; preserve objective and collection schedule.
        for key in config:
            if key not in ('total_steps','device') and config[key] != saved['config'][key]:
                raise ValueError(f'Resume config mismatch: {key}')
        agent.config = dict(config)
        collector.load_state_dict(saved['collector'])
        steps, updates, best = saved['steps'], saved['updates'], saved['best_validation_score']
        elapsed_before = saved['training_seconds']
        restore_rng(saved['rng'])
        previous_best = Path(resume).parent / 'best.pt'
        if previous_best.is_file():
            (output/'checkpoints').mkdir()
            shutil.copy2(previous_best, output/'checkpoints/best.pt')
    if config['total_steps'] <= steps:
        raise ValueError('Total step target must exceed saved steps')
    metadata = dict(config=config, device=device, torch=torch.__version__, numpy=np.__version__,
                    python=platform.python_version(), platform=platform.platform(),
                    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                    resume=str(resume) if resume else None)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    start = time.perf_counter()
    next_validation = (steps // config['validation_interval'] + 1) * config['validation_interval']
    while steps < config['total_steps']:
        budget = min(config['num_envs'] * config['rollout_steps'],
                     config['total_steps'] - steps, next_validation - steps)
        tick = time.perf_counter()
        data, episodes, rewards = collector.collect(agent, budget)
        metrics = agent.update(data)
        steps += budget
        updates += 1
        duration = time.perf_counter()-tick
        metrics.update(steps=steps, rollout=updates, samples=budget, rollout_seconds=duration,
                       samples_per_second=budget/duration, raw_reward=float(rewards[0]),
                       shaping_reward=float(rewards[1]), training_reward=float(rewards[2]))
        with (output/'updates.jsonl').open('a') as f:
            f.write(json.dumps(metrics)+'\n')
        with (output/'episodes.jsonl').open('a') as f:
            for row in episodes:
                f.write(json.dumps(dict(steps=steps, **row))+'\n')
        improved = False
        if steps >= next_validation or steps == config['total_steps']:
            summary, games = evaluate_policy(agent, config['validation_seeds'], deterministic=True)
            improved = summary['average_score'] > best
            best = max(best, summary['average_score'])
            with (output/'validation.jsonl').open('a') as f:
                f.write(json.dumps(dict(steps=steps, **summary, per_game=games))+'\n')
            next_validation = (steps // config['validation_interval']+1)*config['validation_interval']
        state = dict(steps=steps, updates=updates, best_validation_score=best,
                     training_seconds=elapsed_before+time.perf_counter()-start,
                     collector=collector.state_dict(), rng=rng_state(), metadata=metadata,
                     resume_scope='exact at saved update boundary on same software/device; changed rollout partition is a different run')
        agent.save(output/'checkpoints/last.pt', **state)
        if improved:
            agent.save(output/'checkpoints/best.pt', **state)
        print(f'steps={steps} fps={budget/duration:.1f} val_best={best:.1f} kl={metrics["approx_kl"]:.5f}', flush=True)
    return agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/ppo_formal.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    for name in ('total_steps','seed','num_envs','rollout_steps','minibatch_size','update_epochs','validation_interval','max_episode_steps'):
        p.add_argument('--'+name.replace('_','-'),type=int)
    for name in ('reward_scale','shaping_coef','learning_rate','target_kl'):
        p.add_argument('--'+name.replace('_','-'),type=float)
    p.add_argument('--reward-mode',choices=['raw','potential','legacy'])
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'])
    p.add_argument('--corner-filter',action=argparse.BooleanOptionalAction,default=None)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    for key in config:
        value=getattr(args,key,None)
        if value is not None:
            config[key]=value
    for key in ('total_steps','num_envs','rollout_steps','minibatch_size','update_epochs','validation_interval'):
        if config[key]<1:
            p.error(f'{key} must be positive')
    if config['reward_scale']<=0 or not config['validation_seeds']:
        p.error('Positive reward scale and nonempty validation seeds required')
    train(config,args.output_dir,args.resume)


if __name__=='__main__':
    main()

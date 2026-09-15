"""One seed protocol for validation, test, and playback policy loading."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.ppo_support import isolated_rng
from utils.strategy import corner_occupied, snake_quality


def load_policy(path, algorithm='auto', device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('algorithm') not in (None, 'ppo', 'dqn'):
        raise ValueError('Unsupported checkpoint algorithm tag')
    if checkpoint.get('algorithm') == 'ppo':
        detected = 'ppo'
    elif checkpoint.get('algorithm') == 'dqn' or 'model_state_dict' in checkpoint or 'layers.0.weight' in checkpoint:
        detected = 'dqn'
    else:
        raise ValueError('Unknown checkpoint format; refusing to guess algorithm')
    if algorithm != 'auto' and algorithm != detected:
        raise ValueError(f'Requested {algorithm}, checkpoint is {detected}')
    if detected == 'ppo':
        agent, metadata = PPOAgent.load(path, device)
    else:
        agent = DQNAgent()
        agent.load(path)
        metadata = checkpoint if 'model_state_dict' in checkpoint else {}
    metadata = dict(metadata)
    # Selection may choose an earlier checkpoint; keep its step separate from
    # the completed run's interaction budget and elapsed training time.
    last_path = Path(path).parent / 'last.pt'
    if metadata.get('algorithm') in ('ppo', 'dqn') and last_path.is_file():
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        if (last.get('algorithm') == metadata['algorithm'] and
                last.get('config', {}).get('seed') == metadata.get('config', {}).get('seed')):
            metadata['run_steps'] = last.get('steps')
            metadata['run_training_seconds'] = last.get('training_seconds')
    metadata['sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return agent, metadata


def policy_action(agent, env, deterministic=True):
    if isinstance(agent, PPOAgent):
        return agent.select_action(env.get_state(), env=env, deterministic=deterministic)
    if isinstance(agent, DQNAgent):
        return agent.select_action(env.get_state(), env.get_valid_actions(), training=False, env=env)
    return agent.select_action(env)


def evaluate_policy(agent, seeds, deterministic=True):
    rows = []
    for seed in seeds:
        # Each game's policy randomness is independent of previous game lengths.
        with isolated_rng(int(seed) + 1_000_000_000):
            env = Game2048(int(seed))
            env.reset()
            done, moves, corner, snake = False, 0, 0, 0.
            while not done:
                _, _, done, info = env.step(policy_action(agent, env, deterministic))
                moves += 1
                corner += corner_occupied(env.board)
                snake += snake_quality(env.board)
            rows.append(dict(seed=int(seed), score=info['score'], max_tile=info['max_tile'],
                             moves=moves, corner_occupancy=100 * corner / moves, snake_quality=snake / moves))
    scores = np.array([r['score'] for r in rows])
    tiles = np.array([r['max_tile'] for r in rows])
    summary = dict(games=len(rows), average_score=float(scores.mean()), median_score=float(np.median(scores)),
                   score_std=float(scores.std(ddof=1)) if len(rows)>1 else 0.,
                   score_q25=float(np.quantile(scores,.25)), score_q75=float(np.quantile(scores,.75)),
                   average_moves=float(np.mean([r['moves'] for r in rows])),
                   corner_occupancy=float(np.mean([r['corner_occupancy'] for r in rows])),
                   snake_quality=float(np.mean([r['snake_quality'] for r in rows])),
                   max_tile_distribution=json.dumps({int(t):int((tiles==t).sum()) for t in np.unique(tiles)}))
    summary.update({f'reach_{t}_pct':float(100*(tiles>=t).mean()) for t in (128,256,512,1024,2048)})
    return summary, rows

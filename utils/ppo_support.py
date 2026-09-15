"""Independent reward controls and reproducible random-state handling."""
import contextlib
import random
import numpy as np
import torch
from utils.strategy import filter_actions, strategy_reward, strategy_score


def action_mask(env, corner_filter=False):
    legal = env.get_valid_actions()
    actions = filter_actions(env, legal) if corner_filter else legal
    mask = np.zeros(4, dtype=bool)
    mask[actions] = True
    return mask


def reward_components(raw, before, after, terminated, config):
    mode = config['reward_mode']
    if mode == 'raw':
        shaping = 0.0
    elif mode == 'legacy':
        shaping = strategy_reward(before, after)
    elif mode == 'potential':
        potential_next = 0.0 if terminated else strategy_score(after)
        shaping = config['shaping_coef'] * (config['gamma'] * potential_next - strategy_score(before))
    else:
        raise ValueError(f'Unknown reward mode: {mode}')
    return float(raw), float(shaping), float((raw + shaping) * config['reward_scale'])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    if state['mps'] is not None:
        torch.mps.set_rng_state(state['mps'].cpu())


@contextlib.contextmanager
def isolated_rng(seed):
    saved = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng(saved)


def choose_device(request):
    if request == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if request == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    if request == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    return request

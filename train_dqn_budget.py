"""Original DQN update/behavior/rewards, with a separate equal-step experiment harness.

No Bellman-target correction. Validation selection replaces historical single-game
selection only in this new harness. No DQN training resumption is promised.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.evaluation import evaluate_policy
from utils.seeding import set_seed
from utils.strategy import strategy_reward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--total-steps',type=int,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--strategy',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--validation-interval',type=int,default=100000)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/ppo_formal.json')
    a=p.parse_args()
    if a.total_steps<1 or a.validation_interval<1:
        p.error('Positive budgets required')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'checkpoints').mkdir()
    set_seed(a.seed)
    config=dict(seed=a.seed, strategy=a.strategy, total_steps=a.total_steps,
                validation_seeds=json.loads(a.config.read_text())['validation_seeds'])
    (a.output_dir/'config.json').write_text(json.dumps(config,indent=2))
    agent,env=DQNAgent(strategy=a.strategy),Game2048(a.seed)
    state=env.reset()
    best=-float('inf')
    start=time.perf_counter()
    for step in range(1,a.total_steps+1):
        action=agent.select_action(state,env.get_valid_actions(),env=env)
        before=env.board.copy()
        next_state,raw,done,info=env.step(action)
        shaping=strategy_reward(before,env.board) if a.strategy else 0.
        agent.store_transition(state,action,raw+shaping,next_state,done)
        agent.train_step()
        state=env.reset() if done else next_state
        if step%a.validation_interval==0 or step==a.total_steps:
            summary,rows=evaluate_policy(agent,config['validation_seeds'])
            improved=summary['average_score']>best
            best=max(best,summary['average_score'])
            with (a.output_dir/'validation.jsonl').open('a') as f:
                f.write(json.dumps(dict(steps=step,**summary,per_game=rows))+'\n')
            saved=dict(algorithm='dqn',model_state_dict=agent.online_network.state_dict(),strategy=a.strategy,
                       config=config,steps=step,training_seconds=time.perf_counter()-start,
                       baseline='original_strategy_dqn' if a.strategy else 'original_vanilla_dqn',
                       resume_scope='evaluation only; no replay/optimizer resume')
            torch.save(saved,a.output_dir/'checkpoints/last.pt')
            if improved:
                torch.save(saved,a.output_dir/'checkpoints/best.pt')
            print(f'steps={step} elapsed={time.perf_counter()-start:.2f}s val={summary["average_score"]:.1f}',flush=True)


if __name__=='__main__':
    main()

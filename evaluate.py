"""Evaluate checkpoints with shared held-out seeds, separated by training run."""
import argparse
import json
from pathlib import Path
import pandas as pd
from agents.baselines import RandomAgent, GreedyAgent
from utils.evaluation import load_policy, evaluate_policy
from utils.seeding import set_seed

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games',type=int,default=100)
    p.add_argument('--seed',type=int,default=200000)
    p.add_argument('--seeds-file',type=Path,help='JSON list of final test environment seeds')
    p.add_argument('--checkpoint',type=Path,action='append',help='Repeat for any number of PPO/DQN runs')
    p.add_argument('--compare-checkpoint',type=Path)
    p.add_argument('--algorithm',choices=['auto','ppo','dqn'],default='auto')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--strategy','--corner-filter',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--deterministic',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--baselines',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    seeds=json.loads(args.seeds_file.read_text()) if args.seeds_file else list(range(args.seed,args.seed+args.games))
    if not seeds or len(set(seeds))!=len(seeds):
        p.error('Need nonempty unique seeds')
    set_seed(args.seed)
    entries=[('Random',RandomAgent(),{},None),('Greedy',GreedyAgent(),{},None)] if args.baselines else []
    paths=list(args.checkpoint or [])
    if args.compare_checkpoint:
        paths.append(args.compare_checkpoint)
    for path in paths:
        agent,meta=load_policy(path,args.algorithm)
        entries.append((str(path),agent,meta,path))
    if not entries:
        p.error('No policies selected')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'seeds.json').write_text(json.dumps(seeds))
    summaries,all_rows=[],[]
    for name,agent,meta,path in entries:
        saved_filter=getattr(agent,'strategy',False)
        if path and set(seeds)&set(meta.get('config',{}).get('validation_seeds',[])):
            p.error('Test seeds overlap checkpoint validation seeds')
        settings=[saved_filter] if args.strategy is None else [args.strategy]
        if saved_filter and args.strategy is None:
            settings.append(False)
        for setting in settings:
            if path:
                agent.strategy=setting
            summary,rows=evaluate_policy(agent,seeds,args.deterministic)
            condition=dict(agent=name,checkpoint=str(path) if path else '',
                           checkpoint_sha256=meta.get('sha256',''),
                           training_seed=meta.get('config',{}).get('seed'),
                           selected_checkpoint_steps=meta.get('steps'),
                           training_steps=meta.get('run_steps',meta.get('steps')),
                           training_seconds=meta.get('run_training_seconds',meta.get('training_seconds')),
                           evaluation_mode=('deterministic' if args.deterministic else 'sampled') if meta.get('algorithm')=='ppo' else ('greedy_q' if path else 'random_ties'),
                           corner_filter=setting,evaluation_condition_changed=setting!=saved_filter)
            summaries.append(dict(**condition,**summary))
            all_rows.extend(dict(**condition,**r) for r in rows)
    frame=pd.DataFrame(summaries)
    frame.to_csv(args.output_dir/'evaluation.csv',index=False)
    pd.DataFrame(all_rows).to_csv(args.output_dir/'evaluation_games.csv',index=False)
    print(frame[['agent','corner_filter','average_score','median_score','score_std','reach_512_pct','reach_1024_pct','reach_2048_pct']].to_string(index=False))


if __name__=='__main__':
    main()

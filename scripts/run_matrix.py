"""Finite, sequential experiment matrix. Dry-run by default; --execute runs it."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--total-steps',type=int,default=1000000)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--test-games',type=int,default=100)
    p.add_argument('--validation-interval',type=int,default=100000)
    p.add_argument('--device',choices=['cpu','auto','cuda','mps'],default='cpu')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.total_steps<1 or a.test_games<1 or len(set(a.train_seeds))!=len(a.train_seeds):
        p.error('Positive budgets and unique training seeds required')
    commands=[]
    for seed in a.train_seeds:
        for mode in ('raw','potential'):
            for corner in (False,True):
                name=f'ppo_{mode}_corner{int(corner)}'
                run=a.output_dir/name/f'seed{seed}'
                commands.append([sys.executable,'train_ppo.py','--config','configs/ppo_formal.json',
                    '--total-steps',str(a.total_steps),'--seed',str(seed),'--reward-mode',mode,
                    '--corner-filter' if corner else '--no-corner-filter','--device',a.device,
                    '--validation-interval',str(a.validation_interval),'--output-dir',str(run)])
                commands.append([sys.executable,'evaluate.py','--checkpoint',str(run/'checkpoints/best.pt'),
                    '--games',str(a.test_games),'--seed','200000','--no-baselines','--output-dir',str(run/'test')])
        for strategy in (False,True):
            run=a.output_dir/('dqn_strategy' if strategy else 'dqn_vanilla')/f'seed{seed}'
            commands.append([sys.executable,'train_dqn_budget.py','--total-steps',str(a.total_steps),
                '--seed',str(seed),'--strategy' if strategy else '--no-strategy',
                '--validation-interval',str(a.validation_interval),'--output-dir',str(run)])
            commands.append([sys.executable,'evaluate.py','--checkpoint',str(run/'checkpoints/best.pt'),
                '--games',str(a.test_games),'--seed','200000','--no-baselines','--output-dir',str(run/'test')])
    commands.append([sys.executable,'evaluate.py','--games',str(a.test_games),'--seed','200000',
                     '--output-dir',str(a.output_dir/'baselines')])
    if a.execute:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        (a.output_dir/'commands.json').write_text(json.dumps(commands,indent=2))
        (a.output_dir/'protocol.json').write_text(json.dumps(vars(a),default=str,indent=2))
    for index,cmd in enumerate(commands):
        print(shlex.join(cmd),flush=True)
        if a.execute:
            with (a.output_dir/f'command_{index:02d}.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if a.execute:
        subprocess.run([sys.executable,'scripts/summarize_matrix.py',str(a.output_dir)],cwd=ROOT,check=True)


if __name__=='__main__':
    main()

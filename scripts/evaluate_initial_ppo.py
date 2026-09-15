"""Bounded diagnostic: distinguish short training results from initialization."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from agents.ppo_agent import PPOAgent
from utils.evaluation import evaluate_policy
from utils.seeding import set_seed


def main():
    output=Path(sys.argv[1])
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('configs/ppo_formal.json').read_text())
    summaries,rows=[],[]
    for seed in (42,43,44):
        set_seed(seed)
        config['seed']=seed
        summary,games=evaluate_policy(PPOAgent(config),range(200000,200100))
        summaries.append(dict(training_seed=seed,training_steps=0,**summary))
        rows.extend(dict(training_seed=seed,**g) for g in games)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'games.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':
    main()

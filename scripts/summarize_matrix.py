"""Summarize run-level means; never pool games across training seeds."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    frames=[]
    for file in sorted(a.directory.glob('*/seed*/test/evaluation.csv')):
        frame=pd.read_csv(file)
        frame['method']=file.parents[2].name
        frames.append(frame)
    runs=pd.concat(frames,ignore_index=True)
    runs.to_csv(a.directory/'per_training_seed.csv',index=False)
    metrics=['average_score','median_score','score_std','reach_512_pct','reach_1024_pct','reach_2048_pct',
             'training_seconds','training_steps']
    result=runs.groupby(['method','corner_filter','evaluation_condition_changed'])[metrics].agg(['mean','std','count'])
    result.columns=['_'.join(c) for c in result.columns]
    result.to_csv(a.directory/'across_training_seeds.csv')
    print(result[['average_score_mean','average_score_std','reach_512_pct_mean',
                  'reach_1024_pct_mean','reach_2048_pct_mean']].to_string())


if __name__=='__main__':
    main()

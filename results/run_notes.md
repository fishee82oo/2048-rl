# Verification run

These files record a short pipeline smoke test, not converged training.

Commands executed from the project directory:

```bash
python -m unittest discover -s tests -v
python train.py --episodes 10 --seed 42
python evaluate.py --games 100 --seed 10000
python play.py --delay 0 --seed 10000
```

- All 10 unit tests passed.
- All 10 training episodes finished: 1,384 environment actions and 1,257 optimizer updates.
- Mean training score: 1,387.6; highest training score: 2,388.
- The best checkpoint is from episode 3; the final checkpoint is from episode 10.
- Evaluation completed 100 games per agent with no exploration for DQN.
- Playback finished naturally: score 1,656, maximum tile 128, 170 moves.
- The score plot was visually inspected; all three plots were generated.
- Both checkpoint files were produced; model save/load output equality passed in tests.

| Agent | Mean score | Median score | Best score | Best tile |
| --- | ---: | ---: | ---: | ---: |
| Random | 1101.6 | 1082.0 | 2864 | 256 |
| Greedy | 1953.8 | 1534.0 | 5064 | 512 |
| DQN | 1139.6 | 1114.0 | 2940 | 256 |

The smoke-test DQN did not outperform greedy. These short-run observations do
not establish learning quality, convergence, or statistically reliable differences.
Run the default 5,000 episodes for a longer experiment. Existing output paths
are overwritten by subsequent runs; this note describes the bundled initial run.

Runtime: Python 3.12.14, CPU (arm64),
NumPy 2.5.3, PyTorch 2.14.0,
matplotlib 3.11.1, pandas 3.0.5.

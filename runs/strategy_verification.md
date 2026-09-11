# Strategy implementation verification

Modified the existing project in place. Original root checkpoints and results,
including the user's 5,000-episode training history, were not overwritten.

Verified on Python 3.9.6 with the project's existing virtual environment:

- 19 unit tests passed, covering the original game/DQN and 9 new strategy tests.
- Both vanilla and strategy training completed 10 episodes with seed 42.
- Both checkpoints were evaluated over 100 held-out games each (seeds 10000–10099).
- Random and greedy baselines also completed 100 games each.
- Strategy playback completed naturally: score 1160, maximum tile 128, 128 moves.
- Reward accounting passed: score equals summed game rewards, and training reward
  equals game reward plus strategy reward. Vanilla strategy rewards were all zero.
- Corner occupancy, snake quality, and empty-cell metrics passed range checks.
- Both new plots were generated and visually inspected.
- New checkpoint metadata round trips and old weight-only checkpoints passed tests.
- `git diff --check` passed.

Exact smoke commands (from the project directory):

```bash
python -m unittest discover -s tests -v
python train.py --episodes 10 --strategy --seed 42 --output-dir runs/strategy_smoke
python train.py --episodes 10 --no-strategy --seed 42 --output-dir runs/vanilla_smoke
python evaluate.py --games 100 --seed 10000 --checkpoint runs/vanilla_smoke/checkpoints/best_model.pt --compare-checkpoint runs/strategy_smoke/checkpoints/best_model.pt --output-dir runs/comparison_smoke
python play.py --delay 0 --seed 10000 --checkpoint runs/strategy_smoke/checkpoints/best_model.pt
```

| Agent | Mean game score | Corner occupancy (%) | Mean snake quality |
| --- | ---: | ---: | ---: |
| Random | 1101.64 | 3.14 | 0.688 |
| Greedy | 1953.84 | 2.02 | 0.665 |
| Vanilla DQN | 1139.60 | 3.56 | 0.669 |
| Strategy DQN | 1432.68 | 84.25 | 0.815 |

The strategy system scored higher than vanilla in this particular short-run
comparison, but lower than greedy. Only one training seed and ten episodes per
variant were used. This does not establish a robust improvement or convergence,
and it does not isolate reward shaping from action filtering. Multiple training
seeds, longer runs, and unfiltered evaluation are needed for stronger conclusions.

## Source changes

- `utils/strategy.py` (new): constants, log-based snake quality, board potential,
  scaled reward changes, corner bonuses/penalties, and simulated action filtering.
- `env/game_2048.py`: added a side-effect-free `simulate_action` adapter that
  reuses existing move logic and reports whether the board changed.
- `agents/baselines.py`: greedy now reuses the simulation adapter; policy unchanged.
- `agents/dqn_agent.py`: optional filter in both action-selection modes; saves
  strategy metadata and loads old/new checkpoints. Bellman update unchanged.
- `train.py`: strategy flags, run directories, shaped replay reward, separate
  reward totals, and per-episode strategy metrics.
- `evaluate.py`: per-agent strategy metrics, checkpoint comparison, output
  directory, and optional filter override. Baselines stay unfiltered.
- `play.py`: checkpoint-driven filtering and optional override; displays game score.
- `utils/plotting.py`: added corner occupancy and snake plots, clarified training
  reward label, and retained support for old history files without strategy columns.
- `tests/test_strategy.py` (new): scoring, corner transitions, empty-cell bonus,
  filtering/fallback, simulation purity, exploration/exploitation, and checkpoint tests.
- `README.md`: formula, defaults, scoring/filter definitions, commands, metrics,
  legacy checkpoint behavior, and tradeoffs.
- `.gitignore`: excludes checkpoints in nested experiment directories.

Generated artifacts are under `runs/strategy_smoke`, `runs/vanilla_smoke`, and
`runs/comparison_smoke`. Each training run contains best/final checkpoints, a
training CSV, and five plots. The comparison contains aggregate/per-game CSVs.

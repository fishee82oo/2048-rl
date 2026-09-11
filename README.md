# Deep Reinforcement Learning for 2048

A minimal, readable PyTorch project that trains a standard Deep Q-Network to play
2048 and compares it with random and immediate-reward greedy baselines. The game
and learning algorithm are implemented directly, without external RL libraries.

## RL formulation

- **State:** the 4×4 board transformed with `log2(tile)`, empty cells set to zero,
  and flattened to 16 float32 values. The raw board remains a NumPy integer array.
- **Actions:** `0 = up`, `1 = down`, `2 = left`, `3 = right`.
- **Reward:** the score gained by merging tiles on that move, without scaling or shaping.
- **Goal:** maximize long-term discounted cumulative reward.

Each tile merges at most once per move. A changed board receives exactly one new
tile (90% 2, 10% 4); unchanged moves receive no new tile. Games end when no valid
move exists, and continue after reaching 2048. `reset()` returns the observation;
`step(action)` returns `(next_state, reward, done, info)` with score and maximum tile.

## Model and learning

```text
16 → Linear(16,128) → ReLU → Linear(128,128) → ReLU → Linear(128,4)
```

The outputs are Q-values with no softmax. Epsilon-greedy behavior samples only
valid moves during exploration and masks invalid Q-values during exploitation.
Evaluation and playback call `training=False`, giving effective epsilon zero.

The replay buffer stores `(state, action, reward, next_state, done)`. Each update
samples a minibatch and minimizes Huber loss against the standard DQN target:

```text
terminal:      target = reward
non-terminal:  target = reward + gamma * max_a Q_target(next_state, a)
```

The maximum in the learning target uses all four target-network outputs, as in
the requested formula; only behavior action selection is masked. This keeps the
five-field replay format simple, but untrained invalid-action values may affect
bootstrapping. This is ordinary DQN, not Double DQN.

| Setting | Default |
| --- | --- |
| Discount gamma | 0.99 |
| Adam learning rate | 0.001 |
| Batch size | 128 |
| Replay capacity | 100,000 |
| Epsilon | Linear decay from 1.0 to 0.05 over 100,000 training actions |
| Target synchronization | Every 1,000 optimizer updates |
| Gradient norm clipping | 10 |
| Training episodes | 5,000 |
| Evaluation games per agent | 100 |
| Seed | 42 |

Training performs one optimizer update per move once 128 transitions are
available. Huber loss and gradient clipping help control large raw merge rewards.
The greedy baseline simulates each valid slide without spawning tiles or changing
the environment, chooses the highest immediate merge reward, and breaks ties randomly.

## Project structure

```text
2048-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py                  # Training loop and best/final checkpoints
├── evaluate.py               # Comparison table and CSV exports
├── play.py                   # Terminal playback
├── env/
│   ├── __init__.py
│   └── game_2048.py           # Board mechanics and observations
├── agents/
│   ├── __init__.py
│   ├── baselines.py
│   └── dqn_agent.py          # Action selection and learning
├── models/
│   ├── __init__.py
│   └── dqn.py                # 16–128–128–4 MLP
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py
│   ├── plotting.py
│   └── seeding.py
├── tests/
│   ├── __init__.py
│   └── test_project.py
├── checkpoints/
│   ├── .gitkeep
│   ├── best_model.pt
│   └── final_model.pt
└── results/
    ├── run_notes.md
    ├── training_history.csv
    ├── evaluation.csv
    ├── evaluation_games.csv
    ├── score_curve.png
    ├── max_tile_curve.png
    └── reward_curve.png
```

## Installation

Use Python 3.9 or newer (including the macOS Python 3.9.6 environment).
Pip selects dependency releases compatible with your Python version. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

## Training

```bash
python train.py --episodes 5000 --seed 42
```

Progress prints every 100 episodes and at completion. The best checkpoint is
selected by single-episode training game score; the final checkpoint saves the
last policy. This selection is noisy and does not imply the best average policy.
Each run starts fresh and overwrites the standard result and checkpoint paths.
Outputs are resolved relative to the scripts, independent of the working directory.

## Evaluation

```bash
python evaluate.py --games 100 --seed 42
```

This loads `checkpoints/best_model.pt`, runs 100 complete games per agent, and
reports average and median score, average maximum tile, best score/tile, average
moves, and the percentage reaching at least 128, 256, 512, 1024, and 2048.
All agents use the same per-game environment seed sequence; different moves
still lead to different boards. Use a separate seed (e.g. `--seed 10000`) for
held-out evaluation. To evaluate the final policy instead:

```bash
python evaluate.py --checkpoint checkpoints/final_model.pt
```

## Watch the agent play

```bash
python play.py
python play.py --delay 0.05 --seed 10000
```

The board and score print after every move. Use `--delay 0` for immediate playback.
There is no GUI and no artificial move limit.

## Results and verification

Plots and per-episode metrics are in `results/`. Score and total reward curves
coincide because merge rewards sum to game score. `evaluation.csv` contains the
aggregate comparison, and `evaluation_games.csv` records every game. Policy
weights are in `checkpoints/`; binary checkpoints are ignored by Git by default.

The bundled checkpoints and plots come from a **10-episode smoke test**, not a
full training run. See `results/run_notes.md` for the actual commands and measured
results. They verify the pipeline and do not demonstrate convergence or strong
play. No claim is made that this small DQN outperforms greedy play or reaches 2048.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Tests cover merge examples, all four directions, spawning, invalid actions,
terminal detection, simulation side effects, network shape, replay copying,
action masking, Bellman targets, target synchronization, and checkpoint round trips.

The project uses CPU execution with one PyTorch thread and seeds Python, NumPy,
PyTorch, and the environment's local generator. Identical software/hardware and
seeds should reproduce runs; different versions or platforms can differ.
Checkpoints contain policy weights only, not replay, optimizer, or RNG state,
so they support evaluation and playback rather than exact training resumption.

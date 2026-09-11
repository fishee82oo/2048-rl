"""Compare random, greedy, and trained DQN policies over complete games."""
import argparse
from pathlib import Path
import pandas as pd
from agents.baselines import RandomAgent, GreedyAgent
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.seeding import set_seed

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_model.pt")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run python train.py first.")
    set_seed(args.seed)
    dqn = DQNAgent()
    dqn.load(args.checkpoint)
    summaries, all_games = [], []
    for name, agent in (("Random", RandomAgent()), ("Greedy", GreedyAgent()), ("DQN", dqn)):
        set_seed(args.seed)
        games = []
        for game in range(args.games):
            env = Game2048(args.seed + game)
            state, done, moves = env.reset(), False, 0
            while not done:
                action = (agent.select_action(state, env.get_valid_actions(), training=False)
                          if isinstance(agent, DQNAgent) else agent.select_action(env))
                state, _, done, info = env.step(action)
                moves += 1
            row = dict(agent=name, game=game + 1, seed=args.seed + game,
                       score=info["score"], max_tile=info["max_tile"], moves=moves)
            games.append(row)
            all_games.append(row)
        frame = pd.DataFrame(games)
        summary = dict(agent=name, games=args.games, seed=args.seed,
                       average_score=frame.score.mean(), median_score=frame.score.median(),
                       average_max_tile=frame.max_tile.mean(), best_score=frame.score.max(),
                       best_max_tile=frame.max_tile.max(), average_moves=frame.moves.mean())
        for tile in (128, 256, 512, 1024, 2048):
            summary[f"reach_{tile}_pct"] = 100 * (frame.max_tile >= tile).mean()
        summaries.append(summary)
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(results / "evaluation.csv", index=False)
    pd.DataFrame(all_games).to_csv(results / "evaluation_games.csv", index=False)
    print(comparison.iloc[:, [0, 3, 4, 5, 6, 7, 8]].to_string(index=False, float_format="%.1f"))
    print("\nTile reach rates (% of games with maximum tile at least the threshold):")
    print(comparison[["agent"] + [f"reach_{t}_pct" for t in (128, 256, 512, 1024, 2048)]]
          .to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()

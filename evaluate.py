"""Compare random, greedy, and trained DQN policies over complete games."""
import argparse
from pathlib import Path
import pandas as pd
from agents.baselines import RandomAgent, GreedyAgent
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.seeding import set_seed
from utils.strategy import corner_occupied, snake_quality

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_model.pt")
    parser.add_argument("--compare-checkpoint", type=Path,
                        help="Optional second DQN checkpoint for a side-by-side comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--strategy", action=argparse.BooleanOptionalAction, default=None,
                        help="Override filtering for all evaluated DQNs; default uses checkpoint metadata")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run python train.py first.")
    set_seed(args.seed)
    dqn = DQNAgent()
    dqn.load(args.checkpoint)
    agents = [("Random", RandomAgent()), ("Greedy", GreedyAgent()), ("DQN", dqn)]
    if args.compare_checkpoint:
        if not args.compare_checkpoint.is_file():
            parser.error(f"Checkpoint not found: {args.compare_checkpoint}")
        other = DQNAgent()
        other.load(args.compare_checkpoint)
        agents.append(("DQN_compare", other))
    summaries, all_games = [], []
    for name, agent in agents:
        is_dqn = isinstance(agent, DQNAgent)
        if is_dqn and args.strategy is not None:
            agent.strategy = args.strategy
        set_seed(args.seed)
        games = []
        for game in range(args.games):
            env = Game2048(args.seed + game)
            state, done, moves = env.reset(), False, 0
            corner_moves, snake_total = 0, 0.0
            while not done:
                action = (agent.select_action(state, env.get_valid_actions(), training=False, env=env)
                          if isinstance(agent, DQNAgent) else agent.select_action(env))
                state, _, done, info = env.step(action)
                moves += 1
                corner_moves += int(corner_occupied(env.board))
                snake_total += snake_quality(env.board)
            row = dict(agent=name, game=game + 1, seed=args.seed + game,
                       score=info["score"], max_tile=info["max_tile"], moves=moves,
                       corner_occupancy=100 * corner_moves / moves,
                       snake_score=snake_total / moves)
            games.append(row)
            all_games.append(row)
        frame = pd.DataFrame(games)
        summary = dict(agent=name, games=args.games, seed=args.seed,
                       average_score=frame.score.mean(), median_score=frame.score.median(),
                       average_max_tile=frame.max_tile.mean(), best_score=frame.score.max(),
                       best_max_tile=frame.max_tile.max(), average_moves=frame.moves.mean(),
                       corner_occupancy=frame.corner_occupancy.mean(),
                       average_snake_quality=frame.snake_score.mean(),
                       strategy_filter=agent.strategy if is_dqn else False,
                       checkpoint=str(args.compare_checkpoint if name == "DQN_compare"
                                      else args.checkpoint) if is_dqn else "")
        for tile in (128, 256, 512, 1024, 2048):
            summary[f"reach_{tile}_pct"] = 100 * (frame.max_tile >= tile).mean()
        summaries.append(summary)
    results = args.output_dir
    results.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(results / "evaluation.csv", index=False)
    pd.DataFrame(all_games).to_csv(results / "evaluation_games.csv", index=False)
    print(comparison[["agent", "average_score", "median_score", "average_max_tile",
                      "best_score", "best_max_tile", "average_moves", "corner_occupancy",
                      "average_snake_quality", "strategy_filter"]].to_string(
                          index=False, float_format="%.1f",
                          formatters={"average_snake_quality": "{:.3f}".format}))
    print("\nTile reach rates (% of games with maximum tile at least the threshold):")
    print(comparison[["agent"] + [f"reach_{t}_pct" for t in (128, 256, 512, 1024, 2048)]]
          .to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()

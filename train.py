"""Train one standard DQN, saving metrics, plots, and policy weights."""
import argparse
from pathlib import Path
import pandas as pd
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.plotting import plot_training
from utils.seeding import set_seed
from utils.strategy import strategy_reward, corner_occupied, snake_quality
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Run directory containing checkpoints/ and results/")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    set_seed(args.seed)
    env, agent = Game2048(args.seed), DQNAgent(strategy=args.strategy)
    results = args.output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    history, best_score = [], -1
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done, total_reward, moves = False, 0.0, 0
        total_strategy_reward = 0.0
        corner_moves, snake_total, empty_total = 0, 0.0, 0
        while not done:
            action = agent.select_action(state, env.get_valid_actions(), env=env)
            old_board = env.board.copy()
            next_state, game_reward, done, info = env.step(action)
            shaped_reward = strategy_reward(old_board, env.board) if args.strategy else 0.0
            training_reward = game_reward + shaped_reward
            agent.store_transition(state, action, training_reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += training_reward
            total_strategy_reward += shaped_reward
            corner_moves += int(corner_occupied(env.board))
            snake_total += snake_quality(env.board)
            empty_total += int(np.count_nonzero(env.board == 0))
            moves += 1
        history.append(dict(episode=episode, score=info["score"],
                            total_reward=total_reward, max_tile=info["max_tile"],
                            moves=moves, epsilon=agent.epsilon, seed=args.seed,
                            strategy=args.strategy, game_reward=info["score"],
                            strategy_reward=total_strategy_reward, training_reward=total_reward,
                            corner_occupancy=100 * corner_moves / moves,
                            snake_score=snake_total / moves, empty_cells=empty_total / moves))
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save(args.output_dir / "checkpoints/best_model.pt")
        if episode % 100 == 0 or episode == args.episodes:
            recent = pd.DataFrame(history[-100:])
            print(f"Episode {episode} | Avg Score: {recent.score.mean():.1f} | "
                  f"Avg Max Tile: {recent.max_tile.mean():.1f} | "
                  f"Corner Occupancy: {recent.corner_occupancy.mean():.1f}% | "
                  f"Avg Snake Score: {recent.snake_score.mean():.3f} | "
                  f"Empty Cells: {recent.empty_cells.mean():.1f} | Epsilon: {agent.epsilon:.3f}",
                  flush=True)
            pd.DataFrame(history).to_csv(results / "training_history.csv", index=False)
    agent.save(args.output_dir / "checkpoints/final_model.pt")
    plot_training(pd.DataFrame(history), results)
    print(f"Saved checkpoints and results under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

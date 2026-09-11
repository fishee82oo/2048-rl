"""Train one standard DQN, saving metrics, plots, and policy weights."""
import argparse
from pathlib import Path
import pandas as pd
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.plotting import plot_training
from utils.seeding import set_seed

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    set_seed(args.seed)
    env, agent = Game2048(args.seed), DQNAgent()
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    history, best_score = [], -1
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done, total_reward, moves = False, 0, 0
        while not done:
            action = agent.select_action(state, env.get_valid_actions())
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            moves += 1
        history.append(dict(episode=episode, score=info["score"],
                            total_reward=total_reward, max_tile=info["max_tile"],
                            moves=moves, epsilon=agent.epsilon, seed=args.seed))
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save(ROOT / "checkpoints/best_model.pt")
        if episode % 100 == 0 or episode == args.episodes:
            recent = pd.DataFrame(history[-100:])
            print(f"Episode {episode} | Avg Score: {recent.score.mean():.1f} | "
                  f"Avg Max Tile: {recent.max_tile.mean():.1f} | Epsilon: {agent.epsilon:.3f}",
                  flush=True)
            pd.DataFrame(history).to_csv(results / "training_history.csv", index=False)
    agent.save(ROOT / "checkpoints/final_model.pt")
    plot_training(pd.DataFrame(history), results)
    print(f"Saved checkpoints and results under {ROOT}")


if __name__ == "__main__":
    main()

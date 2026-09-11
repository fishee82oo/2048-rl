"""Watch a greedy DQN policy play one complete game in the terminal."""
import argparse
from pathlib import Path
import time
from agents.dqn_agent import DQNAgent
from env.game_2048 import Game2048
from utils.seeding import set_seed

ROOT = Path(__file__).resolve().parent


def display(env: Game2048, move: str) -> None:
    border = "+" + "+".join(["------"] * 4) + "+"
    print(border)
    for row in env.board:
        print("|" + "|".join(f"{int(value) if value else '':>5} " for value in row) + "|")
    print(border)
    print(f"Score: {env.score} | Move: {move}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_model.pt")
    parser.add_argument("--strategy", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the strategy filter setting saved in the checkpoint")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run python train.py first.")
    set_seed(args.seed)
    env, agent = Game2048(args.seed), DQNAgent()
    agent.load(args.checkpoint)
    if args.strategy is not None:
        agent.strategy = args.strategy
    state, done, moves = env.reset(), False, 0
    display(env, "START")
    while not done:
        action = agent.select_action(state, env.get_valid_actions(), training=False, env=env)
        state, _, done, info = env.step(action)
        moves += 1
        display(env, env.ACTION_NAMES[action])
        if args.delay:
            time.sleep(args.delay)
    print(f"Game over | Score: {info['score']} | Max tile: {info['max_tile']} | Moves: {moves}")


if __name__ == "__main__":
    main()

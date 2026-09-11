from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training(history: pd.DataFrame, output_dir: Path, window: int = 100) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for column, label, filename in (
        ("score", "Game score", "score_curve.png"),
        ("max_tile", "Maximum tile", "max_tile_curve.png"),
        ("total_reward", "Training reward", "reward_curve.png"),
        ("corner_occupancy", "Corner occupancy (%)", "corner_occupancy_curve.png"),
        ("snake_score", "Snake quality (0–1)", "snake_score_curve.png"),
    ):
        if column not in history:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.episode, history[column], alpha=0.35, label="Per episode")
        ax.plot(history.episode, history[column].rolling(window, min_periods=1).mean(),
                label=f"Moving average (up to {window} episodes)")
        ax.set(xlabel="Episode", ylabel=label, title=f"DQN training: {label.lower()}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

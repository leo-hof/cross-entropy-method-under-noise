"""Evaluation of a trained policy, and plots."""
import matplotlib

matplotlib.use("Agg")  # write figures to files, never open windows
import matplotlib.pyplot as plt
import numpy as np


def evaluate(agent, env, n_episodes=300, warmup_steps=100):
    """Run the agent's current policy for `n_episodes` episodes from the environment's default start.

    Rewards collected before step `warmup_steps` are discarded, so an episode that ends
    before then scores 0. This measures whether the policy keeps the pole up over the long run.
    """
    returns, lengths = [], []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total, step, done = 0.0, 0, False
        while not done:
            s, r, terminated, truncated, _ = env.step(agent.get_action(s))
            total += r
            step += 1
            if step < warmup_steps:
                total = 0.0
            done = terminated or truncated
        returns.append(float(total))
        lengths.append(step)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "returns": returns,
        "episode_lengths": lengths,
    }


def plot_training(history, title, path):
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Best elite score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_evaluation(stats, title, path):
    returns = stats["returns"]
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(returns, "o-", alpha=0.5)
    plt.axhline(stats["mean_return"], color="r", linestyle="--",
                label=f"Mean: {stats['mean_return']:.1f} ± {stats['std_return']:.1f}")
    plt.axhline(stats["median_return"], color="g", linestyle=":", label=f"Median: {stats['median_return']:.1f}")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(returns, bins=min(20, max(1, len(returns) // 2)), alpha=0.7)
    plt.title("Returns distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

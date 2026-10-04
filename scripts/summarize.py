"""Summarize the 5-seed grid (results/*_eval.json from scripts/run_grid.sh).

Writes summary/per_seed_returns.csv and figures/results_5seeds.png, and prints the
README results table: mean over seeds of the mean evaluation return, with a 95%
t-interval across seeds, and the paired difference to vanilla CEM (same seeds).

Usage: python scripts/summarize.py
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEEDS = range(5)
ENVS = [("reward", "Noisy rewards"), ("transition", "Noisy transitions"),
        ("obs", "Noisy observations"), ("start", "Random start (no noise)")]
VARIANTS = [("cem", 1, "Vanilla"), ("cem", 5, "5 rollouts"), ("baseline", 1, "Baseline"), ("advantage", 1, "Advantage")]

SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def load(agent, env, rollouts):
    returns = []
    for s in SEEDS:
        with open(os.path.join(ROOT, "results", f"{agent}_{env}_r{rollouts}_s{s}_eval.json")) as f:
            returns.append(json.load(f)["mean_return"])
    return np.array(returns)


def mean_ci(x):
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - half, x.mean() + half


data = {(env, label): load(agent, env, r) for env, _ in ENVS for agent, r, label in VARIANTS}

# 1. CSV of every run
os.makedirs(os.path.join(ROOT, "summary"), exist_ok=True)
with open(os.path.join(ROOT, "summary", "per_seed_returns.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["env", "variant", "seed", "mean_return"])
    for (env, label), x in data.items():
        for s, v in zip(SEEDS, x):
            w.writerow([env, label, s, round(float(v), 1)])

# 2. README table
print("| Noise | " + " | ".join(label for *_, label in VARIANTS) + " |")
print("|---" * (len(VARIANTS) + 1) + "|")
for env, title in ENVS:
    cells = []
    for *_, label in VARIANTS:
        m, lo, hi = mean_ci(data[env, label])
        cells.append(f"{m:.0f} [{lo:.0f}, {hi:.0f}]")
    print(f"| {title} | " + " | ".join(cells) + " |")

print("\nPaired difference vs vanilla (mean [95% CI], paired t-test p):")
for env, title in ENVS:
    base = data[env, "Vanilla"]
    for *_, label in VARIANTS[1:]:
        d = data[env, label] - base
        m, lo, hi = mean_ci(d)
        print(f"  {title:24s} {label:10s} {m:+5.0f} [{lo:+5.0f}, {hi:+5.0f}]  p={stats.ttest_rel(data[env, label], base).pvalue:.3f}")

# 3. Figure: one panel per environment, every seed as a dot, mean +/- 95% CI as a black tick
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
fig, axes = plt.subplots(1, len(ENVS), figsize=(13, 3.8), sharey=True, facecolor=SURFACE)
rng = np.random.default_rng(0)  # horizontal jitter only, so overlapping seeds stay visible
for ax, (env, title) in zip(axes, ENVS):
    ax.set_facecolor(SURFACE)
    for i, (*_, label) in enumerate(VARIANTS):
        x = data[env, label]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(x)), x, s=36, color=SERIES, alpha=0.85,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        m, lo, hi = mean_ci(x)
        ax.errorbar(i + 0.3, m, yerr=[[m - max(lo, 0)], [hi - m]], fmt="_", color=INK,
                    markersize=12, elinewidth=2, capsize=0, zorder=4)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks(range(len(VARIANTS)), [label for *_, label in VARIANTS], rotation=25, ha="right")
    ax.set_xlim(-0.5, len(VARIANTS) - 0.3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Mean evaluation return\n(one dot per training seed)")
axes[0].set_ylim(bottom=0)
fig.text(0.99, 0.01, "Black tick: mean over 5 seeds ± 95% CI", ha="right", va="bottom", fontsize=9, color=INK_2)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "results_5seeds.png"), dpi=150, facecolor=SURFACE)
print("\nwrote summary/per_seed_returns.csv and figures/results_5seeds.png")

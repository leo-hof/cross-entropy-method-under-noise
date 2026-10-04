"""Train and/or evaluate one CEM variant on one environment.

Example:
    python run.py --env transition --agent advantage --seed 0
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from agents import AGENTS
from envs import ENVS, make_env
from evaluate import evaluate, plot_evaluation, plot_training


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # the environments' noise and the CEM sampling use NumPy's global RNG
    torch.manual_seed(seed)  # initial policy parameters and the value network


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=ENVS, required=True,
                   help="plain | obs (noisy observations) | reward (noisy rewards) | "
                        "transition (noisy state transitions) | start (random initial state)")
    p.add_argument("--agent", choices=AGENTS, required=True, help="cem | baseline | advantage")
    p.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollouts", type=int, default=1, help="episodes per candidate when scoring")
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--pop", type=int, default=100, help="population size")
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--noise", type=float, default=None, help="override the environment's noise level")
    p.add_argument("--episodes", type=int, default=300, help="evaluation episodes")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    env = make_env(args.env, args.noise)
    env.reset(seed=args.seed)  # seeds gymnasium's own RNG (default initial states)

    agent = AGENTS[args.agent](env, n_pop=args.pop, elite_frac=args.elite_frac, n_rollouts=args.rollouts)

    tag = f"{args.agent}_{args.env}_r{args.rollouts}_s{args.seed}"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    checkpoint = os.path.join("checkpoints", f"{tag}.pt")
    title = f"{agent.name} on {args.env} ({args.rollouts} rollout(s), seed {args.seed})"

    if args.mode in ("train", "both"):
        history = agent.train(args.generations)
        agent.save(checkpoint)
        plot_training(history, f"Training: {title}", os.path.join("results", f"{tag}_training.png"))

    if args.mode in ("eval", "both"):
        agent.load(checkpoint)
        stats = evaluate(agent, env, n_episodes=args.episodes)
        print(f"\n{title}: mean {stats['mean_return']:.1f} ± {stats['std_return']:.1f}, "
              f"median {stats['median_return']:.1f}, mean length {stats['mean_episode_length']:.1f} steps")
        with open(os.path.join("results", f"{tag}_eval.json"), "w") as f:
            json.dump({"args": vars(args), **stats}, f)
        plot_evaluation(stats, title, os.path.join("results", f"{tag}_evaluation.png"))

    env.close()


if __name__ == "__main__":
    main()

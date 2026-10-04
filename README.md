# Cross-Entropy Method under noise: don't pick the lucky policies

The Cross-Entropy Method (CEM) is a simple, strong policy-search method in deterministic environments. It keeps the top 20% of policies ("elites") by return. **When returns are noisy, the elites are partly the policies that got lucky**, and CEM can fail to converge. This project measures when that happens and compares three fixes.

> The same failure mode appears whenever you select the best of many candidates using noisy scores: model selection on a small validation set, A/B tests, strategy backtests. Selecting on noisy estimates biases the winners upward.

## Setup

**Environment:** CartPole with a continuous action in [−1, 1] (scaled to a force of ±10 N), episodes of at most 500 steps, and a shaped per-step reward `1 + 10·|θ| + 0.3·|ẋ|`, so returns can exceed 500.

**Four kinds of noise** (zero-mean Gaussian):

| `--env` | What is noisy | Std. dev. | Real-world analogy |
|---|---|---|---|
| `reward` | Reward signal | 5.0 | Noisy outcome measurement |
| `transition` | Action, before it is applied | 1.0 | Mechanical disturbances |
| `obs` | State seen by the agent (true dynamics unchanged) | 0.3 | Imperfect sensors |
| `start` | Initial state only | – | Varying starting conditions |

**Policy and search:** linear policy `a = tanh(w·s + b)` (5 parameters). Each generation samples 100 parameter vectors from N(μ, σ²), scores them, and refits μ and σ to the top 20%. 50 generations. The policy kept at the end is the best-scoring candidate.

**Three ways to score a candidate**, compared with vanilla CEM (score = return of one episode):

1. **5 rollouts** (`--agent cem --rollouts 5`): average return over 5 episodes. This directly reduces variance, but costs 5× more simulation.
2. **Baseline** (`--agent baseline`): return minus an exponential moving average of previous returns.
3. **Advantage** (`--agent advantage`): mean advantage A(s,a) = G_t − V(s_t) along the episode, with V a small neural network trained on the returns seen so far. It rewards *good actions* rather than *lucky trajectories*.

**Evaluation:** each trained policy runs 300 episodes. Rewards from the first 100 steps are discarded, so an episode that fails early scores 0. Every configuration is trained with **5 seeds**.

## Results

Mean evaluation return over 5 training seeds, with a 95% confidence interval across seeds:

| Noise | Vanilla | 5 rollouts | Baseline | Advantage |
|---|---|---|---|---|
| **Noisy rewards** | 398 [131, 666] | 492 [483, 502] | 476 [376, 576] | **694 [608, 781]** |
| **Noisy transitions** | 201 [−3, 406] | 236 [27, 446] | 165 [−60, 390] | **369 [293, 445]** |
| Noisy observations | 367 [105, 629] | 441 [162, 719] | 446 [272, 620] | 476 [424, 529] |
| Random start (no noise) | **723 [663, 783]** | 672 [544, 800] | 656 [453, 858] | 513 [143, 883] |

![Mean evaluation return per training seed, for each variant and noise type](figures/results_5seeds.png)

**Findings**
- **Noisy rewards: the advantage score clearly wins.** It improves on vanilla by +296 (paired 95% CI [+10, +582]), and all 5 Advantage seeds (621–779) beat all 5 vanilla seeds (14–505). The learned V(s) acts as a stable reference that absorbs part of the reward noise.
- **Under noise, the fixes mainly make training reliable.** In each noisy environment, vanilla CEM had one training run that collapsed (mean returns of 14, 15 and 19): the search locked onto lucky policies early. Advantage's worst seed in the same environments was 285, which is why its confidence intervals are about 3–5× narrower.
- **Noisy transitions are hard for every variant.** No variant averages above 400. Advantage is the best and most consistent. 5 rollouts helps only slightly (+35 on average) at 5× the simulation cost.
- **The EMA baseline has no reliable effect.** This is expected: elites are selected by *rank*, and subtracting a baseline barely changes the ranking.
- **Without noise during the episode (random start), vanilla CEM is enough.** The fixes bring no gain. Advantage was the least stable here: 2 of 5 seeds collapsed.

## Limitations

- **5 seeds per configuration:** small differences are within noise (see the intervals). With 12 comparisons against vanilla, only the noisy-reward advantage result is convincing on its own.
- **Train/evaluation mismatch in start states (noisy environments):** vanilla CEM and Baseline train from random starts, Advantage from the default start, and evaluation uses the default start. In the `start` environment every phase uses random starts.
- One environment (CartPole), one noise level per type, and a 5-parameter linear policy.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# one run: train (50 generations) then evaluate (300 episodes), ~10–60 s on a laptop CPU
python run.py --env reward --agent advantage --seed 0
python run.py --env transition --agent cem --rollouts 5 --seed 0

# full grid (4 noise types x 4 variants x 5 seeds = 80 runs, ~13 min), then the table and figure above
bash scripts/run_grid.sh
python scripts/summarize.py
```

Main options: `--env {plain,obs,reward,transition,start}`, `--agent {cem,baseline,advantage}`, `--rollouts N`, `--seed S`, `--mode {train,eval,both}`, `--noise SD`. Run `python run.py -h` for all options.
Outputs: `checkpoints/<run>.pt`, and `results/<run>_eval.json` (all 300 returns) + training and evaluation plots.

## Repository structure

```
envs.py                       continuous CartPole and the 4 noisy variants
agents.py                     CEM, CEM + baseline, CEM + advantage
evaluate.py                   300-episode evaluation and plots
run.py                        command-line entry point (seeding, training, evaluation)
scripts/run_grid.sh           all 80 runs
scripts/summarize.py          results table, figure, summary/per_seed_returns.csv
scripts/make_gif.py           the GIF below (needs: pip install pygame pillow)
summary/per_seed_returns.csv  mean evaluation return of every run
figures/                      figure and GIF used in this README
```

## What it looks like

A typical episode with noisy transitions (each action is perturbed before it is applied). Vanilla CEM lets the cart drift off the track after 186 steps. Advantage CEM keeps the pole up until step 300 (the maximum is 500).

![Side-by-side CartPole episodes with noisy transitions: vanilla CEM fails at step 186, Advantage CEM at step 300](figures/noisy_transitions.gif)

*Illustration, not evidence. Episodes were chosen by a fixed rule: for each variant, take the training seed with the median result, run 300 new episodes, and show the one whose return is closest to the median. Played in real time. Generated by `scripts/make_gif.py`.*

## Context

Individual project for *IE540 Dynamic Programming and Reinforcement Learning* (KAIST, Spring 2025). The continuous CartPole environment and shaped reward come from the course's Homework 5 template. The research question, the noisy environments, the CEM variants and the experiments are my own. In 2026 the code was reorganized and documented for publication with [Claude Code](https://claude.com/claude-code), and the experiments were rerun with 5 seeds. The algorithms are unchanged.

Related work: [arXiv:1609.09449](https://arxiv.org/abs/1609.09449), [arXiv:2008.06389](https://arxiv.org/abs/2008.06389), [arXiv:2009.09043](https://arxiv.org/abs/2009.09043).

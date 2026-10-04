"""Cross-Entropy Method agents: vanilla, EMA baseline, and advantage-based scoring.

All agents search over the parameters of a linear policy a = tanh(w . s + b).
Each generation: sample a population of parameter vectors from N(mu, sigma^2),
score every candidate, keep the top `elite_frac` ("elites"), and refit mu and
sigma to the elites. The variants differ only in how candidates are scored.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

GAMMA = 0.99


class LinearPolicy(nn.Module):
    """Deterministic policy a = tanh(w . s + b), with flat get/set of parameters for CEM."""

    def __init__(self, dim_state):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_state, 1), nn.Tanh())

    def forward(self, s):
        return self.net(s)

    def get_params(self):
        return np.hstack([p.data.numpy().flatten() for p in self.parameters()]).copy()

    def set_params(self, params):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(params[i:i + n]).view(p.size()))
            i += n


class CEM:
    """Vanilla CEM: a candidate's score is its mean return over `n_rollouts` episodes."""

    name = "CEM"

    def __init__(self, env, n_pop=100, elite_frac=0.2, n_rollouts=1):
        self.env = env
        self.act_limit = env.action_space.high[0]
        self.gamma = GAMMA

        self.policy = LinearPolicy(env.observation_space.shape[0])
        self.param_size = self.policy.get_params().shape[0]
        self.mu = self.policy.get_params()
        self.sigma = np.ones(self.param_size)
        self.eps = 0.001  # floor on sigma so the search never collapses completely

        self.n_pop = n_pop
        self.elite_frac = elite_frac
        self.n_elites = int(n_pop * elite_frac)
        self.n_rollouts = n_rollouts
        self.population = np.empty((n_pop, self.param_size))
        self.scores = np.zeros(n_pop)

        self.best_params = None
        self.best_score = -np.inf

    def get_action(self, s):
        with torch.no_grad():
            return self.policy(torch.FloatTensor(s)).item() * self.act_limit

    def run_episode(self):
        """Run one episode with the current policy from a random "recoverable" start; return its return."""
        s, _ = self.env.reset(init_mode=1)
        total, done = 0.0, False
        while not done:
            s, r, terminated, truncated, _ = self.env.step(self.get_action(s))
            total += r
            done = terminated or truncated
        return total

    def sample_population(self):
        self.population = np.random.normal(self.mu, self.sigma, (self.n_pop, self.param_size))

    def score_population(self):
        for i in range(self.n_pop):
            self.policy.set_params(self.population[i])
            self.scores[i] = np.mean([self.run_episode() for _ in range(self.n_rollouts)])

    def update_distribution(self):
        """Refit mu and sigma to the elites. Returns (best, median, worst) elite score and the best candidate."""
        elite_idx = np.argsort(-self.scores)[:self.n_elites]
        elites = self.population[elite_idx]
        self.mu = elites.mean(axis=0)
        self.sigma = elites.std(axis=0) + self.eps
        self.policy.set_params(self.mu)

        elite_scores = self.scores[elite_idx]
        return elite_scores[0], np.median(elite_scores), elite_scores[-1], self.population[elite_idx[0]].copy()

    def train(self, n_gen=50):
        """Run `n_gen` generations; return the best elite score of each generation."""
        history = []
        for gen in range(n_gen):
            self.sample_population()
            self.score_population()
            top, med, low, gen_best = self.update_distribution()
            history.append(top)
            if top > self.best_score:
                self.best_score, self.best_params = top, gen_best
            print(f"[{self.name}] gen {gen + 1:3d}/{n_gen}  elite score max/med/min = {top:8.1f} / {med:8.1f} / {low:8.1f}")
        return history

    def save(self, path):
        """Save the best-scoring candidate seen during training."""
        current = self.policy.get_params()
        self.policy.set_params(self.best_params)
        torch.save(self.policy.state_dict(), path)
        self.policy.set_params(current)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))


class CEMBaseline(CEM):
    """Score = mean return minus an exponential moving average (EMA) of past returns."""

    name = "CEM_Baseline"

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.baseline = 0.0
        self.baseline_alpha = 0.1
        self.returns = np.zeros(self.n_pop)

    def score_population(self):
        for i in range(self.n_pop):
            self.policy.set_params(self.population[i])
            self.returns[i] = np.mean([self.run_episode() for _ in range(self.n_rollouts)])
            # NOTE: the baseline is updated after every candidate, so candidates in the same
            # generation are compared against different baselines (see CHANGES / README).
            self.scores[i] = self.returns[i] - self.baseline
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * self.returns[i]

    def update_distribution(self):
        z = (self.scores - self.scores.mean()) / (self.scores.std() + 1e-8)
        elite_idx = np.argsort(-z)[:self.n_elites]
        elites = self.population[elite_idx]
        self.mu = elites.mean(axis=0)
        self.sigma = elites.std(axis=0) + self.eps
        self.policy.set_params(self.mu)

        elite_returns = self.returns[elite_idx]
        return elite_returns[0], np.median(elite_returns), elite_returns[-1], self.population[elite_idx[0]].copy()


class CEMAdvantage(CEM):
    """Score = mean advantage A(s, a) = G_t - V(s_t) along the episode, with V a learned value network."""

    name = "CEM_Advantage"

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        dim_state = env.observation_space.shape[0]
        self.vnet = nn.Sequential(nn.Linear(dim_state, 64), nn.ReLU(), nn.Linear(64, 1))
        self.vnet_optimizer = Adam(self.vnet.parameters(), lr=1e-3)
        self.scores = np.zeros(self.n_pop, dtype=np.float32)

    def train_value_net(self, states, returns):
        """One gradient step of V(s) towards the observed discounted returns."""
        states = torch.FloatTensor(np.array(states))
        returns = torch.FloatTensor(np.array(returns)).unsqueeze(1)
        loss = F.mse_loss(self.vnet(states), returns)
        self.vnet_optimizer.zero_grad()
        loss.backward()
        self.vnet_optimizer.step()

    def compute_advantages(self, states, rewards):
        """Advantages from TD errors accumulated backwards (GAE with lambda = 1)."""
        with torch.no_grad():
            values = self.vnet(torch.FloatTensor(np.array(states))).squeeze().numpy()
        advantages = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            next_value = self.gamma * values[t + 1] if t + 1 < len(values) else 0.0
            running = rewards[t] + next_value - values[t] + self.gamma * running
            advantages[t] = running
        return advantages

    def rollout(self):
        """One episode; returns the visited states, rewards and discounted returns-to-go."""
        # NOTE: unlike CEM.run_episode, this uses the default start (init_mode=0), as in the original code.
        s, _ = self.env.reset()
        states, rewards = [], []
        done = False
        while not done:
            states.append(s)
            s, r, terminated, truncated, _ = self.env.step(self.get_action(s))
            rewards.append(r)
            done = terminated or truncated
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        return states, rewards, returns

    def score_population(self):
        all_states, all_returns = [], []
        for i in range(self.n_pop):
            self.policy.set_params(self.population[i])
            states, rewards = [], []
            for _ in range(self.n_rollouts):
                s, r, g = self.rollout()
                states += s
                rewards += r
                all_returns += g
            self.scores[i] = np.mean(self.compute_advantages(states, rewards))
            all_states += states
        self.train_value_net(all_states, all_returns)


AGENTS = {"cem": CEM, "baseline": CEMBaseline, "advantage": CEMAdvantage}

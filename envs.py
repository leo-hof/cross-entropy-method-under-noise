"""Continuous-action CartPole and four noisy variants.

The base environment (continuous action, shaped reward) is adapted from the
IE540 (KAIST, Spring 2025) Homework 5 template. The four noisy variants are
the subject of this project.
"""
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.spaces import Box

MAX_STEPS = 500

# Shaped per-step reward from the course template:
#   r = 1 + ANGLE_FACTOR * |theta| + SPEED_FACTOR * |x_dot| - FALL_PENALTY * terminated
# so returns can exceed MAX_STEPS.
ANGLE_FACTOR = 10
SPEED_FACTOR = 0.3
FALL_PENALTY = 0

# Default noise levels (standard deviations of zero-mean Gaussian noise),
# set to the values used for the reported experiments.
OBS_NOISE = 0.3
REWARD_NOISE = 5.0
TRANSITION_NOISE = 1.0


class ContCartPoleEnv(CartPoleEnv):
    """CartPole with a continuous action in [-1, 1], scaled to a force."""

    def __init__(self, render_mode=None, max_episode_steps=MAX_STEPS):
        super().__init__(render_mode=render_mode)
        self.max_episode_steps = max_episode_steps
        self.act_limit = 1.0
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.max_force = self.force_mag  # 10.0, same magnitude as the discrete version

    @staticmethod
    def shaped_reward(state, terminated):
        return 1 + ANGLE_FACTOR * abs(state[2]) + SPEED_FACTOR * abs(state[1]) - FALL_PENALTY * terminated

    def step(self, action):
        action = np.clip(action, -self.act_limit, self.act_limit)
        force = self.max_force * action

        # Same physics as CartPoleEnv.step, with a continuous force instead of +/- force_mag
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = self.shaped_reward(self.state, terminated)

        self._elapsed_steps += 1
        truncated = (not terminated) and (self._elapsed_steps >= self.max_episode_steps)
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, init_mode=0):
        """init_mode 0: gymnasium default (all state values in +/-0.05).
        init_mode 1: "recoverable" random start, cart position and pole angle drawn jointly.
        init_mode 2: fully random start."""
        s, info = super().reset(seed=seed)
        self._elapsed_steps = 0
        if init_mode == 0:
            return s, info
        if init_mode == 1:
            u = np.random.random()
            s[0] = (1 - u) * (-2.2) + u * (+2.2)
            s[2] = (1 - u) * (+0.2) + u * (-0.2)
        else:
            s[0] = np.random.uniform(-2.4, 2.4)
            s[1] = np.random.uniform(-3, 3)
            s[2] = np.random.uniform(-0.209, 0.209)
            s[3] = np.random.uniform(-4, 4)
        self.unwrapped.state = s
        return s, info


class NoisyObservationCartPoleEnv(ContCartPoleEnv):
    """The agent sees a noisy state; the true dynamics are unchanged (imperfect sensors)."""

    def __init__(self, render_mode=None, max_episode_steps=MAX_STEPS, noise_scale=OBS_NOISE):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.noise_scale = noise_scale

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        noisy_state = state + np.random.normal(0, self.noise_scale, size=state.shape)
        return noisy_state, reward, terminated, truncated, info


class NoisyRewardCartPoleEnv(ContCartPoleEnv):
    """The reward signal is noisy."""

    def __init__(self, render_mode=None, max_episode_steps=MAX_STEPS, noise_scale=REWARD_NOISE):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.noise_scale = noise_scale

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        return state, reward + np.random.normal(0, self.noise_scale), terminated, truncated, info


class NoisyTransitionCartPoleEnv(ContCartPoleEnv):
    """The action is perturbed before it is applied (mechanical disturbances)."""

    def __init__(self, render_mode=None, max_episode_steps=MAX_STEPS, noise_scale=TRANSITION_NOISE):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.noise_scale = noise_scale

    def step(self, action):
        return super().step(action + np.random.normal(0, self.noise_scale))


class RandomStartCartPoleEnv(ContCartPoleEnv):
    """Only the initial state is random; the episode itself is deterministic."""

    def reset(self, seed=None, init_mode=None):
        return super().reset(seed=seed, init_mode=1)


ENVS = {
    "plain": ContCartPoleEnv,
    "obs": NoisyObservationCartPoleEnv,
    "reward": NoisyRewardCartPoleEnv,
    "transition": NoisyTransitionCartPoleEnv,
    "start": RandomStartCartPoleEnv,
}


def make_env(name, noise_scale=None):
    """Build an environment by short name; noise_scale overrides the default noise level."""
    cls = ENVS[name]
    if noise_scale is not None and name in ("obs", "reward", "transition"):
        return cls(noise_scale=noise_scale)
    return cls()

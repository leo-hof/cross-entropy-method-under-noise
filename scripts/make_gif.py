"""Side-by-side GIF of a typical episode: vanilla CEM vs Advantage CEM on noisy transitions.

Selection rule (no cherry-picking): for each variant, use the training seed with the
median mean return (seed 1 for both, see summary/per_seed_returns.csv), run 300 fresh
evaluation episodes, and replay the episode whose return is closest to that policy's
median return.

Requires the checkpoints from scripts/run_grid.sh, plus: pip install pygame pillow
Usage: python scripts/make_gif.py
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render off-screen, no window
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agents import AGENTS
from envs import NoisyTransitionCartPoleEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")
PANELS = [("cem", 1, "Vanilla CEM"), ("advantage", 1, "Advantage CEM")]
N_EPISODES, WARMUP, FIRST_EP_SEED = 300, 100, 10_000
SCALE, FRAME_STEP, FPS = 0.5, 2, 25  # half-size frames, every 2nd step at 25 fps = real time


def run_episode(agent, env, ep_seed, render=False):
    """Same scoring as evaluate.evaluate (rewards before step 100 dropped), seeded per episode."""
    np.random.seed(ep_seed)
    s, _ = env.reset(seed=ep_seed)
    ret, step, frames, done = 0.0, 0, [], False
    while not done:
        if render:
            frames.append(env.render())
        s, r, terminated, truncated, _ = env.step(agent.get_action(s))
        ret = 0.0 if step < WARMUP else ret + r
        step += 1
        done = terminated or truncated
    return ret, step, terminated, frames


def typical_episode(agent_name, seed):
    env = NoisyTransitionCartPoleEnv(render_mode="rgb_array")
    agent = AGENTS[agent_name](env)
    agent.load(os.path.join(ROOT, "checkpoints", f"{agent_name}_transition_r1_s{seed}.pt"))
    returns = [run_episode(agent, env, FIRST_EP_SEED + i)[0] for i in range(N_EPISODES)]
    i = int(np.argmin(np.abs(np.array(returns) - np.median(returns))))
    ret, length, fell, frames = run_episode(agent, env, FIRST_EP_SEED + i, render=True)
    env.close()
    print(f"{agent_name}: median return {np.median(returns):.0f}, chosen episode return {ret:.0f}, "
          f"{length} steps, {'failed' if fell else 'survived'}")
    return frames, length, fell


def label_frame(frame, title, step, length, fell, font, small):
    img = Image.fromarray(frame).resize((int(frame.shape[1] * SCALE), int(frame.shape[0] * SCALE)))
    canvas = Image.new("RGB", (img.width, img.height + 28), "white")
    canvas.paste(img, (0, 28))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill="#0b0b0b", font=font)
    if fell and step >= length:
        d.text((img.width - 8, 8), f"failed at step {length}", fill="#e34948", font=small, anchor="ra")
    else:
        d.text((img.width - 8, 8), f"step {min(step, length)}", fill="#52514e", font=small, anchor="ra")
    return canvas


def main():
    episodes = [typical_episode(agent, seed=1) for agent, _, _ in PANELS]
    try:
        font, small = ImageFont.truetype("Arial.ttf", 15), ImageFont.truetype("Arial.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()

    n_steps = max(length for _, length, _ in episodes)
    gif = []
    for t in range(0, n_steps + 1, FRAME_STEP):
        panels = []
        for (frames, length, fell), (_, _, title) in zip(episodes, PANELS):
            panels.append(label_frame(frames[min(t, len(frames) - 1)], title, t, length, fell, font, small))
        row = Image.new("RGB", (sum(p.width for p in panels) + 8, panels[0].height), "#d6d5d0")
        x = 0
        for p in panels:
            row.paste(p, (x, 0))
            x += p.width + 8
        gif.append(row)
    gif += [gif[-1]] * FPS * 2  # hold the last frame for 2 s before looping

    out = os.path.join(ROOT, "figures", "noisy_transitions.gif")
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=1000 // FPS, loop=0, optimize=True)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(gif)} frames)")


if __name__ == "__main__":
    main()

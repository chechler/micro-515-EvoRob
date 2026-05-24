"""
Record a video of the best evolved robot navigating the hill terrain.

Usage:
    python record_hill_video.py --checkpoint results/body_brain_019/seed_0/975
    python record_hill_video.py --checkpoint results/body_brain_019/seed_0/975 --out hill.mp4
"""

import argparse
import os
import platform

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if platform.system() == "Darwin" else "osmesa"

import numpy as np
import evorob.world  # registers HillEnv-v0

from final_project_train import FinalWorld

MAX_STEPS = 1000
SEED = 0


def record_hill_video(checkpoint_dir: str, out_path: str, fps: int = 30) -> None:
    x_best_path = os.path.join(checkpoint_dir, "x_best.npy")
    if not os.path.isfile(x_best_path):
        raise FileNotFoundError(f"x_best.npy not found in {checkpoint_dir}")

    genotype = np.load(x_best_path, allow_pickle=True)
    print(f"Loaded genotype  shape={genotype.shape}  from {x_best_path}")

    world = FinalWorld(co_evolve_body=True)
    world.update_robot_xml(genotype)

    env = world.create_env(render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    world.controller.reset_controller(batch_size=1)
    obs, _ = env.reset(seed=SEED)

    frames = []
    total_reward = 0.0
    for step in range(MAX_STEPS):
        frames.append(env.render())
        action = world.controller.get_action(obs)
        if action.ndim > 1:
            action = action.squeeze(0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}  "
                  f"({'terminated' if terminated else 'truncated'})")
            break

    env.close()

    import imageio
    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"Video saved: {out_path}  ({len(frames)} frames @ {fps} fps)")
    print(f"Total episode reward: {total_reward:.2f}")
    x_pos = float(info.get("x_position", 0.0))
    z_pos = float(info.get("z_position", 0.0))
    print(f"Final position: x={x_pos:.2f}  z={z_pos:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record hill terrain video of best robot")
    parser.add_argument(
        "--checkpoint",
        default="results/body_brain_019/seed_0/975",
        help="Checkpoint directory containing x_best.npy",
    )
    parser.add_argument(
        "--out",
        default="hill_video.mp4",
        help="Output video path (default: hill_video.mp4)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    args = parser.parse_args()
    record_hill_video(args.checkpoint, args.out, args.fps)

#!/usr/bin/env python3
"""
Collect training trajectories in simulation.
Supports: random policy, scripted policy, or trained VLA policy.
Saves as LeRobot-compatible dataset.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import gym_hil
import numpy as np
import torch

# Headless rendering
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def scripted_pick_policy(obs, info, step, env):
    """
    Simple scripted pick-and-place policy for Franka Panda.
    Demonstrates the phases: approach → lower → grasp → lift → move → release.
    """
    action_dim = env.action_space.shape[0]
    action = np.zeros(action_dim)
    
    # Simple 4-phase policy based on step count
    # Phase timings (adjust based on env fps and physics)
    if step < 30:
        # Phase 1: Move toward cube (assumes cube is roughly centered)
        action[0] = 0.0    # x
        action[1] = 0.3    # y (forward)
        action[2] = -0.2   # z (down slightly)
        if action_dim > 3:
            action[3] = 1.0  # gripper open
    elif step < 60:
        # Phase 2: Lower to grasp
        action[2] = -0.5   # z (down)
        if action_dim > 3:
            action[3] = 1.0  # gripper open
    elif step < 80:
        # Phase 3: Close gripper
        if action_dim > 3:
            action[3] = -1.0  # gripper close
    elif step < 120:
        # Phase 4: Lift
        action[2] = 0.5    # z (up)
        if action_dim > 3:
            action[3] = -1.0  # gripper stay closed
    else:
        # Phase 5: Hold
        if action_dim > 3:
            action[3] = -1.0
    
    # Add small noise for diversity
    action[:3] += np.random.normal(0, 0.02, 3)
    
    return action.astype(np.float32)


def random_policy(obs, info, step, env):
    """Random actions — useful for exploring the action space."""
    return env.action_space.sample()


def load_vla_policy(model_path):
    """Load a trained SmolVLA model for data collection."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SmolVLAPolicy.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    def vla_policy(obs, info, step, env):
        with torch.no_grad():
            obs_dict = {
                "observation.images.front": torch.tensor(obs["pixels"]).unsqueeze(0).to(device),
                "observation.state": torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0).to(device),
            }
            action = model.select_action(obs_dict)
        return action.squeeze(0).cpu().numpy()
    
    return vla_policy


def collect_episodes(policy_fn, num_episodes, max_steps, task_description):
    """Run policy in simulation and collect trajectories."""
    
    env = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode="rgb_array", image_obs=True)
    
    all_episodes = []
    successes = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_frames = []
        total_reward = 0.0
        success = False
        
        for step in range(max_steps):
            action = policy_fn(obs, info, step, env)
            next_obs, reward, done, truncated, info = env.step(action)
            
            episode_frames.append({
                "image": obs["pixels"]["front"] if isinstance(obs, dict) and "pixels" in obs else obs,
                "state": obs["agent_pos"] if isinstance(obs, dict) and "agent_pos" in obs else np.zeros(18),
                "action": action,
                "reward": float(reward),
            })
            
            total_reward += reward
            obs = next_obs
            
            if done:
                success = True
                successes += 1
                break
            if truncated:
                break
        
        episode_result = {
            "frames": episode_frames,
            "total_reward": total_reward,
            "success": success,
            "num_steps": len(episode_frames),
        }
        all_episodes.append(episode_result)
        
        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep+1}/{num_episodes}: {status} | reward={total_reward:.3f} | steps={len(episode_frames)}")
    
    env.close()
    
    success_rate = successes / num_episodes if num_episodes > 0 else 0
    avg_reward = sum(e["total_reward"] for e in all_episodes) / num_episodes if num_episodes > 0 else 0
    
    print(f"\nCollection complete:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Successes: {successes} ({success_rate*100:.1f}%)")
    print(f"  Avg reward: {avg_reward:.3f}")
    
    return all_episodes, {"success_rate": success_rate, "avg_reward": avg_reward}


def save_as_lerobot_dataset(episodes, repo_id, task_description, output_dir=None):
    """Convert collected episodes to LeRobot dataset format."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    # Infer shapes from first episode
    sample_frame = episodes[0]["frames"][0]
    img_shape = sample_frame["image"].shape if hasattr(sample_frame["image"], "shape") else (128, 128, 3)
    state_shape = sample_frame["state"].shape if hasattr(sample_frame["state"], "shape") else (18,)
    action_shape = sample_frame["action"].shape if hasattr(sample_frame["action"], "shape") else (7,)
    
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": list(img_shape),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": list(state_shape),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": list(action_shape),
            "names": ["action"],
        },
    }
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type="panda",
        features=features,
        image_writer_threads=4,
    )
    
    for ep in episodes:
        for frame in ep["frames"]:
            dataset.add_frame({
                "observation.images.front": frame["image"],
                "observation.state": torch.tensor(frame["state"], dtype=torch.float32),
                "action": torch.tensor(frame["action"], dtype=torch.float32),
            })
        dataset.save_episode(task=task_description)
    
    dataset.consolidate()
    print(f"\nDataset saved: {repo_id}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total frames: {sum(len(e['frames']) for e in episodes)}")
    
    return dataset


def save_raw_episodes(episodes, output_path):
    """Save raw episode data as JSON for later processing."""
    
    serializable = []
    for ep in episodes:
        serializable.append({
            "total_reward": ep["total_reward"],
            "success": ep["success"],
            "num_steps": ep["num_steps"],
            # Don't save raw images in JSON — too large
        })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2))
    print(f"Raw episode metadata saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect simulation trajectories")
    parser.add_argument("--policy", choices=["random", "scripted", "vla"], default="scripted",
                        help="Policy to use for data collection")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to VLA model (required if policy=vla)")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--task", type=str, default="Pick up the cube",
                        help="Task description for the dataset")
    parser.add_argument("--dataset-id", type=str, default="user/sim_panda_pick",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--save-lerobot", action="store_true",
                        help="Save as LeRobot dataset format")
    parser.add_argument("--output-dir", type=str, default="./data/trajectories",
                        help="Directory for raw trajectory output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hermes Embodied — Trajectory Collection")
    print("=" * 60)
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Task: {args.task}")
    
    # Select policy
    if args.policy == "random":
        policy_fn = random_policy
    elif args.policy == "scripted":
        policy_fn = scripted_pick_policy
    elif args.policy == "vla":
        if not args.model_path:
            print("ERROR: --model-path required when policy=vla")
            sys.exit(1)
        policy_fn = load_vla_policy(args.model_path)
    
    # Collect
    episodes, stats = collect_episodes(
        policy_fn, args.num_episodes, args.max_steps, args.task
    )
    
    # Save raw metadata
    save_raw_episodes(episodes, f"{args.output_dir}/metadata.json")
    
    # Save as LeRobot dataset
    if args.save_lerobot:
        save_as_lerobot_dataset(episodes, args.dataset_id, args.task)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

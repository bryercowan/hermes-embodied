#!/usr/bin/env python3
"""
Evaluate VLA checkpoints in simulation.
Supports open-loop (trajectory comparison) and closed-loop (live sim) evaluation.
Can compare two checkpoints for A/B testing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import gym_hil
import numpy as np
import torch

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def load_model(model_path, device="cpu"):
    """Load a SmolVLA model from path or HF hub."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    model = SmolVLAPolicy.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model


def prepare_obs(obs, device="cpu"):
    """Format gym observation for SmolVLA input."""
    obs_dict = {}
    
    if isinstance(obs, dict):
        if "pixels" in obs:
            img = obs["pixels"]["front"]  # front camera
            if isinstance(img, np.ndarray):
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_dict["observation.images.front"] = img.unsqueeze(0).to(device)
            
            if "wrist" in obs["pixels"]:
                wrist_img = obs["pixels"]["wrist"]
                if isinstance(wrist_img, np.ndarray):
                    wrist_img = torch.tensor(wrist_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                obs_dict["observation.images.wrist"] = wrist_img.unsqueeze(0).to(device)
        
        if "agent_pos" in obs:
            state = obs["agent_pos"]
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            obs_dict["observation.state"] = state.unsqueeze(0).to(device)
    
    return obs_dict


def evaluate_model(model, num_episodes=50, max_steps=300, device="cpu", verbose=True):
    """
    Closed-loop evaluation: run model in simulation, measure success rate.
    """
    env = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode="rgb_array", image_obs=True)
    
    successes = 0
    total_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        success = False
        
        for step in range(max_steps):
            with torch.no_grad():
                obs_dict = prepare_obs(obs, device)
                action = model.select_action(obs_dict)
            
            action_np = action.squeeze(0).cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
            obs, reward, done, truncated, info = env.step(action_np)
            ep_reward += reward
            
            if done:
                success = True
                successes += 1
                break
            if truncated:
                break
        
        total_rewards.append(ep_reward)
        episode_lengths.append(step + 1)
        
        if verbose:
            status = "OK" if success else "--"
            print(f"  [{status}] Episode {ep+1}/{num_episodes}: reward={ep_reward:.3f}, steps={step+1}")
    
    env.close()
    
    metrics = {
        "success_rate": successes / num_episodes,
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "num_episodes": num_episodes,
        "num_successes": successes,
    }
    
    return metrics


def compare_models(model_a_path, model_b_path, num_episodes=50, device="cpu"):
    """A/B test two checkpoints."""
    
    print(f"\n{'='*60}")
    print(f"A/B Test: Comparing two checkpoints")
    print(f"{'='*60}")
    
    print(f"\nModel A: {model_a_path}")
    model_a = load_model(model_a_path, device)
    metrics_a = evaluate_model(model_a, num_episodes, device=device)
    del model_a
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nModel B: {model_b_path}")
    model_b = load_model(model_b_path, device)
    metrics_b = evaluate_model(model_b, num_episodes, device=device)
    del model_b
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Model A':>12} {'Model B':>12} {'Delta':>12}")
    print(f"{'-'*61}")
    
    for key in ["success_rate", "avg_reward", "avg_episode_length"]:
        a_val = metrics_a[key]
        b_val = metrics_b[key]
        delta = b_val - a_val
        
        if key == "success_rate":
            print(f"{key:<25} {a_val*100:>11.1f}% {b_val*100:>11.1f}% {delta*100:>+11.1f}%")
        else:
            print(f"{key:<25} {a_val:>12.3f} {b_val:>12.3f} {delta:>+12.3f}")
    
    # Determine winner
    if metrics_b["success_rate"] > metrics_a["success_rate"] + 0.05:
        winner = "B"
        print(f"\nWINNER: Model B (>{5}% improvement)")
    elif metrics_a["success_rate"] > metrics_b["success_rate"] + 0.05:
        winner = "A"
        print(f"\nWINNER: Model A (>{5}% better)")
    else:
        winner = "TIE"
        print(f"\nTIE: No significant difference (<5%)")
    
    return {
        "model_a": {"path": model_a_path, "metrics": metrics_a},
        "model_b": {"path": model_b_path, "metrics": metrics_b},
        "winner": winner,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA checkpoints")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (or HF repo ID)")
    parser.add_argument("--compare-with", type=str, default=None,
                        help="Second model path for A/B testing")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    print("=" * 60)
    print("Hermes Embodied — VLA Evaluation")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    
    if args.compare_with:
        # A/B test
        results = compare_models(
            args.model, args.compare_with,
            num_episodes=args.num_episodes,
            device=args.device,
        )
    else:
        # Single model eval
        print(f"\nLoading model: {args.model}")
        model = load_model(args.model, args.device)
        
        print(f"\nRunning {args.num_episodes} episodes...")
        metrics = evaluate_model(model, args.num_episodes, args.max_steps, args.device)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Avg reward:   {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}")
        print(f"  Avg length:   {metrics['avg_episode_length']:.1f} steps")
        
        results = {"model": args.model, "metrics": metrics}
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

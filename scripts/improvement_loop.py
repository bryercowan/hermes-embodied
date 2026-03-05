#!/usr/bin/env python3
"""
The self-improvement loop — runs autonomously.

Cycle:
1. Load current best model
2. Collect episodes in simulation  
3. Curate successful trajectories
4. Check if enough data for retraining
5. If yes: train new model, evaluate, promote if better
6. Save state for next cycle

Designed to be called by Hermes Agent cron or manually.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Headless rendering
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ── Configuration ────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Collection
    "episodes_per_cycle": 20,
    "max_steps_per_episode": 300,
    "task_description": "Pick up the cube",
    
    # Curation
    "reward_threshold": 0.5,
    "require_success": False,
    
    # Retraining
    "retrain_threshold": 50,
    "training_steps": 5000,
    "batch_size": 64,
    "learning_rate": 1e-5,
    "base_model": "lerobot/smolvla_base",
    
    # Evaluation
    "eval_episodes": 50,
    "improvement_threshold": 0.05,
    
    # Infrastructure
    "train_mode": "local",  # "local" or "vastai"
    "gpu_type": "A100_SXM4",
    "max_gpu_cost": 2.0,
    
    # Paths
    "workspace": os.path.expanduser("~/hermes-embodied"),
}


# ── State Management ────────────────────────────────────────

def load_state(workspace):
    state_file = Path(workspace) / "loop_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {
        "current_best": "lerobot/smolvla_base",
        "generation": 0,
        "episodes_since_last_train": 0,
        "total_episodes_collected": 0,
        "best_success_rate": 0.0,
        "training_history": [],
        "cycle_count": 0,
        "created_at": datetime.now().isoformat(),
        "last_cycle": None,
    }


def save_state(state, workspace):
    state_file = Path(workspace) / "loop_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


# ── Pipeline Steps ──────────────────────────────────────────

def step_collect(state, config):
    """Step 1-2: Deploy model and collect trajectories."""
    import torch
    import gymnasium as gym
    import gym_hil
    
    print(f"\n{'─'*40}")
    print(f"COLLECT: Running {config['episodes_per_cycle']} episodes")
    print(f"Model: {state['current_best']}")
    print(f"{'─'*40}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    
    # Load model
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    try:
        model = SmolVLAPolicy.from_pretrained(state["current_best"])
        model.to(device)
        model.eval()
        using_vla = True
    except Exception as e:
        print(f"Could not load VLA model ({e}), using scripted policy")
        using_vla = False
    
    env = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode="rgb_array", image_obs=True)
    
    episodes = []
    successes = 0
    
    for ep in range(config["episodes_per_cycle"]):
        obs, info = env.reset()
        frames = []
        total_reward = 0.0
        success = False
        
        for step in range(config["max_steps_per_episode"]):
            if using_vla:
                with torch.no_grad():
                    # Format obs for model
                    obs_dict = {}
                    if isinstance(obs, dict):
                        if "pixels" in obs:
                            img = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1) / 255.0
                            obs_dict["observation.images.front"] = img.unsqueeze(0).to(device)
                        if "agent_pos" in obs:
                            obs_dict["observation.state"] = torch.tensor(
                                obs["agent_pos"], dtype=torch.float32
                            ).unsqueeze(0).to(device)
                    
                    action = model.select_action(obs_dict)
                    action_np = action.squeeze(0).cpu().numpy()
            else:
                action_np = env.action_space.sample()
            
            next_obs, reward, done, truncated, info = env.step(action_np)
            
            frames.append({
                "state": obs["agent_pos"] if isinstance(obs, dict) and "agent_pos" in obs else np.zeros(18),
                "action": action_np,
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
        
        episodes.append({
            "frames": frames,
            "total_reward": total_reward,
            "success": success,
            "num_steps": len(frames),
        })
        
        status = "OK" if success else "--"
        print(f"  [{status}] Ep {ep+1}: reward={total_reward:.3f}, steps={len(frames)}")
    
    env.close()
    if using_vla:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    success_rate = successes / config["episodes_per_cycle"]
    print(f"\nSuccess rate: {success_rate*100:.1f}%")
    
    return episodes, success_rate


def step_curate(episodes, config):
    """Step 3: Filter to keep only good trajectories."""
    
    print(f"\n{'─'*40}")
    print(f"CURATE: Filtering episodes")
    print(f"{'─'*40}")
    
    good = []
    for ep in episodes:
        passes = True
        
        if config["require_success"] and not ep["success"]:
            passes = False
        
        if ep["total_reward"] < config["reward_threshold"] and not ep["success"]:
            passes = False
        
        if passes:
            good.append(ep)
    
    print(f"Kept {len(good)}/{len(episodes)} episodes")
    return good


def step_should_retrain(state, config, new_episode_count):
    """Step 4: Check if we have enough data to retrain."""
    
    total = state["episodes_since_last_train"] + new_episode_count
    threshold = config["retrain_threshold"]
    
    print(f"\n{'─'*40}")
    print(f"RETRAIN CHECK")
    print(f"{'─'*40}")
    print(f"Episodes since last train: {state['episodes_since_last_train']}")
    print(f"New episodes this cycle: {new_episode_count}")
    print(f"Total: {total} / {threshold} needed")
    
    if total >= threshold:
        print(f"READY TO RETRAIN")
        return True
    else:
        print(f"Not enough data yet. Need {threshold - total} more episodes.")
        return False


def step_train(state, config):
    """Step 5a: Trigger training."""
    
    print(f"\n{'─'*40}")
    print(f"TRAIN: Starting fine-tuning")
    print(f"{'─'*40}")
    
    new_gen = state["generation"] + 1
    
    import subprocess
    
    cmd = [
        sys.executable, "scripts/train_smolvla.py",
        f"--mode={config['train_mode']}",
        f"--base-model={state['current_best']}",
        f"--dataset-id=user/robot_loop_gen{new_gen}",
        f"--steps={config['training_steps']}",
        f"--batch-size={config['batch_size']}",
        f"--output-dir={config['workspace']}/checkpoints/gen{new_gen}",
        f"--job-name=hermes_embodied_gen{new_gen}",
    ]
    
    if config.get("learning_rate"):
        cmd.append(f"--learning-rate={config['learning_rate']}")
    
    if config["train_mode"] == "vastai":
        cmd.extend([
            f"--gpu-type={config['gpu_type']}",
            f"--max-cost={config['max_gpu_cost']}",
        ])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Training FAILED (exit code {result.returncode})")
        return None
    
    checkpoint_path = f"{config['workspace']}/checkpoints/gen{new_gen}"
    print(f"Training complete: {checkpoint_path}")
    return checkpoint_path


def step_evaluate(state, config, new_checkpoint):
    """Step 5b: A/B test new checkpoint vs current best."""
    
    print(f"\n{'─'*40}")
    print(f"EVALUATE: A/B testing")
    print(f"{'─'*40}")
    
    import subprocess
    
    result_file = f"{config['workspace']}/evaluations/ab_test_gen{state['generation']+1}.json"
    
    cmd = [
        sys.executable, "scripts/evaluate.py",
        f"--model={state['current_best']}",
        f"--compare-with={new_checkpoint}",
        f"--num-episodes={config['eval_episodes']}",
        f"--output={result_file}",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr}")
        return None
    
    if os.path.exists(result_file):
        return json.loads(Path(result_file).read_text())
    
    return None


def step_promote(state, config, eval_results, new_checkpoint):
    """Step 6: Promote new model if it's better."""
    
    print(f"\n{'─'*40}")
    print(f"PROMOTE: Deciding on new checkpoint")
    print(f"{'─'*40}")
    
    if eval_results is None:
        print("No evaluation results — keeping current model")
        return False
    
    winner = eval_results.get("winner", "TIE")
    
    if winner == "B":  # Model B is the new checkpoint
        new_gen = state["generation"] + 1
        new_success_rate = eval_results["model_b"]["metrics"]["success_rate"]
        
        print(f"PROMOTING gen {new_gen}!")
        print(f"Success rate: {state['best_success_rate']*100:.1f}% → {new_success_rate*100:.1f}%")
        
        state["current_best"] = new_checkpoint
        state["generation"] = new_gen
        state["best_success_rate"] = new_success_rate
        state["episodes_since_last_train"] = 0
        state["training_history"].append({
            "generation": new_gen,
            "success_rate": new_success_rate,
            "avg_reward": eval_results["model_b"]["metrics"]["avg_reward"],
            "timestamp": datetime.now().isoformat(),
            "promoted": True,
        })
        
        return True
    else:
        print(f"New model not significantly better (winner={winner}). Keeping current.")
        state["training_history"].append({
            "generation": state["generation"],
            "timestamp": datetime.now().isoformat(),
            "promoted": False,
            "reason": "no_improvement",
        })
        return False


# ── Main Loop ───────────────────────────────────────────────

def run_cycle(config):
    """Execute one full improvement cycle."""
    
    workspace = config["workspace"]
    state = load_state(workspace)
    
    state["cycle_count"] += 1
    state["last_cycle"] = datetime.now().isoformat()
    
    print("=" * 60)
    print(f"HERMES EMBODIED — Improvement Cycle #{state['cycle_count']}")
    print(f"Generation: {state['generation']}")
    print(f"Current best: {state['current_best']}")
    print(f"Best success rate: {state['best_success_rate']*100:.1f}%")
    print("=" * 60)
    
    # Step 1-2: Collect
    episodes, current_success_rate = step_collect(state, config)
    
    # Step 3: Curate
    good_episodes = step_curate(episodes, config)
    state["episodes_since_last_train"] += len(good_episodes)
    state["total_episodes_collected"] += len(episodes)
    
    # Step 4: Check if ready to retrain
    should_train = step_should_retrain(state, config, len(good_episodes))
    
    if should_train:
        # Step 5a: Train
        new_checkpoint = step_train(state, config)
        
        if new_checkpoint:
            # Step 5b: Evaluate
            eval_results = step_evaluate(state, config, new_checkpoint)
            
            # Step 6: Promote
            promoted = step_promote(state, config, eval_results, new_checkpoint)
    
    # Save state
    save_state(state, workspace)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"CYCLE #{state['cycle_count']} COMPLETE")
    print(f"{'='*60}")
    print(f"Episodes collected this cycle: {len(episodes)}")
    print(f"Good episodes kept: {len(good_episodes)}")
    print(f"Current success rate: {current_success_rate*100:.1f}%")
    print(f"Generation: {state['generation']}")
    print(f"Best success rate: {state['best_success_rate']*100:.1f}%")
    print(f"Total episodes ever: {state['total_episodes_collected']}")
    
    return state


def main():
    parser = argparse.ArgumentParser(description="Run the self-improvement loop")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON file")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Workspace directory")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes per cycle")
    parser.add_argument("--continuous", action="store_true",
                        help="Run continuously (loop forever)")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between cycles in continuous mode")
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    
    if args.workspace:
        config["workspace"] = args.workspace
    if args.episodes:
        config["episodes_per_cycle"] = args.episodes
    
    if args.continuous:
        print("Running in continuous mode. Press Ctrl+C to stop.")
        while True:
            try:
                run_cycle(config)
                print(f"\nSleeping {args.interval}s until next cycle...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
    else:
        run_cycle(config)


if __name__ == "__main__":
    main()

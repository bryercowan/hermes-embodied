#!/usr/bin/env python3
"""
Setup and verify the LeRobot + MuJoCo simulation environment.
Run this first to make sure everything works before training.
"""

import subprocess
import sys
import os


def run(cmd, check=True):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)
    return result


def check_python_version():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.minor != 10:
        print("WARNING: LeRobot recommends Python 3.10. You may encounter issues.")


def install_dependencies():
    """Install LeRobot with simulation and SmolVLA support."""
    
    # Check if already installed
    try:
        import lerobot
        print(f"LeRobot already installed: {lerobot.__version__}")
        return
    except ImportError:
        pass
    
    print("\nInstalling LeRobot with simulation support...")
    
    # Check if we're in the lerobot repo
    if os.path.exists("lerobot") and os.path.exists("lerobot/setup.py"):
        run("pip install -e './lerobot[hilserl,smolvla]'")
    else:
        run("git clone https://github.com/huggingface/lerobot.git")
        run("pip install -e './lerobot[hilserl,smolvla]'")


def setup_headless():
    """Configure headless rendering for cloud/CI environments."""
    if sys.platform == "linux":
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("Set MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl for headless rendering")
    elif sys.platform == "darwin":
        print("macOS detected — using CGL (native), no special config needed")


def verify_simulation():
    """Run a quick simulation test."""
    print("\n=== Verifying MuJoCo + gym_hil ===")
    
    import gymnasium as gym
    import gym_hil
    import numpy as np
    
    env = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode="rgb_array", image_obs=True)
    obs, info = env.reset()
    
    print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    # Run 20 random steps
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            obs, info = env.reset()
    
    env.close()
    print(f"\nRan 20 random steps. Total reward: {total_reward:.3f}")
    print("Simulation verified!")
    
    return True


def verify_smolvla():
    """Check that SmolVLA can be loaded."""
    print("\n=== Verifying SmolVLA ===")
    
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        print("SmolVLA policy class found!")
        
        # Don't actually download the model yet — just verify the import works
        print("SmolVLA verified! (model will be downloaded on first use)")
        return True
    except ImportError as e:
        print(f"SmolVLA import failed: {e}")
        print("Try: pip install -e './lerobot[smolvla]'")
        return False


def main():
    print("=" * 60)
    print("Hermes Embodied — Simulation Environment Setup")
    print("=" * 60)
    
    check_python_version()
    setup_headless()
    install_dependencies()
    
    sim_ok = verify_simulation()
    vla_ok = verify_smolvla()
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Simulation (MuJoCo + gym_hil): {'OK' if sim_ok else 'FAILED'}")
    print(f"  SmolVLA policy:                {'OK' if vla_ok else 'FAILED'}")
    print("=" * 60)
    
    if sim_ok and vla_ok:
        print("\nReady to go! Next steps:")
        print("  1. python scripts/collect_trajectories.py  — collect training data")
        print("  2. python scripts/train_smolvla.py         — fine-tune SmolVLA")
        print("  3. python scripts/evaluate.py              — evaluate checkpoint")
        print("  4. python scripts/improvement_loop.py      — run the full loop")
    else:
        print("\nSome checks failed. Fix the issues above and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hermes Embodied — Environment Setup & Verification

This script is the FIRST thing that must run before any training or sim work.
It handles:
  1. Conda environment creation (lerobot, Python 3.10)
  2. System dependency installation (ffmpeg via conda-forge)
  3. LeRobot + MuJoCo + SmolVLA + gym_hil installation
  4. Vast.ai SDK installation
  5. Full verification of sim, model imports, and obs structure
  6. Writes a .setup_complete marker so it only runs once

Designed to be called by Hermes Agent automatically before any other script.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Configuration ────────────────────────────────────────────

CONDA_ENV_NAME = "lerobot"
PYTHON_VERSION = "3.10"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
LEROBOT_DIR = PROJECT_DIR / "lerobot"
SETUP_MARKER = PROJECT_DIR / ".setup_complete"
WORKSPACE = Path.home() / "hermes-embodied"

# Verified working versions (pinned from our tested setup)
EXPECTED_VERSIONS = {
    "python": "3.10",
    "gym-hil": "0.1.13",
    "mujoco": "3.5",
    # torch and transformers float — just check they import
}


# ── Helpers ──────────────────────────────────────────────────

def run(cmd, check=True, capture=True, shell=True):
    """Run a shell command, print it, return result."""
    print(f"  >>> {cmd}")
    result = subprocess.run(
        cmd, shell=shell, capture_output=capture, text=True,
        timeout=600  # 10 min max per command
    )
    if result.stdout and capture:
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"      {line}")
    if result.returncode != 0 and check:
        if result.stderr:
            print(f"  ERROR: {result.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def conda_cmd(cmd):
    """Run a command inside the conda lerobot env."""
    # Need to init conda in the subshell
    prefix = 'eval "$(/opt/homebrew/bin/conda shell.bash hook 2>/dev/null || conda shell.bash hook 2>/dev/null)" && conda activate lerobot && '
    return run(prefix + cmd)


def is_setup_complete():
    """Check if setup has already been done successfully."""
    if not SETUP_MARKER.exists():
        return False
    try:
        data = json.loads(SETUP_MARKER.read_text())
        # Check if the marker is from a successful run
        if data.get("status") == "complete":
            print(f"Setup already completed on {data.get('timestamp', 'unknown')}")
            return True
    except (json.JSONDecodeError, KeyError):
        pass
    return False


def mark_setup_complete(verified_config):
    """Write marker file indicating setup is done."""
    SETUP_MARKER.write_text(json.dumps({
        "status": "complete",
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "arch": platform.machine(),
        "config": verified_config,
    }, indent=2))


# ── Setup Steps ──────────────────────────────────────────────

def step_1_conda():
    """Ensure conda is installed and lerobot env exists."""
    print("\n[1/6] Checking conda environment...")

    # Check if conda exists
    conda_path = shutil.which("conda")
    if not conda_path:
        # Try homebrew location
        conda_path = "/opt/homebrew/bin/conda"
        if not os.path.exists(conda_path):
            print("  Installing miniconda via Homebrew...")
            run("brew install --cask miniconda")
            conda_path = "/opt/homebrew/bin/conda"

    # Check if lerobot env exists
    result = run(f"{conda_path} env list", check=False)
    if CONDA_ENV_NAME in result.stdout:
        print(f"  Conda env '{CONDA_ENV_NAME}' already exists")
        return

    print(f"  Creating conda env '{CONDA_ENV_NAME}' with Python {PYTHON_VERSION}...")

    # Accept TOS if needed (non-interactive)
    run(f"{conda_path} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true")
    run(f"{conda_path} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true")

    run(f"{conda_path} create -y -n {CONDA_ENV_NAME} python={PYTHON_VERSION}")
    print(f"  Conda env created!")


def step_2_ffmpeg():
    """Install ffmpeg with svtav1 support via conda-forge."""
    print("\n[2/6] Checking ffmpeg...")

    result = conda_cmd("conda list ffmpeg 2>/dev/null | grep ffmpeg || echo 'missing'")
    if "missing" in result.stdout:
        print("  Installing ffmpeg from conda-forge...")
        conda_cmd("conda install -y ffmpeg -c conda-forge")
    else:
        print("  ffmpeg already installed")


def step_3_lerobot():
    """Clone and install LeRobot with sim + SmolVLA support."""
    print("\n[3/6] Checking LeRobot installation...")

    # Check if already importable
    result = conda_cmd("python -c 'import lerobot; print(lerobot.__version__)' 2>/dev/null || echo 'missing'")
    if "missing" not in result.stdout:
        print(f"  LeRobot already installed: {result.stdout.strip()}")
        return

    # Clone if needed
    if not LEROBOT_DIR.exists():
        print("  Cloning LeRobot repository...")
        run(f"git clone https://github.com/huggingface/lerobot.git {LEROBOT_DIR}")
    else:
        print("  LeRobot repo already cloned")

    # Install with sim + smolvla extras
    print("  Installing LeRobot with [hilserl,smolvla] extras (this takes a few minutes)...")
    conda_cmd(f"cd {LEROBOT_DIR} && pip install -e '.[hilserl,smolvla]'")
    print("  LeRobot installed!")


def step_4_vastai():
    """Install Vast.ai SDK for cloud GPU provisioning."""
    print("\n[4/6] Checking Vast.ai SDK...")

    result = conda_cmd("python -c 'from vastai import VastAI; print(\"ok\")' 2>/dev/null || echo 'missing'")
    if "missing" in result.stdout:
        print("  Installing vastai-sdk + paramiko...")
        # IMPORTANT: --no-deps on vastai-sdk to avoid downgrading transformers
        # vastai-sdk pins transformers<4.53 but SmolVLA needs >=5.0
        # The SDK works fine with transformers 5.x despite the pin
        conda_cmd("pip install --no-deps vastai-sdk && pip install paramiko")
    else:
        print("  Vast.ai SDK already installed")


def step_5_headless():
    """Configure headless rendering for Linux (cloud/CI)."""
    print("\n[5/6] Configuring rendering backend...")

    system = platform.system()
    if system == "Linux":
        # Check for EGL support
        print("  Linux detected — setting up EGL headless rendering")
        run("sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev 2>/dev/null || true", check=False)
        # These get set at runtime by the scripts themselves
        print("  Scripts will use MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl")
    elif system == "Darwin":
        print("  macOS detected — using CGL (native). No extra config needed.")
    else:
        print(f"  Unknown platform: {system}. MuJoCo may need manual GL config.")


def step_6_verify():
    """Run full verification of the installed environment."""
    print("\n[6/6] Verifying complete environment...\n")

    verification_script = """
import sys, json

results = {}
errors = []

# 1. Check gym_hil simulation
try:
    import gymnasium as gym
    import gym_hil
    env = gym.make('gym_hil/PandaPickCubeBase-v0', render_mode='rgb_array', image_obs=True)
    obs, _ = env.reset()

    # Verify obs structure
    assert 'pixels' in obs, 'obs missing pixels key'
    assert 'front' in obs['pixels'], 'obs.pixels missing front camera'
    assert 'wrist' in obs['pixels'], 'obs.pixels missing wrist camera'
    assert obs['pixels']['front'].shape == (128, 128, 3), f'front cam wrong shape: {obs["pixels"]["front"].shape}'
    assert obs['pixels']['wrist'].shape == (128, 128, 3), f'wrist cam wrong shape: {obs["pixels"]["wrist"].shape}'
    assert 'agent_pos' in obs, 'obs missing agent_pos'
    assert obs['agent_pos'].shape == (18,), f'agent_pos wrong shape: {obs["agent_pos"].shape}'
    assert env.action_space.shape == (7,), f'action space wrong shape: {env.action_space.shape}'
    env.close()

    results['sim'] = 'OK'
    results['obs_pixels_front'] = list(obs['pixels']['front'].shape)
    results['obs_pixels_wrist'] = list(obs['pixels']['wrist'].shape)
    results['obs_agent_pos'] = list(obs['agent_pos'].shape)
    results['action_space'] = list(env.action_space.shape)
except Exception as e:
    results['sim'] = f'FAIL: {e}'
    errors.append(f'Simulation: {e}')

# 2. Check SmolVLA import
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    results['smolvla'] = 'OK'
except ImportError as e:
    results['smolvla'] = f'FAIL: {e}'
    errors.append(f'SmolVLA: {e}')

# 3. Check LeRobot dataset tools
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    results['dataset'] = 'OK'
except ImportError as e:
    results['dataset'] = f'FAIL: {e}'
    errors.append(f'Dataset: {e}')

# 4. Check torch + device
try:
    import torch
    results['torch'] = torch.__version__
    if torch.cuda.is_available():
        results['device'] = f'cuda ({torch.cuda.get_device_name(0)})'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        results['device'] = 'mps (Apple Silicon)'
    else:
        results['device'] = 'cpu'
except Exception as e:
    results['torch'] = f'FAIL: {e}'
    errors.append(f'Torch: {e}')

# 5. Check Vast.ai SDK
try:
    from vastai import VastAI
    results['vastai'] = 'OK'
except ImportError:
    results['vastai'] = 'NOT INSTALLED (optional — needed for cloud training)'

# Output
results['errors'] = errors
results['all_passed'] = len(errors) == 0
print(json.dumps(results))
"""

    result = conda_cmd(f'python -c "{verification_script}" 2>/dev/null')

    try:
        verify = json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        print("  VERIFICATION FAILED — could not parse results")
        print(f"  Raw output: {result.stdout}")
        return None

    # Pretty print results
    print("  VERIFICATION RESULTS:")
    print(f"    Simulation (MuJoCo + gym_hil): {verify.get('sim', '?')}")
    print(f"    SmolVLA policy:                {verify.get('smolvla', '?')}")
    print(f"    LeRobot dataset:               {verify.get('dataset', '?')}")
    print(f"    PyTorch:                        {verify.get('torch', '?')}")
    print(f"    Compute device:                 {verify.get('device', '?')}")
    print(f"    Vast.ai SDK:                    {verify.get('vastai', '?')}")

    if verify.get('sim') == 'OK':
        print(f"\n    Obs structure:")
        print(f"      pixels.front:  {verify.get('obs_pixels_front', '?')}")
        print(f"      pixels.wrist:  {verify.get('obs_pixels_wrist', '?')}")
        print(f"      agent_pos:     {verify.get('obs_agent_pos', '?')}")
        print(f"      action_space:  {verify.get('action_space', '?')}")

    if verify.get("all_passed"):
        print("\n  ALL CHECKS PASSED")
        return verify
    else:
        print(f"\n  FAILURES: {verify.get('errors', [])}")
        return None


def step_create_workspace():
    """Create the workspace directory for loop state, checkpoints, etc."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "checkpoints").mkdir(exist_ok=True)
    (WORKSPACE / "trajectories").mkdir(exist_ok=True)
    (WORKSPACE / "evaluations").mkdir(exist_ok=True)


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HERMES EMBODIED — Environment Setup")
    print("=" * 60)
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Project:  {PROJECT_DIR}")

    # Check if already done
    if "--force" not in sys.argv and is_setup_complete():
        print("\n  To re-run setup, use: python setup_sim.py --force")

        # Still run quick verification
        print("\n  Running quick verification...")
        verify = step_6_verify()
        if verify and verify.get("all_passed"):
            return 0
        else:
            print("  Verification failed — re-running setup...")

    # Run setup
    try:
        step_1_conda()
        step_2_ffmpeg()
        step_3_lerobot()
        step_4_vastai()
        step_5_headless()
        verify = step_6_verify()
        step_create_workspace()
    except Exception as e:
        print(f"\n  SETUP FAILED: {e}")
        print("  Fix the error above and re-run.")
        return 1

    if verify and verify.get("all_passed"):
        mark_setup_complete(verify)
        print("\n" + "=" * 60)
        print("  SETUP COMPLETE!")
        print("=" * 60)
        print(f"\n  Activate with: conda activate lerobot")
        print(f"  Then run:      python scripts/demo.py")
        print(f"  Or train:      python scripts/train_smolvla.py --dataset-id Abderlrahman/panda_mujoco_lerobot")
        return 0
    else:
        print("\n  Setup completed but verification failed.")
        print("  Check the errors above and try --force to re-run.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

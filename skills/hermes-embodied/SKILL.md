---
name: hermes-embodied
description: "Self-improving robotics via Hermes Agent. One skill that provisions cloud GPUs, fine-tunes VLA models (SmolVLA/GR00T), runs sim evaluation, and autonomously improves robot policies through a collect→curate→train→eval→promote loop. Just say 'train my robot to pick up cubes' and it handles everything."
version: 1.0.0
author: bryer
dependencies:
  - lerobot
  - mujoco
  - gym-hil>=0.1.13
  - vastai-sdk>=0.5.0
  - paramiko
  - wandb
compatibility:
  platforms: [linux, macos]
metadata:
  hermes:
    tags: [robotics, vla, training, simulation, self-improvement, smolvla, groot, lerobot, vast.ai]
---

# Hermes Embodied — Self-Improving Robotics

One skill. Say "train my robot" and Hermes handles everything: GPU provisioning, data collection, VLA fine-tuning, evaluation, and autonomous improvement.

## FIRST RUN — Setup Gate

Before doing ANYTHING with this skill, check if the setup marker exists:
```bash
test -f ~/projects/hermes-embodied/.setup_complete && echo "ready" || echo "needs setup"
```

If it says "needs setup", run the setup script FIRST:
```bash
python ~/projects/hermes-embodied/scripts/setup_sim.py
```

This creates the conda env, installs all dependencies, and verifies everything works.
It only needs to run once — subsequent calls detect the marker and skip.
After setup, ALWAYS activate the env before running any other script:
```bash
eval "$(/opt/homebrew/bin/conda shell.bash hook)" && conda activate lerobot
```

## When To Use

- User mentions robots, arms, manipulation, VLA, SmolVLA, GR00T, LeRobot
- User says "train the robot", "improve the policy", "start the loop"
- User wants to provision a GPU for training anything robotics-related
- User wants to evaluate or compare robot policies

## Quick Reference

### Environment
- Conda env: `lerobot` (Python 3.10)
- Activate: `eval "$(/opt/homebrew/bin/conda shell.bash hook)" && conda activate lerobot`
- Sim: `gym_hil/PandaPickCubeBase-v0` (Franka Panda, MuJoCo)
- Obs: `pixels` (dict: front 128x128, wrist 128x128), `agent_pos` (18-dim)
- Actions: 7-dim continuous (joint deltas + gripper)

### Key Imports
```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import gymnasium as gym
import gym_hil
```

### Ready-to-use Dataset
`Abderlrahman/panda_mujoco_lerobot` — 50 episodes, 12k frames, Panda 7-DOF, two cameras

### Training Command
```bash
cd ~/projects/hermes-embodied/lerobot
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=Abderlrahman/panda_mujoco_lerobot \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/smolvla_panda \
  --policy.device=cuda --wandb.enable=true
```

## Full Pipeline

### Phase 1: Setup (one-time)
```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
conda install ffmpeg -c conda-forge
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[hilserl,smolvla]"
# Linux headless: export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl
```

### Phase 2: Collect Data in Sim
```python
import gymnasium as gym
import gym_hil
env = gym.make("gym_hil/PandaPickCubeBase-v0", render_mode="rgb_array", image_obs=True)
obs, info = env.reset()
# obs["pixels"]["front"] = (128,128,3), obs["agent_pos"] = (18,)
# env.action_space.shape = (7,)
for step in range(200):
    action = env.action_space.sample()  # or your policy
    obs, reward, done, truncated, info = env.step(action)
```

### Phase 3: Provision GPU on Vast.ai

Use the CLI (preferred — works directly from terminal):
```bash
# Set API key (one-time)
vastai set api-key $VAST_API_KEY

# Search for cheapest A100
vastai search offers 'gpu_name=A100_SXM4 num_gpus=1 reliability>0.95 dph<2.0' -o 'dph'

# Create instance (use ID from search results)
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
  --disk 100 --ssh --direct \
  --onstart-cmd "apt-get update && apt-get install -y git && pip install wandb"

# Check status
vastai show instances --raw

# Get SSH connection info
vastai ssh-url <INSTANCE_ID>

# View logs
vastai logs <INSTANCE_ID> --tail 50

# ALWAYS destroy when done (stops billing)
vastai destroy instance <INSTANCE_ID>
```

Use --raw flag for JSON output that's easy to parse programmatically.
The --explain flag shows the underlying API call for debugging.

### Phase 4: Train SmolVLA
On GPU instance via SSH:
```bash
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[smolvla]"
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=Abderlrahman/panda_mujoco_lerobot \
  --batch_size=64 --steps=20000 --policy.device=cuda
```
~4 hours on A100, ~$4 total.

### Phase 5: Evaluate
```python
model = SmolVLAPolicy.from_pretrained("path/to/checkpoint")
model.eval()
# Run in sim, measure success rate over 50 episodes
# Compare old vs new checkpoint for A/B testing
# Promote if >5% improvement
```

### Phase 6: Self-Improvement Loop
```
Deploy model → Collect episodes → Curate successes → 
Retrain when 50+ new episodes → Eval A/B → Promote if better → Repeat
```
Schedule via Hermes cron: `every 6h`

## Scripts (at ~/projects/hermes-embodied/scripts/)
- `setup_sim.py` — Install & verify deps
- `collect_trajectories.py` — Run policies in sim, save data
- `train_smolvla.py` — Local or Vast.ai training
- `evaluate.py` — A/B test checkpoints
- `improvement_loop.py` — Full autonomous cycle
- `demo.py` — Interactive walkthrough

## Telegram Reporting

All training events output structured reports to stdout. When run via Hermes cron
with `deliver: "telegram"`, these reports drop right into the user's chat.

### Set up monitoring cron:
```
schedule: "every 2h"
deliver: "telegram"
prompt: "Run ~/projects/hermes-embodied/scripts/training_monitor.py and report the status.
         Read ~/hermes-embodied/loop_state.json and ~/hermes-embodied/training_log.jsonl.
         Format a concise status update with: current generation, success rate, episodes
         buffered, and whether training is due. If a Vast.ai instance is running, include
         GPU utilization and cost so far."
```

### Training events logged to ~/hermes-embodied/training_log.jsonl:
- cycle_complete: episodes collected, success rate, episodes buffered
- training_started: GPU type, cost/hr, dataset
- training_complete: steps, final loss, duration, total cost
- model_promoted: new generation, success rate delta
- model_rejected: reason

### Report functions in scripts/training_monitor.py:
- `format_status_report()` — full status overview
- `format_cycle_report()` — single cycle summary  
- `format_training_complete_report()` — training results with A/B test
- `log_event(event, details, metrics)` — append to training log

## Pitfalls
1. SmolVLA import: `lerobot.policies.smolvla` (NOT `lerobot.common.policies`)
2. Dataset import: `lerobot.datasets.lerobot_dataset` (NOT `lerobot.common.datasets`)
3. gym_hil obs: `pixels` is a DICT (`front`, `wrist`), not a direct array
4. Stats normalization: MUST use YOUR dataset's stats.json for inference, not pretrained
5. Mac: Use conda not uv, set image_obs=True, render_mode="rgb_array"
6. Headless Linux: MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl
7. Vast.ai CLI: `pip install vastai` (CLI) vs `pip install vastai-sdk` (Python SDK) — both installed
8. Vast.ai CLI: use `vastai` commands from terminal (preferred). Use `--raw` for JSON output.
9. Vast.ai SDK pins transformers<4.53 — installed with --no-deps to avoid breaking SmolVLA
9. LIBERO needs Linux GPU (robosuite + EGL) — use gym_hil on Mac

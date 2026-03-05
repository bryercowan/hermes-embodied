# Hermes Embodied — Quickstart

## Install the Skill (for any Hermes Agent)

```bash
# Add the tap
hermes skills tap add bryercowan/hermes-embodied

# Install the skill
hermes skills install bryercowan/hermes-embodied/skills/hermes-embodied
```

Or just tell your Hermes Agent: "Install the hermes-embodied skill from bryercowan's repo"

## One-Time Setup

Tell Hermes: "Set up the robotics simulation environment"

What it does:
1. Creates conda env `lerobot` with Python 3.10
2. Installs LeRobot + MuJoCo + SmolVLA
3. Verifies sim works (Franka Panda pick-and-place)

## Train a Robot

Tell Hermes: "Train SmolVLA on the panda pick-and-place dataset"

What it does:
1. Downloads `Abderlrahman/panda_mujoco_lerobot` (50 episodes)
2. Provisions A100 on Vast.ai (~$1/hr)
3. Runs fine-tuning for 20k steps (~4 hours)
4. Downloads checkpoint, destroys GPU instance

## Run the Self-Improvement Loop

Tell Hermes: "Start the robot improvement loop"

What it does:
1. Loads current best VLA checkpoint
2. Runs 20 episodes in simulation
3. Keeps successful trajectories
4. When 50+ episodes accumulated, retrains
5. A/B tests new model vs old
6. Promotes if >5% better
7. Repeats on schedule

## Environment Variables Needed

- `VAST_API_KEY` — Get from https://console.vast.ai/
- `WANDB_API_KEY` — Get from https://wandb.ai/ (optional)
- `HF_TOKEN` — Get from https://huggingface.co/ (optional)

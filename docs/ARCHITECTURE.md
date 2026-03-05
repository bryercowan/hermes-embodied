# Architecture: Hermes Embodied

## Core Thesis

The same self-improvement loop works at every level of the abstraction stack:

- **Text tasks**: Hermes Agent → generates trajectories → RL via Tinker-Atropos → better agent
- **Robot tasks**: Hermes Agent → orchestrates VLA → collects trajectories → fine-tune VLA → better robot

Hermes Agent is the constant. It's always the reasoning/planning layer that orchestrates
training, evaluation, and deployment. The only thing that changes is what's being trained.

## System Layers

### Layer 1: Reasoning (Runtime)

Hermes Agent acts as the "System 2" — slow, deliberative planning.

In the robotics context:
- Interprets user intent ("pick up the red block")
- Decomposes into subtasks if needed
- Monitors execution and handles failures
- Decides when to retrain

The VLA acts as "System 1" — fast, reactive motor control at 10-50Hz.

### Layer 2: Training (On-Demand)

User-initiated model improvement. "Train my robot to do X."

Pipeline:
1. User describes task
2. Hermes identifies or downloads relevant dataset (HuggingFace)
3. Hermes provisions GPU on Vast.ai
4. Hermes configures training (model, hyperparams, data config)
5. Training runs with WandB monitoring
6. Hermes evaluates results
7. Hermes deploys new checkpoint

### Layer 3: Continuous Improvement (Autonomous)

The closed loop. No human in the loop.

1. VLA runs in simulation, executing tasks
2. Each rollout is recorded (observations, actions, rewards)
3. Successful rollouts (reward > threshold) become training data
4. When enough new data accumulates (configurable, e.g., 50 episodes)
5. Hermes automatically triggers a retraining run
6. New checkpoint is evaluated against current best
7. If better → promoted to active policy
8. If worse → discarded, old policy continues
9. Loop repeats via Hermes cron scheduling

## Data Flow

```
Simulation Environment (MuJoCo / gym_hil)
    │
    ├── observations: camera images (640x480, front+wrist)
    ├── state: joint positions, gripper state
    ├── actions: delta joint positions (action chunks)
    └── reward: task-specific (e.g., cube lifted > 5cm)
    │
    ▼
LeRobot Dataset Format (HuggingFace compatible)
    │
    ├── videos/ (mp4 per camera per episode)
    ├── data/ (parquet with state/action columns)
    └── meta/ (info.json, stats.json, tasks.json)
    │
    ▼
Vast.ai GPU Instance (A100/A6000)
    │
    ├── SmolVLA fine-tuning (lerobot-train)
    ├── WandB logging (loss curves, eval metrics)
    └── Checkpoint upload (HuggingFace Hub or local)
    │
    ▼
Evaluation
    │
    ├── Open-loop: predicted vs ground truth trajectory
    ├── Closed-loop: run in sim, measure success rate
    └── Comparison: new checkpoint vs current best
    │
    ▼
Deployment (sim or physical arm)
```

## Simulation Environment

Using LeRobot's gym_hil with MuJoCo:
- Franka Panda robot (7DOF + gripper)
- Pick-and-place tasks (cube → bowl)
- 10Hz control frequency
- Dual camera: global view + wrist cam
- Keyboard/gamepad teleop for demo collection

Why Franka Panda (not SO-101 sim):
- Native LeRobot support with gym_hil
- SmolVLA was pretrained on LIBERO (Franka Panda tasks)
- Transfers better to fine-tuning (same embodiment as pretraining)
- SO-101 in sim would require custom environment creation

## SmolVLA Training Details

- Base model: lerobot/smolvla_base (450M params)
- Architecture: Vision encoder + language encoder → action expert
- Action output: chunks of 30 future actions (3 seconds at 10Hz)
- Training: ~4 hours on A100 for 20k steps, batch_size=64
- Data: ~50 episodes per task for good generalization
- Key insight: stats.json must match YOUR data, not pretraining data

## Vast.ai Integration

SDK-based provisioning:
```python
from vastai import VastAI
client = VastAI(api_key=VAST_API_KEY)

# Search for cheapest A100
offers = client.search_offers(
    gpu_name="A100",
    num_gpus=1,
    disk_space=50,
    sort="dph_total"  # dollars per hour
)

# Create instance
instance = client.create_instance(
    id=offers[0].id,
    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
    disk=50
)
```

## Key Decisions

1. **SmolVLA over GR00T for v1**: 450M vs 3B params, 6x faster iteration
2. **Vast.ai over Lambda/RunPod**: Best price, good Python SDK, CLI
3. **Sim-first**: Prove the loop works, physical arm is optional icing
4. **Franka Panda sim**: Native LeRobot support, matches SmolVLA pretraining
5. **Skill-based**: Everything is a Hermes skill — composable, shareable, reusable

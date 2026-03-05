# Hermes Embodied: Self-Improving Robotics via Hermes Agent

> "Any robot owner can fine-tune a state-of-the-art VLA by talking to their agent. No ML expertise needed."

## What Is This?

Hermes Embodied turns [Hermes Agent](https://github.com/NousResearch/hermes-agent) into a **self-improving robotics trainer**. It adds three Hermes skills that close the loop between robot execution, training data collection, and model improvement — all orchestrated through natural language.

The same self-improvement loop that Hermes uses to get better at coding tasks (via Tinker-Atropos RL) now extends to **physical robot control** via Vision-Language-Action models.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   HERMES AGENT                       │
│  (Reasoning Layer — plans, monitors, orchestrates)   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ vast-gpu  │  │  vla-trainer │  │  robot-loop   │  │
│  │  (skill)  │  │   (skill)    │  │   (skill)     │  │
│  │           │  │              │  │               │  │
│  │ Provision │  │ SmolVLA /    │  │ Deploy model  │  │
│  │ & manage  │  │ GR00T fine-  │  │ Collect traj  │  │
│  │ cloud GPU │  │ tuning on    │  │ Auto-retrain  │  │
│  │ instances │  │ LeRobot data │  │ when improved │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
│                                                      │
├─────────────────────────────────────────────────────┤
│              SIMULATION / HARDWARE                   │
│                                                      │
│  MuJoCo + LeRobot gym_hil    OR    SO-ARM101 + USB  │
│  (Franka Panda sim tasks)          (Physical arm)    │
└─────────────────────────────────────────────────────┘
```

## The Self-Improvement Loop

1. **Deploy** — Hermes loads a VLA checkpoint and runs it in sim (or on hardware)
2. **Collect** — Every rollout is recorded as a LeRobot trajectory (state, action, camera, reward)
3. **Curate** — Hermes filters successful trajectories (reward > threshold)
4. **Train** — Provisions a GPU on Vast.ai and fine-tunes SmolVLA on the new data
5. **Evaluate** — Runs open-loop eval comparing new checkpoint vs. old
6. **Promote** — If new model is better, it becomes the active policy
7. **Repeat** — Scheduled via Hermes cron, runs autonomously

## Skills

### `vast-gpu` — Cloud GPU Infrastructure
Provision, monitor, and teardown GPU instances on Vast.ai through natural language.
- "Spin up an A100 for training" → finds cheapest A100, creates instance, returns SSH access
- "How's my training instance?" → checks status, GPU utilization, cost so far
- "Tear down the GPU" → destroys instance, confirms billing stopped

### `vla-trainer` — VLA Fine-Tuning Pipeline
End-to-end fine-tuning of Vision-Language-Action models.
- Supports SmolVLA (450M, fast) and GR00T N1.5 (3B, powerful)
- Handles data prep, LeRobot format conversion, stats validation
- Runs training on Vast.ai with WandB monitoring
- Open-loop evaluation with trajectory visualization

### `robot-loop` — Continuous Improvement
The autonomous improvement cycle.
- Runs VLA inference in MuJoCo simulation
- Collects and scores trajectories
- Triggers retraining when enough new data accumulates
- A/B tests new checkpoints against current best
- Promotes winners, logs everything

## Quick Start

```bash
# Tell Hermes what you want
"Set up a simulation environment for pick-and-place tasks"

# Hermes installs MuJoCo, LeRobot, configures the Franka Panda env

"Train SmolVLA on the pick-and-place demo dataset"

# Hermes provisions a Vast.ai GPU, downloads data, runs fine-tuning

"Deploy the trained model and start the improvement loop"

# Hermes runs inference in sim, collects trajectories, schedules retraining
```

## Hardware Support (Optional)

For physical deployment on SO-ARM101:
- Leader arm (teleoperation/demo recording)
- Follower arm (autonomous execution)  
- USB cameras (wrist + global view)
- Any Linux machine with USB ports

## Models Supported

| Model | Params | Train Time (A100) | VRAM | Best For |
|-------|--------|-------------------|------|----------|
| SmolVLA | 450M | ~4h / 20k steps | 22GB | Fast iteration, prototyping |
| GR00T N1.5 | 3B | ~4h / 10k steps | 25GB | Production, complex tasks |
| GR00T N1.6 | 3B | ~4h / 10k steps | 25GB | Latest, best performance |

## Cost Estimate

- Vast.ai A100 80GB: ~$1/hr → ~$4 per training run
- Vast.ai A6000 48GB: ~$0.50/hr → ~$2 per training run
- Simulation: Free (local CPU/GPU)
- Physical arm (optional): ~$200-$440

## Project Structure

```
hermes-embodied/
├── README.md
├── skills/
│   ├── vast-gpu/
│   │   └── SKILL.md
│   ├── vla-trainer/
│   │   └── SKILL.md
│   └── robot-loop/
│       └── SKILL.md
├── scripts/
│   ├── setup_sim.py          # MuJoCo + LeRobot environment setup
│   ├── collect_trajectories.py # Run VLA in sim, save rollouts
│   ├── train_smolvla.py      # Fine-tuning wrapper
│   ├── evaluate.py           # Open-loop eval + metrics
│   └── improvement_loop.py   # Full autonomous loop
├── configs/
│   ├── sim_env.json          # Simulation environment config
│   ├── training.yaml         # Training hyperparameters
│   └── vast_instance.yaml    # GPU instance specs
└── docs/
    └── ARCHITECTURE.md
```

## Built With

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — AI agent framework with skills, memory, and RL training
- [LeRobot](https://github.com/huggingface/lerobot) — Open-source robotics framework by Hugging Face
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — 450M parameter Vision-Language-Action model
- [Vast.ai](https://vast.ai) — Affordable cloud GPU rental
- [MuJoCo](https://mujoco.org/) — Physics simulation for robotics
- [WandB](https://wandb.ai) — Experiment tracking

## License

MIT

#!/usr/bin/env python3
"""
Fine-tune SmolVLA on collected trajectories.
Supports local training or remote training on Vast.ai.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def train_local(args):
    """Train SmolVLA on local GPU."""
    
    cmd = [
        "lerobot-train",
        f"--policy.path={args.base_model}",
        f"--dataset.repo_id={args.dataset_id}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
    ]
    
    if args.learning_rate:
        cmd.append(f"--optimizer.lr={args.learning_rate}")
    
    if args.wandb:
        cmd.append("--wandb.enable=true")
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\nTraining complete! Checkpoint saved to: {args.output_dir}")


def train_vastai(args):
    """Train SmolVLA on a Vast.ai GPU instance."""
    
    try:
        from vastai import VastAI
    except ImportError:
        print("ERROR: vastai-sdk not installed. Run: pip install vastai-sdk")
        sys.exit(1)
    
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("ERROR: VAST_API_KEY environment variable not set")
        sys.exit(1)
    
    sdk = VastAI(api_key=api_key)
    
    # Build the onstart script
    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    
    onstart_script = f"""#!/bin/bash
set -e
apt-get update && apt-get install -y git cmake build-essential
pip install wandb accelerate
git clone https://github.com/huggingface/lerobot.git /workspace/lerobot
cd /workspace/lerobot && pip install -e ".[smolvla]"

# Download dataset
pip install huggingface-hub
huggingface-cli download --repo-type dataset {args.dataset_id} --local-dir /workspace/data/{args.dataset_id}

# Signal that setup is complete
touch /workspace/.setup_complete
echo "Setup complete at $(date)"
"""
    
    training_cmd = (
        f"cd /workspace/lerobot && "
        f"lerobot-train "
        f"--policy.path={args.base_model} "
        f"--dataset.repo_id={args.dataset_id} "
        f"--batch_size={args.batch_size} "
        f"--steps={args.steps} "
        f"--output_dir=/workspace/outputs/{args.job_name} "
        f"--policy.device=cuda "
    )
    if args.wandb and wandb_key:
        training_cmd += "--wandb.enable=true "
    
    env_vars = f"-e HF_TOKEN={hf_token}"
    if wandb_key:
        env_vars += f" -e WANDB_API_KEY={wandb_key}"
    
    print(f"\n1. Searching for {args.gpu_type} on Vast.ai...")
    
    # Search for offers
    query = f"gpu_name={args.gpu_type} num_gpus=1 rentable=True reliability>0.95 disk_space>=100"
    if args.max_cost:
        query += f" dph_total<{args.max_cost}"
    
    offers_raw = sdk.search_offers(query=query)
    
    if not offers_raw:
        print("ERROR: No GPU offers found matching criteria")
        sys.exit(1)
    
    # Parse offers
    offers = json.loads(offers_raw) if isinstance(offers_raw, str) else offers_raw
    if not offers:
        print("ERROR: No GPU offers available")
        sys.exit(1)
    
    best_offer = offers[0]  # Already sorted by price
    offer_id = best_offer["id"]
    cost = best_offer.get("dph_total", "unknown")
    print(f"   Found: {args.gpu_type} @ ${cost}/hr (offer {offer_id})")
    
    print(f"\n2. Launching instance...")
    result = sdk.create_instance(
        ID=offer_id,
        image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
        disk=100,
        onstart=onstart_script,
        env=env_vars,
    )
    
    instance_result = json.loads(result) if isinstance(result, str) else result
    instance_id = instance_result.get("new_contract")
    print(f"   Instance created: {instance_id}")
    
    print(f"\n3. Waiting for instance to be ready...")
    ssh_host = None
    ssh_port = None
    
    for attempt in range(60):  # Max 15 minutes
        instances = sdk.show_instances()
        data = json.loads(instances) if isinstance(instances, str) else instances
        for inst in data:
            if inst["id"] == instance_id:
                status = inst.get("actual_status", "unknown")
                if status == "running":
                    ssh_host = inst.get("ssh_host")
                    ssh_port = inst.get("ssh_port")
                    print(f"   Instance running! SSH: {ssh_host}:{ssh_port}")
                    break
                else:
                    print(f"   Status: {status}...", end="\r")
        if ssh_host:
            break
        time.sleep(15)
    
    if not ssh_host:
        print("ERROR: Instance failed to start within 15 minutes")
        sdk.destroy_instance(ID=instance_id)
        sys.exit(1)
    
    # Wait for setup to complete
    print(f"\n4. Waiting for setup to complete (installing deps, downloading data)...")
    import paramiko
    
    for attempt in range(60):  # Max 15 more minutes
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ssh_host, port=int(ssh_port), username="root",
                       key_filename=os.path.expanduser("~/.ssh/id_ed25519"),
                       timeout=10)
            
            stdin, stdout, stderr = ssh.exec_command("test -f /workspace/.setup_complete && echo 'ready'")
            output = stdout.read().decode().strip()
            ssh.close()
            
            if output == "ready":
                print("   Setup complete!")
                break
            else:
                print(f"   Still setting up...", end="\r")
        except Exception:
            print(f"   Connecting...", end="\r")
        time.sleep(15)
    
    print(f"\n5. Starting training...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_host, port=int(ssh_port), username="root",
               key_filename=os.path.expanduser("~/.ssh/id_ed25519"))
    
    stdin, stdout, stderr = ssh.exec_command(training_cmd, timeout=18000)  # 5hr timeout
    
    # Stream output
    for line in iter(stdout.readline, ""):
        print(f"   [train] {line.rstrip()}")
    
    exit_code = stdout.channel.recv_exit_status()
    ssh.close()
    
    if exit_code != 0:
        print(f"ERROR: Training failed with exit code {exit_code}")
        # Don't destroy yet — let user inspect
        print(f"Instance still running for debugging: ssh -p {ssh_port} root@{ssh_host}")
        sys.exit(1)
    
    print(f"\n6. Downloading checkpoint...")
    sdk.copy(
        src=f"{instance_id}:/workspace/outputs/{args.job_name}/",
        dst=args.output_dir,
        identity=os.path.expanduser("~/.ssh/id_ed25519"),
    )
    
    print(f"\n7. Destroying instance...")
    sdk.destroy_instance(ID=instance_id)
    print("   Instance destroyed. Billing stopped.")
    
    print(f"\nTraining complete! Checkpoint saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA")
    parser.add_argument("--mode", choices=["local", "vastai"], default="local",
                        help="Where to train")
    parser.add_argument("--base-model", type=str, default="lerobot/smolvla_base",
                        help="Base model to fine-tune from")
    parser.add_argument("--dataset-id", type=str, required=True,
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Number of training steps")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (default: model default)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/latest",
                        help="Output directory for checkpoint")
    parser.add_argument("--job-name", type=str, default="smolvla_hermes",
                        help="Job name for WandB")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for local training")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging")
    
    # Vast.ai specific
    parser.add_argument("--gpu-type", type=str, default="A100_SXM4",
                        help="GPU type for Vast.ai")
    parser.add_argument("--max-cost", type=float, default=2.0,
                        help="Max $/hr for Vast.ai GPU")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hermes Embodied — SmolVLA Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    
    if args.mode == "local":
        train_local(args)
    else:
        train_vastai(args)


if __name__ == "__main__":
    main()

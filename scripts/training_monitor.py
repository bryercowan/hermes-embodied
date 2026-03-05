#!/usr/bin/env python3
"""
Training monitor — outputs structured status reports.
Designed to be called by Hermes cron and delivered to Telegram.

Reads from:
  - loop_state.json (improvement loop status)
  - training_log.jsonl (all training runs)
  - WandB API (if available)
  - Vast.ai instance status (if running)

Outputs a human-readable status report to stdout.
Hermes cron captures stdout and sends it to Telegram.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(os.environ.get("HERMES_EMBODIED_WORKSPACE", 
                                 os.path.expanduser("~/hermes-embodied")))
LOG_FILE = WORKSPACE / "training_log.jsonl"
STATE_FILE = WORKSPACE / "loop_state.json"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def load_recent_logs(n=5):
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text().strip().split("\n")
    entries = [json.loads(l) for l in lines if l.strip()]
    return entries[-n:]


def format_status_report():
    """Generate a Telegram-friendly status report."""
    
    state = load_state()
    recent_logs = load_recent_logs(5)
    
    lines = []
    lines.append("HERMES EMBODIED — Status Report")
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    if state:
        lines.append(f"Generation: {state.get('generation', 0)}")
        lines.append(f"Current best: {state.get('current_best', 'none')}")
        lines.append(f"Best success rate: {state.get('best_success_rate', 0)*100:.1f}%")
        lines.append(f"Total episodes: {state.get('total_episodes_collected', 0)}")
        lines.append(f"Episodes buffered: {state.get('episodes_since_last_train', 0)}/50")
        lines.append(f"Cycles completed: {state.get('cycle_count', 0)}")
        lines.append("")
        
        # Training history trend
        history = state.get("training_history", [])
        if history:
            lines.append("Training History:")
            for h in history[-5:]:
                gen = h.get("generation", "?")
                sr = h.get("success_rate", 0) * 100
                promoted = "PROMOTED" if h.get("promoted") else "discarded"
                lines.append(f"  Gen {gen}: {sr:.1f}% success ({promoted})")
            lines.append("")
    else:
        lines.append("No loop state found. Run the improvement loop first.")
        lines.append("")
    
    if recent_logs:
        lines.append("Recent Training Runs:")
        for log in recent_logs:
            ts = log.get("timestamp", "?")[:16]
            event = log.get("event", "?")
            details = log.get("details", "")
            lines.append(f"  [{ts}] {event}: {details}")
        lines.append("")
    
    # Check for active Vast.ai instance
    try:
        from vastai import VastAI
        api_key = os.environ.get("VAST_API_KEY")
        if api_key:
            sdk = VastAI(api_key=api_key)
            instances = sdk.show_instances()
            data = json.loads(instances) if isinstance(instances, str) else instances
            active = [i for i in data if i.get("actual_status") == "running"]
            if active:
                lines.append(f"Active GPU Instances: {len(active)}")
                for inst in active:
                    gpu = inst.get("gpu_name", "?")
                    cost = inst.get("dph_total", 0)
                    lines.append(f"  {gpu} @ ${cost:.2f}/hr")
                lines.append("")
    except Exception:
        pass
    
    return "\n".join(lines)


def log_event(event, details="", metrics=None):
    """Append a training event to the log file."""
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "details": details,
    }
    if metrics:
        entry["metrics"] = metrics
    
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def format_training_complete_report(metrics):
    """Format a report for when training finishes."""
    
    lines = []
    lines.append("TRAINING COMPLETE")
    lines.append("")
    lines.append(f"Steps: {metrics.get('steps', '?')}")
    lines.append(f"Final loss: {metrics.get('final_loss', '?')}")
    lines.append(f"Duration: {metrics.get('duration_minutes', '?')} minutes")
    lines.append(f"Cost: ${metrics.get('cost_usd', '?'):.2f}")
    lines.append("")
    
    if "eval_before" in metrics and "eval_after" in metrics:
        before = metrics["eval_before"]
        after = metrics["eval_after"]
        lines.append("A/B Test Results:")
        lines.append(f"  Before: {before.get('success_rate', 0)*100:.1f}% success")
        lines.append(f"  After:  {after.get('success_rate', 0)*100:.1f}% success")
        delta = (after.get("success_rate", 0) - before.get("success_rate", 0)) * 100
        lines.append(f"  Delta:  {delta:+.1f}%")
        lines.append("")
        
        if delta > 5:
            lines.append("NEW MODEL PROMOTED")
        else:
            lines.append("No significant improvement. Keeping current model.")
    
    if metrics.get("wandb_url"):
        lines.append(f"WandB: {metrics['wandb_url']}")
    
    return "\n".join(lines)


def format_cycle_report(state, episodes_collected, success_rate, good_episodes):
    """Format a report for a single improvement cycle."""
    
    lines = []
    lines.append(f"IMPROVEMENT CYCLE #{state.get('cycle_count', '?')}")
    lines.append("")
    lines.append(f"Episodes collected: {episodes_collected}")
    lines.append(f"Current success rate: {success_rate*100:.1f}%")
    lines.append(f"Good episodes kept: {good_episodes}")
    lines.append(f"Buffer: {state.get('episodes_since_last_train', 0)}/50")
    
    if state.get('episodes_since_last_train', 0) >= 50:
        lines.append("")
        lines.append("Threshold reached! Training will start.")
    else:
        need = 50 - state.get('episodes_since_last_train', 0)
        lines.append(f"Need {need} more episodes before training.")
    
    lines.append("")
    lines.append(f"Generation: {state.get('generation', 0)}")
    lines.append(f"Best: {state.get('best_success_rate', 0)*100:.1f}%")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", nargs=2, metavar=("EVENT", "DETAILS"),
                        help="Log an event")
    parser.add_argument("--report", choices=["status", "training", "cycle"],
                        default="status",
                        help="Report type to generate")
    
    args = parser.parse_args()
    
    if args.log:
        log_event(args.log[0], args.log[1])
        print(f"Logged: {args.log[0]}")
    else:
        print(format_status_report())

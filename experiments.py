#!/usr/bin/env python
"""
Runs three hyper-parameter sweeps (shared, DQN-specific, PPO-specific) on three warehouse levels
and prints evaluation metrics to the console.
"""

import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

# ───────────────────────── constants ─────────────────────────── #
ROOT         = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"
RESULTS_DIR  = ROOT / "results"
EVAL_NAME    = "eval_metrics.csv"

LEVEL_FILES  = [
    "warehouse_level_1",
    "warehouse_level_2",
    "warehouse_level_3",
]
LEVEL_NAMES = ["level1", "level2", "level3"]

RESULT_KEYS  = ["mean_return", "success_rate", "steps", "time_min"]

# sweep values
LR_VALUES      = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
BATCH_VALUES   = [64, 128, 256, 512, 1024]
GAMMA_VALUES   = [0.90, 0.95, 0.99, 0.995, 0.999]
HIDDEN_VALUES  = [64, 128, 256, 512, 1024]

BUFFER_VALUES  = [5000, 10000, 20000, 50000, 100000]  # DQN only
EPS_DECAY_VALS = [0.95, 0.97, 0.98, 0.99, 0.995]     # DQN only
TAU_VALS       = [0.005, 0.01, 0.02, 0.05, 0.1]       # DQN only

CLIP_EPS_VALS  = [0.1, 0.2, 0.3, 0.4, 0.5]            # PPO only
ENTROPY_VALS   = [0.0, 0.01, 0.05, 0.1, 0.4]          # PPO only
LAMDA_VALS     = [0.90, 0.93, 0.95, 0.97, 0.99]       # PPO only
PPO_EPOCH_VALS = [2, 4, 6, 8, 10]                    # PPO only

# ───────────────────────── helpers ───────────────────────────── #
def parse_eval(path: Path) -> dict:
    """Read evaluation metrics from a csv and return a dict of floats."""
    with path.open() as f:
        row = next(csv.DictReader(f))
        return {k: float(row[k]) for k in RESULT_KEYS if k in row}


def run_cfg(agent: str, part: str, params: dict, level_file: str, level_name: str) -> dict:
    """Run a single configuration, parse metrics, and return a result row."""
    out_row = {
        "agent": agent,
        "part": part,
        "level": level_name,
        **params,
    }
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--agent", agent,
        "--level_file", level_file,
        "--num_episodes", "300",
        "--max_steps", "3000",
        "--test-gui"
    ] + [f"--{k} {v}" for k, v in params.items()]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    out_row["time_min"] = (time.time() - t0) / 60.0

    # locate the latest eval_metrics.csv for this agent
    eval_dir = RESULTS_DIR / agent
    eval_files = list(eval_dir.rglob(EVAL_NAME))
    if not eval_files:
        print(f"No {EVAL_NAME} found for agent '{agent}' in '{eval_dir}'. Skipping.")
        return None
    eval_csv = max(eval_files, key=lambda p: p.stat().st_mtime)
    metrics  = parse_eval(eval_csv)
    out_row.update(metrics)
    return out_row

# ──────────────────── sweep generators ───────────────────────── #
def part1_shared():
    for agent in ["dqn", "ppo"]:
        for lr in LR_VALUES:
            yield agent, {"lr": lr}, "part1"
        for batch in BATCH_VALUES:
            yield agent, {"batch": batch}, "part1"
        for gamma in GAMMA_VALUES:
            yield agent, {"gamma": gamma}, "part1"
        for hidden in HIDDEN_VALUES:
            yield agent, {"hidden_dim": hidden}, "part1"

def part2_dqn():
    agent = "dqn"
    for eps in EPS_DECAY_VALS:
        yield agent, {"epsilon_decay": eps}, "part2"
    for tau in TAU_VALS:
        yield agent, {"tau": tau}, "part2"
    for buf in BUFFER_VALUES:
        yield agent, {"buffer": buf}, "part2"

def part3_ppo():
    agent = "ppo"
    for ce in CLIP_EPS_VALS:
        yield agent, {"clip_eps": ce}, "part3"
    for ent in ENTROPY_VALS:
        yield agent, {"entropy_coeff": ent}, "part3"
    for lam in LAMDA_VALS:
        yield agent, {"lamda": lam}, "part3"
    for pe in PPO_EPOCH_VALS:
        yield agent, {"ppo_epochs": pe}, "part3"

if __name__ == "__main__":
    try:
        sweeps = itertools.chain(
            # Uncomment the parts you want to run:
            # part1_shared(),
            # part2_dqn(),
            # part3_ppo(),
        )
        for agent, params, part in sweeps:
            for level_file, level_name in zip(LEVEL_FILES, LEVEL_NAMES):
                print(f"\n>>> {part} | {agent} | {params} | {level_name}")
                try:
                    row = run_cfg(agent, part, params, level_file, level_name)
                    if row:
                        print("Result:", row)
                except subprocess.CalledProcessError as e:
                    print(f"Failed (exit {e.returncode}) – continuing…")
                except Exception as e:
                    print(f"Unexpected error: {e} – continuing…")
    except KeyboardInterrupt:
        print("\nInterrupted – partial results printed.")

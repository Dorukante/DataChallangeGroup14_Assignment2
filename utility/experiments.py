"""
experiments.py
Runs hyper-parameter sweeps (shared, DQN-specific, PPO-specific) on three warehouse levels,
and appends results (agent, level, parameter, runtime, collisions, goals) to a CSV under ./results.
"""

import csv
import itertools
import subprocess
import sys
import time
import re
from pathlib import Path

# ───────────────────────── constants ─────────────────────────── #
ROOT         = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT / "train.py"
RESULTS_DIR  = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV   = RESULTS_DIR / "experiments_results.csv"

LEVEL_FILES  = [
    "warehouse_level_1",
    "warehouse_level_2",
    "warehouse_level_3",
]
LEVEL_NAMES = ["level1", "level2", "level3"]

# sweep values (shared)
LR_VALUES      = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
BATCH_VALUES   = [64, 128, 256, 512, 1024]
GAMMA_VALUES   = [0.90, 0.95, 0.99, 0.995, 0.999]
HIDDEN_VALUES  = [64, 128, 256, 512, 1024]

# DQN-specific
BUFFER_VALUES  = [5000, 10000, 20000, 50000, 100000]
EPS_DECAY_VALS = [0.95, 0.97, 0.98, 0.99, 0.995]
TAU_VALS       = [0.005, 0.01, 0.02, 0.05, 0.1]

# PPO-specific
CLIP_EPS_VALS  = [0.1, 0.2, 0.3, 0.4, 0.5]
ENTROPY_VALS   = [0.0, 0.01, 0.05, 0.1, 0.4]
LAMDA_VALS     = [0.90, 0.93, 0.95, 0.97, 0.99]
PPO_EPOCH_VALS = [2, 4, 6, 8, 10]

# ───────────────────────── helpers ───────────────────────────── #

def append_to_csv(row: dict):
    """
    Append a result row to OUTPUT_CSV, adding header if file is new.

    Args:
        row (dict): Dictionary of result fields to write.
    """
    write_header = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_eval_txt(path: Path) -> dict:
    """
    Parse evaluation .txt file to extract collision_count and goals_reached.

    Args:
        path (Path): Path to the evaluation results text file.

    Returns:
        dict: {'collision_count': int, 'goals_reached': int}
    """
    text = path.read_text()
    coll = re.search(r"collision_count:\s*(\d+)", text)
    goals = re.search(r"goals_reached:\s*(\d+)", text)
    return {
        'collision_count': int(coll.group(1)) if coll else None,
        'goals_reached': int(goals.group(1)) if goals else None
    }


def run_cfg(agent: str, part: str, params: dict, level_file: str, level_name: str) -> dict:
    """
    Run one configuration, parse collisions and goals, and return a result dict.

    Args:
        agent (str): 'dqn' or 'ppo'.
        part (str): 'part1', 'part2', or 'part3'.
        params (dict): Single hyperparameter for this run.
        level_file (str): Path to level file.
        level_name (str): Descriptive level name.

    Returns:
        dict: {'agent', 'level', 'param_key', 'param_val', 'time_min', 'collision_count', 'goals_reached'}
    """
    metric_row = {
        "agent": agent,
        "level": level_name,
        "param_key": next(iter(params.keys())),
        "param_val": next(iter(params.values()))
    }
    # Construct command
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--agent", agent,
        "--level_file", level_file,
        "--num_episodes", "300",
        "--max_steps", "3000",
        "--test-gui"
    ]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    # Execute
    start = time.time()
    subprocess.run(cmd, check=True)
    metric_row["time_min"] = (time.time() - start) / 60.0

    # Locate the latest evaluation text file
    eval_dir = RESULTS_DIR / agent
    txt_files = list(eval_dir.rglob(f"*evaluation_results*{level_file}*.txt"))
    if txt_files:
        latest_txt = max(txt_files, key=lambda p: p.stat().st_mtime)
        eval_metrics = parse_eval_txt(latest_txt)
        metric_row.update(eval_metrics)
    else:
        metric_row.update({'collision_count': None, 'goals_reached': None})

    return metric_row

# ──────────────────── sweep generators ───────────────────────── #

def part1_shared():
    """Shared hyperparameter sweeps for DQN and PPO: lr, batch, gamma, hidden_dim."""
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
    """DQN-only sweeps: epsilon_decay, tau, buffer."""
    agent = "dqn"
    for eps in EPS_DECAY_VALS:
        yield agent, {"epsilon_decay": eps}, "part2"
    for tau in TAU_VALS:
        yield agent, {"tau": tau}, "part2"
    for buf in BUFFER_VALUES:
        yield agent, {"buffer": buf}, "part2"

def part3_ppo():
    """PPO-only sweeps: clip_eps, entropy_coeff, lamda, ppo_epochs."""
    agent = "ppo"
    for ce in CLIP_EPS_VALS:
        yield agent, {"clip_eps": ce}, "part3"
    for ent in ENTROPY_VALS:
        yield agent, {"entropy_coeff": ent}, "part3"
    for lam in LAMDA_VALS:
        yield agent, {"lamda": lam}, "part3"
    for pe in PPO_EPOCH_VALS:
        yield agent, {"ppo_epochs": pe}, "part3"

# ─────────────────────────── main ────────────────────────────── #
if __name__ == "__main__":
    sweeps = itertools.chain(
        part1_shared(),
        part2_dqn(),
        part3_ppo()
    )
    for agent, params, part in sweeps:
        for level_file, level_name in zip(LEVEL_FILES, LEVEL_NAMES):
            try:
                rec = run_cfg(agent, part, params, level_file, level_name)
                append_to_csv(rec)
            except subprocess.CalledProcessError:
                continue
            except Exception:
                continue

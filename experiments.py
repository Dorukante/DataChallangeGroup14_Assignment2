#!/usr/bin/env python
"""
experiment_runner.py  – 23 June 2025, with per-level results, part-wise CSVs.
Runs three hyper-parameter sweeps (shared, DQN-specific, PPO-specific) on three warehouse levels,
and aggregates evaluation metrics into results/part#/part#.csv.
"""

import csv, itertools, os, subprocess, sys, time
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

# Per-part CSV files
PART_CSVS = {
    "part1": RESULTS_DIR / "part1" / "part1.csv",
    "part2": RESULTS_DIR / "part2" / "part2.csv",
    "part3": RESULTS_DIR / "part3" / "part3.csv",
}

RESULT_KEYS  = ["mean_return", "success_rate", "steps", "time_min"]

# sweep values – same as before
LR_VALUES      = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
BATCH_VALUES   = [64, 128, 256, 512, 1024]
GAMMA_VALUES   = [0.90, 0.95, 0.99, 0.995, 0.999]
HIDDEN_VALUES  = [64, 128, 256, 512, 1024]
BUFFER_VALUES  = [5000, 10000, 20000, 50000, 100000]

EPS_DECAY_VALS = [0.95, 0.97, 0.98, 0.99, 0.995]   # DQN only
TAU_VALS       = [0.005, 0.01, 0.02, 0.05, 0.1]    # DQN only

CLIP_EPS_VALS  = [0.1, 0.2, 0.3, 0.4, 0.5]         # PPO only
ENTROPY_VALS   = [0.0, 0.01, 0.05, 0.1, 0.4]        # PPO only
LAMDA_VALS     = [0.90, 0.93, 0.95, 0.97, 0.99]     # PPO only
PPO_EPOCH_VALS = [2, 4, 6, 8, 10]                  # PPO only

# ───────────────────────── helpers ───────────────────────────── #
def parse_eval(path: Path) -> dict:
    with path.open() as f:
        row = next(csv.DictReader(f))
        return {k: float(row[k]) for k in RESULT_KEYS if k in row}

def save_row(row: dict, csv_file: Path):
    write_hdr = not csv_file.exists()
    with csv_file.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_hdr:
            w.writeheader()
        w.writerow(row)

def run_cfg(agent: str, part: str, params: dict, level_file: str, level_name: str):
    # Only include swept params and common fields
    out_row = {
        "agent": agent,
        "part": part,
        "level": level_name,
        **params,
    }
    # Assemble command
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

    t0 = time.time()
    subprocess.run(cmd, check=True)
    runtime = (time.time() - t0) / 60.0

    # Find latest eval_metrics.csv in agent's results subdir
    eval_dir = RESULTS_DIR / agent
    eval_csv = max(list(eval_dir.rglob(EVAL_NAME)), key=lambda p: p.stat().st_mtime)
    metrics  = parse_eval(eval_csv)
    out_row.update(metrics)
    out_row["time_min"] = runtime
    return out_row

# ──────────────────── sweep generators ───────────────────────── #
def part1_shared():
    # Only sweep SHARED params: lr, batch, gamma, hidden_dim
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

# ─────────────────────────── main ────────────────────────────── #
if __name__ == "__main__":
    try:
        for agent, params, part in itertools.chain(
            # comment the part you don't want to run
            part1_shared(),
            part2_dqn(),
            part3_ppo()
        ):
            for level_file, level_name in zip(LEVEL_FILES, LEVEL_NAMES):
                print(f"\n>>> {part} | {agent} | {params} | {level_name}")
                try:
                    row = run_cfg(agent, part, params, level_file, level_name)
                    part_csv = PART_CSVS[part]
                    part_csv.parent.mkdir(exist_ok=True, parents=True)
                    save_row(row, part_csv)
                    print(f"✓ appended to {part_csv}")
                except subprocess.CalledProcessError as e:
                    print(f"✗ failed (exit {e.returncode}) – continuing…")
                except Exception as e:
                    print(f"✗ unexpected error: {e} – continuing…")
    except KeyboardInterrupt:
        print("\nInterrupted – partial results kept.")

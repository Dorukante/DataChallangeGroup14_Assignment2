#!/usr/bin/env python
"""
experiment_runner.py  –  final version per 23 June 2025 request
Runs three hyper-parameter sweeps (shared, DQN-specific, PPO-specific) and
aggregates evaluation metrics into all_results.csv.

Assumptions
-----------
• train.py must write evaluation CSV to results/<agent>/.../eval_metrics.csv
  with at least: mean_return, success_rate, steps, time_min  (wall-clock mins).
• train.py accepts every CLI flag we pass (lr, batch, gamma, etc.).
• No --seed flag is sent; each run uses whatever randomness train.py chooses.
"""

import csv, itertools, os, subprocess, sys, time
from pathlib import Path

# ───────────────────────── constants ─────────────────────────── #
ROOT         = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"
RESULTS_DIR  = ROOT / "results"
EVAL_NAME    = "eval_metrics.csv"          # produced by Helper.save_eval_results
AGG_CSV      = ROOT / "all_results.csv"
RESULT_KEYS  = ["mean_return", "success_rate", "steps", "time_min"]

# sweep values – updated ---------------------------------------------------- #
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
def newest_eval(dir_root: Path) -> Path:
    files = list(dir_root.rglob(EVAL_NAME))
    if not files:
        raise FileNotFoundError(f"No {EVAL_NAME} under {dir_root}")
    return max(files, key=lambda p: p.stat().st_mtime)

def parse_eval(path: Path) -> dict:
    with path.open() as f:
        row = next(csv.DictReader(f))
        return {k: float(row[k]) for k in RESULT_KEYS if k in row}

def row_exists(id_fields: dict) -> bool:
    if not AGG_CSV.exists():
        return False
    with AGG_CSV.open() as f:
        rdr = csv.DictReader(f)
        return any(all(r.get(k) == str(v) for k, v in id_fields.items()) for r in rdr)

def save_row(row: dict):
    write_hdr = not AGG_CSV.exists()
    with AGG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_hdr:
            w.writeheader()
        w.writerow(row)

def run_cfg(agent: str, part: str, params: dict):
    base_id = {"part": part, "agent": agent, **params}
    if row_exists(base_id):
        print(f"• Skipping (done) {base_id}")
        return

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--agent", agent,
        "--level_file", "warehouse_level_1",
        "--num_episodes", "300",
        "--max_steps", "3000",
        "--test-gui"
    ]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    runtime = (time.time() - t0) / 60.0

    eval_csv = newest_eval(RESULTS_DIR / agent)
    metrics  = parse_eval(eval_csv)
    metrics["time_min"] = runtime
    save_row({**base_id, **metrics})
    print("✓ appended to all_results.csv")

# ──────────────────── sweep generators ───────────────────────── #
def part1_shared():
    for agent in ["dqn", "ppo"]:
        for lr in LR_VALUES:
            yield agent, "part1_lr", {"lr": lr}
        for batch in BATCH_VALUES:
            yield agent, "part1_batch", {"batch": batch}
        for gamma in GAMMA_VALUES:
            yield agent, "part1_gamma", {"gamma": gamma}
        for hidden in HIDDEN_VALUES:
            yield agent, "part1_hidden", {"hidden_dim": hidden}
        for buf in BUFFER_VALUES:
            yield agent, "part1_buffer", {"buffer": buf}

def part2_dqn():
    agent = "dqn"
    for eps in EPS_DECAY_VALS:
        yield agent, "part2_eps_decay", {"epsilon_decay": eps}
    for tau in TAU_VALS:
        yield agent, "part2_tau", {"tau": tau}

def part3_ppo():
    agent = "ppo"
    for ce in CLIP_EPS_VALS:
        yield agent, "part3_clip_eps", {"clip_eps": ce}
    for ent in ENTROPY_VALS:
        yield agent, "part3_entropy", {"entropy_coeff": ent}
    for lam in LAMDA_VALS:
        yield agent, "part3_lamda", {"lamda": lam}
    for pe in PPO_EPOCH_VALS:
        yield agent, "part3_epochs", {"ppo_epochs": pe}

# ─────────────────────────── main ────────────────────────────── #
if __name__ == "__main__":
    try:
        for agent, part, params in itertools.chain(
            part1_shared(), part2_dqn(), part3_ppo()
        ):
            print(f"\n>>> {part} | {agent} | {params}")
            try:
                run_cfg(agent, part, params)
            except subprocess.CalledProcessError as e:
                print(f"✗ failed (exit {e.returncode}) – continuing…")
            except Exception as e:
                print(f"✗ unexpected error: {e} – continuing…")
    except KeyboardInterrupt:
        print("\nInterrupted – partial results kept.")

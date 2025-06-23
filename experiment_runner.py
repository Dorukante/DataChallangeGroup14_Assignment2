#!/usr/bin/env python
"""
experiment_runner.py  —  *Part-2-only* edition
Runs just the DQN-specific hyper-parameter sweep:
    • epsilon_decay  ∈ {0.95, 0.98, 0.995}
    • tau            ∈ {0.005, 0.02, 0.05}

The script looks for evaluation CSVs under results/dqn/**/eval*.csv
and appends one row per successful run to all_results.csv.
"""

import csv, itertools, os, subprocess, sys, time, re
from pathlib import Path

# ───────── paths & constants ───────── #
ROOT         = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train.py"
RESULTS_DIR  = ROOT / "results"
AGG_CSV      = ROOT / "all_results.csv"
EVAL_GLOB    = re.compile(r"eval.*\.csv", re.I)
METRIC_KEYS  = ["mean_return", "success_rate", "steps", "time_min"]

# ───────── parameter values (Part 2) ───────── #
EPS_DECAY_VALS = [0.95, 0.98, 0.995]
TAU_VALS       = [0.005, 0.02, 0.05]

# ───────── helper functions ───────── #
def newest_eval(agent_root: Path):
    files = [p for p in agent_root.rglob("*.csv") if EVAL_GLOB.search(p.name)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def parse_eval(path: Path) -> dict:
    with path.open() as f:
        row = next(csv.DictReader(f))
        return {k: float(row[k]) for k in METRIC_KEYS if k in row}

def row_done(id_fields: dict) -> bool:
    if not AGG_CSV.exists():
        return False
    with AGG_CSV.open() as f:
        rdr = csv.DictReader(f)
        return any(all(r.get(k) == str(v) for k, v in id_fields.items()) for r in rdr)

def append_row(row: dict):
    hdr_needed = not AGG_CSV.exists()
    with AGG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if hdr_needed:
            w.writeheader()
        w.writerow(row)

def run_cfg(params: dict):
    row_id = {"part": 2, "agent": "dqn", **params}
    if row_done(row_id):
        print(f"• skipping (done) {row_id}")
        return

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--agent", "dqn",
        "--level_file", "warehouse_level_3",
        "--num_episodes", "300",
        "--max_steps", "3000",
        "--test-gui"
    ]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ train.py exit {e.returncode} — skipped")
        return

    runtime = (time.time() - t0) / 60.0
    eval_csv = newest_eval(RESULTS_DIR / "dqn")
    if not eval_csv:
        print("✗ no evaluation CSV found — skipped")
        return

    metrics = parse_eval(eval_csv)
    metrics["time_min"] = runtime
    append_row({**row_id, **metrics})
    print("✓ logged")

# ───────── main loop ───────── #
if __name__ == "__main__":
    try:
        sweep = itertools.chain(
            ({"epsilon_decay": v} for v in EPS_DECAY_VALS),
            ({"tau": v}          for v in TAU_VALS)
        )
        for params in sweep:
            print(f"\n>>> Part-2 | dqn | {params}")
            run_cfg(params)
    except KeyboardInterrupt:
        print("\nInterrupted — partial results kept.")

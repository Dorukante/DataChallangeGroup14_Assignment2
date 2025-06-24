#!/usr/bin/env python3
"""
exp_part2.py  –  DQN ε-decay / τ / buffer sweeps on two levels
----------------------------------------------------------------
Creates  (3 ε) + (3 τ) + (3 buffer)  × 2 levels = 18 runs.
Outputs:
    - logs_part2/*.log              (console for each run)
    - part2_results.csv             (aggregated final metrics)
"""

import subprocess, json, csv, datetime, pathlib

# ---------------------------------------------------------------------------
TRAIN_SCRIPT = "train.py"
RESULT_DIR   = pathlib.Path("results")          # produced by train.py
LOG_DIR      = pathlib.Path("logs_part2")
CSV_OUT      = pathlib.Path("part2_results.csv")
LOG_DIR.mkdir(exist_ok=True)

LEVELS = ["warehouse_level_1", "warehouse_level_2"]
AGENT  = "dqn"                                  # Part-2 is DQN-specific

# single-parameter grids
GRID = dict(
    epsilon_decay = [0.95, 0.98, 0.995],
    tau           = [0.005, 0.01, 0.02],
    buffer        = [5_000, 20_000, 100_000],
)

FLAG = dict(epsilon_decay="--epsilon_decay",
            tau           ="--tau",
            buffer        ="--buffer")

BASE_ARGS = [
    "--num_episodes", "300",
    "--max_steps",    "3000",
    "--gamma",        "0.99",
    "--lr",           "3e-4",
    "--batch",        "128",
    "--hidden_dim",   "128",
    "--test-gui"
]

# ---------------------------------------------------------------------------
def run_once(level, param_key, val):
    """Launch train.py with the specified single-parameter change."""
    cli = ["python", TRAIN_SCRIPT,
           "--level_file", level,
           "--agent",     AGENT,
           *BASE_ARGS,
           FLAG[param_key], str(val)]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_p = LOG_DIR / f"p2_{param_key}_{val}_{level}_{stamp}.log"
    with log_p.open("w") as lf:
        subprocess.run(cli, stdout=lf, stderr=subprocess.STDOUT)

    # pick the most recent *training_metrics*.json written by Helper
    json_files = sorted(RESULT_DIR.rglob("*training_metrics*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    if not json_files:
        print(f"✗ metrics JSON missing for {param_key}={val} on {level}")
        return None
    metrics_path = json_files[0]
    data  = json.loads(metrics_path.read_text())
    last  = data[-1]                                   # final episode entry

    return dict(
        level_file   = level,
        param        = param_key,
        value        = val,
        total_time   = last.get("total_time"),
        collisions   = last.get("collision_count"),
        goals        = last.get("goals_reached"),
        results_file = metrics_path.name,
    )

# ---------------------------------------------------------------------------
rows = []
for level in LEVELS:
    for param_key, values in GRID.items():
        for val in values:
            print(f">>> {level} | {param_key} = {val}")
            rec = run_once(level, param_key, val)
            if rec:
                rows.append(rec)

# write CSV
fieldnames = ["level_file","param","value",
              "total_time","collisions","goals","results_file"]

with CSV_OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames)
    w.writeheader();  w.writerows(rows)

print(f"\n✓ Part-2 sweep complete — {len(rows)} runs logged to {CSV_OUT}")

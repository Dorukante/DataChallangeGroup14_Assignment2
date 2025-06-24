#!/usr/bin/env python3
"""
exp_part1.py – Part-1 sweeps on L1 & L2, DQN + PPO
--------------------------------------------------
Parameters varied (one-at-a-time, 3 values each)
    lr       : 1e-4 , 3e-4 , 1e-3
    batch    : 64 , 256 , 1024
    gamma    : 0.95 , 0.99 , 0.999
    hidden   : 64 , 256 , 512

Levels      : warehouse_level_1 , warehouse_level_2
Agents      : dqn , ppo
Totals      : 4 sweeps × 3 values × 2 levels × 2 agents = 48 runs
"""
import subprocess, json, csv, datetime, pathlib, time
from itertools import product

# locations
TRAIN_SCRIPT = "train.py"
RESULT_DIR   = pathlib.Path("results")
LOG_DIR      = pathlib.Path("logs_part1"); LOG_DIR.mkdir(exist_ok=True)
CSV_OUT      = pathlib.Path("part1_results.csv")

# sweep grids
GRID   = dict(
    lr     =[1e-4, 3e-4, 1e-3],
    batch  =[64, 256, 1024],
    gamma  =[0.95, 0.99, 0.999],
    hidden =[64, 256, 512],
)
LEVELS = ["warehouse_level_1", "warehouse_level_2"]
AGENTS = ["dqn", "ppo"]          # ← now includes PPO

# map to CLI flags
FLAG = dict(lr="--lr", batch="--batch",
            gamma="--gamma", hidden="--hidden_dim")

BASE_ARGS = [
    "--num_episodes", "300",
    "--max_steps",    "3000",
    "--test-gui"                  # headless for speed
]

def run_cfg(level, agent, sweep, key, val):
    cli = ["python", TRAIN_SCRIPT,
           "--level_file", level,
           "--agent", agent,
           *BASE_ARGS,
           FLAG[key], str(val)]

    ts  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = LOG_DIR / f"{sweep}_{agent}_{level}_{val}_{ts}.log"
    with log.open("w") as lf:
        try:
            subprocess.run(cli, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=3600, check=False)
        except subprocess.TimeoutExpired:
            print(f"✗ timeout: {cli}")
            return None

    # newest training_metrics JSON
    jfiles = sorted(RESULT_DIR.rglob("*training_metrics*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not jfiles:
        print(f"✗ no JSON for {level}/{agent}/{key}={val}")
        return None

    data, last = json.loads(jfiles[0].read_text()), None
    if isinstance(data, list) and data:
        last = data[-1]

    return dict(
        part=sweep, level_file=level, agent=agent,
        param=key, value=val,
        total_time   = last.get("total_time")    if last else None,
        collisions   = last.get("collision_count") if last else None,
        goals        = last.get("goals_reached")   if last else None,
        results_file = jfiles[0].name
    )

# main loop with ETA
rows, total = [], len(AGENTS)*len(LEVELS)*sum(len(v) for v in GRID.values())
start = time.time()

for key, vals in GRID.items():
    sweep = f"p1_{key}"
    for val, level, agent in product(vals, LEVELS, AGENTS):
        idx = len(rows)+1
        print(f"[{idx:02}/{total}] {sweep} | {agent} | {level} | {key}={val}")
        rec = run_cfg(level, agent, sweep, key, val)
        if rec:
            rows.append(rec)
        eta = ((time.time()-start)/(idx)) * (total-idx)
        print(f"    ETA ≈ {eta/60:5.1f} min")

# write CSV
hdr = ["part","level_file","agent","param","value",
       "total_time","collisions","goals","results_file"]
CSV_OUT.write_text("")
with CSV_OUT.open("w", newline="") as f:
    csv.DictWriter(f, hdr).writerows([dict(zip(hdr,hdr))])  # header
    csv.DictWriter(f, hdr).writerows(rows)

print(f"\n✓ {len(rows)} runs complete – results → {CSV_OUT}")

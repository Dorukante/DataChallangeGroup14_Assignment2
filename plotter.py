import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- load data --------------------------------------------------- #
csv_path = "part1_variation_aggregates.csv"
df = pd.read_csv(csv_path)

# ---- make a plot for each swept parameter ------------------------ #
param_cols = ["lr", "batch", "gamma", "hidden_dim", "buffer"]

for param in param_cols:
    sub = df[df[param].notna()].copy()
    if sub.empty:
        continue

    sub.sort_values(by=param, inplace=True)

    plt.figure()
    for agent, g in sub.groupby("agent"):
        plt.plot(g[param], g["mean_reward"], marker="o", label=agent)

    plt.xlabel(param)
    plt.ylabel("Mean reward per step")
    plt.title(f"Part-1 sweep: {param}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---- simple console preview -------------------------------------- #
print("\n=== Part-1 aggregates used for plots ===")
print(df.head())

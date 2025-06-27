#!/usr/bin/env python3
"""
ppo_eval.py

Reads PPO-specific results from results/PPO/PPO_results.csv and generates heatmap
visualizations for goals reached across pairs of hyperparameters.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────── constants ─────────────────────────── #
# Data file is located alongside this script in the results/PPO directory
CSV_PATH = Path(__file__).resolve().parent / "PPO_results.csv"

# Define parameter pairs and chart titles
param_plots = [
    ('lam', 'clip_eps', r'$\lambda$ vs clip_eps'),
    ('entropy', 'ppo_epochs', 'entropy vs PPO epochs'),
    ('lam', 'ppo_epochs', r'$\lambda$ vs PPO epochs'),
    ('entropy', 'clip_eps', 'entropy vs clip_eps')
]

# ─────────────────────────── main ────────────────────────────── #
def main():
    # Load the data
    df = pd.read_csv(CSV_PATH)

    # Ensure required columns are present
    required = {'goals_reached', 'lam', 'entropy', 'clip_eps', 'ppo_epochs'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Generate subplots for each parameter pair
    fig, axes = plt.subplots(1, len(param_plots), figsize=(5 * len(param_plots), 5))
    for ax, (row_param, col_param, title) in zip(axes, param_plots):
        # pivot for heatmap
        pivot = df.pivot_table(
            values='goals_reached',
            index=row_param,
            columns=col_param,
            aggfunc='mean'
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            ax=ax,
            cbar=False
        )
        ax.set_title(title)
        ax.set_xlabel(col_param)
        ax.set_ylabel(row_param)

    # Single shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap="Blues",
        norm=plt.Normalize(
            vmin=df['goals_reached'].min(),
            vmax=df['goals_reached'].max()
        )
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Goals Reached')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

if __name__ == '__main__':
    main()

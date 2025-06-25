import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("part2_metrics.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Figure 1 – heat-maps (same as before)
# ─────────────────────────────────────────────────────────────────────────────
piv_time = df.pivot(index="epsilon_decay", columns="tau", values="total_time")
piv_coll = df.pivot(index="epsilon_decay", columns="tau", values="collision_count")

epses, taus = piv_time.index.tolist(), piv_time.columns.tolist()
vmin_t, vmax_t = piv_time.min().min(), piv_time.max().max()
vmin_c, vmax_c = piv_coll.min().min(), piv_coll.max().max()

plt.rcParams.update({"font.size": 10})
fig1, ax = plt.subplots(1, 2, figsize=(8.5, 3.8))

# (A) total-time
im1 = ax[0].imshow(piv_time, cmap="Blues_r", vmin=vmin_t, vmax=vmax_t)
ax[0].set_title("Total time (s)")
ax[0].set_xlabel("τ")
ax[0].set_ylabel("ε-decay")
ax[0].set_xticks(range(len(taus)), taus)
ax[0].set_yticks(range(len(epses)), epses)
for i, eps in enumerate(epses):
    for j, tau in enumerate(taus):
        val = piv_time.iloc[i, j]
        if pd.notna(val):
            ax[0].text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=8)
fig1.colorbar(im1, ax=ax[0], shrink=.8, pad=.02, label="seconds")

# (B) collisions
im2 = ax[1].imshow(piv_coll, cmap="Reds_r", vmin=vmin_c, vmax=vmax_c)
ax[1].set_title("Collision count")
ax[1].set_xlabel("τ")
ax[1].set_ylabel("ε-decay")
ax[1].set_xticks(range(len(taus)), taus)
ax[1].set_yticks(range(len(epses)), epses)
for i, eps in enumerate(epses):
    for j, tau in enumerate(taus):
        val = piv_coll.iloc[i, j]
        if pd.notna(val):
            ax[1].text(j, i, f"{int(val)}", ha='center', va='center', fontsize=8)
fig1.colorbar(im2, ax=ax[1], shrink=.8, pad=.02, label="collisions")

fig1.suptitle("Figure 1 – Effect of ε-decay × τ", y=1.02)
fig1.tight_layout()


# ── helper: build a colour array of correct length ───────────────────────────
def make_colors(n):
    """Return n visually-distinct colours from a matplotlib colormap."""
    cmap = plt.cm.get_cmap("tab10") if n <= 10 else plt.cm.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]

# ── helper: improved scatter with discrete legend ────────────────────────────
def tradeoff_scatter(sub, param_col, title, fixed_txt):
    """
    sub        : dataframe for ONE sweep (3–4 rows)
    param_col  : "epsilon_decay" or "tau"
    """
    sub = sub.sort_values(param_col).reset_index(drop=True)
    colours = make_colors(len(sub))

    fig, ax = plt.subplots(figsize=(4.2, 3.3))

    for idx, row in sub.iterrows():
        ax.scatter(row.total_time, row.collision_count,
                   s=130, marker="o", edgecolor="k",
                   color=colours[idx], zorder=3)
    

    ax.set_xlabel("Total time (s)")
    ax.set_ylabel("Collision count")
    ax.set_title(f"{title}\n({fixed_txt})")
    ax.grid(alpha=.25)

    # discrete legend
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=9,
                      markeredgecolor='k', markerfacecolor=colours[i],
                      label=f"{param_col[0]}={v:.3f}")
               for i, v in enumerate(sub[param_col])]
    ax.legend(handles=handles, title=param_col.replace('_', ' '),
              loc="best", frameon=True)

    fig.tight_layout()
    return fig

# ── Figure 2 : vary ε-decay (τ fixed) ────────────────────────────────────────
tau_ref = df.tau.mode().iloc[0]            # most frequent τ (e.g. 0.01)
eps_df  = df[df.tau == tau_ref].copy()
tradeoff_scatter(eps_df,
                 param_col="epsilon_decay",
                 title="Figure 2 – ε-decay trade-off",
                 fixed_txt=f"τ = {tau_ref:.3f}")

# ── Figure 3 : vary τ (ε-decay fixed) ────────────────────────────────────────
eps_ref = df.epsilon_decay.mode().iloc[0]  # most frequent ε (e.g. 0.98)
tau_df  = df[df.epsilon_decay == eps_ref].copy()
tradeoff_scatter(tau_df,
                 param_col="tau",
                 title="Figure 3 – τ trade-off",
                 fixed_txt=f"ε-decay = {eps_ref:.3f}")

plt.show()

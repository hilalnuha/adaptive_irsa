#!/usr/bin/env python3
"""Regenerate the article figures from results/run_level_results.csv.

Usage: python code/make_figures.py results/run_level_results.csv figures
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
import os; os.makedirs(out, exist_ok=True)
d["G"] = d.actual_g.round(2)
C = {"adaptive_irsa": "#2a78d6", "standard_irsa": "#eb6834", "crdsa": "#1baf7a", "slotted_aloha": "#eda100"}
NAME = {"adaptive_irsa": "Adaptive IRSA", "standard_irsa": "Standard IRSA", "crdsa": "CRDSA", "slotted_aloha": "Slotted ALOHA"}
MK = {"adaptive_irsa": "o", "standard_irsa": "s", "crdsa": "^", "slotted_aloha": "D"}
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def style(ax, ylab):
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Normalized offered load, G")
    ax.set_ylabel(ylab)
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.axvspan(0.725, 1.025, color="#f1f0ec", zorder=0)
    ax.text(0.875, ax.get_ylim()[1] * 0.98, "high-load region", ha="center", va="top", color=MUTED, fontsize=8.5)


def band_plot(metric, ylab, fname, legend_loc):
    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=300)
    for p in ["standard_irsa", "adaptive_irsa"]:
        per_n = d[d.protocol == p].groupby(["node_count", "G"])[metric].mean().unstack(0)
        ax.fill_between(per_n.index, per_n.min(axis=1), per_n.max(axis=1), color=C[p], alpha=0.18, linewidth=0)
        ax.plot(per_n.index, per_n.mean(axis=1), color=C[p], lw=2, marker=MK[p], ms=5,
                markeredgecolor="white", markeredgewidth=1, label=NAME[p])
    ax.set_xlim(0.18, 1.02)
    if metric == "primary_plr":
        ax.set_ylim(0, 0.8)
    else:
        ax.set_ylim(0, 0.65)
    style(ax, ylab)
    ax.legend(loc=legend_loc)
    fig.tight_layout(); fig.savefig(f"{out}/{fname}", dpi=300); plt.close(fig)


band_plot("primary_plr", "Packet loss ratio, (–)", "fig_plr_all.png", "upper left")
band_plot("normalized_throughput", "Normalized throughput, (packet/slot)", "fig_throughput_all.png", "lower right")

# four protocols, N = 10000
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=300)
t = d[d.node_count == 10000].groupby(["G", "protocol"]).normalized_throughput.mean().unstack()
for p in ["adaptive_irsa", "standard_irsa", "crdsa", "slotted_aloha"]:
    ax.plot(t.index, t[p], color=C[p], lw=2, marker=MK[p], ms=5, markeredgecolor="white", markeredgewidth=1, label=NAME[p])
ax.set_xlim(0.18, 1.02); ax.set_ylim(0, 0.65)
style(ax, "Normalized throughput, (packet/slot)")
ax.legend(loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig(f"{out}/fig_four_protocol.png", dpi=300); plt.close(fig)

# adaptive reliability decomposition (mean across N)
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=300)
a = d[d.protocol == "adaptive_irsa"].groupby("G")[["primary_plr", "transmission_plr", "final_backlog_ratio"]].mean()
spec = [("transmission_plr", "Transmission PLR", "#eb6834", "s", "-"),
        ("final_backlog_ratio", "Final backlog ratio", "#1baf7a", "^", "-"),
        ("primary_plr", "Primary PLR", "#2a78d6", "o", "--")]
for col, lab, c, m, ls in spec:
    ax.plot(a.index, a[col], color=c, lw=2, marker=m, ms=5, ls=ls, markeredgecolor="white", markeredgewidth=1, label=lab)
ax.set_xlim(0.18, 1.02); ax.set_ylim(0, 0.5)
style(ax, "Ratio of generated or admitted packets, (–)")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{out}/fig_backlog.png", dpi=300); plt.close(fig)
print("ok")

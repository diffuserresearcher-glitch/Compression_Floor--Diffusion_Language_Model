#!/usr/bin/env python3
"""Regenerate the paper's figures from the shipped results.

CPU only, no GPU and no model downloads. Reads results/ and writes
results/figures/.

    python scripts/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

# =============================================================== FIG 1
rows = list(csv.DictReader(open(RESULTS / "floor_grid_per_problem.csv")))
vals = [float(r["repeat4"]) for r in rows if r["repeat4"] not in ("", "nan")]
n = len(vals)
lo = sum(1 for v in vals if v < 0.2)
mid = sum(1 for v in vals if 0.3 <= v <= 0.6)
hi = sum(1 for v in vals if v > 0.7)
print(f"FIG1 pooled n={n} fluent={lo/n:.0%} middle={mid/n:.0%} collapsed={hi/n:.0%}")

fig, ax = plt.subplots(figsize=(4.6, 2.3))
ax.hist(vals, bins=20, range=(0, 1), color="#4477aa", edgecolor="white", linewidth=0.5)
ax.axvline(0.5, color="0.35", linestyle=":", linewidth=1)
ax.set_xlabel("repeat-4 (per generation)")
ax.set_ylabel("# generations")
ax.set_xlim(0, 1)
ymax = ax.get_ylim()[1]
ax.text(0.03, ymax * 0.92, f"fluent mode\n{lo/n:.0%} below 0.2", fontsize=8, va="top", ha="left")
ax.text(0.97, ymax * 0.92, f"collapsed mode\n{hi/n:.0%} above 0.7", fontsize=8, va="top", ha="right")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig1_bimodality.png", bbox_inches="tight")
plt.close(fig)

# =============================================================== FIG 2
B = json.load(open(RESULTS / "benchmark_100_all_arms.json"))
ORDER = ["raw",
         "masked_s1", "masked_s2", "masked_s3",
         "dense500_s1", "dense500_s2", "dense500_s3",
         "dense1500_s1", "dense1500_s2", "dense1500_s3"]
LABEL = {"raw": "raw",
         "masked_s1": "masked s1", "masked_s2": "masked s2", "masked_s3": "masked s3",
         "dense500_s1": "dense500 s1", "dense500_s2": "dense500 s2", "dense500_s3": "dense500 s3",
         "dense1500_s1": "dense1500 s1", "dense1500_s2": "dense1500 s2", "dense1500_s3": "dense1500 s3"}
COL = {"raw": "0.35", "masked": "#4477aa", "dense500": "#ee7733", "dense1500": "#cc3311"}
def colour(k):
    return COL["raw"] if k == "raw" else COL[k.rsplit("_", 1)[0]]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
for ax, tps in zip(axes, (1, 2)):
    d = B[f"tokens_per_step_{tps}"]
    x = np.arange(len(ORDER))
    acc = [d[k]["acc"] for k in ORDER]
    err_lo = [d[k]["acc"] - d[k]["ci"][0] for k in ORDER]
    err_hi = [d[k]["ci"][1] - d[k]["acc"] for k in ORDER]
    ax.bar(x, acc, yerr=[err_lo, err_hi], capsize=3,
           color=[colour(k) for k in ORDER], edgecolor="white")
    ax.axhline(d["raw"]["acc"], ls=":", c="0.4", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.60)
    ax.set_title(f"tokens/step = {tps}")
fig.suptitle("n = 100 benchmark, 95% bootstrap CIs (dotted line: base model)", y=1.02, fontsize=9.5)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig2_benchmark100.png", bbox_inches="tight")
plt.close(fig)
print("FIG2 n=100 accuracy with CIs")

# =============================================================== FIG 3
fig, ax = plt.subplots(figsize=(6.6, 2.6))
for tps, mk, ls in ((1, "o", "-"), (2, "s", "--")):
    d = B[f"tokens_per_step_{tps}"]
    ax.plot(range(len(ORDER)), [d[k]["repeat4"] for k in ORDER],
            marker=mk, linestyle=ls, markersize=4, linewidth=1.3, label=f"t={tps}")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("repeat-4 (lower better)")
ax.set_title("degeneration signature: dense supervision raises repetition at every stage")
ax.legend(frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig3_repeat4_100.png", bbox_inches="tight")
plt.close(fig)
print("FIG3 repeat-4 signature")

# =============================================================== FIG 4
O = json.load(open(RESULTS / "overlap_topk_results.json"))
A, Bt = O["teacher_A_privileged_prefix"], O["teacher_B_self_future_lookahead"]
fig, ax = plt.subplots(figsize=(4.4, 2.6))
names = ["A: privileged\nprefix (OPSD-1)", "B: self-future\nlookahead (OPSD-2)"]
means = [A["mean_overlap"], Bt["mean_overlap"]]
pairs = [(A["mean_overlap"], A["ci95"]), (Bt["mean_overlap"], Bt["ci95"])]
errs = [[m - c[0] for m, c in pairs], [c[1] - m for m, c in pairs]]
ax.bar(names, means, yerr=errs, capsize=4, color=["#cc3311", "#4477aa"],
       edgecolor="white", width=0.55)
ax.axhline(1.0, ls=":", c="0.4", lw=1)
ax.text(0.5, 1.02, "total agreement: no learning signal", fontsize=7.5, ha="center", color="0.35")
ax.set_ylabel("Overlap Top-K")
ax.set_ylim(0, 1.15)
for i, m in enumerate(means):
    ax.text(i, m + 0.04, f"{m:.3f}", ha="center", fontsize=8.5)
ax.set_title(f"teacher-student agreement ({O['setup']['n_states_measured']:,} states)")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig4_overlap_topk.png", bbox_inches="tight")
plt.close(fig)
print("FIG4 Overlap Top-K")

print(f"\nwrote 4 figures to {OUT}")

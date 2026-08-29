"""Regenerate the paper's figures from the shipped results. CPU only, no GPU
or model downloads needed.

    python scripts/make_figures.py

Reads results/*.csv and writes results/figures/*.png.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------------- FIG 1
# Bimodality of per-problem repeat-4, POOLED over block 16 and block 32, all schedules.
rows = list(csv.DictReader(open(
    RESULTS / "floor_grid_per_problem.csv")))
vals = [float(r["repeat4"]) for r in rows if r["repeat4"] not in ("", "nan")]
lo = sum(1 for v in vals if v < 0.2)
mid = sum(1 for v in vals if 0.3 <= v <= 0.6)
hi = sum(1 for v in vals if v > 0.7)
n = len(vals)
print(f"FIG1 pooled b16+b32 n={n} fluent={lo} ({lo/n:.0%}) middle={mid} ({mid/n:.0%}) collapsed={hi} ({hi/n:.0%})")

fig, ax = plt.subplots(figsize=(4.6, 2.3))
ax.hist(vals, bins=20, range=(0, 1), color="#4477aa", edgecolor="white", linewidth=0.5)
ax.axvline(0.5, color="0.35", linestyle=":", linewidth=1)
ax.set_xlabel("repeat-4 (per generation)")
ax.set_ylabel("# generations")
ax.set_xlim(0, 1)
ymax = ax.get_ylim()[1]
ax.text(0.03, ymax * 0.92, f"fluent mode\n{lo/n:.0%} below 0.2",
        fontsize=8, va="top", ha="left")
ax.text(0.97, ymax * 0.92, f"collapsed mode\n{hi/n:.0%} above 0.7",
        fontsize=8, va="top", ha="right")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig1_bimodality.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- FIG 2
# Base (block 16 and block 32) vs dense-supervision SFT control (block 32), common schedules.
base16, base32 = {}, {}
for r in csv.DictReader(open(
        RESULTS / "floor_grid_aggregate.csv")):
    if r["block_size"] == "16":
        base16[int(r["tokens_per_step"])] = r
    elif r["block_size"] == "32":
        base32[int(r["tokens_per_step"])] = r
sft = {}
for r in csv.DictReader(open(RESULTS / "sft_dense_schedule_grid.csv")):
    if r["arm"] == "sft_only":
        sft[int(r["tokens_per_step"])] = r

T = [1, 2, 4, 8, 16]                      # schedules measured in ALL three series
panels = [("correct", "correct rate", False),
          ("repeat4", "repeat-4 (lower better)", False),
          ("redundancy", "redundancy (lower better)", False),
          ("coverage", "reference-solution coverage", False)]

fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.05))
for ax, (key, title, _) in zip(axes, panels):
    ax.plot(range(len(T)), [float(base16[t][key]) for t in T],
            "^-", color="0.55", label="base (b16)", markersize=4, linewidth=1.2)
    ax.plot(range(len(T)), [float(base32[t][key]) for t in T],
            "o-", color="#4477aa", label="base (b32)", markersize=4, linewidth=1.4)
    ax.plot(range(len(T)), [float(sft[t][key]) for t in T],
            "s-", color="#cc3311", label="SFT, dense (b32)", markersize=4, linewidth=1.4)
    ax.set_title(title)
    ax.set_xticks(range(len(T)))
    ax.set_xticklabels(T)
    ax.set_xlabel("tokens/step")
axes[0].legend(frameon=False, loc="upper right", fontsize=7)
fig.suptitle("Base (block 16 and 32) vs dense-supervision SFT (block 32)", y=1.06, fontsize=9.5)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig2_base_vs_sft.png", bbox_inches="tight")
plt.close(fig)

print(f"wrote {OUT}/fig1_bimodality.png and {OUT}/fig2_base_vs_sft.png")
for t in T:
    print(f"  t={t:<3} b16 correct {float(base16[t]['correct']):.3f}  b32 correct {float(base32[t]['correct']):.3f}  "
          f"sft correct {float(sft[t]['correct']):.3f}")

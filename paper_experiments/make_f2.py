# -*- coding: utf-8 -*-
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import report_from_results as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

D = R.load(120000)
order = [fk for fk, _, _ in R.SWEEP if any(fk in d["rows"] for d in D)]
data = [[d["rows"][fk]["range_pct"] for d in D if fk in d["rows"] and d["rows"][fk].get("range_pct") is not None] for fk in order]

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(7.2, 2.9))

bp = ax.boxplot([x if x else [np.nan] for x in data], showfliers=False, patch_artist=True,
                medianprops=dict(color="#d95f02", lw=1.3),
                boxprops=dict(lw=1.0),
                whiskerprops=dict(lw=0.9),
                capprops=dict(lw=1.0))

for b, fk in zip(bp["boxes"], order):
    cls_ = dict((a, c) for a, _, c in R.SWEEP)[fk]
    b.set(facecolor="#c6dbef" if cls_ == "inv" else ("#fed9a6" if cls_ == "approx" else "#fcbba1"))

ax.set_xticks(range(1, len(order) + 1))
labels = [dict((a, b_) for a, b_, _ in R.SWEEP)[fk].replace(" (", "\n(") for fk in order]
ax.set_xticklabels(labels, fontsize=7.5)

ax.set_yscale("log")
ax.set_ylim(0.4, 85)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
ax.set_ylabel("açı aralığı (%)", fontsize=8.5)

n_inv = sum(1 for fk in order if dict((a, c) for a, _, c in R.SWEEP)[fk] == "inv")
ax.axvline(n_inv + 0.5, color="k", ls="--", lw=0.8)

fig.tight_layout()
out_dir = os.path.join(ROOT, "makale_v3", "figures")
fig.savefig(os.path.join(out_dir, "F2_duyarlilik_kutu.pdf"))
fig.savefig(os.path.join(out_dir, "F2_duyarlilik_kutu.png"), dpi=180)
plt.close(fig)
print("F2_duyarlilik_kutu successfully written to makale_v3/figures!")

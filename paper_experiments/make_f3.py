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
C = [(d["rows"]["fs_ge8"]["range_pct"], d["rows"]["fs_ge8_jitter"]["range_pct"], d["rows"]["fs_ge8_detied"]["range_pct"], d["rows"]["fs_ge8_detied"].get("tie_frac_before", 0), d["n"])
     for d in D if all(k in d["rows"] for k in ("fs_ge8", "fs_ge8_jitter", "fs_ge8_detied"))]

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif"})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

# Panel 1: rot vs jitter
ax = axes[0]
ax.scatter([c[1] for c in C], [c[0] for c in C], s=14, alpha=0.75, color="#1f4e79", edgecolors="none")
lim = [0.35, 20]
ax.plot(lim, lim, "k--", lw=0.9, label="$y=x$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.35, 20)
ax.set_ylim(0.35, 20)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
ax.set_yticks([0.5, 1, 2, 5, 10, 20])
ax.set_xlabel("jitter aralığı, $\\theta=0$ (%)", fontsize=8.5)
ax.set_ylabel("rotasyon taraması aralığı (%)", fontsize=8.5)
ax.set_title("GE: rotasyon = beraberlik kırma", fontsize=9)
ax.legend(fontsize=7.5, loc="lower right")

# Panel 2: range vs n
ax = axes[1]
ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
ax.scatter([c[4] for c in C], [c[0] for c in C], s=14, label="rotasyon", color="#1f4e79", alpha=0.75)
ax.scatter([c[4] for c in C], [c[2] for c in C], s=18, marker="x", lw=1.2, label="beraberliksiz", color="#c62828", alpha=0.85)
ax.set_xscale("log")
ax.set_xlim(80, 150000)
ax.set_ylim(-0.6, 13.5)
ax.set_xlabel("$n$ (örnek boyutu)", fontsize=8.5)
ax.set_ylabel("GE açı aralığı (%)", fontsize=8.5)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Aralık $n$ ile küçülür; beraberliksizde $\\approx 0$", fontsize=9)

fig.tight_layout()

out_pdf = os.path.join(ROOT, "makale_v3", "figures", "F3_ge_beraberlik.pdf")
out_png = os.path.join(ROOT, "makale_v3", "figures", "F3_ge_beraberlik.png")
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=160)
plt.close(fig)
print("Updated F3_ge_beraberlik.pdf and F3_ge_beraberlik.png")

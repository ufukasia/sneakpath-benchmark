# -*- coding: utf-8 -*-
"""parallel_experiment.py ciktisindan paralel-GE tablolari ve figurleri.

    python paper_experiments/make_parallel_report.py [--out paper_experiments/out_parallel]

Yazilanlar (out_parallel/report/): summary.md, tables/T7*.md/.tex, figures/F7*.png
  T7  : strateji x k -> makespan orani (makespan / (L_GE/k); 1.0 ideal), toplam/L_GE,
        denge, paralel hiz kazanci (t_GE / t_par) -- ortalama, siniflara gore
  T7b : her k icin en iyi strateji (makespan) -- kac ornekte hangi strateji kazandi
  T7c : bant bolumlemesinin cerceve bagimliligi: theta* / theta=0 / theta=22.5 ve
        k=4 aci egrisi araligi
  F7  : makespan orani vs k (strateji egrileri), F7b: bant aci egrisi, F7c: hiz kazanci
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from make_report import Tables, sign_test, wilcoxon, fmt, pfmt, mean, q  # noqa: E402

LABELS = {"band_star": "band (theta*)", "band_zero": "band (theta=0)", "band_mis": "band (theta=22.5, misaligned)",
          "kmeans": "k-means", "split": "split global GE tour"}
ORDER = ["band_star", "band_zero", "band_mis", "kmeans", "split"]


def load(out):
    rows = []
    for f in sorted(os.listdir(os.path.join(out, "per_instance"))):
        if f.endswith(".json"):
            d = json.load(open(os.path.join(out, "per_instance", f), encoding="utf-8"))
            if "error" not in d:
                rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out_parallel"))
    args = ap.parse_args()
    rep = os.path.join(args.out, "report")
    os.makedirs(os.path.join(rep, "figures"), exist_ok=True)
    D = load(args.out)
    if not D:
        print("sonuc yok")
        return
    tabs = Tables(os.path.join(rep, "tables"))
    S = [f"# Parallel-GE experiment -- summary", "", f"Instances: {len(D)} "
         f"({sum(1 for d in D if d['cls'] == 'vlsi')} VLSI, {sum(1 for d in D if d['cls'] == 'tsplib')} TSPLIB)", ""]
    N = {}

    # ---- hucre olculeri: makespan orani, toplam orani, denge, hiz -------------
    def cells(d):
        L, t = d["ge"]["cost"], d["ge"]["time"]
        out = []
        for c in d["cells"]:
            k = c["k_requested"]
            out.append(dict(label=c["label"], k=k, mk_ratio=c["makespan"] / (L / k), tot_ratio=c["total"] / L,
                            imb=c["imbalance"], speed=t / max(c["t_par"], 1e-9), t_par=c["t_par"]))
        return out

    ks = sorted({c["k_requested"] for d in D for c in d["cells"]})
    rows = []
    agg = {}
    for cls in ("all", "vlsi", "tsplib"):
        Ds = D if cls == "all" else [d for d in D if d["cls"] == cls]
        if not Ds:
            continue
        for lab in ORDER:
            for k in ks:
                v = [c for d in Ds for c in cells(d) if c["label"] == lab and c["k"] == k]
                if len(v) < 3:
                    continue
                a = dict(n=len(v), mk=mean([c["mk_ratio"] for c in v]), mk_med=st.median([c["mk_ratio"] for c in v]),
                         tot=mean([c["tot_ratio"] for c in v]), imb=mean([c["imb"] for c in v]),
                         speed=st.median([c["speed"] for c in v]))
                agg[(cls, lab, k)] = a
                rows.append([cls.upper() if cls != "all" else "All", LABELS[lab], k, a["n"], fmt(a["mk"], 3), fmt(a["mk_med"], 3),
                             fmt(a["tot"], 3), fmt(a["imb"], 3), fmt(a["speed"], 1)])
    tabs.write("T7", "Parallel GE: partition strategy x k (means over instances)",
               ["Class", "Partition", "k", "N", "Makespan / (L_GE/k) mean", "median", "Total / L_GE", "Imbalance (max/mean)",
                "Build speed-up t_GE / t_par (median)"], rows,
               "L_GE = global Greedy-Edge tour (k=1). Makespan ratio 1.0 = k agents finish in exactly 1/k of the single-agent tour; total ratio 1.0 = no extra distance from partitioning.")

    # ---- her k icin kazanan strateji --------------------------------------------
    rows = []
    best_by_k = {}
    for cls in ("vlsi", "tsplib"):
        Ds = [d for d in D if d["cls"] == cls]
        for k in ks:
            wins = defaultdict(int)
            mk_gain = []
            for d in Ds:
                cs = {c["label"]: c for c in cells(d) if c["k"] == k and c["label"] in ("band_star", "kmeans", "split")}
                if len(cs) < 3:
                    continue
                w = min(cs, key=lambda l: cs[l]["mk_ratio"])
                wins[w] += 1
                mk_gain.append(cs["kmeans"]["mk_ratio"] - cs["band_star"]["mk_ratio"])
            if not mk_gain:
                continue
            wb, lb, p = sign_test(mk_gain)      # negatif = kmeans daha iyi
            best_by_k[(cls, k)] = dict(wins=dict(wins), kmeans_minus_band=mean(mk_gain), p=p)
            rows.append([cls.upper(), k, len(mk_gain), wins.get("band_star", 0), wins.get("kmeans", 0), wins.get("split", 0),
                         fmt(mean(mk_gain), 3), f"{wb}/{lb}", pfmt(p)])
    tabs.write("T7b", "Which partition gives the shortest makespan? (band at theta*, k-means, split)",
               ["Class", "k", "N", "band wins", "k-means wins", "split wins", "mean(k-means - band) makespan ratio",
                "k-means better / worse", "p (sign)"], rows)

    # ---- bant bolumlemesinin cerceve bagimliligi -----------------------------------
    rows = []
    frame = {}
    for cls in ("vlsi", "tsplib"):
        Ds = [d for d in D if d["cls"] == cls]
        for k in ks:
            d_mis, d_zero = [], []
            for d in Ds:
                cs = {c["label"]: c for c in cells(d) if c["k"] == k}
                if "band_star" in cs and "band_mis" in cs and "band_zero" in cs:
                    d_mis.append(cs["band_mis"]["mk_ratio"] - cs["band_star"]["mk_ratio"])
                    d_zero.append(cs["band_zero"]["mk_ratio"] - cs["band_star"]["mk_ratio"])
            if len(d_mis) < 3:
                continue
            w, l, p = sign_test([-x for x in d_mis])
            frame[(cls, k)] = dict(mis=mean(d_mis), zero=mean(d_zero), p=p, n=len(d_mis), mis_worse=w)
            rows.append([cls.upper(), k, len(d_mis), fmt(mean(d_mis), 3), f"{w}/{len(d_mis)}", pfmt(p), fmt(mean(d_zero), 3)])
        curve = [d["band_angle_curve_k4"] for d in Ds if d.get("band_angle_curve_k4")]
        if curve:
            rng = [100 * (max(c[a]["makespan"] for a in c) - min(c[a]["makespan"] for a in c)) / min(c[a]["makespan"] for a in c) for c in curve]
            frame[(cls, "curve")] = dict(range_med=st.median(rng), range_mean=mean(rng))
            rows.append([cls.upper(), "k=4 angle sweep", len(rng), f"range med {st.median(rng):.1f}% / mean {mean(rng):.1f}%", "", "", ""])
    tabs.write("T7c", "Frame dependence of band partitioning: misaligned (theta=22.5) vs aligned (theta*) bands",
               ["Class", "k", "N", "mean(mis - aligned) makespan ratio", "misaligned worse", "p (sign)", "mean(theta=0 - theta*)"], rows,
               "Positive = misaligned bands give a longer makespan. The k=4 angle sweep row gives the range of makespan over 12 band angles.")
    N.update(agg={f"{c}|{l}|{k}": v for (c, l, k), v in agg.items()},
             best_by_k={f"{c}|{k}": v for (c, k), v in best_by_k.items()},
             frame={f"{c}|{k}": v for (c, k), v in frame.items()})

    # ---- figurler --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
        COL = {"band_star": "#b45309", "band_zero": "#d97706", "band_mis": "#ef4444", "kmeans": "#0f766e", "split": "#475569"}
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, cls in zip(axes[:2], ("vlsi", "tsplib")):
            if not any((cls, lab, k) in agg for lab in ORDER for k in ks):
                ax.set_title(f"{cls.upper()}: no data yet")
                continue
            for lab in ORDER:
                xs_ = [k for k in ks if (cls, lab, k) in agg]
                if xs_:
                    ax.plot(xs_, [agg[(cls, lab, k)]["mk"] for k in xs_], "-o", ms=3, color=COL[lab], label=LABELS[lab])
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("k agents")
            ax.set_ylabel("makespan / (L_GE / k)")
            ax.set_title(f"{cls.upper()}: how close to ideal 1/k finish?")
        axes[0].legend(fontsize=7)
        ax = axes[2]
        for lab in ("band_star", "kmeans"):
            xs_ = [k for k in ks if ("all", lab, k) in agg]
            if xs_:
                ax.plot(xs_, [max(agg[("all", lab, k)]["speed"], 1e-3) for k in xs_], "-o", ms=3, color=COL[lab], label=LABELS[lab])
        ax.plot(ks, ks, "k--", lw=0.8, label="ideal k")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("k agents")
        ax.set_ylabel("build speed-up t_GE / t_par (median)")
        ax.set_title("Parallel construction speed-up")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(rep, "figures", "F7_parallel_ge.png"), dpi=170)
        plt.close(fig)
        # bant aci egrisi: normalize makespan(theta)/min
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        vl = [d for d in D if d["cls"] == "vlsi" and d.get("band_angle_curve_k4")]
        import numpy as np
        angs = sorted({float(a) for d in vl for a in d["band_angle_curve_k4"]})
        M = []
        for d in vl:
            c = d["band_angle_curve_k4"]
            m = min(c[a]["makespan"] for a in c)
            M.append([c[f"{a:g}"]["makespan"] / m for a in angs])
        M = np.array(M)
        if len(M):
            ax.plot(angs, np.median(M, 0), "-o", ms=3, color="#b45309", label="median over VLSI")
            ax.fill_between(angs, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color="#b45309", alpha=0.2, label="IQR")
        ax.set_xlabel("band angle theta (deg)")
        ax.set_ylabel("makespan / best-angle makespan (k=4)")
        ax.set_title("Band partition is frame-dependent")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(rep, "figures", "F7b_band_angle.png"), dpi=170)
        plt.close(fig)
    except Exception as ex:
        import traceback
        N["figure_error"] = traceback.format_exc()
        print("figur hatasi:", ex)

    # ---- summary ---------------------------------------------------------------
    S.append("## Makespan ratio (mean) by partition and k")
    for cls in ("vlsi", "tsplib"):
        for lab in ORDER:
            vals = [(k, agg[(cls, lab, k)]["mk"]) for k in ks if (cls, lab, k) in agg]
            if vals:
                S.append(f"- {cls} {LABELS[lab]}: " + ", ".join(f"k={k}: {v:.3f}" for k, v in vals))
    S.append("")
    S.append("## Winner by k (makespan)")
    for key, v in N["best_by_k"].items():
        S.append(f"- {key}: wins {v['wins']}, mean(kmeans-band) {v['kmeans_minus_band']:+.3f}, p={pfmt(v['p'])}")
    S.append("")
    S.append("## Frame dependence of bands")
    for key, v in N["frame"].items():
        if "mis" in v:
            S.append(f"- {key}: misaligned - aligned = {v['mis']:+.3f} (worse in {v['mis_worse']}/{v['n']}, p={pfmt(v['p'])}); theta=0 - theta* = {v['zero']:+.3f}")
        else:
            S.append(f"- {key}: k=4 band angle sweep makespan range median {v['range_med']:.1f}%, mean {v['range_mean']:.1f}%")
    S.append("")
    S.append("## Tables / figures")
    for k, t in tabs.index:
        S.append(f"- {k}: {t}")
    for f in sorted(os.listdir(os.path.join(rep, "figures"))):
        S.append(f"- figures/{f}")
    open(os.path.join(rep, "summary.md"), "w", encoding="utf-8").write("\n".join(S) + "\n")
    json.dump(N, open(os.path.join(rep, "numbers.json"), "w", encoding="utf-8"), indent=1, default=str)
    print("\n".join(S))


if __name__ == "__main__":
    main()

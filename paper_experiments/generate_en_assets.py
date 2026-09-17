# -*- coding: utf-8 -*-
"""Generate English publication assets (tables and figures) for Journal of Heuristics.

Reads data from results/*.json and writes:
    makale_v3/tables/T*.tex
    makale_v3/figures/F*.pdf / F*.png
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p_ in (ROOT, HERE):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import vlsi_datasets as V
from make_report import (sign_test, wilcoxon, spearman, ks_2samp, q, mean,
                         fmt, pfmt)

SWEEP = [("fs_ge8", "GE ($k{=}8$)", "inv"), ("fs_ge15", "GE ($k{=}15$)", "inv"),
         ("fs_nn", "NN (exact)", "inv"), ("fs_fi", "Farthest Ins.", "inv"),
         ("fs_nn_grid", "NN (grid)", "approx"),
         ("fs_strip", "Strip", "dep"), ("fs_hilbert", "Hilbert", "dep"),
         ("fs_morton", "Morton", "dep"), ("fs_band2ge", "2-band GE", "dep")]

PHIS = [("", 22.5), ("@manual:7.5", 7.5), ("@manual:37.5", 37.5)]

PGE_S = [("band", "band (1-D, balanced)"), ("kd", "k-d (2-D, balanced)"),
         ("kmeans", "k-means (2-D, unbalanced)"), ("split", "global tour cutting")]


def tex_table(path, caption, label, header, rows, note="", col=None, small=True):
    cols = col or ("l" + "r" * (len(header) - 1))
    L = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        L.append("\\footnotesize")
    if len(header) >= 6:
        L[-1] = "\\scriptsize"
    L += ["\\setlength{\\tabcolsep}{2.5pt}",
          f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
          " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MIDRULE":
            L.append("\\midrule")
            continue
        L.append(" & ".join(str(c) for c in r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    if note:
        L.append("\\begin{flushleft}\\footnotesize " + note + "\\end{flushleft}")
    L.append("\\end{table}")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        if r != "MIDRULE":
            md.append("| " + " | ".join(str(c) for c in r) + " |")
    open(path[:-4] + ".md", "w", encoding="utf-8").write(f"**{caption}**\n\n" + "\n".join(md) + ("\n\n" + note if note else "") + "\n")


def pct(x, nd=1):
    return fmt(x, nd) + "\\%"


def prel(p, nd=3):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 1e-4:
        return "<10^{-4}"
    if p < 10 ** (-nd) / 2:
        return f"<10^{{-{nd}}}"
    return f"={p:.{nd}f}"


def ptex(p, nd=3):
    r = prel(p, nd)
    if r == "--":
        return r
    return f"${r}$" if r.startswith("<") else r[1:]


def load(max_n=120000):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        nm = d.get("dataset")
        if nm not in V.CATALOG or d["n"] > max_n or not d.get("bks"):
            continue
        rows = {m["row_id"]: m for m in d.get("methods", []) if m.get("cost") is not None}
        if "greedy_edge@knn8_greedy" not in rows:
            continue
        out.append(dict(name=nm, n=d["n"], bks=float(d["bks"]), rows=rows))
    out.sort(key=lambda d: d["n"])
    return out


def main():
    out_dir = os.path.join(ROOT, "makale_v3")
    T = os.path.join(out_dir, "tables")
    F = os.path.join(out_dir, "figures")
    os.makedirs(T, exist_ok=True)
    os.makedirs(F, exist_ok=True)

    D = load()

    def g(d, c):
        return 100.0 * (c / d["bks"] - 1.0)

    def gap_of(d, rid):
        m = d["rows"].get(rid)
        return g(d, m["cost"]) if m else None

    # ================= T1: Baseline constructors theta=0 =====================
    base_rows = [("greedy_edge", "Greedy-Edge, $k{=}15$"), ("greedy_edge@knn8_greedy", "Greedy-Edge, $k{=}8$"),
                 ("nn_exact", "NN (exact)"), ("nn", "NN (grid)"),
                 ("farthest_insertion", "Farthest Insertion"), ("strip", "Strip"),
                 ("hilbert", "Hilbert"), ("morton", "Morton"), ("rotation_strip", "Strip (best angle)"),
                 ("lkh3", "LKH-3 (reference)")]
    rows = []
    for rid, nm in base_rows:
        gs = [gap_of(d, rid) for d in D if rid in d["rows"]]
        ts = [d["rows"][rid]["time"] for d in D if rid in d["rows"]]
        if not gs:
            continue
        rows.append([nm, len(gs), pct(mean(gs)), pct(st.median(gs)), pct(min(gs)), pct(max(gs)), fmt(st.median(ts), 3)])
    fi_ns = [d["n"] for d in D if "farthest_insertion" in d["rows"]]
    tex_table(os.path.join(T, "T1_kurucular.tex"),
              "Baseline constructive heuristics, $\\theta=0^\\circ$: gap (\\%) relative to BKS and runtime.",
              "tab:kurucular", ["Constructor", "$N$", "mean", "median", "min", "max", "time (s)"], rows,
              f"Greedy-Edge $k{{=}}15$: Johnson--McGeoch candidate list. Farthest Insertion pure $O(n^2)$: completed on {len(fi_ns)} instances within construction budget ($n\\le{max(fi_ns) if fi_ns else 0}$); in rotation sweep (Table~\\ref{{tab:aci}}) limited to $n\\le3000$. Strip (best angle): refined continuous sweep (\\texttt{{rotation\\_strip}}). Runtime in pure Python, single thread, median. LKH-3 is a time-limited baseline reference, not a direct competitor ($N=71$; 28 large instances with $n>13\\,000$ skipped due to 300 s cutoff).")

    # ================= T2: Angle sensitivity =================================
    rows = []
    sens = {}
    for fk, nm, cls in SWEEP:
        ms = [d for d in D if fk in d["rows"] and d["rows"][fk].get("sweep")]
        if not ms:
            continue
        rng = [d["rows"][fk]["range_pct"] for d in ms]
        worst, best, sd = [], [], []
        for d in ms:
            m = d["rows"][fk]
            gs = [g(d, c) for c in m["sweep"].values()]
            g0 = g(d, m["cost0"])
            worst.append(max(gs) - g0)
            best.append(g0 - min(gs))
            sd.append(st.pstdev(gs))
        sens[fk] = dict(n=len(ms), med=st.median(rng), mean=mean(rng), p90=q(rng, .9), worst=mean(worst), best=mean(best), sd=mean(sd))
        dk = fk + "_detied"
        det_r = [d["rows"][dk]["range_pct"] for d in D if dk in d["rows"] and d["rows"][dk].get("range_pct") is not None]
        cls_lab = {"inv": "invariant", "approx": "inv.\\ $+$ approx.", "dep": "dependent"}[cls]
        det_cell = (fmt(st.median(det_r), 3) + f" ({len(det_r)})") if det_r else "--"
        rows.append([nm, cls_lab, len(ms), fmt(st.median(rng), 1) + " / " + fmt(mean(rng), 1), fmt(q(rng, .9), 1),
                     fmt(mean(worst)) + " / " + fmt(mean(best)), det_cell])
    tex_table(os.path.join(T, "T2_aci_duyarlilik.tex"),
              "Sensitivity to coordinate orientation: tour cost range ($100\\,(\\max-\\min)/\\min$; median / mean, p90) in $[-90^\\circ,90^\\circ)$ sweep, and worst-angle loss / best-angle gain relative to $\\theta=0^\\circ$ (points).",
              "tab:aci", ["Constructor", "Class", "$N$", "range med./mean", "p90", "worst/best angle", "detied ($N$)"], rows,
              "Range, p90, and detied in percent; worst/best angle in percentage points. Step: $5^\\circ$ ($n\\le 3000$), $10^\\circ$ ($\\le 15\\,000$), $15^\\circ$ ($\\le 50\\,000$), $30^\\circ$ (above). Cost is always evaluated on original coordinates. "
              "Class (theoretical): reliance strictly on pairwise Euclidean distances (invariant) versus coordinate axes (dependent); NN (grid) is distance-based but uses axis-aligned grid cells for candidate lookup (approx). "
              "Detied: median range after adding $10^{-4}\\times$ median neighbor distance perturbation to break coordinate ties; invariant constructors approach 0, while grid approximation retains residuals. "
              "NN (exact) detied has $N=98$ (one missing record for dan59296). In GE detied, median is 0.000\\% (69\\%--72\\% exact zero), but grid $k$-NN boundary shifts leave tail residuals in two extreme instances: \\%5.040 in GE $k{=}15$ and \\%7.043 in GE $k{=}8$ (vanishes to exact 0.000\\% under exact $k$-NN, see Table~\\ref{tab:gekontrol}).")

    # ================= T3: GE Controls ======================================
    def ctrl(d):
        r = d["rows"]
        if not all(k in r for k in ("fs_ge8", "fs_ge8_jitter", "fs_ge8_detied")):
            return None
        rot = [g(d, c) for c in r["fs_ge8"]["sweep"].values()]
        jit = [g(d, c) for c in r["fs_ge8_jitter"]["jitter"].values()]
        det = [g(d, c) for c in r["fs_ge8_detied"]["sweep"].values()]
        ex = [g(d, c) for c in r["fs_ge8_exactknn"]["sweep"].values()] if "fs_ge8_exactknn" in r else None
        g0 = g(d, r["fs_ge8"]["cost0"])
        return dict(n=d["n"], rng_rot=r["fs_ge8"]["range_pct"], rng_jit=r["fs_ge8_jitter"]["range_pct"],
                    rng_det=r["fs_ge8_detied"]["range_pct"], rng_ex=(r["fs_ge8_exactknn"]["range_pct"] if ex else None),
                    ks_p=ks_2samp(rot, jit)[1], best_rot=g0 - min(rot), best_jit=g0 - min(jit), mean_shift=mean(rot) - g0,
                    tie=r["fs_ge8_detied"].get("tie_frac_before"), gap0=g0)
    C = [c for c in (ctrl(d) for d in D) if c]
    rows = []
    for lab, sel in (("All", C), ("$n<500$", [c for c in C if c["n"] < 500]), ("$500\\le n<2000$", [c for c in C if 500 <= c["n"] < 2000]),
                     ("$2000\\le n<10^4$", [c for c in C if 2000 <= c["n"] < 10000]), ("$n\\ge 10^4$", [c for c in C if c["n"] >= 10000])):
        if not sel:
            continue
        rows.append([lab, len(sel), fmt(st.median([c["rng_rot"] for c in sel]), 2) + " / " + fmt(st.median([c["rng_jit"] for c in sel]), 2),
                     fmt(st.median([c["rng_det"] for c in sel]), 3),
                     fmt(100 * mean([c["ks_p"] > 0.05 for c in sel]), 0),
                     fmt(mean([c["best_rot"] for c in sel])) + " / " + fmt(mean([c["best_jit"] for c in sel]))])
    tex_table(os.path.join(T, "T3_ge_kontrol.tex"),
              "Greedy-Edge ($k{=}8$): rotation sweep versus tie-breaking controls (ranges are median, equal sample sizes). KS: proportion of instances where rotation and jitter distributions cannot be distinguished by Kolmogorov--Smirnov test.",
              "tab:gekontrol", ["Subset", "$N$", "range rot.\\ / jit.\\ (\\%)", "detied (\\%)", "KS (\\%)", "gain angle / jit.\\ (pts)"], rows,
              "Jitter: $\\theta=0^\\circ$ fixed orientation, only equal-length candidate edges perturbed by deterministic hash $\\pm10^{-9}$ (identical sample size to rotation sweep). Detied: coordinate perturbation. KS: two-sample Kolmogorov--Smirnov test between rotation and jitter distributions ($p>0.05$). Gain: $\\mathrm{gap}_0 - \\min\\mathrm{gap}$. Exact $k$-NN control evaluated on $N=65$ instances ($n\\le6000$); maximum range across all instances is exactly $0.000\\%$. Equal-budget oracle sign test between best angle and best jitter seed: 50/47 ($p=0.84$). Spearman rank correlation between rotation and jitter ranges is $\\rho_s=0.93$ ($p<10^{-4}$); between $\\mathrm{gap}_0$ and best angle gain is $\\rho_s=0.84$ ($p<10^{-4}$, regression to the mean).")

    # ================= T5: Detectors =========================================
    snap_ok, snap_tot, ident = 0, 0, 0
    eff = {m: dict(orig=[], mis=[], re=[]) for m in ("ge8", "strip", "hilbert", "morton")}
    det_err = defaultdict(list)
    for d in D:
        r = d["rows"]
        for meth, base in (("ge8", "greedy_edge@knn8_greedy"), ("strip", "strip"), ("hilbert", "hilbert"), ("morton", "morton")):
            if base not in r:
                continue
            for suf, phi in PHIS:
                m = r.get(f"realign_{meth}{suf}")
                if not m:
                    continue
                eff[meth]["orig"].append(g(d, r[base]["cost"]))
                eff[meth]["mis"].append(g(d, m["initial_cost"]))
                eff[meth]["re"].append(g(d, m["cost"]))
                if meth == "ge8":
                    for dk, x in (m.get("det") or {}).items():
                        det_err[dk].append(x["err"])
                    det_err["snap"].append(m["err_deg"])
                    snap_tot += 1
                    snap_ok += bool((m.get("snap") or {}).get("exact_recovery"))
                    ident += bool(m.get("ge8_tour_identical"))
    rows = []
    for dk, nm in (("comb", "Comb projection (grid\\_theta)"), ("nndir", "NN edge direction"), ("pca", "PCA boundary baseline"), ("snap", "Hierarchical snap (coarse + fine)")):
        e = det_err.get(dk, [])
        if not e:
            continue
        rows.append([nm, len(e), fmt(st.median(e), 4), fmt(q(e, .9), 2), fmt(100 * mean([x < 1 for x in e]), 0) + "\\%", fmt(100 * mean([x < 0.1 for x in e]), 0) + "\\%"])
    tex_table(os.path.join(T, "T5_dedektor.tex"),
              f"Lattice orientation estimation error $|\\hat\\theta-\\varphi|$ (degrees, mod 90) on misaligned instances ($\\varphi\\in\\{{7.5^\\circ,22.5^\\circ,37.5^\\circ\\}}$) and exact lattice recovery.",
              "tab:dedektor", ["Detector", "trials", "median", "p90", "$<1^\\circ$", "$<0.1^\\circ$"], rows,
              f"Snap: integer lattice was exactly recovered in {snap_ok} out of {snap_tot} trials ({100 * snap_ok / max(snap_tot, 1):.0f}\\%), with GE tour bit-identical in {ident} trials.")

    # ================= T5b: Noise ablation ===================================
    nz = [d["rows"][k_] for d in D for k_ in d["rows"] if k_.startswith("realign_strip_noise")]
    nzd = [(d, d["rows"][k_]) for d in D for k_ in d["rows"] if k_.startswith("realign_strip_noise")]
    rows = []
    if len(nz) >= 3:
        e0 = [m["err_deg"] for m in nz]
        rows = [["$\\sigma=0.1\\,d_{NN}$, no deletion", len(e0), fmt(st.median(e0), 4), fmt(q(e0, .9), 3), fmt(100 * mean([x < 0.1 for x in e0]), 0) + "\\%", fmt(100 * mean([x < 1 for x in e0]), 0) + "\\%"]]
        for fr in ("0.01", "0.05"):
            e = [m["drop"][fr]["err_deg"] for m in nz if fr in (m.get("drop") or {})]
            if e:
                rows.append([f"$+$ {float(fr) * 100:g}\\% node deletion", len(e), fmt(st.median(e), 4), fmt(q(e, .9), 3), fmt(100 * mean([x < 0.1 for x in e]), 0) + "\\%", fmt(100 * mean([x < 1 for x in e]), 0) + "\\%"])
        dre = [g(d, m["cost"]) - g(d, m["oracle_cost"]) for d, m in nzd]
        dmis = [g(d, m["mis_cost"]) - g(d, m["oracle_cost"]) for d, m in nzd]
        tex_table(os.path.join(T, "T5b_gurultu.tex"),
                  r"Realignment under field noise: random $\varphi\sim U(-45^\circ,45^\circ)$, coordinate Gaussian noise $\sigma=0.1\times$ median neighbor distance, and point deletion; snap angle error $|\hat\theta-\varphi|$ (degrees).",
                  "tab:gurultu", ["Condition", "$N$", "median", "p90", "$<0.1^\\circ$", "$<1^\\circ$"], rows,
                  f"Tour constructed on noisy coordinates, cost evaluated on true coordinates. Tour cost gap difference between realigned strip and true-angle (oracle) strip averages {mean(dre):.2f} points (median $|$diff$|$ {st.median([abs(x) for x in dre]):.3f}); difference between unaligned strip and oracle is {mean(dmis):.1f} points (negative = misaligned strip is shorter; constructive paradox, Step~4).")

    # ================= T6: Misalignment and realignment ======================
    rows = []
    for meth in ("ge8", "strip", "hilbert", "morton"):
        e = eff[meth]
        if not e["orig"]:
            continue
        dm = [a - b for a, b in zip(e["mis"], e["orig"])]
        dr = [a - b for a, b in zip(e["re"], e["orig"])]
        nm = {"ge8": "Greedy-Edge ($k{=}8$)", "strip": "Strip", "hilbert": "Hilbert", "morton": "Morton"}[meth]
        rows.append([nm, len(dm), pct(mean(e["orig"])), pct(mean(e["mis"])), fmt(mean(dm), 2), fmt(mean([abs(x) for x in dm]), 2), ptex(wilcoxon(dm)),
                     fmt(mean([abs(x) for x in dr]), 3), fmt(100 * mean([abs(x) < 1e-9 for x in dr]), 0) + "\\%"])
    tex_table(os.path.join(T, "T6_hizasizlik.tex"),
              r"Cost impact of misalignment and autonomous realignment (gap \%, mean; instances $\times$ 3 angles).",
              "tab:hizasizlik", ["Constructor", "trials", "original", "misaligned", "diff", "$|$diff$|$", "$p$", "$|$realigned$-$orig.$|$", "exact"], rows,
              r"Original: $\theta=0^\circ$; Misaligned: rotated by $\varphi\in\{7.5^\circ, 22.5^\circ, 37.5^\circ\}$; Realigned: after autonomous snap. Invariant GE is virtually unaffected ($p=0.161$); Strip cost decreases under rotation due to construction paradox ($p<10^{-4}$). Exact: fraction of instances where snap produces bit-identical tours to original.")

    # ================= T7: Stitched banding + RSGE/RGGE ======================
    rows = []
    bset = sorted({b["b"] for d in D if "fs_band_ge8" in d["rows"] for b in d["rows"]["fs_band_ge8"]["bands"] if not b.get("tag")})
    for b in bset + ["ks"]:
        pr = []
        for d in D:
            m = d["rows"].get("fs_band_ge8")
            if not m:
                continue
            g1 = next((x for x in m["bands"] if x["b"] == 1 and not x.get("tag")), None)
            gb = next((x for x in m["bands"] if x.get("tag") == "ks"), None) if b == "ks" else next((x for x in m["bands"] if x["b"] == b and not x.get("tag")), None)
            if g1 and gb:
                pr.append((g(d, g1["cost"]), g(d, gb["cost"]), gap_of(d, "strip")))
        if len(pr) < 3:
            continue
        dg = [y - x for x, y, _ in pr]
        w, l, p = sign_test(dg)
        lab = b if b != "ks" else "$k_s{=}\\mathrm{round}\\sqrt{n/2}$"
        rows.append([lab, len(pr), pct(mean([y for _, y, _ in pr])), fmt(mean(dg), 2), fmt(st.median(dg), 2), f"{w}/{len(pr)}", ptex(p)])
    rows.append("MIDRULE")
    for rid, nm in (("rsge_fixed2", "RSGE $b{=}2$, lattice angle"), ("rsge_fixed2@rotation_strip", "RSGE $b{=}2$, strip angle"),
                    ("rsge_corridor", "RSGE corridor"), ("rgge_x", "RGGE, strip angle"), ("rgge_x@zero", r"RGGE, $\theta=0^\circ$ (anchor)")):
        pr = [(gap_of(d, "greedy_edge@knn8_greedy"), gap_of(d, rid), d["rows"][rid].get("rsge_b")) for d in D if rid in d["rows"]]
        if len(pr) < 3:
            continue
        dg = [y - x for x, y, _ in pr]
        w, l, p = sign_test(dg)
        bs = [b for _, _, b in pr if b is not None]
        extra = f" ($b{{=}}1$: {sum(1 for b in bs if b == 1)}/{len(bs)})" if bs and "corridor" in rid else ""
        rows.append([nm + extra, len(pr), pct(mean([y for _, y, _ in pr])), fmt(mean(dg), 2), fmt(st.median(dg), 2), f"{w}/{len(pr)}", ptex(p)])
    tex_table(os.path.join(T, "T7_bantlama.tex"),
              "Stitched banding ($\\theta=0^\\circ$, internal GE $k{=}8$, $b$ equal-width bands stitched into single tour) and RSGE/RGGE family lines: difference from global GE ($k{=}8$).",
              "tab:bant", ["$b$ / Line", "$N$", "gap", "diff (pts)", "median diff", "wins", "$p$ (sign)"], rows,
              "Wins: instances where the line achieves lower cost than GE. $k_s$: strip heuristic's own band count (\\texttt{snake\\_order}); tests banded GE at the same band count as strip. For $b=32$, $N=97$ (two small instances with $n<250$ skipped because $n/b<8$). RSGE resource/budget rules set $b=1$ for $n<20\\,000$ (identical to GE).")

    # ================= T8: Parallel GE =======================================
    def cells(d):
        r = d["rows"]
        out = {}
        m = r.get("pge_ksweep")
        if m and m.get("ksweep"):
            Lg = m["cost"]
            for c in m["ksweep"]:
                out[(c["strategy"], c["k_requested"])] = dict(mk=c["makespan"] / (Lg / c["k_requested"]), tot=c["total"] / Lg, imb=c["imbalance"], t_par=c["t_par"])
        for s, _ in PGE_S:
            for k in (2, 3, 4, 6, 8, 12, 16):
                mm = r.get(f"pge_{s}_k{k}")
                if mm and mm.get("makespan_ratio") is not None:
                    out[(s, k)] = dict(mk=mm["makespan_ratio"], tot=mm["total"] / mm["ge_global_cost"], imb=mm["imbalance"], t_par=mm["t_par"], speed=mm["speedup"])
        return out
    CE = [(d, cells(d)) for d in D]
    ks = sorted({k for _, c in CE for (_, k) in c})
    rows = []
    for s, nm in PGE_S:
        for k in ks:
            v = [c[(s, k)] for _, c in CE if (s, k) in c]
            if len(v) < 3:
                continue
            sp = [x["speed"] for x in v if "speed" in x]
            rows.append([nm if k == ks[0] else "", k, len(v), fmt(mean([x["mk"] for x in v]), 3) + " / " + fmt(st.median([x["mk"] for x in v]), 3),
                         fmt(mean([x["tot"] for x in v]), 3), fmt(mean([x["imb"] for x in v]), 2), fmt(st.median(sp), 1) if sp else "--"])
        rows.append("MIDRULE")
    if rows and rows[-1] == "MIDRULE":
        rows.pop()
    tex_table(os.path.join(T, "T8_paralel.tex"),
              "Parallel GE: partitioning strategy $\\times$ $k$ vehicles. Makespan ratio $=\\max_i L_i/(L_{\\mathrm{GE}}/k)$ (1.0 ideal), total ratio $=\\sum_i L_i/L_{\\mathrm{GE}}$, imbalance $=\\max_i L_i/\\overline{L}$, speedup $=t_{\\mathrm{GE}}/t_{\\mathrm{par}}$.",
              "tab:paralel", ["Partitioning", "$k$", "$N$", "makespan mean / med.", "total", "imbalance", "speedup"], rows,
              "Band angle from lattice detector (grid\\_theta). Speedup: $t_{\\mathrm{GE}}/t_{\\mathrm{par}}$ in pure Python single-thread, $t_{\\mathrm{par}} = t_{\\text{part}} + \\max_j t_j$.")

    # ================= T8b: Winners by k =====================================
    rows = []
    for k in ks:
        wins = defaultdict(int); dkb = []
        for _, c in CE:
            cs = {s: c[(s, k)]["mk"] for s, _ in PGE_S if (s, k) in c}
            if len(cs) < 4:
                continue
            wins[min(cs, key=cs.get)] += 1
            dkb.append(cs["kmeans"] - cs["band"])
        if not dkb:
            continue
        w, l, p = sign_test(dkb)
        rows.append([k, len(dkb)] + [wins.get(s, 0) for s, _ in PGE_S] + [fmt(mean(dkb), 3) + f" ($p{prel(p)}$)"])
    tex_table(os.path.join(T, "T8b_paralel_kazanan.tex"),
              "Partitioning strategy achieving minimum makespan for each $k$ (instance count) and k-means $-$ band difference (sign test $p$).",
              "tab:paralelkazanan", ["$k$", "$N$"] + [nm.split(" (")[0] for _, nm in PGE_S] + ["k-means $-$ band"], rows)

    # ================= T8c: Band frame variants ==============================
    rows = []
    for k in (2, 4, 8):
        base_k = f"pge_band_k{k}"
        for suf, nm in (("@zero", "$\\theta=0^\\circ$"), ("@rotation_strip", "strip angle"), ("@manual:22.5", "$22.5^\\circ$ (unaligned)")):
            pr = [(d["rows"][base_k]["makespan_ratio"], d["rows"][base_k + suf]["makespan_ratio"]) for d in D
                  if base_k in d["rows"] and base_k + suf in d["rows"] and d["rows"][base_k].get("makespan_ratio") is not None]
            if len(pr) < 3:
                continue
            dg = [y - x for x, y in pr]
            w, l, p = sign_test([-x for x in dg])
            rows.append([k, nm, len(pr), fmt(mean(dg), 3), f"{w}/{len(pr)}", ptex(p)])
    tex_table(os.path.join(T, "T8c_bant_cerceve.tex"),
              "Frame dependence of band partitioning: makespan ratio difference relative to lattice orientation.",
              "tab:bantcerceve", ["$k$", "Frame", "$N$", "diff", "worse", "$p$"], rows,
              "Positive difference indicates larger makespan under variant coordinate frame.")

    # ================= T8d: k-means++ seed sensitivity =======================
    rows = []
    for k in (2, 4, 8):
        per = []
        for d in D:
            m = d["rows"].get("pge_kmseed")
            if not m or not m.get("kmseed"):
                continue
            v = [x["makespan_ratio"] for x in m["kmseed"] if x["k"] == k]
            if len(v) >= 2:
                band = (d["rows"].get(f"pge_band_k{k}") or {}).get("makespan_ratio")
                kd = (d["rows"].get(f"pge_kd_k{k}") or {}).get("makespan_ratio")
                per.append(dict(mean=mean(v), rng=max(v) - min(v), sd=st.pstdev(v), best=min(v), worst=max(v), band=band, kd=kd))
        if len(per) < 3:
            continue
        wb = [x for x in per if x["band"] is not None]
        wk = [x for x in per if x["kd"] is not None]
        rows.append([k, len(per), fmt(mean([x["mean"] for x in per]), 3), fmt(mean([x["sd"] for x in per]), 3), fmt(mean([x["rng"] for x in per]), 3),
                     (fmt(100 * mean([x["best"] < x["band"] for x in wb]), 0) + "\\% / " + fmt(100 * mean([x["worst"] < x["band"] for x in wb]), 0) + "\\%") if wb else "--",
                     fmt(100 * mean([x["best"] < x["kd"] for x in wk]), 0) + "\\%" if wk else "--"])
    if rows:
        tex_table(os.path.join(T, "T8d_kmeans_tohum.tex"),
                  "Sensitivity to k-means$++$ initialization seed: makespan ratio across 5 seeds (per-instance mean, inter-seed std.\\ dev.\\ and range) and fraction beating band and k-d.",
                  "tab:kmeanstohum", ["$k$", "$N$", "mean", "std", "range", "beats band (best / worst)", "beats k-d (best)"], rows,
                  "Band and k-d are deterministic single-seed methods; k-means remains inferior to k-d on average even under best-of-5 seed selection.")

    # ================= T9: Repair and ILS ====================================
    rows = []
    for rid, nm in (("repair_vnd", "VND $\\leftarrow$ Greedy-Edge"), ("repair_vnd@strip", "VND $\\leftarrow$ Strip"),
                    ("repair_vnd@rsge_fixed2", "VND $\\leftarrow$ RSGE $b{=}2$"), ("repair_vnd@realign_strip", "VND $\\leftarrow$ Realigned strip"),
                    ("ils@repair_vnd", "ILS $\\leftarrow$ VND(GE)"), ("ils", "ILS $\\leftarrow$ Greedy-Edge"), ("ils@rsge_fixed2", "ILS $\\leftarrow$ RSGE $b{=}2$")):
        pr = [(gap_of(d, rid), g(d, d["rows"][rid]["initial_cost"]) if d["rows"][rid].get("initial_cost") else None, d["rows"][rid]["time"]) for d in D if rid in d["rows"]]
        if len(pr) < 3:
            continue
        fin = [x for x, _, _ in pr]; ini = [y for _, y, _ in pr if y is not None]
        rows.append([nm, len(pr), pct(mean(ini)) if ini else "--", pct(mean(fin)), pct(st.median(fin)), fmt(st.median([t for _, _, t in pr]), 1)])
    if rows:
        tex_table(os.path.join(T, "T9_onarim.tex"),
                  "Do construction differences persist after deep local search? Same repair (VND) and ILS across different initial tours.",
                  "tab:onarim", ["Line", "$N$", "initial gap", "final gap mean", "median", "time (s)"], rows,
                  "Initial tour for VND $\\leftarrow$ GE is default $k{=}15$ GE (Table~\\ref{tab:kurucular}, 17.2\\%); not $k{=}8$. ILS evaluated only on $n\\le 1000$ due to runtime budget; perturbation is adaptive (double-bridge or subpath reversal depending on stagnation).")

    # ================= T10: Single-tour frame race ===========================
    def gap_seedmean(d, rid):
        m = d["rows"].get(rid)
        if not m:
            return None
        stc = m.get("stochastic") or {}
        return g(d, stc["mean"]) if stc.get("mean") is not None else g(d, m["cost"])
    stages = [("insa", "strip", "rotation_strip", "Construction (strip)", gap_of),
              ("vnd", "repair_vnd@strip", "repair_vnd@rotation_strip", "$+$ composite repair (VND)", gap_of),
              ("ils", "ils@repair_vnd@strip", "ils@repair_vnd@rotation_strip", "$+$ VND $+$ ILS (seed mean)", gap_seedmean),
              ("ilsbest", "ils@repair_vnd@strip", "ils@repair_vnd@rotation_strip", "$+$ VND $+$ ILS (best seed)", gap_of)]
    race_rows = []
    race = {}
    for key, ra, rs, nm, gf in stages:
        pr = [(gf(d, ra), gf(d, rs)) for d in D if ra in d["rows"] and rs in d["rows"]]
        if len(pr) < 3:
            continue
        dg = [a - s_ for a, s_ in pr]
        w, l, p = sign_test(dg)
        race[key] = dict(n=len(pr), aligned=mean([a for a, _ in pr]), sparse=mean([s_ for _, s_ in pr]),
                         diff=mean(dg), med=st.median(dg), aligned_wins=w, sparse_wins=l, p=p, wil=wilcoxon(dg))
        race_rows.append([nm, len(pr), pct(race[key]["aligned"]), pct(race[key]["sparse"]), fmt(mean(dg), 2), fmt(st.median(dg), 2),
                          f"{w} / {l}", ptex(p)])
    if race_rows:
        tex_table(os.path.join(T, "T10_yaris_tekli.tex"),
                  r"Single-tour frame race: lattice-aligned ($\theta=0^\circ$) strip vs.\ sparse-sweep (\texttt{rotation\_strip}) strip, across improvement depth (gap \%).",
                  "tab:yaristekli", ["Stage", "$N$", "aligned", "sparse", "diff", "median", "a / s wins", "$p$"], race_rows,
                  "Both frames undergo identical composite repair (VND: 2-opt $+$ Or-opt $+$ relocate) and subsequent ILS; ILS evaluated for $n\\le3000$ with 10 seeds ($n\\le2500$) and 5 seeds above; seed mean is primary measure, best seed shown separately. Negative difference indicates aligned frame advantage; $p$ from two-sided sign test.")

    # ================= T11: Parallel frame race ==============================
    prows = []
    prace = {}
    for k in (2, 4, 8):
        base_k = f"pgr_band_k{k}"
        for suf, nm in (("@rotation_strip", "sparse"), ("@zero", r"$\theta=0^\circ$"), ("@manual:22.5", "$22.5^\\circ$")):
            pr = []
            for d in D:
                a = d["rows"].get(base_k)
                v = d["rows"].get(base_k + suf)
                if a and v and a.get("makespan_ratio_rep") is not None and v.get("makespan_ratio_rep") is not None:
                    pr.append((a["makespan_ratio"], v["makespan_ratio"], a["makespan_ratio_rep"], v["makespan_ratio_rep"]))
            if len(pr) < 3:
                continue
            d_before = [va - aa for aa, va, _, _ in pr]
            d_after = [vr - ar for _, _, ar, vr in pr]
            wb, lb, pb = sign_test([-x for x in d_before])
            wa, la, pa = sign_test([-x for x in d_after])
            prace[(k, suf)] = dict(n=len(pr), aligned_before=mean([a for a, _, _, _ in pr]), var_before=mean([v for _, v, _, _ in pr]),
                                   aligned_after=mean([a for _, _, a, _ in pr]), var_after=mean([v for _, _, _, v in pr]),
                                   d_before=mean(d_before), d_after=mean(d_after), aligned_wins_before=wb, aligned_wins_after=wa,
                                   p_before=pb, p_after=pa)
            e = prace[(k, suf)]
            prows.append([k, nm, len(pr), fmt(e["aligned_before"], 3) + " $\\to$ " + fmt(e["aligned_after"], 3),
                          fmt(e["var_before"], 3) + " $\\to$ " + fmt(e["var_after"], 3),
                          f"{wb} $\\to$ {wa}", ptex(pa)])
    if prows:
        tex_table(os.path.join(T, "T11_yaris_paralel.tex"),
                  "Parallel frame race: band partitioning + GE + per-band VND. Makespan ratio for lattice-aligned vs.\\ variant frame; before $\\to$ after repair.",
                  "tab:yarisparalel", ["$k$", "Variant", "$N$", "aligned", "variant", "aligned wins", "$p$ (after)"], prows,
                  "Winner: lower makespan ratio; numbers indicate before $\\to$ after per-band VND repair. Repair applies identical VND operator and budget to each piece.")

    # ================= T11b: Post-repair partitioning ========================
    kk_rows = []
    for k in (2, 4, 8):
        pr = [(d["rows"][f"pgr_band_k{k}"]["makespan_ratio_rep"], d["rows"][f"pgr_kmeans_k{k}"]["makespan_ratio_rep"],
               (d["rows"].get(f"pgr_kd_k{k}") or {}).get("makespan_ratio_rep")) for d in D
              if f"pgr_band_k{k}" in d["rows"] and f"pgr_kmeans_k{k}" in d["rows"]
              and d["rows"][f"pgr_band_k{k}"].get("makespan_ratio_rep") is not None and d["rows"][f"pgr_kmeans_k{k}"].get("makespan_ratio_rep") is not None]
        if len(pr) < 3:
            continue
        dg = [b - m for b, m, _ in pr]
        w, l, p = sign_test(dg)
        kd = [(b, m, x) for b, m, x in pr if x is not None]
        kd_cell, kd_win = "--", "--"
        if len(kd) >= 3:
            dk_ = [b - x for b, _, x in kd]
            wk, lk, pk = sign_test(dk_)
            dkm = [m - x for _, m, x in kd]
            wkm, lkm, pkm = sign_test(dkm)
            kd_cell = fmt(mean([x for _, _, x in kd]), 3)
            kd_win = f"{wk}/{lk} ({ptex(pk)})"
            km_kd_win = f"{wkm}/{lkm} ({ptex(pkm)})"
        kk_rows.append([k, fmt(mean([b for b, _, _ in pr]), 3), fmt(mean([m for _, m, _ in pr]), 3), kd_cell,
                        f"{w}/{l} ({ptex(p)})", kd_win, km_kd_win])
    if kk_rows:
        tex_table(os.path.join(T, "T11b_yaris_kmeans.tex"),
                  "Post-repair partitioning comparison: band vs.\\ k-means and k-d (makespan ratio after per-band VND).",
                  "tab:yariskmeans", ["$k$", "band", "k-means", "k-d", "band vs.\\ k-means", "k-d vs.\\ band", "k-d vs.\\ k-means"], kk_rows,
                  "All strategies repaired with per-piece VND. Pairwise sign test wins/losses and $p$-value.")

    # ================= ENGLISH FIGURES =======================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif"})
    COL = {"fs_ge8": "#1f4e79", "fs_nn": "#2e7d32", "fs_strip": "#c62828", "fs_hilbert": "#ef6c00", "fs_morton": "#6a1b9a", "fs_band2ge": "#795548"}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(F, name + ".pdf"))
        fig.savefig(os.path.join(F, name + ".png"), dpi=160)
        plt.close(fig)

    # F1: angle curves
    picks = [d for d in D if "fs_strip" in d["rows"] and "fs_ge8" in d["rows"]]
    if picks:
        sel = [min(picks, key=lambda d: abs(d["n"] - 1000)), max(picks, key=lambda d: d["n"])]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, d in zip(axes, sel):
            for fk in ("fs_ge8", "fs_nn", "fs_strip", "fs_hilbert", "fs_morton", "fs_band2ge"):
                m = d["rows"].get(fk)
                if not m or not m.get("sweep"):
                    continue
                pts = sorted((float(a), g(d, c)) for a, c in m["sweep"].items())
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=2, lw=1, color=COL[fk],
                        label={"fs_ge8": "Greedy-Edge", "fs_nn": "NN (exact)", "fs_strip": "Strip", "fs_hilbert": "Hilbert", "fs_morton": "Morton", "fs_band2ge": "2-band GE"}[fk])
            ax.set_title(f"{d['name']} ($n$={d['n']})")
            ax.set_xlabel("Rotation angle $\\theta$ (°)")
            ax.set_ylabel("Gap (%)")
        axes[0].legend(fontsize=6, ncol=2)
        save(fig, "F1_aci_egrileri")

    # F2: boxplot
    order = [fk for fk, _, _ in SWEEP if any(fk in d["rows"] for d in D)]
    data = [[d["rows"][fk]["range_pct"] for d in D if fk in d["rows"] and d["rows"][fk].get("range_pct") is not None] for fk in order]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bp = ax.boxplot([x if x else [np.nan] for x in data], showfliers=False, patch_artist=True,
                    medianprops=dict(color="#d95f02", lw=1.3),
                    boxprops=dict(lw=1.0),
                    whiskerprops=dict(lw=0.9),
                    capprops=dict(lw=1.0))
    for b, fk in zip(bp["boxes"], order):
        cls_ = dict((a, c) for a, _, c in SWEEP)[fk]
        b.set(facecolor="#c6dbef" if cls_ == "inv" else ("#fed9a6" if cls_ == "approx" else "#fcbba1"))
    ax.set_xticks(range(1, len(order) + 1))
    en_labels = {"fs_ge8": "GE\n($k{=}8$)", "fs_ge15": "GE\n($k{=}15$)", "fs_nn": "NN\n(exact)", "fs_fi": "Farthest\nIns.",
                 "fs_nn_grid": "NN\n(grid)", "fs_strip": "Strip", "fs_hilbert": "Hilbert", "fs_morton": "Morton", "fs_band2ge": "2-band\nGE"}
    ax.set_xticklabels([en_labels.get(fk, fk) for fk in order], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 85)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_ylabel("Angle range (%)", fontsize=8.5)
    n_inv = sum(1 for fk in order if dict((a, c) for a, _, c in SWEEP)[fk] == "inv")
    ax.axvline(n_inv + 0.5, color="k", ls="--", lw=0.8)
    save(fig, "F2_duyarlilik_kutu")

    # F3: rot vs jitter
    C_f3 = [(d["rows"]["fs_ge8"]["range_pct"], d["rows"]["fs_ge8_jitter"]["range_pct"], d["rows"]["fs_ge8_detied"]["range_pct"], d["rows"]["fs_ge8_detied"].get("tie_frac_before", 0), d["n"])
            for d in D if all(k in d["rows"] for k in ("fs_ge8", "fs_ge8_jitter", "fs_ge8_detied"))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    ax.scatter([c[1] for c in C_f3], [c[0] for c in C_f3], s=14, alpha=0.75, color="#1f4e79", edgecolors="none")
    lim = [0.35, 20]
    ax.plot(lim, lim, "k--", lw=0.9, label="$y=x$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.35, 20); ax.set_ylim(0.35, 20)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xlabel("Jitter range at $\\theta=0^\\circ$ (%)", fontsize=8.5); ax.set_ylabel("Rotation sweep range (%)", fontsize=8.5)
    ax.set_title("GE: Rotation = Tie-breaking", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1]
    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.scatter([c[4] for c in C_f3], [c[0] for c in C_f3], s=14, label="Rotation", color="#1f4e79", alpha=0.75)
    ax.scatter([c[4] for c in C_f3], [c[2] for c in C_f3], s=18, marker="x", lw=1.2, label="Detied", color="#c62828", alpha=0.85)
    ax.set_xscale("log"); ax.set_xlim(80, 150000); ax.set_ylim(-0.6, 13.5)
    ax.set_xlabel("$n$ (instance size)", fontsize=8.5); ax.set_ylabel("GE angle range (%)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Range decays with $n$; $\\approx 0$ when detied", fontsize=9)
    save(fig, "F3_ge_beraberlik")

    # F4: realignment CDF
    errs = defaultdict(list)
    for d in D:
        for suf, _ in PHIS:
            m = d["rows"].get("realign_ge8" + suf)
            if m:
                for dk, x in (m.get("det") or {}).items():
                    errs[dk].append(max(x["err"], 1e-3))
                errs["snap"].append(max(m["err_deg"], 1e-3))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for dk, lab, c in (("comb", "Comb projection (grid_theta)", "#1f4e79"), ("nndir", "NN edge direction", "#2e7d32"), ("pca", "PCA", "#7f7f7f"), ("snap", "Snap", "#c62828")):
        e = sorted(errs.get(dk, []))
        if e:
            ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", label=lab, color=c)
    ax.set_xscale("log"); ax.set_xlim(8e-4, 50); ax.set_xlabel("$|\\hat\\theta-\\varphi|$ (°, 0.001 baseline)"); ax.set_ylabel("Fraction of trials (CDF)")
    ax.legend(fontsize=7)
    save(fig, "F4_hizalama_cdf")

    # F5: banding + parallel
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    bset_f5 = sorted({b["b"] for d in D if "fs_band_ge8" in d["rows"] for b in d["rows"]["fs_band_ge8"]["bands"] if not b.get("tag")})
    ys = [mean([g(d, x["cost"]) for d in D if "fs_band_ge8" in d["rows"] for x in d["rows"]["fs_band_ge8"]["bands"] if x["b"] == b and not x.get("tag")]) for b in bset_f5]
    ax.plot(bset_f5, ys, "-o", color="#795548"); ax.set_xscale("log", base=2); ax.set_xlabel("Bands $b$ (stitched into single tour)"); ax.set_ylabel("Mean gap (%)"); ax.set_title("Stitched Banding")
    ax = axes[1]
    COLP = {"band": "#b45309", "kd": "#7c2d12", "kmeans": "#0f766e", "split": "#475569"}
    for s, nm in PGE_S:
        pts = []
        for k in (2, 3, 4, 6, 8, 12, 16):
            v = [c[(s, k)]["mk"] for _, c in CE if (s, k) in c]
            if v:
                pts.append((k, mean(v)))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3, color=COLP[s], label=nm)
    ax.axhline(1.0, color="k", ls="--", lw=.8); ax.set_xscale("log", base=2); ax.set_xlabel("$k$ vehicles"); ax.set_ylabel("Makespan / ($L_{\\mathrm{GE}}/k$)"); ax.set_title("Parallel GE"); ax.legend(fontsize=6.5)
    save(fig, "F5_bant_paralel")

    print("English tables and figures successfully generated in makale_v3/!")


if __name__ == "__main__":
    main()

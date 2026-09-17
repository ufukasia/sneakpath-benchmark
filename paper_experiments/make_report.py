# -*- coding: utf-8 -*-
"""frame_experiment.py ciktisindan makale tablolarini ve figurlerini uretir.

    python paper_experiments/make_report.py [--out paper_experiments/out]

Yazilanlar (out/report/):
    summary.md          -- anahtar sayilar, testler, okunabilir ozet
    tables/T*.md, .tex  -- makale tablolari (booktabs)
    figures/F*.png      -- figurler
    per_instance.csv    -- ornek basina duz tablo (ek malzeme)
    numbers.json        -- summary'de gecen tum sayilar

Istatistik (scipy yok): isaret testi (tam binom), Wilcoxon isaretli-sira
(normal yaklasim), Spearman (t yaklasimi), iki-orneklem KS (asimptotik).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

GRID_CONF_MIN = 0.6
INVARIANT = ("ge8", "ge15", "nn", "nn_grid", "fi")
DEPENDENT = ("strip", "hilbert", "morton", "band2ge")
PRETTY = {"ge8": "Greedy-Edge (k=8)", "ge15": "Greedy-Edge (k=15)",
          "nn": "Nearest Neighbour (exact)", "nn_grid": "Nearest Neighbour (grid-acc.)",
          "fi": "Farthest Insertion", "strip": "Strip (boustrophedon)",
          "hilbert": "Hilbert curve", "morton": "Morton curve",
          "band2ge": "2-band serpentine GE (RSGE b=2)"}
CLASS_PRETTY = {"vlsi": "VLSI", "tsplib": "TSPLIB"}


# ---------------------------------------------------------------------------
#  Istatistik
# ---------------------------------------------------------------------------
def sign_test(diffs, eps=1e-9):
    w = sum(1 for d in diffs if d < -eps)
    l = sum(1 for d in diffs if d > eps)
    m = w + l
    if m == 0:
        return w, l, 1.0
    k = min(w, l)
    p = sum(math.comb(m, i) for i in range(0, k + 1)) / 2 ** m * 2
    return w, l, min(1.0, p)


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon(diffs, eps=1e-9):
    d = [x for x in diffs if abs(x) > eps]
    n = len(d)
    if n < 6:
        return float("nan")
    ranks = _rankdata([abs(x) for x in d])
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (wp - mu) / sd
    return 2 * _norm_sf(abs(z))


def _rankdata(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return float("nan"), float("nan")
    rho = num / den
    if abs(rho) >= 1:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # t dagilimi -> normal yaklasimi (n>30 icin yeterli)
    p = 2 * _norm_sf(abs(t))
    return rho, p


def ks_2samp(a, b):
    a, b = sorted(a), sorted(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    allv = sorted(set(a) | set(b))
    i = j = 0
    d = 0.0
    for v in allv:
        while i < n1 and a[i] <= v:
            i += 1
        while j < n2 and b[j] <= v:
            j += 1
        d = max(d, abs(i / n1 - j / n2))
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, max(0.0, min(1.0, p))


def q(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    k = (len(s) - 1) * p
    f, c = int(math.floor(k)), int(math.ceil(k))
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}"


def pfmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
#  Tablo yazicilar
# ---------------------------------------------------------------------------
class Tables:
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)
        self.index = []

    def write(self, key, title, header, rows, note=""):
        md = [f"**{key}. {title}**", "", "| " + " | ".join(header) + " |",
              "|" + "|".join("---" for _ in header) + "|"]
        for r in rows:
            md.append("| " + " | ".join(str(c) for c in r) + " |")
        if note:
            md += ["", note]
        txt = "\n".join(md) + "\n"
        open(os.path.join(self.d, f"{key}.md"), "w", encoding="utf-8").write(txt)
        cols = "l" + "r" * (len(header) - 1)
        tex = ["\\begin{table}[H]", "\\centering", f"\\caption{{{title}}}",
               f"\\label{{tab:{key}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
               " & ".join(header) + " \\\\", "\\midrule"]
        for r in rows:
            tex.append(" & ".join(str(c).replace("%", "\\%").replace("_", "\\_")
                                  for c in r) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        if note:
            tex.append("\\begin{flushleft}\\footnotesize " + note.replace("%", "\\%") + "\\end{flushleft}")
        tex.append("\\end{table}")
        open(os.path.join(self.d, f"{key}.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
        self.index.append((key, title))
        return txt


# ---------------------------------------------------------------------------
#  Yukleme ve turetilmis olculer
# ---------------------------------------------------------------------------
def load(out):
    rows, errs = [], []
    for f in sorted(os.listdir(os.path.join(out, "per_instance"))):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(out, "per_instance", f), encoding="utf-8"))
        if "error" in d:
            errs.append((d["name"], d["error"].strip().splitlines()[-1]))
            continue
        rows.append(d)
    rows.sort(key=lambda d: d["n"])
    return rows, errs


def derive(d):
    """Ornek basina turetilmis olculer."""
    bks = d["bks"]
    g = lambda c: 100.0 * (c / bks - 1.0)
    m = dict(name=d["name"], cls=d["cls"], n=d["n"], integer=d["integer_coords"],
             tie=d["tie"]["edge_tie_frac"], tie_boundary=d["tie"]["boundary"],
             comb0=d["detect_orig"]["comb"][0], comb0_conf=d["detect_orig"]["comb"][1],
             grid=d["detect_orig"]["comb"][1] >= GRID_CONF_MIN)
    m["gap0"] = {k: g(v["0"]) for k, v in d["sweep"].items() if "0" in v}
    m["range"], m["sd"], m["worst"], m["best"] = {}, {}, {}, {}
    for k, sw in d["sweep"].items():
        gs = [g(c) for c in sw.values()]
        if not gs:
            continue
        lo, hi = min(gs), max(gs)
        m["range"][k] = 100.0 * (max(sw.values()) - min(sw.values())) / min(sw.values())
        m["sd"][k] = st.pstdev(gs)
        g0 = m["gap0"].get(k, gs[0])
        m["worst"][k] = hi - g0
        m["best"][k] = g0 - lo
    jit = [g(c) for c in d["jitter"].values()]
    det = [g(c) for c in d["detied_sweep"].values()]
    rotg = [g(c) for c in d["sweep"]["ge8"].values()]
    m["ge_rot"], m["ge_jit"], m["ge_det"] = rotg, jit, det
    m["range_jit"] = 100.0 * (max(d["jitter"].values()) - min(d["jitter"].values())) / min(d["jitter"].values())
    m["range_det"] = 100.0 * (max(d["detied_sweep"].values()) - min(d["detied_sweep"].values())) / min(d["detied_sweep"].values())
    m["sd_jit"], m["sd_det"] = st.pstdev(jit), st.pstdev(det)
    m["best_jit"] = m["gap0"]["ge8"] - min(jit)
    m["ks_rot_jit"] = ks_2samp(rotg, jit)
    m["mean_rot_minus0"] = mean(rotg) - m["gap0"]["ge8"]
    m["realign"] = []
    for r in d["realign"]:
        e = dict(phi=r["phi"], mis={k: g(v) for k, v in r["mis"].items()}, det={})
        for dk, x in r["det"].items():
            e["det"][dk] = dict(theta=x["theta"], err=x["err"], conf=x["conf"],
                                re={k: g(v) for k, v in x["realigned"].items()})
        e["snap"] = None
        if r.get("snap"):
            s = dict(r["snap"])
            s["re"] = {k: g(v) for k, v in s["realigned"].items()}
            s["raw_resid"] = s["raw_resid_full"][s["coarse_from"]]
            e["snap"] = s
        m["realign"].append(e)
    m["bands"] = [dict(b=b["b"], nb=b["n_bands"], gap=g(b["cost"]), time=b["time"]) for b in d["bands"]]
    m["ge8_time"] = d["sweep_time"].get("ge8")
    m["angles"] = d["angles"]
    m["sweep_gap"] = {k: {a: g(c) for a, c in sw.items()} for k, sw in d["sweep"].items()}
    return m


# ---------------------------------------------------------------------------
#  Rapor
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    rep = os.path.join(args.out, "report")
    os.makedirs(rep, exist_ok=True)
    raw, errs = load(args.out)
    M = [derive(d) for d in raw]
    if not M:
        print("sonuc yok")
        return
    tabs = Tables(os.path.join(rep, "tables"))
    figdir = os.path.join(rep, "figures")
    os.makedirs(figdir, exist_ok=True)
    N = {}
    S = []          # summary satirlari

    def say(s=""):
        S.append(s)

    classes = ["all", "vlsi", "tsplib"]

    def sub(cls):
        return M if cls == "all" else [m for m in M if m["cls"] == cls]

    # ---------------- T1: ornek kumesi ----------------------------------
    rows = []
    for cls in classes:
        ms = sub(cls)
        if not ms:
            continue
        rows.append([CLASS_PRETTY.get(cls, "All"), len(ms),
                     f"{min(m['n'] for m in ms)}--{max(m['n'] for m in ms)}",
                     fmt(100 * mean([m["integer"] for m in ms]), 0) + "%",
                     fmt(st.median([m["tie"] for m in ms]), 3),
                     fmt(100 * mean([m["grid"] for m in ms]), 0) + "%",
                     fmt(100 * mean([abs(m["comb0"]) < 0.05 for m in ms if m["grid"]]), 0) + "%"
                     if any(m["grid"] for m in ms) else "--"])
    tabs.write("T1", "Instance set", ["Class", "Instances", "n range", "Integer coords",
                                       "Median tie fraction (k=8 candidate edges)",
                                       "Lattice detected (comb conf >= 0.6)",
                                       "Detected angle = 0 (among detected)"], rows)
    N["n_instances"] = len(M)
    N["n_vlsi"] = len(sub("vlsi"))
    N["n_tsplib"] = len(sub("tsplib"))
    N["errors"] = errs

    # ---------------- T2: aci duyarliligi -------------------------------
    rows = []
    sens = {}
    for k in INVARIANT + DEPENDENT:
        for cls in classes:
            ms = [m for m in sub(cls) if k in m["range"]]
            if not ms:
                continue
            r = [m["range"][k] for m in ms]
            sens[(k, cls)] = dict(n=len(ms), med=st.median(r), mean=mean(r), p90=q(r, 0.9),
                                  sd=mean([m["sd"][k] for m in ms]),
                                  worst=mean([m["worst"][k] for m in ms]),
                                  best=mean([m["best"][k] for m in ms]),
                                  gap0=mean([m["gap0"][k] for m in ms if k in m["gap0"]]))
            s = sens[(k, cls)]
            rows.append([PRETTY[k], CLASS_PRETTY.get(cls, "All"), s["n"], fmt(s["gap0"]),
                         fmt(s["med"]), fmt(s["mean"]), fmt(s["p90"]), fmt(s["sd"]),
                         fmt(s["worst"]), fmt(s["best"])])
    tabs.write("T2", "Angle sensitivity of constructors over the full [-90, 90) sweep",
               ["Constructor", "Class", "N", "Gap at 0 (%)", "Range median (%)",
                "Range mean (%)", "Range p90 (%)", "SD of gap (pts)",
                "Worst-angle loss (pts)", "Best-angle gain (pts)"], rows,
               "Range = 100 (max - min) / min of tour cost across angles. Loss/gain are relative to theta = 0.")
    N["sens"] = {f"{k}|{c}": v for (k, c), v in sens.items()}

    # ---------------- T3: GE beraberlik kontrolleri ---------------------
    rows = []
    for cls in classes:
        ms = sub(cls)
        if not ms:
            continue
        rot = [m["range"]["ge8"] for m in ms]
        jit = [m["range_jit"] for m in ms]
        det = [m["range_det"] for m in ms]
        ks_ok = mean([m["ks_rot_jit"][1] > 0.05 for m in ms])
        rows.append([CLASS_PRETTY.get(cls, "All"), len(ms),
                     f"{fmt(st.median(rot))} / {fmt(mean(rot))}",
                     f"{fmt(st.median(jit))} / {fmt(mean(jit))}",
                     f"{fmt(st.median(det), 3)} / {fmt(mean(det), 3)}",
                     fmt(100 * mean([r < 0.05 for r in det]), 0) + "%",
                     fmt(100 * ks_ok, 0) + "%",
                     fmt(mean([m["mean_rot_minus0"] for m in ms]), 3),
                     fmt(mean([m["best"]["ge8"] for m in ms])),
                     fmt(mean([m["best_jit"] for m in ms]))])
    tabs.write("T3", "Greedy-Edge: rotation sweep versus tie-breaking controls (equal sample sizes)",
               ["Class", "N", "Rotation range med/mean (%)", "Tie-jitter range med/mean (%)",
                "De-tied rotation range med/mean (%)", "De-tied range < 0.05%",
                "KS rot vs jitter not rejected (p>0.05)", "Mean(rot) - gap0 (pts)",
                "Best-of-angles gain (pts)", "Best-of-jitter gain (pts)"], rows,
               "Rotation range and tie-jitter range use the same number of samples per instance; de-tied = coordinates perturbed by 1e-4 x median NN distance (no exact ties remain).")
    allm = M
    rho_tie, p_tie = spearman([m["tie"] for m in allm], [m["range"]["ge8"] for m in allm])
    rho_rtm, p_rtm = spearman([m["gap0"]["ge8"] for m in allm], [m["best"]["ge8"] for m in allm])
    rho_mean0, p_mean0 = spearman([m["gap0"]["ge8"] for m in allm], [m["mean_rot_minus0"] for m in allm])
    d_or = [m["best"]["ge8"] - m["best_jit"] for m in allm]
    w, l, p_or = sign_test(d_or)
    N["ge"] = dict(spearman_tie_range=(rho_tie, p_tie), spearman_gap0_bestgain=(rho_rtm, p_rtm),
                   spearman_gap0_meanshift=(rho_mean0, p_mean0),
                   oracle_vs_jitter=dict(mean_best_rot=mean([m["best"]["ge8"] for m in allm]),
                                         mean_best_jit=mean([m["best_jit"] for m in allm]),
                                         sign=(w, l, p_or), wilcoxon_p=wilcoxon(d_or)),
                   frac_ks_ok=mean([m["ks_rot_jit"][1] > 0.05 for m in allm]),
                   detied_zero_frac=mean([m["range_det"] < 0.05 for m in allm]),
                   n_tied=sum(1 for m in allm if m["tie"] > 0.2))
    # beraberlik sinifina gore
    rows = []
    for lab, ms in (("tie fraction >= 0.5", [m for m in allm if m["tie"] >= 0.5]),
                    ("0.05 <= tie fraction < 0.5", [m for m in allm if 0.05 <= m["tie"] < 0.5]),
                    ("tie fraction < 0.05", [m for m in allm if m["tie"] < 0.05])):
        if not ms:
            continue
        rows.append([lab, len(ms), fmt(mean([m["range"]["ge8"] for m in ms])),
                     fmt(mean([m["range_jit"] for m in ms])),
                     fmt(mean([m["range_det"] for m in ms]), 3),
                     fmt(mean([m["range"]["strip"] for m in ms])),
                     fmt(mean([m["range"]["hilbert"] for m in ms]))])
    tabs.write("T3b", "Angle range by tie abundance", ["Tie class", "N", "GE rotation range (%)",
               "GE jitter range (%)", "GE de-tied range (%)", "Strip range (%)", "Hilbert range (%)"], rows)
    # boyuta gore: beraberlik gurultusu kucuk n'de buyuk (tek karar cok sey degistirir)
    rows = []
    for lo, hi in ((0, 500), (500, 2000), (2000, 10000), (10000, 10 ** 9)):
        ms = [m for m in allm if lo <= m["n"] < hi]
        if not ms:
            continue
        rows.append([f"{lo}--{hi - 1}" if hi < 10 ** 9 else f">= {lo}", len(ms),
                     fmt(st.median([m["range"]["ge8"] for m in ms])), fmt(st.median([m["range_jit"] for m in ms])),
                     fmt(st.median([m["range_det"] for m in ms]), 3),
                     fmt(100 * mean([m["ks_rot_jit"][1] > 0.05 for m in ms]), 0) + "%",
                     fmt(st.median([m["range"]["strip"] for m in ms])), fmt(st.median([m["range"]["hilbert"] for m in ms]))])
    tabs.write("T3c", "Angle range by instance size (medians)", ["n", "N", "GE rotation range (%)", "GE jitter range (%)",
               "GE de-tied range (%)", "KS rot vs jitter not rejected", "Strip range (%)", "Hilbert range (%)"], rows,
               "Tie-breaking noise is a relative effect: one different tie decision moves a small tour by more percent than a large one; the strip range does not shrink with n.")
    N["ge"]["spearman_n_range"] = spearman([m["n"] for m in allm], [m["range"]["ge8"] for m in allm])
    N["ge"]["spearman_rot_jit"] = spearman([m["range"]["ge8"] for m in allm], [m["range_jit"] for m in allm])
    # strip'in en iyi acisi kafes ekseninde mi?
    rows = []
    for cls in ("vlsi", "tsplib"):
        ms = sub(cls)
        best = []
        for m in ms:
            sw = m["sweep_gap"]["strip"]
            best.append(float(min(sw, key=sw.get)))
        on_axis = mean([min(abs(a), abs(abs(a) - 90)) <= 5.0 for a in best])
        rows.append([CLASS_PRETTY[cls], len(ms), fmt(mean([m["gap0"]["strip"] for m in ms])),
                     fmt(mean([m["gap0"]["strip"] - m["best"]["strip"] for m in ms])),
                     fmt(100 * on_axis, 0) + "%"])
        N[f"strip_best_on_axis_{cls}"] = on_axis
    tabs.write("T2b", "Strip: is the best sweep angle the lattice axis?", ["Class", "N", "Strip gap at theta=0 (%)",
               "Strip gap at best angle (%)", "Best angle within 5 deg of an axis"], rows,
               "For the strip heuristic the optimal frame is instance-specific and mostly NOT the lattice axis: a cost sweep (rotation_strip), not a geometric detector, is the right tool.")

    # ---------------- T4: dedektor dogrulugu ----------------------------
    rows = []
    det_stats = {}
    for cls in ("vlsi", "tsplib"):
        ms = sub(cls)
        for gridonly in (True, False):
            sel = [m for m in ms if (m["grid"] or not gridonly)]
            if not sel:
                continue
            for dk in ("comb", "nndir", "pca", "snap"):
                errsv = []
                for m in sel:
                    for r in m["realign"]:
                        if dk == "snap":
                            if r["snap"]:
                                errsv.append(r["snap"]["err"])
                        else:
                            errsv.append(r["det"][dk]["err"])
                if not errsv:
                    continue
                key = (cls, gridonly, dk)
                det_stats[key] = dict(n=len(errsv), med=st.median(errsv), p90=q(errsv, 0.9),
                                      lt1=mean([e < 1.0 for e in errsv]), lt05=mean([e < 0.5 for e in errsv]),
                                      lt01=mean([e < 0.1 for e in errsv]))
                s = det_stats[key]
                rows.append([CLASS_PRETTY[cls], "lattice detected" if gridonly else "all",
                             {"comb": "comb (projection histogram)", "nndir": "NN edge direction",
                              "pca": "PCA axis", "snap": "best coarse + hierarchical lattice refinement"}[dk],
                             s["n"], fmt(s["med"], 3), fmt(s["p90"], 2), fmt(100 * s["lt1"], 0) + "%",
                             fmt(100 * s["lt05"], 0) + "%", fmt(100 * s["lt01"], 0) + "%"])
    tabs.write("T4", "Re-alignment of deliberately misaligned instances: detector error |theta_hat - phi| (degrees, mod 90)",
               ["Class", "Subset", "Detector", "Trials", "Median err", "p90 err", "err < 1 deg",
                "err < 0.5 deg", "err < 0.1 deg"], rows,
               "Each instance is rotated by phi in {7.5, 22.5, 37.5, U(-45,45)} degrees; the detector sees only the rotated coordinates.")
    # snap exact recovery
    snap_rows = []
    for cls in ("vlsi", "tsplib"):
        sel = [m for m in sub(cls) if m["grid"] and m["integer"]]
        tot = ex = ident = cid = 0
        raw_res, ref_res = [], []
        for m in sel:
            for r in m["realign"]:
                s = r["snap"]
                if not s:
                    continue
                tot += 1
                ex += bool(s["exact_recovery"])
                ident += bool(s["ge8_tour_identical"])
                cid += bool(s["ge8_cost_identical"])
                raw_res.append(s["raw_resid"])
                ref_res.append(s["resid_mean"])
        if tot:
            snap_rows.append([CLASS_PRETTY[cls], len(sel), tot, fmt(mean(raw_res), 3),
                              fmt(mean(ref_res), 6), fmt(100 * ex / tot, 0) + "%",
                              fmt(100 * ident / tot, 0) + "%", fmt(100 * cid / tot, 0) + "%"])
            N[f"snap_{cls}"] = dict(instances=len(sel), trials=tot, exact=ex / tot, ident=ident / tot, cost_ident=cid / tot)
    tabs.write("T4b", "Exact lattice recovery after re-alignment (integer-coordinate instances with detected lattice)",
               ["Class", "Instances", "Trials", "Mean lattice residual, raw detector",
                "Mean lattice residual, refined", "Exact lattice recovered", "GE tour bit-identical to original",
                "GE cost identical"], snap_rows,
               "Residual = mean L1 distance of re-aligned coordinates to the nearest integer lattice point.")
    N["det_stats"] = {f"{c}|{'grid' if g else 'all'}|{d}": v for (c, g, d), v in det_stats.items()}

    # ---------------- T5: yeniden hizalamanin MALIYET etkisi ------------
    rows = []
    cost_eff = {}
    for cls in ("vlsi", "tsplib"):
        sel = [m for m in sub(cls)]
        for meth in ("ge8", "strip", "hilbert", "morton"):
            orig, mis, re_c, re_nn, re_sn = [], [], [], [], []
            for m in sel:
                g0 = m["gap0"][meth]
                for r in m["realign"]:
                    orig.append(g0)
                    mis.append(r["mis"][meth])
                    re_c.append(r["det"]["comb"]["re"][meth])
                    re_nn.append(r["det"]["nndir"]["re"][meth])
                    # onerilen hat: snap (tamsayi kafes) varsa o, yoksa nndir
                    re_sn.append(r["snap"]["re"][meth] if r["snap"] else r["det"]["nndir"]["re"][meth])
            if not orig:
                continue
            d_mis = [a - b for a, b in zip(mis, orig)]
            d_re = [a - b for a, b in zip(re_sn, orig)]
            d_fix = [a - b for a, b in zip(re_sn, mis)]
            w1, l1, p1 = sign_test(d_mis)
            w2, l2, p2 = sign_test(d_fix)
            cost_eff[(cls, meth)] = dict(orig=mean(orig), mis=mean(mis), re_comb=mean(re_c),
                                         re_nn=mean(re_nn), re=mean(re_sn), d_mis=mean(d_mis), d_re=mean(d_re),
                                         abs_mis=mean([abs(x) for x in d_mis]),
                                         d_fix=mean(d_fix), abs_re=mean([abs(x) for x in d_re]),
                                         frac_re_exact=mean([abs(x) < 1e-6 for x in d_re]),
                                         sign_mis=(w1, l1, p1), sign_fix=(w2, l2, p2),
                                         wil_mis=wilcoxon(d_mis), wil_fix=wilcoxon(d_fix), n=len(orig))
            e = cost_eff[(cls, meth)]
            rows.append([CLASS_PRETTY[cls], PRETTY[meth], e["n"], fmt(e["orig"]), fmt(e["mis"]),
                         fmt(e["d_mis"], 2), fmt(e["abs_mis"], 2), pfmt(e["wil_mis"]), fmt(e["re_comb"]), fmt(e["re_nn"]),
                         fmt(e["re"]), fmt(e["d_re"], 2), fmt(e["abs_re"], 2),
                         fmt(100 * e["frac_re_exact"], 0) + "%", fmt(e["d_fix"], 2), pfmt(e["wil_fix"])])
    tabs.write("T5", "Cost effect of misalignment and re-alignment (gap %, mean over instances x 4 rotations)",
               ["Class", "Constructor", "Trials", "Original (aligned)", "Misaligned", "Mis - orig", "|Mis - orig|",
                "p (Wilcoxon)", "Re-aligned: comb", "Re-aligned: NN-dir", "Re-aligned: lattice snap",
                "Snap - orig", "|Snap - orig|", "Snap = orig exactly", "Snap - mis", "p (Wilcoxon)"], rows,
               "Snap = best coarse detector + hierarchical lattice refinement + rounding to the integer lattice (falls back to NN-dir on non-integer instances). A frame-dependent constructor shows a large misalignment penalty that re-alignment removes; a frame-invariant one shows neither.")
    N["cost_eff"] = {f"{c}|{m}": v for (c, m), v in cost_eff.items()}

    # ---------------- T6: bantlama --------------------------------------
    rows = []
    band_stats = {}
    for cls in classes:
        ms = sub(cls)
        bset = sorted({b["b"] for m in ms for b in m["bands"]})
        for b in bset:
            pairs = []
            for m in ms:
                g1 = next((x for x in m["bands"] if x["b"] == 1), None)
                gb = next((x for x in m["bands"] if x["b"] == b), None)
                if g1 and gb:
                    pairs.append((g1, gb))
            if len(pairs) < 3:
                continue
            dg = [gb["gap"] - g1["gap"] for g1, gb in pairs]
            w, l, p = sign_test(dg)      # w = #(gap_b < gap_1) = b kazandi
            band_stats[(cls, b)] = dict(n=len(pairs), gap=mean([gb["gap"] for _, gb in pairs]),
                                        dgap=mean(dg), med=st.median(dg), wins=w, p=p,
                                        tratio=mean([gb["time"] / g1["time"] for g1, gb in pairs if g1["time"] > 0]))
            s = band_stats[(cls, b)]
            rows.append([CLASS_PRETTY.get(cls, "All"), b, s["n"], fmt(s["gap"]), fmt(s["dgap"], 2),
                         fmt(s["med"], 2), f"{s['wins']}/{s['n']}", pfmt(s["p"]), fmt(s["tratio"], 2)])
    tabs.write("T6", "Banding cost at the aligned frame (theta = 0): serpentine GE with b equal-width bands versus global GE (b = 1)",
               ["Class", "b", "N", "Mean gap (%)", "Gap - gap(b=1) (pts)", "Median diff", "b wins", "p (sign)",
                "Time / time(b=1)"], rows)
    N["bands"] = {f"{c}|{b}": v for (c, b), v in band_stats.items()}

    # ---------------- T7: buyuk-n RGGE/RSGE'nin sonuc dosyasi ozeti (opsiyonel) ----
    # (eski results/ kosumlari ayri; buraya alinmaz -- makale bu tek kosumdan beslenir)

    # ---------------- per-instance CSV ----------------------------------
    with open(os.path.join(rep, "per_instance.csv"), "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        hdr = ["name", "class", "n", "integer", "tie_frac", "comb_theta", "comb_conf", "lattice",
               "gap0_ge8", "range_ge8", "range_jit", "range_detied", "ks_p_rot_jit",
               "best_gain_ge8", "best_gain_jit"] + [f"range_{k}" for k in INVARIANT + DEPENDENT] + \
              [f"gap0_{k}" for k in INVARIANT + DEPENDENT] + \
              ["comb_err_mean", "nndir_err_mean", "pca_err_mean", "snap_exact_all", "ge8_time_s"]
        wtr.writerow(hdr)
        for m in M:
            snaps = [r["snap"] for r in m["realign"] if r["snap"]]
            wtr.writerow([m["name"], m["cls"], m["n"], int(m["integer"]), f"{m['tie']:.4f}",
                          f"{m['comb0']:.3f}", f"{m['comb0_conf']:.3f}", int(m["grid"]),
                          f"{m['gap0']['ge8']:.3f}", f"{m['range']['ge8']:.3f}", f"{m['range_jit']:.3f}",
                          f"{m['range_det']:.4f}", f"{m['ks_rot_jit'][1]:.3f}", f"{m['best']['ge8']:.3f}",
                          f"{m['best_jit']:.3f}"] +
                         [f"{m['range'][k]:.3f}" if k in m["range"] else "" for k in INVARIANT + DEPENDENT] +
                         [f"{m['gap0'][k]:.3f}" if k in m["gap0"] else "" for k in INVARIANT + DEPENDENT] +
                         [f"{mean([r['det'][dk]['err'] for r in m['realign']]):.3f}" for dk in ("comb", "nndir", "pca")] +
                         [int(all(s["exact_recovery"] for s in snaps)) if snaps else "",
                          f"{m['ge8_time']:.3f}" if m["ge8_time"] else ""])

    # ---------------- Figurler -----------------------------------------
    try:
        make_figures(M, figdir, N)
    except Exception as ex:      # figur hatasi raporu dusurmesin
        import traceback
        N["figure_error"] = traceback.format_exc()
        print("figur hatasi:", ex)

    # ---------------- summary.md ---------------------------------------
    say("# Frame-sensitivity experiment -- summary")
    say()
    say(f"Instances: {N['n_instances']} ({N['n_vlsi']} VLSI, {N['n_tsplib']} TSPLIB). "
        f"Errors: {len(errs)}" + (f" -> {errs}" if errs else "."))
    say()
    say("## 1. Frame-dependent vs frame-invariant constructors (T2)")
    for k in INVARIANT + DEPENDENT:
        s = sens.get((k, "all"))
        if s:
            say(f"- {PRETTY[k]}: range over angles median {s['med']:.2f}%, mean {s['mean']:.2f}%, "
                f"worst-angle loss {s['worst']:.2f} pts, best-angle gain {s['best']:.2f} pts (N={s['n']}).")
    say()
    say("## 2. Greedy-Edge: rotation is tie-breaking (T3, T3b)")
    g = N["ge"]
    say(f"- Rotation vs tie-jitter distributions indistinguishable (KS p>0.05) in "
        f"{100 * g['frac_ks_ok']:.0f}% of instances.")
    say(f"- De-tied instances: rotation range < 0.05% in {100 * g['detied_zero_frac']:.0f}% of instances.")
    say(f"- Spearman(tie fraction, GE rotation range) = {g['spearman_tie_range'][0]:.2f} (p={pfmt(g['spearman_tie_range'][1])}).")
    say(f"- Regression to the mean: Spearman(gap at 0, best-angle gain) = {g['spearman_gap0_bestgain'][0]:.2f} "
        f"(p={pfmt(g['spearman_gap0_bestgain'][1])}); Spearman(gap at 0, mean-over-angles minus gap0) = "
        f"{g['spearman_gap0_meanshift'][0]:.2f} (p={pfmt(g['spearman_gap0_meanshift'][1])}).")
    o = g["oracle_vs_jitter"]
    say(f"- Equal-budget oracle: best-of-angles gain {o['mean_best_rot']:.2f} pts vs best-of-jitter gain "
        f"{o['mean_best_jit']:.2f} pts; sign test angles better/worse = {o['sign'][0]}/{o['sign'][1]}, "
        f"p={pfmt(o['sign'][2])}; Wilcoxon p={pfmt(o['wilcoxon_p'])}.")
    say()
    say("## 3. Misaligned chips can be re-aligned (T4, T4b, T5)")
    for key, v in sorted(N["det_stats"].items()):
        say(f"- {key}: median err {v['med']:.3f} deg, p90 {v['p90']:.2f}, <1 deg {100 * v['lt1']:.0f}%, <0.1 deg {100 * v['lt01']:.0f}% (N={v['n']}).")
    for cls in ("vlsi", "tsplib"):
        s = N.get(f"snap_{cls}")
        if s:
            say(f"- {cls}: exact lattice recovery {100 * s['exact']:.0f}% of {s['trials']} trials; "
                f"GE tour bit-identical {100 * s['ident']:.0f}%; GE cost identical {100 * s['cost_ident']:.0f}%.")
    for cls in ("vlsi", "tsplib"):
        if f"strip_best_on_axis_{cls}" in N:
            say(f"- {cls}: strip's best sweep angle lies within 5 deg of a lattice axis in only "
                f"{100 * N[f'strip_best_on_axis_{cls}']:.0f}% of instances (T2b).")
    say(f"- Spearman(n, GE rotation range) = {N['ge']['spearman_n_range'][0]:.2f}; "
        f"Spearman(GE rotation range, jitter range) = {N['ge']['spearman_rot_jit'][0]:.2f} (T3c).")
    for key, e in N["cost_eff"].items():
        say(f"- {key}: aligned {e['orig']:.2f}%, misaligned {e['mis']:.2f}% (diff {e['d_mis']:+.2f}, |diff| {e['abs_mis']:.2f}, p={pfmt(e['wil_mis'])}), "
            f"re-aligned/snap {e['re']:.2f}% (snap-mis {e['d_fix']:+.2f}, p={pfmt(e['wil_fix'])}; |snap-orig| {e['abs_re']:.2f}; "
            f"snap==orig exactly {100 * e['frac_re_exact']:.0f}%).")
    say()
    say("## 4. Banding at the aligned frame (T6)")
    for key, b in sorted(N["bands"].items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[1]))):
        say(f"- {key}: gap {b['gap']:.2f}%, diff vs b=1 {b['dgap']:+.2f} pts, b wins {b['wins']}/{b['n']} (p={pfmt(b['p'])}), time ratio {b['tratio']:.2f}.")
    say()
    say("## Tables")
    for k, t in tabs.index:
        say(f"- {k}: {t}  (tables/{k}.md, tables/{k}.tex)")
    say()
    say("## Figures")
    for f in sorted(os.listdir(figdir)):
        say(f"- figures/{f}")
    open(os.path.join(rep, "summary.md"), "w", encoding="utf-8").write("\n".join(S) + "\n")
    json.dump(N, open(os.path.join(rep, "numbers.json"), "w", encoding="utf-8"), indent=1, default=str)
    print("\n".join(S))


# ---------------------------------------------------------------------------
#  Figurler
# ---------------------------------------------------------------------------
def make_figures(M, figdir, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    COL = {"ge8": "#1f77b4", "ge15": "#4c9be8", "nn": "#2ca02c", "nn_grid": "#98df8a", "fi": "#17becf",
           "strip": "#d62728", "hilbert": "#ff7f0e", "morton": "#9467bd", "band2ge": "#8c564b"}

    # F1: temsili orneklerde aci egrileri
    picks = []
    vl = [m for m in M if m["cls"] == "vlsi" and m["grid"]]
    ts = [m for m in M if m["cls"] == "tsplib"]
    if vl:
        picks.append(min(vl, key=lambda m: abs(m["n"] - 1000)))
        big = max(vl, key=lambda m: m["n"])
        if big is not picks[0]:
            picks.append(big)
    for nm in ("kroA100", "pcb3038", "rat783", "fnl4461", "d493"):
        c = [m for m in ts if m["name"] == nm]
        if c and len(picks) < 4 and c[0] not in picks:
            picks.append(c[0])
    picks = picks[:4]
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.4), squeeze=False)
        for ax, m in zip(axes[0], picks):
            for k in ("ge8", "nn", "strip", "hilbert", "morton", "band2ge"):
                if k not in m["sweep_gap"]:
                    continue
                pts = sorted(((float(a), g) for a, g in m["sweep_gap"][k].items()))
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=2.5, lw=1.2, color=COL[k], label=PRETTY[k])
            ax.set_title(f"{m['name']} (n={m['n']}, {m['cls'].upper()}, tie={m['tie']:.2f})")
            ax.set_xlabel("rotation angle (deg)")
            ax.set_ylabel("gap to BKS (%)")
        axes[0][0].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "F1_angle_curves.png"), dpi=170)
        plt.close(fig)

    # F2: yontem basina aci-aralik dagilimi (kutu), siniflara gore
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    order = [k for k in INVARIANT + DEPENDENT if any(k in m["range"] for m in M)]
    data_v = [[m["range"][k] for m in M if m["cls"] == "vlsi" and k in m["range"]] for k in order]
    data_t = [[m["range"][k] for m in M if m["cls"] == "tsplib" and k in m["range"]] for k in order]
    pos = np.arange(len(order))
    bp1 = ax.boxplot([d if d else [np.nan] for d in data_v], positions=pos - 0.18, widths=0.32, patch_artist=True,
                     showfliers=False)
    bp2 = ax.boxplot([d if d else [np.nan] for d in data_t], positions=pos + 0.18, widths=0.32, patch_artist=True,
                     showfliers=False)
    for b in bp1["boxes"]:
        b.set(facecolor="#9ecae1")
    for b in bp2["boxes"]:
        b.set(facecolor="#fdd0a2")
    ax.set_xticks(pos)
    ax.set_xticklabels([PRETTY[k].replace(" (", "\n(") for k in order], fontsize=6.5, rotation=20, ha="right")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("cost range over angles (%)  [symlog]")
    ax.axvline(len(INVARIANT) - 0.5 + (0 if len(order) == len(INVARIANT + DEPENDENT) else -1), color="k", lw=0.8, ls="--")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["VLSI", "TSPLIB"], loc="upper left", fontsize=8)
    ax.set_title("Angle sensitivity: frame-invariant (left) vs frame-dependent (right) constructors")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "F2_sensitivity_boxplot.png"), dpi=170)
    plt.close(fig)

    # F3: GE rotasyon vs jitter vs detied
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    for cls, c in (("vlsi", "#1f77b4"), ("tsplib", "#ff7f0e")):
        ms = [m for m in M if m["cls"] == cls]
        ax.scatter([max(m["range_jit"], 1e-3) for m in ms], [max(m["range"]["ge8"], 1e-3) for m in ms],
                   s=14, alpha=0.7, color=c, label=cls.upper())
    lim = [1e-3, max(max(m["range"]["ge8"] for m in M), max(m["range_jit"] for m in M)) * 1.3]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tie-jitter range at theta=0 (%)")
    ax.set_ylabel("rotation-sweep range (%)")
    ax.set_title("GE: rotation spread = tie-break spread")
    ax.legend(fontsize=8)
    ax = axes[1]
    ms = sorted(M, key=lambda m: m["tie"])
    ax.scatter([m["tie"] for m in ms], [m["range"]["ge8"] for m in ms], s=14, label="rotation sweep", color="#1f77b4")
    ax.scatter([m["tie"] for m in ms], [m["range_det"] for m in ms], s=14, marker="x", label="rotation sweep, de-tied", color="#d62728")
    ax.set_xlabel("tie fraction among k=8 candidate edges")
    ax.set_ylabel("GE cost range over angles (%)")
    ax.set_title("Removing exact ties removes the angle effect")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "F3_ge_tiebreak.png"), dpi=170)
    plt.close(fig)

    # F4: regresyon-ortalamaya: gap0 vs best-angle gain
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for cls, c in (("vlsi", "#1f77b4"), ("tsplib", "#ff7f0e")):
        ms = [m for m in M if m["cls"] == cls]
        ax.scatter([m["gap0"]["ge8"] for m in ms], [m["best"]["ge8"] for m in ms], s=14, alpha=0.75, color=c, label=cls.upper())
    ax.set_xlabel("GE gap at theta = 0 (%)")
    ax.set_ylabel("best-of-angles gain (pts)")
    rho = N["ge"]["spearman_gap0_bestgain"][0]
    ax.set_title(f"'Best angle' gain tracks how unlucky theta=0 was (rho={rho:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "F4_regression_to_mean.png"), dpi=170)
    plt.close(fig)

    # F5: yeniden hizalama -- dedektor hata CDF + maliyet cubuklari
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    vl = [m for m in M if m["cls"] == "vlsi" and m["grid"]]
    for dk, lab, c in (("comb", "comb (projection)", "#1f77b4"), ("nndir", "NN edge direction", "#2ca02c"),
                       ("pca", "PCA", "#7f7f7f")):
        e = sorted(max(r["det"][dk]["err"], 1e-3) for m in vl for r in m["realign"])
        if e:
            ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", label=lab, color=c)
    e = sorted(max(r["snap"]["err"], 1e-3) for m in vl for r in m["realign"] if r["snap"])
    if e:
        ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post",
                label="best coarse + lattice refinement (snap)", color="#d62728")
    ax.set_xscale("log")
    ax.set_xlim(8e-4, 50)
    ax.set_xlabel("|estimated - true rotation| (deg, mod 90; floored at 0.001)")
    ax.set_ylabel("CDF over trials")
    ax.set_title("Re-alignment accuracy (VLSI, lattice detected)")
    ax.legend(fontsize=7)
    ax = axes[1]
    labels, o, mi, re = [], [], [], []
    for meth in ("ge8", "nn", "strip", "hilbert", "morton"):
        if meth not in ("ge8", "strip", "hilbert", "morton"):
            continue
        og, mg, rg = [], [], []
        for m in vl:
            for r in m["realign"]:
                og.append(m["gap0"][meth])
                mg.append(r["mis"][meth])
                rg.append(r["snap"]["re"][meth] if r["snap"] else r["det"]["nndir"]["re"][meth])
        if og:
            labels.append(PRETTY[meth].split(" (")[0])
            o.append(mean(og))
            mi.append(mean(mg))
            re.append(mean(rg))
    x = np.arange(len(labels))
    ax.bar(x - 0.27, o, 0.26, label="aligned (original)", color="#4c9be8")
    ax.bar(x, mi, 0.26, label="misaligned", color="#d62728")
    ax.bar(x + 0.27, re, 0.26, label="re-aligned (lattice snap)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean gap to BKS (%)")
    ax.set_title("Misalignment moves frame-dependent costs, leaves GE; snap restores exactly")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "F5_realignment.png"), dpi=170)
    plt.close(fig)

    # F6: bantlama
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for cls, c in (("vlsi", "#1f77b4"), ("tsplib", "#ff7f0e")):
        ms = [m for m in M if m["cls"] == cls]
        bs = sorted({b["b"] for m in ms for b in m["bands"]})
        ys, xs_ = [], []
        for b in bs:
            v = [x["gap"] for m in ms for x in m["bands"] if x["b"] == b]
            if len(v) >= 3:
                xs_.append(b)
                ys.append(mean(v))
        ax.plot(xs_, ys, "-o", color=c, label=cls.upper())
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of equal-width bands b (theta = 0)")
    ax.set_ylabel("mean gap to BKS (%)")
    ax.set_title("Serpentine banding of GE: cost grows with b")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "F6_banding.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

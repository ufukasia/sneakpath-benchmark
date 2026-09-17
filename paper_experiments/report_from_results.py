# -*- coding: utf-8 -*-
"""MAKALE v3 KANITI -- panel sonuclarindan (results/*.json, yalniz Waterloo VLSI)
tablolar, figurler ve evidence.json.

    python paper_experiments/report_from_results.py [--out makale_v3] [--max-n 120000]

Yazilanlar:
    <out>/tables/T*.tex  (+ .md)     booktabs tablolar
    <out>/figures/F*.pdf (+ .png)    figurler
    <out>/evidence.json              her sayi bir id ile (hakem/atif ajanlari icin)
    <out>/evidence_summary.md        okunabilir ozet

Kaynak satirlar (row_id): greedy_edge, greedy_edge@knn8_greedy, nn, nn_exact,
farthest_insertion, strip, hilbert, morton, rotation_strip, rgge_*, rsge_*,
fs_*, fs_ge8_jitter/detied/exactknn, fs_band_ge8, realign_*(@manual:7.5/37.5),
pge_*_k*, pge_ksweep, pge_band_k*@zero/@rotation_strip/@manual:22.5,
repair_vnd(@strip/@rsge_fixed2/@realign_strip), ils(@...).
"""
from __future__ import annotations

import argparse
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
import vlsi_datasets as V                                                   # noqa: E402
from make_report import (sign_test, wilcoxon, spearman, ks_2samp, q, mean,  # noqa: E402
                         fmt, pfmt)

SWEEP = [("fs_ge8", "GE ($k{=}8$)", "inv"), ("fs_ge15", "GE ($k{=}15$)", "inv"),
         ("fs_nn", "NN (kesin)", "inv"), ("fs_fi", "Farthest Ins.", "inv"),
         ("fs_nn_grid", "NN (ızgara)", "approx"),
         ("fs_strip", "Strip", "dep"), ("fs_hilbert", "Hilbert", "dep"),
         ("fs_morton", "Morton", "dep"), ("fs_band2ge", "2-bant GE", "dep")]
REALIGN_BASE = {"ge8": "greedy_edge@knn8_greedy", "strip": "strip", "hilbert": "hilbert", "morton": "morton"}
REALIGN_PRETTY = {"ge8": "Greedy-Edge (k=8)", "strip": "Strip", "hilbert": "Hilbert", "morton": "Morton"}
PHIS = [("", 22.5), ("@manual:7.5", 7.5), ("@manual:37.5", 37.5)]
PGE_S = [("band", "bant (1-B, dengeli)"), ("kd", "k-d (2-B, dengeli)"),
         ("kmeans", "k-means (2-B, dengesiz)"), ("split", "global tur kesme")]


class Ev:
    """evidence.json: id -> {value, unit, desc}."""

    def __init__(self):
        self.d = {}

    def put(self, key, value, desc="", unit=""):
        if isinstance(value, float):
            value = round(value, 6)
        self.d[key] = dict(value=value, unit=unit, desc=desc)
        return value


def tex_table(path, caption, label, header, rows, note="", col=None, small=True):
    cols = col or ("l" + "r" * (len(header) - 1))
    L = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        L.append("\\footnotesize")
    if len(header) >= 6:
        L[-1] = "\\scriptsize"           # sn-jnl tabular'i sarmalamaya izin vermez: kucult
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
    """p degeri BAGINTIYLA (hakem C1: p hicbir zaman 0.000 yazilmaz)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 1e-4:
        return "<10^{-4}"
    if p < 10 ** (-nd) / 2:
        return f"<10^{{-{nd}}}"
    return f"={p:.{nd}f}"


def ptex(p, nd=3):
    """Tablo hucresi: 0.001 ya da $<10^{-4}$."""
    r = prel(p, nd)
    if r == "--":
        return r
    return f"${r}$" if r.startswith("<") else r[1:]


def load(max_n):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "makale_v3"))
    ap.add_argument("--max-n", type=int, default=120000)
    args = ap.parse_args()
    T = os.path.join(args.out, "tables")
    F = os.path.join(args.out, "figures")
    os.makedirs(T, exist_ok=True)
    os.makedirs(F, exist_ok=True)
    D = load(args.max_n)
    ev = Ev()
    S = []
    N = len(D)
    ev.put("n_instances", N, "VLSI ornek sayisi")
    ev.put("n_min", min(d["n"] for d in D), "en kucuk n")
    ev.put("n_max", max(d["n"] for d in D), "en buyuk n")
    S.append(f"# Makale v3 kaniti -- {N} Waterloo VLSI ornegi (n = {ev.d['n_min']['value']} .. {ev.d['n_max']['value']})\n")

    def g(d, c):
        return 100.0 * (c / d["bks"] - 1.0)

    def gap_of(d, rid):
        m = d["rows"].get(rid)
        return g(d, m["cost"]) if m else None

    # ================= T1: kurucular theta=0 (taban) =========================
    base_rows = [("greedy_edge", "Greedy-Edge, $k{=}15$"), ("greedy_edge@knn8_greedy", "Greedy-Edge, $k{=}8$"),
                 ("nn_exact", "NN (kesin)"), ("nn", "NN (ızgara)"),
                 ("farthest_insertion", "Farthest Insertion"), ("strip", "Strip"),
                 ("hilbert", "Hilbert"), ("morton", "Morton"), ("rotation_strip", "Strip, en iyi açı"),
                 ("lkh3", "LKH-3 (referans)")]
    rows = []
    for rid, nm in base_rows:
        gs = [gap_of(d, rid) for d in D if rid in d["rows"]]
        ts = [d["rows"][rid]["time"] for d in D if rid in d["rows"]]
        if not gs:
            continue
        ev.put(f"gap0.{rid}.mean", mean(gs), f"{nm} ortalama gap (%)")
        ev.put(f"gap0.{rid}.median", st.median(gs))
        ev.put(f"gap0.{rid}.n", len(gs))
        rows.append([nm, len(gs), pct(mean(gs)), pct(st.median(gs)), pct(min(gs)), pct(max(gs)), fmt(st.median(ts), 3)])
    fi_ns = [d["n"] for d in D if "farthest_insertion" in d["rows"]]
    ev.put("gap0.farthest_insertion.max_n", max(fi_ns) if fi_ns else None, "FI'nin kosuldugu en buyuk n")
    tex_table(os.path.join(T, "T1_kurucular.tex"), "Taban kurucular, $\\theta=0$: BKS'ye göre gap (\\%) ve süre.",
              "tab:kurucular", ["Kurucu", "$N$", "ort.", "medyan", "min", "maks", "süre (s)"], rows,
              f"Greedy-Edge $k{{=}}15$: Johnson--McGeoch aday listesi. Farthest Insertion saf $O(n^2)$: inşa bütçesi içinde bitirdiği {len(fi_ns)} örnek ($n\\le{max(fi_ns) if fi_ns else 0}$); açı taramasında (Tablo~\\ref{{tab:aci}}) $n\\le3000$. Strip, en iyi açı: maliyet taramasıyla (\\texttt{{rotation\\_strip}}). Süre saf Python, tek iş parçacığı, medyan. LKH-3 süre sınırlı referans çizgisidir, rakip değil ($N=71$; $n>13\\,000$ olan 28 büyük örnekte 300 s süre sınırı nedeniyle atlanmıştır).")

    # ================= T2: aci duyarliligi ===================================
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
        # 2026-09-09 (hakem A3/E9): beraberliksiz kontrol her kuramsal-degismez kurucuda
        dk = fk + "_detied"
        det_r = [d["rows"][dk]["range_pct"] for d in D if dk in d["rows"] and d["rows"][dk].get("range_pct") is not None]
        if det_r:
            sens[fk]["det_n"] = len(det_r); sens[fk]["det_med"] = st.median(det_r); sens[fk]["det_max"] = max(det_r)
            sens[fk]["det_zero_frac"] = mean([x < 0.05 for x in det_r])
        for k_, v_ in sens[fk].items():
            ev.put(f"sens.{fk}.{k_}", v_, f"{nm} aci taramasi {k_}")
        cls_lab = {"inv": "değişmez", "approx": "değ.\\ $+$ yakl.", "dep": "bağımlı"}[cls]
        det_cell = (fmt(st.median(det_r), 3) + f" ({len(det_r)})") if det_r else "--"
        rows.append([nm, cls_lab, len(ms), fmt(st.median(rng), 1) + " / " + fmt(mean(rng), 1), fmt(q(rng, .9), 1),
                     fmt(mean(worst)) + " / " + fmt(mean(best)), det_cell])
    tex_table(os.path.join(T, "T2_aci_duyarlilik.tex"),
              "Açı duyarlılığı: kurucu $[-90^\\circ,90^\\circ)$ taramasında tur maliyetinin aralığı ($100\\,(\\max-\\min)/\\min$; medyan / ortalama, p90) ve $\\theta=0$'a göre en kötü açı kaybı / en iyi açı kazancı (puan).",
              "tab:aci", ["Kurucu", "sınıf", "$N$", "aralık med./ort.", "p90", "kötü/iyi açı", "ber.siz ($N$)"], rows,
              "Aralık, p90 ve ber.siz yüzde; kötü/iyi açı puan. Adım $5^\\circ$ ($n\\le 3000$), $10^\\circ$ ($\\le 15\\,000$), $15^\\circ$ ($\\le 50\\,000$), $30^\\circ$ (üstü). Maliyet daima orijinal koordinatlarda. "
              "Sınıf (kuramsal): kurucunun yalnız Öklid mesafelerine (değişmez) ya da eksen koordinatlarına (bağımlı) dayanması; NN (ızgara) mesafe tabanlıdır ama aday listesini eksen-hizalı ızgara hücrelerinden alır (yaklaşıklık). "
              "Ber.siz: koordinatlara $10^{-4}\\times$ medyan komşu mesafesi gürültü eklenip beraberlikler kırıldıktan sonra aynı açı taramasının aralık medyanı; kuramsal-değişmez kurucuda $0$ beklenir, ızgara yaklaşıklığında kalıntı kalır. "
              "NN (kesin) detied sütununda bir örnekte (dan59296) tarama kaydı boş olduğundan $N=98$'dir. GE detied sütunlarında medyan 0.000 olmakla birlikte ızgara k-NN kaynaklı kuyruk kalıntısı iki uç örnekte GE $k{=}15$'te \\%5.040 ve GE $k{=}8$'de \\%7.043'tür (kesin $k$-NN altında tam sıfıra iner, bkz. Tablo~\\ref{tab:gekontrol}).")

    # ================= T3: GE kontrolleri ====================================
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
    for lab, sel in (("Tümü", C), ("$n<500$", [c for c in C if c["n"] < 500]), ("$500\\le n<2000$", [c for c in C if 500 <= c["n"] < 2000]),
                     ("$2000\\le n<10^4$", [c for c in C if 2000 <= c["n"] < 10000]), ("$n\\ge 10^4$", [c for c in C if c["n"] >= 10000])):
        if not sel:
            continue
        exs = [c["rng_ex"] for c in sel if c["rng_ex"] is not None]
        rows.append([lab, len(sel), fmt(st.median([c["rng_rot"] for c in sel]), 2) + " / " + fmt(st.median([c["rng_jit"] for c in sel]), 2),
                     fmt(st.median([c["rng_det"] for c in sel]), 3),
                     fmt(100 * mean([c["ks_p"] > 0.05 for c in sel]), 0),
                     fmt(mean([c["best_rot"] for c in sel])) + " / " + fmt(mean([c["best_jit"] for c in sel]))])
    w, l, p = sign_test([c["best_rot"] - c["best_jit"] for c in C])
    rho_rj = spearman([c["rng_rot"] for c in C], [c["rng_jit"] for c in C])[0]
    rho_n = spearman([c["n"] for c in C], [c["rng_rot"] for c in C])[0]
    rho_rtm = spearman([c["gap0"] for c in C], [c["best_rot"] for c in C])
    ev.put("ge.n", len(C)); ev.put("ge.ks_ok_frac", mean([c["ks_p"] > 0.05 for c in C]), "KS rot~jitter reddedilmeyen oran")
    ev.put("ge.spearman_rot_jit", rho_rj); ev.put("ge.spearman_n_range", rho_n)
    ev.put("ge.spearman_gap0_bestgain", rho_rtm[0]); ev.put("ge.spearman_gap0_bestgain_p", rho_rtm[1])
    ev.put("ge.oracle_sign", f"{w}/{l}", "en iyi aci vs en iyi jitter: aci daha iyi/kotu"); ev.put("ge.oracle_p", p)
    ev.put("ge.best_rot_mean", mean([c["best_rot"] for c in C])); ev.put("ge.best_jit_mean", mean([c["best_jit"] for c in C]))
    ev.put("ge.rng_rot_median", st.median([c["rng_rot"] for c in C])); ev.put("ge.rng_jit_median", st.median([c["rng_jit"] for c in C]))
    ev.put("ge.rng_det_median", st.median([c["rng_det"] for c in C]))
    ev.put("ge.det_zero_frac", mean([c["rng_det"] < 0.05 for c in C]), "de-tied aralik <0.05% oran")
    exs = [c for c in C if c["rng_ex"] is not None]
    if exs:
        ev.put("ge.exact_n", len(exs)); ev.put("ge.exact_max_range", max(c["rng_ex"] for c in exs), "kesin kNN ile en buyuk aralik (%)")
        ev.put("ge.exact_zero_frac", mean([c["rng_ex"] < 1e-9 for c in exs]))
    ev.put("ge.mean_shift", mean([c["mean_shift"] for c in C]), "aci ortalamasi - theta0 (puan)")
    tex_table(os.path.join(T, "T3_ge_kontrol.tex"),
              "Greedy-Edge ($k{=}8$): açı taraması ile beraberlik-kırma kontrolleri (aralıklar medyan, eşit örneklem büyüklüğü). KS: rotasyon ve jitter dağılımlarının Kolmogorov--Smirnov testiyle ayırt edilemediği örnek oranı.",
              "tab:gekontrol", ["Küme", "$N$", "aralık rot.\\ / jit.\\ (\\%)", "ber.siz (\\%)", "KS (\\%)", "kazanç açı / jit.\\ (puan)"], rows,
              f"Beraberliksiz koordinatlarda medyan aralık \\%{st.median([c['rng_det'] for c in C]):.3f} (iki uç örnekte ızgara k-NN kaynaklı maks. \\%{max(c['rng_det'] for c in C):.3f}); kesin $k$-NN altında tüm örneklerde tam sıfırdır (\\%{max(c['rng_ex'] for c in exs):.3f}, $N={len(exs)}$, $n\\le 6000$). "
              f"Spearman(rotasyon aralığı, jitter aralığı) $= {rho_rj:.2f}$ ($p<10^{{-4}}$); Spearman($n$, rotasyon aralığı) $= {rho_n:.2f}$ ($p<10^{{-4}}$); "
              f"eşit bütçeli oracle işaret testi (açı daha iyi / daha kötü) $= {w}/{l}$, $p={p:.2f}$. Jitter: $\\theta=0$'da yalnız tam-eşit kenar uzunluklarının sırasını değiştiren tohumlar; beraberliksiz: koordinatlara $10^{{-4}}\\times$ medyan komşu mesafesi gürültü.")

    # ================= T4: strip'in en iyi acisi ==============================
    sb = []
    for d in D:
        r = d["rows"]
        if "fs_strip" in r and "rotation_strip" in r:
            sw = r["fs_strip"]["sweep"]
            ba = float(min(sw, key=sw.get))
            sb.append(dict(g0=g(d, r["fs_strip"]["cost0"]), gbest=g(d, min(sw.values())), ba=ba,
                           on_axis=min(abs(ba), abs(abs(ba) - 90)) <= 5.0, grot=g(d, r["rotation_strip"]["cost"])))
    ev.put("strip.n", len(sb)); ev.put("strip.gap0_mean", mean([x["g0"] for x in sb])); ev.put("strip.gapbest_mean", mean([x["gbest"] for x in sb]))
    ev.put("strip.on_axis_frac", mean([x["on_axis"] for x in sb]), "en iyi aci eksene 5 derece icinde oran")
    ev.put("strip.rotation_strip_gap_mean", mean([x["grot"] for x in sb]))

    # ================= T5/T6: yeniden hizalama ================================
    det_err = defaultdict(list)
    snap_ok, snap_tot, ident = 0, 0, 0
    eff = {m: dict(orig=[], mis=[], re=[]) for m in REALIGN_BASE}
    for d in D:
        r = d["rows"]
        for meth, base in REALIGN_BASE.items():
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
    for dk, nm in (("comb", "Tarak izdüşümü (grid\\_theta)"), ("nndir", "NN-kenar yönü"), ("pca", "PCA ana ekseni"), ("snap", "Snap (kaba + kafes inceltme)")):
        e = det_err.get(dk, [])
        if not e:
            continue
        ev.put(f"det.{dk}.n", len(e)); ev.put(f"det.{dk}.median", st.median(e)); ev.put(f"det.{dk}.p90", q(e, .9))
        ev.put(f"det.{dk}.lt1", mean([x < 1 for x in e])); ev.put(f"det.{dk}.lt01", mean([x < 0.1 for x in e]))
        rows.append([nm, len(e), fmt(st.median(e), 4), fmt(q(e, .9), 2), fmt(100 * mean([x < 1 for x in e]), 0) + "\\%", fmt(100 * mean([x < 0.1 for x in e]), 0) + "\\%"])
    ev.put("snap.trials", snap_tot); ev.put("snap.exact_frac", snap_ok / snap_tot if snap_tot else None); ev.put("snap.ge_identical_frac", ident / snap_tot if snap_tot else None)
    tex_table(os.path.join(T, "T5_dedektor.tex"),
              f"Hizasızlaştırılmış kartta ($\\varphi\\in\\{{7.5^\\circ,22.5^\\circ,37.5^\\circ\\}}$) açı tahmin hatası $|\\hat\\theta-\\varphi|$ (derece, mod 90) ve kafes geri kazanımı.",
              "tab:dedektor", ["Dedektör", "deneme", "medyan", "p90", "$<1^\\circ$", "$<0.1^\\circ$"], rows,
              f"Snap: {snap_tot} denemenin {snap_ok}'inde kafes birebir geri geldi ({100 * snap_ok / max(snap_tot, 1):.0f}\\%), GE turu {ident} denemede bit-aynı.")
    rows = []
    for meth in ("ge8", "strip", "hilbert", "morton"):
        e = eff[meth]
        if not e["orig"]:
            continue
        dm = [a - b for a, b in zip(e["mis"], e["orig"])]
        dr = [a - b for a, b in zip(e["re"], e["orig"])]
        ev.put(f"realign.{meth}.n", len(dm)); ev.put(f"realign.{meth}.orig", mean(e["orig"])); ev.put(f"realign.{meth}.mis", mean(e["mis"]))
        ev.put(f"realign.{meth}.re", mean(e["re"])); ev.put(f"realign.{meth}.abs_mis", mean([abs(x) for x in dm])); ev.put(f"realign.{meth}.abs_re", mean([abs(x) for x in dr]))
        ev.put(f"realign.{meth}.exact_frac", mean([abs(x) < 1e-9 for x in dr])); ev.put(f"realign.{meth}.wil_mis", wilcoxon(dm))
        rows.append([REALIGN_PRETTY[meth], len(dm), pct(mean(e["orig"])), pct(mean(e["mis"])), fmt(mean(dm), 2), fmt(mean([abs(x) for x in dm]), 2), ptex(wilcoxon(dm)),
                     fmt(mean([abs(x) for x in dr]), 3), fmt(100 * mean([abs(x) < 1e-9 for x in dr]), 0) + "\\%"])
    tex_table(os.path.join(T, "T6_hizasizlik.tex"),
              "Hizasızlaştırma ve yeniden hizalamanın maliyete etkisi (gap \\%, ortalama; ornek $\\times$ 3 açı).",
              "tab:hizasizlik", ["Kurucu", "deneme", "orijinal", "hizasız", "fark", "$|$fark$|$", "$p$", "$|$geri$-$orij.$|$", "birebir"], rows,
              "Dönme-değişmez kurucuda hizasızlık ve geri hizalama maliyeti değiştirmez; çerçeveye bağımlı kurucuda hizasızlık maliyeti $\\pm$ oynatır, geri hizalama orijinali birebir geri getirir.")

    # ================= T7: dikisli bantlama + RSGE/RGGE ======================
    rows = []
    bset = sorted({b["b"] for d in D if "fs_band_ge8" in d["rows"] for b in d["rows"]["fs_band_ge8"]["bands"] if not b.get("tag")})
    for b in bset + ["ks"]:
        pr = []
        for d in D:
            m = d["rows"].get("fs_band_ge8")
            if not m:
                continue
            g1 = next((x for x in m["bands"] if x["b"] == 1 and not x.get("tag")), None)
            if b == "ks":
                gb = next((x for x in m["bands"] if x.get("tag") == "ks"), None)
            else:
                gb = next((x for x in m["bands"] if x["b"] == b and not x.get("tag")), None)
            if g1 and gb:
                pr.append((g(d, g1["cost"]), g(d, gb["cost"]), gap_of(d, "strip")))
        if len(pr) < 3:
            continue
        dg = [y - x for x, y, _ in pr]
        w, l, p = sign_test(dg)
        ev.put(f"band.b{b}.n", len(pr)); ev.put(f"band.b{b}.gap", mean([y for _, y, _ in pr])); ev.put(f"band.b{b}.dgap", mean(dg)); ev.put(f"band.b{b}.wins", w); ev.put(f"band.b{b}.p", p)
        lab = b
        if b == "ks":
            # 2026-09-09 (hakem B5/E6): strip'in kendi serit sayisiyla dogrudan kiyas
            ds = [y - z for _, y, z in pr if z is not None]
            wb_, ws_, ps = sign_test(ds)                              # sign_test: w = negatif sayisi -> bant daha iyi
            ev.put("band.ks.strip_gap", mean([z for _, _, z in pr if z is not None])); ev.put("band.ks.minus_strip", mean(ds))
            ev.put("band.ks.abs_minus_strip_med", st.median([abs(x) for x in ds])); ev.put("band.ks.band_better", wb_); ev.put("band.ks.strip_better", ws_); ev.put("band.ks.p_vs_strip", ps)
            ev.put("band.ks.n", len(pr)); ev.put("band.ks.gap", mean([y for _, y, _ in pr]))
            lab = "$k_s{=}\\mathrm{round}\\sqrt{n/2}$"
        rows.append([lab, len(pr), pct(mean([y for _, y, _ in pr])), fmt(mean(dg), 2), fmt(st.median(dg), 2), f"{w}/{len(pr)}", ptex(p)])
    rows.append("MIDRULE")
    for rid, nm in (("rsge_fixed2", "RSGE $b{=}2$, kafes açısı"), ("rsge_fixed2@rotation_strip", "RSGE $b{=}2$, strip açısı"),
                    ("rsge_corridor", "RSGE koridor"), ("rgge_x", "RGGE, strip açısı"), ("rgge_x@zero", "RGGE, $\\theta=0$ (çapa)")):
        pr = [(gap_of(d, "greedy_edge@knn8_greedy"), gap_of(d, rid), d["rows"][rid].get("rsge_b")) for d in D if rid in d["rows"]]
        if len(pr) < 3:
            continue
        dg = [y - x for x, y, _ in pr]
        w, l, p = sign_test(dg)
        bs = [b for _, _, b in pr if b is not None]
        extra = f" ($b{{=}}1$: {sum(1 for b in bs if b == 1)}/{len(bs)})" if bs and "corridor" in rid else ""
        ev.put(f"fam.{rid}.n", len(pr)); ev.put(f"fam.{rid}.dgap", mean(dg)); ev.put(f"fam.{rid}.wins", w); ev.put(f"fam.{rid}.p", p)
        rows.append([nm + extra, len(pr), pct(mean([y for _, y, _ in pr])), fmt(mean(dg), 2), fmt(st.median(dg), 2), f"{w}/{len(pr)}", ptex(p)])
    tex_table(os.path.join(T, "T7_bantlama.tex"),
              "Dikişli bantlama ($\\theta=0$, bant içi GE $k{=}8$, $b$ eşit-genişlik bant tek tura dikilir) ve RSGE/RGGE aile satırları: global GE ($k{=}8$) ile fark.",
              "tab:bant", ["$b$ / satır", "$N$", "gap", "fark (puan)", "medyan fark", "kazanan", "$p$ (işaret)"], rows,
              "Kazanan: satırın GE'den daha iyi olduğu örnek sayısı. $k_s$: strip kurucusunun kendi şerit sayısı (\\texttt{snake\\_order}); bu satır bantlı GE'yi strip ile aynı şerit sayısında sınar. $b=32$ satırında $N=97$'dir ($n<250$ olan iki küçük örnekte $n/b<8$ olduğundan atlanmıştır). RSGE kaynak/bütçe kuralları $n<20\\,000$'de $b=1$ verir ve GE ile özdeştir (tabloda yok).")

    # ================= T8: paralel GE ========================================
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
            ev.put(f"pge.{s}.k{k}.n", len(v)); ev.put(f"pge.{s}.k{k}.mk", mean([x["mk"] for x in v])); ev.put(f"pge.{s}.k{k}.mk_med", st.median([x["mk"] for x in v]))
            ev.put(f"pge.{s}.k{k}.tot", mean([x["tot"] for x in v])); ev.put(f"pge.{s}.k{k}.imb", mean([x["imb"] for x in v]))
            if sp:
                ev.put(f"pge.{s}.k{k}.speed_med", st.median(sp))
            rows.append([nm if k == ks[0] else "", k, len(v), fmt(mean([x["mk"] for x in v]), 3) + " / " + fmt(st.median([x["mk"] for x in v]), 3),
                         fmt(mean([x["tot"] for x in v]), 3), fmt(mean([x["imb"] for x in v]), 2), fmt(st.median(sp), 1) if sp else "--"])
        rows.append("MIDRULE")
    if rows and rows[-1] == "MIDRULE":
        rows.pop()
    tex_table(os.path.join(T, "T8_paralel.tex"),
              "Paralel GE: bölümleme stratejisi $\\times$ $k$ araç. Makespan oranı $=\\max_i L_i/(L_{GE}/k)$ (1.0 ideal), toplam oranı $=\\sum_i L_i/L_{GE}$, denge $=\\max_i L_i/\\overline{L}$, hız $=t_{GE}/t_{par}$.",
              "tab:paralel", ["Bölümleme", "$k$", "$N$", "makespan ort.\\ / med.", "toplam", "denge", "hız"], rows,
              "Bant açısı kafes dedektöründen (grid\\_theta). Hız: tek iş parçacıklı saf Python'da $t_{GE}/t_{par}$, $t_{par}$ = bölümleme $+$ en uzun parça.")
    # T8b: k'ya gore kazanan (makespan)
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
        for s, _ in PGE_S:
            ev.put(f"pgewin.k{k}.{s}", wins.get(s, 0))
        ev.put(f"pgewin.k{k}.kmeans_minus_band", mean(dkb)); ev.put(f"pgewin.k{k}.p", p)
        rows.append([k, len(dkb)] + [wins.get(s, 0) for s, _ in PGE_S] + [fmt(mean(dkb), 3) + f" ($p{prel(p)}$)"])
    tex_table(os.path.join(T, "T8b_paralel_kazanan.tex"), "Her $k$ için en düşük makespan'ı veren bölümleme (örnek sayısı) ve k-means $-$ bant farkı (işaret testi $p$).",
              "tab:paralelkazanan", ["$k$", "$N$"] + [nm.split(" (")[0] for _, nm in PGE_S] + ["k-means $-$ bant"], rows)
    # T8c: bant cerceve varyantlari
    rows = []
    for k in (2, 4, 8):
        base_k = f"pge_band_k{k}"
        for suf, nm in (("@zero", "$\\theta=0$"), ("@rotation_strip", "strip açısı"), ("@manual:22.5", "$22.5^\\circ$ (hizasız)")):
            pr = [(d["rows"][base_k]["makespan_ratio"], d["rows"][base_k + suf]["makespan_ratio"]) for d in D
                  if base_k in d["rows"] and base_k + suf in d["rows"] and d["rows"][base_k].get("makespan_ratio") is not None]
            if len(pr) < 3:
                continue
            dg = [y - x for x, y in pr]
            w, l, p = sign_test([-x for x in dg])
            ev.put(f"pgeframe.k{k}.{suf}.n", len(pr)); ev.put(f"pgeframe.k{k}.{suf}.diff", mean(dg)); ev.put(f"pgeframe.k{k}.{suf}.worse", w); ev.put(f"pgeframe.k{k}.{suf}.p", p)
            rows.append([k, nm, len(pr), fmt(mean(dg), 3), f"{w}/{len(pr)}", ptex(p)])
    tex_table(os.path.join(T, "T8c_bant_cerceve.tex"), "Bant bölümlemesinin çerçeve bağımlılığı: kafes açısına (grid\\_theta) göre makespan oranı farkı.",
              "tab:bantcerceve", ["$k$", "çerçeve", "$N$", "fark", "daha kötü", "$p$"], rows,
              "Pozitif fark = varyant çerçevede daha uzun makespan.")

    # ================= T8d: k-means++ tohum duyarliligi (hakem E10) ============
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
        ev.put(f"kmseed.k{k}.n", len(per)); ev.put(f"kmseed.k{k}.mean", mean([x["mean"] for x in per])); ev.put(f"kmseed.k{k}.range", mean([x["rng"] for x in per]))
        ev.put(f"kmseed.k{k}.sd", mean([x["sd"] for x in per]))
        wb = [x for x in per if x["band"] is not None]
        wk = [x for x in per if x["kd"] is not None]
        if wb:
            ev.put(f"kmseed.k{k}.worst_beats_band", mean([x["worst"] < x["band"] for x in wb])); ev.put(f"kmseed.k{k}.best_beats_band", mean([x["best"] < x["band"] for x in wb]))
        if wk:
            ev.put(f"kmseed.k{k}.best_beats_kd", mean([x["best"] < x["kd"] for x in wk]))
        rows.append([k, len(per), fmt(mean([x["mean"] for x in per]), 3), fmt(mean([x["sd"] for x in per]), 3), fmt(mean([x["rng"] for x in per]), 3),
                     (fmt(100 * mean([x["best"] < x["band"] for x in wb]), 0) + "\\% / " + fmt(100 * mean([x["worst"] < x["band"] for x in wb]), 0) + "\\%") if wb else "--",
                     fmt(100 * mean([x["best"] < x["kd"] for x in wk]), 0) + "\\%" if wk else "--"])
    if rows:
        tex_table(os.path.join(T, "T8d_kmeans_tohum.tex"), "k-means$++$ başlatma tohumu duyarlılığı: 5 tohumda makespan oranı (örnek başına ortalama, tohumlar arası s.s.\\ ve aralık) ve en iyi / en kötü tohumun bantı, en iyi tohumun k-d'yi geçtiği örnek oranı.",
                  "tab:kmeanstohum", ["$k$", "$N$", "ort.", "s.s.", "aralık", "bantı geçer (en iyi / en kötü)", "k-d'yi geçer (en iyi)"], rows,
                  "Bant ve k-d tek tohumlu (deterministik) satırlardır; k-means'in tohum seçimiyle bile onları geçemediği yerde fark tohuma bağlı değildir.")

    # ================= T9: onarim / ILS =====================================
    rows = []
    for rid, nm in (("repair_vnd", "VND $\\leftarrow$ Greedy-Edge"), ("repair_vnd@strip", "VND $\\leftarrow$ Strip"),
                    ("repair_vnd@rsge_fixed2", "VND $\\leftarrow$ RSGE $b{=}2$"), ("repair_vnd@realign_strip", "VND $\\leftarrow$ geri hizalanmış strip"),
                    ("ils@repair_vnd", "ILS $\\leftarrow$ VND(GE)"), ("ils", "ILS $\\leftarrow$ Greedy-Edge"), ("ils@rsge_fixed2", "ILS $\\leftarrow$ RSGE $b{=}2$")):
        pr = [(gap_of(d, rid), g(d, d["rows"][rid]["initial_cost"]) if d["rows"][rid].get("initial_cost") else None, d["rows"][rid]["time"]) for d in D if rid in d["rows"]]
        if len(pr) < 3:
            continue
        fin = [x for x, _, _ in pr]; ini = [y for _, y, _ in pr if y is not None]
        ev.put(f"rep.{rid}.n", len(pr)); ev.put(f"rep.{rid}.final", mean(fin)); ev.put(f"rep.{rid}.initial", mean(ini) if ini else None)
        rows.append([nm, len(pr), pct(mean(ini)) if ini else "--", pct(mean(fin)), pct(st.median(fin)), fmt(st.median([t for _, _, t in pr]), 1)])
    if rows:
        tex_table(os.path.join(T, "T9_onarim.tex"), "İnşa farkı derin düzenlemeden sonra kalıyor mu? Aynı onarım (VND) ve ILS farklı tohumlarla.",
                  "tab:onarim", ["Satır", "$N$", "başlangıç gap", "son gap ort.", "medyan", "süre (s)"], rows,
                  "VND $\\leftarrow$ Greedy-Edge satırının tohumu panel varsayılanı $k{=}15$ GE'dir (Tablo~\\ref{tab:kurucular}, \\%17.2); diğer bölümlerdeki $k{=}8$ değil. ILS yalnız $n\\le 1000$ örneklerde koşuldu (süre bütçesi); pertürbasyon uyarlamalı (durgunluğa göre double-bridge ya da segment ters çevirme).")

    # ================= T10: CERCEVE YARISI -- tekli tur ========================
    # Hizali (kafes, theta=0) strip <-> seyrek tarama (rotation_strip) stripi:
    # insa -> 2-opt -> VND -> ILS basamaklarinda kim onde?
    race_rows = []
    race = {}
    # Kullanici sozlesmesi: tekil komsuluk yok; birlesik onarim (VND) ve onun USTUNE ILS.
    # 2026-09-09 (hakem A2/E1): ILS satiri stokastik -- ana olcu TOHUM ORTALAMASI
    # (n<=2500: 10 tohum, 2500<n<=3000: 5 tohum); en iyi tohum ayri satirda.
    def gap_seedmean(d, rid):
        m = d["rows"].get(rid)
        if not m:
            return None
        stc = m.get("stochastic") or {}
        return g(d, stc["mean"]) if stc.get("mean") is not None else g(d, m["cost"])
    stages = [("insa", "strip", "rotation_strip", "İnşa (strip)", gap_of),
              ("vnd", "repair_vnd@strip", "repair_vnd@rotation_strip", "$+$ birleşik onarım (VND)", gap_of),
              ("ils", "ils@repair_vnd@strip", "ils@repair_vnd@rotation_strip", "$+$ VND $+$ ILS (tohum ort.)", gap_seedmean),
              ("ilsbest", "ils@repair_vnd@strip", "ils@repair_vnd@rotation_strip", "$+$ VND $+$ ILS (en iyi tohum)", gap_of)]
    ils_seeds = sorted({(d["rows"][r_].get("stochastic") or {}).get("n_seeds") for d in D for r_ in ("ils@repair_vnd@strip",) if r_ in d["rows"]} - {None})
    ev.put("race.ils_seed_counts", ",".join(str(x) for x in ils_seeds), "ILS yaris satirlarinda tohum sayilari")
    for key, ra, rs, nm, gf in stages:
        pr = [(gf(d, ra), gf(d, rs)) for d in D if ra in d["rows"] and rs in d["rows"]]
        if len(pr) < 3:
            continue
        dg = [a - s_ for a, s_ in pr]                       # pozitif = hizali daha kotu
        w, l, p = sign_test(dg)                             # w = hizali daha iyi
        race[key] = dict(n=len(pr), aligned=mean([a for a, _ in pr]), sparse=mean([s_ for _, s_ in pr]),
                         diff=mean(dg), med=st.median(dg), aligned_wins=w, sparse_wins=l, p=p, wil=wilcoxon(dg))
        for k_, v_ in race[key].items():
            ev.put(f"race.{key}.{k_}", v_, f"cerceve yarisi {key} {k_}")
        race_rows.append([nm, len(pr), pct(race[key]["aligned"]), pct(race[key]["sparse"]), fmt(mean(dg), 2), fmt(st.median(dg), 2),
                          f"{w} / {l}", ptex(p)])
    # ILS asamasi, buyuk dilim (n >= 2000): fark n ile buyuyor mu?
    ra, rs = "ils@repair_vnd@strip", "ils@repair_vnd@rotation_strip"
    big = [(gap_seedmean(d, ra), gap_seedmean(d, rs)) for d in D if ra in d["rows"] and rs in d["rows"] and d["n"] >= 2000]
    if len(big) >= 3:
        dg = [a - s_ for a, s_ in big]
        w, l, p = sign_test(dg)
        for k_, v_ in dict(n=len(big), diff=mean(dg), aligned_wins=w, sparse_wins=l, p=p, wil=wilcoxon(dg),
                           aligned=mean([a for a, _ in big]), sparse=mean([s_ for _, s_ in big])).items():
            ev.put(f"race.ilsbig.{k_}", v_, f"cerceve yarisi ILS n>=2000 {k_} (tohum ort.)")
    if race_rows:
        tex_table(os.path.join(T, "T10_yaris_tekli.tex"),
                  "Çerçeve yarışı, tek tur: hizalı (kafes açısı, $\\theta=0$) strip ile seyrek-tarama (\\texttt{rotation\\_strip}) stripi, iyileştirme derinliğine göre (gap \\%).",
                  "tab:yaristekli", ["Aşama", "$N$", "hizalı", "seyrek", "fark", "medyan", "h / s kazanır", "$p$"], race_rows,
                  "Her iki çerçeve aynı birleşik onarımdan (VND: 2-opt $+$ Or-opt $+$ relocate) ve onun üstüne aynı ILS'den geçer; ILS yalnız $n\\le3000$, $n\\le2500$'de 10, üstünde 5 bağımsız tohum; tohum ort.\\ = tohumlar üzerinden ortalama tur maliyetinin gap'i (ana ölçü), en iyi tohum = tohumların en iyisi. Negatif fark = hizalı önde; $p$ işaret testi.")

    # ================= T11: CERCEVE YARISI -- paralel (bant basina onarim) =========
    prows = []
    prace = {}
    for k in (2, 4, 8):
        base_k = f"pgr_band_k{k}"
        for suf, nm in (("@rotation_strip", "seyrek"), ("@zero", "$\\theta=0$"), ("@manual:22.5", "$22.5^\\circ$")):
            pr = []
            for d in D:
                a = d["rows"].get(base_k)
                v = d["rows"].get(base_k + suf)
                if a and v and a.get("makespan_ratio_rep") is not None and v.get("makespan_ratio_rep") is not None:
                    pr.append((a["makespan_ratio"], v["makespan_ratio"], a["makespan_ratio_rep"], v["makespan_ratio_rep"]))
            if len(pr) < 3:
                continue
            d_before = [va - aa for aa, va, _, _ in pr]      # pozitif = varyant daha kotu (hizali onde)
            d_after = [vr - ar for _, _, ar, vr in pr]
            wb, lb, pb = sign_test([-x for x in d_before])   # wb = varyant daha kotu (hizali kazanir)
            wa, la, pa = sign_test([-x for x in d_after])
            prace[(k, suf)] = dict(n=len(pr), aligned_before=mean([a for a, _, _, _ in pr]), var_before=mean([v for _, v, _, _ in pr]),
                                   aligned_after=mean([a for _, _, a, _ in pr]), var_after=mean([v for _, _, _, v in pr]),
                                   d_before=mean(d_before), d_after=mean(d_after), aligned_wins_before=wb, aligned_wins_after=wa,
                                   p_before=pb, p_after=pa)
            for k_, v_ in prace[(k, suf)].items():
                ev.put(f"prace.k{k}.{suf}.{k_}", v_)
            e = prace[(k, suf)]
            prows.append([k, nm, len(pr), fmt(e["aligned_before"], 3) + " $\\to$ " + fmt(e["aligned_after"], 3),
                          fmt(e["var_before"], 3) + " $\\to$ " + fmt(e["var_after"], 3),
                          f"{wb} $\\to$ {wa}", ptex(pa)])
    if prows:
        tex_table(os.path.join(T, "T11_yaris_paralel.tex"),
                  "Çerçeve yarışı, paralel: bant bölümlemesi + GE + bant-başına VND. Makespan oranı, kafes açısı (hizalı) ile varyant çerçeve; onarım öncesi $\\to$ sonrası.",
                  "tab:yarisparalel", ["$k$", "varyant", "$N$", "hizalı", "varyant", "hizalı kazanır ($N$ içinde)", "$p$ (sonra)"], prows,
                  "Kazanma: makespan oranı düşük olan; sayılar onarım öncesi $\\to$ sonrası. Onarım her parçada aynı VND ve bütçe.")
    # k-means kontrol: onarim sonrasi bant (kafes) vs kmeans
    kk_rows = []
    kd_note = []
    for k in (2, 4, 8):
        pr = [(d["rows"][f"pgr_band_k{k}"]["makespan_ratio_rep"], d["rows"][f"pgr_kmeans_k{k}"]["makespan_ratio_rep"],
               (d["rows"].get(f"pgr_kd_k{k}") or {}).get("makespan_ratio_rep")) for d in D
              if f"pgr_band_k{k}" in d["rows"] and f"pgr_kmeans_k{k}" in d["rows"]
              and d["rows"][f"pgr_band_k{k}"].get("makespan_ratio_rep") is not None and d["rows"][f"pgr_kmeans_k{k}"].get("makespan_ratio_rep") is not None]
        if len(pr) < 3:
            continue
        dg = [b - m for b, m, _ in pr]
        w, l, p = sign_test(dg)
        ev.put(f"prace_km.k{k}.band", mean([b for b, _, _ in pr])); ev.put(f"prace_km.k{k}.kmeans", mean([m for _, m, _ in pr]))
        ev.put(f"prace_km.k{k}.band_wins", w); ev.put(f"prace_km.k{k}.kmeans_wins", l); ev.put(f"prace_km.k{k}.n", len(pr)); ev.put(f"prace_km.k{k}.p", p)
        # 2026-09-09 (hakem B4/E2): onarim sonrasi k-d
        kd = [(b, m, x) for b, m, x in pr if x is not None]
        kd_cell, kd_win = "--", "--"
        if len(kd) >= 3:
            dk_ = [b - x for b, _, x in kd]
            wk, lk, pk = sign_test(dk_)
            dkm = [m - x for _, m, x in kd]
            wkm, lkm, pkm = sign_test(dkm)                    # wkm = k-means k-d'den iyi
            ev.put(f"prace_kd.k{k}.n", len(kd)); ev.put(f"prace_kd.k{k}.kd", mean([x for _, _, x in kd])); ev.put(f"prace_kd.k{k}.band", mean([b for b, _, _ in kd]))
            ev.put(f"prace_kd.k{k}.band_wins", wk); ev.put(f"prace_kd.k{k}.kd_wins", lk); ev.put(f"prace_kd.k{k}.p", pk)
            ev.put(f"prace_kd.k{k}.kmeans_wins_vs_kd", wkm); ev.put(f"prace_kd.k{k}.kd_wins_vs_kmeans", lkm); ev.put(f"prace_kd.k{k}.p_kmeans", pkm)
            best = defaultdict(int)
            for b, m, x in kd:
                best[min((("bant", b), ("k-d", x), ("k-means", m)), key=lambda t: t[1])[0]] += 1
            for nm_ in ("bant", "k-d", "k-means"):
                ev.put(f"prace_kd.k{k}.best_{nm_}", best.get(nm_, 0))
            kd_cell = fmt(mean([x for _, _, x in kd]), 3) + ("$^{*}$" if len(kd) != len(pr) else "")
            if len(kd) != len(pr):
                kd_note.append(f"$^{{*}}$ $k={k}$: k-d $N={len(kd)}$ (bir örnekte onarım sonrası makespan kaydı yok).")
            kd_win = f"{wk} / {lk} ({ptex(pk)})"
        kk_rows.append([k, len(pr), fmt(mean([b for b, _, _ in pr]), 3), kd_cell, fmt(mean([m for _, m, _ in pr]), 3),
                        kd_win, f"{w} / {l} ({ptex(p)})"])
    if kk_rows:
        tex_table(os.path.join(T, "T11b_yaris_kmeans.tex"), "Onarım sonrası bölümleme karşılaştırması: hizalı bant, k-d ve k-means (makespan oranı, bant-başına VND sonrası).",
                  "tab:yariskmeans", ["$k$", "$N$", "bant (hizalı)", "k-d", "k-means", "bant/k-d kazanır ($p$)", "bant/k-means kazanır ($p$)"], kk_rows,
                  "Her parçada aynı VND ve bütçe; $p$ işaret testi. " + " ".join(kd_note))

    # ================= T5b (hakem E8): gurultu altinda snap ======================
    nz = [d["rows"][k_] for d in D for k_ in d["rows"] if k_.startswith("realign_strip_noise")]
    nzd = [(d, d["rows"][k_]) for d in D for k_ in d["rows"] if k_.startswith("realign_strip_noise")]
    if len(nz) >= 3:
        e0 = [m["err_deg"] for m in nz]
        ev.put("noise.n", len(nz)); ev.put("noise.err_med", st.median(e0)); ev.put("noise.err_p90", q(e0, .9)); ev.put("noise.lt01", mean([x < 0.1 for x in e0])); ev.put("noise.lt1", mean([x < 1 for x in e0]))
        ev.put("noise.sigma_rel", nz[0].get("noise_rel"))
        rows = [["$\\sigma=0.1\\,d_{NN}$, silme yok", len(e0), fmt(st.median(e0), 4), fmt(q(e0, .9), 3), fmt(100 * mean([x < 0.1 for x in e0]), 0) + "\\%", fmt(100 * mean([x < 1 for x in e0]), 0) + "\\%"]]
        for fr in ("0.01", "0.05"):
            e = [m["drop"][fr]["err_deg"] for m in nz if fr in (m.get("drop") or {})]
            if e:
                ev.put(f"noise.drop{fr}.err_med", st.median(e)); ev.put(f"noise.drop{fr}.lt01", mean([x < 0.1 for x in e])); ev.put(f"noise.drop{fr}.lt1", mean([x < 1 for x in e]))
                rows.append([f"$+$ \\%{float(fr) * 100:g} nokta silme", len(e), fmt(st.median(e), 4), fmt(q(e, .9), 3), fmt(100 * mean([x < 0.1 for x in e]), 0) + "\\%", fmt(100 * mean([x < 1 for x in e]), 0) + "\\%"])
        dre = [g(d, m["cost"]) - g(d, m["oracle_cost"]) for d, m in nzd]
        dmis = [g(d, m["mis_cost"]) - g(d, m["oracle_cost"]) for d, m in nzd]
        ev.put("noise.re_minus_oracle", mean(dre)); ev.put("noise.re_minus_oracle_absmed", st.median([abs(x) for x in dre])); ev.put("noise.mis_minus_oracle", mean(dmis))
        ev.put("noise.re_within_1pt", mean([abs(x) <= 1.0 for x in dre]))
        tex_table(os.path.join(T, "T5b_gurultu.tex"),
                  "Saha gürültüsü altında yeniden hizalama: rastgele $\\varphi\\sim U(-45^\\circ,45^\\circ)$, koordinat gürültüsü $\\sigma=0.1\\times$ medyan komşu mesafesi ve nokta silme; snap açı hatası $|\\hat\\theta-\\varphi|$ (derece).",
                  "tab:gurultu", ["Koşul", "$N$", "medyan", "p90", "$<0.1^\\circ$", "$<1^\\circ$"], rows,
                  f"Tur gürültülü koordinatta kurulur, maliyet gerçek koordinatta ölçülür. Geri hizalanmış strip ile gerçek açıyla hizalanmış (oracle) strip arasındaki gap farkı ortalama {mean(dre):.2f} puan (medyan $|$fark$|$ {st.median([abs(x) for x in dre]):.3f}); hizasız bırakılan strip ile oracle arasındaki fark {mean(dmis):.1f} puan (negatif = hizasız strip daha kısa; inşa paradoksu, Adım~4).")

    # ================= figurler ==============================================
    try:
        make_figures(D, F, ev, g)
    except Exception:
        import traceback
        ev.put("figure_error", traceback.format_exc())

    json.dump(ev.d, open(os.path.join(args.out, "evidence.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_macros(ev, os.path.join(args.out, "sayilar.tex"))
    S.append("## Anahtar sayilar")
    for k in sorted(ev.d):
        if any(k.startswith(p) for p in ("ge.", "strip.", "snap.", "n_")):
            S.append(f"- {k} = {ev.d[k]['value']}")
    open(os.path.join(args.out, "evidence_summary.md"), "w", encoding="utf-8").write("\n".join(S) + "\n")
    print("\n".join(S[:40]))
    print("tablolar:", sorted(f for f in os.listdir(T) if f.endswith(".tex")))
    print("figurler:", sorted(os.listdir(F)))


def write_macros(ev, path):
    """Metinde gecen sayilar icin LaTeX makrolari (evidence.json'dan; tablolar
    yenilendiginde metin de yenilenir). Makro adi = harf, sayi biciminde."""
    E = ev.d

    def v(k, nd=None, default="--"):
        x = E.get(k, {}).get("value")
        if x is None:
            return default
        if isinstance(x, str):
            return x
        if nd is None:
            return str(x)
        return f"{x:.{nd}f}"

    def pctv(k, nd=0):
        x = E.get(k, {}).get("value")
        return "--" if x is None else f"{100 * x:.{nd}f}"

    def pv(k, nd=3):
        """p makrosu BAGINTIYLA: metinde $p\\evX$ yazilir -> p=0.001 / p<10^{-4}."""
        x = E.get(k, {}).get("value")
        return "--" if x is None else prel(float(x), nd)
    M = {
        "evN": v("n_instances"), "evNmin": v("n_min"), "evNmax": v("n_max"),
        "evGeN": v("ge.n"), "evGeRotMed": v("ge.rng_rot_median", 2), "evGeJitMed": v("ge.rng_jit_median", 2),
        "evGeDetMed": v("ge.rng_det_median", 3), "evGeDetZero": pctv("ge.det_zero_frac"), "evGeKsOk": pctv("ge.ks_ok_frac"),
        "evGeSpRotJit": v("ge.spearman_rot_jit", 2), "evGeSpN": v("ge.spearman_n_range", 2),
        "evGeSpRtm": v("ge.spearman_gap0_bestgain", 2), "evGeOracle": v("ge.oracle_sign"), "evGeOracleP": pv("ge.oracle_p", 2),
        "evGeBestRot": v("ge.best_rot_mean", 2), "evGeBestJit": v("ge.best_jit_mean", 2), "evGeMeanShift": v("ge.mean_shift", 2),
        "evGeExactN": v("ge.exact_n"), "evGeExactMax": v("ge.exact_max_range", 3),
        "evStripGapZero": v("strip.gap0_mean", 1), "evStripGapBest": v("strip.gapbest_mean", 1), "evStripOnAxis": pctv("strip.on_axis_frac"),
        "evStripRotGap": v("strip.rotation_strip_gap_mean", 1),
        "evSnapTrials": v("snap.trials"), "evSnapExact": pctv("snap.exact_frac"), "evSnapIdent": pctv("snap.ge_identical_frac"),
        "evCombMed": v("det.comb.median", 3), "evCombPninety": v("det.comb.p90", 1), "evCombLtOne": pctv("det.comb.lt1"),
        "evNndirMed": v("det.nndir.median", 2), "evNndirLtOne": pctv("det.nndir.lt1"), "evPcaMed": v("det.pca.median", 1),
        "evSnapMed": v("det.snap.median", 4), "evSnapLtPointOne": pctv("det.snap.lt01"),
        "evReGeAbsMis": v("realign.ge8.abs_mis", 2), "evReGeAbsRe": v("realign.ge8.abs_re", 3), "evReGeExact": pctv("realign.ge8.exact_frac"),
        "evReStripAbsMis": v("realign.strip.abs_mis", 1), "evReStripMis": v("realign.strip.mis", 1), "evReStripOrig": v("realign.strip.orig", 1),
        "evReStripExact": pctv("realign.strip.exact_frac"), "evReStripP": pv("realign.strip.wil_mis"),
        "evBandTwo": v("band.b2.dgap", 1), "evBandFour": v("band.b4.dgap", 1), "evBandEight": v("band.b8.dgap", 1), "evBandSixteen": v("band.b16.dgap", 1),
        "evBandTwoWins": v("band.b2.wins"), "evBandTwoN": v("band.b2.n"),
        "evRsgeFixedD": v("fam.rsge_fixed2.dgap", 2), "evRsgeFixedStripD": v("fam.rsge_fixed2@rotation_strip.dgap", 2),
        "evRggeD": v("fam.rgge_x.dgap", 2), "evRggeWins": v("fam.rgge_x.wins"), "evRggeN": v("fam.rgge_x.n"),
        "evGeGapEight": v("gap0.greedy_edge@knn8_greedy.mean", 2), "evGeGapFifteen": v("gap0.greedy_edge.mean", 2),
        "evNnGap": v("gap0.nn_exact.mean", 1), "evFiGap": v("gap0.farthest_insertion.mean", 1), "evFiN": v("gap0.farthest_insertion.n"),
        "evHilbertGap": v("gap0.hilbert.mean", 1), "evMortonGap": v("gap0.morton.mean", 1),
    }
    for s in ("band", "kd", "kmeans", "split"):
        for k in (2, 4, 8, 16):
            M[f"evPge{s.capitalize()}K{ {2: 'Two', 4: 'Four', 8: 'Eight', 16: 'Sixteen'}[k]}"] = v(f"pge.{s}.k{k}.mk", 2)
            if s in ("band", "kd", "kmeans"):
                M[f"evPge{s.capitalize()}Speed{ {2: 'Two', 4: 'Four', 8: 'Eight', 16: 'Sixteen'}[k]}"] = v(f"pge.{s}.k{k}.speed_med", 1)
    for k in (4, 8, 16):
        kk = {4: "Four", 8: "Eight", 16: "Sixteen"}[k]
        M[f"evPgeWinKmeansK{kk}"] = v(f"pgewin.k{k}.kmeans"); M[f"evPgeWinBandK{kk}"] = v(f"pgewin.k{k}.band")
        M[f"evPgeWinKdK{kk}"] = v(f"pgewin.k{k}.kd"); M[f"evPgeWinNK{kk}"] = v(f"pgewin.k{k}.n") if f"pgewin.k{k}.n" in E else v("n_instances")
        M[f"evPgeKmBandK{kk}"] = v(f"pgewin.k{k}.kmeans_minus_band", 3); M[f"evPgeKmBandPK{kk}"] = pv(f"pgewin.k{k}.p")
    for k in (4, 8):
        kk = {4: "Four", 8: "Eight"}[k]
        M[f"evFrameMisK{kk}"] = v(f"pgeframe.k{k}.@manual:22.5.diff", 3); M[f"evFrameMisWorseK{kk}"] = v(f"pgeframe.k{k}.@manual:22.5.worse")
        M[f"evFrameMisNK{kk}"] = v(f"pgeframe.k{k}.@manual:22.5.n"); M[f"evFrameStripK{kk}"] = v(f"pgeframe.k{k}.@rotation_strip.diff", 3); M[f"evFrameMisPK{kk}"] = pv(f"pgeframe.k{k}.@manual:22.5.p"); M[f"evFrameStripPK{kk}"] = pv(f"pgeframe.k{k}.@rotation_strip.p")
    M["evRepGeFinal"] = v("rep.repair_vnd.final", 2); M["evRepStripFinal"] = v("rep.repair_vnd@strip.final", 2)
    M["evRepStripInit"] = v("rep.repair_vnd@strip.initial", 1); M["evRepGeInit"] = v("rep.repair_vnd.initial", 1)
    M["evRepRsgeFinal"] = v("rep.repair_vnd@rsge_fixed2.final", 2); M["evRepN"] = v("rep.repair_vnd.n")
    M["evIlsGeFinal"] = v("rep.ils.final", 2); M["evIlsVndFinal"] = v("rep.ils@repair_vnd.final", 2); M["evIlsRsgeFinal"] = v("rep.ils@rsge_fixed2.final", 2); M["evIlsN"] = v("rep.ils.n")
    for st_ in ("insa", "vnd", "ils", "ilsbest"):
        S_ = {"insa": "Insa", "vnd": "Vnd", "ils": "Ils", "ilsbest": "IlsBest"}[st_]
        M[f"evRace{S_}N"] = v(f"race.{st_}.n"); M[f"evRace{S_}Aligned"] = v(f"race.{st_}.aligned", 1); M[f"evRace{S_}Sparse"] = v(f"race.{st_}.sparse", 1)
        M[f"evRace{S_}Diff"] = v(f"race.{st_}.diff", 2); M[f"evRace{S_}AW"] = v(f"race.{st_}.aligned_wins"); M[f"evRace{S_}SW"] = v(f"race.{st_}.sparse_wins"); M[f"evRace{S_}P"] = pv(f"race.{st_}.p"); M[f"evRace{S_}Wil"] = pv(f"race.{st_}.wil")
    M["evRaceIlsBigN"] = v("race.ilsbig.n"); M["evRaceIlsBigDiff"] = v("race.ilsbig.diff", 2); M["evRaceIlsBigAW"] = v("race.ilsbig.aligned_wins"); M["evRaceIlsBigSW"] = v("race.ilsbig.sparse_wins"); M["evRaceIlsBigP"] = pv("race.ilsbig.p"); M["evRaceIlsBigWil"] = pv("race.ilsbig.wil"); M["evRaceIlsSeeds"] = v("race.ils_seed_counts")
    M["evRaceIlsBigAligned"] = v("race.ilsbig.aligned", 1); M["evRaceIlsBigSparse"] = v("race.ilsbig.sparse", 1)
    for k in (2, 4, 8):
        kk = {2: "Two", 4: "Four", 8: "Eight"}[k]
        M[f"evPRaceK{kk}AlignedBefore"] = v(f"prace.k{k}.@rotation_strip.aligned_before", 3); M[f"evPRaceK{kk}SparseBefore"] = v(f"prace.k{k}.@rotation_strip.var_before", 3)
        M[f"evPRaceK{kk}AlignedAfter"] = v(f"prace.k{k}.@rotation_strip.aligned_after", 3); M[f"evPRaceK{kk}SparseAfter"] = v(f"prace.k{k}.@rotation_strip.var_after", 3)
        M[f"evPRaceK{kk}WinsBefore"] = v(f"prace.k{k}.@rotation_strip.aligned_wins_before"); M[f"evPRaceK{kk}WinsAfter"] = v(f"prace.k{k}.@rotation_strip.aligned_wins_after")
        M[f"evPRaceK{kk}N"] = v(f"prace.k{k}.@rotation_strip.n"); M[f"evPRaceK{kk}PAfter"] = pv(f"prace.k{k}.@rotation_strip.p_after")
        M[f"evPRaceKmK{kk}Band"] = v(f"prace_km.k{k}.band", 3); M[f"evPRaceKmK{kk}Kmeans"] = v(f"prace_km.k{k}.kmeans", 3); M[f"evPRaceKmK{kk}BandWins"] = v(f"prace_km.k{k}.band_wins"); M[f"evPRaceKmK{kk}N"] = v(f"prace_km.k{k}.n"); M[f"evPRaceKmK{kk}P"] = pv(f"prace_km.k{k}.p"); M[f"evPRaceKmK{kk}KmeansWins"] = v(f"prace_km.k{k}.kmeans_wins")
        M[f"evPRaceKdK{kk}Kd"] = v(f"prace_kd.k{k}.kd", 3); M[f"evPRaceKdK{kk}BandWins"] = v(f"prace_kd.k{k}.band_wins"); M[f"evPRaceKdK{kk}KdWins"] = v(f"prace_kd.k{k}.kd_wins"); M[f"evPRaceKdK{kk}P"] = pv(f"prace_kd.k{k}.p"); M[f"evPRaceKdK{kk}N"] = v(f"prace_kd.k{k}.n")
        M[f"evPRaceKdK{kk}KmWinsVsKd"] = v(f"prace_kd.k{k}.kmeans_wins_vs_kd"); M[f"evPRaceKdK{kk}KdWinsVsKm"] = v(f"prace_kd.k{k}.kd_wins_vs_kmeans"); M[f"evPRaceKdK{kk}PKm"] = pv(f"prace_kd.k{k}.p_kmeans")
        M[f"evPRaceKdK{kk}BestBand"] = v(f"prace_kd.k{k}.best_bant"); M[f"evPRaceKdK{kk}BestKd"] = v(f"prace_kd.k{k}.best_k-d"); M[f"evPRaceKdK{kk}BestKm"] = v(f"prace_kd.k{k}.best_k-means")
        M[f"evKmSeedK{kk}Range"] = v(f"kmseed.k{k}.range", 3); M[f"evKmSeedK{kk}Sd"] = v(f"kmseed.k{k}.sd", 3); M[f"evKmSeedK{kk}BestBeatsBand"] = pctv(f"kmseed.k{k}.best_beats_band"); M[f"evKmSeedK{kk}WorstBeatsBand"] = pctv(f"kmseed.k{k}.worst_beats_band"); M[f"evKmSeedK{kk}BestBeatsKd"] = pctv(f"kmseed.k{k}.best_beats_kd"); M[f"evKmSeedK{kk}N"] = v(f"kmseed.k{k}.n")
    # 2026-09-09 (hakem): k-d kazanma sayilari her k'da, gurultu, band k_s, FI, beraberliksiz kontroller
    for k in (2, 3, 4, 6, 8, 12, 16):
        kk = {2: "Two", 3: "Three", 4: "Four", 6: "Six", 8: "Eight", 12: "Twelve", 16: "Sixteen"}[k]
        for s_ in ("band", "kd", "kmeans", "split"):
            M[f"evPgeWin{s_.capitalize()}K{kk}"] = v(f"pgewin.k{k}.{s_}")
            M[f"evPge{s_.capitalize()}K{kk}"] = v(f"pge.{s_}.k{k}.mk", 3)
            M[f"evPge{s_.capitalize()}SpeedK{kk}"] = v(f"pge.{s_}.k{k}.speed_med", 1)
            M[f"evPge{s_.capitalize()}ImbK{kk}"] = v(f"pge.{s_}.k{k}.imb", 2)
        M[f"evPgeWinNK{kk}"] = v(f"pgewin.k{k}.n") if f"pgewin.k{k}.n" in E else v("n_instances")
    M["evFiMaxN"] = v("gap0.farthest_insertion.max_n")
    M["evNoiseN"] = v("noise.n"); M["evNoiseErrMed"] = v("noise.err_med", 3); M["evNoiseLtPointOne"] = pctv("noise.lt01"); M["evNoiseLtOne"] = pctv("noise.lt1")
    M["evNoiseDropOneMed"] = v("noise.drop0.01.err_med", 3); M["evNoiseDropFiveMed"] = v("noise.drop0.05.err_med", 3); M["evNoiseDropFiveLtOne"] = pctv("noise.drop0.05.lt1"); M["evNoiseDropFiveLtPointOne"] = pctv("noise.drop0.05.lt01")
    M["evNoiseReOracle"] = v("noise.re_minus_oracle", 2); M["evNoiseReOracleAbsMed"] = v("noise.re_minus_oracle_absmed", 3); M["evNoiseMisOracle"] = v("noise.mis_minus_oracle", 1); M["evNoiseWithinOne"] = pctv("noise.re_within_1pt")
    M["evBandKsGap"] = v("band.ks.gap", 1); M["evBandKsN"] = v("band.ks.n"); M["evBandKsMinusStrip"] = v("band.ks.minus_strip", 2); M["evBandKsAbsMed"] = v("band.ks.abs_minus_strip_med", 2)
    M["evBandKsBetter"] = v("band.ks.band_better"); M["evBandKsStripBetter"] = v("band.ks.strip_better"); M["evBandKsP"] = pv("band.ks.p_vs_strip");    M["evBandThirtyTwo"] = v("band.b32.gap", 1)
    _bwords = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen"}
    for b_ in (1, 2, 4, 8, 16):
        M[f"evBandGapB{_bwords[b_]}"] = v(f"band.b{b_}.gap", 1)
    for fk, nm_ in (("fs_ge15", "GeFifteen"), ("fs_nn", "Nn"), ("fs_nn_grid", "NnGrid"), ("fs_fi", "Fi"), ("fs_ge8", "GeEight")):
        M[f"evDet{nm_}Med"] = v(f"sens.{fk}.det_med", 3); M[f"evDet{nm_}Max"] = v(f"sens.{fk}.det_max", 3); M[f"evDet{nm_}N"] = v(f"sens.{fk}.det_n"); M[f"evDet{nm_}Zero"] = pctv(f"sens.{fk}.det_zero_frac")
        M[f"evSens{nm_}Med"] = v(f"sens.{fk}.med", 1); M[f"evSens{nm_}Mean"] = v(f"sens.{fk}.mean", 1); M[f"evSens{nm_}Best"] = v(f"sens.{fk}.best", 2)
    # metinde gecen strip aci-duyarliligi degerleri (main.tex Adim 1) makrolastirildi
    M["evSensStripMed"] = v("sens.fs_strip.med", 1)
    M["evSensStripPninety"] = v("sens.fs_strip.p90", 1)
    M["evSensStripWorst"] = v("sens.fs_strip.worst", 1)
    M["evSensStripBest"] = v("sens.fs_strip.best", 2)
    _sw = E.get("sens.fs_strip.worst", {}).get("value")
    _sb = E.get("sens.fs_strip.best", {}).get("value")
    M["evSensStripSpan"] = "--" if (_sw is None or _sb is None) else f"{_sw + _sb:.0f}"
    L = ["% OTOMATIK: report_from_results.py -> evidence.json. Elle duzenlemeyin."]
    for k, val in M.items():
        L.append(f"\\newcommand{{\\{k}}}{{{val}}}")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def make_figures(D, F, ev, g):
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

    # F1: aci egrileri (kucuk + buyuk VLSI)
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
                        label={"fs_ge8": "Greedy-Edge", "fs_nn": "NN (kesin)", "fs_strip": "Strip", "fs_hilbert": "Hilbert", "fs_morton": "Morton", "fs_band2ge": "2-bant GE"}[fk])
            ax.set_title(f"{d['name']} ($n$={d['n']})")
            ax.set_xlabel("döndürme açısı $\\theta$ (°)")
            ax.set_ylabel("gap (%)")
        axes[0].legend(fontsize=6, ncol=2)
        save(fig, "F1_aci_egrileri")
    # F2: kutu grafigi
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
    ax.set_xticklabels([dict((a, b_) for a, b_, _ in SWEEP)[fk].replace(" (", "\n(") for fk in order], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 85)
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_ylabel("açı aralığı (%)", fontsize=8.5)
    n_inv = sum(1 for fk in order if dict((a, c) for a, _, c in SWEEP)[fk] == "inv")
    ax.axvline(n_inv + 0.5, color="k", ls="--", lw=0.8)
    save(fig, "F2_duyarlilik_kutu")
    # F3: rot vs jitter
    C = [(d["rows"]["fs_ge8"]["range_pct"], d["rows"]["fs_ge8_jitter"]["range_pct"], d["rows"]["fs_ge8_detied"]["range_pct"], d["rows"]["fs_ge8_detied"].get("tie_frac_before", 0), d["n"])
         for d in D if all(k in d["rows"] for k in ("fs_ge8", "fs_ge8_jitter", "fs_ge8_detied"))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    ax.scatter([c[1] for c in C], [c[0] for c in C], s=14, alpha=0.75, color="#1f4e79", edgecolors="none")
    lim = [0.35, 20]
    ax.plot(lim, lim, "k--", lw=0.9, label="$y=x$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.35, 20); ax.set_ylim(0.35, 20)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xlabel("jitter aralığı, $\\theta=0$ (%)", fontsize=8.5); ax.set_ylabel("rotasyon taraması aralığı (%)", fontsize=8.5)
    ax.set_title("GE: rotasyon = beraberlik kırma", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1]
    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.scatter([c[4] for c in C], [c[0] for c in C], s=14, label="rotasyon", color="#1f4e79", alpha=0.75)
    ax.scatter([c[4] for c in C], [c[2] for c in C], s=18, marker="x", lw=1.2, label="beraberliksiz", color="#c62828", alpha=0.85)
    ax.set_xscale("log"); ax.set_xlim(80, 150000); ax.set_ylim(-0.6, 13.5)
    ax.set_xlabel("$n$ (örnek boyutu)", fontsize=8.5); ax.set_ylabel("GE açı aralığı (%)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Aralık $n$ ile küçülür; beraberliksizde $\\approx 0$", fontsize=9)
    save(fig, "F3_ge_beraberlik")
    # F4: yeniden hizalama CDF
    errs = defaultdict(list)
    for d in D:
        for suf, _ in PHIS:
            m = d["rows"].get("realign_ge8" + suf)
            if m:
                for dk, x in (m.get("det") or {}).items():
                    errs[dk].append(max(x["err"], 1e-3))
                errs["snap"].append(max(m["err_deg"], 1e-3))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for dk, lab, c in (("comb", "tarak (grid_theta)", "#1f4e79"), ("nndir", "NN-kenar yönü", "#2e7d32"), ("pca", "PCA", "#7f7f7f"), ("snap", "snap", "#c62828")):
        e = sorted(errs.get(dk, []))
        if e:
            ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", label=lab, color=c)
    ax.set_xscale("log"); ax.set_xlim(8e-4, 50); ax.set_xlabel("$|\\hat\\theta-\\varphi|$ (°, 0.001 tabanlı)"); ax.set_ylabel("deneme oranı (CDF)")
    ax.legend(fontsize=7)
    save(fig, "F4_hizalama_cdf")
    # F5: bantlama + paralel
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    bset = sorted({b["b"] for d in D if "fs_band_ge8" in d["rows"] for b in d["rows"]["fs_band_ge8"]["bands"] if not b.get("tag")})
    ys = [mean([g(d, x["cost"]) for d in D if "fs_band_ge8" in d["rows"] for x in d["rows"]["fs_band_ge8"]["bands"] if x["b"] == b and not x.get("tag")]) for b in bset]
    ax.plot(bset, ys, "-o", color="#795548"); ax.set_xscale("log", base=2); ax.set_xlabel("bant sayısı $b$ (tek tura dikişli)"); ax.set_ylabel("ort. gap (%)"); ax.set_title("Dikişli bantlama")
    ax = axes[1]
    COLP = {"band": "#b45309", "kd": "#7c2d12", "kmeans": "#0f766e", "split": "#475569"}
    for s, nm in PGE_S:
        pts = [(k, ev.d[f"pge.{s}.k{k}.mk"]["value"]) for k in (2, 3, 4, 6, 8, 12, 16) if f"pge.{s}.k{k}.mk" in ev.d]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3, color=COLP[s], label=nm)
    ax.axhline(1.0, color="k", ls="--", lw=.8); ax.set_xscale("log", base=2); ax.set_xlabel("$k$ araç"); ax.set_ylabel("makespan / ($L_{GE}/k$)"); ax.set_title("Paralel GE"); ax.legend(fontsize=6.5)
    save(fig, "F5_bant_paralel")


if __name__ == "__main__":
    main()

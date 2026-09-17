# -*- coding: utf-8 -*-
"""CERCEVE (ACI) DENEYI -- panelden kosulabilen yontemler.

Makalenin uc iddiasini (bkz. paper_experiments/README.md) admin panelinden
tek tek yeniden uretebilmek icin runner.py'nin cagirdigi ince katman.
Kurucu / dedektor / kafes-inceltme ILKELLERI TEK KAYNAKTAN gelir:
`paper_experiments/frame_experiment.py` (ayni kod makale kosumunu da yapti;
burada yalniz panel satirina cevrilir). Iki yerde iki kopya olsaydi panel ile
makale sessizce ayrisirdi.

Satir aileleri (runner.METHOD_ORDER'daki anahtarlar):

  nn_exact              KESIN en-yakin-komsu (O(n^2) numpy). `nn` izgara-
                        hizlandirmali ve yaklasiktir; degismezlik iddiasi
                        kesin surumde okunmali.

  fs_<kurucu>           ACI TARAMASI (TANI): kurucu [-90, 90) araligindaki
                        her acida dondurulmus koordinatlarla kurulur, maliyet
                        ORIJINAL koordinatlarda. Satirin turu/maliyeti theta=0
                        turudur (kanonik); `savings` = en kotu - en iyi aci
                        maliyeti (rotation_strip ile ayni anlam), `worst_cost`
                        / `best_cost` / `sweep` alanlari tam taramayi tasir.
                        Aci adimi panelin "rotation step" degeri (n buyudukce
                        kabalastirilir, bkz. `angles_for`).
                        Kurucular: ge8, ge15, nn, nn_grid, fi, strip, hilbert,
                        morton, band2ge (RSGE b=2).

  fs_ge8_jitter         KONTROL: theta=0, yalniz TAM beraberlikleri bozan
                        tohumlu jitter (tie_eps=1e-9), aci sayisi kadar tohum.
                        Rotasyon taramasiyla AYNI orneklem buyuklugu.
  fs_ge8_detied         KONTROL: koordinatlara 1e-4 x medyan-NN gurultu
                        (beraberlik kalmaz), sonra aci taramasi. Tez dogruysa
                        yayilim ~0.

  fs_band_ge8           BANTLAMA (TANI): theta=0'da b in {1,2,3,4,6,8,12,16}
                        esit-genislik bant, bant ici GE k=8, serpantin dikis.
                        Satir turu = en iyi b; `bands` alani hepsini tasir.

  realign_<kurucu>      HIZASIZ KART -> YENIDEN HIZALAMA: kart panelde secili
                        aciyla (varsayilan manual:22.5) DONDURULUR ("hizasiz
                        cip"), uc dedektor (comb / nndir / pca) aciyi tahmin
                        eder, snap hatti (en iyi kaba aci + hiyerarsik kafes
                        inceltme + tamsayiya yuvarlama; tamsayi olmayan
                        kumede nndir) karti geri hizalar, kurucu geri
                        hizalanmis kartta kurulur. Satir maliyeti = geri
                        hizalanmis tur; `initial_cost` = hizasiz kartta tur;
                        `savings` = hizasiz - geri hizalanmis. Alanlar:
                        phi, theta_hat, err_deg, det_err (uc dedektor),
                        exact_recovery, ge8_tour_identical (yalniz ge8).
"""
from __future__ import annotations

import math
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PE = os.path.join(_HERE, "paper_experiments")
if _PE not in sys.path:
    sys.path.insert(0, _PE)

import frame_experiment as FX  # noqa: E402  (paper_experiments/frame_experiment.py)
import snake_alt as SA         # noqa: E402
import parallel_ge as PG       # noqa: E402

# --- PARALEL GE (k drone): bolumleme stratejisi x k --------------------------
PGE_K_ALL = (2, 3, 4, 6, 8, 12, 16)     # 2026-09-09 (hakem E7): hiz her k icin olculur
PGE_KEYS = {f"pge_{s}_k{k}": (s, k) for s in PG.STRATEGIES for k in PGE_K_ALL}
PGE_KMSEED_KEY = "pge_kmseed"           # 2026-09-09 (hakem E10): k-means++ tohum duyarliligi
PGE_SWEEP_KEY = "pge_ksweep"
PGE_BAND_KEYS = tuple(k for k, (s, _) in PGE_KEYS.items() if s == "band")   # aci-secilebilir
_PGE_S = {"band": "bant (θ açı kutusundan, eşit-sayılı, 1-B)", "kd": "k-d ikiye bölme (dengeli 2-B)",
          "kmeans": "k-means kümeleri (2-B, dengesiz)", "split": "global GE turu kesilir"}
_PGE_PRETTY = {k: f"Paralel GE — {_PGE_S[s]}, k={kk} drone (satır=toplam yol; makespan/denge alanlarda)"
               for k, (s, kk) in PGE_KEYS.items()}
_PGE_PRETTY[PGE_SWEEP_KEY] = "⚠ TANI Paralel GE k-taraması — 4 strateji × k∈{2,3,4,6,8,12,16} (satır=global GE; en iyi makespan alanlarda)"
_PGE_PRETTY[PGE_KMSEED_KEY] = "⚠ KONTROL Paralel GE k-means — 5 k-means++ tohumu × k∈{2,4,8} (satır=global GE; tohum başına makespan alanlarda)"
_PGE_COMPLEXITY = {k: ("k × O((n/k) log(n/k)) GE, paralelde O((n/k) log(n/k))" if s != "split"
                       else "O(n log n) global GE + O(n) kesme (paralel kazanç yok)")
                   for k, (s, kk) in PGE_KEYS.items()}
_PGE_COMPLEXITY[PGE_SWEEP_KEY] = "28 hücre × yukarıdaki"
_PGE_COMPLEXITY[PGE_KMSEED_KEY] = "15 hücre × (k-means + k × GE)"
_PGE_ROLE = {k: (f"paralel GE — {s} bölümleme, k={kk} " + {"band": "(1-B, dengeli, çerçeveye bağımlı)", "kd": "(2-B, dengeli, eksen-hizalı kesim)",
                                                               "kmeans": "(2-B, dengesiz, dönme-değişmez)", "split": "(referans, dönme-değişmez)"}[s])
             for k, (s, kk) in PGE_KEYS.items()}
_PGE_ROLE[PGE_SWEEP_KEY] = "TANI — hangi bölümleme / hangi k en iyi makespan"
_PGE_ROLE[PGE_KMSEED_KEY] = "KONTROL — k-means makespan başlatma tohumuna ne kadar duyarlı"

# --- PARALEL GE + BANT BASINA ONARIM (cerceve yarisi, 2026-09-08) -----------
# Kullanici hipotezi: hizali (kafes) cerceve insada seyrek taramaya kaybetse de
# onarim/ILS bittiginde kazanir. Bu satirlar her parcada VND onarimi yapar;
# bant satirlari aci-secilebilir (kafes varsayilan; kopyalar: strip acisi, 0, 22.5).
# 2026-09-09 (hakem B4/E2): onarim oncesi sampiyon k-d de yarisa girer.
PGR_STRATEGIES = ("band", "kd", "kmeans")
PGR_KEYS = {f"pgr_{s}_k{k}": (s, k) for s in PGR_STRATEGIES for k in PG.K_CHOICES}
PGR_BAND_KEYS = tuple(k for k, (s, _) in PGR_KEYS.items() if s == "band")
_PGR_PRETTY = {k: f"Paralel GE + bant-başına VND onarımı — {_PGE_S[s]}, k={kk} drone (satır=onarım sonrası toplam; makespan öncesi/sonrası alanlarda)"
               for k, (s, kk) in PGR_KEYS.items()}
_PGR_COMPLEXITY = {k: "bölümleme + k × (GE + VND), paralelde tek parça" for k in PGR_KEYS}
_PGR_ROLE = {k: (f"çerçeve yarışı — {s} bölümleme + onarım, k={kk} " + {"band": "(açı kutusu: kafes / seyrek / 22.5)", "kd": "(kontrol, dengeli 2-B, eksen-hizalı kesim)", "kmeans": "(kontrol, dönme-değişmez)"}[s])
             for k, (s, kk) in PGR_KEYS.items()}

SWEEP_METHODS = ("ge8", "ge15", "nn", "nn_grid", "fi", "strip", "hilbert", "morton", "band2ge")
SWEEP_KEYS = {f"fs_{m}": m for m in SWEEP_METHODS}
REALIGN_METHODS = ("ge8", "strip", "hilbert", "morton")
REALIGN_KEYS = {f"realign_{m}": m for m in REALIGN_METHODS}
# 2026-09-09 (hakem A3/E9): beraberliksiz kontrol yalniz GE k=8'e degil,
# "degismez" etiketli her kurucuya uygulanir (fs_<m>_detied).
DETIED_METHODS = ("ge15", "nn", "nn_grid", "fi")
DETIED_KEYS = {f"fs_{m}_detied": m for m in DETIED_METHODS}
CONTROL_KEYS = ("fs_ge8_jitter", "fs_ge8_detied", "fs_ge8_exactknn") + tuple(DETIED_KEYS)
# 2026-09-09 (hakem E8): rastgele hizasizlik + konum gurultusu + nokta silme
REALIGN_NOISE_KEY = "realign_strip_noise"
NOISE_REL = 0.10          # konum gurultusu sigma = 0.10 x medyan komsu mesafesi
NOISE_DROPS = (0.01, 0.05)  # dedektor/snap icin nokta silme oranlari
EXACTKNN_MAX_N = 6000
BAND_KEY = "fs_band_ge8"
ALL_KEYS = ("nn_exact",) + tuple(SWEEP_KEYS) + CONTROL_KEYS + (BAND_KEY,) + tuple(REALIGN_KEYS) + (REALIGN_NOISE_KEY,)

FI_MAX_N = FX.FI_MAX_N
# 2026-09-09 (hakem B5/E6): b=32 ve strip'in kendi serit sayisi k_s=round(sqrt(n/2))
BAND_COUNTS = tuple(FX.BAND_COUNTS) + (32,)
BAND_KS = "ks"
DEFAULT_MISALIGN_DEG = 22.5

PRETTY = {
    "nn_exact": "Nearest Neighbor — KESİN (O(n²); ızgara-hızlandırmasız kontrol)",
    "fs_ge8": "⚠ TANI Açı taraması — Greedy-Edge k=8 (satır=θ=0; savings=en kötü−en iyi açı)",
    "fs_ge15": "⚠ TANI Açı taraması — Greedy-Edge k=15",
    "fs_nn": "⚠ TANI Açı taraması — Nearest Neighbor (kesin)",
    "fs_nn_grid": "⚠ TANI Açı taraması — Nearest Neighbor (ızgara-hızlandırmalı)",
    "fs_fi": "⚠ TANI Açı taraması — Farthest Insertion (n≤3000)",
    "fs_strip": "⚠ TANI Açı taraması — Strip (boustrophedon)",
    "fs_hilbert": "⚠ TANI Açı taraması — Hilbert eğrisi",
    "fs_morton": "⚠ TANI Açı taraması — Morton eğrisi",
    "fs_band2ge": "⚠ TANI Açı taraması — 2-bantlı serpantin GE (RSGE b=2)",
    "fs_ge8_jitter": "⚠ KONTROL Greedy-Edge k=8 — θ=0, beraberlik-jitter tohumları (açı sayısı kadar)",
    "fs_ge8_detied": "⚠ KONTROL Greedy-Edge k=8 — beraberliksiz koordinatta açı taraması",
    "fs_ge8_exactknn": "⚠ KONTROL Greedy-Edge k=8 — beraberliksiz koordinat + KESİN k-NN aday listesi, açı taraması (n≤6000; beklenen aralık tam 0)",
    "fs_band_ge8": "⚠ TANI Bant taraması — θ=0, b∈{1,2,3,4,6,8,12,16,32, k_s=round√(n/2)}, bant-içi GE k=8 (satır=en iyi b)",
    "fs_ge15_detied": "⚠ KONTROL Greedy-Edge k=15 — beraberliksiz koordinatta açı taraması",
    "fs_nn_detied": "⚠ KONTROL Nearest Neighbor (kesin) — beraberliksiz koordinatta açı taraması",
    "fs_nn_grid_detied": "⚠ KONTROL Nearest Neighbor (ızgara) — beraberliksiz koordinatta açı taraması (yaklaşıklık kalıntısı beklenir)",
    "fs_fi_detied": "⚠ KONTROL Farthest Insertion — beraberliksiz koordinatta açı taraması (n≤3000)",
    "realign_strip_noise": "Hizasız + gürültülü kart (φ rastgele, σ=0.1×komşu mesafesi) → snap → Strip; %1/%5 nokta silme dedektör tanısı alanlarda",
    "realign_ge8": "Hizasız kart → yeniden hizalama (snap) → Greedy-Edge k=8",
    "realign_strip": "Hizasız kart → yeniden hizalama (snap) → Strip",
    "realign_hilbert": "Hizasız kart → yeniden hizalama (snap) → Hilbert eğrisi",
    "realign_morton": "Hizasız kart → yeniden hizalama (snap) → Morton eğrisi",
    **_PGE_PRETTY,
    **_PGR_PRETTY,
}

COMPLEXITY = {
    "nn_exact": "O(n²) kesin NN (numpy)",
    "fs_ge8": "A × O(n log n) GE, A = açı sayısı", "fs_ge15": "A × O(n log n) GE",
    "fs_nn": "A × O(n²) kesin NN", "fs_nn_grid": "A × O(n) ızgara-NN",
    "fs_fi": "A × O(n²) FI (n≤3000)", "fs_strip": "A × O(n log n)",
    "fs_hilbert": "A × O(n log n)", "fs_morton": "A × O(n log n)",
    "fs_band2ge": "A × O(n log n) bantlı GE",
    "fs_ge8_jitter": "A × O(n log n) GE (tohumlu)", "fs_ge8_detied": "A × O(n log n) GE",
    "fs_ge8_exactknn": "A × (O(n²) kesin k-NN + GE), n≤6000",
    "fs_band_ge8": "10 × O(n log n) bantlı GE",
    "fs_ge15_detied": "A × O(n log n) GE", "fs_nn_detied": "A × O(n²) kesin NN",
    "fs_nn_grid_detied": "A × O(n) ızgara-NN", "fs_fi_detied": "A × O(n²) FI (n≤3000)",
    "realign_strip_noise": "dedektörler O(n) + kafes inceltme + strip, ×(1 + silme oranı sayısı)",
    "realign_ge8": "dedektörler O(n) + kafes inceltme ~150×O(n) + GE",
    "realign_strip": "dedektörler O(n) + kafes inceltme + strip",
    "realign_hilbert": "dedektörler O(n) + kafes inceltme + Hilbert",
    "realign_morton": "dedektörler O(n) + kafes inceltme + Morton",
    **_PGE_COMPLEXITY,
    **_PGR_COMPLEXITY,
}

ROLE = {
    "nn_exact": "kontrol (kesin NN; dönme-değişmez sınıf)",
    "fs_ge8": "TANI — açı duyarlılığı (dönme-değişmez)",
    "fs_ge15": "TANI — açı duyarlılığı (dönme-değişmez)",
    "fs_nn": "TANI — açı duyarlılığı (dönme-değişmez)",
    "fs_nn_grid": "TANI — açı duyarlılığı (ızgara yaklaşıklığı)",
    "fs_fi": "TANI — açı duyarlılığı (dönme-değişmez)",
    "fs_strip": "TANI — açı duyarlılığı (çerçeveye bağımlı)",
    "fs_hilbert": "TANI — açı duyarlılığı (çerçeveye bağımlı)",
    "fs_morton": "TANI — açı duyarlılığı (çerçeveye bağımlı)",
    "fs_band2ge": "TANI — açı duyarlılığı (çerçeveye bağımlı)",
    "fs_ge8_jitter": "KONTROL — beraberlik kırma (rotasyonla ayırt edilemez olmalı)",
    "fs_ge8_detied": "KONTROL — beraberliksiz (yayılım ~0 olmalı)",
    "fs_ge8_exactknn": "KONTROL — beraberliksiz + kesin k-NN (yayılım TAM 0: GE dönme-değişmez)",
    "fs_band_ge8": "TANI — bant sayısı maliyeti (b=k_s: strip ile doğrudan kıyas)",
    "fs_ge15_detied": "KONTROL — beraberliksiz (yayılım ~0 olmalı)",
    "fs_nn_detied": "KONTROL — beraberliksiz (yayılım ~0 olmalı)",
    "fs_nn_grid_detied": "KONTROL — beraberliksiz (ızgara yaklaşıklığı kalır: sınıf 'değişmez + yaklaşıklık')",
    "fs_fi_detied": "KONTROL — beraberliksiz (yayılım ~0 olmalı)",
    "realign_strip_noise": "deney — saha gürültüsü altında dedektör/snap sağlamlığı",
    "realign_ge8": "deney — hizasız kartı hizalama (GE: fark yok beklenir)",
    "realign_strip": "deney — hizasız kartı hizalama (strip: kurtarma beklenir)",
    "realign_hilbert": "deney — hizasız kartı hizalama",
    "realign_morton": "deney — hizasız kartı hizalama",
    **_PGE_ROLE,
    **_PGR_ROLE,
}


def pgr(strategy, k, xs, ys, ewt, theta=0.0, cfg=None, budget_s=None, global_tour=None, t_global=None):
    """Paralel GE + parca basina VND (bkz. parallel_ge.evaluate_repaired)."""
    return PG.evaluate_repaired(strategy, xs, ys, k, ewt, theta=theta, cfg=cfg, budget_s=budget_s,
                                global_tour=global_tour, t_global=t_global)


def pge(strategy, k, xs, ys, ewt, theta=0.0, global_tour=None, t_global=None):
    """Tek paralel-GE hucresi (bkz. parallel_ge.evaluate)."""
    return PG.evaluate(strategy, xs, ys, k, ewt, theta=theta,
                       global_tour=global_tour, t_global=t_global)


def pge_sweep(xs, ys, ewt, theta=0.0, global_tour=None, t_global=None, ks=PG.K_SWEEP):
    """3 strateji x k taramasi; her hucrenin metrikleri (turlar atilir)."""
    n = len(xs)
    rows = []
    for k in ks:
        if k > 1 and n // k < 8:
            continue
        for s in PG.STRATEGIES:
            r = PG.evaluate(s, xs, ys, k, ewt, theta=theta, global_tour=global_tour, t_global=t_global)
            r.pop("tours", None)
            rows.append(r)
    return rows


def angles_for(n, step):
    """Aci kumesi: panel adimi, n buyudukce kabalastirilir (sure)."""
    step = float(step) if step and step > 0 else 5.0
    if n > 50000:
        step = max(step, 30.0)
    elif n > 15000:
        step = max(step, 15.0)
    elif n > 3000:
        step = max(step, 10.0)
    k = max(1, int(round(180.0 / step)))
    step = 180.0 / k
    return [round(-90.0 + i * step, 6) for i in range(k)]


def sweep(method, xs, ys, cost_fn, angles):
    """Aci taramasi. Doner: dict(cost0, tour0, sweep{aci: maliyet}, best, worst, times)."""
    if method == "fi" and len(xs) > FI_MAX_N:
        return None
    out, times = {}, []
    tour0 = None
    for a in angles:
        rx, ry = FX.rot(xs, ys, a)
        t0 = time.perf_counter()
        t = FX.build(method, rx, ry)
        times.append(time.perf_counter() - t0)
        c = cost_fn(t)
        out[f"{a:g}"] = c
        if abs(a) < 1e-9:
            tour0 = t
    if tour0 is None:
        tour0 = FX.build(method, xs, ys)
        out["0"] = cost_fn(tour0)
    vals = list(out.values())
    best_a = min(out, key=out.get)
    worst_a = max(out, key=out.get)
    return dict(cost0=out["0"], tour0=tour0, sweep=out, best_cost=min(vals), worst_cost=max(vals),
                best_theta=float(best_a), worst_theta=float(worst_a),
                range_pct=100.0 * (max(vals) - min(vals)) / min(vals),
                build_time=sum(times), n_angles=len(out))


def jitter(xs, ys, cost_fn, count):
    """theta=0, tohumlu beraberlik-jitter (tie_eps=1e-9)."""
    out = {}
    t0 = time.perf_counter()
    tour0 = FX.build("ge8", xs, ys)
    c0 = cost_fn(tour0)
    for s in range(1, int(count) + 1):
        out[str(s)] = cost_fn(FX.build("ge8", xs, ys, seed=s))
    vals = list(out.values()) + [c0]
    return dict(cost0=c0, tour0=tour0, jitter=out, best_cost=min(vals), worst_cost=max(vals),
                range_pct=100.0 * (max(vals) - min(vals)) / min(vals),
                build_time=time.perf_counter() - t0, n_seeds=len(out))


def detied(xs, ys, cost_fn, angles, method="ge8"):
    """Beraberliksiz koordinatta aci taramasi (varsayilan GE k=8; 2026-09-09:
    her kurucu -- ayni gurultu, ayni tohum, ayni acilar)."""
    n = len(xs)
    rng = random.Random(20260907 + n)
    eps = FX.DETIED_REL * FX.median_nn_dist(xs, ys)
    dxs = [v + rng.uniform(-eps, eps) for v in xs]
    dys = [v + rng.uniform(-eps, eps) for v in ys]
    r = sweep(method, dxs, dys, cost_fn, angles)
    r["method"] = method
    r["detied_eps"] = eps
    r["tie_frac_before"] = FX.edge_tie_fraction(xs, ys)
    r["tie_frac_after"] = FX.edge_tie_fraction(dxs, dys)
    return r


def exact_knn(xs, ys, k):
    """KESIN k-NN (O(n^2), numpy, parca parca). grid_knn sozlesmesi: en yakin once."""
    import numpy as np
    X = np.asarray(xs, float)
    Y = np.asarray(ys, float)
    n = len(X)
    out = []
    for s in range(0, n, 2000):
        D = np.hypot(X[s:s + 2000, None] - X[None, :], Y[s:s + 2000, None] - Y[None, :])
        for i in range(D.shape[0]):
            D[i, s + i] = np.inf
        idx = np.argpartition(D, k, axis=1)[:, :k]
        for i in range(D.shape[0]):
            row = idx[i]
            out.append(row[np.argsort(D[i, row], kind="stable")].tolist())
    return out


def detied_exactknn(xs, ys, cost_fn, angles):
    """Beraberliksiz koordinat + KESIN aday listesiyle aci taramasi (GE k=8).
    grid_knn gecici olarak kesin surumle degistirilir (tek is parcacigi)."""
    import tsplib_engine as E
    orig = E.grid_knn
    E.grid_knn = exact_knn
    try:
        r = detied(xs, ys, cost_fn, angles)
    finally:
        E.grid_knn = orig
    r["knn"] = "exact"
    return r


def bands(xs, ys, cost_fn, counts=BAND_COUNTS):
    """theta=0'da bant taramasi; en iyi b'nin turu doner."""
    n = len(xs)
    rows, best = [], None
    t_all = time.perf_counter()
    ks = max(1, int(round(math.sqrt(n / 2.0))))     # strip'in kendi serit sayisi (E.snake_order)
    for b in tuple(counts) + (BAND_KS,):
        tag = None
        if b == BAND_KS:
            b, tag = ks, BAND_KS
        if b > 1 and b > n // 8 and tag is None:
            continue
        t0 = time.perf_counter()
        t = SA._band_hybrid_tour(xs, ys, SA._greedy_band_order_k(8), 0.0,
                                 dks=(0,), axes=(True,), k0=b)
        c = cost_fn(t)
        row = dict(b=b, n_bands=int(SA.LAST_BUILD.get("n_bands", b)), cost=c,
                   time=round(time.perf_counter() - t0, 3))
        if tag:
            row["tag"] = tag
        rows.append(row)
        if best is None or c < best[0]:
            best = (c, b, t)
    c1 = next((r["cost"] for r in rows if r["b"] == 1 and not r.get("tag")), None)
    return dict(cost=best[0], best_b=best[1], tour=best[2], bands=rows, cost_b1=c1, ks=ks,
                worst_cost=max(r["cost"] for r in rows), build_time=time.perf_counter() - t_all)


def realign(method, xs, ys, cost_fn, phi, base_tour_ge8=None):
    """Hizasiz kart (phi ile dondurulmus) -> dedektorler -> snap -> kurucu."""
    n = len(xs)
    t_all = time.perf_counter()
    mx, my = FX.rot(xs, ys, float(phi))
    mis_tour = FX.build(method, mx, my)
    mis_cost = cost_fn(mis_tour)
    det = {}
    for dk, f in FX.DETECTORS.items():
        th, conf = f(mx, my)
        det[dk] = dict(theta=round(th, 5), conf=round(float(conf), 4),
                       err=round(abs(FX.fold90(th - phi)), 5))
    integer = all(float(v).is_integer() for v in xs) and all(float(v).is_integer() for v in ys)
    snap = dict(integer_coords=integer)
    if integer:
        L = FX._Lattice(mx, my)
        spacing = 1.0          # tamsayi kafes: birim 1 (bkz. FX.run_instance notu)
        R0 = max(L.extent * 0.02, 50.0 * spacing)
        cands = {dk: det[dk]["theta"] for dk in ("comb", "nndir")}
        # iki kaba adayi da incelt, nihai tam-kart kalintisi kucuk olani al
        refined = {dk: FX.refine_lattice(mx, my, t, spacing=spacing, L=L) for dk, t in cands.items()}
        pick = min(refined, key=lambda k: refined[k][1])
        th_r, resid_r = refined[pick]
        ax, ay = FX.rot(mx, my, -th_r)
        sx = [float(int(round(v))) for v in ax]
        sy = [float(int(round(v))) for v in ay]
        exact = False
        for q in (0, 90, 180, 270):
            qx, qy = FX.rot(xs, ys, q)
            if sorted(zip([int(round(v)) for v in qx], [int(round(v)) for v in qy])) == \
                    sorted(zip([int(v) for v in sx], [int(v) for v in sy])):
                exact = True
                break
        snap.update(coarse_from=pick, theta=round(th_r, 6), resid_mean=resid_r,
                    exact_recovery=exact)
        rx, ry = sx, sy
    else:
        th_r = det["nndir"]["theta"]
        snap.update(coarse_from="nndir", theta=round(th_r, 6), exact_recovery=False)
        rx, ry = FX.rot(mx, my, -th_r)
    tour = FX.build(method, rx, ry)
    cost = cost_fn(tour)
    out = dict(phi=float(phi), theta_hat=snap["theta"], err_deg=round(abs(FX.fold90(snap["theta"] - phi)), 6),
               det=det, snap=snap, mis_cost=mis_cost, cost=cost, tour=tour,
               build_time=time.perf_counter() - t_all)
    if method == "ge8":
        if base_tour_ge8 is None:
            base_tour_ge8 = FX.build("ge8", xs, ys)
        out["ge8_tour_identical"] = (FX.canon(list(tour)) == FX.canon(list(base_tour_ge8)))
    return out


def nn_exact(xs, ys):
    return FX.nn_exact_tour(xs, ys, 0)


def realign_noise(method, xs, ys, cost_fn, phi, noise_rel=NOISE_REL, drops=NOISE_DROPS, seed=20260909):
    """2026-09-09 (hakem E8): saha gurultusu altinda dedektor + snap.
    Kart phi ile dondurulur, her koordinata N(0, sigma) eklenir
    (sigma = noise_rel x medyan komsu mesafesi); dedektorler + kafes inceltme
    yalniz gurultulu bulutu gorur. Tur GURULTULU koordinatta kurulur, maliyet
    ORIJINAL koordinatta olculur (algilama gurultulu, ucus gercek konumda).
    Ayrica drops oranlarinda nokta silinmis alt kumelerde yalniz dedektor/snap
    hatasi olculur (tur yok)."""
    n = len(xs)
    rng = random.Random(seed + n)
    t_all = time.perf_counter()
    sig = noise_rel * FX.median_nn_dist(xs, ys)
    mx, my = FX.rot(xs, ys, float(phi))
    mx = [v + rng.gauss(0.0, sig) for v in mx]
    my = [v + rng.gauss(0.0, sig) for v in my]

    def _detect(px, py):
        det = {}
        for dk, f in FX.DETECTORS.items():
            th, conf = f(px, py)
            det[dk] = dict(theta=round(th, 5), conf=round(float(conf), 4),
                           err=round(abs(FX.fold90(th - phi)), 5))
        L = FX._Lattice(px, py)
        refined = {dk: FX.refine_lattice(px, py, det[dk]["theta"], spacing=1.0, L=L) for dk in ("comb", "nndir")}
        pick = min(refined, key=lambda k: refined[k][1])
        th_r, resid_r = refined[pick]
        return det, dict(coarse_from=pick, theta=round(th_r, 6), resid_mean=resid_r,
                         err_deg=round(abs(FX.fold90(th_r - phi)), 6))

    det, snap = _detect(mx, my)
    mis_tour = FX.build(method, mx, my)
    rx, ry = FX.rot(mx, my, -snap["theta"])
    tour = FX.build(method, rx, ry)
    ox, oy = FX.rot(mx, my, -float(phi))          # oracle: gercek aciyla geri dondur
    or_tour = FX.build(method, ox, oy)
    out = dict(phi=float(phi), sigma=sig, noise_rel=noise_rel, theta_hat=snap["theta"],
               err_deg=snap["err_deg"], det=det, snap=snap,
               mis_cost=cost_fn(mis_tour), cost=cost_fn(tour), oracle_cost=cost_fn(or_tour), tour=tour)
    dd = {}
    for fr in drops:
        keep = sorted(rng.sample(range(n), max(10, int(round(n * (1.0 - fr))))))
        d_, s_ = _detect([mx[i] for i in keep], [my[i] for i in keep])
        dd[f"{fr:g}"] = dict(n_kept=len(keep), err_deg=s_["err_deg"], resid_mean=s_["resid_mean"],
                             det={k: v["err"] for k, v in d_.items()})
    out["drop"] = dd
    out["build_time"] = time.perf_counter() - t_all
    return out


def pge_kmeans_seeds(xs, ys, ewt, seeds=5, ks=PG.K_CHOICES, global_tour=None, t_global=None):
    """2026-09-09 (hakem E10): k-means++ tohum duyarliligi. Her k icin `seeds`
    farkli tohumla bolumleme + parca GE; tohum basina makespan."""
    ecost = PG.edge_cost_fn(ewt)
    n = len(xs)
    rows = []
    for k in ks:
        if k > 1 and n // k < 8:
            continue
        for sd in range(1, int(seeds) + 1):
            t0 = time.perf_counter()
            parts = PG.partition_kmeans(xs, ys, k, seed=20260907 + sd)
            tours, lens, times = PG.build_parts(parts, xs, ys, ecost)
            rows.append(dict(k=k, seed=sd, makespan=float(max(lens)), total=float(sum(lens)),
                             imbalance=float(max(lens) / (sum(lens) / len(lens))), sizes=[len(p) for p in parts],
                             time=round(time.perf_counter() - t0, 4)))
    return rows

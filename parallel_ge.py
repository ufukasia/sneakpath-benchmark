# -*- coding: utf-8 -*-
"""PARALEL GREEDY-EDGE -- k arac (drone) icin bolumleme + bant-ici GE.

SENARYO (kullanici, 2026-09-07): bir tarlada elma toplayan k drone. Tek drone
tek GE turu izlemek yerine, tarla k parcaya bolunur ve her drone KENDI
parcasinda GE turunu ayni anda izler. Amac: butun tarlanin en kisa surede
bitmesi. Bu, tek dikisli turun uzunlugundan (RSGE'nin olctugu sey) FARKLI bir
amactir; dogru olcutler:

    total     = sum_i L_i          (toplam yol; enerji)
    makespan  = max_i L_i          (en uzun tur; tarla ne zaman biter)
    imbalance = makespan / mean_i L_i
    t_seq     = sum_i t_i + t_part (tek makinede kurma suresi)
    t_par     = max_i t_i + t_part (k makinede / k drone'da kurma suresi)
    speedup   = t_GE_global / t_par

Her drone kendi parcasinda KAPALI bir tur izler (basladigi yere doner);
parcalar arasinda dikis YOKTUR -- paralel senaryoda buna gerek yok.

BOLUMLEME STRATEJILERI (makalenin ekseni: cerceveye bagimli mi?):

    band    : theta acisinda dondurulmus x ekseni boyunca k ESIT-SAYILI bant
              (her drone'a ayni sayida agac). CERCEVEYE BAGIMLI -- bantlarin
              yonu aciya baglidir; hizali cercevede bantlar kafes satirlarini
              izler, hizasizda testere disi olur. Aci panelden / dedektorden.
    kmeans  : k-means kumeleri (k-means++ baslatma, sabit tohum, Lloyd).
              DONME-DEGISMEZ. Dengesiz olabilir (kume boyutlari esit degil).
    split   : once GLOBAL GE turu, sonra tur k esit-sayili ardisik parcaya
              kesilir, her parca kapatilir (klasik "route-first, cluster-
              second"; mTSP literaturu). DONME-DEGISMEZ; paralel KURMA
              kazanci yoktur (global tur zaten kurulur), ama parca kalitesi
              iyi bir referanstir.

Bant-ici kurucu her stratejide AYNI: greedy_edge_tour(k=8). Boylece uc
satir arasindaki tek degisken BOLUMLEMEdir.
"""
from __future__ import annotations

import math
import time

import numpy as np

import tsplib_engine as E

KNN = 8
#: kd (2026-09-08, sef karari): DENGELI 2-B bolumleme -- k-d agaci tarzi ardisik
#: medyan ikiye bolme (en genis eksen boyunca), her yaprak esit sayida nokta.
#: 2x2 tasarimi tamamlar: {1-B bant, 2-B} x {dengeli, dengesiz(k-means)}; k-means'in
#: kazanci "2-B" mi "dengesizlik" mi ayrisir. Donme-degismez DEGIL (eksen-hizali
#: kesimler) ama bant gibi tek eksene bagli da degil.
STRATEGIES = ("band", "kd", "kmeans", "split")
K_CHOICES = (2, 4, 8)
K_SWEEP = (2, 3, 4, 6, 8, 12, 16)


def edge_cost_fn(ewt):
    """TSPLIB kenar maliyeti (yuvarlama dahil) -- alt-tur maliyeti icin."""
    w = str(ewt).upper()
    if w == "CEIL_2D":
        return E.tsplib_ceil
    if w == "ATT":
        return E.tsplib_att
    if w == "GEO":
        return E.tsplib_geo
    return E.tsplib_euc


def tour_cost(tour, xs, ys, ecost):
    m = len(tour)
    if m < 2:
        return 0.0
    return float(sum(ecost(xs[tour[i]], ys[tour[i]], xs[tour[(i + 1) % m]], ys[tour[(i + 1) % m]])
                     for i in range(m)))


# ---------------------------------------------------------------------------
#  Bolumleme
# ---------------------------------------------------------------------------
def partition_band(xs, ys, k, theta=0.0):
    """theta'da dondurulmus x boyunca k esit-SAYILI bant (indeks listeleri)."""
    n = len(xs)
    if abs(theta) > 1e-12:
        rx, _ = E.rotate_coords(xs, ys, float(theta))
    else:
        rx = xs
    order = sorted(range(n), key=lambda i: (rx[i], i))
    k = max(1, min(int(k), n))
    parts, base, rem = [], n // k, n % k
    p = 0
    for j in range(k):
        m = base + (1 if j < rem else 0)
        parts.append(order[p:p + m])
        p += m
    return [b for b in parts if b]


def partition_kd(xs, ys, k, theta=0.0):
    """Dengeli 2-B bolumleme: k parcaya ulasacak sekilde ardisik medyan ikiye
    bolme; her adimda EN GENIS eksen kesilir, sayilar esit paylasilir (k'nin
    2'nin kuvveti olmasi gerekmez: bir parca m noktali ve j parcaya
    bolunecekse, ilk parca floor(j/2)/j oraninda nokta alir)."""
    n = len(xs)
    X = np.asarray(xs, float)
    Y = np.asarray(ys, float)
    if abs(theta) > 1e-12:
        rx, ry = E.rotate_coords(xs, ys, float(theta))
        X, Y = np.asarray(rx), np.asarray(ry)
    k = max(1, min(int(k), n))
    out = []

    def rec(idx, j):
        if j <= 1 or len(idx) <= 1:
            out.append(idx.tolist())
            return
        j1 = j // 2
        m1 = int(round(len(idx) * j1 / j))
        ext_x = X[idx].max() - X[idx].min()
        ext_y = Y[idx].max() - Y[idx].min()
        key = X[idx] if ext_x >= ext_y else Y[idx]
        order = idx[np.lexsort((idx, key))]
        rec(order[:m1], j1)
        rec(order[m1:], j - j1)

    rec(np.arange(n), k)
    return [p for p in out if p]


def partition_kmeans(xs, ys, k, seed=20260907, iters=40):
    P = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    n = len(P)
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    # k-means++ baslatma
    C = np.empty((k, 2))
    C[0] = P[rng.integers(n)]
    d2 = ((P - C[0]) ** 2).sum(1)
    for j in range(1, k):
        pr = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        C[j] = P[rng.choice(n, p=pr)]
        d2 = np.minimum(d2, ((P - C[j]) ** 2).sum(1))
    lab = None
    for _ in range(iters):
        D = ((P[:, None, :] - C[None, :, :]) ** 2).sum(2) if n * k <= 4_000_000 else None
        if D is None:      # bellek: parca parca
            lab_new = np.empty(n, dtype=int)
            for s in range(0, n, 20000):
                blk = P[s:s + 20000]
                lab_new[s:s + 20000] = ((blk[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
        else:
            lab_new = D.argmin(1)
        if lab is not None and np.array_equal(lab, lab_new):
            break
        lab = lab_new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = P[m].mean(0)
    parts = [np.flatnonzero(lab == j).tolist() for j in range(k)]
    return [p for p in parts if p]


def partition_split(xs, ys, k, global_tour=None):
    """Global GE turunu k esit-sayili ardisik parcaya keser."""
    n = len(xs)
    t = global_tour if global_tour is not None else E.greedy_edge_tour(xs, ys, k=KNN)
    k = max(1, min(int(k), n))
    parts, base, rem, p = [], n // k, n % k, 0
    for j in range(k):
        m = base + (1 if j < rem else 0)
        parts.append(list(t[p:p + m]))
        p += m
    return [b for b in parts if b]


# ---------------------------------------------------------------------------
#  Degerlendirme
# ---------------------------------------------------------------------------
def build_parts(parts, xs, ys, ecost, already_tours=False):
    """Her parcada GE (k=8) kapali tur; (turlar, uzunluklar, sureler)."""
    tours, lens, times = [], [], []
    for b in parts:
        t0 = time.perf_counter()
        if already_tours or len(b) <= 3:
            seg = list(b)
        else:
            lx = [xs[i] for i in b]
            ly = [ys[i] for i in b]
            loc = E.greedy_edge_tour(lx, ly, k=min(KNN, len(b) - 1))
            seg = [b[j] for j in loc]
        times.append(time.perf_counter() - t0)
        tours.append(seg)
        lens.append(tour_cost(seg, xs, ys, ecost))
    return tours, lens, times


def evaluate(strategy, xs, ys, k, ewt="EUC_2D", theta=0.0, global_tour=None,
             t_global=None):
    """Tek (strateji, k) hucresi. Doner: sozluk (metrikler + parcalar)."""
    ecost = edge_cost_fn(ewt)
    t0 = time.perf_counter()
    if strategy == "band":
        parts = partition_band(xs, ys, k, theta)
    elif strategy == "kd":
        parts = partition_kd(xs, ys, k, theta)
    elif strategy == "kmeans":
        parts = partition_kmeans(xs, ys, k)
    elif strategy == "split":
        if global_tour is None:
            tg0 = time.perf_counter()
            global_tour = E.greedy_edge_tour(xs, ys, k=KNN)
            t_global = time.perf_counter() - tg0
        parts = partition_split(xs, ys, k, global_tour)
    else:
        raise ValueError(strategy)
    t_part = time.perf_counter() - t0
    if strategy == "split":
        # parcalar zaten tur sirasinda: yalniz kapatma maliyeti; kurma suresi
        # global turun suresidir (paralel kazanc YOK)
        tours, lens, times = build_parts(parts, xs, ys, ecost, already_tours=True)
        t_seq = (t_global or 0.0) + t_part
        t_par = (t_global or 0.0) + t_part
    else:
        tours, lens, times = build_parts(parts, xs, ys, ecost)
        t_seq = sum(times) + t_part
        t_par = max(times) + t_part
    total = float(sum(lens))
    mk = float(max(lens))
    mean = total / len(lens)
    return dict(strategy=strategy, k_requested=int(k), k=len(parts), theta=float(theta),
                total=total, makespan=mk, mean=mean, imbalance=(mk / mean if mean > 0 else 1.0),
                sizes=[len(p) for p in parts], lens=[round(v, 2) for v in lens],
                t_part=t_part, t_seq=t_seq, t_par=t_par, times=[round(v, 4) for v in times],
                tours=tours)


def repair_parts(tours, xs, ys, ewt, cfg=None, budget_s=None, mode="vnd"):
    """Her parcanin turunu BAGIMSIZ olarak onarir (drone kendi parcasinda
    yerel arama yapar). Motor: repair.run_single_neighborhood (panelin
    repair_vnd satiriyla AYNI VND: 2-opt + Or-opt + relocate). Parca basina
    ayni butce; paralel duvar-saati = en uzun parca. Doner: (yeni turlar,
    uzunluklar, sureler)."""
    import repair as REP
    cfg = cfg or {}
    ecost = edge_cost_fn(ewt)
    out, lens, times = [], [], []
    for t in tours:
        t0 = time.perf_counter()
        if len(t) < 5:
            out.append(list(t)); lens.append(tour_cost(t, xs, ys, ecost)); times.append(0.0)
            continue
        coords = [(xs[i], ys[i]) for i in t]
        inst, _ = E.make_instance(coords, ewt)
        local = list(range(len(t)))
        dl = (t0 + budget_s) if budget_s else None
        r = REP.run_single_neighborhood(inst, local, mode,
                                        neighbor_limit_2opt=cfg.get("nb2", 20),
                                        neighbor_limit_oropt=cfg.get("nbo", 20),
                                        max_oropt_segment=3, k_relocate=cfg.get("rk", 30),
                                        relocate_neighbor_limit=cfg.get("rnb", 120),
                                        max_rounds=40, deadline=dl)
        seg = [t[j] for j in r.tour]
        out.append(seg)
        lens.append(tour_cost(seg, xs, ys, ecost))
        times.append(time.perf_counter() - t0)
    return out, lens, times


def ils_parts(tours, xs, ys, ewt, budget_s=1.0, seed=1):
    """Her parcanin turunda BAGIMSIZ ILS. Makale sozlesmesi: ILS tek basina
    degil, birlesik onarimin (VND) USTUNE tohumlanir (runner'daki `ils`
    satirinin tohum kuraliyla ayni). Motor: lro.build_ils_result --
    benchmark'taki `ils` satiriyla AYNI kod, gorsellestirme icin kucultulmus
    butce. Doner: (yeni turlar, uzunluklar, sureler)."""
    import repair as REP
    import line_reassign_optimizer as lro
    ecost = edge_cost_fn(ewt)
    out, lens, times = [], [], []
    for t in tours:
        t0 = time.perf_counter()
        if len(t) < 5:
            out.append(list(t)); lens.append(tour_cost(t, xs, ys, ecost)); times.append(0.0)
            continue
        coords = [(xs[i], ys[i]) for i in t]
        inst, _ = E.make_instance(coords, ewt)
        dl = (t0 + budget_s) if budget_s else None
        r0 = REP.run_single_neighborhood(inst, list(range(len(t))), "vnd",
                                         neighbor_limit_2opt=20, neighbor_limit_oropt=20,
                                         max_oropt_segment=3, k_relocate=30,
                                         relocate_neighbor_limit=120, max_rounds=40,
                                         deadline=dl)
        remain = (dl - time.perf_counter()) if dl else 5.0
        _, diag = lro.build_ils_result(
            inst, list(r0.tour), max_iterations=120, max_no_improve=40,
            max_time_seconds=max(0.05, remain), seed=seed, max_ls_rounds=10,
            neighbor_limit_2opt=20, neighbor_limit_oropt=20,
            k_nearest_edges_relocate=30, city_neighbor_limit_relocate=120)
        seg = [t[j] for j in diag.tour]
        out.append(seg)
        lens.append(tour_cost(seg, xs, ys, ecost))
        times.append(time.perf_counter() - t0)
    return out, lens, times


def evaluate_repaired(strategy, xs, ys, k, ewt="EUC_2D", theta=0.0, cfg=None, budget_s=None,
                      global_tour=None, t_global=None):
    """evaluate(...) + parca basina VND onarimi. Alanlar: *_rep."""
    r = evaluate(strategy, xs, ys, k, ewt, theta=theta, global_tour=global_tour, t_global=t_global)
    tours, lens, times = repair_parts(r["tours"], xs, ys, ewt, cfg=cfg, budget_s=budget_s)
    tot = float(sum(lens)); mk = float(max(lens)); mean = tot / len(lens)
    r.update(tours_rep=tours, lens_rep=[round(v, 2) for v in lens], total_rep=tot, makespan_rep=mk,
             imbalance_rep=(mk / mean if mean > 0 else 1.0), rep_times=[round(v, 3) for v in times],
             t_rep_par=max(times), t_rep_seq=sum(times), t_par_total=r["t_par"] + max(times))
    return r


def concat_tour(tours):
    """Panel gorsellestirmesi icin parca turlarinin birlestirilmis permutasyonu."""
    out = []
    for t in tours:
        out.extend(t)
    return out

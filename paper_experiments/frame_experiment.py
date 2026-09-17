# -*- coding: utf-8 -*-
"""CERCEVE (ACI) DUYARLILIGI DENEYI -- makalenin tum olcumleri, tek kosum.

Tez (uc iddia, uc bolum):
  (1) Cerceveye BAGIMLI kurucular (strip, Hilbert, Morton, bantli GE) aciya
      buyuk olcude duyarlidir; dogru hizalama onlari kurtarir.
  (2) Donme-DEGISMEZ kurucular (Greedy-Edge, NN, FI) icin aci yalnizca bir
      BERABERLIK KIRMA zaridir: kazanc dagilimi sifir ortalamali, tie-jitter
      ile ayirt edilemez, beraberlikler kaldirilinca yok olur.
  (3) Hizasiz verilmis bir VLSI karti, bir aci dedektoruyle 0 hizasina geri
      getirilebilir (hata < 1 derece) -- bu strip'i kurtarir, GE'ye hic bir
      sey yapmaz. (Kullanicinin "hizasiz cipi hizala" deneyi.)
  (+) Bantlama (RSGE b ekseni) hizalama dogru olsa bile GE'ye maliyet ekler.

Kosum:
    python paper_experiments/frame_experiment.py --workers 8
    python paper_experiments/frame_experiment.py --quick --only xqf131,kroA100
    python paper_experiments/make_report.py

Her ornek icin tek bir JSON yazilir (out/per_instance/<ad>.json); var olan
dosya atlanir (resume). Kesilirse ayni komutla devam eder.

TASARIM KARARLARI (deneysel):
  * Tur MALIYETI her zaman ORIJINAL koordinatlarda, TSPLIB yuvarlamasiyla
    hesaplanir (make_instance(..., force_sparse=True).tour_cost). Dondurme
    yalnizca kurucunun GORDUGU koordinatlari degistirir; gap BKS'ye gore
    gercek kalir.
  * Aci taramasi [-90, 90) araligi, adim n'e gore 5/10/15/30 derece. Strip
    180 derece periyodiktir; GE zaten degismez. Tum yontemler AYNI acilarda
    kosulur ki dagilimlar dogrudan kiyaslanabilsin.
  * GE icin uc kontrol, ayni ornek sayisiyla (esit butce):
      rot    : theta taramasi (dondur -> GE)
      jitter : theta = 0, yalniz TAM beraberlikleri bozan tohumlu jitter
               (tie_eps = 1e-9)
      detied : koordinatlara ~1e-4 x medyan-NN-mesafesi kadar gurultu
               eklenmis (beraberliksiz) ornekte theta taramasi
    Beklenti: rot ~ jitter (ayirt edilemez), detied ~ 0 (yalniz grid_knn
    yaklasiklik hatasi kalir).
  * Yeniden hizalama: orijinal (hizali) kart phi in {7.5, 22.5, 37.5, rastgele}
    ile dondurulur ("hizasiz cip"), uc dedektor (comb = grid_theta.grid_angle,
    nndir = en-yakin-komsu kenar yonu dairesel ortalamasi, pca = ana eksen)
    aciyi tahmin eder, kart geri dondurulur, GE / strip / Hilbert / Morton
    yeniden kurulur. Ayrica "snap": geri dondurulmus koordinatlar tamsayiya
    yuvarlanir; orijinal kafesin BIREBIR geri geldigi ve GE turunun orijinalle
    bit-ayni oldugu sinanir.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import tsplib_engine as E          # noqa: E402
import snake_alt as SA             # noqa: E402
import grid_theta as GT            # noqa: E402
import rgge as RG                  # noqa: E402
import vlsi_datasets as V          # noqa: E402
import tsplib_datasets as T        # noqa: E402

VERSION = "fx-7"

OUT_DEFAULT = os.path.join(HERE, "out")

#: Ek TSPLIB kontrolleri (kafes olmayan / kumeli / duzgun-rastgele). results/
#: icinde zaten bulunanlar tekrar eklenmez.
EXTRA_TSPLIB = [
    "d493", "dsj1000", "pr1002", "u1060", "rd400", "pcb442", "pr2392", "d2103",
    "u2152", "pcb1173", "nrw1379", "d1291", "rl1323", "fl1400", "vm1084",
    "u1817", "d1655", "lin318", "a280", "pr136", "pr144", "pr299", "pr439",
    "u724", "rat575", "p654", "d657", "gil262", "kroB200", "tsp225",
    "bier127", "ch150", "eil101", "u159", "d198", "rl1889", "pr226", "pr264",
    "ts225",
]
GEO_SKIP = {"ali535", "gr666"}

INVARIANT = ("ge8", "ge15", "nn", "nn_grid", "fi")
DEPENDENT = ("strip", "hilbert", "morton", "band2ge")
ALL_METHODS = INVARIANT + DEPENDENT

FI_MAX_N = 3000            # FI saf O(n^2); ustunde atlanir
FI_MAX_ANGLES = 18
REALIGN_PHIS = (7.5, 22.5, 37.5, "random")
REALIGN_METHODS = ("ge8", "strip", "hilbert", "morton")
BAND_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16)
JITTER_EPS = 1e-9
DETIED_REL = 1e-4


# ---------------------------------------------------------------------------
#  Ornek yukleme
# ---------------------------------------------------------------------------
def load_instances(only=None, min_n=0, max_n=None):
    seen, out = set(), []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        name = d.get("dataset")
        if not name or name in seen or name in GEO_SKIP or d.get("is_geo"):
            continue
        coords = d.get("coords")
        if not coords or not d.get("bks"):
            continue
        if str(d.get("ewt", "EUC_2D")).upper() not in ("EUC_2D", "CEIL_2D"):
            continue
        cls = "vlsi" if name in V.CATALOG else "tsplib"
        out.append(dict(name=name, cls=cls, coords=[tuple(c) for c in coords],
                        ewt=d["ewt"], bks=float(d["bks"])))
        seen.add(name)
    for name in EXTRA_TSPLIB:
        if name in seen or name not in T.CATALOG:
            continue
        p = os.path.join(ROOT, "data_tsplib", f"{name}.tsp")
        if not os.path.exists(p):
            continue
        try:
            _h, coords, ewt = E.parse_tsp(p)
        except Exception:
            continue
        if str(ewt).upper() not in ("EUC_2D", "CEIL_2D"):
            continue
        out.append(dict(name=name, cls="tsplib", coords=[tuple(c) for c in coords],
                        ewt=ewt, bks=float(T.CATALOG[name][1])))
        seen.add(name)
    if only:
        want = set(only)
        out = [o for o in out if o["name"] in want]
    out = [o for o in out if len(o["coords"]) >= min_n
           and (max_n is None or len(o["coords"]) <= max_n)]
    out.sort(key=lambda o: -len(o["coords"]))       # buyuk once: yuk dengesi
    return out


# ---------------------------------------------------------------------------
#  Yardimcilar
# ---------------------------------------------------------------------------
def angle_grid(n, quick=False):
    if quick:
        step = 30
    elif n <= 3000:
        step = 5
    elif n <= 15000:
        step = 10
    elif n <= 50000:
        step = 15
    else:
        step = 30
    return [float(a) for a in range(-90, 90, step)]


def fold90(a):
    """Aciyi (-45, 45] araligina katlar (kafesin 90 derece simetrisi)."""
    return (float(a) + 45.0) % 90.0 - 45.0


def rot(xs, ys, deg):
    if abs(deg) < 1e-12:
        return list(xs), list(ys)
    return E.rotate_coords(xs, ys, float(deg))


def build(method, xs, ys, seed=None):
    if method == "ge8":
        return E.greedy_edge_tour(xs, ys, k=8, tie_seed=seed,
                                  tie_eps=(JITTER_EPS if seed is not None else 0.0))
    if method == "ge15":
        return E.greedy_edge_tour(xs, ys, k=15)
    if method == "nn":
        return nn_exact_tour(xs, ys, 0)
    if method == "nn_grid":
        return E.grid_nn_tour(xs, ys, 0)
    if method == "fi":
        return E.farthest_insertion_tour(xs, ys, 0)
    if method == "strip":
        return E.snake_order(xs, ys, 0)
    if method == "hilbert":
        return E.hilbert_curve_tour(xs, ys)
    if method == "morton":
        return E.morton_curve_tour(xs, ys)
    if method == "band2ge":
        return SA._band_hybrid_tour(xs, ys, SA._greedy_band_order_k(8), 0.0,
                                    dks=(0,), axes=(True,), k0=2)
    raise ValueError(method)


def canon(t):
    """Kapali turu kanonik bicime getirir (baslangic + yon bagimsiz)."""
    i = t.index(0)
    a = t[i:] + t[:i]
    return a if len(a) < 3 or a[1] <= a[-1] else [a[0]] + a[1:][::-1]


def median_nn_dist(xs, ys):
    knn = E.grid_knn(xs, ys, 1)
    d = sorted(math.hypot(xs[i] - xs[knn[i][0]], ys[i] - ys[knn[i][0]])
               for i in range(len(xs)) if knn[i])
    d = [v for v in d if v > 0] or [1.0]
    return d[len(d) // 2]


def edge_tie_fraction(xs, ys, k=8):
    """GE'nin GERCEKTEN gordugu aday kenar kumesinde (k-NN, kanonik cift,
    cift sayim yok) tekrarli uzunluk orani. Taban yok: beraberliksiz bulutta
    ~0 cikar."""
    knn = E.grid_knn(xs, ys, min(k, len(xs) - 1))
    seen, lens = set(), []
    for i, nb in enumerate(knn):
        for j in nb:
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            lens.append(round(math.hypot(xs[i] - xs[j], ys[i] - ys[j]), 9))
    if not lens:
        return 0.0
    from collections import Counter
    c = Counter(lens)
    tied = sum(v for v in c.values() if v > 1)
    return tied / len(lens)


# ---------------------------------------------------------------------------
#  Aci dedektorleri -- hepsi "hizalamak icin -theta ile dondur" sozlesmesinde
#  (-45, 45] doner. Isaret sozlesmesi smoke testte dogrulanir (phi verilmis
#  dondurulmus kartta hata ~0 olmali).
# ---------------------------------------------------------------------------
def det_comb(xs, ys):
    # grid_angle dogrudan "kart phi ile dondurulmus" -> phi dondurur
    # (duman testi: phi=7.5 -> 7.5; isaret cevrilmez).
    th, conf = GT.grid_angle(xs, ys)
    return fold90(th), float(conf)


def nn_exact_tour(xs, ys, start=0):
    """KESIN en-yakin-komsu (O(n^2), numpy). Beraberlikte en kucuk indeks.
    `grid_nn_tour` izgara-hizlandirmali ve yaklasiktir; degismezlik iddiasi
    kesin surumde okunmali, izgara etkisi ayri satirda gorunmeli."""
    X = np.asarray(xs, float)
    Y = np.asarray(ys, float)
    n = len(X)
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        d = np.hypot(X - X[cur], Y - Y[cur])
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
    return tour


def det_nndir(xs, ys):
    knn = E.grid_knn(xs, ys, 1)
    X = np.asarray(xs, float)
    Y = np.asarray(ys, float)
    j = np.asarray([nb[0] if nb else i for i, nb in enumerate(knn)])
    dx, dy = X[j] - X, Y[j] - Y
    ok = (dx != 0) | (dy != 0)
    if not ok.any():
        return 0.0, 0.0
    alpha = np.arctan2(dy[ok], dx[ok])
    z = np.exp(1j * 4.0 * alpha).mean()
    th = math.degrees(np.angle(z)) / 4.0
    return fold90(th), float(abs(z))


def det_pca(xs, ys):
    P = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    C = np.cov((P - P.mean(axis=0)).T)
    w, v = np.linalg.eigh(C)
    main = v[:, int(np.argmax(w))]
    th = math.degrees(math.atan2(main[1], main[0]))
    ratio = float(w.max() / (w.min() + 1e-12))
    return fold90(th), ratio


DETECTORS = {"comb": det_comb, "nndir": det_nndir, "pca": det_pca}


class _Lattice:
    """Kafes kalintisi hesaplayicisi. Kart merkez etrafinda -th ile geri
    dondurulunce koordinatlarin en yakin tamsayiya ortalama L1 uzakligi.
    Kafes tam geri geldiyse ~1e-13.

    `radius` verilirse yalniz merkeze o kadar yakin noktalar kullanilir:
    kalinti fonksiyonunun aci ekseninde havza genisligi ~ aralik / yaricap
    (radyan) oldugundan, kucuk yaricap = genis havza = kaba arama; buyuk
    yaricap = dar havza = ince arama. Hiyerarsik inceltme buna dayanir."""

    def __init__(self, mx, my):
        P = np.column_stack([np.asarray(mx, float), np.asarray(my, float)])
        self.c = P.mean(axis=0)            # E.rotate_coords ile AYNI merkez
        self.C = P - self.c
        self.r = np.hypot(self.C[:, 0], self.C[:, 1])
        self.order = np.argsort(self.r)
        self.extent = float(self.r.max()) or 1.0

    MIN_PTS = 60

    def subset(self, radius=None):
        """(alt kume, GERCEK yaricap). Merkez bos olabilir (PCB'lerde sik):
        istenen yaricapta MIN_PTS'ten az nokta varsa en yakin MIN_PTS nokta
        alinir ve gercek yaricap buna gore buyur -- adim buyuklugu bu GERCEK
        yaricaptan turetilmeli, yoksa havza kacirilir (pcb3038, phi=7.5)."""
        if radius is None:
            return self.C, self.extent
        k = max(self.MIN_PTS, int((self.r <= radius).sum()))
        idx = self.order[:k]
        return self.C[idx], float(self.r[idx[-1]]) or 1.0

    def resid(self, th, radius=None):
        a = math.radians(-th)
        ca, sa = math.cos(a), math.sin(a)
        C, _ = self.subset(radius)
        qx = self.c[0] + C[:, 0] * ca - C[:, 1] * sa
        qy = self.c[1] + C[:, 0] * sa + C[:, 1] * ca
        return float((np.abs(qx - np.round(qx)) + np.abs(qy - np.round(qy))).mean())


def lattice_residual(mx, my, th):
    return _Lattice(mx, my).resid(th)


def refine_lattice(mx, my, th0, spacing=1.0, halfwidth=1.0, L=None):
    """Kaba aci tahminini kafes kalintisini MINIMIZE ederek inceltir --
    HIYERARSIK: once merkeze yakin kucuk bir yaricapta (genis havza, kaba
    adim), sonra yaricap ikiye katlanip adim daraltilir, tum kart dahil
    olunca en ince adimla biter. Adim her kademede 0.25 x (aralik/yaricap)
    radyan, yani havzanin dortte biri; toplam degerlendirme ~100-200,
    her biri O(alt kume). Buyuk kartta tek olcekli tarama neden yetmez:
    n=50k, yayilim 10^4 birim -> havza 0.006 derece; 0.005'lik sabit adim
    ya kacirir ya da +-1 derecelik pencerede 400 tam-kart degerlendirme
    ister. Hiyerarsi bunu ~30 alt-kume + ~10 tam-kart degerlendirmeye
    indirir."""
    L = L or _Lattice(mx, my)
    best_th = float(th0)
    hw = float(halfwidth)
    R = max(L.extent * 0.02, 50.0 * spacing)
    while True:
        R = min(R, L.extent)
        _, R_act = L.subset(R)            # merkez bossa gercek yaricap > R
        step = 0.25 * math.degrees(spacing / R_act)
        best_r = L.resid(best_th, R)
        c = best_th
        for t in np.arange(c - hw, c + hw + 1e-12, step):
            r = L.resid(float(t), R)
            if r < best_r:
                best_r, best_th = r, float(t)
        hw = 2.0 * step
        if R >= L.extent:
            break
        R *= 2.0
    # son cila: tam kartta cok ince adim
    step = 0.05 * math.degrees(spacing / L.extent)
    best_r = L.resid(best_th)
    c = best_th
    for t in np.arange(c - 3 * step, c + 3 * step + 1e-12, step / 4):
        r = L.resid(float(t))
        if r < best_r:
            best_r, best_th = r, float(t)
    return best_th, best_r


# ---------------------------------------------------------------------------
#  Tek ornek
# ---------------------------------------------------------------------------
def run_instance(job):
    inst = job["inst"]
    quick = job.get("quick", False)
    name, cls = inst["name"], inst["cls"]
    coords, ewt, bks = inst["coords"], inst["ewt"], inst["bks"]
    n = len(coords)
    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    t_start = time.perf_counter()
    T_inst, _ = E.make_instance(coords, ewt, force_sparse=True)

    def cost(t):
        return float(T_inst.tour_cost(t))

    def gap(c):
        return 100.0 * (c / bks - 1.0)

    res = dict(version=VERSION, name=name, cls=cls, n=n, ewt=ewt, bks=bks)
    res["integer_coords"] = all(float(v).is_integer() for c in coords for v in c)
    tb, tr = RG.tie_measure(xs, ys)
    res["tie"] = dict(boundary=tb, repeat=tr, edge_tie_frac=edge_tie_fraction(xs, ys))
    res["median_nn"] = median_nn_dist(xs, ys)
    res["detect_orig"] = {k: list(f(xs, ys)) for k, f in DETECTORS.items()}

    # ---- (A) aci taramasi: tum yontemler, ayni acilar --------------------
    angles = angle_grid(n, quick)
    res["angles"] = angles
    methods = [m for m in ALL_METHODS if not (m == "fi" and (n > FI_MAX_N or quick))]
    fi_angles = angles if len(angles) <= FI_MAX_ANGLES else angles[::2]
    sweep = {m: {} for m in methods}
    stime = {m: [] for m in methods}
    for a in angles:
        rx, ry = rot(xs, ys, a)
        for m in methods:
            if m == "fi" and a not in fi_angles:
                continue
            t0 = time.perf_counter()
            t = build(m, rx, ry)
            stime[m].append(time.perf_counter() - t0)
            sweep[m][f"{a:g}"] = cost(t)
    res["sweep"] = sweep
    res["sweep_time"] = {m: (sum(v) / len(v) if v else None) for m, v in stime.items()}
    res["gap0"] = {m: gap(sweep[m]["0"]) for m in methods if "0" in sweep[m]}

    # ---- (B) GE beraberlik kontrolleri ---------------------------------
    jit = {}
    for s in range(1, len(angles) + 1):
        jit[str(s)] = cost(build("ge8", xs, ys, seed=s))
    res["jitter"] = jit
    rng = random.Random(20260907 + n)
    eps = DETIED_REL * res["median_nn"]
    dxs = [v + rng.uniform(-eps, eps) for v in xs]
    dys = [v + rng.uniform(-eps, eps) for v in ys]
    res["detied_eps"] = eps
    res["detied_tie_frac"] = edge_tie_fraction(dxs, dys)
    det_sw = {}
    for a in angles:
        rx, ry = rot(dxs, dys, a)
        det_sw[f"{a:g}"] = cost(build("ge8", rx, ry))
    res["detied_sweep"] = det_sw

    # ---- (C) hizasiz cip -> yeniden hizalama ------------------------------
    base_tour = canon(build("ge8", xs, ys))
    phis = list(REALIGN_PHIS[:1]) + ["random"] if quick else list(REALIGN_PHIS)
    rr = random.Random(777 + n)
    realign = []
    for ph in phis:
        phi = rr.uniform(-45.0, 45.0) if ph == "random" else float(ph)
        mx, my = rot(xs, ys, phi)
        entry = dict(phi=phi, random=(ph == "random"), mis={}, det={})
        for m in REALIGN_METHODS:
            entry["mis"][m] = cost(build(m, mx, my))
        for dk, f in DETECTORS.items():
            th, conf = f(mx, my)
            err = abs(fold90(th - phi))
            ax, ay = rot(mx, my, -th)
            d = dict(theta=th, conf=conf, err=err, realigned={})
            for m in REALIGN_METHODS:
                d["realigned"][m] = cost(build(m, ax, ay))
            entry["det"][dk] = d
        # ---- SNAP hatti: kaba aci = {comb, nndir} icinden merkez-alt-kume
        # kafes kalintisi kucuk olani (yer gercegi KULLANILMAZ), sonra
        # hiyerarsik kafes inceltmesi, sonra tamsayiya yuvarlama.
        if res["integer_coords"]:
            L = _Lattice(mx, my)
            # Tamsayi kafeste birim HER ZAMAN 1'dir: kalinti "en yakin tamsayiya"
            # olculur, dolayisiyla havza genisligi 1/yaricap radyandir; medyan
            # komsu mesafesiyle olceklemek (onceki surum) adimi 10x kabalastirip
            # havzayi kacirtiyordu (pcb3038'de 0.28 derecede takildi).
            spacing = 1.0
            R0 = max(L.extent * 0.02, 50.0 * spacing)
            cands = {dk: entry["det"][dk]["theta"] for dk in ("comb", "nndir")}
            cres = {dk: L.resid(t, R0) for dk, t in cands.items()}
            # IKI kaba adayi da incelt, NIHAI tam-kart kalintisi kucuk olani al
            # (yer gercegi kullanilmaz). Tek adaya guvenmek pcb3038/phi=7.5'te
            # 0.85 derecede takilmisti.
            refined = {dk: refine_lattice(mx, my, t, spacing=spacing, L=L) for dk, t in cands.items()}
            pick = min(refined, key=lambda k: refined[k][1])
            th_r, resid_r = refined[pick]
            ax, ay = rot(mx, my, -th_r)
            sx = [int(round(v)) for v in ax]
            sy = [int(round(v)) for v in ay]
            snap = dict(coarse_from=pick, coarse_theta=cands[pick],
                        coarse_resid_center={k: v for k, v in cres.items()},
                        raw_resid_full={dk: L.resid(t) for dk, t in cands.items()},
                        theta=th_r, err=abs(fold90(th_r - phi)), resid_mean=resid_r,
                        resid_max=max(abs(ax[i] - sx[i]) + abs(ay[i] - sy[i]) for i in range(n)))
            exact = False
            for q_ in (0, 90, 180, 270):
                qx, qy = rot(xs, ys, q_)
                if sorted(zip([int(round(v)) for v in qx],
                              [int(round(v)) for v in qy])) == sorted(zip(sx, sy)):
                    exact = True
                    break
            snap["exact_recovery"] = exact
            fsx = [float(v) for v in sx]
            fsy = [float(v) for v in sy]
            snap["realigned"] = {m: cost(build(m, fsx, fsy)) for m in REALIGN_METHODS}
            st = build("ge8", fsx, fsy)
            snap["ge8_cost"] = snap["realigned"]["ge8"]
            snap["ge8_tour_identical"] = (canon(st) == base_tour)
            snap["ge8_cost_identical"] = abs(snap["ge8_cost"] - sweep["ge8"]["0"]) < 0.5
            snap["strip_cost"] = snap["realigned"]["strip"]
            entry["snap"] = snap
        realign.append(entry)
    res["realign"] = realign

    # ---- (D) bantlama: theta = 0 (hizali), GE k=8 bant icinde -------------
    bands = []
    for b in (BAND_COUNTS[:4] if quick else BAND_COUNTS):
        if b > n // 8:
            continue
        t0 = time.perf_counter()
        t = SA._band_hybrid_tour(xs, ys, SA._greedy_band_order_k(8), 0.0,
                                 dks=(0,), axes=(True,), k0=b)
        dt = time.perf_counter() - t0
        bands.append(dict(b=b, n_bands=int(SA.LAST_BUILD.get("n_bands", b)),
                          cost=cost(t), time=dt))
    res["bands"] = bands
    res["elapsed"] = time.perf_counter() - t_start
    return res


def _worker(job):
    inst = job["inst"]
    path = os.path.join(job["out"], "per_instance", f"{inst['name']}.json")
    t0 = time.perf_counter()
    try:
        res = run_instance(job)
    except Exception:
        res = dict(version=VERSION, name=inst["name"], cls=inst["cls"],
                   n=len(inst["coords"]), error=traceback.format_exc())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    return inst["name"], len(inst["coords"]), time.perf_counter() - t0, "error" in res


# ---------------------------------------------------------------------------
#  Orkestra
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--only", default="", help="virgulle ayrilmis ornek adlari")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="duman testi: kaba aci, FI yok")
    ap.add_argument("--force", action="store_true", help="var olan sonuclari yeniden kos")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    insts = load_instances(only=only, min_n=args.min_n, max_n=args.max_n)
    if args.list:
        for o in insts:
            print(f"{o['name']:12s} {o['cls']:7s} n={len(o['coords'])}")
        print(len(insts), "ornek")
        return
    os.makedirs(os.path.join(args.out, "per_instance"), exist_ok=True)
    log_path = os.path.join(args.out, "progress.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    jobs = []
    for o in insts:
        p = os.path.join(args.out, "per_instance", f"{o['name']}.json")
        if not args.force and os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                if d.get("version") == VERSION and "error" not in d:
                    continue
            except Exception:
                pass
        jobs.append(dict(inst=o, out=args.out, quick=args.quick))
    log(f"{len(insts)} ornek, {len(jobs)} kosulacak, {args.workers} isci, "
        f"quick={args.quick}, version={VERSION}")
    if not jobs:
        return
    t0 = time.perf_counter()
    done = 0
    if args.workers <= 1:
        for j in jobs:
            nm, n, dt, err = _worker(j)
            done += 1
            log(f"{done}/{len(jobs)} {nm} n={n} {dt:.1f}s{'  HATA' if err else ''}")
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for nm, n, dt, err in pool.imap_unordered(_worker, jobs):
                done += 1
                log(f"{done}/{len(jobs)} {nm} n={n} {dt:.1f}s{'  HATA' if err else ''}")
    log(f"bitti: {len(jobs)} ornek, toplam {(time.perf_counter() - t0) / 60:.1f} dk")


if __name__ == "__main__":
    main()

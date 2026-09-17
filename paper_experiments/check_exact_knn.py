# -*- coding: utf-8 -*-
"""KONTROL: beraberliksiz (de-tied) koordinatta GE'nin kalan aci yayilimi
grid_knn YAKLASIKLIGINDAN mi geliyor?

fx-7 kosumunda 145 ornegin 33'unde de-tied aralik > 0.05% cikti (en buyuk
xqd4966: %7.0). Hipotez: kafes kumelerinde k=8 aday sinirinda es-uzaklikli
kabuk bulunur; de-tied gurultu (1e-4) siralamayi belirler ama `grid_knn`
(eksen-hizali izgara, halka arama) DONDURULMUS koordinatta farkli bir yaklasik
komsu kumesi dondurur -> GE farkli aday listesi gorur.

Test: ayni de-tied koordinatlarda ayni acilarla, kesin O(n^2) k-NN ile GE.
Beklenti: aralik ~0 (yalniz float 1e-13 kirilmalari kalir).

    python paper_experiments/check_exact_knn.py --top 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                  # noqa: E402
import tsplib_engine as E           # noqa: E402
import frame_experiment as FX       # noqa: E402


def exact_knn(xs, ys, k):
    X = np.asarray(xs, float)
    Y = np.asarray(ys, float)
    n = len(X)
    out = []
    for s in range(0, n, 2000):
        dx = X[s:s + 2000, None] - X[None, :]
        dy = Y[s:s + 2000, None] - Y[None, :]
        D = np.hypot(dx, dy)
        for i in range(D.shape[0]):
            D[i, s + i] = np.inf
        idx = np.argpartition(D, k, axis=1)[:, :k]
        # mesafeye gore sirala (grid_knn sozlesmesi: en yakin once)
        for i in range(D.shape[0]):
            row = idx[i]
            out.append(row[np.argsort(D[i, row], kind="stable")].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    rows = []
    for f in glob.glob(os.path.join(args.out, "per_instance", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if "error" in d or d["n"] > 6000:
            continue
        v = list(d["detied_sweep"].values())
        rows.append((100.0 * (max(v) - min(v)) / min(v), d))
    rows.sort(key=lambda r: -r[0])
    res = []
    for rng_det, d in rows[: args.top]:
        name, n = d["name"], d["n"]
        inst = next(i for i in FX.load_instances(only=[name]))
        xs = [float(c[0]) for c in inst["coords"]]
        ys = [float(c[1]) for c in inst["coords"]]
        T, _ = E.make_instance(inst["coords"], inst["ewt"], force_sparse=True)
        rng = random.Random(20260907 + n)
        eps = FX.DETIED_REL * FX.median_nn_dist(xs, ys)
        dxs = [v + rng.uniform(-eps, eps) for v in xs]
        dys = [v + rng.uniform(-eps, eps) for v in ys]
        angles = FX.angle_grid(n)[::3]          # 12 aci yeter
        costs_grid, costs_exact = [], []
        orig = E.grid_knn
        for a in angles:
            rx, ry = FX.rot(dxs, dys, a)
            costs_grid.append(T.tour_cost(E.greedy_edge_tour(rx, ry, k=8)))
            E.grid_knn = exact_knn
            try:
                costs_exact.append(T.tour_cost(E.greedy_edge_tour(rx, ry, k=8)))
            finally:
                E.grid_knn = orig
        rg = 100.0 * (max(costs_grid) - min(costs_grid)) / min(costs_grid)
        rex = 100.0 * (max(costs_exact) - min(costs_exact)) / min(costs_exact)
        res.append(dict(name=name, n=n, detied_range_fx=rng_det, detied_range_grid12=rg,
                        detied_range_exact12=rex, n_unique_exact=len(set(costs_exact))))
        print(f"{name:10s} n={n:5d} de-tied aralik: fx7 {rng_det:6.3f}%  grid_knn(12 aci) {rg:6.3f}%  "
              f"KESIN kNN(12 aci) {rex:6.3f}%  benzersiz maliyet {len(set(costs_exact))}/{len(angles)}")
    json.dump(res, open(os.path.join(args.out, "report", "exact_knn_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

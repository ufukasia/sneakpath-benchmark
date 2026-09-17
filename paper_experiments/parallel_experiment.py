# -*- coding: utf-8 -*-
"""PARALEL GE DENEYI -- k drone icin bolumleme x k x cerceve.

Soru: k arac bir ornegi paylasirken hangi bolumleme (band / kmeans / split)
ve hangi k, en kisa MAKESPAN'i (en uzun tur) verir; bant bolumlemesi
cerceveye ne kadar bagimli; paralel kurma ne kadar hizlandirir?

Her ornek icin:
  * global GE (k=1) referansi: L, t
  * k in K_SWEEP, strateji in {band(theta*), band(theta=0), band(theta=22.5
    hizasiz), kmeans, split}: total, makespan, imbalance, t_par, t_seq
  * bant bolumlemesinin aci taramasi: k=4, theta in [-90,90) adim 15 ->
    makespan(theta) egrisi (cerceve bagimliligi)

Kosum:
    python paper_experiments/parallel_experiment.py --workers 2
Cikti: out_parallel/per_instance/<ad>.json ; rapor: make_report.py (T7/F7).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p_ in (ROOT, HERE):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import tsplib_engine as E          # noqa: E402
import parallel_ge as PG           # noqa: E402
import frame_experiment as FX      # noqa: E402

VERSION = "px-1"
OUT_DEFAULT = os.path.join(HERE, "out_parallel")
K_SWEEP = (2, 3, 4, 6, 8, 12, 16)
BAND_ANGLES = [float(a) for a in range(-90, 90, 15)]
MISALIGN = 22.5


def run_instance(job):
    inst = job["inst"]
    coords, ewt, bks = inst["coords"], inst["ewt"], inst["bks"]
    n = len(coords)
    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    t_start = time.perf_counter()
    res = dict(version=VERSION, name=inst["name"], cls=inst["cls"], n=n, ewt=ewt, bks=bks)
    ecost = PG.edge_cost_fn(ewt)
    t0 = time.perf_counter()
    g = E.greedy_edge_tour(xs, ys, k=PG.KNN)
    t_g = time.perf_counter() - t0
    L_g = PG.tour_cost(g, xs, ys, ecost)
    res["ge"] = dict(cost=L_g, time=t_g)
    # hizali aci: comb/nndir dedektoru (aile vekili degil: bagimsiz)
    th_c, conf = FX.det_comb(xs, ys)
    th_n, _ = FX.det_nndir(xs, ys)
    theta_star = th_c if conf >= 0.6 else th_n
    res["theta_star"] = dict(theta=theta_star, comb=th_c, comb_conf=conf, nndir=th_n)
    strategies = [("band_star", "band", theta_star), ("band_zero", "band", 0.0),
                  ("band_mis", "band", MISALIGN), ("kmeans", "kmeans", 0.0), ("split", "split", 0.0)]
    cells = []
    for k in K_SWEEP:
        if n // k < 8:
            continue
        for label, s, th in strategies:
            r = PG.evaluate(s, xs, ys, k, ewt, theta=th, global_tour=g, t_global=t_g)
            r.pop("tours", None)
            r["label"] = label
            cells.append(r)
    res["cells"] = cells
    # bant bolumlemesinin aci egrisi (k=4)
    curve = {}
    for a in BAND_ANGLES:
        r = PG.evaluate("band", xs, ys, 4, ewt, theta=a)
        curve[f"{a:g}"] = dict(total=r["total"], makespan=r["makespan"], imbalance=r["imbalance"])
    res["band_angle_curve_k4"] = curve
    res["elapsed"] = time.perf_counter() - t_start
    return res


def _worker(job):
    inst = job["inst"]
    path = os.path.join(job["out"], "per_instance", f"{inst['name']}.json")
    t0 = time.perf_counter()
    try:
        res = run_instance(job)
    except Exception:
        res = dict(version=VERSION, name=inst["name"], cls=inst["cls"], n=len(inst["coords"]),
                   error=traceback.format_exc())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    return inst["name"], len(inst["coords"]), time.perf_counter() - t0, "error" in res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    insts = FX.load_instances(only=only, max_n=args.max_n)
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
        jobs.append(dict(inst=o, out=args.out))
    log(f"{len(insts)} ornek, {len(jobs)} kosulacak, {args.workers} isci, version={VERSION}")
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
    log(f"bitti: {len(jobs)} ornek, {(time.perf_counter() - t0) / 60:.1f} dk")


if __name__ == "__main__":
    main()

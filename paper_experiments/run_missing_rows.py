# -*- coding: utf-8 -*-
"""Panel kosumunda EKSIK kalan satirlari VLSI kumelerinde tamamlar (makale v3).

Her kume icin runner.py'yi asagidaki GECISLERLE cagirir (devam modu: diskte
olan satir yeniden kosulmaz; P1 --force ile yeniden kosulur cunku pge_ksweep
eski bant acisiyla uretilmisti):

  P1  pge_kd_k{2,4,8} + pge_ksweep (yeniden) + fs_ge8_exactknn         --force
  P2  realign_* @manual:7.5            P3  realign_* @manual:37.5
  P4  pge_band_k* @zero                P5  pge_band_k* @rotation_strip
  P6  pge_band_k* @manual:22.5
  P7  repair_vnd (<- greedy_edge)      P8  repair_vnd <- strip
  P9  repair_vnd <- rsge_fixed2        P10 repair_vnd <- realign_strip
  P11 ils (<- repair_vnd, panel)  P12 ils <- rsge_fixed2  P13 ils <- greedy_edge   [n <= ILS_MAX_N]

Kosum:  python paper_experiments/run_missing_rows.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import vlsi_datasets as V  # noqa: E402

MAX_N = 120000
ILS_MAX_N = 1000
REALIGN = "realign_ge8,realign_strip,realign_hilbert,realign_morton"
PBAND = "pge_band_k2,pge_band_k4,pge_band_k8"


def passes(n):
    P = [
        (["--methods", "pge_kd_k2,pge_kd_k4,pge_kd_k8,pge_ksweep,fs_ge8_exactknn", "--force"], "P1"),
        (["--methods", REALIGN, "--choices", ",".join(f"{k}=manual:7.5" for k in REALIGN.split(","))], "P2"),
        (["--methods", REALIGN, "--choices", ",".join(f"{k}=manual:37.5" for k in REALIGN.split(","))], "P3"),
        (["--methods", PBAND, "--choices", ",".join(f"{k}=zero" for k in PBAND.split(","))], "P4"),
        (["--methods", PBAND + ",rotation_strip", "--choices", ",".join(f"{k}=rotation_strip" for k in PBAND.split(","))], "P5"),
        (["--methods", PBAND, "--choices", ",".join(f"{k}=manual:22.5" for k in PBAND.split(","))], "P6"),
        (["--methods", "repair_vnd,greedy_edge"], "P7"),
        (["--methods", "repair_vnd,strip", "--choices", "repair_vnd=strip"], "P8"),
        (["--methods", "repair_vnd,rsge_fixed2", "--choices", "repair_vnd=rsge_fixed2"], "P9"),
        (["--methods", "repair_vnd,realign_strip", "--choices", "repair_vnd=realign_strip"], "P10"),
    ]
    if n <= ILS_MAX_N:
        P += [
            (["--methods", "ils,repair_vnd,greedy_edge"], "P11"),
            (["--methods", "ils,rsge_fixed2", "--choices", "ils=rsge_fixed2"], "P12"),
            (["--methods", "ils,greedy_edge", "--choices", "ils=greedy_edge"], "P13"),
        ]
    return P


def run_one(name, n, log):
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for args, tag in passes(n):
        cmd = [sys.executable, os.path.join(ROOT, "runner.py"), "--dataset", name] + args
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log(f"  [{name}] {tag} HATA rc={r.returncode}: {(r.stderr or r.stdout)[-300:]}")
    return name, n, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-n", type=int, default=MAX_N)
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    sets = sorted(((v[0], k) for k, v in V.CATALOG.items() if v[0] <= args.max_n
                   and os.path.exists(os.path.join(ROOT, "data_vlsi", f"{k}.tsp"))
                   and (not only or k in only)))
    logp = os.path.join(HERE, "run_missing_rows.log")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        open(logp, "a", encoding="utf-8").write(line + "\n")

    log(f"{len(sets)} VLSI kumesi, {args.workers} isci")
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, nm, n, log) for n, nm in sets]
        for f in as_completed(futs):
            nm, n, dt = f.result()
            done += 1
            log(f"{done}/{len(sets)} {nm} n={n} {dt:.0f}s")
    log("bitti")


if __name__ == "__main__":
    main()

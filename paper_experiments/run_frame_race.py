# -*- coding: utf-8 -*-
"""CERCEVE YARISI kosumu (makale v3): hizali (kafes acisi) <-> seyrek tarama (rotation_strip).

Gecisler (devam modu; diskte olan satir yeniden kosulmaz):
  R1  pgr_band_k{2,4,8} (kafes acisi) + pgr_kmeans_k{2,4,8}
  R2  pgr_band_k* @rotation_strip       R3  pgr_band_k* @zero      R4  pgr_band_k* @manual:22.5
  R5  repair_vnd <- strip               R6  repair_vnd <- rotation_strip   (BIRLESIK onarim; tekil komsuluk yok)
  R7  ils <- repair_vnd <- strip        R8  ils <- repair_vnd <- rotation_strip   [n <= ILS_MAX_N]
Her gecis icin duvar saati siniri PASS_TIMEOUT_S; asan gecis ZAMAN ASIMI olarak loglanir ve atlanir.

Kosum:  python paper_experiments/run_frame_race.py --workers 4
"""
from __future__ import annotations

import argparse
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
ILS_MAX_N = 3000
PASS_TIMEOUT_S = 1800  # gecis basina ust sinir (butceli onarimlar 100k dugumde ~90 s surer)
PGR = "pgr_band_k2,pgr_band_k4,pgr_band_k8"


def passes(n):
    P = [
        (["--methods", PGR + ",pgr_kmeans_k2,pgr_kmeans_k4,pgr_kmeans_k8"], "R1"),
        (["--methods", PGR + ",rotation_strip", "--choices", ",".join(f"{k}=rotation_strip" for k in PGR.split(","))], "R2"),
        (["--methods", PGR, "--choices", ",".join(f"{k}=zero" for k in PGR.split(","))], "R3"),
        (["--methods", PGR, "--choices", ",".join(f"{k}=manual:22.5" for k in PGR.split(","))], "R4"),
        # birlesik onarim (VND) -- tekil komsuluk (2-opt vb.) YOK (kullanici sozlesmesi)
        (["--methods", "repair_vnd,strip", "--choices", "repair_vnd=strip"], "R5"),
        (["--methods", "repair_vnd,rotation_strip", "--choices", "repair_vnd=rotation_strip"], "R6"),
    ]
    if n <= ILS_MAX_N:
        # ILS onarimli turun USTUNE: ils <- repair_vnd <- {strip | rotation_strip}
        # (ayni adimda iki gecersiz kilma; satir kimligi ils@repair_vnd@<tohum>)
        P += [
            (["--methods", "ils,repair_vnd,strip", "--choices", "repair_vnd=strip,ils=repair_vnd"], "R7"),
            (["--methods", "ils,repair_vnd,rotation_strip", "--choices", "repair_vnd=rotation_strip,ils=repair_vnd"], "R8"),
        ]
    return P


def run_one(name, n, log):
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for args, tag in passes(n):
        cmd = [sys.executable, os.path.join(ROOT, "runner.py"), "--dataset", name] + args
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=PASS_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log(f"  [{name}] {tag} ZAMAN ASIMI (> {PASS_TIMEOUT_S}s) -- atlandi")
            continue
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
    logp = os.path.join(HERE, "run_frame_race.log")

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

# -*- coding: utf-8 -*-
"""HAKEM RAPORU eksik kosumlari (2026-09-09; bkz. HAKEM_RAPORU.md, E-tablosu).

Gecisler (devam modu; diskte olan satir yeniden kosulmaz, H3 haric):
  H1  pgr_kd_k{2,4,8}                         E2 -- onarim sonrasi yarisa k-d
  H2  fs_{ge15,nn,nn_grid,fi}_detied          E9 -- "degismez" etiketinin kontrolu (fi: n<=3000)
  H3  fs_band_ge8 --force                     E6 -- b=32 ve b=k_s=round(sqrt(n/2)) eklendi
  H4  realign_strip_noise (@random)           E8 -- rastgele phi + konum gurultusu + nokta silme
  H5  pge_*_k{3,6,12,16} + pge_kmseed         E7 -- hiz her k'da; E10 -- k-means tohum duyarliligi
  H6  fs_fi  (yalniz dbj2924, xva2993)        B3 -- iki eksik aci taramasi
  H7  fs_ge8 + fs_ge8_jitter (yalniz sra104815, insa butcesi 1500 s)   B2
Her gecis icin duvar saati siniri PASS_TIMEOUT_S; asan gecis ZAMAN ASIMI olarak loglanir ve atlanir.

Kosum:  python paper_experiments/run_hakem_fixes.py --workers 4
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
PASS_TIMEOUT_S = 1800
PGE_EXTRA = ",".join(f"pge_{s}_k{k}" for s in ("band", "kd", "kmeans", "split") for k in (3, 6, 12, 16))
FI_MISSING = {"dbj2924", "xva2993"}
GE8_MISSING = {"sra104815"}


def passes(name, n):
    P = [
        (["--methods", "pgr_kd_k2,pgr_kd_k4,pgr_kd_k8"], "H1"),
        (["--methods", "fs_ge15_detied,fs_nn_detied,fs_nn_grid_detied,fs_fi_detied"], "H2"),
        (["--methods", "fs_band_ge8", "--force"], "H3"),
        (["--methods", "realign_strip_noise"], "H4"),
        (["--methods", PGE_EXTRA + ",pge_kmseed"], "H5"),
    ]
    if name in FI_MISSING:
        P.append((["--methods", "fs_fi"], "H6"))
    if name in GE8_MISSING:
        P.append((["--methods", "fs_ge8,fs_ge8_jitter", "--construction-max-s", "1500"], "H7"))
    return P


def run_one(name, n, log, timeout):
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for args, tag in passes(name, n):
        cmd = [sys.executable, os.path.join(ROOT, "runner.py"), "--dataset", name] + args
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            log(f"  [{name}] {tag} ZAMAN ASIMI (> {timeout}s) -- atlandi")
            continue
        if r.returncode != 0:
            log(f"  [{name}] {tag} HATA rc={r.returncode}: {(r.stderr or r.stdout)[-300:]}")
    return name, n, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-n", type=int, default=MAX_N)
    ap.add_argument("--timeout", type=int, default=PASS_TIMEOUT_S)
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    sets = sorted(((v[0], k) for k, v in V.CATALOG.items() if v[0] <= args.max_n
                   and os.path.exists(os.path.join(ROOT, "data_vlsi", f"{k}.tsp"))
                   and (not only or k in only)))
    logp = os.path.join(HERE, "run_hakem_fixes.log")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        open(logp, "a", encoding="utf-8").write(line + "\n")

    log(f"{len(sets)} VLSI kumesi, {args.workers} isci")
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, nm, n, log, args.timeout) for n, nm in sets]
        for f in as_completed(futs):
            nm, n, dt = f.result()
            done += 1
            log(f"{done}/{len(sets)} {nm} n={n} {dt:.0f}s")
    log("bitti")


if __name__ == "__main__":
    main()

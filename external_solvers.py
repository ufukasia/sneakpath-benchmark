# -*- coding: utf-8 -*-
"""Optional bridges to external "gold standard" TSP solvers: Concorde (exact
branch-and-cut) and LKH-3 (near-optimal Lin-Kernighan-Helsgaun; effectively
optimal on TSPLIB-sized Euclidean instances in practice).

Both read the SAME `data/*.tsp` TSPLIB file that `tsplib_engine.parse_tsp`
reads for every other method in this project, so the comparison starts from
identical coordinates and EDGE_WEIGHT_TYPE. The resulting tour is then scored
by the CALLER using `inst.tour_cost` (TSPLIB-exact EUC_2D nint / GEO), the
same cost function every other method's result is measured with -- so a
solver's reported cost and this project's reported cost cannot silently
diverge.

LKH-3 ships with this repo: `bin/LKH.exe` is a Windows build of LKH-3.0.6
(cross-compiled from Keld Helsgaun's official source with mingw-w64; the only
source change was making GetTime.c's getrusage() call POSIX-only, since it
doesn't exist on Windows -- LKH already had a clock()-based fallback for that
case). `find_lkh` picks it up automatically with no PATH/env var setup;
LKH_PATH or a `LKH`/`LKH.exe` on PATH still take priority if present.

Concorde ALSO ships with this repo: `bin/concorde` is the official Linux
build from the pyconcorde project's CI (github.com/jvkersch/pyconcorde,
`external/pyconcorde-build`, itself built from Applegate/Bixby/Chvatal/Cook's
official source + the QSopt LP backend -- see that project's
`build-concorde-linux.sh` for the exact upstream URLs). It is an ELF binary,
so on Windows it is run inside a `debian:bookworm-slim` Docker container
(`docker` must be on PATH; Docker Desktop's default Windows setup is
sufficient, no manual image pull needed -- it's pulled on first use). A
native `bin/concorde.exe` (or one on PATH / CONCORDE_PATH) always takes
priority over the Dockerized Linux binary if present, since it avoids the
container-startup overhead. Verified against kroA100 (EXACT optimum 21282 in
0.09s) and pr2392 during setup.

Install (only needed if you want to replace either bundled solver):
  LKH-3:    http://webhotel4.ruc.dk/~keld/research/LKH-3/  -> put LKH(.exe)
            on PATH, set LKH_PATH, or drop it at bin/LKH.exe.
  Concorde: https://www.math.uwaterloo.ca/tsp/concorde.html -> put
            concorde(.exe) on PATH, set CONCORDE_PATH, or drop it at
            bin/concorde.exe. Concorde needs a working QSopt/CPLEX LP
            backend to build; prebuilt Windows/Linux binaries bundle QSopt.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


_BIN_DIR = Path(__file__).resolve().parent / "bin"
_BUNDLED_LKH = _BIN_DIR / "LKH.exe"
_BUNDLED_CONCORDE = _BIN_DIR / ("concorde.exe" if platform.system() == "Windows" else "concorde")
_BUNDLED_CONCORDE_LINUX = _BIN_DIR / "concorde"  # ELF; run via Docker on non-Linux hosts
_CONCORDE_DOCKER_IMAGE = "debian:bookworm-slim"


def find_lkh() -> str | None:
    p = os.environ.get("LKH_PATH")
    if p and Path(p).exists():
        return p
    for name in ("LKH", "LKH.exe", "LKH3", "LKH3.exe"):
        found = shutil.which(name)
        if found:
            return found
    if _BUNDLED_LKH.exists():
        return str(_BUNDLED_LKH)
    return None


def find_concorde() -> str | None:
    """Returns a path `run_concorde` can execute. On Windows this may be the
    bundled Linux ELF binary (`bin/concorde`, no execute permission needed
    natively) -- `run_concorde` detects that case and routes through Docker."""
    p = os.environ.get("CONCORDE_PATH")
    if p and Path(p).exists():
        return p
    for name in ("concorde", "concorde.exe"):
        found = shutil.which(name)
        if found:
            return found
    if _BUNDLED_CONCORDE.exists():  # native-platform binary (e.g. a real concorde.exe)
        return str(_BUNDLED_CONCORDE)
    if _BUNDLED_CONCORDE_LINUX.exists() and shutil.which("docker"):
        return str(_BUNDLED_CONCORDE_LINUX)
    return None


def _parse_tsplib_tour_file(path: Path, n: int) -> list[int] | None:
    """Parses a TSPLIB TOUR_SECTION (LKH's OUTPUT_TOUR_FILE format) into a
    0-indexed tour. Returns None if the file doesn't yield exactly n cities."""
    tour: list[int] = []
    in_section = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("TOUR_SECTION"):
            in_section = True
            continue
        if not in_section:
            continue
        if up.startswith("EOF"):
            break
        stop = False
        for tok in s.split():
            v = int(tok)
            if v == -1:
                stop = True
                break
            tour.append(v - 1)  # TSPLIB tours are 1-indexed
        if stop:
            break
    if len(tour) != n or sorted(tour) != list(range(n)):
        return None
    return tour


def lkh_hard_cap(time_limit: float) -> float:
    """Dis emniyet kapagi: LKH kendi TIME_LIMIT'ini denemeler ARASINDA
    kontrol ettigi icin on-isleme sirasinda hic bakmaz; bu yuzden disaridan
    verdigimiz oldurme suresi TIME_LIMIT'ten belirgin sekilde buyuk olmali.

    TEK KAYNAK: sure asimi notunu geriye donuk okurken (runner'daki
    `migrate_solver_notes`) bu formulun TERSI aliniyor -- iki yerde ayri ayri
    yazilirsa biri degistiginde eski notlarin butcesi yanlis cozulur."""
    return max(time_limit * 1.5, time_limit + 300.0)


def lkh_time_limit_from_hard_cap(hard_cap: float) -> float | None:
    """`lkh_hard_cap`in tersi: kapak degerinden yapilandirilmis TIME_LIMIT.

    Eski notlarda metne KAPAK yazilmis (ornegin 600 s), oysa kullanicinin
    `--exact-time` ile ayarladigi deger TIME_LIMIT'tir (300 s). Butce
    hafizasi yanlis sayiyla karsilastirirsa kullanici butceyi 500'e
    cikardiginda satir hala engelli kalirdi. Iki adayi da deneyip kapagi
    yeniden ureteni seciyoruz; hicbiri tutmuyorsa None (not oldugu gibi
    birakilir, yanlis pozitif yok)."""
    for cand in (hard_cap - 300.0, hard_cap / 1.5):
        if cand > 0 and abs(lkh_hard_cap(cand) - hard_cap) < 1e-6:
            return cand
    return None


def run_lkh(tsp_path: str | Path, n: int, time_limit: float = 60.0,
            seed: int = 1, runs: int = 1,
            lkh_path: str | None = None) -> tuple[list[int] | None, str]:
    """Runs LKH-3 on the SAME TSPLIB .tsp file the rest of the pipeline reads.

    `time_limit` maps to LKH's own TIME_LIMIT parameter (seconds): LKH checks
    the clock between trials and, once it fires, stops and writes whatever
    tour it currently holds -- so a time-limited run is a normal, EXPECTED
    outcome that should still produce a (near-optimal, not-yet-optimal) tour,
    not an empty result.

    Two things make that promise fragile in practice, both handled here:
      1. LKH only checks the clock BETWEEN trials/runs, and for large n (e.g.
         usa13509) candidate-set preprocessing (alpha-nearness ascent) alone
         can take well over a minute before the first trial even starts -- a
         flat "+60s" outer safety margin can fire mid-preprocessing, killing
         the process before it ever reaches a point where it would write
         OUTPUT_TOUR_FILE. The margin below scales with time_limit AND has a
         large fixed floor so big instances get real preprocessing headroom.
      2. Even a hard kill isn't necessarily a total loss: OUTPUT_TOUR_FILE is
         written as soon as an internal run/trial completes, which can race
         with our kill signal. So this ALWAYS checks for -- and uses -- a
         valid tour file before reporting a timeout, whether LKH exited on
         its own or had to be killed."""
    exe = lkh_path or find_lkh()
    if not exe:
        return None, "LKH-3 binary not found (set LKH_PATH or put LKH on PATH)"
    tsp_path = Path(tsp_path).resolve()
    with tempfile.TemporaryDirectory(prefix="lkh_") as td:
        # LKH's fopen() on the .par-declared PROBLEM_FILE uses the ANSI/locale
        # codepage, not UTF-8, so a path with non-ASCII characters (accented
        # letters in a OneDrive/user-profile path, common on this project)
        # fails with "Cannot open PROBLEM_FILE" even though the file exists.
        # Sidestep it entirely by working from a copy in the (ASCII) temp dir.
        problem_copy = Path(td) / "problem.tsp"
        problem_copy.write_bytes(tsp_path.read_bytes())
        tour_out = Path(td) / "out.tour"
        par_path = Path(td) / "run.par"
        par_path.write_text(
            f"PROBLEM_FILE = {problem_copy}\n"
            f"OUTPUT_TOUR_FILE = {tour_out}\n"
            f"RUNS = {runs}\n"
            f"SEED = {seed}\n"
            f"TIME_LIMIT = {time_limit}\n"
            f"TRACE_LEVEL = 0\n",
            encoding="utf-8")
        hard_cap = lkh_hard_cap(time_limit)
        t0 = time.perf_counter()
        proc = subprocess.Popen([exe, str(par_path)], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        hard_killed = False
        try:
            _, stderr = proc.communicate(timeout=hard_cap)
        except subprocess.TimeoutExpired:
            hard_killed = True
            proc.kill()
            _, stderr = proc.communicate()
        dt = time.perf_counter() - t0

        tour = _parse_tsplib_tour_file(tour_out, n) if tour_out.exists() else None
        if tour is not None:
            note = " [outer hard-kill raced with LKH's own output write]" if hard_killed else ""
            return tour, f"LKH-3 ok in {dt:.1f}s (runs={runs}, time_limit={time_limit}s){note}"
        if hard_killed:
            return None, (f"LKH-3 hard-killed after {hard_cap:.0f}s without writing a tour "
                           f"-- preprocessing (candidate-set construction) for n={n} likely "
                           f"still in progress; raise the exact-solver time budget")
        return None, (f"LKH-3 produced no tour file (rc={proc.returncode}): "
                       f"{(stderr or '')[-300:]}")


def run_concorde(tsp_path: str | Path, n: int, time_limit: float = 300.0,
                  seed: int = 1,
                  concorde_path: str | None = None) -> tuple[list[int] | None, str]:
    """Runs Concorde (exact branch-and-cut) on the SAME .tsp file.

    Concorde has no built-in wall-clock cutoff for exact solving, so this
    wraps the process with a hard subprocess timeout. On timeout it returns
    None rather than a partial tour: Concorde's exact-optimality guarantee
    only holds if the solve actually terminates, and a killed process's
    working files should not be reinterpreted as a final answer."""
    exe = concorde_path or find_concorde()
    if not exe:
        return None, "Concorde binary not found (set CONCORDE_PATH or put concorde on PATH)"
    use_docker = (platform.system() != "Linux" and Path(exe) == _BUNDLED_CONCORDE_LINUX)
    tsp_path = Path(tsp_path).resolve()
    with tempfile.TemporaryDirectory(prefix="concorde_") as td:
        tdp = Path(td)
        # same non-ASCII-path workaround as run_lkh (see its comment); also
        # required here so the Docker bind-mount below is a plain ASCII path.
        problem_copy = tdp / "problem.tsp"
        problem_copy.write_bytes(tsp_path.read_bytes())
        t0 = time.perf_counter()
        try:
            if use_docker:
                bin_copy = tdp / "concorde_bin"
                bin_copy.write_bytes(Path(exe).read_bytes())
                bin_copy.chmod(0o755)
                docker = shutil.which("docker")
                cmd = [docker, "run", "--rm", "-v", f"{tdp}:/work", "-w", "/work",
                       _CONCORDE_DOCKER_IMAGE, "./concorde_bin", "-s", str(seed),
                       "-o", "out.sol", "problem.tsp"]
            else:
                cmd = [exe, "-s", str(seed), "-o", "out.sol", str(problem_copy)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=td,
                                   timeout=time_limit)
        except subprocess.TimeoutExpired:
            return None, (f"Concorde did not finish within {time_limit:.0f}s "
                           f"(exact solve too slow at this instance size)")
        dt = time.perf_counter() - t0
        sol_path = tdp / "out.sol"
        if not sol_path.exists():
            return None, (f"Concorde produced no solution file (rc={proc.returncode}): "
                           f"{proc.stderr[-300:]}")
        toks = sol_path.read_text(encoding="utf-8", errors="ignore").split()
        try:
            declared_n = int(toks[0])
            tour = [int(v) for v in toks[1:1 + declared_n]]
        except (ValueError, IndexError):
            return None, "Concorde .sol file did not parse"
        if len(tour) != n or sorted(tour) != list(range(n)):
            return None, "Concorde tour is not a valid n-city permutation"
        via = " via Docker" if use_docker else ""
        return tour, f"Concorde ok (EXACT/OPTIMAL) in {dt:.1f}s{via}"

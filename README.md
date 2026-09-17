# Banded Parallel Greedy-Edge

**Reproduction package for a study on frame selection and spatial partitioning in
multi-vehicle routing over grid-structured fields.**

The study asks a narrow question with a practical consequence. When a fleet of *k*
vehicles has to cover a field whose points sit on an integer lattice — a VLSI board,
an orchard, a solar farm — the objective is no longer the length of one tour but the
**makespan**, the longest subtour, which decides when the mission ends. Two things
could plausibly decide that makespan: the *angle of the coordinate frame* the tours
are built in, or the *way the field is cut into k parts*. This package contains the
code, the data and the measurements that separate the two.

The answer, in one line: **the frame is not a free parameter to optimise for the
constructor — it is a design decision that belongs to the partitioning.**

---

## Live figure

The paper's Figure F0 runs live on Streamlit Community Cloud — pick an instance,
change *k*, switch the partitioning strategy, and watch the three panels redraw:

> **[→ open the live figure](https://share.streamlit.io/)** *(link added after the
> first deploy; see [Deploying](#deploying-your-own-instance))*

It is a **single read-only page**: no runs, no administration, nothing to break.
The tables, statistics and the benchmark runner live in the full dashboard you
start locally (below).

---

## What the measurements show

All figures below are recomputed from the result files shipped in `results/`
(111 instances: 102 Waterloo VLSI + 9 TSPLIB, *n* from 100 to 744 710, median 3 649).
They are produced by the same code paths the dashboard reports, so a reader can
reproduce every number in this table from this repository alone.

### 1. Constructors at θ = 0°, mean optimality gap to the best known solution

| Constructor | N | mean gap % | median gap % | class |
|---|---:|---:|---:|---|
| Farthest Insertion | 83 | 14.54 | 14.78 | rotation-invariant, O(n²) |
| **Greedy-Edge (k = 8 candidates)** | 109 | **17.17** | 16.17 | rotation-invariant |
| Nearest Neighbour (exact) | 110 | 26.80 | 25.30 | rotation-invariant |
| Nearest Neighbour (grid-accelerated) | 111 | 36.13 | 29.22 | rotation-invariant |
| θ★ sparse strip scan | 111 | 46.23 | 42.72 | frame-dependent |
| Strip / boustrophedon | 111 | 72.18 | 54.87 | frame-dependent |
| Hilbert curve | 111 | 65.63 | 51.84 | frame-dependent |
| Morton / Z-order curve | 111 | 137.19 | 111.08 | frame-dependent |

Greedy-Edge is the strongest constructor that is also **rotation-invariant**: rotating
the instance leaves its tour unchanged up to floating-point tie-breaking. The
space-filling family is not — the same instance rotated gives a materially different
tour. That is the asymmetry the rest of the study is built on.

### 2. Improvement layers, same instances

| Layer | N | mean gap % | median gap % |
|---|---:|---:|---:|
| VND (2-opt + Or-opt + relocate) | 100 | 5.54 | 5.52 |
| ILS (seeded on VND) | 15 | 1.56 | 1.32 |
| LKH-3 (gold standard reference) | 80 | 0.02 | 0.00 |

LKH-3 and Concorde are **reference baselines, not competitors**; the dashboard
excludes them from rank and win-rate pools for that reason.

### 3. Partitioning strategy decides the makespan, not the frame angle

Balance = makespan / mean subtour length (1.00 = perfectly balanced).
Computed over the 99 VLSI instances that carry all four base rows, *k* = 8 drones:

| Partitioning strategy | mean balance | median balance |
|---|---:|---:|
| **2-D k-d median cut** | **1.134** | **1.119** |
| 1-D equal-cardinality band | 1.152 | 1.129 |
| k-means clusters | 1.224 | 1.187 |
| global tour split | 1.330 | 1.258 |

The balanced 2-D k-d median cut is ahead of every alternative. The 1-D band cut —
the only strategy whose result depends on the frame angle — is equivalent only at
small *k* and falls behind as *k* grows.

> **Note on the figures.** The explanatory text embedded in `dashboard/index.html`
> quotes balance values of 1.19 / 1.39 / 1.27 / 1.48 for these four strategies. The
> table above is recomputed from the result files in this repository and does not
> reproduce those values, though the ordering is unchanged. The definition used here
> is stated explicitly above; the panel prose does not state which statistic or
> subset it summarises. Prefer the numbers you can recompute.

---

## Quick start

Requires Python 3.9–3.13 and about 25 MB of disk. No compilation step.

```bash
git clone https://github.com/ufukasia/sneakpath-benchmark.git
cd sneakpath-benchmark
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
```

Then pick an interface.

**Streamlit figure** — one page, read-only, the live Figure F0:

```bash
.venv/bin/streamlit run streamlit_app/app.py
```

**Full dashboard** — four tabs (visualization, benchmark tables, statistics,
administration) and the only interface that can *launch* runs:

```bash
.venv/bin/python dashboard/server.py
# http://localhost:7100
```

**Headless run** — reproduce one instance from scratch:

```bash
.venv/bin/python -m benchmark.run --sets kroA100
```

Both interfaces are bilingual (English / Turkish); the toggle is in the top bar of
the dashboard and in the sidebar of the Streamlit page.

> **Platform note.** Streamlit depends on `pyarrow`, which publishes no wheels for
> `win-arm64`. On an ARM64 Windows machine, create the environment from an x64
> interpreter (`py -3.11 -m venv .venv`); the dashboard and the headless runner have
> no such constraint and work on the native ARM64 build.

---

## Repository layout

```
core.py                 instance model, distance metrics, tour cost
runner.py               method catalogue + orchestration (METHOD_ORDER, PRETTY)
parallel_ge.py          k-way partitioning (band / k-d / k-means / tour split)
frame_methods.py        rotated-frame constructors, angle sweeps
rgge.py                 rotated-grid Greedy-Edge (RGGE)
repair.py               local search: 2-opt, Or-opt, relocate, VND
grid_theta.py           lattice angle detectors (comb scan, NN-direction, PCA)
tsplib_engine.py        TSPLIB parsing, EUC_2D / GEO, constructors
external_solvers.py     LKH-3 / Concorde adapters (binaries not shipped, see below)
snake_alt.py            strip / boustrophedon family
tsplib_datasets.py      TSPLIB catalogue and download URLs
vlsi_datasets.py        Waterloo VLSI catalogue and download URLs

benchmark/run.py        headless runner  (python -m benchmark.run --sets <name>)
benchmark/method_table.py  method → stage / role mapping

dashboard/              original single-file panel
  index.html              UI, charts, bilingual dictionary (source of truth)
  server.py               JSON API + job queue + admin endpoints

streamlit_app/          live Figure F0, one page
  app.py                  the whole page (no pages/ directory, so no nav list)
  sp/backend.py           imports dashboard/server.py directly (no HTTP hop)
  sp/charts.py            tour plot
  sp/meta.json            TR/EN dictionary, extracted from index.html
  extract_meta.js         regenerates sp/meta.json from index.html

data_tsplib/            94 TSPLIB instances
data_vlsi/              43 small Waterloo VLSI instances (n ≤ 3000)
data/                   9 instances used by the classic quick-look set
results/                111 result files, one per instance
paper_experiments/      scripts that produced the paper's figures and tables
```

### Two interfaces, one source of truth

The Streamlit page does **not** reimplement anything. `sp/backend.py` imports
`dashboard/server.py` as a module and calls `_compute_paradoks` directly —
`server.py` guards its entry point with `if __name__ == "__main__"`, so importing it
does not start a server. The tours you see are produced by the same code the
benchmark runs, not by a client-side approximation. The bilingual dictionary lives
in `dashboard/index.html` and is extracted into `streamlit_app/sp/meta.json` by

```bash
node streamlit_app/extract_meta.js
```

Run that after editing text in the panel; otherwise the two interfaces drift.

---

## Data

**TSPLIB** instances (Reinelt, Universität Heidelberg) and the **Waterloo VLSI**
collection are redistributed here under their original terms — both are published
for free academic use. Best-known solution values travel with the catalogues in
`tsplib_datasets.py` and `vlsi_datasets.py`.

Only VLSI instances with *n* ≤ 3000 are included, because those are the ones the
visualization page can draw interactively. The dashboard downloads the rest on
demand from the Waterloo mirror (Benchmark tab → *Download the entire collection*).

### About `results/`

The result files here are **slimmed**: per-method `tour` and `parts` arrays and
per-instance `coords` have been removed. Those three fields are 98.8 % of the raw
size (1.17 GB → 13.7 MB) and no view in either interface reads them — the tours
drawn on the Visualization page are recomputed live from the `.tsp` files. Everything
a reader needs is preserved verbatim, including the per-iteration `history` arrays
that drive the convergence plot. Re-running `benchmark.run` regenerates the full
files locally.

---

## External solvers

`bin/LKH.exe` and `bin/concorde` are **not distributed in this repository.** Both
are third-party solvers under academic-use licences that do not grant
redistribution rights. The adapters in `external_solvers.py` look for them and
simply skip those rows when they are absent — every other method runs normally.

To enable them, obtain the binaries from their authors and place them in `bin/`:

| Solver | Source | Licence |
|---|---|---|
| LKH-3 | `http://webhotel4.ruc.dk/~keld/research/LKH-3/` | free for academic and non-commercial use |
| Concorde | `https://www.math.uwaterloo.ca/tsp/concorde.html` | free for academic research use |

---

## Reproducing the paper's tables

Each table in the paper corresponds to a preset in the dashboard's Statistics tab
(*Paper tables*), which selects exactly the rows that table reports — including the
seed and angle clones it was built from. Start the dashboard with
`python dashboard/server.py` to use them.

| Preset | Content |
|---|---|
| T1 Constructors | θ = 0° baselines: rotation-invariant vs frame-dependent, plus reference solvers |
| T2 Angle sensitivity | [−90°, 90°) angle sweep per constructor; row = θ = 0 tour, spread in the fields |
| T3 Tie controls | rotation-invariance evidence: jitter / tie-free coordinates / exact k-NN |
| T5 / T6 Realignment | misaligned board → detector → lattice snap → constructor, with noise ablation |
| T7 Banding | band sweep b ∈ {1…32} and the RGGE / RSGE families |
| T8 Parallel partitioning | k-drone partitioning: band / k-d / k-means / tour split × k |
| S3 Repair seeds | final gap of VND and ILS from different construction seeds |
| T10 / T11 Frame race | aligned ↔ sparse: construction → VND → ILS, single tour and parallel |

To recompute rather than browse:

```bash
python -m benchmark.run --sets xqf131              # one instance, all visible methods
python -m benchmark.run --sets kroA100 rat783      # several
```

Results are written to `results/<instance>.json` and appear in both interfaces on
the next reload.

### Reading the frame race

The Streamlit page is the paper's Figure F0, live. Pick an instance, then switch
*Improvement* between **None**, **VND** and **ILS**. All three panels pass through
the same improvement layer, which is the point: at construction the sparse frame (b)
is usually ahead of the lattice-aligned frame (a), and after deep local search the
ordering commonly reverses. On `xqf131` with *k* = 4:

| | construction | after ILS |
|---|---:|---:|
| (a) lattice-aligned strip | 854 | 579 |
| (b) sparse strip scan (θ̂ₛ) | **764** | 581 |
| (c) k = 4 partition, makespan | 211 | **183** |

The frame that looks better at construction time is not the frame that wins after
optimisation — which is why choosing it as a constructor parameter is the wrong move.

---

## Administration

The dashboard's Admin tab changes method visibility, permanently deletes result
rows and starts benchmark subprocesses. **There is no authentication.** This is a
research tool you run on your own machine against your own result files, and a
password only got in the way.

Two things follow from that, and they are worth knowing:

- `dashboard/server.py` binds to `localhost` by default, so nothing outside your
  machine can reach it. If you pass `--host 0.0.0.0` to share it on a network,
  anyone who can reach the port can delete your results and start runs. Don't do
  that on an untrusted network.
- Deleting is irreversible. `results/*.json` is the only copy of a run; re-running
  `benchmark.run` recomputes it, which for the large instances takes a long time.

The published **Streamlit page has no administration at all** and never writes — it
reads instance files and draws tours. A public deployment therefore exposes no
destructive operation.

---

## Deploying your own instance

1. Fork or clone this repository into your own GitHub account.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `streamlit_app/app.py`.
3. No secrets and no configuration are needed. The page computes its tours from the
   `.tsp` files in the repository.

Instances above *n* = 3000 are excluded from the picker because they cannot be drawn
interactively; the full collection is available in the local dashboard.

---

## Citation

If you use this code or the measurements, please cite the accompanying paper. A
`CITATION.cff` file is included so GitHub's *Cite this repository* button produces a
correct entry; update it with the final bibliographic details once the paper appears.

---

## Licence

Code in this repository is released under the MIT Licence (see `LICENSE`).

The bundled instance files are redistributed under the terms of their original
collections (TSPLIB, Waterloo VLSI). LKH-3 and Concorde are not included; see
[External solvers](#external-solvers).

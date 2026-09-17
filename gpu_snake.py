# -*- coding: utf-8 -*-
"""GPU Parallel Multi-Start Snake + Repair ("gpu_snake").

The user's three CPU stages fused into ONE massively parallel GPU method:

  1. MULTI-START SNAKE (construction): instead of building a single
     snake-grid tour and only then improving it, this builds a whole BATCH of
     snake tours at once — every combination of rotation angle x strip-width
     factor x snake phase is one batch row, and all of them are constructed
     in a single batched argsort on the GPU. This is the "start from many
     points at once" replacement for the sequential snake+rotation scan.
  2. WINDOW REPAIR (parallel): a candidate-list 2-opt pass — for every tour
     in the batch simultaneously, the GPU evaluates the edge exchange between
     each position and its k nearest-neighbour cities' positions in one shot
     (a GEOMETRIC repair window, so long-range crossings between far-apart
     tour positions are found too). Improving, non-overlapping moves are
     applied between passes.
  3. ADAPTIVE LR (parallel): an or-opt relocation pass — segments of length
     1..3 are tested against the k nearest-neighbour insertion edges of
     their head city (the vectorised equivalent of line-reassignment's
     "move a point/short run to a better line"), again for every tour in
     the batch at once.

Passes 2 and 3 alternate until no tour improves, a round cap, or the time
budget. The final winner is picked by EXACT TSPLIB cost (inst.tour_cost in
the runner), so the reported number is always comparable with every other
method. Move deltas use unrounded metric distances (float32 EUC / float64
GEO great-circle), which is standard for local search; only the final
scoring is TSPLIB-exact.

Speed levers ("hız aşırtması"): more starts (`n_starts`), more 2-opt
partners (`k_2opt`), more insertion candidates (`k_or`) — all of them widen
the GPU batch instead of adding wall-clock, until VRAM is the limit. Runs
on CPU torch as a fallback (same code path, just slower).

No torch import is needed anywhere else: the runner imports this module
lazily and reports the method as unavailable if torch is missing.
"""
from __future__ import annotations

import math
import os
import time
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import torch

# duck-typed like app.HistoryPoint (runner's _downsample_history reads
# .elapsed_time / .cost) without importing the engine here
HistPt = namedtuple("HistPt", ["iteration", "elapsed_time", "cost"])

_RRR = 6378.388  # TSPLIB earth radius (km), same constant as gnn_trained


@dataclass
class GpuSnakeDiag:
    """Duck-typed like the lro result objects the runner's record() reads."""
    tour: list[int]
    cost: float                       # metric cost (exact cost set by caller)
    iterations_completed: int = 0
    improvements_accepted: int = 0
    total_candidates_evaluated: int = 0
    total_saving: float = 0.0
    history: list = field(default_factory=list)
    n_starts: int = 0
    device: str = "cpu"
    construction_cost: float = 0.0
    winner: str = "snake"             # origin of the best final tour
                                      # ("snake" or a seed_tours label)


def _pdist(A: torch.Tensor, B: torch.Tensor, is_geo: bool) -> torch.Tensor:
    """Metric distance between coordinate tensors (..., 2). EUC: unrounded
    Euclid. GEO: continuous TSPLIB great-circle (no +1/floor — monotone in
    the true rounded cost, which is all a move-delta needs)."""
    if not is_geo:
        d = A - B
        return torch.sqrt((d * d).sum(-1) + 1e-12)
    la1, lo1 = torch.deg2rad(A[..., 0]), torch.deg2rad(A[..., 1])
    la2, lo2 = torch.deg2rad(B[..., 0]), torch.deg2rad(B[..., 1])
    q1 = torch.cos(lo1 - lo2)
    q2 = torch.cos(la1 - la2)
    q3 = torch.cos(la1 + la2)
    val = torch.clamp(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    return _RRR * torch.arccos(val)


def _tour_lengths(C: torch.Tensor, tours: torch.Tensor, is_geo: bool) -> torch.Tensor:
    P = C[tours]                                   # (M, n, 2)
    return _pdist(P, torch.roll(P, -1, dims=1), is_geo).sum(dim=1)


# ---------------------------------------------------------------------------
#  1) batched multi-start snake construction
# ---------------------------------------------------------------------------
def build_multi_snake(C: torch.Tensor, n_starts: int,
                      hint_deg: float | None = None) -> torch.Tensor:
    """Builds ~n_starts snake tours in ONE batched argsort: variants are the
    cartesian product of rotation angles x strip-width factors x snake phase
    (which strip parity runs upward). Returns (M, n) city-index tours.

    `hint_deg`: theta* MIRASI (runner'in theta dedektorlerinden gelen en iyi
    aci, derece). Verilirse esit-aralikli aci havuzuna hint merkezli acilar
    (+-0.6 derece komsulariyla, iki isaret) EKLENIR — boylece dedektorlerin
    buldugu yonelim GPU havuzunda kesin olarak temsil edilir; esit-aralikli
    taban AYNEN kalir (kucultulmez — cesitlilik sigortasidir; hint satirlari
    batch'i buyutur, GPU'da marjinal maliyet)."""
    n = C.shape[0]
    factors = (0.7, 1.0, 1.45)
    phases = (0, 1)
    n_variants = len(factors) * len(phases)
    hint_list: list[float] = []
    if hint_deg is not None:
        # runner theta'si noktayi +theta dondurup x'i seritler; burada u =
        # cos(a)x + sin(a)y, yani cerceve donusu = nokta donusunun tersi ->
        # a = -theta. Isaret/mod belirsizligine karsi iki isaret de eklenir.
        for s in (-hint_deg, hint_deg):
            for off in (-0.6, 0.0, 0.6):
                hint_list.append(math.radians((s + off) % 180.0))
    n_angles = max(2, n_starts // n_variants)
    angles = torch.linspace(0.0, math.pi * (1 - 1.0 / n_angles), n_angles,
                            device=C.device, dtype=C.dtype)
    if hint_list:
        angles = torch.cat([angles, torch.tensor(hint_list, device=C.device,
                                                 dtype=C.dtype)])
    n_angles = angles.shape[0]

    variants = []
    for f in factors:
        for ph in phases:
            variants.append((f, ph))
    M = n_angles * len(variants)

    cos = torch.cos(angles)[:, None]               # (A, 1)
    sin = torch.sin(angles)[:, None]
    x, y = C[:, 0][None, :], C[:, 1][None, :]      # (1, n)
    u = cos * x + sin * y                          # (A, n) rotated abscissa
    v = -sin * x + cos * y

    keys = torch.empty((M, n), device=C.device, dtype=C.dtype)
    row = 0
    for a in range(n_angles):
        ua, va = u[a], v[a]
        umin, umax = ua.min(), ua.max()
        vmin, vmax = va.min(), va.max()
        vnorm = (va - vmin) / torch.clamp(vmax - vmin, min=1e-9)
        uspan = torch.clamp(umax - umin, min=1e-9)
        for f, ph in variants:
            strips = max(1, int(round(f * math.sqrt(n))))
            idx = torch.floor(torch.clamp((ua - umin) / uspan * strips,
                                          max=strips - 0.5))
            up = (idx.long() + ph) % 2 == 0
            keys[row] = idx * 1.5 + torch.where(up, vnorm, 1.0 - vnorm)
            row += 1
    return torch.argsort(keys, dim=1)


# ---------------------------------------------------------------------------
#  2) window repair: batched candidate-list 2-opt (kNN partners, so the
#     repair window is geometric, not positional -- long-range crossings
#     between far-apart tour positions are still found)
# ---------------------------------------------------------------------------
def _knn_2opt_moves(C, tours, is_geo, knn, eps=1e-4,
                    pen_rm=None, pen_knn=None, lam=0.0):
    """For every tour and position i: best 2-opt exchange whose partner edge
    starts at a kNN city of t[i]. Returns per-tour improving move lists as
    CPU arrays (m_idx, lo, hi, delta) for the greedy applier.

    Optional GLS augmentation (ngls_gpu): `pen_rm` (M, n) holds the penalty
    of the tour edge at each position (removed edges), `pen_knn`
    (M, n_cities, k) the penalty of each (city, j-th kNN) pair (the added
    edge (t[i], candidate) is by construction a kNN pair; the second added
    edge's penalty is approximated as 0). Deltas are then augmented with
    lam * (pen_added - pen_removed), i.e. moves are ranked/gated by the
    GLS-augmented objective."""
    M, n = tours.shape
    k = knn.shape[1]
    pos = torch.empty_like(tours)
    pos.scatter_(1, tours, torch.arange(n, device=tours.device).expand(M, n))
    P = C[tours]
    d_next = _pdist(P, torch.roll(P, -1, dims=1), is_geo)     # (M, n)

    c_pos = pos.gather(1, knn[tours].reshape(M, -1)).reshape(M, n, k)
    i_pos = torch.arange(n, device=tours.device)[None, :, None].expand(M, n, k)
    lo = torch.minimum(i_pos, c_pos)
    hi = torch.maximum(i_pos, c_pos)

    def city(idx):                                  # (M,n,k) pos -> city coords
        return C[tours.gather(1, idx.reshape(M, -1)).reshape(M, n, k)]

    a = city(lo)
    b = city((lo + 1) % n)
    c2 = city(hi)
    d2 = city((hi + 1) % n)
    delta = (_pdist(a, c2, is_geo) + _pdist(b, d2, is_geo)
             - d_next.gather(1, lo.reshape(M, -1)).reshape(M, n, k)
             - d_next.gather(1, hi.reshape(M, -1)).reshape(M, n, k))
    if lam > 0.0 and pen_rm is not None and pen_knn is not None:
        pen_add = pen_knn.gather(1, tours.unsqueeze(-1).expand(-1, -1, k))
        delta = delta + lam * (
            pen_add
            - pen_rm.gather(1, lo.reshape(M, -1)).reshape(M, n, k)
            - pen_rm.gather(1, hi.reshape(M, -1)).reshape(M, n, k))
    delta = torch.where(hi - lo < 2, torch.full_like(delta, float("inf")), delta)
    delta = torch.where((lo == 0) & (hi == n - 1),
                        torch.full_like(delta, float("inf")), delta)

    bd, bk = delta.min(dim=2)                       # best partner per (m, i)
    blo = lo.gather(2, bk.unsqueeze(2)).squeeze(2)
    bhi = hi.gather(2, bk.unsqueeze(2)).squeeze(2)
    imp = bd < -eps
    m_idx, _ = torch.nonzero(imp, as_tuple=True)
    return (m_idx.cpu().numpy(), blo[imp].cpu().numpy(),
            bhi[imp].cpu().numpy(), bd[imp].cpu().numpy())


def _apply_2opt(tours_np: np.ndarray, moves, eps=1e-4):
    """Greedy non-overlapping application of improving 2-opt moves per tour
    (boundary-edge-inclusive disjointness keeps every applied delta exact).
    Returns (moves_applied, metric_saving).

    Sort MUST be stable (2026-07-20). The sort spans the whole batch while the
    disjointness test `used[m]` is per tour, so a tour's outcome depends only
    on the order its OWN moves are visited. With an unstable sort, moves with
    exactly equal delta -- common in grid-like VLSI instances -- may be
    reordered depending on what else is in the batch, so splitting the pool
    across devices changes the tours. Measured on bnd7168: 28/32 tours diverged
    when the pool was split in two, with cost differences up to ~3%; with a
    stable sort the split is bit-identical (measurements_jpdc/results/
    shard_parity{,_stable}.json). Verified NOT to change single-device results
    on rat783/fnl4461/usa13509 (stable_sort_impact.json); costs ~1% more time."""
    M, n = tours_np.shape
    m_idx, lo, hi, deltas = moves
    applied = 0
    saving = 0.0
    if len(m_idx) == 0:
        return 0, 0.0
    order = np.argsort(deltas, kind="stable")
    used = np.zeros((M, n + 1), dtype=bool)
    for t in order:
        m, i, j, d = int(m_idx[t]), int(lo[t]), int(hi[t]), float(deltas[t])
        if d >= -eps or used[m, i:j + 2].any():
            continue
        tours_np[m, i + 1:j + 1] = tours_np[m, i + 1:j + 1][::-1]
        used[m, i:j + 2] = True
        applied += 1
        saving -= d
    return applied, saving


# ---------------------------------------------------------------------------
#  3) adaptive LR: batched or-opt segment relocation
# ---------------------------------------------------------------------------
def _oropt_moves(C, tours, is_geo, knn, seg_lens=(1, 2, 3),
                 pen_rm=None, pen_knn=None, lam=0.0):
    """For every tour and segment start: best relocation (segment of length
    L to a kNN insertion edge). Returns per-L (delta, cand_pos) CPU arrays.
    `pen_rm`/`pen_knn`/`lam`: optional GLS augmentation, see _knn_2opt_moves
    (removed edges: prev->s0, sE->nxt, u->w; added kNN pair: u->s0; the
    other added edges' penalties are approximated as 0)."""
    M, n = tours.shape
    pos = torch.empty_like(tours)
    pos.scatter_(1, tours, torch.arange(n, device=tours.device
                                        ).expand(M, n))
    out = []
    Ct = C[tours]                                   # (M, n, 2) coords by pos
    k = knn.shape[1]
    pen_add_all = (pen_knn.gather(1, tours.unsqueeze(-1).expand(-1, -1, k))
                   if (lam > 0.0 and pen_knn is not None) else None)
    for L in seg_lens:
        if n < L + 3:
            out.append(None)
            continue
        prev = torch.roll(Ct, 1, dims=1)            # coords of t[i-1]
        s0 = Ct                                     # t[i]
        sE = torch.roll(Ct, -(L - 1), dims=1)       # t[i+L-1]
        nxt = torch.roll(Ct, -L, dims=1)            # t[i+L]
        gain = (_pdist(prev, s0, is_geo) + _pdist(sE, nxt, is_geo)
                - _pdist(prev, nxt, is_geo))        # (M, n)

        head = tours                                # city id at pos i
        cand_city = knn[head]                       # (M, n, k)
        c_pos = pos.gather(1, cand_city.reshape(M, -1)).reshape(cand_city.shape)
        u = C[cand_city]                            # insertion edge left city
        w_city = torch.gather(tours, 1, ((c_pos + 1) % n).reshape(M, -1)
                              ).reshape(cand_city.shape)
        w = C[w_city]
        s0e = s0.unsqueeze(2)
        sEe = sE.unsqueeze(2)
        add = (_pdist(u, s0e, is_geo) + _pdist(sEe, w, is_geo)
               - _pdist(u, w, is_geo))              # (M, n, k)
        delta = add - gain.unsqueeze(2)
        if lam > 0.0 and pen_rm is not None and pen_add_all is not None:
            pen_gain = (torch.roll(pen_rm, 1, dims=1)          # edge i-1 -> i
                        + torch.roll(pen_rm, -(L - 1), dims=1))  # edge i+L-1 -> i+L
            pen_uw = pen_rm.gather(1, c_pos.reshape(M, -1)).reshape(c_pos.shape)
            delta = delta + lam * (pen_add_all - pen_uw - pen_gain.unsqueeze(2))

        i_pos = torch.arange(n, device=tours.device)[None, :, None]
        rel = (c_pos - i_pos + 1) % n               # c in {i-1..i+L-1} invalid
        delta = torch.where(rel <= L, torch.full_like(delta, float("inf")), delta)
        # segment must not wrap: i in [1, n-L-1]
        delta[:, 0, :] = float("inf")
        delta[:, n - L:, :] = float("inf")

        bd, bk = delta.min(dim=2)                   # (M, n)
        bc = torch.gather(c_pos, 2, bk.unsqueeze(2)).squeeze(2)
        out.append((bd.cpu().numpy(), bc.cpu().numpy()))
    return out


def _apply_oropt(tours_np: np.ndarray, per_len, seg_lens=(1, 2, 3), eps=1e-4):
    """Greedy non-overlapping or-opt application. Moves are collected with a
    position dirty-mask, then each tour is rebuilt once (segments pulled out,
    re-inserted after their target city). Returns (moves, metric_saving).

    HIZ: aday toplama/siralama numpy'ye (lexsort ile ayni (d,i,L,c) toplam
    sirasi) ve tur-yeniden-kurma saf-Python `for p in range(n)` dongusu (tur
    basi n dict.get) yerine tamamen vektorize scatter'a tasindi -- kabul
    edilen hamle kumesini ureten sirali cakisma-secim dongusu AYNEN korundu,
    dolayisiyla cikti BIT-AYNI (parity_oropt.py: gercek _oropt_moves ciktisi
    uzerinde eski vs yeni, tours + applied + saving birebir). Bu fonksiyon
    gpu_snake/repair/ngls/gnn_gpu tarafindan paylasildigi icin hepsi ayni
    sonucu daha hizli uretir."""
    M, n = tours_np.shape
    applied = 0
    saving = 0.0
    for m in range(M):
        d_l, i_l, L_l, c_l = [], [], [], []
        for li, L in enumerate(seg_lens):
            if per_len[li] is None:
                continue
            bd, bc = per_len[li]
            bdm = bd[m]
            idx = np.nonzero(bdm < -eps)[0]
            if idx.size == 0:
                continue
            d_l.append(bdm[idx])
            i_l.append(idx.astype(np.int64))
            L_l.append(np.full(idx.size, L, dtype=np.int64))
            c_l.append(bc[m][idx].astype(np.int64))
        if not d_l:
            continue
        d_a = np.concatenate(d_l)
        i_a = np.concatenate(i_l)
        L_a = np.concatenate(L_l)
        c_a = np.concatenate(c_l)
        # ayni toplam sira: birincil d, sonra i, L, c (eski cands.sort() ile ozdes)
        order = np.lexsort((c_a, L_a, i_a, d_a))
        i_s = i_a[order].tolist()
        L_s = L_a[order].tolist()
        c_s = c_a[order].tolist()
        d_s = d_a[order].tolist()
        # --- sirali greedy cakisma-secimi (aynen korundu) ---
        dirty = np.zeros(n, dtype=bool)
        mv_i, mv_L, mv_c = [], [], []
        for t in range(len(i_s)):
            i = i_s[t]; L = L_s[t]; c = c_s[t]
            lo = i - 1 if i - 1 > 0 else 0
            hi = i + L if i + L < n - 1 else n - 1
            if dirty[lo:hi + 1].any() or dirty[c] or dirty[(c + 1) % n]:
                continue
            dirty[lo:hi + 1] = True
            dirty[c] = True
            dirty[(c + 1) % n] = True
            mv_i.append(i); mv_L.append(L); mv_c.append(c)
            saving -= float(d_s[t])
        if not mv_i:
            continue
        # --- vektorize yeniden-kurma (eski dict + for p in range(n) yerine) ---
        tour = tours_np[m]
        mi = np.asarray(mv_i, dtype=np.int64)
        mL = np.asarray(mv_L, dtype=np.int64)
        mc = np.asarray(mv_c, dtype=np.int64)
        tot = int(mL.sum())
        # cikarilan konumlar: her hamlenin [i, i+L) araligi (birlesim)
        removed = np.zeros(n, dtype=bool)
        rm_base = np.repeat(mi, mL)
        rm_intra = np.arange(tot) - np.repeat(np.cumsum(mL) - mL, mL)
        removed[rm_base + rm_intra] = True
        keep = ~removed
        kept_cities = tour[keep]
        K = int(keep.sum())
        kept_rank = np.cumsum(keep) - 1             # tutulan-diziideki sira
        ranks = kept_rank[mc]                        # hedef c daima tutulur
        ins_len = np.zeros(K, dtype=np.int64)
        np.add.at(ins_len, ranks, mL)               # her tutulan sonrasi eklenen
        base_out = np.arange(K) + (np.cumsum(ins_len) - ins_len)
        out = np.empty(n, dtype=tour.dtype)
        out[base_out] = kept_cities
        # ayni hedefe giden segmentler kabul sirasinda (kararli sirala)
        o2 = np.argsort(ranks, kind="stable")
        r_s = ranks[o2]; Ls2 = mL[o2]; is2 = mi[o2]
        cum = np.cumsum(Ls2) - Ls2
        first = np.empty(len(r_s), dtype=bool)
        first[0] = True
        first[1:] = r_s[1:] != r_s[:-1]
        grp_base = np.maximum.accumulate(np.where(first, cum, 0))
        within = cum - grp_base                      # grup-ici birikimli ofset
        seg_start = base_out[r_s] + 1 + within
        so_rep = np.repeat(seg_start, Ls2)
        ss_rep = np.repeat(is2, Ls2)
        intra = np.arange(tot) - np.repeat(np.cumsum(Ls2) - Ls2, Ls2)
        out[so_rep + intra] = tour[ss_rep + intra]
        tours_np[m] = out
        applied += len(mv_i)
    return applied, saving


# ---------------------------------------------------------------------------
#  kNN table (chunked, works for EUC + GEO, no n x n matrix in memory at once)
# ---------------------------------------------------------------------------
def _knn_table(C: torch.Tensor, k: int, is_geo: bool, chunk: int = 2048) -> torch.Tensor:
    n = C.shape[0]
    k = min(k, n - 1)
    out = torch.empty((n, k), device=C.device, dtype=torch.long)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = _pdist(C[s:e, None, :], C[None, :, :], is_geo)   # (chunk, n)
        rows = torch.arange(e - s, device=C.device)
        d[rows, s + rows] = float("inf")                     # exclude self
        out[s:e] = torch.topk(d, k, dim=1, largest=False).indices
    return out


# ---------------------------------------------------------------------------
#  Public entry points
# ---------------------------------------------------------------------------
def repair_tours(
    xs, ys, is_geo: bool,
    tours: list[list[int]],
    time_limit: float = 0.0,
    device: str | None = None,
) -> tuple[np.ndarray, GpuSnakeDiag]:
    """The repair layer alone, on caller-supplied tours: exactly the same
    batched candidate-list 2-opt + or-opt loop `run_gpu_snake` uses, without
    the snake construction. This is the construction-agnostic half of the
    method — used by the `ge_gpu` referee baseline (Greedy-Edge + GPU
    repair) and by gpu_repair_ablation.py's protocol. Returns the improved
    tour batch (same order as input) and a diagnostics object."""
    t_start = time.perf_counter()
    if device is None:
        # 2026-07-24 (kullanici karari): bu calisma bir GPU calismasi DEGIL.
        # Toplu (batched) onarim katmani CPU'da kosar; hamle semantigi ve
        # sonuc cihazdan bagimsizdir (ayni kNN aday listeleri, ayni delta
        # hesaplari, hamleler zaten numpy'da uygulanir), yalniz hiz degisir.
        # Cihazi bilerek zorlamak isteyen device="cuda" gecebilir.
        device = os.environ.get("GREEDY_SNAKE_DEVICE", "cpu")
    tours_np = np.stack([np.asarray(t, dtype=np.int64) for t in tours])
    n = tours_np.shape[1]
    dtype = torch.float64 if is_geo else torch.float32
    C = torch.stack([torch.tensor(np.asarray(xs), device=device, dtype=dtype),
                     torch.tensor(np.asarray(ys), device=device, dtype=dtype)],
                    dim=1)
    k_2opt = 12 if n <= 6000 else 10
    # kNN chunk'i (chunk x n) mesafe matrisini ~512MB'ta tutacak sekilde
    # olceklenir; sabit 2048 satir n>~300k'da 8GB karti tasiriyor. Sonuc
    # chunk boyutundan bagimsiz (ayni topk), yalniz bellek profili degisir.
    bytes_per = 8 if is_geo else 4
    knn_chunk = int(max(128, min(2048, (512 << 20) // max(1, n * bytes_per))))
    knn_full = _knn_table(C, max(k_2opt, 8), is_geo, chunk=knn_chunk)
    knn2 = knn_full[:, :k_2opt]
    knn_or = knn_full[:, :8]

    lens0 = _tour_lengths(C, torch.from_numpy(tours_np).to(device), is_geo)
    history = [HistPt(0, time.perf_counter() - t_start, float(lens0.min()))]
    total_moves = 0
    rounds = 0
    while True:
        if time_limit > 0 and time.perf_counter() - t_start >= time_limit:
            break
        t = torch.from_numpy(tours_np).to(device)
        mv = _knn_2opt_moves(C, t, is_geo, knn2)
        m1, _ = _apply_2opt(tours_np, mv)
        t = torch.from_numpy(tours_np).to(device)
        pl = _oropt_moves(C, t, is_geo, knn_or)
        m2, _ = _apply_oropt(tours_np, pl)
        total_moves += m1 + m2
        rounds += 1
        t = torch.from_numpy(tours_np).to(device)
        best_now = float(_tour_lengths(C, t, is_geo).min())
        history.append(HistPt(rounds, time.perf_counter() - t_start, best_now))
        if m1 + m2 == 0:
            break
    diag = GpuSnakeDiag(
        tour=[], cost=history[-1].cost,
        iterations_completed=rounds,
        improvements_accepted=total_moves,
        total_saving=float(lens0.min()) - history[-1].cost,
        history=history,
        n_starts=tours_np.shape[0],
        device=(f"cuda:{torch.cuda.get_device_name(0)}"
                if device == "cuda" else "cpu"),
        construction_cost=float(lens0.min()),
    )
    return tours_np, diag


def run_gpu_snake(
    xs, ys, is_geo: bool,
    n_starts: int | None = None,
    k_2opt: int | None = None,
    k_or: int = 8,
    max_rounds: int = 200,
    time_limit: float = 0.0,
    device: str | None = None,
    seed_tours: list[tuple[str, list[int]]] | None = None,
    theta_hint: float | None = None,
) -> tuple[list[np.ndarray], GpuSnakeDiag]:
    """Runs the fused multi-start snake + parallel window-repair + adaptive-LR
    method. Returns (final_tours, diag): ALL final batch tours (the caller
    scores them with the exact TSPLIB cost and picks the winner) plus a
    diagnostics object whose history tracks the metric best-so-far curve.

    `seed_tours`: optional HYBRID START POOL — extra (label, tour) pairs
    appended to the snake batch and repaired alongside it (e.g. an NN and a
    Greedy-Edge tour: near-zero extra GPU cost, and the classical seeds'
    strong repaired results become the method's floor while the snake
    variants provide diversity on pathological layouts). The origin of the
    winning tour is reported in diag.winner ("snake" or the seed's label).

    `theta_hint`: theta* mirasi (derece) — runner'in theta dedektorlerinin en
    iyi acisi; snake aci havuzuna hint merkezli acilar eklenir (bkz.
    build_multi_snake). None ise havuz esit-araliklidir."""
    t_start = time.perf_counter()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(xs)
    dtype = torch.float64 if is_geo else torch.float32
    C = torch.stack([torch.tensor(np.asarray(xs), device=device, dtype=dtype),
                     torch.tensor(np.asarray(ys), device=device, dtype=dtype)],
                    dim=1)

    if n_starts is None:
        n_starts = 66 if n <= 2000 else (48 if n <= 6000 else 36)
    if k_2opt is None:
        k_2opt = 12 if n <= 6000 else 10

    tours_t = build_multi_snake(C, n_starts, hint_deg=theta_hint)
    tours_np = tours_t.cpu().numpy().copy()
    origins = ["snake"] * tours_np.shape[0]
    if seed_tours:
        extra = np.stack([np.asarray(t, dtype=np.int64) for _, t in seed_tours])
        tours_np = np.concatenate([tours_np, extra], axis=0)
        origins += [label for label, _ in seed_tours]
    M = tours_np.shape[0]
    lens = _tour_lengths(C, torch.from_numpy(tours_np).to(device), is_geo)
    construction_best = float(lens.min())
    knn_full = _knn_table(C, max(k_or, k_2opt), is_geo)
    knn2 = knn_full[:, :k_2opt]
    knn = knn_full[:, :k_or]
    history = [HistPt(0, time.perf_counter() - t_start, construction_best)]
    total_moves = 0
    total_saving = 0.0
    rounds_done = 0

    for rnd in range(1, max_rounds + 1):
        if time_limit > 0 and time.perf_counter() - t_start >= time_limit:
            break
        t = torch.from_numpy(tours_np).to(device)

        moves = _knn_2opt_moves(C, t, is_geo, knn2)
        m1, s1 = _apply_2opt(tours_np, moves)

        t = torch.from_numpy(tours_np).to(device)
        per_len = _oropt_moves(C, t, is_geo, knn)
        m2, s2 = _apply_oropt(tours_np, per_len)

        rounds_done = rnd
        total_moves += m1 + m2
        total_saving += s1 + s2
        t = torch.from_numpy(tours_np).to(device)
        best_now = float(_tour_lengths(C, t, is_geo).min())
        history.append(HistPt(rnd, time.perf_counter() - t_start, best_now))
        if m1 + m2 == 0:
            break

    t = torch.from_numpy(tours_np).to(device)
    final_lens = _tour_lengths(C, t, is_geo).cpu().numpy()
    order = np.argsort(final_lens)
    diag = GpuSnakeDiag(
        tour=[], cost=float(final_lens[order[0]]),
        iterations_completed=rounds_done,
        improvements_accepted=total_moves,
        total_candidates_evaluated=M * rounds_done * n * (k_2opt + 3 * k_or),
        total_saving=total_saving,
        history=history,
        n_starts=M,
        device=(f"cuda:{torch.cuda.get_device_name(0)}"
                if device == "cuda" else "cpu"),
        construction_cost=construction_best,
        winner=origins[order[0]],
    )
    return [tours_np[i] for i in order], diag

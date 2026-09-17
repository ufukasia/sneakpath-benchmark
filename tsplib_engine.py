# -*- coding: utf-8 -*-
"""TSPLIB engine: parsing, TSPLIB-correct distances, and a sparse (matrix-free)
instance so the user's optimizers run on instances up to usa13509 (13509 cities)
without building an O(n^2) distance matrix.

Bridges TSPLIB .tsp files to the user's `app.TSPInstance` API. The optimizers in
`line_reassign_optimizer.py` only ever access distances via `dist_matrix[a][b]`
and nearest-neighbour lists via `_precompute_city_nearest_neighbors`, so we:
  * provide a lazy `dist_matrix` proxy that computes distances on demand, and
  * replace the O(n^2) kNN precompute with a grid-based approximate kNN.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

# ---- locate the user's engine (app.py / line_reassign_optimizer.py now live
#      alongside this file in the tsplib project directory) ---------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core import City, TSPInstance, DistanceMetric  # noqa: E402
import line_reassign_optimizer as lro  # noqa: E402


# ---- best-known / optimal tour lengths (TSPLIB) --------------------------------
BKS: dict[str, int] = {
    # ali535 (2026-07-28 DUZELTMESI): 202310 -> 202339. Eski deger bu projenin
    # KENDI GEO mesafe fonksiyonuyla ULASILAMAZ bir sayiydi -- Concorde (kesin
    # cozucu) ve LKH-3 ikisi de 202339 uretiyor. Yani kayitli "en iyi bilinen"
    # deger, kanitlanmis optimumun ALTINDAydi; bu imkansizdir ve iki sonuc
    # doguruyordu: (a) tum ali535 gap'leri ~0.0143 puan sisiyordu, (b) kesin
    # cozucu satiri tabloda gap=+0.014% ile gorunup optimumu bulamamis gibi
    # okunuyordu. BKS her zaman bu motorun metrigiyle ayni olcude olmalidir.
    "kroA100": 21282, "rat783": 8806, "ali535": 202339, "gr666": 294358,
    "pcb3038": 137694, "fl3795": 28772, "fnl4461": 182566, "rl5915": 565530,
    "usa13509": 19982859,
}

# above this many cities we use the sparse (lazy) instance
SPARSE_THRESHOLD = 2500


# ---- insa (construction) duvar-saati butcesi ------------------------------------
# runner her insa yontemini kosmadan once CONSTRUCTION_DEADLINE'i (perf_counter
# cinsinden mutlak son an) kurar; asagidaki yavas-olabilen kurucular sicak
# donglerinde check_construction_deadline() cagirir. Butce asilirsa yontem
# ConstructionTimeout ile kesilir ve runner onu "denendi, sonuc yok" olarak
# kaydeder (sessiz kaybolma yok). None iken (varsayilan; ve metasezgisel
# asamalarin icinden yapilan kurucu cagrilarinda) kontroller no-op'tur.
CONSTRUCTION_DEADLINE: float | None = None


class ConstructionTimeout(RuntimeError):
    """Insa, runner'in verdigi duvar-saati butcesini asti (bkz. runner
    _CONSTRUCTION_MAX_S): kismi sonuc atilir, yontem o veri kumesinde
    'denendi, sonuc yok' olur."""


def check_construction_deadline() -> None:
    dl = CONSTRUCTION_DEADLINE
    if dl is not None and time.perf_counter() > dl:
        raise ConstructionTimeout("insa zaman butcesi asildi")


# ======================================================================
#  Parsing
# ======================================================================
def parse_tsp(path: str | Path) -> tuple[dict, list[tuple[float, float]], str]:
    """Returns (header, coords, edge_weight_type). GEO coords are converted to
    decimal degrees (lat, lon)."""
    header: dict[str, str] = {}
    coords: list[tuple[float, float]] = []
    in_section = False
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("NODE_COORD_SECTION"):
            in_section = True
            continue
        if up.startswith(("EOF", "DISPLAY_DATA_SECTION", "TOUR_SECTION")):
            if in_section:
                break
            continue
        if not in_section:
            if ":" in s:
                k, _, v = s.partition(":")
                header[k.strip().upper()] = v.strip()
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                coords.append((float(parts[1]), float(parts[2])))
            except ValueError:
                pass
    ewt = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if "GEO" in ewt:
        coords = [(_ddmm_to_deg(x), _ddmm_to_deg(y)) for (x, y) in coords]
    return header, coords, ewt


def _ddmm_to_deg(v: float) -> float:
    """TSPLIB DDD.MM coordinate -> decimal degrees."""
    deg = int(v)
    minutes = v - deg
    return deg + 5.0 * minutes / 3.0


# ======================================================================
#  TSPLIB-correct distances
# ======================================================================
_RRR = 6378.388  # TSPLIB earth radius (km)

# ---------------------------------------------------------------------------
# MESAFE TIPI COZUMLEMESI (2026-07-31)
# ---------------------------------------------------------------------------
# Eskiden yalniz `is_geo: bool` vardi ve GEO DISINDAKI HER SEY EUC_2D
# sayiliyordu. Waterloo koleksiyonlarinda bu dogruydu (hepsi EUC_2D), ama
# orijinal TSPLIB'de degil: CEIL_2D ve ATT ornekleri sessizce YANLIS
# olculurdu. CEIL_2D ozellikle onemli -- o tipteki uc ornegin ucu de DEVRE
# verisi (pla7397/pla33810/pla85900, "programmed logic array").
#
# Geriye donuk uyum: fonksiyonlar bool da kabul eder (True -> "geo").
_METRICS = ("euc", "geo", "ceil", "att")
#: metric_kind() anahtari -> uzaklik fonksiyonu. Asagida tanimlanan
#: fonksiyonlara BAGLANMASI icin modul sonunda doldurulur.
_METRIC_FN: dict = {}


def metric_kind(ewt) -> str:
    """EDGE_WEIGHT_TYPE -> mesafe tipi anahtari. bool da kabul edilir
    (eski `is_geo` cagrilarinin bit-ayni calismasi icin)."""
    if isinstance(ewt, bool):
        return "geo" if ewt else "euc"
    e = (ewt or "EUC_2D").upper()
    if "GEO" in e:
        return "geo"
    if "CEIL_2D" in e:
        return "ceil"
    if e.startswith("ATT"):
        return "att"
    return "euc"


def tsplib_euc(ax: float, ay: float, bx: float, by: float) -> float:
    """EUC_2D: nearest-integer rounding (TSPLIB nint convention)."""
    return float(int(math.hypot(ax - bx, ay - by) + 0.5))


def tsplib_ceil(ax: float, ay: float, bx: float, by: float) -> float:
    """CEIL_2D: Oklid uzakligin TAVANI (nint DEGIL).

    TSPLIB'de yalniz uc ornek bu tipi kullanir ve ucu de DEVRE verisidir:
    pla7397 / pla33810 / pla85900 ("programmed logic array", Johnson) --
    yani kullanicinin oncelikli sinifinin en buyuk uyeleri. EUC_2D sanilirsa
    her kenar 0-1 birim kisa olcusulur ve gap SESSIZCE yanlis cikar."""
    return float(math.ceil(math.hypot(ax - bx, ay - by)))


def tsplib_att(ax: float, ay: float, bx: float, by: float) -> float:
    """ATT: TSPLIB'in "pseudo-Euclidean" uzakligi (att48 / att532).

    Tanim TSPLIB belgesindeki gibidir: r = sqrt((dx^2+dy^2)/10),
    t = nint(r); t < r ise t+1, degilse t."""
    dx = ax - bx
    dy = ay - by
    r = math.sqrt((dx * dx + dy * dy) / 10.0)
    t = float(int(r + 0.5))
    return t + 1.0 if t < r else t


def tsplib_geo(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """GEO distance from decimal-degree lat/lon (TSPLIB convention)."""
    la1 = math.radians(lat1); lo1 = math.radians(lon1)
    la2 = math.radians(lat2); lo2 = math.radians(lon2)
    q1 = math.cos(lo1 - lo2)
    q2 = math.cos(la1 - la2)
    q3 = math.cos(la1 + la2)
    val = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    val = max(-1.0, min(1.0, val))
    return float(int(_RRR * math.acos(val) + 1.0))


_METRIC_FN.update({"euc": tsplib_euc, "geo": tsplib_geo,
                   "ceil": tsplib_ceil, "att": tsplib_att})


# ======================================================================
#  Lazy distance proxy (matrix-free)
# ======================================================================
_hypot = math.hypot  # yerel alias: sicak dm[a][b] yolunda attr aramasini keser


class _LazyRow:
    """`dist_matrix[i]` proxy'si. Sicak yol (EUC, i!=j) satir-ici hesaplanir:
    koordinatlar satirda onbelleklenir ve `LazyDistance.dist()` metot cagri
    katmani + is_geo/i==j dallari atlanir. GEO ve i==j `dist()`'e yonlendirilir
    (birebir ayni deger). LazyDistance satirlari onbellekledigi icin bu daha
    agir __init__ tur/sehir basina bir kez odenir, her erisimde degil."""
    __slots__ = ("d", "i", "_geo", "_xs", "_ys", "_xi", "_yi")

    def __init__(self, d: "LazyDistance", i: int):
        self.d = d
        self.i = i
        self._geo = d.is_geo
        xs = d.xs
        ys = d.ys
        self._xs = xs
        self._ys = ys
        self._xi = xs[i]
        self._yi = ys[i]

    def __getitem__(self, j: int) -> float:
        # Sicak yol YALNIZ duz EUC_2D icindir; digerleri (geo/ceil/att) ve
        # i==j `dist()`'e gider -- birebir ayni deger, sadece daha yavas yol.
        if self._geo or j == self.i:
            return self.d.dist(self.i, j)
        return float(int(_hypot(self._xi - self._xs[j], self._yi - self._ys[j]) + 0.5))


class LazyDistance:
    """Behaves like `dist_matrix[a][b]` but computes distances on demand."""

    def __init__(self, xs, ys, is_geo=False, metric: "str | None" = None):
        self.xs = xs
        self.ys = ys
        self.metric = metric or metric_kind(is_geo)
        # `is_geo` GERIYE DONUK alan olarak KALIYOR (snake_numpy ve dis
        # betikler okuyor). Artik "duz EUC_2D degil" anlamina gelir: ceil/att
        # de sicak yoldan CIKARILMALI, aksi halde yanlis yuvarlanirlar.
        self.is_geo = self.metric != "euc"
        self.n = len(xs)
        self._rowcache: dict[int, _LazyRow] = {}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> _LazyRow:
        r = self._rowcache.get(i)
        if r is None:
            r = self._rowcache[i] = _LazyRow(self, i)
        return r

    def dist(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        xs, ys = self.xs, self.ys
        m = self.metric
        if m == "geo":
            return tsplib_geo(xs[i], ys[i], xs[j], ys[j])
        if m == "ceil":
            return tsplib_ceil(xs[i], ys[i], xs[j], ys[j])
        if m == "att":
            return tsplib_att(xs[i], ys[i], xs[j], ys[j])
        return float(int(math.hypot(xs[i] - xs[j], ys[i] - ys[j]) + 0.5))


# ======================================================================
#  Instances
# ======================================================================
class _RoundedInstance(TSPInstance):
    """Dense instance with TSPLIB-correct rounded distances (small/medium n).

    `dist_coords` (if given) holds the ORIGINAL coordinates used for distance
    computation, while `coords` holds the (possibly rotated) coordinates used for
    the cities' spatial layout. This lets the whole pipeline run in a rotated
    frame (snake grid, window repair, projections) while distances/costs stay the
    true TSPLIB values (rotation-invariant for EUC, exact for GEO)."""

    def __init__(self, coords, is_geo=False, dist_coords=None,
                 metric: "str | None" = None):
        self.metric = metric or metric_kind(is_geo)
        self._is_geo = self.metric != "euc"   # geriye donuk alan; bkz. LazyDistance
        self._fn = _METRIC_FN[self.metric]
        dc = dist_coords if dist_coords is not None else coords
        self._dx = [c[0] for c in dc]
        self._dy = [c[1] for c in dc]
        cities = [City(i, str(i), c[0], c[1]) for i, c in enumerate(coords)]
        super().__init__(cities, DistanceMetric.EUCLIDEAN, axis_labels=("X", "Y"))

    def _pairwise_distance(self, a: City, b: City) -> float:
        i = a.identifier; j = b.identifier
        return self._fn(self._dx[i], self._dy[i], self._dx[j], self._dy[j])


class _SparseInstance(TSPInstance):
    """Matrix-free instance: skips the O(n^2) build, uses a lazy distance proxy.
    `dist_coords` plays the same role as in `_RoundedInstance` (true-distance
    coordinates while the cities may carry a rotated layout)."""

    def __init__(self, coords, is_geo=False, dist_coords=None,
                 metric: "str | None" = None):
        if len(coords) < 3:
            raise ValueError("A TSP instance requires at least three cities.")
        self.metric = metric or metric_kind(is_geo)
        self._is_geo = self.metric != "euc"   # geriye donuk alan
        self.cities = [City(i, str(i), c[0], c[1]) for i, c in enumerate(coords)]
        self.distance_metric = DistanceMetric.EUCLIDEAN
        self.axis_labels = ("X", "Y")
        dc = dist_coords if dist_coords is not None else coords
        xs = [c[0] for c in dc]
        ys = [c[1] for c in dc]
        self._distance_matrix = LazyDistance(  # type: ignore[assignment]
            xs, ys, metric=self.metric)


def rotate_coord_pairs(coords, deg):
    """Rotates a list of (x, y) pairs around their centroid by `deg` degrees."""
    if not deg:
        return list(coords)
    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    a = math.radians(deg)
    ca = math.cos(a); sa = math.sin(a)
    out = []
    for x, y in coords:
        dx = x - cx; dy = y - cy
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def make_instance(coords, ewt: str, force_dense: bool = False, theta_deg: float = 0.0,
                  force_sparse: bool = False):
    """Builds the lightest instance that fits: dense rounded matrix for small n,
    sparse lazy proxy for large n. When `theta_deg` is non-zero the cities' layout
    is rotated by theta (so snake/repair/line-reassign operate in the optimal
    snake-sweep frame) while distances are computed on the ORIGINAL coordinates,
    keeping costs/gaps true. Returns (instance, is_sparse).

    `force_sparse` skips the O(n^2) dense matrix even below SPARSE_THRESHOLD —
    for callers that only ever evaluate tour edges from coordinates (e.g. the
    stratified-subsample SEARCH, which scores variants with the vectorized
    coordinate backend and never touches the matrix). Distances are computed
    by the same TSPLIB formulas either way, so results are bit-identical."""
    metric = metric_kind(ewt)
    n = len(coords)
    layout = rotate_coord_pairs(coords, theta_deg) if theta_deg else coords
    if force_dense or (n <= SPARSE_THRESHOLD and not force_sparse):
        return _RoundedInstance(layout, dist_coords=coords, metric=metric), False
    inst = _SparseInstance(layout, dist_coords=coords, metric=metric)
    install_sparse_knn(inst)
    return inst, True


# ======================================================================
#  Grid-based approximate kNN (replaces the O(n^2) precompute)
# ======================================================================
def grid_knn(xs, ys, k: int) -> list[list[int]]:
    """Approximate k nearest neighbours per point via a uniform grid. Returns a
    list per point of nearest city indices (self excluded), nearest first."""
    n = len(xs)
    k = min(k, n - 1)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = (maxx - minx) or 1.0
    h = (maxy - miny) or 1.0
    g = max(1, int(math.sqrt(n)))
    inv_w = g / (w * (1.0 + 1e-9))
    inv_h = g / (h * (1.0 + 1e-9))

    cells: list[list[int]] = [[] for _ in range(g * g)]
    cx_of = [0] * n
    cy_of = [0] * n
    for i in range(n):
        cx = int((xs[i] - minx) * inv_w)
        cy = int((ys[i] - miny) * inv_h)
        if cx >= g:
            cx = g - 1
        if cy >= g:
            cy = g - 1
        cx_of[i] = cx
        cy_of[i] = cy
        cells[cy * g + cx].append(i)

    result: list[list[int]] = [None] * n  # type: ignore[list-item]
    hypot = math.hypot
    for i in range(n):
        xi = xs[i]; yi = ys[i]
        cx = cx_of[i]; cy = cy_of[i]
        cand: list[int] = []
        r = 0
        enough_at = None
        while True:
            x0 = max(0, cx - r); x1 = min(g - 1, cx + r)
            y0 = max(0, cy - r); y1 = min(g - 1, cy + r)
            # only the ring at Chebyshev distance r
            for gy in range(y0, y1 + 1):
                on_y_edge = (gy == cy - r) or (gy == cy + r)
                row = gy * g
                if on_y_edge:
                    for gx in range(x0, x1 + 1):
                        cand.extend(cells[row + gx])
                else:
                    if cx - r >= 0:
                        cand.extend(cells[row + cx - r])
                    if cx + r <= g - 1 and r != 0:
                        cand.extend(cells[row + cx + r])
            if enough_at is None and len(cand) >= 2 * k + 4:
                enough_at = r
            # stop one full ring after we first had enough candidates
            if enough_at is not None and r >= enough_at + 1:
                break
            if x0 == 0 and y0 == 0 and x1 == g - 1 and y1 == g - 1:
                break
            r += 1
        cand = [c for c in cand if c != i]
        cand.sort(key=lambda j: hypot(xi - xs[j], yi - ys[j]))
        result[i] = cand[:k]
    return result


# Süreç içinde AYNI ANDA birden çok boyutta örnek yaşayabilir: tam küme +
# stratified-subsample ARAMASININ 1200'lük alt-örneklem örneği (force_sparse).
# Eski "son kuran kazanır" monkeypatch'i yüzünden alt-örneklem kurulduktan
# sonra TAM kümenin ILS'i 1200 satırlık komşu listesi alıyordu -> n>1200 her
# örnekte koşum ortasında IndexError (bck2217 vb. "sonuç dosyası hiç yazılmadı"
# hatasının kökü). Çözüm: nokta-sayısına anahtarlı kayıt; sevkiyat çağıran
# örneğin BOYUTUNA bakar, kayıt yoksa orijinal matris-tabanlı hesaba düşer.
# (xs/ys her zaman ORİJİNAL koordinatlardır -- dist_coords; bu yüzden aynı
# veri kümesinin döndürülmüş kopyaları da aynı kayıttan DOĞRU listeyi alır.)
_GRID_KNN_REGISTRY: dict[int, tuple[list, list, dict]] = {}
_LRO_ORIG_PRECOMPUTE = None


def install_sparse_knn(instance) -> None:
    """Registers a cached grid-kNN backend for this instance's SIZE and (once
    per process) monkeypatches the optimizer's kNN precompute with a
    size-dispatching wrapper, so the full pipeline runs matrix-free. Sizes
    without a registered backend fall back to the original matrix-based
    precompute -- a subsample instance can no longer poison the full
    instance's neighbor lists."""
    global _LRO_ORIG_PRECOMPUTE
    xs = instance._distance_matrix.xs
    ys = instance._distance_matrix.ys
    _GRID_KNN_REGISTRY[len(xs)] = (xs, ys, {})
    if _LRO_ORIG_PRECOMPUTE is not None:
        return  # sevkiyatçı zaten kurulu; sadece kayıt güncellendi
    _LRO_ORIG_PRECOMPUTE = lro._precompute_city_nearest_neighbors

    def _grid_precompute(inst, neighbor_count):  # signature match
        entry = _GRID_KNN_REGISTRY.get(inst.size)
        if entry is None:
            return _LRO_ORIG_PRECOMPUTE(inst, neighbor_count)
        gxs, gys, cache = entry
        kk = min(neighbor_count, len(gxs) - 1)
        # serve from the largest cached table if it covers kk
        for have_k, table in cache.items():
            if have_k >= kk:
                if have_k == kk:
                    return table
                return [row[:kk] for row in table]
        table = grid_knn(gxs, gys, kk)
        cache[kk] = table
        return table

    lro._precompute_city_nearest_neighbors = _grid_precompute


def nearest_insertion_tour(xs, ys, start: int = 0) -> list[int]:
    """Classic Nearest-Insertion construction (Rosenkrantz, Stearns & Lewis
    1977, 'An analysis of several heuristics for the traveling salesman
    problem'): repeatedly inserts the unvisited city NEAREST to the current
    partial tour, at the tour edge that minimises insertion cost. O(n^2),
    deliberately un-accelerated (no candidate-list shortcuts) so it matches the
    standard literature baseline exactly and is directly gap-comparable to
    published Nearest-Insertion results. Same input convention as
    `grid_nn_tour`/`snake_order`: raw (xs, ys) Euclidean proximity drives
    construction; the caller computes TRUE (TSPLIB-rounded / GEO) cost via
    `inst.tour_cost`."""
    n = len(xs)
    hypot = math.hypot
    if n < 3:
        return list(range(n))
    nearest = min((j for j in range(n) if j != start),
                  key=lambda j: hypot(xs[start] - xs[j], ys[start] - ys[j]))
    tour = [start, nearest]
    in_tour = bytearray(n)
    in_tour[start] = 1
    in_tour[nearest] = 1
    min_dist = [hypot(xs[i] - xs[start], ys[i] - ys[start]) for i in range(n)]
    for i in range(n):
        if not in_tour[i]:
            d = hypot(xs[i] - xs[nearest], ys[i] - ys[nearest])
            if d < min_dist[i]:
                min_dist[i] = d
    remaining = [i for i in range(n) if not in_tour[i]]
    for step in range(n - 2):
        if not (step & 0xFF):
            check_construction_deadline()
        c = min(remaining, key=lambda i: min_dist[i])
        remaining.remove(c)
        best_pos, best_delta = 0, float("inf")
        m = len(tour)
        for p in range(m):
            a = tour[p]; b = tour[(p + 1) % m]
            delta = (hypot(xs[a] - xs[c], ys[a] - ys[c]) +
                     hypot(xs[c] - xs[b], ys[c] - ys[b]) -
                     hypot(xs[a] - xs[b], ys[a] - ys[b]))
            if delta < best_delta:
                best_delta, best_pos = delta, p
        tour.insert(best_pos + 1, c)
        in_tour[c] = 1
        xc, yc = xs[c], ys[c]
        for i in remaining:
            d = hypot(xs[i] - xc, ys[i] - yc)
            if d < min_dist[i]:
                min_dist[i] = d
    return tour


def farthest_insertion_tour(xs, ys, start: int = 0) -> list[int]:
    """Classic Farthest-Insertion construction (Rosenkrantz, Stearns & Lewis
    1977): repeatedly inserts the unvisited city FARTHEST from the current
    partial tour (maximises the minimum distance to the tour), at its cheapest
    insertion edge. Farthest insertion typically beats nearest insertion on
    Euclidean instances (better global coverage before local refinement holds
    it together). O(n^2), un-accelerated -- literature-comparable. Same input
    convention as `nearest_insertion_tour`."""
    n = len(xs)
    hypot = math.hypot
    if n < 3:
        return list(range(n))
    farthest = max((j for j in range(n) if j != start),
                   key=lambda j: hypot(xs[start] - xs[j], ys[start] - ys[j]))
    tour = [start, farthest]
    in_tour = bytearray(n)
    in_tour[start] = 1
    in_tour[farthest] = 1
    min_dist = [hypot(xs[i] - xs[start], ys[i] - ys[start]) for i in range(n)]
    for i in range(n):
        if not in_tour[i]:
            d = hypot(xs[i] - xs[farthest], ys[i] - ys[farthest])
            if d < min_dist[i]:
                min_dist[i] = d
    remaining = [i for i in range(n) if not in_tour[i]]
    for step in range(n - 2):
        if not (step & 0xFF):
            check_construction_deadline()
        c = max(remaining, key=lambda i: min_dist[i])
        remaining.remove(c)
        best_pos, best_delta = 0, float("inf")
        m = len(tour)
        for p in range(m):
            a = tour[p]; b = tour[(p + 1) % m]
            delta = (hypot(xs[a] - xs[c], ys[a] - ys[c]) +
                     hypot(xs[c] - xs[b], ys[c] - ys[b]) -
                     hypot(xs[a] - xs[b], ys[a] - ys[b]))
            if delta < best_delta:
                best_delta, best_pos = delta, p
        tour.insert(best_pos + 1, c)
        in_tour[c] = 1
        xc, yc = xs[c], ys[c]
        for i in remaining:
            d = hypot(xs[i] - xc, ys[i] - yc)
            if d < min_dist[i]:
                min_dist[i] = d
    return tour


# Onarim ciftlerinin sirali-numpy yolu icin ust E siniri: E^2/2 cift saklanir
# (E=12000 -> 72M cift; du+order+int32 pa/pb ~1.7GB tepe bellek). E, n ile
# ~oransal buyur (sra104815 n=104k -> E=4571), 12000 siniri n~270k'ya kadar
# yeter (ara238025 dahil). Ustunde eski O(E^3) tarama + butce kontrolu devreye
# girer (o boyutta insa zaten runner'in 300s kapagini asar ve atlanir).
_REPAIR_SORTED_PAIRS_MAX_E = 12000


def greedy_edge_tour(xs, ys, k: int = 15,
                     tie_seed: "int | None" = None,
                     tie_eps: float = 0.0) -> list[int]:
    """Greedy-Edge construction (the 'Greedy' baseline of Johnson & McGeoch
    1997, 'The Traveling Salesman Problem: A Case Study in Local
    Optimization' -- already the benchmark_methodology_reference for this
    project): sorts candidate edges by length and adds each edge unless it
    would push a city's degree above 2 or close a sub-cycle before the tour is
    complete (Union-Find cycle check). Candidate edges come from a k-NN list
    (`grid_knn`, the same grid-based structure used for the sparse optimizer
    kNN) instead of the full O(n^2) edge set, so this scales to the largest
    TSPLIB instances (usa13509, ~7s). Any path fragments left after the
    candidate pass (k too small to directly connect every city) are stitched
    by greedily joining the closest pair of endpoints from different
    fragments -- a small set in practice, so this repair stays cheap.

    tie_seed / tie_eps (2026-07-24, ge_pool_repair havuz çeşitlendirmesi):
    tie_seed verilirse her aday kenarın sıralama ağırlığı
    d * (1 + tie_eps * u), u ∈ [-1, 1) — u, (tie_seed, i, j)'nin splitmix64
    benzeri deterministik hash'inden üretilir (kenarın hangi yönden
    görüldüğünden bağımsız: kanonik (min, max) çiftiyle). Eşit/near-eşit
    uzunluklu kenarların (ızgara-tipi VLSI levhalarında çok yaygın) sırası
    tohumdan tohuma değişir -> aynı kurucudan FARKLI ama tamamen
    tekrarlanabilir turlar. Kurucu mekaniği (derece/union-find/dikiş)
    değişmez; tie_seed=None eski davranışla birebir aynıdır."""
    n = len(xs)
    if n < 3:
        return list(range(n))
    hypot = math.hypot
    knn = grid_knn(xs, ys, min(k, n - 1))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    seen = set()
    cand = []
    jitter = tie_seed is not None and tie_eps > 0.0
    if jitter:
        _M1 = 0xBF58476D1CE4E5B9
        _M2 = 0x94D049BB133111EB
        _MASK = 0xFFFFFFFFFFFFFFFF

        def _pw(d: float, i: int, j: int) -> float:
            # splitmix64-benzeri deterministik hash -> u ∈ [-1, 1) perturbasyonu
            h = (tie_seed * 0x9E3779B97F4A7C15 + i * _M1 + j * _M2) & _MASK
            h = ((h ^ (h >> 30)) * _M1) & _MASK
            h = ((h ^ (h >> 27)) * _M2) & _MASK
            h ^= h >> 31
            u = (h >> 11) / float(1 << 53)          # [0, 1)
            return d * (1.0 + tie_eps * (2.0 * u - 1.0))
    for i in range(n):
        if not (i & 0x3FFF):
            check_construction_deadline()
        for j in knn[i]:
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            d = hypot(xs[i] - xs[j], ys[i] - ys[j])
            cand.append((_pw(d, key[0], key[1]) if jitter else d,
                         key[0], key[1]))
    cand.sort(key=lambda e: e[0])

    n_edges = 0
    for _, i, j in cand:
        if n_edges == n - 1:
            break
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        degree[i] += 1; degree[j] += 1
        adj[i].append(j); adj[j].append(i)
        n_edges += 1

    # repair: stitch remaining fragment endpoints (degree < 2) into one cycle.
    # Kural degismedi: her seferinde FARKLI-fragman en yakin uc cifti (esitlikte
    # kucuk (a, b) indeksi). Eski hali bu kurali her dikiste tum ciftleri saf
    # Python'la tarayip uc listesini O(n) yeniden kurarak uyguluyordu --
    # E uc icin O(E^3 + E*n): sra104815'te (E=4571, 2327 fragman) tek basina
    # ~4089s (olcum 2026-07-16; aday gecisi yalnizca 5.3s idi). Gecerlilik
    # MONOTON dustugu icin (derece yalniz artar, fragmanlar yalniz birlesir)
    # "her adimda gecerli en yakin cift" == "ciftleri BIR KEZ mesafeye gore
    # sirala, sirayla gez, gecersizleri atla" -- birebir ayni dikis dizisi.
    # (parity_greedy_repair.py: 10 gercek enstans + 60 bant-boyu altkume,
    # eski/yeni tur listeleri AYNI; sra104815 4094s -> ~10s.) Cok buyuk E'de
    # (cift matrisi bellege sigmayacaksa) eski tarama korunur, orada butce
    # kontrolu keser.
    endpoints = [i for i in range(n) if degree[i] < 2]
    if len(endpoints) > 2 and len(endpoints) <= _REPAIR_SORTED_PAIRS_MAX_E:
        import numpy as _np
        ep = endpoints
        ex = _np.array([xs[i] for i in ep])
        ey = _np.array([ys[i] for i in ep])
        pa, pb = _np.triu_indices(len(ep), k=1)      # satir-major = leksikografik
        du = _np.hypot(ex[pa] - ex[pb], ey[pa] - ey[pb])
        pa = pa.astype(_np.int32, copy=False)        # bellek: E=12k'da ~1.7GB tepe
        pb = pb.astype(_np.int32, copy=False)
        order = _np.argsort(du, kind="stable")       # esitlikte kucuk (a, b) once
        del du
        frags_left = len({find(i) for i in ep})
        for t in range(order.size):
            if frags_left == 1:
                break
            if not (t & 0xFFFF):
                check_construction_deadline()
            a = ep[int(pa[order[t]])]
            b = ep[int(pb[order[t]])]
            if degree[a] >= 2 or degree[b] >= 2:
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            parent[ra] = rb
            degree[a] += 1; degree[b] += 1
            adj[a].append(b); adj[b].append(a)
            frags_left -= 1
        endpoints = [i for i in range(n) if degree[i] < 2]
    while len(endpoints) > 2:
        check_construction_deadline()
        best = None
        for a_idx in range(len(endpoints)):
            a = endpoints[a_idx]
            ra = find(a)
            for b_idx in range(a_idx + 1, len(endpoints)):
                b = endpoints[b_idx]
                if find(b) == ra:
                    continue
                d = hypot(xs[a] - xs[b], ys[a] - ys[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        parent[find(a)] = find(b)
        degree[a] += 1; degree[b] += 1
        adj[a].append(b); adj[b].append(a)
        endpoints = [i for i in range(n) if degree[i] < 2]
    if len(endpoints) == 2:
        a, b = endpoints
        adj[a].append(b); adj[b].append(a)
        degree[a] += 1; degree[b] += 1

    # walk the resulting degree-2 graph into a tour order.
    tour = [0]
    prev, cur = -1, 0
    for _ in range(n - 1):
        a0, a1 = adj[cur][0], adj[cur][1]
        nxt = a0 if a0 != prev else a1
        tour.append(nxt)
        prev, cur = cur, nxt
    return tour


# ---------------------------------------------------------------------------
# GREEDY-EDGE DUYARLILIK ABLASYONLARI (2026-07-26, hakem savunmasi)
# ---------------------------------------------------------------------------
# Itiraz (kullanici, 2026-07-26): "Greedy-Edge rakip satiri haksiz guclu --
# (1) k-NN aday budamasi ona uzamsal zeka veriyor, (2) aday kenarlar bitince
# yaptigi akilli uc-eslestirme ikinci bir silah, (3) arkasinda derlenmis C
# kodu var." Uc iddia da OLCULDU ve DOGRULANMADI; ama iddiayi metinle degil
# VERIYLE kapatmak icin uc karsi-olgusal varyant burada uretilir ve panelden
# secilerek tabloda ASIL SATIRIN YANINA kosar (uzerine degil).
#
# Olcum (kroA100 / rat783 / pcb3038, 2026-07-26):
#   kNN k=15 + akilli dikis   13.70 / 18.67 / 18.12 %   0.00 / 0.05 / 0.28 s
#   TAM O(n^2) (kNN YOK)      13.70 / 19.63 / 19.43 %   0.00 / 0.43 / 8.27 s
#   kNN k=3                   13.70 / 19.34 / 18.75 %   0.00 / 0.11 / 2.85 s
#   kNN k=15 + RASTGELE dikis 28.55 / 57.81 / 101.06 %
# Yani: (1) k-NN budamasi KALITE KAZANDIRMIYOR -- kaldirilinca tur AYNI ya da
# DAHA KOTU; kazandirdigi tek sey hiz (30x). Budama bir KISIT, kenar EKLEMEZ.
# (2) Akilli dikis "ekstra" degil, kanonik Greedy'nin ta kendisi: tam O(n^2)
# greedy'de dikis fazi HIC YOKTUR (tum kenarlar zaten adaydir) ve sonucu
# kNN+akilli-dikis ile ayni seviyededir -- dikis, budanan kenarlari ayni
# greedy sirayla geri koyuyor. Rastgelelestirmek Johnson & McGeoch'un
# Greedy'sini degil ondan ~5x kotu bir kuklayi uretir.
# (3) Bu dosyada derlenmis kod YOK; greedy_edge_tour bastan sona saf Python
# (tek numpy kullanimi buyuk-E dikisindeki argsort).
#
# greedy_edge_tour KASITLI OLARAK degistirilmedi: parity_greedy_repair.py o
# fonksiyonun bit-esligini 68+60 vakada kanitliyor ve snake_alt._greedy_band_
# order (Greedy Snake'in KENDI bant kurucusu) onu cagiriyor -- ablasyon icin
# oraya dokunmak kanitlanmis yolu riske atardi.

# Tam O(n^2) varyantin ust n siniri: n(n-1)/2 kenar numpy'de tutulur
# (du float64 + pa/pb int32 + argsort int64 ~ 36 B/kenar). n=6000 -> 18M
# kenar ~ 650MB tepe. Ustunde anlamli bir olcum yapilamaz -- ki "saf greedy
# olceklenmez" bulgusunun kendisi de budur, sessizce kirpmak yerine acikca
# hata verilir.
_FULL_GREEDY_MAX_N = 6000

GE_ABLATION_MODES = ("knn15_greedy", "knn8_greedy", "knn3_greedy",
                     "full_greedy", "knn15_random")


def _ge_walk(adj, n):
    tour = [0]
    prev, cur = -1, 0
    for _ in range(n - 1):
        a0, a1 = adj[cur][0], adj[cur][1]
        nxt = a0 if a0 != prev else a1
        tour.append(nxt)
        prev, cur = cur, nxt
    return tour


def greedy_edge_ablation_tour(xs, ys, mode: str = "knn15_greedy",
                              stitch_seed: int = 0) -> list[int]:
    """Greedy-Edge'in karsi-olgusal varyantlari (yukaridaki nota bakin).

    mode:
      knn15_greedy  -- rakip satirin KENDISI (k=15 + kanonik dikis). Dogrudan
                       greedy_edge_tour'a devreder: bit-ayni tur.
      knn8_greedy   -- k=8; Greedy Snake'in bant ici kurucusuyla AYNI aday
                       genisligi (snake_alt._greedy_band_order, k=min(8,m-1)).
      knn3_greedy   -- k=3; "aday listesini bant basina dusen nokta sayisina
                       cekin" itirazinin uc hali.
      full_greedy   -- k-NN YOK: tum n(n-1)/2 kenar siralanir. Kanonik Greedy'nin
                       budanmamis hali; dikis fazi dogal olarak devre disi kalir
                       (fragman kalmaz). n > _FULL_GREEDY_MAX_N -> ValueError.
      knn15_random  -- k=15 aday gecisi AYNI, ama aday kenarlar bittikten sonra
                       acikta kalan uclar EN YAKIN degil RASTGELE (stitch_seed
                       ile deterministik) baglanir. "Akilli dikisi kapat"
                       itirazinin birebir karsiligi.

    Donen tur her zaman gecerli bir Hamilton cevrimidir; tum modlar
    deterministiktir (knn15_random verilen stitch_seed icin tekrarlanabilir).
    """
    if mode not in GE_ABLATION_MODES:
        raise ValueError(f"bilinmeyen Greedy-Edge ablasyon modu: {mode!r}")
    n = len(xs)
    if n < 3:
        return list(range(n))
    if mode == "knn15_greedy":
        return greedy_edge_tour(xs, ys, k=15)
    if mode == "knn8_greedy":
        return greedy_edge_tour(xs, ys, k=8)
    if mode == "knn3_greedy":
        return greedy_edge_tour(xs, ys, k=3)

    hypot = math.hypot
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]

    if mode == "full_greedy":
        if n > _FULL_GREEDY_MAX_N:
            raise ValueError(
                f"full_greedy: n={n} > {_FULL_GREEDY_MAX_N} — budanmamis "
                f"O(n^2) kenar kumesi bu boyutta bellege sigmaz")
        import numpy as _np
        ax = _np.asarray(xs, dtype=_np.float64)
        ay = _np.asarray(ys, dtype=_np.float64)
        pa, pb = _np.triu_indices(n, k=1)
        du = _np.hypot(ax[pa] - ax[pb], ay[pa] - ay[pb])
        pa = pa.astype(_np.int32, copy=False)
        pb = pb.astype(_np.int32, copy=False)
        order = _np.argsort(du, kind="stable")
        del du
        n_edges = 0
        for t in range(order.size):
            if n_edges == n - 1:
                break
            if not (t & 0xFFFF):
                check_construction_deadline()
            e = int(order[t])
            i = int(pa[e]); j = int(pb[e])
            if degree[i] >= 2 or degree[j] >= 2:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            parent[ri] = rj
            degree[i] += 1; degree[j] += 1
            adj[i].append(j); adj[j].append(i)
            n_edges += 1
        # Budanmamis kumede aday kenarlar TUKENMEZ: yukaridaki dongu her zaman
        # n-1 kenarla biter, geriye tek bir acik yol kalir -> DIKIS FAZI YOK.
        # (Bu, "akilli dikis ekstra bir silah degil" kanitinin ta kendisi.)
        ep = [i for i in range(n) if degree[i] < 2]
        if len(ep) == 2:
            a, b = ep
            adj[a].append(b); adj[b].append(a)
            degree[a] += 1; degree[b] += 1
        return _ge_walk(adj, n)

    # ---- knn15_random: aday gecisi greedy_edge_tour ile BIREBIR AYNI,
    #      yalniz dikis kurali rastgele. ----
    knn = grid_knn(xs, ys, min(15, n - 1))
    seen = set()
    cand = []
    for i in range(n):
        if not (i & 0x3FFF):
            check_construction_deadline()
        for j in knn[i]:
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            cand.append((hypot(xs[i] - xs[j], ys[i] - ys[j]), key[0], key[1]))
    cand.sort(key=lambda e: e[0])
    n_edges = 0
    for _, i, j in cand:
        if n_edges == n - 1:
            break
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        degree[i] += 1; degree[j] += 1
        adj[i].append(j); adj[j].append(i)
        n_edges += 1

    rng = random.Random(stitch_seed)
    endpoints = [i for i in range(n) if degree[i] < 2]
    while len(endpoints) > 2:
        check_construction_deadline()
        # RASTGELE ama GECERLI: uclari karistir, ilk gordugu farkli-fragman
        # ciftini bagla (mesafeye BAKMAZ -- "akilli dikis" tam olarak budur).
        rng.shuffle(endpoints)
        joined = False
        for ia in range(len(endpoints)):
            a = endpoints[ia]
            if degree[a] >= 2:
                continue
            ra = find(a)
            for ib in range(len(endpoints)):
                if ib == ia:
                    continue
                b = endpoints[ib]
                if degree[b] >= 2 or find(b) == ra:
                    continue
                parent[find(a)] = find(b)
                degree[a] += 1; degree[b] += 1
                adj[a].append(b); adj[b].append(a)
                joined = True
                break
            if joined:
                break
        if not joined:
            break        # baglanabilecek farkli-fragman cifti kalmadi
        endpoints = [i for i in range(n) if degree[i] < 2]
    if len(endpoints) == 2:
        a, b = endpoints
        adj[a].append(b); adj[b].append(a)
        degree[a] += 1; degree[b] += 1
    return _ge_walk(adj, n)


def quick_boruvka_tour(xs, ys, k: int = 10) -> list[int]:
    """Quick-Boruvka yaklaşımı (DIMACS güçlü kurucusu; Bentley 1992, Johnson &
    McGeoch DIMACS protokolünün standart 'Boruvka' kurucusunun k-NN'li hızlı
    karşılığı).

    Greedy-Edge kenarları GLOBAL uzunluk sırasıyla işlerken Borůvka TURLAR
    halinde çalışır: her turda HER parça (fragment) kendi en yakın komşu
    parçasına giden en kısa kenarını ÖNERİR, öneriler kısadan uzuna uygulanır
    (derece<=2 + union-find alt-tur yasağı, greedy_edge ile aynı sözleşme).
    Aday kenarlar grid_knn listelerinden gelir (tam O(n^2) tarama yok) ->
    büyük örneklerde de ölçeklenir. Kalan parçalar greedy_edge'in kanıtlanmış
    en-yakın-çift dikişiyle kapatılır ve derece-2 çizgesi tura yürünür."""
    n = len(xs)
    if n < 3:
        return list(range(n))
    hypot = math.hypot
    knn = grid_knn(xs, ys, min(k, n - 1))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    n_edges = 0

    # ---- Boruvka turları: her tur O(n*k); parça sayısı her turda en az
    #      yarıya indiğinden toplam O(n*k*log n) ----
    while n_edges < n - 1:
        check_construction_deadline()
        best_edge: dict[int, tuple[float, int, int]] = {}
        for i in range(n):
            if degree[i] >= 2:
                continue
            ri = find(i)
            for j in knn[i]:
                if degree[j] >= 2:
                    continue
                rj = find(j)
                if rj == ri:
                    continue
                d = hypot(xs[i] - xs[j], ys[i] - ys[j])
                cur = best_edge.get(ri)
                if cur is None or d < cur[0]:
                    best_edge[ri] = (d, i, j)
        if not best_edge:
            break   # k-NN listesi tükendi; kalanlar dikişle kapanır
        added = 0
        for d, i, j in sorted(best_edge.values()):
            if degree[i] >= 2 or degree[j] >= 2:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            parent[ri] = rj
            degree[i] += 1; degree[j] += 1
            adj[i].append(j); adj[j].append(i)
            n_edges += 1
            added += 1
        if added == 0:
            break

    # ---- dikiş: kalan parça uçlarını en-yakın-çift kuralıyla kapat
    #      (greedy_edge_tour'daki kanıtlanmış akışın birebir kopyası) ----
    endpoints = [i for i in range(n) if degree[i] < 2]
    if len(endpoints) > 2 and len(endpoints) <= _REPAIR_SORTED_PAIRS_MAX_E:
        import numpy as _np
        ep = endpoints
        ex = _np.array([xs[i] for i in ep])
        ey = _np.array([ys[i] for i in ep])
        pa, pb = _np.triu_indices(len(ep), k=1)
        du = _np.hypot(ex[pa] - ex[pb], ey[pa] - ey[pb])
        pa = pa.astype(_np.int32, copy=False)
        pb = pb.astype(_np.int32, copy=False)
        order = _np.argsort(du, kind="stable")
        del du
        frags_left = len({find(i) for i in ep})
        for t in range(order.size):
            if frags_left == 1:
                break
            if not (t & 0xFFFF):
                check_construction_deadline()
            a = ep[int(pa[order[t]])]
            b = ep[int(pb[order[t]])]
            if degree[a] >= 2 or degree[b] >= 2:
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            parent[ra] = rb
            degree[a] += 1; degree[b] += 1
            adj[a].append(b); adj[b].append(a)
            frags_left -= 1
        endpoints = [i for i in range(n) if degree[i] < 2]
    while len(endpoints) > 2:
        check_construction_deadline()
        best = None
        for a_idx in range(len(endpoints)):
            a = endpoints[a_idx]
            ra = find(a)
            for b_idx in range(a_idx + 1, len(endpoints)):
                b = endpoints[b_idx]
                if find(b) == ra:
                    continue
                d = hypot(xs[a] - xs[b], ys[a] - ys[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        parent[find(a)] = find(b)
        degree[a] += 1; degree[b] += 1
        adj[a].append(b); adj[b].append(a)
        endpoints = [i for i in range(n) if degree[i] < 2]
    if len(endpoints) == 2:
        a, b = endpoints
        adj[a].append(b); adj[b].append(a)
        degree[a] += 1; degree[b] += 1

    # derece-2 çizgeyi tura yürü
    tour = [0]
    prev, cur = -1, 0
    for _ in range(n - 1):
        a0, a1 = adj[cur][0], adj[cur][1]
        nxt = a0 if a0 != prev else a1
        tour.append(nxt)
        prev, cur = cur, nxt
    return tour


def grid_nn_tour(xs, ys, start: int = 0) -> list[int]:
    """Grid-accelerated nearest-neighbour construction tour (matrix-free)."""
    n = len(xs)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = (maxx - minx) or 1.0
    h = (maxy - miny) or 1.0
    g = max(1, int(math.sqrt(n)))
    inv_w = g / (w * (1.0 + 1e-9))
    inv_h = g / (h * (1.0 + 1e-9))
    cells: list[set] = [set() for _ in range(g * g)]
    cell_of = [0] * n
    for i in range(n):
        cx = min(g - 1, int((xs[i] - minx) * inv_w))
        cy = min(g - 1, int((ys[i] - miny) * inv_h))
        cell_of[i] = cy * g + cx
        cells[cy * g + cx].add(i)

    hypot = math.hypot
    used = bytearray(n)
    tour = [start]
    used[start] = 1
    cells[cell_of[start]].discard(start)
    cur = start
    for _ in range(n - 1):
        xi = xs[cur]; yi = ys[cur]
        cx = min(g - 1, int((xi - minx) * inv_w))
        cy = min(g - 1, int((yi - miny) * inv_h))
        best = -1; bd = float("inf")
        r = 0
        while True:
            x0 = max(0, cx - r); x1 = min(g - 1, cx + r)
            y0 = max(0, cy - r); y1 = min(g - 1, cy + r)
            for gy in range(y0, y1 + 1):
                on_y_edge = (gy == cy - r) or (gy == cy + r)
                row = gy * g
                gxs = range(x0, x1 + 1) if on_y_edge else (
                    [v for v in (cx - r, cx + r) if 0 <= v <= g - 1 and (r != 0 or v == cx)])
                for gx in gxs:
                    for j in cells[row + gx]:
                        d = hypot(xi - xs[j], yi - ys[j])
                        if d < bd:
                            bd = d; best = j
            # once found, expand one safety ring then stop
            if best >= 0:
                ring_min = r * min(w * inv_w, h * inv_h)  # not exact; do one extra ring
                if r >= 1:
                    break
            if x0 == 0 and y0 == 0 and x1 == g - 1 and y1 == g - 1:
                break
            r += 1
        if best < 0:
            for j in range(n):
                if not used[j]:
                    best = j; break
        used[best] = 1
        cells[cell_of[best]].discard(best)
        tour.append(best)
        cur = best
    return tour


# ======================================================================
#  Optimal snake-sweep rotation angle (theta-star)
#  Mirrors the dashboard's main-page theta-star scan exactly: rotate around the
#  centroid, build a boustrophedon (serpentine) snake in the rotated frame,
#  minimise the raw Euclidean tour length. This angle then seeds the benchmark
#  pipeline's construction stage so every dataset starts from its best sweep.
# ======================================================================
def rotate_coords(xs, ys, deg):
    """Rotates (xs, ys) around their centroid by `deg` degrees."""
    n = len(xs)
    cx = sum(xs) / n
    cy = sum(ys) / n
    a = math.radians(deg)
    ca = math.cos(a); sa = math.sin(a)
    rx = [0.0] * n; ry = [0.0] * n
    for i in range(n):
        dx = xs[i] - cx; dy = ys[i] - cy
        rx[i] = cx + dx * ca - dy * sa
        ry[i] = cy + dx * sa + dy * ca
    return rx, ry


def snake_order(rx, ry, strips: int = 0):
    """Boustrophedon order in the given frame: strips along x, serpentine in y.
    Matches `snakeTour` in index.html (default strips = round(sqrt(n/2)))."""
    n = len(rx)
    minx = min(rx); maxx = max(rx)
    k = strips if strips > 0 else max(1, round(math.sqrt(n / 2.0)))
    w = (maxx - minx) or 1.0
    buckets = [[] for _ in range(k)]
    inv = k / (w * (1.0 + 1e-9))
    for i in range(n):
        b = int((rx[i] - minx) * inv)
        if b >= k:
            b = k - 1
        elif b < 0:
            b = 0
        buckets[b].append(i)
    order = []
    for b in range(k):
        arr = buckets[b]
        arr.sort(key=lambda p: ry[p])
        if b % 2 == 1:
            arr.reverse()
        order.extend(arr)
    return order


# ======================================================================
#  Space-filling-curve constructors — same family as `snake_order` (strip)
#  and the Snake-Grid initializer: normalize coordinates onto an integer
#  grid, compute each point's 1-D position along the curve, sort by it.
#  O(n log n); no local search. Literature: Platzman & Bartholdi (1989),
#  "Spacefilling curves and the planar travelling salesman problem",
#  J. ACM 36(4) — the Hilbert-curve heuristic is the classic reference point
#  for this family; Morton/Z-order is the simpler bit-interleaved cousin.
# ======================================================================
def _normalize_to_grid(xs, ys, bits: int):
    """Maps (xs, ys) into [0, 2**bits - 1] integer coordinates, preserving
    relative order on each axis (needed for both curve constructions)."""
    side = (1 << bits) - 1
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = (maxx - minx) or 1.0
    h = (maxy - miny) or 1.0
    ix = [min(side, int((x - minx) / w * side)) for x in xs]
    iy = [min(side, int((y - miny) / h * side)) for y in ys]
    return ix, iy


def _hilbert_d(bits: int, x: int, y: int) -> int:
    """Maps a (x, y) grid cell to its 1-D distance along the Hilbert curve
    (standard xy2d bit-rotation algorithm; side is the FIXED square side
    length 2**bits -- the quadrant-flip in `rot()` mirrors around that fixed
    side, not around the shrinking per-level block size `s`)."""
    side = 1 << bits
    d = 0
    s = side >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # rotate/reflect the quadrant into the canonical orientation
        if ry == 0:
            if rx == 1:
                x = side - 1 - x
                y = side - 1 - y
            x, y = y, x
        s >>= 1
    return d


def hilbert_curve_tour(xs, ys, bits: int = 16) -> list[int]:
    """Hilbert-curve space-filling-curve construction (Platzman & Bartholdi
    1989): grid the points onto a 2**bits square, sort by Hilbert distance.
    Same family/complexity as `snake_order` (strip) and Snake-Grid, no local
    optimisation of cell size/axis/direction (that search is what Snake-Grid
    adds on top of this family)."""
    n = len(xs)
    if n <= 2:
        return list(range(n))
    ix, iy = _normalize_to_grid(xs, ys, bits)
    keyed = sorted(range(n), key=lambda i: (_hilbert_d(bits, ix[i], iy[i]), i))
    return keyed


def _morton_d(bits: int, x: int, y: int) -> int:
    """Interleaves the bits of x and y into a Z-order (Morton) code."""
    d = 0
    for b in range(bits):
        d |= ((x >> b) & 1) << (2 * b)
        d |= ((y >> b) & 1) << (2 * b + 1)
    return d


def morton_curve_tour(xs, ys, bits: int = 16) -> list[int]:
    """Morton/Z-order space-filling-curve construction: same grid-and-sort
    recipe as the Hilbert-curve tour above, cheaper bit-interleave key
    instead of the rotation-based Hilbert distance. Included as the
    simpler, lower-locality member of the same family for a same-class
    comparison against Snake-Grid."""
    n = len(xs)
    if n <= 2:
        return list(range(n))
    ix, iy = _normalize_to_grid(xs, ys, bits)
    keyed = sorted(range(n), key=lambda i: (_morton_d(bits, ix[i], iy[i]), i))
    return keyed


def _euclid_tour_len(order, xs, ys):
    hypot = math.hypot
    L = 0.0
    for i in range(1, len(order)):
        a = order[i - 1]; b = order[i]
        L += hypot(xs[a] - xs[b], ys[a] - ys[b])
    a = order[-1]; b = order[0]
    L += hypot(xs[a] - xs[b], ys[a] - ys[b])
    return L


def scan_optimal_angle(xs, ys, strips: int = 0):
    """Finds theta-star: the rotation (deg) minimising the snake-sweep tour
    length. Coarse 0..180 step 3 deg, then refine +/-3 deg step 0.5 deg (snake is
    180-periodic). Returns (theta_deg in [-90, 90], best_length)."""
    best_a = 0.0
    best_l = float("inf")

    def ev(deg):
        rx, ry = rotate_coords(xs, ys, deg)
        return _euclid_tour_len(snake_order(rx, ry, strips), rx, ry)

    a = 0.0
    while a < 180.0:
        L = ev(a)
        if L < best_l:
            best_l = L; best_a = a
        a += 3.0
    a = best_a - 3.0
    while a <= best_a + 3.0 + 1e-9:
        L = ev(a)
        if L < best_l:
            best_l = L; best_a = a
        a += 0.5
    norm = best_a
    while norm > 90.0:
        norm -= 180.0
    while norm < -90.0:
        norm += 180.0
    return norm, best_l


if __name__ == "__main__":
    # quick self-check
    for nm in ("kroA100", "ali535", "pcb3038"):
        h, c, e = parse_tsp(f"data/{nm}.tsp")
        inst, sparse = make_instance(c, e)
        print(f"{nm}: n={len(c)} ewt={e} sparse={sparse} d(0,1)={inst.distance_matrix[0][1]}")

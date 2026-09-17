"""Line Reassignment Optimizer for TSP tours.

Takes the output of AdaptiveSnakeWindowRepair as starting point.
For each point, measures a 'sensitivity' score — how much closer the point
is to a nearby edge (line segment between two consecutive tour cities)
compared to its own adjacent edges in the tour.

High-sensitivity points are candidates for relocation: remove the point
from its current position and reinsert it into the better neighbouring edge.
If the move shortens the total tour, the improvement is accepted and all
subsequent evaluations use the improved tour.

The process iterates up to a configurable number of rounds (default 500).
"""

from __future__ import annotations

import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from core import (
    AdaptiveSnakeWindowRepair,
    HistoryPoint,
    RunMetrics,
    SnakeGridPathInitializer,
    TSPDataLoader,
    TSPInstance,
    TSPRunResult,
    TSPVisualizer,
    TourFrame,
    TourInitializers,
    build_adaptive_window_repair_result,
    build_sampled_initialization_result,
    random_initialization,
    nearest_neighbor_initialization,
    snake_path_initialization_result,
    build_window_repair_result,
    tour_cost_from_matrix,
)


@dataclass(frozen=True)
class PointSensitivity:
    """Measures how 'misplaced' a point is in the current tour."""

    city_index: int
    tour_position: int
    own_edge_cost: float        # cost of edges currently using this point
    best_other_edge_index: int  # tour position of the best alternative edge
    best_other_insert_cost: float  # cost if inserted into that edge
    sensitivity: float          # own_edge_cost - best_other_insert_cost (positive = improvable)
    removal_gain: float = 0.0   # R(v) = d(p,v) + d(v,n) - d(p,n)
    local_deviation: float = 0.0  # S(v), geometric distance to segment p-n
    delta_total: float = 0.0    # Delta_remove + Delta_insert; negative is improving


@dataclass(frozen=True)
class LineReassignResult:
    """Stores the outcome of the line reassignment optimization."""

    tour: list[int]
    cost: float
    iterations_completed: int
    improvements_accepted: int
    total_candidates_evaluated: int
    total_saving: float
    frames: list[TourFrame]
    history: list[HistoryPoint]
    # --- diagnostics (defaults keep every existing caller/return valid) ---
    # Why the ILS loop stopped: "time" (wall-clock budget), "no_improve"
    # (max_no_improve consecutive non-improving iters), "max_iter" (iteration
    # cap reached), or "" when not an ILS run. Answers "neden bitti?".
    stop_reason: str = ""
    # Did the Phase-0 initial descent reach a true local optimum (a round with
    # zero accepted moves) or was it cut off by the round cap / deadline? None
    # for non-ILS runs. False => the curve was still descending at the stop
    # (i.e. it "platoya ulaşmadan durdu").
    phase0_converged: "Optional[bool]" = None
    # Perturbation iterations that actually ran a local search (0 => ILS never
    # got past the initial descent; it was a single descent, not "iterated").
    perturbation_iterations: int = 0


DEFAULT_ADAPTIVE_K_SCHEDULE: tuple[int, ...] = (15, 30, 60)
DEFAULT_ILS_TIME_LIMIT_SECONDS = 60.0

# When False, playback frames keep their metadata (cost/time/count) but DROP the
# per-move full-tour snapshot copy. The interactive dashboard needs the snapshots
# for animation; batch benchmarking does not, and on large instances storing one
# O(n) tour copy per accepted move costs gigabytes. Set to False from the runner.
RECORD_FRAME_TOURS = True

# Master switch for the optimizers' human-readable progress output (headers and
# per-iteration "\r" status lines). Batch benchmarking sets this False so hot
# loops don't waste time formatting strings that are discarded anyway.
VERBOSE = True


def _frame_tour(tour: list[int]) -> list[int]:
    """Returns a snapshot copy of the tour, or an empty list when frame-tour
    recording is disabled (keeps frame count/timing without the O(n) copy)."""
    return tour[:] if RECORD_FRAME_TOURS else []


# The in-pass deadline is a HARD safety cap, set to this multiple of the soft
# per-method time budget. The soft (between-iteration) check remains the primary
# stop, so normal runs are unaffected; the in-pass cap only fires when a single
# uninterruptible local-search descent would massively overrun (e.g. a Phase-0
# sweep on a very large instance).
_LS_DEADLINE_SAFETY = 1.5


def _past_deadline(deadline: Optional[float]) -> bool:
    """True when a wall-clock deadline (perf_counter timestamp) has passed.
    Lets a single local-search descent be interrupted mid-pass so a time budget
    is honored within seconds instead of only between outer iterations."""
    return deadline is not None and time.perf_counter() >= deadline


@dataclass(frozen=True)
class DenseAdaptiveLevel:
    """One dense-region search level for multi-point reassignment."""

    k_nearest_edges: int
    city_neighbor_limit: int
    dense_fraction: float
    max_batch_size: int


@dataclass(frozen=True)
class DenseBatchMove:
    """A relocate move selected from one dense-region sensitivity snapshot."""

    city_index: int
    remove_pos: int
    insert_after_pos: int
    insert_after_city: int
    insert_before_city: int
    delta_total: float
    removal_gain: float
    local_deviation: float
    density_radius: float


DEFAULT_DENSE_ADAPTIVE_LEVELS: tuple[DenseAdaptiveLevel, ...] = (
    DenseAdaptiveLevel(18, 64, 0.40, 2),
    DenseAdaptiveLevel(36, 120, 0.60, 3),
    DenseAdaptiveLevel(72, 240, 0.78, 4),
    DenseAdaptiveLevel(120, 400, 0.92, 5),
    DenseAdaptiveLevel(168, 640, 0.99, 6),
)


def _point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Euclidean distance from point (px,py) to line segment (ax,ay)-(bx,by).
    
    Used for geometric proximity detection, NOT for tour cost (which uses the
    instance distance matrix).
    """
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def compute_projected_coords(instance: TSPInstance) -> list[tuple[float, float]]:
    """Returns (x, y) projected coordinates for all cities using the SnakeGrid projector."""
    points = SnakeGridPathInitializer._project_points(instance)
    coords = [(0.0, 0.0)] * instance.size
    for p in points:
        coords[p.city_index] = (p.x, p.y)
    return coords


def _precompute_city_nearest_neighbors(
    instance: TSPInstance,
    neighbor_count: int,
) -> list[list[int]]:
    """Caches the nearest cities for each city to prune candidate edges."""

    dist_matrix = instance.distance_matrix
    size = instance.size
    take = min(neighbor_count + 1, size)
    city_nearest_neighbors: list[list[int]] = []
    for city_index in range(size):
        row = dist_matrix[city_index]
        # Partial selection (O(n log k)) instead of a full sort (O(n log n)) per
        # city. heapq.nsmallest is stable on ties (decorates with the original
        # index), so the neighbour lists are identical to sorted()[1:k+1]; the
        # first element is the city itself (distance 0) and is dropped.
        nearest = heapq.nsmallest(take, range(size), key=lambda other: row[other])
        city_nearest_neighbors.append(nearest[1 : neighbor_count + 1])
    return city_nearest_neighbors


def compute_sensitivities(
    tour: list[int],
    dist_matrix: list[list[float]],
    projected: list[tuple[float, float]],
    city_nearest_neighbors: list[list[int]],
    k_nearest_edges: int = 15,
    city_neighbor_limit: Optional[int] = None,
    candidate_city_mask: Optional[Sequence[bool]] = None,
) -> list[PointSensitivity]:
    """Computes sensitivity for every point in the tour.
    
    For each point P at tour position i:
    - own_edge_cost = dist(tour[i-1], P) + dist(P, tour[i+1])
    - removal_saving = own_edge_cost - dist(tour[i-1], tour[i+1])
      (what we save by skipping P)
    - For each candidate edge (tour[j], tour[j+1]) where j is NOT adjacent to i:
      insertion_cost = dist(tour[j], P) + dist(P, tour[j+1]) - dist(tour[j], tour[j+1])
    - sensitivity = removal_saving - best_insertion_cost
      (positive means net improvement)
    
    To avoid O(n^2) for every point, we use geometric proximity to find the
    k nearest edges.
    """
    n = len(tour)
    if n < 5:
        return []

    # Map city -> its position in the tour to easily find its edges
    max_city_id = max(tour) if tour else 0
    city_to_pos = [0] * (max_city_id + 1)
    for idx, c in enumerate(tour):
        city_to_pos[c] = idx

    sensitivities: list[PointSensitivity] = []
    
    for i in range(n):
        city = tour[i]
        if candidate_city_mask is not None and not candidate_city_mask[city]:
            continue
        prev_city = tour[(i - 1) % n]
        next_city = tour[(i + 1) % n]

        # Cost of current edges through this point
        own_edge_cost = dist_matrix[prev_city][city] + dist_matrix[city][next_city]
        # Cost of shortcut if we remove this point
        shortcut_cost = dist_matrix[prev_city][next_city]
        removal_saving = own_edge_cost - shortcut_cost

        # Find k nearest edges by geometric distance
        # Candidate edges: look at edges connected to the nearest cities
        px, py = projected[city]
        local_deviation = _point_to_segment_distance(
            px, py,
            projected[prev_city][0], projected[prev_city][1],
            projected[next_city][0], projected[next_city][1],
        )
        candidate_edges = set()
        stage_neighbors = city_nearest_neighbors[city]
        if city_neighbor_limit is not None:
            stage_neighbors = stage_neighbors[:city_neighbor_limit]
        for neighbor_city in stage_neighbors:
            pos = city_to_pos[neighbor_city]
            candidate_edges.add(pos)             # Edge starting at neighbor
            candidate_edges.add((pos - 1) % n)   # Edge ending at neighbor

        edge_dists: list[tuple[float, int]] = []
        for j in candidate_edges:
            # Skip edges adjacent to position i (they share the point)
            if j == i or j == (i - 1) % n or j == (i + 1) % n:
                continue
            # Also skip edge at (i-2) and (i+1) to avoid near-degenerate moves
            if j == (i - 2) % n or j == (i + 2) % n:
                continue
            
            a_idx = tour[j]
            b_idx = tour[(j + 1) % n]
            seg_dist = _point_to_segment_distance(
                px, py,
                projected[a_idx][0], projected[a_idx][1],
                projected[b_idx][0], projected[b_idx][1],
            )
            edge_dists.append((seg_dist, j))
        
        if not edge_dists:
            continue

        # Sort by geometric distance and take top-k
        edge_dists.sort()
        candidates = edge_dists[:k_nearest_edges]

        best_insertion_cost = float("inf")
        best_edge_idx = -1

        for _, j in candidates:
            a_idx = tour[j]
            b_idx = tour[(j + 1) % n]
            insert_cost = (
                dist_matrix[a_idx][city]
                + dist_matrix[city][b_idx]
                - dist_matrix[a_idx][b_idx]
            )
            if insert_cost < best_insertion_cost:
                best_insertion_cost = insert_cost
                best_edge_idx = j

        if best_edge_idx < 0:
            continue

        sensitivity = removal_saving - best_insertion_cost
        delta_total = best_insertion_cost - removal_saving

        sensitivities.append(PointSensitivity(
            city_index=city,
            tour_position=i,
            own_edge_cost=own_edge_cost,
            best_other_edge_index=best_edge_idx,
            best_other_insert_cost=best_insertion_cost,
            sensitivity=sensitivity,
            removal_gain=removal_saving,
            local_deviation=local_deviation,
            delta_total=delta_total,
        ))

    return sensitivities


def relocate_point(
    tour: list[int],
    remove_pos: int,
    insert_after_pos: int,
) -> list[int]:
    """Removes the city at remove_pos and inserts it after insert_after_pos.
    
    Returns a new tour list. Positions are tour indices.
    The insert_after_pos refers to position BEFORE the removal.
    """
    n = len(tour)
    city = tour[remove_pos]
    
    # Build new tour by removing and reinserting
    new_tour = tour[:remove_pos] + tour[remove_pos + 1:]
    
    # Adjust insert position for the removal
    if insert_after_pos > remove_pos:
        adjusted_pos = insert_after_pos - 1
    else:
        adjusted_pos = insert_after_pos
    
    # Insert after the adjusted position
    insert_idx = adjusted_pos + 1
    new_tour = new_tour[:insert_idx] + [city] + new_tour[insert_idx:]

    return new_tour


def _append_line_reassign_frame(
    frames: list[TourFrame],
    tour: list[int],
    cost: float,
    best_cost: float,
    elapsed_time: float,
    phase: str,
    stage_index: int,
    constructed_cities: int,
) -> None:
    """Appends a playback frame for a line-reassignment event."""

    frames.append(
        TourFrame(
            step=len(frames),
            phase=phase,
            stage_index=stage_index,
            elapsed_time=elapsed_time,
            cost=cost,
            best_cost=best_cost,
            constructed_cities=constructed_cities,
            tour=_frame_tour(tour),
        )
    )


def _try_best_reassignment_move(
    tour: list[int],
    current_cost: float,
    dist_matrix: list[list[float]],
    projected: list[tuple[float, float]],
    city_nearest_neighbors: list[list[int]],
    k_nearest_edges: int,
    min_sensitivity_threshold: float,
    city_neighbor_limit: int,
    priority_mode: str = "saving",
) -> tuple[int, Optional[tuple[list[int], float]]]:
    """Finds and evaluates the best improving relocate move under the active budget."""

    sensitivities = compute_sensitivities(
        tour,
        dist_matrix,
        projected,
        city_nearest_neighbors,
        k_nearest_edges,
        city_neighbor_limit=city_neighbor_limit,
    )
    candidates = [
        sensitivity
        for sensitivity in sensitivities
        if sensitivity.delta_total < -min_sensitivity_threshold
    ]
    if not candidates:
        return len(sensitivities), None

    if priority_mode == "node_geometry":
        candidates.sort(
            key=lambda sensitivity: (
                sensitivity.removal_gain,
                sensitivity.local_deviation,
                -sensitivity.delta_total,
            ),
            reverse=True,
        )
    else:
        candidates.sort(key=lambda sensitivity: sensitivity.sensitivity, reverse=True)

    tour_size = len(tour)
    for candidate in candidates:
        remove_pos = candidate.tour_position
        city = tour[remove_pos]
        prev_city = tour[(remove_pos - 1) % tour_size]
        next_city = tour[(remove_pos + 1) % tour_size]
        delta_remove = (
            dist_matrix[prev_city][next_city]
            - dist_matrix[prev_city][city]
            - dist_matrix[city][next_city]
        )

        insert_after_pos = candidate.best_other_edge_index
        if (
            insert_after_pos == remove_pos
            or insert_after_pos == (remove_pos - 1) % tour_size
            or insert_after_pos == (remove_pos + 1) % tour_size
        ):
            continue

        left_city = tour[insert_after_pos]
        right_city = tour[(insert_after_pos + 1) % tour_size]
        insertion_cost = (
            dist_matrix[left_city][city]
            + dist_matrix[city][right_city]
            - dist_matrix[left_city][right_city]
        )
        delta_total = delta_remove + insertion_cost
        if delta_total >= -min_sensitivity_threshold:
            continue

        # Incremental cost update: delta_total is the EXACT change of this
        # relocate (the insertion edge is guaranteed disjoint from the removed
        # edges by the position checks above), so we avoid the O(n) full recompute.
        new_tour = relocate_point(tour, remove_pos, insert_after_pos)
        new_cost = current_cost + delta_total
        if new_cost < current_cost - 1e-9:
            return len(sensitivities), (new_tour, new_cost)

    return len(sensitivities), None


def _normalize_dense_adaptive_levels(
    levels: Sequence[DenseAdaptiveLevel | tuple[int, int, float, int]],
    city_count: int,
) -> tuple[DenseAdaptiveLevel, ...]:
    """Validates dense adaptive levels and caps neighbor budgets to the instance size."""

    normalized: list[DenseAdaptiveLevel] = []
    max_neighbors = max(1, city_count - 1)
    for raw_level in levels:
        if isinstance(raw_level, DenseAdaptiveLevel):
            raw_k = raw_level.k_nearest_edges
            raw_neighbor_limit = raw_level.city_neighbor_limit
            raw_dense_fraction = raw_level.dense_fraction
            raw_batch_size = raw_level.max_batch_size
        else:
            raw_k, raw_neighbor_limit, raw_dense_fraction, raw_batch_size = raw_level

        k_nearest_edges = max(1, int(raw_k))
        city_neighbor_limit = min(max_neighbors, max(k_nearest_edges, int(raw_neighbor_limit)))
        dense_fraction = min(1.0, max(0.01, float(raw_dense_fraction)))
        max_batch_size = max(1, int(raw_batch_size))
        normalized.append(
            DenseAdaptiveLevel(
                k_nearest_edges=k_nearest_edges,
                city_neighbor_limit=city_neighbor_limit,
                dense_fraction=dense_fraction,
                max_batch_size=max_batch_size,
            )
        )

    if not normalized:
        raise ValueError("Dense adaptive levels must contain at least one k:neighbors:fraction:batch tuple.")

    return tuple(normalized)


def _compute_density_radii(
    dist_matrix: list[list[float]],
    city_nearest_neighbors: list[list[int]],
    density_neighbor_count: int,
) -> list[float]:
    """Returns each city's average distance to its nearest k cities; lower means denser."""

    density_radii: list[float] = []
    for city_index, neighbors in enumerate(city_nearest_neighbors):
        local_neighbors = neighbors[:density_neighbor_count]
        if not local_neighbors:
            density_radii.append(float("inf"))
            continue
        density_radii.append(
            sum(dist_matrix[city_index][neighbor] for neighbor in local_neighbors)
            / len(local_neighbors)
        )
    return density_radii


def _dense_city_mask(
    density_radii: Sequence[float],
    dense_fraction: float,
) -> tuple[list[bool], float, int]:
    """Builds a mask for the densest fraction of cities."""

    if not density_radii:
        return [], float("inf"), 0

    sorted_radii = sorted(density_radii)
    dense_count = max(1, min(len(sorted_radii), math.ceil(len(sorted_radii) * dense_fraction)))
    threshold = sorted_radii[dense_count - 1]
    mask = [radius <= threshold for radius in density_radii]
    return mask, threshold, sum(1 for included in mask if included)


def _select_dense_batch_moves(
    tour: list[int],
    dist_matrix: list[list[float]],
    projected: list[tuple[float, float]],
    city_nearest_neighbors: list[list[int]],
    density_radii: Sequence[float],
    dense_mask: Sequence[bool],
    level: DenseAdaptiveLevel,
    min_sensitivity_threshold: float,
) -> tuple[list[DenseBatchMove], int]:
    """Selects non-overlapping improving moves from one dense-region snapshot."""

    sensitivities = compute_sensitivities(
        tour,
        dist_matrix,
        projected,
        city_nearest_neighbors,
        level.k_nearest_edges,
        city_neighbor_limit=level.city_neighbor_limit,
        candidate_city_mask=dense_mask,
    )
    candidates = [
        sensitivity
        for sensitivity in sensitivities
        if sensitivity.delta_total < -min_sensitivity_threshold
    ]
    if not candidates:
        return [], len(sensitivities)

    candidates.sort(
        key=lambda sensitivity: (
            -sensitivity.delta_total,
            -density_radii[sensitivity.city_index],
            sensitivity.removal_gain,
            sensitivity.local_deviation,
        ),
        reverse=True,
    )

    selected: list[DenseBatchMove] = []
    blocked_remove_positions: set[int] = set()
    blocked_edge_positions: set[int] = set()
    tour_size = len(tour)

    for candidate in candidates:
        remove_pos = candidate.tour_position
        insert_after_pos = candidate.best_other_edge_index
        if remove_pos in blocked_remove_positions or insert_after_pos in blocked_edge_positions:
            continue
        if (insert_after_pos + 1) % tour_size in blocked_edge_positions:
            continue

        selected.append(
            DenseBatchMove(
                city_index=candidate.city_index,
                remove_pos=remove_pos,
                insert_after_pos=insert_after_pos,
                insert_after_city=tour[insert_after_pos],
                insert_before_city=tour[(insert_after_pos + 1) % tour_size],
                delta_total=candidate.delta_total,
                removal_gain=candidate.removal_gain,
                local_deviation=candidate.local_deviation,
                density_radius=density_radii[candidate.city_index],
            )
        )

        for offset in (-2, -1, 0, 1, 2):
            blocked_remove_positions.add((remove_pos + offset) % tour_size)
            blocked_edge_positions.add((insert_after_pos + offset) % tour_size)

        if len(selected) >= level.max_batch_size:
            break

    return selected, len(sensitivities)


def _apply_dense_batch_moves(
    tour: list[int],
    current_cost: float,
    dist_matrix: list[list[float]],
    moves: Sequence[DenseBatchMove],
    min_sensitivity_threshold: float,
) -> tuple[list[int], float, int]:
    """Applies a selected dense batch with exact delta checks after each move."""

    working_tour = list(tour)
    working_cost = current_cost
    accepted_moves = 0

    for move in moves:
        city_to_pos = {city: pos for pos, city in enumerate(working_tour)}
        remove_pos = city_to_pos.get(move.city_index)
        insert_after_pos = city_to_pos.get(move.insert_after_city)
        if remove_pos is None or insert_after_pos is None:
            continue

        tour_size = len(working_tour)
        if working_tour[(insert_after_pos + 1) % tour_size] != move.insert_before_city:
            continue
        if (
            insert_after_pos == remove_pos
            or insert_after_pos == (remove_pos - 1) % tour_size
            or insert_after_pos == (remove_pos + 1) % tour_size
        ):
            continue

        city = working_tour[remove_pos]
        prev_city = working_tour[(remove_pos - 1) % tour_size]
        next_city = working_tour[(remove_pos + 1) % tour_size]
        left_city = working_tour[insert_after_pos]
        right_city = working_tour[(insert_after_pos + 1) % tour_size]
        delta_remove = (
            dist_matrix[prev_city][next_city]
            - dist_matrix[prev_city][city]
            - dist_matrix[city][next_city]
        )
        delta_insert = (
            dist_matrix[left_city][city]
            + dist_matrix[city][right_city]
            - dist_matrix[left_city][right_city]
        )
        delta_total = delta_remove + delta_insert
        if delta_total >= -min_sensitivity_threshold:
            continue

        new_tour = relocate_point(working_tour, remove_pos, insert_after_pos)
        new_cost = tour_cost_from_matrix(new_tour, dist_matrix)
        if new_cost < working_cost - 1e-9:
            working_tour = new_tour
            working_cost = new_cost
            accepted_moves += 1

    return working_tour, working_cost, accepted_moves


def run_line_reassignment(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 1000,
    k_nearest_edges: int = 15,
    min_sensitivity_threshold: float = 1e-6,
    deadline: Optional[float] = None,
) -> LineReassignResult:
    """Runs the fixed-k line reassignment optimizer.

    `deadline` (perf_counter zaman damgası) verilirse iterasyon döngüsü bu
    duvar-saati sınırını geçince nazikçe durur -- uzun örneklerde koşucunun
    süresiz takılmasını önler (sonuç o ana kadarki en iyi turdur)."""

    return _run_line_reassignment_schedule(
        instance=instance,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        k_nearest_schedule=(k_nearest_edges,),
        min_sensitivity_threshold=min_sensitivity_threshold,
        run_label="Line Reassignment Optimizer",
        progress_label="Line Reassign",
        deadline=deadline,
    )


def _run_line_reassignment_schedule(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int,
    k_nearest_schedule: Sequence[int],
    min_sensitivity_threshold: float,
    run_label: str,
    progress_label: str,
    priority_mode: str = "saving",
    deadline: Optional[float] = None,
) -> LineReassignResult:
    """Runs one or more line-reassignment stages on top of the same tour."""

    schedule = tuple(max(1, int(value)) for value in k_nearest_schedule)
    if not schedule:
        raise ValueError("k_nearest_schedule must contain at least one positive value.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    instance.validate_tour(initial_tour)
    projected = compute_projected_coords(instance)
    dist_matrix = instance.distance_matrix

    working_tour = list(initial_tour)
    current_cost = tour_cost_from_matrix(working_tour, dist_matrix)
    initial_cost = current_cost
    best_cost = current_cost

    n_cities = len(working_tour)
    max_neighbor_limit = min(n_cities - 1, max(50, max(schedule) * 2))
    stage_neighbor_limits = {
        k_value: min(max_neighbor_limit, max(50, k_value * 2))
        for k_value in schedule
    }

    print(f"\n{'=' * 55}")
    print(f"[{time.strftime('%H:%M:%S')}] {run_label.upper()} STARTED")
    print(f"{'=' * 55}")
    print("Search parameters:")
    print(f"  > max_iterations           : {max_iterations}")
    print(
        "  > k_nearest_edges schedule : "
        + " -> ".join(str(value) for value in schedule)
    )
    print(
        "  > stage neighbor limits    : "
        + " -> ".join(str(stage_neighbor_limits[value]) for value in schedule)
    )
    print(f"  > candidate priority       : {priority_mode}")
    print(f"  > min_sensitivity_threshold: {min_sensitivity_threshold}")
    print(f"{'-' * 55}")

    t0 = time.perf_counter()
    city_nearest_neighbors = _precompute_city_nearest_neighbors(instance, max_neighbor_limit)
    t1 = time.perf_counter()
    sys.stdout.write(
        f"  [Info] Cached nearest-neighbor lists up to {max_neighbor_limit} cities in {t1 - t0:.2f}s.\n"
    )

    start_time = time.perf_counter()
    iterations_completed = 0
    improvements_accepted = 0
    total_candidates = 0

    frames: list[TourFrame] = [
        TourFrame(
            step=0,
            phase="initial",
            stage_index=0,
            elapsed_time=0.0,
            cost=current_cost,
            best_cost=best_cost,
            constructed_cities=0,
            tour=working_tour[:],
        )
    ]
    history: list[HistoryPoint] = [
        HistoryPoint(iteration=0, elapsed_time=0.0, cost=best_cost)
    ]

    stage_count = len(schedule)
    for stage_number, k_nearest_edges in enumerate(schedule, start=1):
        if iterations_completed >= max_iterations or _past_deadline(deadline):
            break

        stage_neighbor_limit = stage_neighbor_limits[k_nearest_edges]

        sys.stdout.write(
            f"\n  [Stage {stage_number}/{stage_count}] Searching with k={k_nearest_edges} and neighbor_limit={stage_neighbor_limit}.\n"
        )
        sys.stdout.flush()
        stage_improvements = 0

        while iterations_completed < max_iterations and not _past_deadline(deadline):
            iterations_completed += 1
            if VERBOSE:
                elapsed_so_far = time.perf_counter() - start_time
                avg_iter_time = elapsed_so_far / iterations_completed
                eta = avg_iter_time * (max_iterations - iterations_completed)
                sys.stdout.write(
                    "\r"
                    f"[{progress_label}] Stage {stage_number}/{stage_count} | "
                    f"k={k_nearest_edges} | Iter: {iterations_completed}/{max_iterations} | "
                    f"Cost: {current_cost:.2f} | Accepted: {improvements_accepted} | "
                    f"Elapsed: {elapsed_so_far:.1f}s | ETA: {eta:.1f}s   "
                )
                sys.stdout.flush()

            candidates_evaluated, improvement = _try_best_reassignment_move(
                tour=working_tour,
                current_cost=current_cost,
                dist_matrix=dist_matrix,
                projected=projected,
                city_nearest_neighbors=city_nearest_neighbors,
                k_nearest_edges=k_nearest_edges,
                min_sensitivity_threshold=min_sensitivity_threshold,
                city_neighbor_limit=stage_neighbor_limit,
                priority_mode=priority_mode,
            )
            total_candidates += candidates_evaluated
            if improvement is None:
                break

            working_tour, current_cost = improvement
            best_cost = min(best_cost, current_cost)
            improvements_accepted += 1
            stage_improvements += 1

            elapsed = time.perf_counter() - start_time
            _append_line_reassign_frame(
                frames=frames,
                tour=working_tour,
                cost=current_cost,
                best_cost=best_cost,
                elapsed_time=elapsed,
                phase="repair",
                stage_index=improvements_accepted,
                constructed_cities=1,
            )
            history.append(
                HistoryPoint(
                    iteration=improvements_accepted,
                    elapsed_time=elapsed,
                    cost=best_cost,
                )
            )

        sys.stdout.write("\n")
        if stage_number < stage_count and iterations_completed < max_iterations:
            next_k = schedule[stage_number]
            sys.stdout.write(
                f"  [Stage {stage_number}] No more improving moves for k={k_nearest_edges}. "
                f"Expanding to k={next_k}.\n"
            )
            if stage_improvements:
                sys.stdout.write(
                    f"  [Stage {stage_number}] Accepted {stage_improvements} moves before convergence.\n"
                )

    elapsed = time.perf_counter() - start_time
    _append_line_reassign_frame(
        frames=frames,
        tour=working_tour,
        cost=current_cost,
        best_cost=best_cost,
        elapsed_time=elapsed,
        phase="final",
        stage_index=improvements_accepted,
        constructed_cities=0,
    )
    history.append(
        HistoryPoint(
            iteration=improvements_accepted + 1,
            elapsed_time=elapsed,
            cost=best_cost,
        )
    )

    return LineReassignResult(
        tour=working_tour,
        cost=current_cost,
        iterations_completed=iterations_completed,
        improvements_accepted=improvements_accepted,
        total_candidates_evaluated=total_candidates,
        total_saving=initial_cost - current_cost,
        frames=frames,
        history=history,
    )


def run_dense_adaptive_line_reassignment(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 1000,
    search_levels: Sequence[DenseAdaptiveLevel | tuple[int, int, float, int]] = DEFAULT_DENSE_ADAPTIVE_LEVELS,
    density_neighbor_count: int = 10,
    min_sensitivity_threshold: float = 1e-6,
) -> LineReassignResult:
    """Runs dense-region adaptive reassignment with multi-point batches."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if density_neighbor_count <= 0:
        raise ValueError("density_neighbor_count must be positive.")

    instance.validate_tour(initial_tour)
    projected = compute_projected_coords(instance)
    dist_matrix = instance.distance_matrix

    working_tour = list(initial_tour)
    current_cost = tour_cost_from_matrix(working_tour, dist_matrix)
    initial_cost = current_cost
    best_cost = current_cost

    levels = _normalize_dense_adaptive_levels(search_levels, len(working_tour))
    max_neighbor_limit = max(
        density_neighbor_count,
        max(level.city_neighbor_limit for level in levels),
    )

    print(f"\n{'=' * 55}")
    print(f"[{time.strftime('%H:%M:%S')}] DENSE ADAPTIVE LINE REASSIGNMENT STARTED")
    print(f"{'=' * 55}")
    print("Search parameters:")
    print(f"  > max_iterations           : {max_iterations}")
    print(
        "  > dense levels            : "
        + " -> ".join(
            "k="
            f"{level.k_nearest_edges}/neighbors={level.city_neighbor_limit}"
            f"/dense={level.dense_fraction:.2f}/batch={level.max_batch_size}"
            for level in levels
        )
    )
    print(f"  > density_neighbor_count  : {density_neighbor_count}")
    print(f"  > min_sensitivity_threshold: {min_sensitivity_threshold}")
    print(f"{'-' * 55}")

    t0 = time.perf_counter()
    city_nearest_neighbors = _precompute_city_nearest_neighbors(instance, max_neighbor_limit)
    density_radii = _compute_density_radii(
        dist_matrix,
        city_nearest_neighbors,
        density_neighbor_count,
    )
    t1 = time.perf_counter()
    sys.stdout.write(
        f"  [Info] Cached nearest-neighbor and density lists up to {max_neighbor_limit} cities in {t1 - t0:.2f}s.\n"
    )

    start_time = time.perf_counter()
    iterations_completed = 0
    improvements_accepted = 0
    total_candidates = 0
    level_index = 0

    frames: list[TourFrame] = [
        TourFrame(
            step=0,
            phase="initial",
            stage_index=0,
            elapsed_time=0.0,
            cost=current_cost,
            best_cost=best_cost,
            constructed_cities=0,
            tour=working_tour[:],
        )
    ]
    history: list[HistoryPoint] = [
        HistoryPoint(iteration=0, elapsed_time=0.0, cost=best_cost)
    ]

    while iterations_completed < max_iterations:
        level = levels[level_index]
        dense_mask, density_threshold, dense_count = _dense_city_mask(
            density_radii,
            level.dense_fraction,
        )
        iterations_completed += 1

        if VERBOSE:
            elapsed_so_far = time.perf_counter() - start_time
            avg_iter_time = elapsed_so_far / iterations_completed
            eta = avg_iter_time * (max_iterations - iterations_completed)
            sys.stdout.write(
                "\r"
                f"[Dense Adaptive Reassign] Level {level_index + 1}/{len(levels)} | "
                f"k={level.k_nearest_edges} | dense={dense_count} | "
                f"batch<={level.max_batch_size} | Iter: {iterations_completed}/{max_iterations} | "
                f"Cost: {current_cost:.2f} | Accepted moves: {improvements_accepted} | "
                f"Elapsed: {elapsed_so_far:.1f}s | ETA: {eta:.1f}s   "
            )
            sys.stdout.flush()

        moves, candidates_evaluated = _select_dense_batch_moves(
            tour=working_tour,
            dist_matrix=dist_matrix,
            projected=projected,
            city_nearest_neighbors=city_nearest_neighbors,
            density_radii=density_radii,
            dense_mask=dense_mask,
            level=level,
            min_sensitivity_threshold=min_sensitivity_threshold,
        )
        total_candidates += candidates_evaluated

        if moves:
            new_tour, new_cost, accepted_count = _apply_dense_batch_moves(
                tour=working_tour,
                current_cost=current_cost,
                dist_matrix=dist_matrix,
                moves=moves,
                min_sensitivity_threshold=min_sensitivity_threshold,
            )
            if accepted_count:
                working_tour = new_tour
                current_cost = new_cost
                best_cost = min(best_cost, current_cost)
                improvements_accepted += accepted_count

                elapsed = time.perf_counter() - start_time
                _append_line_reassign_frame(
                    frames=frames,
                    tour=working_tour,
                    cost=current_cost,
                    best_cost=best_cost,
                    elapsed_time=elapsed,
                    phase="repair",
                    stage_index=improvements_accepted,
                    constructed_cities=accepted_count,
                )
                history.append(
                    HistoryPoint(
                        iteration=improvements_accepted,
                        elapsed_time=elapsed,
                        cost=best_cost,
                    )
                )
                level_index = 0
                continue

        if level_index + 1 < len(levels):
            next_level = levels[level_index + 1]
            sys.stdout.write(
                f"\n  [Dense] No dense batch improvement at level {level_index + 1} "
                f"(threshold={density_threshold:.4f}). Expanding to "
                f"k={next_level.k_nearest_edges}, dense={next_level.dense_fraction:.2f}.\n"
            )
            level_index += 1
            continue

        sys.stdout.write("\n  [Dense] All dense adaptive levels stalled.\n")
        break

    sys.stdout.write("\n")
    elapsed = time.perf_counter() - start_time
    _append_line_reassign_frame(
        frames=frames,
        tour=working_tour,
        cost=current_cost,
        best_cost=best_cost,
        elapsed_time=elapsed,
        phase="final",
        stage_index=improvements_accepted,
        constructed_cities=0,
    )
    history.append(
        HistoryPoint(
            iteration=improvements_accepted + 1,
            elapsed_time=elapsed,
            cost=best_cost,
        )
    )

    return LineReassignResult(
        tour=working_tour,
        cost=current_cost,
        iterations_completed=iterations_completed,
        improvements_accepted=improvements_accepted,
        total_candidates_evaluated=total_candidates,
        total_saving=initial_cost - current_cost,
        frames=frames,
        history=history,
    )


def run_adaptive_line_reassignment(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 1000,
    k_nearest_schedule: Sequence[int] = DEFAULT_ADAPTIVE_K_SCHEDULE,
    min_sensitivity_threshold: float = 1e-6,
) -> LineReassignResult:
    """Runs the adaptive line reassignment optimizer with an expanding k schedule."""

    return _run_line_reassignment_schedule(
        instance=instance,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        k_nearest_schedule=k_nearest_schedule,
        min_sensitivity_threshold=min_sensitivity_threshold,
        run_label="Adaptive Line Reassignment Optimizer",
        progress_label="Adaptive Line Reassign",
        priority_mode="node_geometry",
    )


def build_line_reassign_result(
    instance: TSPInstance,
    adaptive_tour: list[int],
    max_iterations: int = 50,
    k_nearest_edges: int = 15,
    deadline: Optional[float] = None,
) -> tuple[TSPRunResult, LineReassignResult]:
    """Builds a dashboard-compatible result for the line reassignment optimizer."""

    result = run_line_reassignment(
        instance=instance,
        initial_tour=adaptive_tour,
        max_iterations=max_iterations,
        k_nearest_edges=k_nearest_edges,
        deadline=deadline,
    )

    metrics = RunMetrics(
        initialization_name="Line Reassignment Optimizer",
        initial_cost=instance.tour_cost(adaptive_tour),
        final_cost=result.cost,
        cpu_time_seconds=result.frames[-1].elapsed_time,
        frame_count=len(result.frames),
        history=result.history,
    )

    return (
        TSPRunResult(
            initial_tour=list(adaptive_tour),
            final_tour=result.tour,
            metrics=metrics,
            animation_frames=result.frames,
        ),
        result,
    )


def build_adaptive_line_reassign_result(
    instance: TSPInstance,
    adaptive_tour: list[int],
    max_iterations: int = 50,
    k_nearest_schedule: Sequence[int] = DEFAULT_ADAPTIVE_K_SCHEDULE,
) -> tuple[TSPRunResult, LineReassignResult]:
    """Builds a dashboard-compatible result for the adaptive line reassignment optimizer."""

    result = run_adaptive_line_reassignment(
        instance=instance,
        initial_tour=adaptive_tour,
        max_iterations=max_iterations,
        k_nearest_schedule=k_nearest_schedule,
    )

    metrics = RunMetrics(
        initialization_name="Adaptive Line Reassignment Optimizer",
        initial_cost=instance.tour_cost(adaptive_tour),
        final_cost=result.cost,
        cpu_time_seconds=result.frames[-1].elapsed_time,
        frame_count=len(result.frames),
        history=result.history,
    )

    return (
        TSPRunResult(
            initial_tour=list(adaptive_tour),
            final_tour=result.tour,
            metrics=metrics,
            animation_frames=result.frames,
        ),
        result,
    )


def build_dense_adaptive_line_reassign_result(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 50,
    search_levels: Sequence[DenseAdaptiveLevel | tuple[int, int, float, int]] = DEFAULT_DENSE_ADAPTIVE_LEVELS,
    density_neighbor_count: int = 10,
) -> tuple[TSPRunResult, LineReassignResult]:
    """Builds a dashboard-compatible result for dense adaptive reassignment."""

    result = run_dense_adaptive_line_reassignment(
        instance=instance,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        search_levels=search_levels,
        density_neighbor_count=density_neighbor_count,
    )

    metrics = RunMetrics(
        initialization_name="Dense Adaptive Line Reassignment Optimizer",
        initial_cost=instance.tour_cost(initial_tour),
        final_cost=result.cost,
        cpu_time_seconds=result.frames[-1].elapsed_time,
        frame_count=len(result.frames),
        history=result.history,
    )

    return (
        TSPRunResult(
            initial_tour=list(initial_tour),
            final_tour=result.tour,
            metrics=metrics,
            animation_frames=result.frames,
        ),
        result,
    )


# ---------------------------------------------------------------------------
#  Iterated Local Search (ILS) Metaheuristic Optimizer
# ---------------------------------------------------------------------------


def _two_opt_delta(
    tour: list[int],
    dist_matrix: list[list[float]],
    i: int,
    j: int,
) -> float:
    """Computes the cost delta of reversing the segment tour[i+1..j]."""
    n = len(tour)
    a = tour[i]
    b = tour[(i + 1) % n]
    c = tour[j]
    d = tour[(j + 1) % n]
    return (
        dist_matrix[a][c] + dist_matrix[b][d]
        - dist_matrix[a][b] - dist_matrix[c][d]
    )


def _apply_two_opt(tour: list[int], i: int, j: int) -> list[int]:
    """Returns a new tour with the segment tour[i+1..j] reversed."""
    new_tour = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
    return new_tour


def _two_opt_pass(
    tour: list[int],
    current_cost: float,
    dist_matrix: list[list[float]],
    city_nearest_neighbors: list[list[int]],
    neighbor_limit: int,
    deadline: Optional[float] = None,
) -> tuple[list[int], float, int]:
    """Performs one full pass of neighbor-list accelerated 2-opt.

    Uses nearest-neighbor lists to prune the search space. Only considers
    edge (i, i+1) paired with edges touching cities near tour[i].
    Returns the improved tour, cost, and count of accepted moves.
    """
    n = len(tour)
    total_accepted = 0
    working_tour = list(tour)
    working_cost = current_cost

    max_city_id = max(working_tour)
    city_to_pos = [0] * (max_city_id + 1)
    for pos, city in enumerate(working_tour):
        city_to_pos[city] = pos

    # Don't-look bits: skip a city until one of its tour edges changes. This
    # replaces the previous "restart the whole scan after every accepted move"
    # (which was O(moves * n * k)); the candidate moves and acceptance rule are
    # unchanged, only the scan bookkeeping is.
    dont_look = bytearray(max_city_id + 1)  # 0 = active, 1 = skip
    active = True
    while active:
        if _past_deadline(deadline):
            break
        active = False
        for scan in range(n):
            city_i = working_tour[scan]
            if dont_look[city_i]:
                continue
            i = city_to_pos[city_i]
            found = False
            for neighbor_city in city_nearest_neighbors[city_i][:neighbor_limit]:
                j = city_to_pos[neighbor_city]
                # Single orientation (j after i), matching the original move set;
                # a full DLB sweep still covers both directions because the
                # mirror move is examined when the neighbour is the anchor.
                if j <= i + 1 or (i == 0 and j == n - 1):
                    continue
                a, b = i, j
                delta = _two_opt_delta(working_tour, dist_matrix, a, b)
                if delta < -1e-9:
                    working_tour = _apply_two_opt(working_tour, a, b)
                    working_cost += delta
                    total_accepted += 1
                    # Update positions for the reversed segment a+1..b
                    for pos in range(a + 1, b + 1):
                        city_to_pos[working_tour[pos]] = pos
                    # Reactivate the four cities whose tour edges changed
                    for c in (working_tour[a], working_tour[a + 1],
                              working_tour[b], working_tour[(b + 1) % n]):
                        dont_look[c] = 0
                    found = True
                    active = True
                    # Honor the time budget mid-pass (cold starts do thousands
                    # of moves in a single sweep).
                    if (total_accepted & 511) == 0 and _past_deadline(deadline):
                        return working_tour, working_cost, total_accepted
                    break
            if not found:
                dont_look[city_i] = 1

    return working_tour, working_cost, total_accepted


def _or_opt_delta(
    tour: list[int],
    dist_matrix: list[list[float]],
    seg_start: int,
    seg_len: int,
    insert_after: int,
) -> float:
    """Computes cost delta for relocating a segment of seg_len cities.

    The segment starts at position seg_start and has length seg_len.
    It is removed from its current location and inserted after position
    insert_after (in the original tour).
    """
    n = len(tour)
    # Segment boundaries
    before_seg = tour[(seg_start - 1) % n]
    seg_first = tour[seg_start]
    seg_last = tour[(seg_start + seg_len - 1) % n]
    after_seg = tour[(seg_start + seg_len) % n]

    # Insertion point boundaries
    ins_left = tour[insert_after]
    ins_right = tour[(insert_after + 1) % n]

    # Cost of removing the segment
    removal_delta = (
        dist_matrix[before_seg][after_seg]
        - dist_matrix[before_seg][seg_first]
        - dist_matrix[seg_last][after_seg]
    )

    # Cost of inserting the segment
    insertion_delta = (
        dist_matrix[ins_left][seg_first]
        + dist_matrix[seg_last][ins_right]
        - dist_matrix[ins_left][ins_right]
    )

    return removal_delta + insertion_delta


def _apply_or_opt(
    tour: list[int],
    seg_start: int,
    seg_len: int,
    insert_after: int,
) -> list[int]:
    """Removes a segment and reinserts it after the given position."""
    n = len(tour)
    # Extract the segment: positions seg_start .. seg_start+seg_len-1 (cyclic).
    seg_end = seg_start + seg_len
    if seg_end <= n:
        segment = tour[seg_start:seg_end]
    else:  # segment wraps the array end
        segment = tour[seg_start:] + tour[: seg_end - n]

    # Build tour without the segment. `remaining` is the tour rotated to start at
    # (seg_start+seg_len) with the segment cut out, i.e. the n-seg_len positions
    # walked cyclically from `s`: remaining[m] == tour[(s + m) % n]. Expressed as
    # C-level slices instead of an element-by-element Python loop -- identical
    # output, ~60x cheaper (this runs once per accepted Or-opt move, and on large
    # instances it dominated the whole initial descent).
    s = (seg_start + seg_len) % n
    keep = n - seg_len
    if s + keep <= n:
        remaining = tour[s : s + keep]
    else:
        remaining = tour[s:] + tour[: s + keep - n]

    # Index of `insert_after` inside `remaining`, computed directly: `remaining`
    # is the tour rotated to start at (seg_start+seg_len) with the segment cut
    # out, so remaining[m] == tour[(seg_start+seg_len+m) % n]. Solving for m is
    # O(1) and replaces the previous O(n) `remaining.index(...)` scan. The caller
    # guarantees insert_after is not inside the removed segment.
    ins_idx = (insert_after - (seg_start + seg_len)) % n
    if ins_idx >= len(remaining):  # safety: position fell inside the segment
        return tour

    # Insert segment after ins_idx
    new_tour = remaining[: ins_idx + 1] + segment + remaining[ins_idx + 1 :]
    return new_tour


def _or_opt_move_inplace(
    tour: list[int],
    seg_start: int,
    seg_len: int,
    insert_after: int,
) -> tuple[int, int]:
    """Same Or-opt move as `_apply_or_opt`, but MUTATES `tour` and keeps the
    array's origin instead of rotating it.

    `_apply_or_opt` rebuilds the tour starting just after the moved segment. That
    rotation is an artifact of how it is written -- a tour is a cycle, so where
    the array "starts" carries no meaning -- but it shifts every city's index,
    which forced the caller to rebuild the whole n-entry position map after every
    accepted move (8.2 ms at n=59k, which dominated the entire initial descent).

    Moving the segment in place instead confines the change to one contiguous
    window, so the caller refreshes only that window. Returns the inclusive
    (lo, hi) index range whose contents changed; (0, len(tour)-1) when the rare
    wrapping-segment fallback rebuilt the whole array. The resulting CYCLE is
    identical to `_apply_or_opt`'s (verified exhaustively); only the array's
    rotation differs.
    """
    n = len(tour)
    seg_end = seg_start + seg_len  # exclusive

    if seg_end <= n:
        # `insert_after` is guaranteed by the caller to lie outside
        # [seg_start-1, seg_end] (mod n), so exactly one of these applies.
        if insert_after >= seg_end:
            # Segment moves forward: the block after it slides left by seg_len.
            segment = tour[seg_start:seg_end]
            tour[seg_start : insert_after + 1] = (
                tour[seg_end : insert_after + 1] + segment
            )
            return seg_start, insert_after
        if insert_after < seg_start - 1:
            # Segment moves backward: the block before it slides right by seg_len.
            segment = tour[seg_start:seg_end]
            tour[insert_after + 1 : seg_end] = (
                segment + tour[insert_after + 1 : seg_start]
            )
            return insert_after + 1, seg_end - 1
        # insert_after == seg_start-1: the segment is already there -> no-op.
        return 0, -1

    # Wrapping segment (only when seg_start > n - seg_len, i.e. a handful of the
    # n starting positions): fall back to the general rotating path.
    tour[:] = _apply_or_opt(list(tour), seg_start, seg_len, insert_after)
    return 0, n - 1


def _or_opt_pass(
    tour: list[int],
    current_cost: float,
    dist_matrix: list[list[float]],
    city_nearest_neighbors: list[list[int]],
    neighbor_limit: int,
    max_segment_len: int = 3,
    deadline: Optional[float] = None,
) -> tuple[list[int], float, int]:
    """Performs one pass of Or-opt (segment relocation) with neighbor pruning."""
    n = len(tour)
    working_tour = list(tour)
    working_cost = current_cost
    total_accepted = 0

    max_city_id = max(working_tour)
    city_to_pos = [0] * (max_city_id + 1)
    for pos, city in enumerate(working_tour):
        city_to_pos[city] = pos

    # Don't-look bits keyed on the segment's anchor city: skip a city until one
    # of the tour edges around it changes. Replaces the previous full-scan
    # restart after every accepted move; candidate moves / acceptance unchanged.
    dont_look = bytearray(max_city_id + 1)  # 0 = active, 1 = skip
    active = True
    while active:
        if _past_deadline(deadline):
            break
        active = False
        for scan in range(n):
            anchor_city = working_tour[scan]
            if dont_look[anchor_city]:
                continue
            seg_start = city_to_pos[anchor_city]
            found = False

            for seg_len in range(1, max_segment_len + 1):
                seg_first_city = working_tour[seg_start]
                seg_last_city = working_tour[(seg_start + seg_len - 1) % n]

                # Candidate insertion points from neighbors of segment endpoints
                candidate_positions: set[int] = set()
                for neighbor_city in city_nearest_neighbors[seg_first_city][:neighbor_limit]:
                    candidate_positions.add(city_to_pos[neighbor_city])
                for neighbor_city in city_nearest_neighbors[seg_last_city][:neighbor_limit]:
                    candidate_positions.add(city_to_pos[neighbor_city])

                best_delta = -1e-9
                best_insert_after = -1

                for insert_after in candidate_positions:
                    # Skip positions that overlap with the segment
                    skip = False
                    for offset in range(-1, seg_len + 1):
                        if insert_after == (seg_start + offset) % n:
                            skip = True
                            break
                    if skip:
                        continue

                    delta = _or_opt_delta(
                        working_tour, dist_matrix, seg_start, seg_len, insert_after
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_insert_after = insert_after

                if best_insert_after >= 0:
                    # Cities whose tour edges change (reactivate them afterwards)
                    affected = (
                        working_tour[(seg_start - 1) % n], seg_first_city, seg_last_city,
                        working_tour[(seg_start + seg_len) % n],
                        working_tour[best_insert_after],
                        working_tour[(best_insert_after + 1) % n],
                    )
                    # best_delta is the EXACT cost change of this Or-opt move
                    # (segment and insertion edge are disjoint by the overlap
                    # check above), so update incrementally instead of O(n) recompute.
                    lo, hi = _or_opt_move_inplace(
                        working_tour, seg_start, seg_len, best_insert_after
                    )
                    working_cost += best_delta
                    total_accepted += 1
                    # Refresh positions over ONLY the window that moved (the move
                    # keeps the array's origin, so everything outside is untouched).
                    for pos in range(lo, hi + 1):
                        city_to_pos[working_tour[pos]] = pos
                    for c in affected:
                        dont_look[c] = 0
                    found = True
                    active = True
                    if (total_accepted & 511) == 0 and _past_deadline(deadline):
                        return working_tour, working_cost, total_accepted
                    break  # positions changed; re-evaluate fresh next pass

            if not found:
                dont_look[anchor_city] = 1

    return working_tour, working_cost, total_accepted


def _best_relocate_move_for_city(
    city: int,
    i: int,
    tour: list[int],
    dist_matrix: list[list[float]],
    projected: list[tuple[float, float]],
    city_to_pos: list[int],
    city_nearest_neighbors: list[list[int]],
    k_nearest_edges: int,
    city_neighbor_limit: int,
) -> tuple[float, int]:
    """Best single-point relocation for ONE city (city sits at tour position i).
    Mirrors the per-city body of `compute_sensitivities` exactly, so the returned
    `delta` is the precise cost change of moving `city` onto the chosen edge.
    Returns (delta, insert_after_pos); delta is +inf / pos -1 when no candidate.
    This per-city evaluation (O(k)) lets `_relocate_pass` sweep with don't-look
    bits instead of recomputing ALL n sensitivities after every accepted move."""
    n = len(tour)
    prev_city = tour[(i - 1) % n]
    next_city = tour[(i + 1) % n]
    removal_saving = (dist_matrix[prev_city][city] + dist_matrix[city][next_city]
                      - dist_matrix[prev_city][next_city])

    px, py = projected[city]
    stage_neighbors = city_nearest_neighbors[city]
    if city_neighbor_limit is not None:
        stage_neighbors = stage_neighbors[:city_neighbor_limit]

    candidate_edges = set()
    for neighbor_city in stage_neighbors:
        pos = city_to_pos[neighbor_city]
        candidate_edges.add(pos)
        candidate_edges.add((pos - 1) % n)

    forbidden = {i, (i - 1) % n, (i + 1) % n, (i - 2) % n, (i + 2) % n}
    edge_dists: list[tuple[float, int]] = []
    for j in candidate_edges:
        if j in forbidden:
            continue
        a_idx = tour[j]
        b_idx = tour[(j + 1) % n]
        seg_dist = _point_to_segment_distance(
            px, py,
            projected[a_idx][0], projected[a_idx][1],
            projected[b_idx][0], projected[b_idx][1],
        )
        edge_dists.append((seg_dist, j))

    if not edge_dists:
        return float("inf"), -1

    edge_dists.sort()
    best_insertion_cost = float("inf")
    best_j = -1
    for _, j in edge_dists[:k_nearest_edges]:
        a_idx = tour[j]
        b_idx = tour[(j + 1) % n]
        insert_cost = (dist_matrix[a_idx][city] + dist_matrix[city][b_idx]
                       - dist_matrix[a_idx][b_idx])
        if insert_cost < best_insertion_cost:
            best_insertion_cost = insert_cost
            best_j = j

    if best_j < 0:
        return float("inf"), -1
    return best_insertion_cost - removal_saving, best_j


def _relocate_pass(
    tour: list[int],
    current_cost: float,
    dist_matrix: list[list[float]],
    projected: list[tuple[float, float]],
    city_nearest_neighbors: list[list[int]],
    k_nearest_edges: int,
    city_neighbor_limit: int,
    deadline: Optional[float] = None,
) -> tuple[list[int], float, int]:
    """Single-point relocation sweep with don't-look bits.

    The previous version called `_try_best_reassignment_move`, which recomputed
    ALL n sensitivities (O(n*k)) to pick a single global-best move per iteration
    -> O(n*k) per accepted move. This sweep evaluates one city at a time (O(k))
    and only reactivates the few cities whose tour edges changed, turning the
    per-move cost from O(n*k) into O(k) + O(n) for the tour splice. The move
    delta is exact (same formula), so only improving moves are ever applied;
    first-improvement scan order differs from the old global-best order, so the
    local optimum reached may differ (this pass feeds ILS/GNN, not the reported
    deterministic line-reassignment methods)."""
    n = len(tour)
    working_tour = list(tour)
    working_cost = current_cost
    total_accepted = 0

    max_city_id = max(working_tour)
    city_to_pos = [0] * (max_city_id + 1)
    for pos, c in enumerate(working_tour):
        city_to_pos[c] = pos

    dont_look = bytearray(max_city_id + 1)  # 0 = active, 1 = skip
    active = True
    while active:
        if _past_deadline(deadline):
            break
        active = False
        for scan in range(n):
            city = working_tour[scan]
            if dont_look[city]:
                continue
            i = city_to_pos[city]
            delta, insert_after = _best_relocate_move_for_city(
                city, i, working_tour, dist_matrix, projected, city_to_pos,
                city_nearest_neighbors, k_nearest_edges, city_neighbor_limit,
            )
            if delta < -1e-9 and insert_after not in (i, (i - 1) % n, (i + 1) % n):
                prev_city = working_tour[(i - 1) % n]
                next_city = working_tour[(i + 1) % n]
                left_city = working_tour[insert_after]
                right_city = working_tour[(insert_after + 1) % n]
                # A relocate is an Or-opt move with seg_len=1, and this produces
                # the exact same array as relocate_point(working_tour, i,
                # insert_after) -- verified exhaustively -- but in place and
                # reporting the window that changed, so the position map costs
                # O(window) instead of a full n-entry Python rebuild per move.
                lo, hi = _or_opt_move_inplace(working_tour, i, 1, insert_after)
                working_cost += delta
                total_accepted += 1
                for pos in range(lo, hi + 1):
                    city_to_pos[working_tour[pos]] = pos
                for c in (city, prev_city, next_city, left_city, right_city):
                    dont_look[c] = 0
                active = True
                if (total_accepted & 511) == 0 and _past_deadline(deadline):
                    return working_tour, working_cost, total_accepted
            else:
                dont_look[city] = 1

    return working_tour, working_cost, total_accepted


def _double_bridge_perturbation(
    tour: list[int],
    rng: 'random.Random',
) -> list[int]:
    """Applies a random double-bridge (4-edge break) perturbation.

    This is the standard ILS perturbation for TSP. It breaks the tour
    into 4 segments and reassembles them in a non-sequential order,
    creating a tour that cannot be reached by 2-opt or Or-opt alone.
    This is the key mechanism for escaping local optima.
    """
    n = len(tour)
    if n < 8:
        return tour[:]

    # Pick 3 random cut points (sorted) to create 4 segments
    cuts = sorted(rng.sample(range(1, n), 3))
    a, b, c = cuts

    # Segments: [0..a), [a..b), [b..c), [c..n)
    seg1 = tour[:a]
    seg2 = tour[a:b]
    seg3 = tour[b:c]
    seg4 = tour[c:]

    # Reassemble in double-bridge order: seg1 + seg3 + seg2 + seg4
    return seg1 + seg3 + seg2 + seg4


def _segment_reverse_perturbation(
    tour: list[int],
    rng: 'random.Random',
    min_seg_ratio: float = 0.05,
    max_seg_ratio: float = 0.25,
) -> list[int]:
    """Randomly reverses a medium-sized segment of the tour.

    A softer perturbation than double-bridge, useful for fine-grained
    exploration near the current solution.
    """
    n = len(tour)
    min_seg = max(3, int(n * min_seg_ratio))
    max_seg = max(min_seg + 1, int(n * max_seg_ratio))
    seg_len = rng.randint(min_seg, min(max_seg, n - 2))
    start = rng.randint(0, n - seg_len)
    new_tour = tour[:start] + tour[start:start + seg_len][::-1] + tour[start + seg_len:]
    return new_tour


def _combined_local_search(
    tour: list[int],
    current_cost: float,
    dist_matrix: list[list[float]],
    projected: list[tuple[float, float]],
    city_nearest_neighbors: list[list[int]],
    neighbor_limit_2opt: int,
    neighbor_limit_oropt: int,
    k_nearest_edges_relocate: int,
    city_neighbor_limit_relocate: int,
    max_ls_rounds: int = 10,
    deadline: Optional[float] = None,
) -> tuple[list[int], float, int, bool]:
    """Applies a combined local search: 2-opt → Or-opt → Relocate, repeated until no improvement.

    Returns (tour, cost, total_accepted, converged) where ``converged`` is True
    iff a full round accepted zero moves — i.e. a true local optimum was reached
    rather than the descent being cut off by ``max_ls_rounds`` or the deadline.
    """
    working_tour = list(tour)
    working_cost = current_cost
    total_accepted = 0
    converged = False

    for _ in range(max_ls_rounds):
        if _past_deadline(deadline):
            break
        round_accepted = 0

        # Phase 1: 2-opt
        new_tour, new_cost, accepted = _two_opt_pass(
            working_tour, working_cost, dist_matrix,
            city_nearest_neighbors, neighbor_limit_2opt,
            deadline=deadline,
        )
        if accepted:
            working_tour, working_cost = new_tour, new_cost
            round_accepted += accepted

        # Phase 2: Or-opt
        new_tour, new_cost, accepted = _or_opt_pass(
            working_tour, working_cost, dist_matrix,
            city_nearest_neighbors, neighbor_limit_oropt,
            max_segment_len=3, deadline=deadline,
        )
        if accepted:
            working_tour, working_cost = new_tour, new_cost
            round_accepted += accepted

        # Phase 3: Relocate (single-point reassignment)
        new_tour, new_cost, accepted = _relocate_pass(
            working_tour, working_cost, dist_matrix, projected,
            city_nearest_neighbors,
            k_nearest_edges=k_nearest_edges_relocate,
            city_neighbor_limit=city_neighbor_limit_relocate,
            deadline=deadline,
        )
        if accepted:
            working_tour, working_cost = new_tour, new_cost
            round_accepted += accepted

        total_accepted += round_accepted
        if round_accepted == 0:
            converged = True
            break

    return working_tour, working_cost, total_accepted, converged


def run_neighborhood_vnd(
    instance: TSPInstance,
    initial_tour: list[int],
    *,
    neighbor_limit_2opt: int = 20,
    neighbor_limit_oropt: int = 20,
    max_oropt_segment: int = 3,
    k_relocate: int = 30,
    relocate_neighbor_limit: int = 120,
    dense_levels: "Optional[Sequence]" = None,
    density_neighbor_count: int = 10,
    max_rounds: int = 60,
    deadline: "Optional[float]" = None,
) -> LineReassignResult:
    """Variable-Neighborhood Descent (deterministic): round-robin 2-opt + Or-opt +
    single-point relocate (+ optional dense multi-point batches), repeated until a
    full round makes no improvement. The returned tour is a local optimum w.r.t.
    ALL enabled neighborhoods, so enabling a STRICTLY LARGER neighborhood set
    (bigger candidate lists, longer Or-opt segments, dense batches) can only keep
    or lower the cost. This is the mathematical basis of the monotone hierarchy
    Line-Reassign (relocate) > Adaptive-LR (+2-opt/Or-opt) > Dense-LR (larger
    neighborhood) when each level is chained onto the previous one's tour."""
    instance.validate_tour(initial_tour)
    projected = compute_projected_coords(instance)
    dist_matrix = instance.distance_matrix
    n = len(initial_tour)

    max_nn = min(n - 1, max(relocate_neighbor_limit, neighbor_limit_2opt,
                            neighbor_limit_oropt, 50))
    city_nn = _precompute_city_nearest_neighbors(instance, max_nn)

    working_tour = list(initial_tour)
    working_cost = tour_cost_from_matrix(working_tour, dist_matrix)
    initial_cost = working_cost
    start_time = time.perf_counter()

    norm_levels = None
    density_radii = None
    if dense_levels:
        norm_levels = _normalize_dense_adaptive_levels(dense_levels, n)
        density_radii = _compute_density_radii(dist_matrix, city_nn, density_neighbor_count)

    history: list[HistoryPoint] = [HistoryPoint(iteration=0, elapsed_time=0.0, cost=working_cost)]
    frames: list[TourFrame] = [TourFrame(
        step=0, phase="initial", stage_index=0, elapsed_time=0.0, cost=working_cost,
        best_cost=working_cost, constructed_cities=0, tour=_frame_tour(working_tour))]
    total_accepted = 0
    total_candidates = 0
    rounds = 0

    while rounds < max_rounds:
        if _past_deadline(deadline):
            break
        rounds += 1
        round_accepted = 0

        t, c, a = _two_opt_pass(working_tour, working_cost, dist_matrix, city_nn,
                                neighbor_limit_2opt, deadline=deadline)
        if a:
            working_tour, working_cost = t, c; round_accepted += a

        t, c, a = _or_opt_pass(working_tour, working_cost, dist_matrix, city_nn,
                               neighbor_limit_oropt, max_segment_len=max_oropt_segment,
                               deadline=deadline)
        if a:
            working_tour, working_cost = t, c; round_accepted += a

        t, c, a = _relocate_pass(working_tour, working_cost, dist_matrix, projected,
                                 city_nn, k_relocate, relocate_neighbor_limit,
                                 deadline=deadline)
        if a:
            working_tour, working_cost = t, c; round_accepted += a

        if norm_levels:
            for level in norm_levels:
                dense_mask, _, _ = _dense_city_mask(density_radii, level.dense_fraction)
                moves, cand = _select_dense_batch_moves(
                    working_tour, dist_matrix, projected, city_nn, density_radii,
                    dense_mask, level, 1e-06)
                total_candidates += cand
                if moves:
                    nt, nc, acc = _apply_dense_batch_moves(
                        working_tour, working_cost, dist_matrix, moves, 1e-06)
                    if acc and nc < working_cost - 1e-9:
                        working_tour, working_cost = nt, nc
                        round_accepted += acc

        total_accepted += round_accepted
        if round_accepted:
            elapsed = time.perf_counter() - start_time
            history.append(HistoryPoint(iteration=rounds, elapsed_time=elapsed, cost=working_cost))
            frames.append(TourFrame(
                step=len(frames), phase="repair", stage_index=total_accepted,
                elapsed_time=elapsed, cost=working_cost, best_cost=working_cost,
                constructed_cities=round_accepted, tour=_frame_tour(working_tour)))
        else:
            break

    elapsed = time.perf_counter() - start_time
    history.append(HistoryPoint(iteration=rounds + 1, elapsed_time=elapsed, cost=working_cost))
    return LineReassignResult(
        tour=working_tour, cost=working_cost, iterations_completed=rounds,
        improvements_accepted=total_accepted, total_candidates_evaluated=total_candidates,
        total_saving=initial_cost - working_cost, frames=frames, history=history)


def run_iterated_local_search(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 500,
    max_no_improve: int = 100,
    max_time_seconds: float = DEFAULT_ILS_TIME_LIMIT_SECONDS,
    neighbor_limit_2opt: int = 20,
    neighbor_limit_oropt: int = 20,
    k_nearest_edges_relocate: int = 30,
    city_neighbor_limit_relocate: int = 120,
    max_ls_rounds: int = 10,
    perturbation_strength: str = "adaptive",
    seed: int = 42,
) -> LineReassignResult:
    """Runs the Iterated Local Search (ILS) metaheuristic.

    Algorithm:
      1. Start from the initial tour (e.g., output of Adaptive Line Reassignment).
      2. Apply combined local search (2-opt + Or-opt + Relocate) to reach a local optimum.
      3. Perturb the best-known tour using double-bridge or segment-reverse.
      4. Apply local search to the perturbed tour.
      5. If the new local optimum is better, accept it as the new best.
      6. Repeat from step 3 until max_iterations or max_no_improve or max_time_seconds.

    Parameters:
      max_iterations: Maximum number of perturb+search cycles.
      max_no_improve: Stop after this many consecutive non-improving iterations.
      max_time_seconds: Time limit in seconds (default 60, 0 = unlimited).
      perturbation_strength: "soft" (segment reverse), "hard" (double bridge),
                             or "adaptive" (alternate based on stagnation).
      seed: Random seed for reproducible perturbations.
    """
    import random as random_module

    rng = random_module.Random(seed)

    instance.validate_tour(initial_tour)
    projected = compute_projected_coords(instance)
    dist_matrix = instance.distance_matrix
    n_cities = len(initial_tour)

    # Clamp neighbor limits
    max_nn = min(n_cities - 1, max(
        neighbor_limit_2opt,
        neighbor_limit_oropt,
        city_neighbor_limit_relocate,
    ))

    print(f"\n{'=' * 65}")
    print(f"[{time.strftime('%H:%M:%S')}] ITERATED LOCAL SEARCH (ILS) STARTED")
    print(f"{'=' * 65}")
    print("Search parameters:")
    print(f"  > max_iterations              : {max_iterations}")
    print(f"  > max_no_improve              : {max_no_improve}")
    print(f"  > max_time_seconds            : {max_time_seconds if max_time_seconds > 0 else 'unlimited'}")
    print(f"  > neighbor_limit_2opt         : {neighbor_limit_2opt}")
    print(f"  > neighbor_limit_oropt        : {neighbor_limit_oropt}")
    print(f"  > k_nearest_edges_relocate    : {k_nearest_edges_relocate}")
    print(f"  > city_neighbor_limit_relocate: {city_neighbor_limit_relocate}")
    print(f"  > max_ls_rounds               : {max_ls_rounds}")
    print(f"  > perturbation_strength       : {perturbation_strength}")
    print(f"  > seed                        : {seed}")
    print(f"{'-' * 65}")

    t0 = time.perf_counter()
    city_nearest_neighbors = _precompute_city_nearest_neighbors(instance, max_nn)
    t1 = time.perf_counter()
    sys.stdout.write(
        f"  [Info] Cached nearest-neighbor lists up to {max_nn} cities in {t1 - t0:.2f}s.\n"
    )

    start_time = time.perf_counter()
    # Hard safety deadline shared by every local-search descent so the budget is
    # honored even within the (otherwise uninterruptible) Phase-0 search. The
    # soft per-iteration check below remains the primary stop.
    ls_deadline = (start_time + max_time_seconds * _LS_DEADLINE_SAFETY
                   if max_time_seconds > 0 else None)

    # Initial local search
    sys.stdout.write("  [ILS] Phase 0: Initial local search on starting tour...\n")
    sys.stdout.flush()
    initial_cost = tour_cost_from_matrix(initial_tour, dist_matrix)

    working_tour, working_cost, init_accepted, phase0_converged = _combined_local_search(
        tour=list(initial_tour),
        current_cost=initial_cost,
        dist_matrix=dist_matrix,
        projected=projected,
        city_nearest_neighbors=city_nearest_neighbors,
        neighbor_limit_2opt=neighbor_limit_2opt,
        neighbor_limit_oropt=neighbor_limit_oropt,
        k_nearest_edges_relocate=k_nearest_edges_relocate,
        city_neighbor_limit_relocate=city_neighbor_limit_relocate,
        max_ls_rounds=max_ls_rounds,
        deadline=ls_deadline,
    )

    elapsed_init = time.perf_counter() - start_time
    sys.stdout.write(
        f"  [ILS] Initial LS: {initial_cost:.2f} -> {working_cost:.2f} "
        f"({init_accepted} moves, {elapsed_init:.1f}s)\n"
    )

    best_tour = list(working_tour)
    best_cost = working_cost

    frames: list[TourFrame] = [
        TourFrame(
            step=0,
            phase="initial",
            stage_index=0,
            elapsed_time=0.0,
            cost=initial_cost,
            best_cost=best_cost,
            constructed_cities=0,
            tour=list(initial_tour),
        ),
        TourFrame(
            step=1,
            phase="initial_ls",
            stage_index=0,
            elapsed_time=elapsed_init,
            cost=working_cost,
            best_cost=best_cost,
            constructed_cities=init_accepted,
            tour=working_tour[:],
        ),
    ]
    history: list[HistoryPoint] = [
        HistoryPoint(iteration=0, elapsed_time=0.0, cost=initial_cost),
        HistoryPoint(iteration=1, elapsed_time=elapsed_init, cost=best_cost),
    ]

    total_improvements = 0
    no_improve_count = 0
    stagnation_counter = 0
    # Observability only (does not alter the search): why did the loop stop, and
    # how many perturbation iterations actually ran. "max_iter" is the default
    # because the loop reaching its natural end means the iteration cap bound it.
    stop_reason = "max_iter"
    perturbation_iterations = 0

    for iteration in range(1, max_iterations + 1):
        elapsed_so_far = time.perf_counter() - start_time
        if max_time_seconds > 0 and elapsed_so_far >= max_time_seconds:
            sys.stdout.write(f"\n  [ILS] Time limit reached ({max_time_seconds:.0f}s).\n")
            stop_reason = "time"
            break

        if no_improve_count >= max_no_improve:
            sys.stdout.write(f"\n  [ILS] No improvement for {max_no_improve} consecutive iterations.\n")
            stop_reason = "no_improve"
            break

        perturbation_iterations += 1

        # Determine perturbation type
        if perturbation_strength == "adaptive":
            # Use soft perturbation when recently improved, hard when stagnating
            if stagnation_counter < 5:
                use_hard = rng.random() < 0.3
            elif stagnation_counter < 15:
                use_hard = rng.random() < 0.6
            else:
                use_hard = True
        elif perturbation_strength == "hard":
            use_hard = True
        else:
            use_hard = False

        # Perturb the best-known tour
        if use_hard:
            perturbed_tour = _double_bridge_perturbation(best_tour, rng)
            perturb_label = "bridge"
        else:
            perturbed_tour = _segment_reverse_perturbation(best_tour, rng)
            perturb_label = "seg-rev"

        perturbed_cost = tour_cost_from_matrix(perturbed_tour, dist_matrix)

        # Local search on perturbed tour
        ls_tour, ls_cost, ls_accepted, _ = _combined_local_search(
            tour=perturbed_tour,
            current_cost=perturbed_cost,
            dist_matrix=dist_matrix,
            projected=projected,
            city_nearest_neighbors=city_nearest_neighbors,
            neighbor_limit_2opt=neighbor_limit_2opt,
            neighbor_limit_oropt=neighbor_limit_oropt,
            k_nearest_edges_relocate=k_nearest_edges_relocate,
            city_neighbor_limit_relocate=city_neighbor_limit_relocate,
            max_ls_rounds=max_ls_rounds,
            deadline=ls_deadline,
        )

        elapsed = time.perf_counter() - start_time
        avg_iter_time = elapsed / iteration
        remaining = max_iterations - iteration
        eta = avg_iter_time * remaining
        if max_time_seconds > 0:
            eta = min(eta, max(0, max_time_seconds - elapsed))

        if ls_cost < best_cost - 1e-9:
            improvement = best_cost - ls_cost
            best_tour = list(ls_tour)
            best_cost = ls_cost
            total_improvements += 1
            no_improve_count = 0
            stagnation_counter = 0
            marker = f"*** IMPROVED by {improvement:.2f} ***"

            frames.append(TourFrame(
                step=len(frames),
                phase="repair",
                stage_index=total_improvements,
                elapsed_time=elapsed,
                cost=best_cost,
                best_cost=best_cost,
                constructed_cities=ls_accepted,
                tour=_frame_tour(best_tour),
            ))
            history.append(HistoryPoint(
                iteration=total_improvements + 1,
                elapsed_time=elapsed,
                cost=best_cost,
            ))
        else:
            no_improve_count += 1
            stagnation_counter += 1
            marker = ""

        if VERBOSE:
            sys.stdout.write(
                "\r"
                f"[ILS] Iter: {iteration}/{max_iterations} | "
                f"Perturb: {perturb_label} | "
                f"LS moves: {ls_accepted} | "
                f"Cost: {ls_cost:.2f} | "
                f"Best: {best_cost:.2f} | "
                f"Improved: {total_improvements} | "
                f"Stall: {no_improve_count}/{max_no_improve} | "
                f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s   "
            )
            sys.stdout.flush()
            if marker:
                sys.stdout.write(f"\n  {marker}\n")
                sys.stdout.flush()

    sys.stdout.write("\n")
    elapsed = time.perf_counter() - start_time

    # Final frame
    frames.append(TourFrame(
        step=len(frames),
        phase="final",
        stage_index=total_improvements,
        elapsed_time=elapsed,
        cost=best_cost,
        best_cost=best_cost,
        constructed_cities=0,
        tour=best_tour[:],
    ))
    history.append(HistoryPoint(
        iteration=total_improvements + 2,
        elapsed_time=elapsed,
        cost=best_cost,
    ))

    print(f"\n{'=' * 65}")
    print(f"[ILS] COMPLETED")
    print(f"  Initial cost  : {initial_cost:.2f}")
    print(f"  Best cost     : {best_cost:.2f}")
    print(f"  Total saving  : {initial_cost - best_cost:.2f}")
    print(f"  Improvements  : {total_improvements}")
    print(f"  Iterations    : {iteration if iteration <= max_iterations else max_iterations}")
    print(f"  Total time    : {elapsed:.1f}s")
    print(f"  Stop reason   : {stop_reason} | Phase-0 converged: {phase0_converged} | "
          f"perturbation iters: {perturbation_iterations}")
    print(f"{'=' * 65}\n")

    return LineReassignResult(
        tour=best_tour,
        cost=best_cost,
        iterations_completed=iteration if iteration <= max_iterations else max_iterations,
        improvements_accepted=total_improvements,
        total_candidates_evaluated=0,
        total_saving=initial_cost - best_cost,
        frames=frames,
        history=history,
        stop_reason=stop_reason,
        phase0_converged=phase0_converged,
        perturbation_iterations=perturbation_iterations,
    )


def build_ils_result(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 500,
    max_no_improve: int = 100,
    max_time_seconds: float = DEFAULT_ILS_TIME_LIMIT_SECONDS,
    neighbor_limit_2opt: int = 20,
    neighbor_limit_oropt: int = 20,
    k_nearest_edges_relocate: int = 30,
    city_neighbor_limit_relocate: int = 120,
    max_ls_rounds: int = 10,
    perturbation_strength: str = "adaptive",
    seed: int = 42,
) -> tuple[TSPRunResult, LineReassignResult]:
    """Builds a dashboard-compatible result for the ILS optimizer."""

    result = run_iterated_local_search(
        instance=instance,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        max_no_improve=max_no_improve,
        max_time_seconds=max_time_seconds,
        neighbor_limit_2opt=neighbor_limit_2opt,
        neighbor_limit_oropt=neighbor_limit_oropt,
        k_nearest_edges_relocate=k_nearest_edges_relocate,
        city_neighbor_limit_relocate=city_neighbor_limit_relocate,
        max_ls_rounds=max_ls_rounds,
        perturbation_strength=perturbation_strength,
        seed=seed,
    )

    metrics = RunMetrics(
        initialization_name="Iterated Local Search (ILS)",
        initial_cost=instance.tour_cost(initial_tour),
        final_cost=result.cost,
        cpu_time_seconds=result.frames[-1].elapsed_time,
        frame_count=len(result.frames),
        history=result.history,
    )

    return (
        TSPRunResult(
            initial_tour=list(initial_tour),
            final_tour=result.tour,
            metrics=metrics,
            animation_frames=result.frames,
        ),
        result,
    )


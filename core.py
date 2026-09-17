"""
core.py -- Greedy Snake Benchmark cekirdek model katmani (app.py'den tasindi,
dashboard/HTML disa-aktarimi ve komut-satiri arabirimi makale kapsami disinda
birakilarak silindi).

Icerik: TSPInstance / TSPDataLoader / mesafe metrikleri, Snake-Grid kurucusu,
pencere onarimi (SnakeWindowRepair / AdaptiveSnakeWindowRepair), sonuc veri
siniflari ve (opsiyonel) matplotlib gorsellestirmesi.
"""
from __future__ import annotations

import csv
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Sequence

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional; the benchmark never plots
    # ImportError (not just ModuleNotFoundError): a broken install / blocked
    # native DLL must not take the whole optimization engine down with it.
    plt = None


EARTH_RADIUS_KM: float = 6371.0088
SnakeAxis = Literal["row", "column"]


class DistanceMetric(str, Enum):
    """Supported distance metrics for the TSP instance."""

    HAVERSINE = "haversine"
    EUCLIDEAN = "euclidean"


@dataclass(frozen=True)
class City:
    """Represents a city or point in the TSP instance."""

    identifier: int
    name: str
    coord_1: float
    coord_2: float


@dataclass(frozen=True)
class HistoryPoint:
    """Stores one convergence record for the optimization history."""

    iteration: int
    elapsed_time: float
    cost: float


@dataclass
class RunMetrics:
    """Collects the dashboard metrics for one route-construction run."""

    initialization_name: str
    initial_cost: float
    final_cost: float
    cpu_time_seconds: float
    frame_count: int
    history: list[HistoryPoint] = field(default_factory=list)


@dataclass(frozen=True)
class TourFrame:
    """Stores one route snapshot for interactive playback."""

    step: int
    phase: str
    stage_index: int
    elapsed_time: float
    cost: float
    best_cost: float
    constructed_cities: int
    tour: list[int]


@dataclass
class TSPRunResult:
    """Stores one complete route-construction run."""

    initial_tour: list[int]
    final_tour: list[int]
    metrics: RunMetrics
    animation_frames: list[TourFrame] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectedPoint:
    """Stores a city projected onto a planar coordinate system."""

    city_index: int
    x: float
    y: float


@dataclass(frozen=True)
class SnakeGridCandidate:
    """Represents one deterministic snake-board candidate."""

    rows: int
    cols: int
    cell_size: float
    origin_x: float
    origin_y: float


@dataclass(frozen=True)
class SnakeInitializationResult:
    """Stores the selected snake-grid initialization and diagnostics."""

    tour: list[int]
    rows: int
    cols: int
    occupied_cells: int
    max_cell_load: int
    route_cost: float
    adjusted_score: float
    axis: SnakeAxis
    reverse_primary: bool
    reverse_secondary: bool


@dataclass(frozen=True)
class WindowRepairResult:
    """Stores the outcome of the fixed-endpoint window refinement pass."""

    tour: list[int]
    cost: float
    accepted_windows: int
    passes_completed: int
    windows_scanned: int
    frames: list[TourFrame]
    history: list[HistoryPoint]


@dataclass(frozen=True)
class AdaptiveWindowRepairResult:
    """Stores the outcome of the multi-scale cyclic window refinement pass."""

    tour: list[int]
    cost: float
    accepted_windows: int
    windows_scanned: int
    queued_neighbors: int
    frames: list[TourFrame]
    history: list[HistoryPoint]


class TSPInstance:
    """Encapsulates city data and the precomputed distance matrix."""

    def __init__(
        self,
        cities: Sequence[City],
        distance_metric: DistanceMetric = DistanceMetric.HAVERSINE,
        axis_labels: Optional[tuple[str, str]] = None,
    ) -> None:
        if len(cities) < 3:
            raise ValueError("A TSP instance requires at least three cities.")

        self.cities: list[City] = list(cities)
        self.distance_metric = distance_metric
        self.axis_labels: tuple[str, str] = axis_labels or self._default_axis_labels()
        self._distance_matrix: list[list[float]] = self._compute_distance_matrix()

    @property
    def size(self) -> int:
        """Returns the number of cities."""

        return len(self.cities)

    @property
    def distance_matrix(self) -> list[list[float]]:
        """Returns the precomputed symmetric distance matrix."""

        return self._distance_matrix

    def _default_axis_labels(self) -> tuple[str, str]:
        if self.distance_metric == DistanceMetric.HAVERSINE:
            return ("Longitude", "Latitude")
        return ("X", "Y")

    def _compute_distance_matrix(self) -> list[list[float]]:
        """Builds the full pairwise distance matrix."""

        matrix: list[list[float]] = [
            [0.0 for _ in range(self.size)] for _ in range(self.size)
        ]
        for i in range(self.size):
            for j in range(i + 1, self.size):
                distance = self._pairwise_distance(self.cities[i], self.cities[j])
                matrix[i][j] = distance
                matrix[j][i] = distance
        return matrix

    def _pairwise_distance(self, city_a: City, city_b: City) -> float:
        if self.distance_metric == DistanceMetric.HAVERSINE:
            return haversine_distance(
                city_a.coord_1,
                city_a.coord_2,
                city_b.coord_1,
                city_b.coord_2,
            )
        return euclidean_distance(
            city_a.coord_1,
            city_a.coord_2,
            city_b.coord_1,
            city_b.coord_2,
        )

    def validate_tour(self, tour: Sequence[int]) -> None:
        """Validates whether a tour is a proper Hamiltonian cycle permutation."""

        if len(tour) != self.size:
            raise ValueError("Tour length does not match the number of cities.")
        if set(tour) != set(range(self.size)):
            raise ValueError("Tour must contain each city index exactly once.")

    def tour_cost(self, tour: Sequence[int]) -> float:
        """Computes the total closed-tour length using the distance matrix."""

        self.validate_tour(tour)
        return tour_cost_from_matrix(tour, self.distance_matrix)

    def plot_coordinate(self, city_index: int) -> tuple[float, float]:
        """Returns plotting coordinates with longitude on the x-axis for maps."""

        city = self.cities[city_index]
        if self.distance_metric == DistanceMetric.HAVERSINE:
            return city.coord_2, city.coord_1
        return city.coord_1, city.coord_2

    def route_coordinates(
        self,
        tour: Sequence[int],
        close_cycle: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Returns the ordered coordinates of a tour for plotting."""

        self.validate_tour(tour)
        ordered_tour = list(tour)
        if close_cycle:
            ordered_tour.append(tour[0])

        xs: list[float] = []
        ys: list[float] = []
        for city_index in ordered_tour:
            x_value, y_value = self.plot_coordinate(city_index)
            xs.append(x_value)
            ys.append(y_value)
        return xs, ys


class TSPDataLoader:
    """Loads city data from flexible CSV layouts."""

    _ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1254", "latin-1")
    _NAME_HEADERS: set[str] = {"il_adi", "name", "city", "sehir"}
    _LAT_HEADERS: set[str] = {"lat", "latitude", "enlem"}
    _LON_HEADERS: set[str] = {"lon", "lng", "longitude", "boylam"}
    _X_HEADERS: set[str] = {"x", "coord_x", "x_coord"}
    _Y_HEADERS: set[str] = {"y", "coord_y", "y_coord"}
    _ID_HEADERS: set[str] = {"id", "il_id", "plate", "plaka", "index"}

    @classmethod
    def load_instance(
        cls,
        file_path: str | Path,
        force_metric: Optional[DistanceMetric] = None,
    ) -> TSPInstance:
        """Loads the dataset and infers the proper distance metric."""

        path = Path(file_path)
        raw_text = cls._read_text(path)
        delimiter = cls._detect_delimiter(raw_text)
        rows = cls._read_rows(raw_text, delimiter)
        if not rows:
            raise ValueError(f"No data rows found in {path}.")

        has_header = cls._has_header(rows[0])
        cities, inferred_metric = cls._parse_rows(rows, has_header)
        metric = force_metric or inferred_metric
        axis_labels = ("Longitude", "Latitude")
        if metric == DistanceMetric.EUCLIDEAN:
            axis_labels = ("X", "Y")
        return TSPInstance(cities=cities, distance_metric=metric, axis_labels=axis_labels)

    @classmethod
    def _read_text(cls, path: Path) -> str:
        """Reads the dataset using a small encoding fallback list."""

        last_error: Optional[UnicodeDecodeError] = None
        for encoding in cls._ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as error:
                last_error = error

        if last_error is None:
            raise ValueError(f"Unable to read dataset: {path}")

        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            last_error.reason,
        )

    @staticmethod
    def _detect_delimiter(raw_text: str) -> str:
        """Detects a likely CSV delimiter from the first non-empty line."""

        first_data_line = next(
            (line for line in raw_text.splitlines() if line.strip()),
            "",
        )
        delimiter_candidates = [";", ",", "\t", "|"]
        return max(delimiter_candidates, key=first_data_line.count)

    @staticmethod
    def _read_rows(raw_text: str, delimiter: str) -> list[list[str]]:
        """Parses CSV rows and strips empty lines."""

        rows: list[list[str]] = []
        reader = csv.reader(raw_text.splitlines(), delimiter=delimiter)
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            rows.append([cell.strip() for cell in row])
        return rows

    @classmethod
    def _has_header(cls, first_row: Sequence[str]) -> bool:
        """Checks whether the first row resembles a named header."""

        normalized = {cls._normalize_header(value) for value in first_row}
        known_headers = (
            cls._NAME_HEADERS
            | cls._LAT_HEADERS
            | cls._LON_HEADERS
            | cls._X_HEADERS
            | cls._Y_HEADERS
            | cls._ID_HEADERS
        )
        return any(value in known_headers for value in normalized)

    @staticmethod
    def _normalize_header(header: str) -> str:
        """Normalizes a header name for matching."""

        return header.strip().lower().replace(" ", "_")

    @classmethod
    def _parse_rows(
        cls,
        rows: Sequence[Sequence[str]],
        has_header: bool,
    ) -> tuple[list[City], DistanceMetric]:
        """Parses data rows into City objects and infers the metric."""

        if has_header:
            return cls._parse_headered_rows(rows)
        return cls._parse_headerless_rows(rows)

    @classmethod
    def _parse_headered_rows(
        cls,
        rows: Sequence[Sequence[str]],
    ) -> tuple[list[City], DistanceMetric]:
        header = [cls._normalize_header(value) for value in rows[0]]
        data_rows = rows[1:]

        name_index = cls._find_first_index(header, cls._NAME_HEADERS)
        id_index = cls._find_first_index(header, cls._ID_HEADERS)
        lat_index = cls._find_first_index(header, cls._LAT_HEADERS)
        lon_index = cls._find_first_index(header, cls._LON_HEADERS)
        x_index = cls._find_first_index(header, cls._X_HEADERS)
        y_index = cls._find_first_index(header, cls._Y_HEADERS)

        if lat_index is not None and lon_index is not None:
            metric = DistanceMetric.HAVERSINE
            coord_1_index, coord_2_index = lat_index, lon_index
            coord_roles = ("lat", "lon")
        elif x_index is not None and y_index is not None:
            metric = DistanceMetric.EUCLIDEAN
            coord_1_index, coord_2_index = x_index, y_index
            coord_roles = ("x", "y")
        else:
            raise ValueError("Could not infer coordinate columns from the CSV header.")

        if name_index is None:
            name_index = 0 if id_index != 0 else 1

        cities: list[City] = []
        for sequential_id, row in enumerate(data_rows, start=1):
            identifier = sequential_id
            if id_index is not None and id_index < len(row):
                identifier = int(float(row[id_index]))
            name = row[name_index] if name_index < len(row) else f"City {sequential_id}"
            coord_1 = cls._parse_coordinate(row[coord_1_index], coord_roles[0])
            coord_2 = cls._parse_coordinate(row[coord_2_index], coord_roles[1])
            cities.append(City(identifier=identifier, name=name, coord_1=coord_1, coord_2=coord_2))
        return cities, metric

    @classmethod
    def _parse_headerless_rows(
        cls,
        rows: Sequence[Sequence[str]],
    ) -> tuple[list[City], DistanceMetric]:
        metric = DistanceMetric.HAVERSINE
        cities: list[City] = []

        for sequential_id, row in enumerate(rows, start=1):
            if len(row) >= 4:
                identifier = int(float(row[0]))
                name = row[1]
                coord_1_raw = row[2]
                coord_2_raw = row[3]
            elif len(row) >= 3:
                identifier = sequential_id
                name = row[0]
                coord_1_raw = row[1]
                coord_2_raw = row[2]
            else:
                raise ValueError("Headerless rows must contain at least three columns.")

            coord_1 = cls._parse_coordinate(coord_1_raw, "lat")
            coord_2 = cls._parse_coordinate(coord_2_raw, "lon")
            cities.append(City(identifier=identifier, name=name, coord_1=coord_1, coord_2=coord_2))

        return cities, metric

    @staticmethod
    def _find_first_index(
        header: Sequence[str],
        accepted_names: set[str],
    ) -> Optional[int]:
        """Returns the first matching header index."""

        for index, column_name in enumerate(header):
            if column_name in accepted_names:
                return index
        return None

    @classmethod
    def _parse_coordinate(cls, raw_value: str, role: str) -> float:
        """Parses numeric coordinates and repairs simple geographic formatting issues."""

        normalized = raw_value.strip().replace(" ", "").replace(",", ".")
        try:
            value = float(normalized)
        except ValueError as error:
            raise ValueError(f"Could not parse coordinate value '{raw_value}'.") from error

        if role in {"lat", "lon"}:
            limit = 90.0 if role == "lat" else 180.0
            if abs(value) > limit:
                repaired = cls._repair_geographic_coordinate(raw_value, role)
                if repaired is None:
                    raise ValueError(f"Geographic coordinate '{raw_value}' exceeds valid bounds.")
                return repaired
        return value

    @staticmethod
    def _repair_geographic_coordinate(raw_value: str, role: str) -> Optional[float]:
        """Attempts to fix merged geographic decimals such as '43021596'."""

        token = raw_value.strip().replace(" ", "")
        if "." in token or "," in token:
            return None

        sign = -1.0 if token.startswith("-") else 1.0
        digits = token.lstrip("+-")
        if not digits.isdigit():
            return None

        if role == "lat":
            degree_digits = 2
            limit = 90.0
        else:
            limit = 180.0
            degree_digits = 3 if len(digits) >= 3 and int(digits[:3]) <= 180 else 2

        if len(digits) <= degree_digits:
            return None

        candidate = sign * float(f"{digits[:degree_digits]}.{digits[degree_digits:]}")
        if abs(candidate) <= limit:
            return candidate
        return None


class SnakeGridPathInitializer:
    """Deterministic grid-based snake initializer for TSP tours."""

    _CELL_SIZE_FACTORS: tuple[float, ...] = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.45, 1.70)

    def __init__(self, instance: TSPInstance) -> None:
        self.instance = instance
        self.points = self._project_points(instance)
        self.points_by_city_index: dict[int, ProjectedPoint] = {
            point.city_index: point for point in self.points
        }

    def build(self) -> SnakeInitializationResult:
        """Builds the best snake path among deterministic grid candidates."""

        candidates = self._generate_candidates(self.points)
        if not candidates:
            raise ValueError("No snake-grid candidates could be generated.")

        best_result: Optional[SnakeInitializationResult] = None
        best_key: Optional[tuple[float, float, int]] = None

        for candidate in candidates:
            cells = self._assign_to_cells(candidate)
            if not cells:
                continue

            occupied_cells = len(cells)
            max_cell_load = max(len(cell_points) for cell_points in cells.values())

            for axis in ("row", "column"):
                for reverse_primary in (False, True):
                    for reverse_secondary in (False, True):
                        tour = self._build_variant_tour(
                            candidate=candidate,
                            cells=cells,
                            axis=axis,
                            reverse_primary=reverse_primary,
                            reverse_secondary=reverse_secondary,
                        )
                        if len(tour) != self.instance.size:
                            continue

                        route_cost = tour_cost_from_matrix(tour, self.instance.distance_matrix)
                        adjusted_score = self._adjusted_score(route_cost, candidate, cells)
                        key = (adjusted_score, route_cost, candidate.rows * candidate.cols)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_result = SnakeInitializationResult(
                                tour=tour,
                                rows=candidate.rows,
                                cols=candidate.cols,
                                occupied_cells=occupied_cells,
                                max_cell_load=max_cell_load,
                                route_cost=route_cost,
                                adjusted_score=adjusted_score,
                                axis=axis,
                                reverse_primary=reverse_primary,
                                reverse_secondary=reverse_secondary,
                            )

        if best_result is None:
            raise ValueError("Snake-grid initialization failed to build a valid tour.")
        return best_result

    @staticmethod
    def _project_points(instance: TSPInstance) -> list[ProjectedPoint]:
        """Projects geographic coordinates into a local planar space."""

        if instance.distance_metric == DistanceMetric.HAVERSINE:
            mean_lat = sum(city.coord_1 for city in instance.cities) / instance.size
            mean_lon = sum(city.coord_2 for city in instance.cities) / instance.size
            mean_lat_rad = math.radians(mean_lat)
            return [
                ProjectedPoint(
                    city_index=index,
                    x=EARTH_RADIUS_KM * math.radians(city.coord_2 - mean_lon) * math.cos(mean_lat_rad),
                    y=EARTH_RADIUS_KM * math.radians(city.coord_1 - mean_lat),
                )
                for index, city in enumerate(instance.cities)
            ]

        return [
            ProjectedPoint(city_index=index, x=city.coord_1, y=city.coord_2)
            for index, city in enumerate(instance.cities)
        ]

    def _generate_candidates(
        self,
        points: Sequence[ProjectedPoint],
    ) -> list[SnakeGridCandidate]:
        """Generates square-cell board candidates from robust data scales."""

        min_x, max_x, min_y, max_y = self._bounds(points)
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)
        aspect_ratio = width / height
        dimensions: set[tuple[int, int]] = set()

        for target_occupancy in self._target_occupancies(len(points)):
            target_cells = max(1, round(len(points) / target_occupancy))
            cols = max(1, round(math.sqrt(target_cells * aspect_ratio)))
            rows = max(1, math.ceil(target_cells / cols))
            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    dimensions.add((max(1, rows + row_delta), max(1, cols + col_delta)))

        robust_cell_size = self._robust_cell_size(points)
        if robust_cell_size > 0.0:
            for factor in self._CELL_SIZE_FACTORS:
                cell_size = robust_cell_size * factor
                rows = max(1, math.ceil(height / cell_size))
                cols = max(1, math.ceil(width / cell_size))
                dimensions.add((rows, cols))

        candidates: list[SnakeGridCandidate] = []
        for rows, cols in sorted(dimensions):
            cell_size = max(width / cols, height / rows, 1e-9)
            board_width = cols * cell_size
            board_height = rows * cell_size
            origin_x = min_x - (board_width - width) / 2.0
            origin_y = min_y - (board_height - height) / 2.0
            candidates.append(
                SnakeGridCandidate(
                    rows=rows,
                    cols=cols,
                    cell_size=cell_size,
                    origin_x=origin_x,
                    origin_y=origin_y,
                )
            )
        return candidates

    @staticmethod
    def _target_occupancies(n_points: int) -> tuple[float, ...]:
        """Returns deterministic occupancy targets that scale with instance size."""

        if n_points >= 3000:
            return (1.25, 1.60, 2.00, 2.50, 3.20, 4.00, 5.00)
        if n_points >= 500:
            return (1.10, 1.35, 1.60, 2.00, 2.50, 3.20, 4.00)
        return (0.90, 1.00, 1.15, 1.35, 1.60, 2.00)

    @classmethod
    def _robust_cell_size(cls, points: Sequence[ProjectedPoint]) -> float:
        """Uses a 2D Freedman-Diaconis-style scale as a deterministic anchor."""

        x_values = [point.x for point in points]
        y_values = [point.y for point in points]
        iqr_x = cls._quantile(x_values, 0.75) - cls._quantile(x_values, 0.25)
        iqr_y = cls._quantile(y_values, 0.75) - cls._quantile(y_values, 0.25)
        robust_area_scale = math.sqrt(max(iqr_x, 1e-9) * max(iqr_y, 1e-9))
        return 2.0 * robust_area_scale * (len(points) ** (-1.0 / 3.0))

    @staticmethod
    def _quantile(values: Sequence[float], q_value: float) -> float:
        """Computes a deterministic linear-interpolated quantile."""

        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * q_value
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            return sorted_values[lower_index]
        lower_weight = upper_index - position
        upper_weight = position - lower_index
        return sorted_values[lower_index] * lower_weight + sorted_values[upper_index] * upper_weight

    @staticmethod
    def _bounds(points: Sequence[ProjectedPoint]) -> tuple[float, float, float, float]:
        """Returns x/y bounds for projected points."""

        x_values = [point.x for point in points]
        y_values = [point.y for point in points]
        return min(x_values), max(x_values), min(y_values), max(y_values)

    def _assign_to_cells(
        self,
        candidate: SnakeGridCandidate,
    ) -> dict[tuple[int, int], list[ProjectedPoint]]:
        """Assigns each point to a square grid cell."""

        cells: dict[tuple[int, int], list[ProjectedPoint]] = {}
        for point in self.points:
            col = min(
                candidate.cols - 1,
                max(0, int((point.x - candidate.origin_x) / candidate.cell_size)),
            )
            row = min(
                candidate.rows - 1,
                max(0, int((point.y - candidate.origin_y) / candidate.cell_size)),
            )
            cells.setdefault((row, col), []).append(point)
        return cells

    @staticmethod
    def _cell_center(
        candidate: SnakeGridCandidate,
        cell: tuple[int, int],
    ) -> tuple[float, float]:
        """Returns the center coordinate of a grid cell."""

        row, col = cell
        return (
            candidate.origin_x + (col + 0.5) * candidate.cell_size,
            candidate.origin_y + (row + 0.5) * candidate.cell_size,
        )

    @staticmethod
    def _snake_cells(
        rows: int,
        cols: int,
        axis: SnakeAxis,
        reverse_primary: bool,
        reverse_secondary: bool,
    ) -> list[tuple[int, int]]:
        """Returns a self-collision-free boustrophedon cell traversal."""

        cells: list[tuple[int, int]] = []
        if axis == "row":
            row_range = range(rows - 1, -1, -1) if reverse_primary else range(rows)
            for step, row in enumerate(row_range):
                left_to_right = (step % 2 == 0) != reverse_secondary
                col_range = range(cols) if left_to_right else range(cols - 1, -1, -1)
                for col in col_range:
                    cells.append((row, col))
            return cells

        col_range = range(cols - 1, -1, -1) if reverse_primary else range(cols)
        for step, col in enumerate(col_range):
            bottom_to_top = (step % 2 == 0) != reverse_secondary
            row_range = range(rows) if bottom_to_top else range(rows - 1, -1, -1)
            for row in row_range:
                cells.append((row, col))
        return cells

    def _build_variant_tour(
        self,
        candidate: SnakeGridCandidate,
        cells: dict[tuple[int, int], list[ProjectedPoint]],
        axis: SnakeAxis,
        reverse_primary: bool,
        reverse_secondary: bool,
    ) -> list[int]:
        """Builds a city-level tour for one snake orientation variant."""

        occupied_cells = [
            cell
            for cell in self._snake_cells(
                rows=candidate.rows,
                cols=candidate.cols,
                axis=axis,
                reverse_primary=reverse_primary,
                reverse_secondary=reverse_secondary,
            )
            if cell in cells
        ]

        tour: list[int] = []
        for index, cell in enumerate(occupied_cells):
            previous_center = (
                self._cell_center(candidate, occupied_cells[index - 1])
                if index > 0
                else None
            )
            current_center = self._cell_center(candidate, cell)
            next_center = (
                self._cell_center(candidate, occupied_cells[index + 1])
                if index + 1 < len(occupied_cells)
                else None
            )
            tour.extend(
                self._sort_points_in_cell(
                    points=cells[cell],
                    previous_point=(
                        self.points_by_city_index[tour[-1]] if tour else None
                    ),
                    previous_center=previous_center,
                    current_center=current_center,
                    next_center=next_center,
                    axis=axis,
                )
            )
        return tour

    def _sort_points_in_cell(
        self,
        points: Sequence[ProjectedPoint],
        previous_point: Optional[ProjectedPoint],
        previous_center: Optional[tuple[float, float]],
        current_center: tuple[float, float],
        next_center: Optional[tuple[float, float]],
        axis: SnakeAxis,
    ) -> list[int]:
        """Orders points inside a cell along the local snake travel direction."""

        if previous_center is not None and next_center is not None:
            direction_x = next_center[0] - previous_center[0]
            direction_y = next_center[1] - previous_center[1]
        elif next_center is not None:
            direction_x = next_center[0] - current_center[0]
            direction_y = next_center[1] - current_center[1]
        elif previous_center is not None:
            direction_x = current_center[0] - previous_center[0]
            direction_y = current_center[1] - previous_center[1]
        elif axis == "row":
            direction_x, direction_y = 1.0, 0.0
        else:
            direction_x, direction_y = 0.0, 1.0

        if abs(direction_x) + abs(direction_y) < 1e-12:
            direction_x, direction_y = (1.0, 0.0) if axis == "row" else (0.0, 1.0)

        perpendicular_x = -direction_y
        perpendicular_y = direction_x

        def sorting_key(point: ProjectedPoint) -> tuple[float, float]:
            along_direction = point.x * direction_x + point.y * direction_y
            perpendicular = point.x * perpendicular_x + point.y * perpendicular_y
            return along_direction, perpendicular

        if previous_point is None:
            return [point.city_index for point in sorted(points, key=sorting_key)]

        remaining = list(points)
        ordered_indices: list[int] = []
        current_point = previous_point
        while remaining:
            next_point = min(
                remaining,
                key=lambda point: (
                    (point.x - current_point.x) ** 2 + (point.y - current_point.y) ** 2,
                    sorting_key(point),
                    point.city_index,
                ),
            )
            ordered_indices.append(next_point.city_index)
            remaining.remove(next_point)
            current_point = next_point
        return ordered_indices

    def _adjusted_score(
        self,
        route_cost: float,
        candidate: SnakeGridCandidate,
        cells: dict[tuple[int, int], list[ProjectedPoint]],
    ) -> float:
        """Adds mild deterministic regularization against bad board geometry."""

        total_cells = candidate.rows * candidate.cols
        empty_ratio = max(0.0, (total_cells - len(cells)) / max(total_cells, 1))
        overload_penalty = sum(
            max(0, len(cell_points) - 2) ** 2 for cell_points in cells.values()
        ) / max(self.instance.size, 1)
        fine_grid_penalty = max(0.0, total_cells / max(self.instance.size, 1) - 1.75)
        return route_cost * (
            1.0
            + 0.004 * empty_ratio
            + 0.006 * overload_penalty
            + 0.002 * fine_grid_penalty
        )


class TourInitializers:
    """Initial tour construction heuristics."""

    @staticmethod
    def random_initialization(
        instance: TSPInstance,
        rng: Optional[random.Random] = None,
    ) -> list[int]:
        """Returns a uniformly shuffled permutation of city indices."""

        generator = rng or random.Random()
        tour = list(range(instance.size))
        generator.shuffle(tour)
        return tour

    @staticmethod
    def nearest_neighbor_initialization(
        instance: TSPInstance,
        start_index: int = 0,
    ) -> list[int]:
        """Builds a greedy tour using the nearest-neighbor rule."""

        if not 0 <= start_index < instance.size:
            raise ValueError("start_index is outside the city index range.")

        remaining = set(range(instance.size))
        remaining.remove(start_index)
        tour = [start_index]

        while remaining:
            current_city = tour[-1]
            next_city = min(
                remaining,
                key=lambda candidate: instance.distance_matrix[current_city][candidate],
            )
            tour.append(next_city)
            remaining.remove(next_city)
        return tour

    @staticmethod
    def snake_path_initialization(instance: TSPInstance) -> list[int]:
        """Builds a deterministic grid-based snake path tour."""

        return TourInitializers.snake_path_initialization_result(instance).tour

    @staticmethod
    def snake_path_initialization_result(instance: TSPInstance) -> SnakeInitializationResult:
        """Builds a deterministic snake path tour with board diagnostics."""

        return SnakeGridPathInitializer(instance).build()


class SnakeWindowRepair:
    """Refines a completed snake tour by exactly optimizing small fixed-endpoint windows.

    The key idea is local but stronger than a crossing-only repair:
    each window keeps its first and last city fixed, and we search for the
    best ordering of the internal cities inside that window.
    """

    def __init__(
        self,
        instance: TSPInstance,
        window_size: int = 10,
        stride: Optional[int] = None,
    ) -> None:
        if window_size < 4:
            raise ValueError("window_size must be at least 4.")

        self.instance = instance
        self.window_size = window_size
        self.stride = stride or window_size

    def run(self, initial_tour: Sequence[int],
            deadline: float | None = None) -> WindowRepairResult:
        """Runs the fixed-endpoint window refinement and records playback frames.

        `deadline` (2026-07-28, perf_counter zaman damgası) verilirse pencere
        taraması bu duvar-saati sınırında nazikçe durur; sonuç o ana kadarki
        iyileştirmeleri içerir. Erken çıkış GÜVENLİDİR: her pencere değişimi
        yerinde yapılır ve `working_tour` her adımda geçerli bir permütasyon
        olarak kalır, dolayısıyla hangi noktada kesilirse kesilsin dönen tur
        geçerlidir.

        NEDEN EKLENDİ: `AdaptiveSnakeWindowRepair.run` deadline'ı 2026-07-27'de
        almıştı, bu sınıf ALMAMIŞTI. Sonuç: onarım katmanındaki bütün kipler
        (two_opt / or_opt / relocate / vnd / vnd_dense / window_adaptive) tek
        duvar-saati bütçesine uyarken `window` kipi -- hem ablasyon satırında
        hem havuz yolunda (`runner._repair_pool`) -- SINIRSIZ koşuyordu.
        Bu, iki aileye de aynı biçimde uygulandığı için taraf tutmuyordu ama
        "onarım katmanında TEK bütçe vardır" sözleşmesini bu kip için
        geçersiz kılıyordu. Ayrıntı:
        `HAVUZ_ADALETI_VE_ADAY_GENISLIGI_AKADEMIK_NOT.md` §9.1."""

        self.instance.validate_tour(initial_tour)
        working_tour = list(initial_tour)
        start_time = time.perf_counter()
        current_cost = self.instance.tour_cost(working_tour)
        best_cost = current_cost
        accepted_windows = 0
        windows_scanned = 0

        frames = [
            TourFrame(
                step=0,
                phase="initial",
                stage_index=0,
                elapsed_time=0.0,
                cost=current_cost,
                best_cost=best_cost,
                constructed_cities=self.window_size,
                tour=working_tour[:],
            )
        ]
        history = [
            HistoryPoint(
                iteration=0,
                elapsed_time=0.0,
                cost=best_cost,
            )
        ]

        offsets = self._offsets()
        passes_completed = len(offsets)
        _out_of_time = False
        for pass_index, offset in enumerate(offsets, start=1):
            if _out_of_time:
                break
            for start_index in range(
                offset,
                len(working_tour) - self.window_size + 1,
                self.stride,
            ):
                if deadline is not None and time.perf_counter() >= deadline:
                    sys.stdout.write(
                        f"[Snake Window Repair] deadline "
                        f"({time.perf_counter() - start_time:.1f}s): "
                        f"pass {pass_index}/{passes_completed} yarida kesildi\n")
                    sys.stdout.flush()
                    _out_of_time = True
                    break
                windows_scanned += 1
                if windows_scanned % 100 == 0:
                    sys.stdout.write(f"[Snake Window Repair] Pass {pass_index}/{passes_completed} | Scanned: {windows_scanned} | Accepted: {accepted_windows} | Time: {time.perf_counter() - start_time:.1f}s\\n")
                    sys.stdout.flush()

                end_index = start_index + self.window_size
                segment = working_tour[start_index:end_index]
                new_segment, delta = self._optimize_fixed_endpoint_segment(segment)
                if delta >= -1e-12:
                    continue

                # Only the internal cities change; both window endpoints stay locked.
                working_tour[start_index:end_index] = new_segment
                current_cost += delta
                best_cost = min(best_cost, current_cost)
                accepted_windows += 1

                elapsed_time = time.perf_counter() - start_time
                frames.append(
                    TourFrame(
                        step=len(frames),
                        phase="repair",
                        stage_index=accepted_windows,
                        elapsed_time=elapsed_time,
                        cost=current_cost,
                        best_cost=best_cost,
                        constructed_cities=self.window_size,
                        tour=working_tour[:],
                    )
                )
                history.append(
                    HistoryPoint(
                        iteration=accepted_windows,
                        elapsed_time=elapsed_time,
                        cost=best_cost,
                    )
                )

        elapsed_time = time.perf_counter() - start_time
        frames.append(
            TourFrame(
                step=len(frames),
                phase="final",
                stage_index=accepted_windows,
                elapsed_time=elapsed_time,
                cost=current_cost,
                best_cost=best_cost,
                constructed_cities=self.window_size,
                tour=working_tour[:],
            )
        )
        history.append(
            HistoryPoint(
                iteration=accepted_windows + 1,
                elapsed_time=elapsed_time,
                cost=best_cost,
            )
        )

        return WindowRepairResult(
            tour=working_tour,
            cost=current_cost,
            accepted_windows=accepted_windows,
            passes_completed=passes_completed,
            windows_scanned=windows_scanned,
            frames=frames,
            history=history,
        )

    def _offsets(self) -> tuple[int, ...]:
        """Uses a second half-window pass to catch bends near window boundaries."""

        half_window = self.window_size // 2
        if half_window <= 0 or half_window >= self.window_size:
            return (0,)
        return (0, half_window)

    def _optimize_fixed_endpoint_segment(
        self,
        segment: Sequence[int],
    ) -> tuple[list[int], float]:
        """Optimizes one local segment while keeping the two segment endpoints fixed.

        For a 10-city window this means only the 8 internal cities are permuted.
        That search space is small enough to solve exactly with dynamic programming,
        so each accepted window is the best local path under the fixed endpoints.
        """

        if len(segment) <= 3:
            return list(segment), 0.0

        start_city = segment[0]
        end_city = segment[-1]
        internal_cities = list(segment[1:-1])
        internal_count = len(internal_cities)
        current_cost = self._open_path_cost(segment)

        state_count = 1 << internal_count
        inf = float("inf")

        # dp[mask][j] stores the cheapest path from start_city to internal city j
        # after visiting exactly the cities encoded by mask.
        dp = [[inf for _ in range(internal_count)] for _ in range(state_count)]
        parent = [[-1 for _ in range(internal_count)] for _ in range(state_count)]

        for city_position, city_index in enumerate(internal_cities):
            dp[1 << city_position][city_position] = self.instance.distance_matrix[start_city][
                city_index
            ]

        for mask in range(state_count):
            for last_position in range(internal_count):
                previous_cost = dp[mask][last_position]
                if previous_cost == inf:
                    continue

                last_city = internal_cities[last_position]
                for next_position, next_city in enumerate(internal_cities):
                    if mask & (1 << next_position):
                        continue

                    next_mask = mask | (1 << next_position)
                    candidate_cost = (
                        previous_cost
                        + self.instance.distance_matrix[last_city][next_city]
                    )
                    if candidate_cost < dp[next_mask][next_position]:
                        dp[next_mask][next_position] = candidate_cost
                        parent[next_mask][next_position] = last_position

        full_mask = state_count - 1
        best_cost = inf
        best_last_position = -1
        for last_position, last_city in enumerate(internal_cities):
            candidate_cost = (
                dp[full_mask][last_position]
                + self.instance.distance_matrix[last_city][end_city]
            )
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_last_position = last_position

        if best_last_position < 0:
            return list(segment), 0.0

        delta = best_cost - current_cost
        if delta >= -1e-12:
            return list(segment), 0.0

        ordered_internal: list[int] = []
        mask = full_mask
        last_position = best_last_position
        while last_position != -1:
            ordered_internal.append(internal_cities[last_position])
            previous_position = parent[mask][last_position]
            mask ^= 1 << last_position
            last_position = previous_position

        ordered_internal.reverse()
        return [start_city, *ordered_internal, end_city], delta

    def _open_path_cost(self, segment: Sequence[int]) -> float:
        """Computes the path length of one non-cyclic local window."""

        return sum(
            self.instance.distance_matrix[segment[index]][segment[index + 1]]
            for index in range(len(segment) - 1)
        )


class AdaptiveSnakeWindowRepair(SnakeWindowRepair):
    """Runs a stronger cyclic window refiner on top of the completed snake tour.

    Differences from the plain window refiner:
    1. It uses multiple window sizes instead of only one fixed scale.
    2. It supports wrap-around windows that cross the end/start boundary of the cycle.
    3. When one window improves, nearby windows are re-queued because local changes
       often create fresh opportunities immediately to the left or right.
    """

    def __init__(
        self,
        instance: TSPInstance,
        window_sizes: Sequence[int] = (10, 12),
        requeue_radius: int = 2,
        max_visits_per_window: int = 3,
    ) -> None:
        normalized_sizes = tuple(
            sorted({window_size for window_size in window_sizes if window_size >= 4})
        )
        if not normalized_sizes:
            raise ValueError("At least one window size >= 4 is required.")
        if requeue_radius < 1:
            raise ValueError("requeue_radius must be positive.")
        if max_visits_per_window < 1:
            raise ValueError("max_visits_per_window must be positive.")

        self.instance = instance
        self.window_size = normalized_sizes[0]
        self.stride = max(1, self.window_size // 2)
        self.window_sizes = normalized_sizes
        self.requeue_radius = requeue_radius
        self.max_visits_per_window = max_visits_per_window

    def run(self, initial_tour: Sequence[int],
            deadline: float | None = None) -> AdaptiveWindowRepairResult:
        """Runs the adaptive multi-scale cyclic window refinement and records frames.

        `deadline` (perf_counter zaman damgası) verilirse kuyruk döngüsü bu
        duvar-saati sınırında nazikçe durur; sonuç o ana kadarki iyileştirmeleri
        içerir (çok büyük örneklerde koşucunun süresiz takılmasını önler)."""

        self.instance.validate_tour(initial_tour)
        working_tour = list(initial_tour)
        n_cities = len(working_tour)
        start_time = time.perf_counter()
        current_cost = self.instance.tour_cost(working_tour)
        best_cost = current_cost
        accepted_windows = 0
        windows_scanned = 0
        queued_neighbors = 0

        frames = [
            TourFrame(
                step=0,
                phase="initial",
                stage_index=0,
                elapsed_time=0.0,
                cost=current_cost,
                best_cost=best_cost,
                constructed_cities=self.window_sizes[0],
                tour=working_tour[:],
            )
        ]
        history = [
            HistoryPoint(
                iteration=0,
                elapsed_time=0.0,
                cost=best_cost,
            )
        ]

        pending_windows: deque[tuple[int, int]] = deque()
        scheduled_windows: set[tuple[int, int]] = set()
        visit_counts: dict[tuple[int, int], int] = {}

        def enqueue(window_start: int, window_size: int) -> None:
            key = (window_start % n_cities, window_size)
            if key in scheduled_windows:
                return
            next_visit_count = visit_counts.get(key, 0) + 1
            if next_visit_count > self.max_visits_per_window:
                return
            visit_counts[key] = next_visit_count
            scheduled_windows.add(key)
            pending_windows.append(key)

        # The initial queue covers the whole cycle with overlapping windows.
        for window_size in self.window_sizes:
            stride = max(1, window_size // 2)
            for offset in self._offsets_for_window(window_size):
                for start_index in range(offset, n_cities + offset, stride):
                    enqueue(start_index, window_size)

        while pending_windows:
            if deadline is not None and time.perf_counter() >= deadline:
                sys.stdout.write(
                    f"[Adaptive Repair] deadline ({time.perf_counter() - start_time:.1f}s): "
                    f"kalan {len(pending_windows)} pencere atlanıyor\n")
                break
            window_start, window_size = pending_windows.popleft()
            scheduled_windows.discard((window_start, window_size))
            windows_scanned += 1

            if windows_scanned % 100 == 0:
                sys.stdout.write(f"[Adaptive Repair] Pending: {len(pending_windows):<5} | Scanned: {windows_scanned:<5} | Accepted: {accepted_windows:<5} | Cost: {current_cost:.2f} | Time: {time.perf_counter() - start_time:.1f}s\\n")
                sys.stdout.flush()

            segment = self._extract_cyclic_segment(working_tour, window_start, window_size)
            new_segment, delta = self._optimize_fixed_endpoint_segment(segment)
            if delta >= -1e-12:
                continue

            working_tour = self._replace_cyclic_segment(
                working_tour,
                window_start,
                new_segment,
            )
            current_cost += delta
            best_cost = min(best_cost, current_cost)
            accepted_windows += 1

            elapsed_time = time.perf_counter() - start_time
            frames.append(
                TourFrame(
                    step=len(frames),
                    phase="repair",
                    stage_index=accepted_windows,
                    elapsed_time=elapsed_time,
                    cost=current_cost,
                    best_cost=best_cost,
                    constructed_cities=window_size,
                    tour=working_tour[:],
                )
            )
            history.append(
                HistoryPoint(
                    iteration=accepted_windows,
                    elapsed_time=elapsed_time,
                    cost=best_cost,
                )
            )

            # Re-queue neighboring windows because an accepted local permutation often
            # changes the best choice just outside the updated segment.
            for candidate_size in self.window_sizes:
                neighbor_stride = max(1, candidate_size // 2)
                for shift in range(-self.requeue_radius, self.requeue_radius + 1):
                    neighbor_start = window_start + shift * neighbor_stride
                    enqueue(neighbor_start, candidate_size)
                    queued_neighbors += 1

        elapsed_time = time.perf_counter() - start_time
        frames.append(
            TourFrame(
                step=len(frames),
                phase="final",
                stage_index=accepted_windows,
                elapsed_time=elapsed_time,
                cost=current_cost,
                best_cost=best_cost,
                constructed_cities=self.window_sizes[-1],
                tour=working_tour[:],
            )
        )
        history.append(
            HistoryPoint(
                iteration=accepted_windows + 1,
                elapsed_time=elapsed_time,
                cost=best_cost,
            )
        )

        return AdaptiveWindowRepairResult(
            tour=working_tour,
            cost=current_cost,
            accepted_windows=accepted_windows,
            windows_scanned=windows_scanned,
            queued_neighbors=queued_neighbors,
            frames=frames,
            history=history,
        )

    @staticmethod
    def _offsets_for_window(window_size: int) -> tuple[int, ...]:
        """Returns coarse offsets so the first sweep covers window boundaries better."""

        half_window = window_size // 2
        if half_window <= 0 or half_window >= window_size:
            return (0,)
        return (0, half_window)

    @staticmethod
    def _extract_cyclic_segment(
        tour: Sequence[int],
        start_index: int,
        window_size: int,
    ) -> list[int]:
        """Extracts one local window, allowing the window to wrap around the tour end."""

        n_cities = len(tour)
        return [tour[(start_index + offset) % n_cities] for offset in range(window_size)]

    @staticmethod
    def _replace_cyclic_segment(
        tour: Sequence[int],
        start_index: int,
        segment: Sequence[int],
    ) -> list[int]:
        """Writes one optimized window back into the cyclic tour."""

        updated_tour = list(tour)
        n_cities = len(updated_tour)
        for offset, city_index in enumerate(segment):
            updated_tour[(start_index + offset) % n_cities] = city_index
        return updated_tour


class TSPVisualizer:
    """Visualization utilities for routes and convergence behavior."""

    @staticmethod
    def plot_routes(
        instance: TSPInstance,
        initial_tour: Sequence[int],
        final_tour: Sequence[int],
        title_prefix: str = "TSP",
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> None:
        """Plots the initial and final tours side by side."""

        plt.style.use("seaborn-v0_8-whitegrid")
        figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        TSPVisualizer._draw_route(
            axes[0],
            instance,
            initial_tour,
            title=f"{title_prefix} - Initial Tour",
            color="#d62728",
        )
        TSPVisualizer._draw_route(
            axes[1],
            instance,
            final_tour,
            title=f"{title_prefix} - Final Tour",
            color="#1f77b4",
        )

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(figure)

    @staticmethod
    def plot_convergence_curve(
        history: Sequence[HistoryPoint],
        title: str = "Route Cost Trace",
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> None:
        """Plots convergence by iteration and elapsed time."""

        if not history:
            raise ValueError("History is empty. There is nothing to plot.")

        plt.style.use("seaborn-v0_8-whitegrid")
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        iterations = [point.iteration for point in history]
        elapsed_times = [point.elapsed_time for point in history]
        costs = [point.cost for point in history]

        axes[0].plot(iterations, costs, color="#2ca02c", linewidth=2.0)
        axes[0].set_title(f"{title} - Cost vs Iteration")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Cost")

        axes[1].plot(elapsed_times, costs, color="#ff7f0e", linewidth=2.0)
        axes[1].set_title(f"{title} - Cost vs Time")
        axes[1].set_xlabel("CPU Time (s)")
        axes[1].set_ylabel("Cost")

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(figure)

    @staticmethod
    def _draw_route(
        axis: plt.Axes,
        instance: TSPInstance,
        tour: Sequence[int],
        title: str,
        color: str,
    ) -> None:
        """Draws one closed route on a scatter plot."""

        x_values, y_values = instance.route_coordinates(tour)
        axis.plot(x_values, y_values, "-o", color=color, linewidth=1.3, markersize=4)

        start_x, start_y = instance.plot_coordinate(tour[0])
        axis.scatter([start_x], [start_y], color="black", s=70, zorder=5, label="Start")
        axis.set_title(title)
        axis.set_xlabel(instance.axis_labels[0])
        axis.set_ylabel(instance.axis_labels[1])
        axis.legend()


def haversine_distance(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    """Computes the great-circle distance in kilometers."""

    lat_1_rad, lon_1_rad = math.radians(lat_1), math.radians(lon_1)
    lat_2_rad, lon_2_rad = math.radians(lat_2), math.radians(lon_2)
    delta_lat = lat_2_rad - lat_1_rad
    delta_lon = lon_2_rad - lon_1_rad

    haversine_term = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_1_rad) * math.cos(lat_2_rad) * math.sin(delta_lon / 2.0) ** 2
    )
    arc = 2.0 * math.atan2(math.sqrt(haversine_term), math.sqrt(1.0 - haversine_term))
    return EARTH_RADIUS_KM * arc


def euclidean_distance(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
    """Computes the Euclidean distance between two points."""

    return math.hypot(x_2 - x_1, y_2 - y_1)


def tour_cost_from_matrix(
    tour: Sequence[int],
    dist_matrix: Sequence[Sequence[float]],
) -> float:
    """Computes the length of a closed tour from a distance matrix."""

    return sum(
        dist_matrix[tour[index]][tour[(index + 1) % len(tour)]]
        for index in range(len(tour))
    )


def random_initialization(
    instance: TSPInstance,
    rng: Optional[random.Random] = None,
) -> list[int]:
    """Convenience wrapper for the random initializer."""

    return TourInitializers.random_initialization(instance, rng)


def nearest_neighbor_initialization(
    instance: TSPInstance,
    start_index: int = 0,
) -> list[int]:
    """Convenience wrapper for the nearest-neighbor initializer."""

    return TourInitializers.nearest_neighbor_initialization(instance, start_index)


def snake_path_initialization(instance: TSPInstance) -> list[int]:
    """Convenience wrapper for the deterministic snake-path initializer."""

    return TourInitializers.snake_path_initialization(instance)


def snake_path_initialization_result(instance: TSPInstance) -> SnakeInitializationResult:
    """Convenience wrapper returning snake-path diagnostics."""

    return TourInitializers.snake_path_initialization_result(instance)


def sampled_prefix_sizes(size: int, frame_count: int) -> list[int]:
    """Returns deterministic prefix sizes for route-construction playback."""

    if frame_count < 2:
        return [size]

    prefix_sizes = {
        max(1, round(1 + (size - 1) * step / (frame_count - 1)))
        for step in range(frame_count)
    }
    prefix_sizes.add(size)
    return sorted(prefix_sizes)


def build_progressive_tour(final_tour: Sequence[int], prefix_size: int) -> list[int]:
    """Builds a full permutation where the first prefix follows the target route."""

    prefix = list(final_tour[:prefix_size])
    prefix_set = set(prefix)
    remaining = [
        city_index for city_index in range(len(final_tour)) if city_index not in prefix_set
    ]
    return prefix + remaining


def build_sampled_initialization_result(
    instance: TSPInstance,
    name: str,
    final_tour: Sequence[int],
    frame_count: int,
) -> TSPRunResult:
    """Creates a dashboard-compatible result for sampled construction playback."""

    start_time = time.perf_counter()
    full_final_tour = list(final_tour)
    first_tour = build_progressive_tour(full_final_tour, 1)
    first_cost = instance.tour_cost(first_tour)
    final_cost = instance.tour_cost(full_final_tour)

    frames: list[TourFrame] = []
    history: list[HistoryPoint] = []
    best_cost = first_cost

    prefix_sizes = sampled_prefix_sizes(instance.size, frame_count)
    for step, prefix_size in enumerate(prefix_sizes):
        tour = build_progressive_tour(full_final_tour, prefix_size)
        cost = instance.tour_cost(tour)
        best_cost = min(best_cost, cost)
        elapsed_time = time.perf_counter() - start_time
        phase = "initial" if step == 0 else "construction"
        if prefix_size == instance.size:
            phase = "final"

        frames.append(
            TourFrame(
                step=step,
                phase=phase,
                stage_index=step,
                elapsed_time=elapsed_time,
                cost=cost,
                best_cost=best_cost,
                constructed_cities=prefix_size,
                tour=tour,
            )
        )
        history.append(
            HistoryPoint(
                iteration=step,
                elapsed_time=elapsed_time,
                cost=best_cost,
            )
        )

    metrics = RunMetrics(
        initialization_name=name,
        initial_cost=first_cost,
        final_cost=final_cost,
        cpu_time_seconds=time.perf_counter() - start_time,
        frame_count=len(prefix_sizes),
        history=history,
    )
    return TSPRunResult(
        initial_tour=first_tour,
        final_tour=full_final_tour,
        metrics=metrics,
        animation_frames=frames,
    )


def build_window_repair_result(
    instance: TSPInstance,
    snake_tour: Sequence[int],
    window_size: int = 10,
    deadline: float | None = None,
) -> tuple[TSPRunResult, WindowRepairResult]:
    """Builds a dashboard-compatible result for the exact local window refiner.

    `deadline` (2026-07-28): `build_adaptive_window_repair_result` ile AYNI
    sözleşme -- bkz. `SnakeWindowRepair.run`."""

    repair = SnakeWindowRepair(
        instance=instance,
        window_size=window_size,
    ).run(snake_tour, deadline=deadline)

    metrics = RunMetrics(
        initialization_name="Snake Window Repair",
        initial_cost=instance.tour_cost(snake_tour),
        final_cost=repair.cost,
        cpu_time_seconds=repair.frames[-1].elapsed_time,
        frame_count=len(repair.frames),
        history=repair.history,
    )
    return (
        TSPRunResult(
            initial_tour=list(snake_tour),
            final_tour=repair.tour,
            metrics=metrics,
            animation_frames=repair.frames,
        ),
        repair,
    )


def build_adaptive_window_repair_result(
    instance: TSPInstance,
    snake_tour: Sequence[int],
    window_sizes: Sequence[int] = (10, 12),
    requeue_radius: int = 1,
    deadline: float | None = None,
) -> tuple[TSPRunResult, AdaptiveWindowRepairResult]:
    """Builds a dashboard-compatible result for the adaptive cyclic window refiner."""

    repair = AdaptiveSnakeWindowRepair(
        instance=instance,
        window_sizes=window_sizes,
        requeue_radius=requeue_radius,
    ).run(snake_tour, deadline=deadline)

    metrics = RunMetrics(
        initialization_name="Adaptive Snake Window Repair",
        initial_cost=instance.tour_cost(snake_tour),
        final_cost=repair.cost,
        cpu_time_seconds=repair.frames[-1].elapsed_time,
        frame_count=len(repair.frames),
        history=repair.history,
    )
    return (
        TSPRunResult(
            initial_tour=list(snake_tour),
            final_tour=repair.tour,
            metrics=metrics,
            animation_frames=repair.frames,
        ),
        repair,
    )

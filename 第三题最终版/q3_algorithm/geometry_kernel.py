"""Compatibility geometry kernel for the historical Question 3 v5 solver.

The delivered ``第三问_核心算法.py`` came from an older experiment tree and
expects the small NumPy based API in this module.  This file reconstructs that
API from the problem invariants used by the solver:

* sources lie in the closed 1800 m target disk;
* a positive radio observation is at most 1500 m from the source;
* a negative observation is more than 1000 m from the source;
* a reported bearing has physical error at most one degree and is rounded to
  0.01 degree by the protocol.

All polygon approximations are *outer* approximations.  They may delay a clear
or add a measurement, but they must not certify a clear after discarding a
position that is consistent with the observations.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


TARGET_RADIUS = 1800.0
MIN_RECEPTION_RADIUS = 1000.0
MAX_RECEPTION_RADIUS = 1500.0
NEAR_RADIUS = 5.0
CLEAR_RADIUS = 20.0

# One degree physical error plus half of the protocol's 0.01-degree rounding
# unit.  This is the conservative value documented for the historical v5 run.
ERROR = 1.005

CIRCLE_SIDES = 72
GEOMETRY_TOL = 1.0e-7


def _points(values: Iterable[Sequence[float]], *, name: str = "points") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim == 1 and array.shape == (2,):
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return array.copy()


def _point(value: Sequence[float], *, name: str = "point") -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional point")
    return array.copy()


def _ordered_unique(vertices: np.ndarray, tol: float = GEOMETRY_TOL) -> np.ndarray:
    """Remove only adjacent duplicates, preserving convex cyclic order."""

    if len(vertices) == 0:
        return np.empty((0, 2), dtype=float)
    output = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - output[-1]) > tol:
            output.append(vertex)
    if len(output) > 1 and np.linalg.norm(output[0] - output[-1]) <= tol:
        output.pop()
    return np.asarray(output, dtype=float).reshape(-1, 2)


def _convex_hull(values: Iterable[Sequence[float]], tol: float = GEOMETRY_TOL) -> np.ndarray:
    """Return the counter-clockwise hull of a small two-dimensional point set."""

    raw = _points(values)
    if len(raw) <= 1:
        return raw
    ordered: list[np.ndarray] = []
    for vertex in sorted(raw, key=lambda item: (float(item[0]), float(item[1]))):
        if not ordered or np.linalg.norm(vertex - ordered[-1]) > tol:
            ordered.append(vertex)
    if len(ordered) <= 1:
        return np.asarray(ordered, dtype=float).reshape(-1, 2)

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        a = first - origin
        b = second - origin
        return float(a[0] * b[1] - a[1] * b[0])

    lower: list[np.ndarray] = []
    for vertex in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], vertex) <= tol:
            lower.pop()
        lower.append(vertex)
    upper: list[np.ndarray] = []
    for vertex in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], vertex) <= tol:
            upper.pop()
        upper.append(vertex)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float).reshape(-1, 2)


def clip_polygon(
    polygon: Iterable[Sequence[float]],
    A: Iterable[Sequence[float]],
    b: Iterable[float] | float,
    tol: float = GEOMETRY_TOL,
) -> np.ndarray:
    """Clip an ordered convex polygon by the half-planes ``A @ x <= b``."""

    vertices = _points(polygon, name="polygon")
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim == 1 and matrix.shape == (2,):
        matrix = matrix.reshape(1, 2)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("A must have shape (m, 2) and contain finite values")
    bounds = np.asarray(b, dtype=float)
    if bounds.ndim == 0:
        bounds = bounds.reshape(1)
    if bounds.shape != (len(matrix),) or not np.all(np.isfinite(bounds)):
        raise ValueError("b must contain one finite bound per row of A")

    for normal, bound in zip(matrix, bounds):
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise ValueError("half-plane normals must be non-zero")
        normal = normal / norm
        bound = float(bound) / norm
        if len(vertices) == 0:
            break
        output: list[np.ndarray] = []
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            start_value = float(normal @ start - bound)
            end_value = float(normal @ end - bound)
            start_inside = start_value <= tol
            end_inside = end_value <= tol
            if start_inside:
                output.append(start.copy())
            if start_inside != end_inside:
                denominator = start_value - end_value
                if abs(denominator) > 1.0e-15:
                    fraction = min(1.0, max(0.0, start_value / denominator))
                    output.append(start + fraction * (end - start))
        vertices = _ordered_unique(np.asarray(output, dtype=float).reshape(-1, 2), tol)
    return vertices


def bearing_halfplanes(
    station: Sequence[float],
    bearing_deg: float,
    error_deg: float = ERROR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two forward-wedge inequalities for a bearing observation."""

    origin = _point(station, name="station")
    bearing = float(bearing_deg)
    error = float(error_deg)
    if not math.isfinite(bearing) or not math.isfinite(error):
        raise ValueError("bearing and error must be finite")
    if not 0.0 < error < 90.0:
        raise ValueError("error_deg must lie in (0, 90)")
    lower = math.radians(bearing - error)
    upper = math.radians(bearing + error)
    A = np.asarray(
        [
            [math.sin(lower), -math.cos(lower)],
            [-math.sin(upper), math.cos(upper)],
        ],
        dtype=float,
    )
    return A, A @ origin


def _circle_normals(sides: int = CIRCLE_SIDES) -> np.ndarray:
    if sides < 12:
        raise ValueError("a conservative circle polygon needs at least 12 sides")
    angles = np.arange(sides, dtype=float) * (2.0 * math.pi / sides)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _circumscribed_circle_polygon(
    center: Sequence[float], radius: float, sides: int = CIRCLE_SIDES
) -> np.ndarray:
    center_array = _point(center, name="center")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    angles = (np.arange(sides, dtype=float) + 0.5) * (2.0 * math.pi / sides)
    vertex_radius = radius / math.cos(math.pi / sides)
    return center_array + vertex_radius * np.column_stack((np.cos(angles), np.sin(angles)))


def initial_polygon(station: Sequence[float], bearing_deg: float) -> np.ndarray:
    """Build the conservative first-positive feasible polygon.

    The region is the target disk outer polygon, clipped by the measured
    bearing wedge, a 1500 m reception-disk outer polygon, and the conservative
    linear consequence of a non-``near`` result.
    """

    origin = _point(station, name="station")
    polygon = _circumscribed_circle_polygon((0.0, 0.0), TARGET_RADIUS)
    wedge_A, wedge_b = bearing_halfplanes(origin, bearing_deg, ERROR)
    polygon = clip_polygon(polygon, wedge_A, wedge_b)
    normals = _circle_normals()
    polygon = clip_polygon(
        polygon,
        normals,
        normals @ origin + MAX_RECEPTION_RADIUS,
    )
    angle = math.radians(float(bearing_deg) % 360.0)
    direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
    minimum_projection = NEAR_RADIUS * math.cos(math.radians(ERROR))
    polygon = clip_polygon(
        polygon,
        -direction,
        -float(direction @ origin) - minimum_projection,
    )
    if len(polygon) == 0:
        raise RuntimeError("positive bearing produced an empty initial feasible region")
    return polygon


def _pair_circle(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, float]:
    center = 0.5 * (first + second)
    return center, float(np.linalg.norm(first - center))


def _three_point_circle(
    first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> tuple[np.ndarray, float] | None:
    ax, ay = first
    bx, by = second
    cx, cy = third
    determinant = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    scale = max(1.0, *(abs(value) for value in (ax, ay, bx, by, cx, cy)))
    if abs(determinant) <= 1.0e-13 * scale * scale:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    center = np.asarray(
        [
            (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / determinant,
            (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / determinant,
        ],
        dtype=float,
    )
    return center, float(np.linalg.norm(center - first))


def _covers(center: np.ndarray, radius: float, values: np.ndarray) -> bool:
    if len(values) == 0:
        return True
    distances = np.linalg.norm(values - center, axis=1)
    return bool(np.max(distances) <= radius + GEOMETRY_TOL * max(1.0, radius))


def _circle_for_three(
    first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> tuple[np.ndarray, float]:
    values = np.asarray([first, second, third], dtype=float)
    candidates = []
    for left, right in ((first, second), (first, third), (second, third)):
        center, radius = _pair_circle(left, right)
        if _covers(center, radius, values):
            candidates.append((radius, float(center[0]), float(center[1]), center))
    triple = _three_point_circle(first, second, third)
    if triple is not None and _covers(triple[0], triple[1], values):
        center, radius = triple
        candidates.append((radius, float(center[0]), float(center[1]), center))
    if not candidates:
        raise RuntimeError("could not construct an enclosing circle for three points")
    radius, _, _, center = min(candidates)
    return center.copy(), float(radius)


def minimum_circle(polygon: Iterable[Sequence[float]]) -> dict[str, np.ndarray | float]:
    """Return the exact minimum enclosing circle of the polygon vertices.

    The deterministic incremental algorithm is cubic only in the worst case;
    the historical polygons have at most a few dozen vertices.  The reported
    radius is finally expanded to the actual farthest-vertex distance so that
    round-off cannot create an unsafe clearance certificate.
    """

    points = _points(polygon, name="polygon")
    if len(points) == 0:
        raise ValueError("minimum_circle requires at least one point")
    points = _convex_hull(points)
    center = points[0].copy()
    radius = 0.0
    for first_index, first in enumerate(points):
        if np.linalg.norm(first - center) <= radius + GEOMETRY_TOL:
            continue
        center = first.copy()
        radius = 0.0
        for second_index in range(first_index):
            second = points[second_index]
            if np.linalg.norm(second - center) <= radius + GEOMETRY_TOL:
                continue
            center, radius = _pair_circle(first, second)
            for third_index in range(second_index):
                third = points[third_index]
                if np.linalg.norm(third - center) <= radius + GEOMETRY_TOL:
                    continue
                center, radius = _circle_for_three(first, second, third)
    # Expanding the radius by measured floating-point error is conservative and
    # avoids claiming a 20 m certificate while one vertex is a few ulps beyond.
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    return {"center": center.copy(), "radius": radius}


def _segment_circle_intersections(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    delta = end - start
    relative = start - center
    quadratic = float(delta @ delta)
    if quadratic <= 1.0e-24:
        return []
    linear = 2.0 * float(relative @ delta)
    constant = float(relative @ relative) - radius * radius
    discriminant = linear * linear - 4.0 * quadratic * constant
    scale = max(1.0, linear * linear, abs(4.0 * quadratic * constant))
    if discriminant < -1.0e-13 * scale:
        return []
    root = math.sqrt(max(0.0, discriminant))
    result = []
    for fraction in ((-linear - root) / (2.0 * quadratic), (-linear + root) / (2.0 * quadratic)):
        if -GEOMETRY_TOL <= fraction <= 1.0 + GEOMETRY_TOL:
            fraction = min(1.0, max(0.0, fraction))
            result.append(start + fraction * delta)
    return result


def exclude_disk(
    polygon: Iterable[Sequence[float]],
    center: Sequence[float],
    radius: float = CLEAR_RADIUS,
) -> np.ndarray:
    """Conservatively convexify ``polygon \\ open_disk(center, radius)``.

    A failed clear or a negative radio observation proves that the source is
    outside a disk.  The exact remainder can be non-convex, whereas the legacy
    solver stores one convex polygon.  Its convex hull is generated from all
    surviving polygon vertices and all polygon-edge/circle intersections.
    """

    vertices = _points(polygon, name="polygon")
    disk_center = _point(center, name="center")
    radius_value = float(radius)
    if not math.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError("radius must be finite and positive")
    if len(vertices) == 0:
        return vertices

    # Exclude a microscopically smaller open disk.  This makes the numerical
    # operation an outer approximation even for points on the analytic circle.
    effective_radius = max(0.0, radius_value - GEOMETRY_TOL)
    candidates: list[np.ndarray] = [
        vertex.copy()
        for vertex in vertices
        if np.linalg.norm(vertex - disk_center) >= effective_radius
    ]
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        candidates.extend(
            _segment_circle_intersections(start, end, disk_center, effective_radius)
        )
    if not candidates:
        return np.empty((0, 2), dtype=float)
    return _convex_hull(candidates)


def _positive_negative_halfplanes(
    positives: np.ndarray, negatives: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normals: list[np.ndarray] = []
    bounds: list[float] = []
    for positive in positives:
        for negative in negatives:
            delta = negative - positive
            if np.linalg.norm(delta) <= GEOMETRY_TOL:
                raise RuntimeError("one station cannot be both positive and negative")
            normals.append(delta)
            # |G-P| <= |G-N|, rearranged without subtracting two large norms.
            bounds.append(0.5 * float(delta @ (negative + positive)))
    return np.asarray(normals, dtype=float).reshape(-1, 2), np.asarray(bounds, dtype=float)


def negative_refine(
    polygon: Iterable[Sequence[float]],
    negative_stations: Iterable[Sequence[float]],
    positive_stations: Iterable[Sequence[float]],
    mode: str = "both",
) -> np.ndarray:
    """Apply conservative information carried by Q3 ``no_signal`` results.

    ``mode='both'`` applies (1) every valid positive/negative distance-bisector
    half-plane and (2) the outside-of-1000-m-disk fact using :func:`exclude_disk`.
    Alternate mode names are accepted for compatibility with old experiments.
    """

    result = _points(polygon, name="polygon")
    negatives = _points(negative_stations, name="negative_stations")
    positives = _points(positive_stations, name="positive_stations")
    selected = str(mode).lower()
    use_linear = selected in {"both", "halfplane", "halfplanes", "linear", "bisector"}
    use_disk = selected in {"both", "disk", "disks"}
    if selected in {"none", "off", "false"}:
        return result
    if not use_linear and not use_disk:
        raise ValueError(f"unsupported negative refinement mode: {mode!r}")
    if len(result) == 0:
        raise RuntimeError("cannot refine an empty feasible polygon")

    if use_linear and len(positives) and len(negatives):
        A, b = _positive_negative_halfplanes(positives, negatives)
        result = clip_polygon(result, A, b)
        if len(result) == 0:
            raise RuntimeError("positive/negative observations made the feasible region empty")
    if use_disk:
        for station in negatives:
            result = exclude_disk(result, station, MIN_RECEPTION_RADIUS)
            if len(result) == 0:
                raise RuntimeError("negative observation made the feasible region empty")
    return result


def _line_target_circle_intersections(
    normal: np.ndarray, bound: float, radius: float
) -> list[np.ndarray]:
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-15:
        return []
    unit = normal / length
    distance = float(bound) / length
    if abs(distance) > radius + GEOMETRY_TOL:
        return []
    foot = distance * unit
    tangent = np.asarray([-unit[1], unit[0]], dtype=float)
    offset = math.sqrt(max(0.0, radius * radius - distance * distance))
    return [foot + offset * tangent, foot - offset * tangent]


def _circumcenter(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray | None:
    circle = _three_point_circle(first, second, third)
    return None if circle is None else circle[0]


def continuous_cover(
    stations: Iterable[Sequence[float]],
    target_radius: float = TARGET_RADIUS,
    cover_radius: float = MIN_RECEPTION_RADIUS,
) -> bool:
    """Certify continuous target-disk coverage by equal station disks.

    This computes the largest distance to the nearest station over the complete
    target disk.  A maximum occurs at a Voronoi vertex in the interior, or on
    the target-circle boundary at a Voronoi break point or a stationary point.
    Enumerating triple circumcentres, pair-bisector/boundary intersections and
    per-station antipodes therefore gives a continuous certificate rather than
    a grid sample.
    """

    points = _points(stations, name="stations")
    target = float(target_radius)
    cover = float(cover_radius)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_radius must be finite and positive")
    if not math.isfinite(cover) or cover <= 0.0:
        raise ValueError("cover_radius must be finite and positive")
    if len(points) == 0:
        return False

    # Remove duplicate sites; duplicates do not alter the Voronoi diagram.
    unique: list[np.ndarray] = []
    for station in points:
        if all(np.linalg.norm(station - existing) > GEOMETRY_TOL for existing in unique):
            unique.append(station)
    points = np.asarray(unique, dtype=float)

    candidates: list[np.ndarray] = [np.zeros(2, dtype=float)]
    candidates.extend(
        np.asarray(
            [target, 0.0]
            if np.linalg.norm(station) <= GEOMETRY_TOL
            else -target * station / np.linalg.norm(station),
            dtype=float,
        )
        for station in points
    )
    # Cardinal points make the one-centre and heavily degenerate cases explicit.
    candidates.extend(
        np.asarray(value, dtype=float)
        for value in ((target, 0.0), (-target, 0.0), (0.0, target), (0.0, -target))
    )
    for first, second in combinations(points, 2):
        normal = second - first
        bound = 0.5 * float(normal @ (second + first))
        candidates.extend(_line_target_circle_intersections(normal, bound, target))
    for first, second, third in combinations(points, 3):
        center = _circumcenter(first, second, third)
        if center is not None and np.linalg.norm(center) <= target + GEOMETRY_TOL:
            candidates.append(center)

    candidate_array = np.asarray(candidates, dtype=float).reshape(-1, 2)
    nearest_distances = np.min(
        np.linalg.norm(candidate_array[:, None, :] - points[None, :, :], axis=2),
        axis=1,
    )
    covering_radius = float(np.max(nearest_distances))
    return covering_radius <= cover + GEOMETRY_TOL


def _equal_circle_intersections(
    first: np.ndarray, second: np.ndarray, radius: float
) -> list[np.ndarray]:
    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-15 or distance > 2.0 * radius + GEOMETRY_TOL:
        return []
    half = 0.5 * distance
    height = math.sqrt(max(0.0, radius * radius - half * half))
    unit = delta / distance
    middle = first + half * unit
    perpendicular = np.asarray([-unit[1], unit[0]], dtype=float)
    if height <= GEOMETRY_TOL:
        return [middle]
    return [middle + height * perpendicular, middle - height * perpendicular]


def project_to_disks(
    point: Sequence[float],
    centers: Iterable[Sequence[float]],
    radius: float = 990.0,
) -> np.ndarray:
    """Project a point onto the intersection of equal closed disks.

    In two dimensions the closest feasible point is the point itself, a radial
    projection on one active circle, or an intersection of two active circles.
    All candidates are checked against every disk.  The centres' minimum-circle
    centre supplies a guaranteed feasible fallback whenever the intersection is
    non-empty.
    """

    current = _point(point)
    disk_centers = _points(centers, name="centers")
    radius_value = float(radius)
    if len(disk_centers) == 0:
        raise ValueError("centers must not be empty")
    if not math.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError("radius must be finite and positive")
    unique: list[np.ndarray] = []
    for center in disk_centers:
        if all(np.linalg.norm(center - existing) > GEOMETRY_TOL for existing in unique):
            unique.append(center)
    disk_centers = np.asarray(unique, dtype=float)

    def maximum_distance(candidate: np.ndarray) -> float:
        return float(np.max(np.linalg.norm(disk_centers - candidate, axis=1)))

    fallback_circle = minimum_circle(disk_centers)
    fallback = np.asarray(fallback_circle["center"], dtype=float)
    if maximum_distance(fallback) > radius_value + GEOMETRY_TOL:
        raise ValueError("the equal-disk intersection is empty")
    if maximum_distance(current) <= radius_value:
        return current

    candidates: list[np.ndarray] = [fallback]
    for center in disk_centers:
        delta = current - center
        distance = float(np.linalg.norm(delta))
        if distance > 1.0e-15:
            candidates.append(center + radius_value * delta / distance)
    for first, second in combinations(disk_centers, 2):
        candidates.extend(_equal_circle_intersections(first, second, radius_value))
    feasible = [
        candidate
        for candidate in candidates
        if maximum_distance(candidate) <= radius_value + GEOMETRY_TOL
    ]
    if not feasible:
        raise RuntimeError("failed to recover a feasible equal-disk projection")
    chosen = min(
        feasible,
        key=lambda candidate: (
            float(np.linalg.norm(candidate - current)),
            float(candidate[0]),
            float(candidate[1]),
        ),
    ).copy()
    if maximum_distance(chosen) <= radius_value:
        return chosen

    # Retreat a boundary candidate toward a known feasible point to remove a
    # possible few-ulp violation without materially changing the projection.
    if maximum_distance(fallback) > radius_value:
        return fallback
    low, high = 0.0, 1.0
    for _ in range(60):
        fraction = 0.5 * (low + high)
        trial = fallback + fraction * (chosen - fallback)
        if maximum_distance(trial) <= radius_value:
            low = fraction
        else:
            high = fraction
    return fallback + low * (chosen - fallback)


def first_route_target(
    position: Sequence[float],
    known: Mapping[int, Mapping[str, object]],
    end_costs: Sequence[float] | None = None,
) -> int:
    """Return the first channel of a short deterministic open source route.

    This helper serves the historical ``route_mode='v4'`` fallback (the adopted
    v5 configuration uses its own joint route).  Every channel is tried as the
    first seed, the remainder is appended by nearest neighbour, and open 2-opt
    improves the route while retaining an optional terminal cost.
    """

    start = _point(position, name="position")
    channels = sorted(int(channel) for channel in known)
    if not channels:
        raise ValueError("known must contain at least one source")
    centers = {channel: _point(known[channel]["center"], name="source center") for channel in channels}
    if end_costs is None:
        terminal = {channel: 0.0 for channel in channels}
    else:
        values = np.asarray(end_costs, dtype=float)
        if values.shape != (len(channels),) or not np.all(np.isfinite(values)):
            raise ValueError("end_costs must align with sorted known channels")
        terminal = dict(zip(channels, map(float, values)))

    def route_cost(route: Sequence[int]) -> float:
        total = float(np.linalg.norm(start - centers[route[0]]))
        total += sum(
            float(np.linalg.norm(centers[left] - centers[right]))
            for left, right in zip(route, route[1:])
        )
        return total + terminal[route[-1]]

    routes: list[tuple[float, tuple[int, ...]]] = []
    for first in channels:
        route = [first]
        remaining = set(channels) - {first}
        while remaining:
            following = min(
                remaining,
                key=lambda channel: (
                    float(np.linalg.norm(centers[route[-1]] - centers[channel])),
                    channel,
                ),
            )
            route.append(following)
            remaining.remove(following)
        improved = True
        while improved:
            improved = False
            baseline = route_cost(route)
            for left in range(len(route) - 1):
                for right in range(left + 1, len(route)):
                    candidate = route[:left] + list(reversed(route[left : right + 1])) + route[right + 1 :]
                    if route_cost(candidate) < baseline - 1.0e-8:
                        route = candidate
                        improved = True
                        break
                if improved:
                    break
        routes.append((route_cost(route), tuple(route)))
    return min(routes, key=lambda item: (item[0], item[1]))[1][0]


__all__ = [
    "ERROR",
    "bearing_halfplanes",
    "clip_polygon",
    "continuous_cover",
    "exclude_disk",
    "first_route_target",
    "initial_polygon",
    "minimum_circle",
    "negative_refine",
    "project_to_disks",
]

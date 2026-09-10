"""Robust second-station selection for CUMCM 2026 Problem B, Question 2.

The first bearing and the known target/reception limits define a convex source
set K1.  A second station is guaranteed to receive the omnidirectional source
when it lies within 1000 m of every point of K1.  For the polygonal outer
approximation used here, it is sufficient and necessary to check all vertices.

Among guaranteed-visible candidates, the strategy maximizes the worst acute
intersection angle over a deterministic sample of K1.  Movement distance is a
tie-breaker.  The two sides of the first bearing are retained as alternative
candidates so later routing constraints can choose between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, pi, sin
from typing import Sequence

import numpy as np

from question1_geometry import (
    Circle,
    HalfPlane,
    Observation,
    Point,
    clip_polygon,
    minimum_enclosing_circle,
    observation_halfplanes,
    polygon_diameter,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    point: Point
    guaranteed_visible: bool
    maximum_source_distance_m: float
    minimum_intersection_angle_deg: float
    travel_distance_m: float


@dataclass(frozen=True)
class CandidateGrid:
    x_values: np.ndarray
    y_values: np.ndarray
    angle_scores_deg: np.ndarray
    feasible_mask: np.ndarray
    positive_candidate: CandidateEvaluation
    negative_candidate: CandidateEvaluation
    feasible_count: int


@dataclass(frozen=True)
class PosteriorEvaluation:
    worst_diameter_m: float
    worst_source: Point
    worst_measurement_error_deg: float
    worst_polygon: tuple[Point, ...]


def _unit(angle_deg: float) -> Point:
    angle = (angle_deg % 360.0) * pi / 180.0
    return Point(cos(angle), sin(angle))


def circumscribed_circle_halfplanes(
    center: Point,
    radius_m: float,
    sides: int = 180,
) -> tuple[HalfPlane, ...]:
    """Return tangent half-planes whose regular polygon contains the circle."""

    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    if sides < 12:
        raise ValueError("At least 12 sides are required.")
    planes = []
    for index in range(sides):
        angle = 2.0 * pi * index / sides
        nx, ny = cos(angle), sin(angle)
        planes.append(
            HalfPlane(nx, ny, radius_m + nx * center.x + ny * center.y)
        )
    return tuple(planes)


def circumscribed_circle_polygon(
    center: Point,
    radius_m: float,
    sides: int = 180,
) -> tuple[Point, ...]:
    """Return CCW vertices of a regular polygon circumscribed about a circle."""

    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    if sides < 12:
        raise ValueError("At least 12 sides are required.")
    vertex_radius = radius_m / cos(pi / sides)
    return tuple(
        Point(
            center.x + vertex_radius * cos(2.0 * pi * (index + 0.5) / sides),
            center.y + vertex_radius * sin(2.0 * pi * (index + 0.5) / sides),
        )
        for index in range(sides)
    )


def first_source_region(
    first_observation: Observation,
    error_deg: float = 1.0,
    target_radius_m: float = 1800.0,
    reception_upper_m: float = 1500.0,
    near_threshold_m: float = 5.0,
    circle_sides: int = 180,
) -> tuple[Point, ...]:
    """Construct a conservative convex polygon containing every possible source.

    Both circular constraints use circumscribed polygons, so discretization
    cannot exclude a physically feasible source.  Because a direction rather
    than ``near`` was measured, the source is beyond 5 m.  The convex projection
    constraint used for that fact is also conservative within the narrow wedge.
    """

    target = circumscribed_circle_polygon(Point(0.0, 0.0), target_radius_m, circle_sides)
    constraints = list(
        circumscribed_circle_halfplanes(
            first_observation.station, reception_upper_m, circle_sides
        )
    )
    constraints.extend(observation_halfplanes(first_observation, error_deg))

    direction = _unit(first_observation.bearing_deg)
    minimum_projection = near_threshold_m * cos(error_deg * pi / 180.0)
    constraints.append(
        HalfPlane(
            -direction.x,
            -direction.y,
            -(
                direction.x * first_observation.station.x
                + direction.y * first_observation.station.y
                + minimum_projection
            ),
        )
    )
    region = clip_polygon(target, constraints, tol=1e-7)
    if not region:
        raise ValueError("The first observation is inconsistent with the physical priors.")
    return region


def sample_convex_polygon(
    vertices: Sequence[Point],
    edge_subdivisions: int = 12,
    radial_levels: int = 5,
) -> tuple[Point, ...]:
    """Sample the boundary and interior of a convex polygon deterministically."""

    if not vertices:
        raise ValueError("vertices must not be empty.")
    if edge_subdivisions < 1 or radial_levels < 2:
        raise ValueError("Invalid sampling resolution.")

    center = Point(
        sum(point.x for point in vertices) / len(vertices),
        sum(point.y for point in vertices) / len(vertices),
    )
    boundary = []
    polygon = tuple(vertices)
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        for index in range(edge_subdivisions):
            fraction = index / edge_subdivisions
            boundary.append(
                Point(
                    start.x + fraction * (end.x - start.x),
                    start.y + fraction * (end.y - start.y),
                )
            )

    samples = [center]
    for boundary_point in boundary:
        for level in range(1, radial_levels):
            fraction = level / (radial_levels - 1)
            samples.append(
                Point(
                    center.x + fraction * (boundary_point.x - center.x),
                    center.y + fraction * (boundary_point.y - center.y),
                )
            )
    return tuple(samples)


def maximum_distance_to_region_vertices(
    candidate: Point,
    vertices: Sequence[Point],
) -> float:
    """Maximum candidate-to-source distance over a convex polygon."""

    return max(candidate.distance_to(vertex) for vertex in vertices)


def minimum_intersection_angle(
    candidate: Point,
    first_station: Point,
    source_samples: Sequence[Point],
    near_threshold_m: float = 5.0,
) -> float:
    """Return the worst acute bearing-line intersection angle in degrees."""

    source = np.asarray([(point.x, point.y) for point in source_samples], dtype=float)
    first = np.asarray([first_station.x, first_station.y], dtype=float)
    second = np.asarray([candidate.x, candidate.y], dtype=float)
    first_vectors = source - first
    second_vectors = source - second
    first_norms = np.linalg.norm(first_vectors, axis=1)
    second_norms = np.linalg.norm(second_vectors, axis=1)

    cross_values = np.abs(
        first_vectors[:, 0] * second_vectors[:, 1]
        - first_vectors[:, 1] * second_vectors[:, 0]
    )
    denominators = first_norms * second_norms
    sine_values = np.divide(
        cross_values,
        denominators,
        out=np.ones_like(cross_values),
        where=denominators > 1e-12,
    )
    sine_values = np.clip(sine_values, 0.0, 1.0)
    # A near result permits immediate optical localization, so it is not a bad
    # triangulation outcome and receives the maximal score.
    sine_values[second_norms <= near_threshold_m] = 1.0
    return degrees(asin(float(np.min(sine_values))))


def evaluate_candidate(
    candidate: Point,
    first_station: Point,
    region_vertices: Sequence[Point],
    source_samples: Sequence[Point],
    guaranteed_reception_m: float = 1000.0,
) -> CandidateEvaluation:
    max_distance = maximum_distance_to_region_vertices(candidate, region_vertices)
    guaranteed = max_distance <= guaranteed_reception_m + 1e-7
    score = (
        minimum_intersection_angle(candidate, first_station, source_samples)
        if guaranteed
        else 0.0
    )
    return CandidateEvaluation(
        point=candidate,
        guaranteed_visible=guaranteed,
        maximum_source_distance_m=max_distance,
        minimum_intersection_angle_deg=score,
        travel_distance_m=candidate.distance_to(first_station),
    )


def search_second_station_candidates(
    first_observation: Observation,
    region_vertices: Sequence[Point],
    grid_size: int = 161,
    guaranteed_reception_m: float = 1000.0,
    edge_subdivisions: int = 12,
    radial_levels: int = 5,
) -> CandidateGrid:
    """Search the guaranteed-visible lens and retain one point on each side."""

    if grid_size < 21:
        raise ValueError("grid_size must be at least 21.")
    samples = sample_convex_polygon(region_vertices, edge_subdivisions, radial_levels)
    x_min = max(vertex.x - guaranteed_reception_m for vertex in region_vertices)
    x_max = min(vertex.x + guaranteed_reception_m for vertex in region_vertices)
    y_min = max(vertex.y - guaranteed_reception_m for vertex in region_vertices)
    y_max = min(vertex.y + guaranteed_reception_m for vertex in region_vertices)
    if x_min > x_max or y_min > y_max:
        raise ValueError("No point can guarantee reception over the first feasible region.")

    x_values = np.linspace(x_min, x_max, grid_size)
    y_values = np.linspace(y_min, y_max, grid_size)
    scores = np.full((grid_size, grid_size), np.nan, dtype=float)
    feasible = np.zeros((grid_size, grid_size), dtype=bool)

    direction = _unit(first_observation.bearing_deg)
    normal = Point(-direction.y, direction.x)
    positive: list[CandidateEvaluation] = []
    negative: list[CandidateEvaluation] = []

    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            candidate = Point(float(x_value), float(y_value))
            max_distance = maximum_distance_to_region_vertices(candidate, region_vertices)
            if max_distance > guaranteed_reception_m + 1e-7:
                continue
            evaluation = CandidateEvaluation(
                point=candidate,
                guaranteed_visible=True,
                maximum_source_distance_m=max_distance,
                minimum_intersection_angle_deg=minimum_intersection_angle(
                    candidate, first_observation.station, samples
                ),
                travel_distance_m=candidate.distance_to(first_observation.station),
            )
            feasible[row, column] = True
            scores[row, column] = evaluation.minimum_intersection_angle_deg
            lateral = (
                (candidate.x - first_observation.station.x) * normal.x
                + (candidate.y - first_observation.station.y) * normal.y
            )
            if lateral >= 0.0:
                positive.append(evaluation)
            if lateral <= 0.0:
                negative.append(evaluation)

    if not positive or not negative:
        raise RuntimeError("The grid did not recover candidates on both bearing sides.")

    ranking = lambda item: (
        item.minimum_intersection_angle_deg,
        -item.travel_distance_m,
        -item.maximum_source_distance_m,
    )
    return CandidateGrid(
        x_values=x_values,
        y_values=y_values,
        angle_scores_deg=scores,
        feasible_mask=feasible,
        positive_candidate=max(positive, key=ranking),
        negative_candidate=max(negative, key=ranking),
        feasible_count=int(np.count_nonzero(feasible)),
    )


def refine_candidate(
    seed: CandidateEvaluation,
    first_observation: Observation,
    region_vertices: Sequence[Point],
    side: int,
    x_halfwidth_m: float,
    y_halfwidth_m: float,
    grid_size: int = 101,
    guaranteed_reception_m: float = 1000.0,
    edge_subdivisions: int = 30,
    radial_levels: int = 9,
) -> CandidateEvaluation:
    """Refine a coarse candidate by a deterministic local grid search.

    ``side`` is +1 for the left/positive side of the measured bearing and -1
    for the right/negative side.  Every returned point is rechecked against
    all source-region vertices, so refinement cannot compromise the robust
    reception guarantee.
    """

    if side not in (-1, 1):
        raise ValueError("side must be +1 or -1.")
    if x_halfwidth_m <= 0.0 or y_halfwidth_m <= 0.0:
        raise ValueError("Refinement half-widths must be positive.")
    if grid_size < 11:
        raise ValueError("grid_size must be at least 11.")

    samples = sample_convex_polygon(
        region_vertices,
        edge_subdivisions=edge_subdivisions,
        radial_levels=radial_levels,
    )
    x_values = np.linspace(
        seed.point.x - x_halfwidth_m, seed.point.x + x_halfwidth_m, grid_size
    )
    y_values = np.linspace(
        seed.point.y - y_halfwidth_m, seed.point.y + y_halfwidth_m, grid_size
    )
    direction = _unit(first_observation.bearing_deg)
    normal = Point(-direction.y, direction.x)
    best: CandidateEvaluation | None = None
    ranking = lambda item: (
        item.minimum_intersection_angle_deg,
        -item.travel_distance_m,
        -item.maximum_source_distance_m,
    )

    for y_value in y_values:
        for x_value in x_values:
            candidate = Point(float(x_value), float(y_value))
            lateral = (
                (candidate.x - first_observation.station.x) * normal.x
                + (candidate.y - first_observation.station.y) * normal.y
            )
            if side * lateral < -1e-9:
                continue
            evaluation = evaluate_candidate(
                candidate,
                first_observation.station,
                region_vertices,
                samples,
                guaranteed_reception_m,
            )
            if not evaluation.guaranteed_visible:
                continue
            if best is None or ranking(evaluation) > ranking(best):
                best = evaluation

    if best is None:
        raise RuntimeError("Local refinement found no feasible candidate.")
    return best


def centerline_baseline(
    first_observation: Observation,
    region_vertices: Sequence[Point],
    source_samples: Sequence[Point],
    guaranteed_reception_m: float = 1000.0,
) -> CandidateEvaluation:
    """A simple baseline that moves to the midpoint along the measured ray."""

    direction = _unit(first_observation.bearing_deg)
    projections = [
        (vertex.x - first_observation.station.x) * direction.x
        + (vertex.y - first_observation.station.y) * direction.y
        for vertex in region_vertices
    ]
    midpoint = (min(projections) + max(projections)) / 2.0
    candidate = Point(
        first_observation.station.x + midpoint * direction.x,
        first_observation.station.y + midpoint * direction.y,
    )
    return evaluate_candidate(
        candidate,
        first_observation.station,
        region_vertices,
        source_samples,
        guaranteed_reception_m,
    )


def posterior_polygon(
    first_region: Sequence[Point],
    second_station: Point,
    source: Point,
    measurement_error_deg: float,
    bearing_error_bound_deg: float = 1.0,
) -> tuple[Point, ...]:
    """Update K1 with a possible second measured bearing for a known scenario."""

    true_bearing = degrees(
        atan2(source.y - second_station.y, source.x - second_station.x)
    ) % 360.0
    measured_bearing = true_bearing + measurement_error_deg
    planes = observation_halfplanes(
        Observation(second_station, measured_bearing), bearing_error_bound_deg
    )
    return clip_polygon(first_region, planes, tol=1e-7)


def worst_posterior_diameter(
    first_region: Sequence[Point],
    second_station: Point,
    source_samples: Sequence[Point],
    measurement_errors_deg: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    bearing_error_bound_deg: float = 1.0,
) -> PosteriorEvaluation:
    """Enumerate source/error scenarios and return the largest posterior diameter."""

    worst: PosteriorEvaluation | None = None
    for source in source_samples:
        if source.distance_to(second_station) <= 5.0:
            diameter = 0.0
            polygon = (source,)
            errors = (0.0,)
        else:
            errors = measurement_errors_deg
        for error in errors:
            if source.distance_to(second_station) > 5.0:
                polygon = posterior_polygon(
                    first_region,
                    second_station,
                    source,
                    error,
                    bearing_error_bound_deg,
                )
                if not polygon:
                    raise RuntimeError("A consistent source was lost during polygon clipping.")
                diameter, _ = polygon_diameter(polygon)
            evaluation = PosteriorEvaluation(diameter, source, error, tuple(polygon))
            if worst is None or evaluation.worst_diameter_m > worst.worst_diameter_m:
                worst = evaluation
    if worst is None:
        raise ValueError("source_samples must not be empty.")
    return worst


def representative_posterior(
    first_observation: Observation,
    first_region: Sequence[Point],
    second_station: Point,
    source_distance_m: float = 900.0,
    measurement_errors_deg: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
) -> tuple[Point, float, tuple[Point, ...], float, Circle]:
    """Return the worst error realization for a center-ray representative source."""

    direction = _unit(first_observation.bearing_deg)
    source = Point(
        first_observation.station.x + source_distance_m * direction.x,
        first_observation.station.y + source_distance_m * direction.y,
    )
    best_error = 0.0
    best_polygon: tuple[Point, ...] = ()
    best_diameter = -1.0
    for error in measurement_errors_deg:
        polygon = posterior_polygon(first_region, second_station, source, error)
        diameter, _ = polygon_diameter(polygon)
        if diameter > best_diameter:
            best_error = error
            best_polygon = polygon
            best_diameter = diameter
    circle = minimum_enclosing_circle(best_polygon)
    return source, best_error, best_polygon, best_diameter, circle

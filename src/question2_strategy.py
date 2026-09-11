"""Robust second-station selection for CUMCM 2026 Problem B, Question 2.

The first successful bearing updates the lower bound of the unknown reception
radius from 1000 m to ``max(1000 m, distance(source, first_station))``.  The
module therefore supports both the original fixed-1000 m baseline and the
conditional reception model.  Final candidates are ranked by the sampled
worst posterior-region diameter; intersection angle is retained as a useful
diagnostic rather than used as the final objective.
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
    reception_margin_m: float = float("nan")


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
    clearance_guaranteed: bool
    clearance_radius_lower_bound_m: float
    worst_exact_mec_radius_m: float
    exact_mec_evaluation_count: int
    diameter_pruned_evaluation_count: int


@dataclass(frozen=True)
class ConditionalReceptionEvaluation:
    """Continuous reception audit over a convex polygonal source region."""

    guaranteed_visible: bool
    minimum_margin_m: float
    worst_source: Point
    maximum_source_distance_m: float


@dataclass(frozen=True)
class PosteriorGrid:
    """Posterior-loss search result on a deterministic Cartesian grid."""

    along_values_m: np.ndarray
    lateral_values_m: np.ndarray
    losses_m: np.ndarray
    clearance_feasible_mask: np.ndarray
    feasible_mask: np.ndarray
    pareto_mask: np.ndarray
    optimum: CandidateEvaluation
    recommended: CandidateEvaluation
    optimum_posterior: PosteriorEvaluation
    recommended_posterior: PosteriorEvaluation
    pareto_knee_score: float
    pareto_front_count: int
    evaluated_count: int
    selection_mode: str


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


def _inside_convex_polygon(
    point: Point,
    vertices: Sequence[Point],
    tol: float = 1e-7,
) -> bool:
    """Return whether ``point`` lies in or on a consistently ordered polygon."""

    polygon = tuple(vertices)
    if len(polygon) < 3:
        return any(point.distance_to(vertex) <= tol for vertex in polygon)
    signs = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (end.x - start.x) * (point.y - start.y) - (
            end.y - start.y
        ) * (point.x - start.x)
        if abs(cross) > tol:
            signs.append(cross > 0.0)
    return not signs or all(sign == signs[0] for sign in signs)


def _polygon_circle_boundary_points(
    vertices: Sequence[Point],
    center: Point,
    radius_m: float,
    tol: float = 1e-9,
) -> tuple[Point, ...]:
    """Return polygon vertices and all edge intersections with a circle."""

    polygon = tuple(vertices)
    points = list(polygon)
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = end.x - start.x, end.y - start.y
        ox, oy = start.x - center.x, start.y - center.y
        qa = dx * dx + dy * dy
        qb = 2.0 * (ox * dx + oy * dy)
        qc = ox * ox + oy * oy - radius_m * radius_m
        discriminant = qb * qb - 4.0 * qa * qc
        if qa <= tol or discriminant < -tol:
            continue
        root = max(0.0, discriminant) ** 0.5
        for fraction in ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)):
            if -tol <= fraction <= 1.0 + tol:
                clipped = min(1.0, max(0.0, fraction))
                point = Point(start.x + clipped * dx, start.y + clipped * dy)
                if not any(point.distance_to(existing) <= 1e-7 for existing in points):
                    points.append(point)
    return tuple(points)


def conditional_reception_evaluation(
    candidate: Point,
    first_station: Point,
    region_vertices: Sequence[Point],
    prior_lower_radius_m: float = 1000.0,
    safety_margin_m: float = 0.0,
) -> ConditionalReceptionEvaluation:
    """Audit the conditional reception guarantee without source sampling.

    Inside the prior-radius disk, the maximum distance to ``candidate`` occurs
    at a polygon vertex, a polygon/circle intersection, or the circle point
    opposite the candidate.  Outside that disk, the squared-distance condition
    is affine in the source coordinate, so the same finite set contains its
    worst point.  Consequently this check is continuous over the polygon, not
    merely a dense point-sample audit.
    """

    if prior_lower_radius_m <= 0.0:
        raise ValueError("prior_lower_radius_m must be positive.")
    if not region_vertices:
        raise ValueError("region_vertices must not be empty.")

    boundary = list(
        _polygon_circle_boundary_points(
            region_vertices, first_station, prior_lower_radius_m
        )
    )
    qx = candidate.x - first_station.x
    qy = candidate.y - first_station.y
    qnorm = (qx * qx + qy * qy) ** 0.5
    if qnorm > 1e-12:
        antipode = Point(
            first_station.x - prior_lower_radius_m * qx / qnorm,
            first_station.y - prior_lower_radius_m * qy / qnorm,
        )
        if _inside_convex_polygon(antipode, region_vertices):
            boundary.append(antipode)

    near_points = [
        point
        for point in boundary
        if point.distance_to(first_station) <= prior_lower_radius_m + 1e-7
    ]
    far_points = [
        point
        for point in boundary
        if point.distance_to(first_station) >= prior_lower_radius_m - 1e-7
    ]
    near_audits = [
        (prior_lower_radius_m - candidate.distance_to(source), source)
        for source in near_points
    ]
    far_squared_audits = [
        (
            source.distance_to(first_station) ** 2
            - candidate.distance_to(source) ** 2,
            source,
        )
        for source in far_points
    ]
    if not near_audits and not far_squared_audits:
        raise RuntimeError("Reception audit recovered no boundary points.")
    margin_audits = list(near_audits)
    if far_squared_audits:
        squared_slack, far_worst = min(far_squared_audits, key=lambda item: item[0])
        maximum_first_distance = max(
            source.distance_to(first_station) for source in far_points
        )
        # Since d2 <= d1 whenever the squared slack is non-negative,
        # d1+d2 <= 2*max(d1).  This converts affine squared slack into a
        # conservative continuous lower bound on the distance margin.
        far_margin_lower_bound = squared_slack / (2.0 * maximum_first_distance)
        margin_audits.append((far_margin_lower_bound, far_worst))
    minimum_margin, worst_source = min(margin_audits, key=lambda item: item[0])
    return ConditionalReceptionEvaluation(
        guaranteed_visible=minimum_margin >= safety_margin_m - 1e-7,
        minimum_margin_m=minimum_margin,
        worst_source=worst_source,
        maximum_source_distance_m=maximum_distance_to_region_vertices(
            candidate, region_vertices
        ),
    )


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
    clearance_radius_m: float = 20.0,
) -> PosteriorEvaluation:
    """Return worst posterior diameter and an exact clearance classification.

    Every non-``near`` scenario is classified against the clearance radius.  A
    posterior with ``diameter / 2 > clearance_radius_m`` is rejected by a
    rigorous lower bound.  Only the remaining scenarios need an exact minimum
    enclosing circle, so the Boolean clearance result is exact on the supplied
    source/error grid without paying for unnecessary circle enumeration.
    """

    worst: PosteriorEvaluation | None = None
    clearance_guaranteed = True
    clearance_radius_lower_bound = 0.0
    worst_exact_mec_radius = 0.0
    exact_mec_count = 0
    diameter_pruned_count = 0
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
            radius_lower_bound = diameter / 2.0
            if radius_lower_bound > clearance_radius_m + 1e-9:
                clearance_guaranteed = False
                diameter_pruned_count += 1
                clearance_radius_lower_bound = max(
                    clearance_radius_lower_bound, radius_lower_bound
                )
            else:
                circle = minimum_enclosing_circle(polygon)
                exact_mec_count += 1
                worst_exact_mec_radius = max(worst_exact_mec_radius, circle.radius)
                clearance_radius_lower_bound = max(
                    clearance_radius_lower_bound, circle.radius
                )
                if circle.radius > clearance_radius_m + 1e-9:
                    clearance_guaranteed = False
            evaluation = PosteriorEvaluation(
                diameter,
                source,
                error,
                tuple(polygon),
                clearance_guaranteed,
                clearance_radius_lower_bound,
                worst_exact_mec_radius,
                exact_mec_count,
                diameter_pruned_count,
            )
            if worst is None or evaluation.worst_diameter_m > worst.worst_diameter_m:
                worst = evaluation
    if worst is None:
        raise ValueError("source_samples must not be empty.")
    return PosteriorEvaluation(
        worst.worst_diameter_m,
        worst.worst_source,
        worst.worst_measurement_error_deg,
        worst.worst_polygon,
        clearance_guaranteed,
        clearance_radius_lower_bound,
        worst_exact_mec_radius,
        exact_mec_count,
        diameter_pruned_count,
    )


def _complete_station_search_bounds(
    first_observation: Observation,
    first_region: Sequence[Point],
    prior_lower_radius_m: float,
    direction: Point,
    normal: Point,
    station_domain_center: Point | None,
    station_domain_radius_m: float | None,
) -> tuple[float, float, float, float]:
    """Return local-coordinate bounds containing every feasible station.

    A conditionally feasible station must lie in the reception disk associated
    with every possible source.  Any one source point therefore supplies a
    valid enclosing disk.  We choose the region vertex whose compatible
    reception disk is smallest, then intersect its local bounding box with the
    optional station-domain disk bounding box.
    """

    station = first_observation.station

    def local_coordinates(point: Point) -> tuple[float, float]:
        dx, dy = point.x - station.x, point.y - station.y
        return dx * direction.x + dy * direction.y, dx * normal.x + dy * normal.y

    anchor = min(
        first_region,
        key=lambda point: max(prior_lower_radius_m, point.distance_to(station)),
    )
    anchor_radius = max(prior_lower_radius_m, anchor.distance_to(station))
    anchor_along, anchor_lateral = local_coordinates(anchor)
    along_min = anchor_along - anchor_radius
    along_max = anchor_along + anchor_radius
    lateral_min = anchor_lateral - anchor_radius
    lateral_max = anchor_lateral + anchor_radius

    if station_domain_center is not None and station_domain_radius_m is not None:
        center_along, center_lateral = local_coordinates(station_domain_center)
        along_min = max(along_min, center_along - station_domain_radius_m)
        along_max = min(along_max, center_along + station_domain_radius_m)
        lateral_min = max(lateral_min, center_lateral - station_domain_radius_m)
        lateral_max = min(lateral_max, center_lateral + station_domain_radius_m)

    if along_min > along_max or lateral_min > lateral_max:
        raise RuntimeError("Station domain and conditional reception region do not overlap.")
    return along_min, along_max, lateral_min, lateral_max


def search_posterior_optimal_candidates(
    first_observation: Observation,
    first_region: Sequence[Point],
    grid_size: int = 31,
    source_edge_subdivisions: int = 6,
    source_radial_levels: int = 4,
    measurement_errors_deg: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    prior_lower_radius_m: float = 1000.0,
    reception_safety_margin_m: float = 0.5,
    clearance_radius_m: float = 20.0,
    station_domain_center: Point | None = Point(0.0, 0.0),
    station_domain_radius_m: float | None = 1800.0,
) -> PosteriorGrid:
    """Search the conditional reception domain using posterior diameter.

    The local Cartesian bounds contain the complete conditional-reception
    region and are intersected with the optional circular activity domain.
    The returned optimum is a deterministic-grid result, not a claim of
    continuous global optimality. If any grid point guarantees clearance it
    is preferred; otherwise the recommendation is the maximum-distance-to-chord
    knee of the travel-distance/log-posterior-loss Pareto frontier.
    """

    if grid_size < 11:
        raise ValueError("grid_size must be at least 11.")
    if clearance_radius_m <= 0.0:
        raise ValueError("clearance_radius_m must be positive.")
    if (station_domain_center is None) != (station_domain_radius_m is None):
        raise ValueError(
            "station_domain_center and station_domain_radius_m must both be set or both be None."
        )
    if station_domain_radius_m is not None and station_domain_radius_m <= 0.0:
        raise ValueError("station_domain_radius_m must be positive.")
    samples = sample_convex_polygon(
        first_region,
        edge_subdivisions=source_edge_subdivisions,
        radial_levels=source_radial_levels,
    )
    direction = _unit(first_observation.bearing_deg)
    normal = Point(-direction.y, direction.x)
    along_min, along_max, lateral_min, lateral_max = _complete_station_search_bounds(
        first_observation,
        first_region,
        prior_lower_radius_m,
        direction,
        normal,
        station_domain_center,
        station_domain_radius_m,
    )
    along_values = np.linspace(along_min, along_max, grid_size)
    lateral_values = np.linspace(lateral_min, lateral_max, grid_size)
    losses = np.full((grid_size, grid_size), np.nan, dtype=float)
    feasible = np.zeros((grid_size, grid_size), dtype=bool)
    clearance_feasible = np.zeros((grid_size, grid_size), dtype=bool)
    evaluations: dict[tuple[int, int], tuple[CandidateEvaluation, PosteriorEvaluation]] = {}

    for row, lateral in enumerate(lateral_values):
        for column, along in enumerate(along_values):
            point = Point(
                first_observation.station.x
                + float(along) * direction.x
                + float(lateral) * normal.x,
                first_observation.station.y
                + float(along) * direction.y
                + float(lateral) * normal.y,
            )
            if (
                station_domain_center is not None
                and station_domain_radius_m is not None
                and point.distance_to(station_domain_center)
                > station_domain_radius_m + 1e-9
            ):
                continue
            reception = conditional_reception_evaluation(
                point,
                first_observation.station,
                first_region,
                prior_lower_radius_m=prior_lower_radius_m,
                safety_margin_m=reception_safety_margin_m,
            )
            if not reception.guaranteed_visible:
                continue
            posterior = worst_posterior_diameter(
                first_region,
                point,
                samples,
                measurement_errors_deg=measurement_errors_deg,
                clearance_radius_m=clearance_radius_m,
            )
            candidate = CandidateEvaluation(
                point=point,
                guaranteed_visible=True,
                maximum_source_distance_m=reception.maximum_source_distance_m,
                minimum_intersection_angle_deg=minimum_intersection_angle(
                    point, first_observation.station, samples
                ),
                travel_distance_m=point.distance_to(first_observation.station),
                reception_margin_m=reception.minimum_margin_m,
            )
            feasible[row, column] = True
            losses[row, column] = posterior.worst_diameter_m
            clearance_feasible[row, column] = posterior.clearance_guaranteed
            evaluations[(row, column)] = (candidate, posterior)

    if not evaluations:
        raise RuntimeError("The posterior grid found no conditionally feasible point.")
    clearance_indices = [
        index for index in evaluations if evaluations[index][1].clearance_guaranteed
    ]
    if clearance_indices:
        selection_mode = "guaranteed_clearance"
        recommended_index = min(
            clearance_indices,
            key=lambda index: (
                evaluations[index][0].travel_distance_m,
                -evaluations[index][0].reception_margin_m,
                losses[index],
                evaluations[index][0].point.x,
                evaluations[index][0].point.y,
            ),
        )
        optimum_index = recommended_index
        pareto = clearance_feasible.copy()
        knee_score = 0.0
    else:
        selection_mode = "pareto_knee"
        optimum_index = min(evaluations, key=lambda index: losses[index])
        pareto = np.zeros_like(feasible)
        ordered = sorted(
            evaluations,
            key=lambda index: (
                evaluations[index][0].travel_distance_m,
                losses[index],
                evaluations[index][0].point.x,
                evaluations[index][0].point.y,
            ),
        )
        best_loss = float("inf")
        pareto_indices: list[tuple[int, int]] = []
        for index in ordered:
            loss = float(losses[index])
            if loss < best_loss - 1e-9:
                pareto_indices.append(index)
                pareto[index] = True
                best_loss = loss

        distances = np.asarray(
            [evaluations[index][0].travel_distance_m for index in pareto_indices],
            dtype=float,
        )
        pareto_losses = np.asarray(
            [losses[index] for index in pareto_indices], dtype=float
        )
        distance_span = float(np.ptp(distances))
        log_losses = np.log(pareto_losses)
        log_loss_span = float(np.ptp(log_losses))
        if distance_span <= 1e-12 or log_loss_span <= 1e-12:
            recommended_index = pareto_indices[0]
            knee_score = 0.0
        else:
            normalized_distance = (distances - distances.min()) / distance_span
            normalized_log_loss = (
                (log_losses - log_losses.min()) / log_loss_span
            )
            knee_scores = 1.0 - normalized_distance - normalized_log_loss
            knee_position = int(np.argmax(knee_scores))
            recommended_index = pareto_indices[knee_position]
            knee_score = float(knee_scores[knee_position])
    optimum, optimum_posterior = evaluations[optimum_index]
    recommended, recommended_posterior = evaluations[recommended_index]
    return PosteriorGrid(
        along_values_m=along_values,
        lateral_values_m=lateral_values,
        losses_m=losses,
        clearance_feasible_mask=clearance_feasible,
        feasible_mask=feasible,
        pareto_mask=pareto,
        optimum=optimum,
        recommended=recommended,
        optimum_posterior=optimum_posterior,
        recommended_posterior=recommended_posterior,
        pareto_knee_score=knee_score,
        pareto_front_count=int(np.count_nonzero(pareto)),
        evaluated_count=len(evaluations),
        selection_mode=selection_mode,
    )


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

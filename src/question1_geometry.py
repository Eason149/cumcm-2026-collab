"""Geometry kernel for Question 1 of CUMCM 2026 Problem B.

The module treats every direction-finding observation as a forward angular
wedge with a deterministic error bound.  Each wedge is converted into two
closed half-planes.  Their intersection is classified as empty, unbounded,
or a bounded convex polygon.  For a bounded polygon, the module computes its
diameter, checks circles built on diametral pairs, and computes the minimum
enclosing circle.

Angles follow the problem statement: 0 degrees points east and positive
angles rotate counter-clockwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import cos, hypot, isfinite, pi, sin
from typing import Iterable, Sequence

from scipy.optimize import linprog


DEFAULT_TOL = 1e-8


@dataclass(frozen=True)
class Point:
    """A point in metres in the east-north coordinate system."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Observation:
    """One detector position and its measured bearing in degrees."""

    station: Point
    bearing_deg: float


@dataclass(frozen=True)
class HalfPlane:
    """A closed half-plane represented by a*x + b*y <= c."""

    a: float
    b: float
    c: float

    def normalized(self) -> "HalfPlane":
        scale = hypot(self.a, self.b)
        if scale <= 0.0:
            raise ValueError("A half-plane must have a non-zero normal.")
        return HalfPlane(self.a / scale, self.b / scale, self.c / scale)

    def contains(self, point: Point, tol: float = DEFAULT_TOL) -> bool:
        return self.a * point.x + self.b * point.y <= self.c + tol


@dataclass(frozen=True)
class HalfPlaneIntersection:
    """Classification and vertices of a half-plane intersection."""

    status: str
    vertices: tuple[Point, ...]


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float
    support: tuple[Point, ...]


@dataclass(frozen=True)
class DiametralCircleCheck:
    endpoints: tuple[Point, Point]
    circle: Circle
    farthest_vertex_distance: float
    covers: bool


@dataclass(frozen=True)
class Question1Result:
    intersection: HalfPlaneIntersection
    diameter: float | None
    diametral_pairs: tuple[tuple[Point, Point], ...]
    diametral_circle_checks: tuple[DiametralCircleCheck, ...]
    minimum_enclosing_circle: Circle | None


def _direction(angle_deg: float) -> Point:
    angle_rad = (angle_deg % 360.0) * pi / 180.0
    return Point(cos(angle_rad), sin(angle_rad))


def observation_halfplanes(
    observation: Observation,
    error_deg: float = 1.0,
) -> tuple[HalfPlane, HalfPlane]:
    """Convert one bounded-bearing observation into two half-planes.

    For lower and upper boundary vectors u_minus and u_plus, a candidate
    displacement r must satisfy cross(u_minus, r) >= 0 and
    cross(u_plus, r) <= 0.  The two inequalities retain the forward wedge and
    reject the backward extensions of the bearing lines.
    """

    # With exactly zero angular width, two cross-product half-planes describe
    # the complete bearing line rather than its forward ray.  The competition
    # uses a strictly positive 1-degree bound, so reject the unsupported
    # degenerate case explicitly instead of silently admitting backward points.
    if not 0.0 < error_deg < 90.0:
        raise ValueError("error_deg must lie in (0, 90).")
    if not all(
        isfinite(value)
        for value in (
            observation.station.x,
            observation.station.y,
            observation.bearing_deg,
            error_deg,
        )
    ):
        raise ValueError("Observation values must be finite.")

    lower = _direction(observation.bearing_deg - error_deg)
    upper = _direction(observation.bearing_deg + error_deg)
    sx, sy = observation.station.x, observation.station.y

    # cross(lower, P-S) >= 0
    first = HalfPlane(
        a=lower.y,
        b=-lower.x,
        c=lower.y * sx - lower.x * sy,
    ).normalized()
    # cross(upper, P-S) <= 0
    second = HalfPlane(
        a=-upper.y,
        b=upper.x,
        c=-upper.y * sx + upper.x * sy,
    ).normalized()
    return first, second


def _line_intersection(
    first: HalfPlane,
    second: HalfPlane,
    tol: float,
) -> Point | None:
    det = first.a * second.b - second.a * first.b
    if abs(det) <= tol:
        return None
    x = (first.c * second.b - second.c * first.b) / det
    y = (first.a * second.c - second.a * first.c) / det
    return Point(x, y)


def _deduplicate_points(points: Iterable[Point], tol: float) -> list[Point]:
    unique: list[Point] = []
    for point in points:
        if not any(point.distance_to(existing) <= tol for existing in unique):
            unique.append(point)
    return unique


def convex_hull(points: Sequence[Point], tol: float = DEFAULT_TOL) -> tuple[Point, ...]:
    """Return unique hull vertices counter-clockwise using the monotone chain."""

    ordered = sorted(_deduplicate_points(points, tol), key=lambda p: (p.x, p.y))
    if len(ordered) <= 1:
        return tuple(ordered)

    def cross(origin: Point, first: Point, second: Point) -> float:
        return (first.x - origin.x) * (second.y - origin.y) - (
            first.y - origin.y
        ) * (second.x - origin.x)

    lower: list[Point] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tol:
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tol:
            upper.pop()
        upper.append(point)

    return tuple(lower[:-1] + upper[:-1])


def clip_polygon_by_halfplane(
    vertices: Sequence[Point],
    halfplane: HalfPlane,
    tol: float = DEFAULT_TOL,
) -> tuple[Point, ...]:
    """Clip a convex polygon by one half-plane using Sutherland-Hodgman.

    The input vertices must be ordered around the polygon.  Single-point and
    line-segment degeneracies are supported so the function can also update a
    nearly collapsed localization region.
    """

    polygon = tuple(vertices)
    if not polygon:
        return ()
    plane = halfplane.normalized()

    def signed_value(point: Point) -> float:
        return plane.a * point.x + plane.b * point.y - plane.c

    output: list[Point] = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        start_value = signed_value(start)
        end_value = signed_value(end)
        start_inside = start_value <= tol
        end_inside = end_value <= tol

        if start_inside:
            output.append(start)
        if start_inside != end_inside:
            denominator = start_value - end_value
            if abs(denominator) > tol:
                fraction = start_value / denominator
                output.append(
                    Point(
                        start.x + fraction * (end.x - start.x),
                        start.y + fraction * (end.y - start.y),
                    )
                )

    unique = _deduplicate_points(output, tol)
    if len(unique) <= 2:
        return tuple(unique)
    return convex_hull(unique, tol)


def clip_polygon(
    vertices: Sequence[Point],
    halfplanes: Sequence[HalfPlane],
    tol: float = DEFAULT_TOL,
) -> tuple[Point, ...]:
    """Clip a convex polygon successively by several half-planes."""

    polygon = tuple(vertices)
    for halfplane in halfplanes:
        polygon = clip_polygon_by_halfplane(polygon, halfplane, tol)
        if not polygon:
            break
    return polygon


def intersect_halfplanes(
    halfplanes: Sequence[HalfPlane],
    tol: float = DEFAULT_TOL,
) -> HalfPlaneIntersection:
    """Intersect half-planes and classify the feasible set.

    Linear programming supplies explicit feasibility and boundedness checks.
    When the set is bounded, feasible pairwise boundary intersections are its
    extreme points; their convex hull is the requested polygon.  Degenerate
    single-point and line-segment intersections are retained.
    """

    if not halfplanes:
        return HalfPlaneIntersection("unbounded", ())

    planes = tuple(plane.normalized() for plane in halfplanes)
    matrix = [[plane.a, plane.b] for plane in planes]
    bounds = [plane.c for plane in planes]
    variable_bounds = [(None, None), (None, None)]

    feasibility = linprog(
        [0.0, 0.0],
        A_ub=matrix,
        b_ub=bounds,
        bounds=variable_bounds,
        method="highs",
    )
    if feasibility.status == 2:
        return HalfPlaneIntersection("empty", ())
    if not feasibility.success:
        raise RuntimeError(f"Half-plane feasibility LP failed: {feasibility.message}")

    objectives = ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0])
    for objective in objectives:
        result = linprog(
            objective,
            A_ub=matrix,
            b_ub=bounds,
            bounds=variable_bounds,
            method="highs",
        )
        if result.status == 3:
            return HalfPlaneIntersection("unbounded", ())
        if not result.success:
            raise RuntimeError(f"Half-plane boundedness LP failed: {result.message}")

    scale = max(1.0, max(abs(value) for value in bounds))
    containment_tol = tol * scale
    candidates: list[Point] = []
    for first, second in combinations(planes, 2):
        point = _line_intersection(first, second, tol)
        if point is not None and all(
            plane.contains(point, containment_tol) for plane in planes
        ):
            candidates.append(point)

    vertices = convex_hull(candidates, containment_tol)
    if not vertices:
        # A feasible bounded set in R^2 must have at least one extreme point.
        # Reaching this branch indicates an ill-conditioned input or tolerance.
        raise RuntimeError("Bounded feasible set found, but no vertices were recovered.")
    return HalfPlaneIntersection("bounded", vertices)


def localization_region(
    observations: Sequence[Observation],
    error_deg: float = 1.0,
    extra_halfplanes: Sequence[HalfPlane] = (),
    tol: float = DEFAULT_TOL,
) -> HalfPlaneIntersection:
    """Construct the intersection of all observation wedges and optional priors."""

    planes: list[HalfPlane] = list(extra_halfplanes)
    for observation in observations:
        planes.extend(observation_halfplanes(observation, error_deg))
    return intersect_halfplanes(planes, tol)


def polygon_diameter(
    vertices: Sequence[Point],
    tol: float = DEFAULT_TOL,
) -> tuple[float, tuple[tuple[Point, Point], ...]]:
    """Compute polygon diameter and every diametral vertex pair by enumeration."""

    points = tuple(vertices)
    if not points:
        raise ValueError("At least one vertex is required.")
    if len(points) == 1:
        return 0.0, ((points[0], points[0]),)

    distances = [(first.distance_to(second), (first, second)) for first, second in combinations(points, 2)]
    diameter = max(distance for distance, _ in distances)
    threshold = tol * max(1.0, diameter)
    pairs = tuple(pair for distance, pair in distances if abs(distance - diameter) <= threshold)
    return diameter, pairs


def _circle_from_pair(first: Point, second: Point) -> Circle:
    center = Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
    return Circle(center, first.distance_to(second) / 2.0, (first, second))


def _circle_from_triple(first: Point, second: Point, third: Point, tol: float) -> Circle | None:
    denominator = 2.0 * (
        first.x * (second.y - third.y)
        + second.x * (third.y - first.y)
        + third.x * (first.y - second.y)
    )
    if abs(denominator) <= tol:
        return None

    first_sq = first.x * first.x + first.y * first.y
    second_sq = second.x * second.x + second.y * second.y
    third_sq = third.x * third.x + third.y * third.y
    center = Point(
        (
            first_sq * (second.y - third.y)
            + second_sq * (third.y - first.y)
            + third_sq * (first.y - second.y)
        )
        / denominator,
        (
            first_sq * (third.x - second.x)
            + second_sq * (first.x - third.x)
            + third_sq * (second.x - first.x)
        )
        / denominator,
    )
    return Circle(center, center.distance_to(first), (first, second, third))


def _circle_covers(circle: Circle, points: Sequence[Point], tol: float) -> bool:
    threshold = circle.radius + tol * max(1.0, circle.radius)
    return all(circle.center.distance_to(point) <= threshold for point in points)


def minimum_enclosing_circle(
    vertices: Sequence[Point],
    tol: float = DEFAULT_TOL,
) -> Circle:
    """Compute the exact minimum enclosing circle from 1-, 2-, and 3-point supports.

    A planar minimum enclosing circle is supported by at most three points.
    Exhaustive enumeration is deterministic and intentionally preferred here
    because Question 1 produces small polygons.
    """

    points = tuple(_deduplicate_points(vertices, tol))
    if not points:
        raise ValueError("At least one vertex is required.")

    candidates: list[Circle] = [Circle(point, 0.0, (point,)) for point in points]
    candidates.extend(_circle_from_pair(first, second) for first, second in combinations(points, 2))
    for first, second, third in combinations(points, 3):
        circle = _circle_from_triple(first, second, third, tol)
        if circle is not None:
            candidates.append(circle)

    feasible = [circle for circle in candidates if _circle_covers(circle, points, tol)]
    if not feasible:
        raise RuntimeError("No enclosing circle could be constructed.")
    return min(
        feasible,
        key=lambda circle: (
            circle.radius,
            circle.center.x,
            circle.center.y,
            len(circle.support),
        ),
    )


def check_diametral_circles(
    vertices: Sequence[Point],
    diametral_pairs: Sequence[tuple[Point, Point]],
    tol: float = DEFAULT_TOL,
) -> tuple[DiametralCircleCheck, ...]:
    """Check whether circles built on each diametral pair cover the polygon."""

    checks: list[DiametralCircleCheck] = []
    for endpoints in diametral_pairs:
        circle = _circle_from_pair(*endpoints)
        farthest = max(circle.center.distance_to(vertex) for vertex in vertices)
        threshold = circle.radius + tol * max(1.0, circle.radius)
        checks.append(
            DiametralCircleCheck(
                endpoints=endpoints,
                circle=circle,
                farthest_vertex_distance=farthest,
                covers=farthest <= threshold,
            )
        )
    return tuple(checks)


def solve_question1(
    observations: Sequence[Observation],
    error_deg: float = 1.0,
    extra_halfplanes: Sequence[HalfPlane] = (),
    tol: float = DEFAULT_TOL,
) -> Question1Result:
    """Run the complete Question 1 workflow for one set of observations."""

    intersection = localization_region(observations, error_deg, extra_halfplanes, tol)
    if intersection.status != "bounded":
        return Question1Result(intersection, None, (), (), None)

    diameter, pairs = polygon_diameter(intersection.vertices, tol)
    checks = check_diametral_circles(intersection.vertices, pairs, tol)
    enclosing_circle = minimum_enclosing_circle(intersection.vertices, tol)
    return Question1Result(intersection, diameter, pairs, checks, enclosing_circle)


def rectangle_prior(x_min: float, x_max: float, y_min: float, y_max: float) -> tuple[HalfPlane, ...]:
    """Return four half-planes for an optional rectangular prior."""

    if not (x_min <= x_max and y_min <= y_max):
        raise ValueError("Invalid rectangle bounds.")
    return (
        HalfPlane(1.0, 0.0, x_max),
        HalfPlane(-1.0, 0.0, -x_min),
        HalfPlane(0.0, 1.0, y_max),
        HalfPlane(0.0, -1.0, -y_min),
    )

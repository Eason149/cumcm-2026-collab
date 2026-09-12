"""Guaranteed discovery and removal policy for mixed sources in Question 4.

Directional sources illuminate a closed semicircle.  A triangular survey
lattice guarantees that every possible source and emission orientation sees at
least one station within the minimum 1000 m reception radius.  After a hit, a
two-sided short-step pursuit retains a deterministic signal guarantee even when
the first station lies exactly on the emission boundary.
"""

from __future__ import annotations

from math import ceil, cos, pi, sin, sqrt

from question1_geometry import Observation, Point
from question3_strategy import (
    BEARING_ERROR_DEG,
    CLEAR_RADIUS_M,
    MAX_RECEPTION_RADIUS_M,
    MIN_RECEPTION_RADIUS_M,
    TARGET_RADIUS_M,
    ChannelState,
    RobotInterface,
    StrategyResult,
    _nominal_bearing_intersection,
    _plan_open_route,
    _region_radius_and_center,
    _update_state_with_bearing,
)


LATTICE_EDGE_M = 999.0
LATTICE_VERTICAL_OFFSET_FRACTION = 1.0 / 6.0
FAST_STATION_INDICES = (
    0, 1, 2, 4, 7, 8, 9, 10, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
)


def _bearing_point(origin: Point, bearing_deg: float, distance_m: float) -> Point:
    angle = bearing_deg * pi / 180.0
    return Point(origin.x + distance_m * cos(angle), origin.y + distance_m * sin(angle))


def triangular_lattice_stations(
    target_radius_m: float = TARGET_RADIUS_M,
    edge_m: float = LATTICE_EDGE_M,
) -> tuple[Point, ...]:
    """Return all lattice vertices needed by triangles intersecting the target.

    Every point in the plane belongs to a closed equilateral lattice triangle.
    Its three vertices are at most ``edge_m`` away.  Keeping every vertex within
    ``target_radius_m + edge_m`` therefore retains the full containing triangle
    for each possible source in the target disk.
    """

    if edge_m <= 0.0 or edge_m > MIN_RECEPTION_RADIUS_M:
        raise ValueError("edge_m must lie in (0, minimum reception radius].")
    height = sqrt(3.0) * edge_m / 2.0
    vertical_offset = height * LATTICE_VERTICAL_OFFSET_FRACTION
    outer_radius = target_radius_m + edge_m
    vertical_limit = ceil((outer_radius + abs(vertical_offset)) / height) + 1
    horizontal_limit = ceil(outer_radius / edge_m) + vertical_limit + 1
    selected: dict[tuple[float, float], Point] = {}

    def lattice_point(column: int, row: int) -> Point:
        return Point(
            edge_m * (column + 0.5 * row),
            row * height + vertical_offset,
        )

    def segment_distance_to_origin(first: Point, second: Point) -> float:
        dx, dy = second.x - first.x, second.y - first.y
        squared_length = dx * dx + dy * dy
        projection = -(first.x * dx + first.y * dy) / squared_length
        projection = max(0.0, min(1.0, projection))
        return sqrt((first.x + projection * dx) ** 2 + (first.y + projection * dy) ** 2)

    def triangle_distance_to_origin(vertices: tuple[Point, Point, Point]) -> float:
        crosses = []
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            crosses.append(
                (second.x - first.x) * (-first.y)
                - (second.y - first.y) * (-first.x)
            )
        if all(value >= -1e-9 for value in crosses) or all(
            value <= 1e-9 for value in crosses
        ):
            return 0.0
        return min(
            segment_distance_to_origin(first, second)
            for first, second in zip(vertices, vertices[1:] + vertices[:1])
        )

    for row in range(-vertical_limit, vertical_limit + 1):
        for column in range(-horizontal_limit, horizontal_limit + 1):
            lower_left = lattice_point(column, row)
            lower_right = lattice_point(column + 1, row)
            upper_left = lattice_point(column, row + 1)
            upper_right = lattice_point(column + 1, row + 1)
            triangles = (
                (lower_left, lower_right, upper_left),
                (lower_right, upper_left, upper_right),
            )
            for triangle in triangles:
                if triangle_distance_to_origin(triangle) <= target_radius_m + 1e-7:
                    for point in triangle:
                        selected[(round(point.x, 9), round(point.y, 9))] = point
    return tuple(selected.values())


def _route_length(start: Point, route: list[Point]) -> float:
    points = [start] + route
    return sum(first.distance_to(second) for first, second in zip(points, points[1:]))


def plan_station_route(stations: tuple[Point, ...], start: Point = Point(0.0, 0.0)) -> tuple[Point, ...]:
    """Construct and 2-opt shorten a deterministic open station route."""

    remaining = list(stations)
    route: list[Point] = []
    current = start
    while remaining:
        following = min(remaining, key=current.distance_to)
        route.append(following)
        remaining.remove(following)
        current = following

    best_length = _route_length(start, route)
    improved = True
    while improved:
        improved = False
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                candidate = route[:left] + list(reversed(route[left : right + 1])) + route[right + 1 :]
                candidate_length = _route_length(start, candidate)
                if candidate_length + 1e-7 < best_length:
                    route = candidate
                    best_length = candidate_length
                    improved = True
    return tuple(route)


def survey_stations_for_profile(profile: str) -> tuple[Point, ...]:
    """Select the strict or empirically shortened survey set."""

    stations = triangular_lattice_stations()
    if profile == "certified":
        return stations
    if profile == "balanced":
        retained = [
            point
            for index, point in enumerate(stations)
            if index not in {5, 6, 11, 12}
        ]
        retained.extend((Point(-1720.0, -850.0), Point(1720.0, -850.0)))
        return tuple(retained)
    if profile == "rapid":
        retained = list(stations[index] for index in FAST_STATION_INDICES)
        retained.append(Point(1700.0, -800.0))
        return tuple(retained)
    if profile == "turbo":
        retained = [
            stations[index]
            for index in FAST_STATION_INDICES
            if index != 10
        ]
        retained.append(Point(1700.0, -800.0))
        return tuple(retained)
    if profile == "fast":
        return tuple(stations[index] for index in FAST_STATION_INDICES)
    raise ValueError("survey_profile must be 'certified', 'balanced', 'rapid', 'turbo', or 'fast'.")


def bracketing_candidates(
    observation: Observation,
    step_m: float = CLEAR_RADIUS_M,
    deflection_deg: float = 8.0,
) -> tuple[Point, Point]:
    """Return two conservative bearing-side pursuit points after a failed clear."""

    if step_m <= 0.0:
        raise ValueError("step_m must be positive.")
    return (
        _bearing_point(
            observation.station,
            observation.bearing_deg - deflection_deg,
            step_m,
        ),
        _bearing_point(
            observation.station,
            observation.bearing_deg + deflection_deg,
            step_m,
        ),
    )


def _distance_to_feasible_region(point: Point, polygon: tuple[Point, ...]) -> float:
    """Return the Euclidean distance from a point to a convex feasible polygon."""

    if not polygon:
        return 0.0
    crosses = [
        (second.x - first.x) * (point.y - first.y)
        - (second.y - first.y) * (point.x - first.x)
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    ]
    if all(value >= -1e-7 for value in crosses) or all(
        value <= 1e-7 for value in crosses
    ):
        return 0.0

    distances: list[float] = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = second.x - first.x, second.y - first.y
        squared_length = dx * dx + dy * dy
        if squared_length <= 1e-18:
            distances.append(point.distance_to(first))
            continue
        projection = (
            (point.x - first.x) * dx + (point.y - first.y) * dy
        ) / squared_length
        projection = max(0.0, min(1.0, projection))
        nearest = Point(first.x + projection * dx, first.y + projection * dy)
        distances.append(point.distance_to(nearest))
    return min(distances)


class MixedDirectionalSearch:
    """Batch survey followed by opportunistic localization and certified pursuit."""

    def __init__(
        self,
        *,
        channels: range = range(1, 21),
        circle_sides: int = 180,
        max_sources: int = 16,
        observations_per_channel: int = 2,
        max_pursuit_steps: int = 80,
        lattice_edge_m: float = LATTICE_EDGE_M,
        pursuit_initial_step_m: float = 40.0,
        pursuit_deflection_deg: float = 8.0,
        pursuit_speculative_limit: int = 0,
        survey_profile: str = "turbo",
        enroute_detour_limit_m: float = 500.0,
        enroute_speculative_limit_m: float = 400.0,
    ) -> None:
        self.channels = tuple(channels)
        self.circle_sides = circle_sides
        self.max_sources = max_sources
        self.observations_per_channel = observations_per_channel
        self.max_pursuit_steps = max_pursuit_steps
        self.lattice_edge_m = lattice_edge_m
        self.pursuit_initial_step_m = pursuit_initial_step_m
        self.pursuit_deflection_deg = pursuit_deflection_deg
        self.pursuit_speculative_limit = pursuit_speculative_limit
        self.survey_profile = survey_profile
        self.enroute_detour_limit_m = enroute_detour_limit_m
        self.enroute_speculative_limit_m = enroute_speculative_limit_m
        self._enroute_attempted_channels: set[int] = set()
        self.states = {channel: ChannelState(channel) for channel in self.channels}
        self.measure_count = 0
        self.clear_attempt_count = 0

    def _known_source_count(self) -> int:
        return sum(state.detected or state.cleared for state in self.states.values())

    def _measure(self, robot: RobotInterface, point: Point, channel: int):
        reply = robot.measure(point, channel)
        self.measure_count += 1
        if reply.result not in {"no_signal", "near", "direction"}:
            raise RuntimeError(f"Unexpected measure result: {reply.result}")
        if reply.result == "direction" and reply.bearing_deg is None:
            raise RuntimeError("A direction result must contain bearing_deg.")
        return reply

    def _clear(self, robot: RobotInterface, point: Point, state: ChannelState) -> bool:
        reply = robot.clear(point, state.channel)
        self.clear_attempt_count += 1
        if reply.success:
            state.cleared = True
            state.clear_time_s = reply.virtual_time_s
        return reply.success

    def _scan_station(self, robot: RobotInterface, station: Point) -> None:
        channels = [
            channel
            for channel, state in self.states.items()
            if not state.cleared and len(state.observations) < self.observations_per_channel
        ]
        current_channel = getattr(robot, "current_channel", None)
        if current_channel in channels:
            channels.remove(current_channel)
            channels.insert(0, current_channel)
        for channel in channels:
            if self._known_source_count() >= self.max_sources:
                break
            reply = self._measure(robot, station, channel)
            state = self.states[channel]
            if reply.result == "near":
                if not self._clear(robot, station, state):
                    raise RuntimeError(f"Near clear failed for channel {channel}.")
            elif reply.result == "direction":
                _update_state_with_bearing(
                    state, station, float(reply.bearing_deg), self.circle_sides
                )

    def _clear_certified_en_route(
        self, robot: RobotInterface, remaining_stations: tuple[Point, ...]
    ) -> None:
        """Insert low-detour guaranteed clears into the remaining survey route."""

        while remaining_stations:
            if self._known_source_count() >= self.max_sources:
                return
            direct = robot.position.distance_to(remaining_stations[0])
            choices: list[tuple[float, Point, ChannelState]] = []
            for state in self.states.values():
                if not state.detected or state.cleared:
                    continue
                radius_m, center = _region_radius_and_center(state)
                if radius_m > CLEAR_RADIUS_M + 1e-7:
                    continue
                detour = (
                    robot.position.distance_to(center)
                    + center.distance_to(remaining_stations[0])
                    - direct
                )
                choices.append((detour, center, state))
            if choices:
                detour, center, state = min(choices, key=lambda item: item[0])
                if detour <= self.enroute_detour_limit_m:
                    if not self._clear(robot, center, state):
                        raise RuntimeError(
                            f"Certified en-route clear failed for channel {state.channel}."
                        )
                    continue

            speculative: list[tuple[float, Point, ChannelState]] = []
            for state in self.states.values():
                if (
                    state.cleared
                    or len(state.observations) < 2
                    or state.channel in self._enroute_attempted_channels
                ):
                    continue
                estimate = _nominal_bearing_intersection(state)
                if estimate is None:
                    continue
                detour = (
                    robot.position.distance_to(estimate)
                    + estimate.distance_to(remaining_stations[0])
                    - direct
                )
                speculative.append((detour, estimate, state))
            if not speculative:
                return
            detour, estimate, state = min(speculative, key=lambda item: item[0])
            if detour > self.enroute_speculative_limit_m:
                return
            self._enroute_attempted_channels.add(state.channel)
            if self._clear(robot, estimate, state):
                continue
            reply = self._measure(robot, estimate, state.channel)
            if reply.result == "near":
                raise RuntimeError(
                    f"Clear/measure inconsistency for channel {state.channel}."
                )
            if reply.result == "direction":
                _update_state_with_bearing(
                    state, estimate, float(reply.bearing_deg), self.circle_sides
                )

    def _try_certified_clear(self, robot: RobotInterface, state: ChannelState) -> bool:
        radius_m, center = _region_radius_and_center(state)
        if radius_m <= CLEAR_RADIUS_M + 1e-7:
            if not self._clear(robot, center, state):
                raise RuntimeError(f"Certified clear failed for channel {state.channel}.")
            return True
        return False

    def _pursue_from_observation(
        self,
        robot: RobotInterface,
        state: ChannelState,
        anchor: Observation,
        *,
        anchor_clear_failed: bool = False,
    ) -> None:
        """Clear a detected source with a worst-case finite-step guarantee."""

        current = anchor
        speculative_attempts = 0
        for _ in range(self.max_pursuit_steps):
            if not anchor_clear_failed:
                if self._clear(robot, current.station, state):
                    return
            anchor_clear_failed = False
            next_observation: Observation | None = None
            certified_lower_range_m = _distance_to_feasible_region(
                current.station, state.feasible_region
            )
            step_m = min(
                MAX_RECEPTION_RADIUS_M,
                max(
                    CLEAR_RADIUS_M,
                    self.pursuit_initial_step_m,
                    certified_lower_range_m,
                ),
            )
            while True:
                candidates = list(
                    bracketing_candidates(
                        current,
                        step_m=step_m,
                        deflection_deg=self.pursuit_deflection_deg,
                    )
                )
                candidates.sort(key=robot.position.distance_to)
                for candidate in candidates:
                    reply = self._measure(robot, candidate, state.channel)
                    if reply.result == "near":
                        if not self._clear(robot, candidate, state):
                            raise RuntimeError(f"Near clear failed for channel {state.channel}.")
                        return
                    if reply.result == "direction":
                        _update_state_with_bearing(
                            state, candidate, float(reply.bearing_deg), self.circle_sides
                        )
                        next_observation = state.observations[-1]
                        break
                if next_observation is not None or step_m <= CLEAR_RADIUS_M + 1e-9:
                    break
                step_m = max(CLEAR_RADIUS_M, step_m / 2.0)
            if next_observation is None:
                raise RuntimeError(
                    f"Both certified pursuit measurements lost channel {state.channel}."
                )
            current = next_observation

            if self._try_certified_clear(robot, state):
                return
            if (
                len(state.observations) >= 3
                and speculative_attempts < self.pursuit_speculative_limit
            ):
                estimate = _nominal_bearing_intersection(state)
                if (
                    estimate is not None
                    and current.station.distance_to(estimate) <= MAX_RECEPTION_RADIUS_M
                ):
                    speculative_attempts += 1
                    if self._clear(robot, estimate, state):
                        return
                    reply = self._measure(robot, estimate, state.channel)
                    if reply.result == "near":
                        raise RuntimeError(
                            f"Clear/measure inconsistency for channel {state.channel}."
                        )
                    if reply.result == "direction":
                        _update_state_with_bearing(
                            state, estimate, float(reply.bearing_deg), self.circle_sides
                        )
                        current = state.observations[-1]
                        anchor_clear_failed = True
        raise RuntimeError(f"Channel {state.channel} exceeded pursuit-step bound.")

    def _resolve_channel(self, robot: RobotInterface, state: ChannelState) -> None:
        if self._try_certified_clear(robot, state):
            return
        if len(state.observations) >= 2:
            estimate = _nominal_bearing_intersection(state)
            if estimate is not None:
                if self._clear(robot, estimate, state):
                    return
                reply = self._measure(robot, estimate, state.channel)
                if reply.result == "near":
                    raise RuntimeError(
                        f"Clear/measure inconsistency for channel {state.channel}."
                    )
                if reply.result == "direction":
                    _update_state_with_bearing(
                        state, estimate, float(reply.bearing_deg), self.circle_sides
                    )
                    if self._try_certified_clear(robot, state):
                        return
                    self._pursue_from_observation(
                        robot,
                        state,
                        state.observations[-1],
                        anchor_clear_failed=True,
                    )
                    return
        # A positive observation is a certified point inside the unknown
        # emission half-plane.  Start from the one closest to the current robot.
        anchor = min(state.observations, key=lambda item: robot.position.distance_to(item.station))
        self._pursue_from_observation(robot, state, anchor)

    def run(self, robot: RobotInterface) -> StrategyResult:
        if (
            abs(self.lattice_edge_m - LATTICE_EDGE_M) > 1e-9
            and self.survey_profile != "certified"
        ):
            raise ValueError("Shortened survey profiles require the default lattice edge.")
        station_set = (
            triangular_lattice_stations(edge_m=self.lattice_edge_m)
            if self.survey_profile == "certified"
            else survey_stations_for_profile(self.survey_profile)
        )
        stations = plan_station_route(station_set)
        visited: list[Point] = []
        for index, station in enumerate(stations):
            if self._known_source_count() >= self.max_sources:
                break
            self._clear_certified_en_route(robot, stations[index:])
            self._scan_station(robot, station)
            visited.append(station)

        unresolved = [state for state in self.states.values() if state.detected and not state.cleared]
        while unresolved:
            state = _plan_open_route(robot.position, unresolved)[0]
            unresolved.remove(state)
            self._resolve_channel(robot, state)

        cleared = tuple(channel for channel, state in self.states.items() if state.cleared)
        reached_source_upper_bound = len(cleared) == self.max_sources
        certified_absent = (
            tuple(
                channel
                for channel, state in self.states.items()
                if not state.detected and not state.cleared
            )
            if self.survey_profile == "certified" or reached_source_upper_bound
            else ()
        )
        average = robot.virtual_time_s / len(cleared) if cleared else float("inf")
        return StrategyResult(
            cleared_channels=cleared,
            certified_absent_channels=certified_absent,
            completion_time_s=robot.virtual_time_s,
            average_clear_time_s=average,
            survey_stations_visited=tuple(visited),
            measure_count=self.measure_count,
            clear_attempt_count=self.clear_attempt_count,
        )

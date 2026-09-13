"""Adaptive search, localization, and removal for Problem B, Question 3.

The policy separates completeness from efficiency.  Seven deterministic survey
stations cover the radius-1800 target disk with reception disks of radius 1000,
so every omnidirectional source is detected at least once.  Positive bearings
are then maintained as deterministic feasible polygons and actively refined
until their minimum enclosing circle fits inside the 20 m removal radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, pi, sin, sqrt
from typing import Protocol

from question1_geometry import (
    Observation,
    Point,
    clip_polygon,
    minimum_enclosing_circle,
    observation_halfplanes,
)
from question2_strategy import (
    circumscribed_circle_halfplanes,
    first_source_region,
)


TARGET_RADIUS_M = 1800.0
MIN_RECEPTION_RADIUS_M = 1000.0
MAX_RECEPTION_RADIUS_M = 1500.0
CLEAR_RADIUS_M = 20.0
NEAR_RADIUS_M = 5.0
BEARING_ERROR_DEG = 1.0
SURVEY_RING_RADIUS_M = 999.0
SURVEY_STATION_COUNT = 7


@dataclass(frozen=True)
class MeasureReply:
    result: str
    virtual_time_s: float
    bearing_deg: float | None = None


@dataclass(frozen=True)
class ClearReply:
    success: bool
    virtual_time_s: float


class RobotInterface(Protocol):
    @property
    def position(self) -> Point: ...

    @property
    def virtual_time_s(self) -> float: ...

    def measure(self, position: Point, channel: int) -> MeasureReply: ...

    def clear(self, position: Point, channel: int) -> ClearReply: ...


@dataclass
class ChannelState:
    channel: int
    observations: list[Observation] = field(default_factory=list)
    no_signal_stations: list[Point] = field(default_factory=list)
    feasible_region: tuple[Point, ...] = ()
    cleared: bool = False
    clear_time_s: float | None = None

    @property
    def detected(self) -> bool:
        return bool(self.observations)


@dataclass(frozen=True)
class StrategyResult:
    cleared_channels: tuple[int, ...]
    certified_absent_channels: tuple[int, ...]
    completion_time_s: float
    average_clear_time_s: float
    survey_stations_visited: tuple[Point, ...]
    measure_count: int
    clear_attempt_count: int


def survey_stations(ring_radius_m: float = SURVEY_RING_RADIUS_M) -> tuple[Point, ...]:
    """Return seven equally spaced stations on a survey ring."""

    if ring_radius_m <= 0.0:
        raise ValueError("ring_radius_m must be positive.")
    return tuple(
        Point(
            ring_radius_m * cos(2.0 * index * pi / SURVEY_STATION_COUNT),
            ring_radius_m * sin(2.0 * index * pi / SURVEY_STATION_COUNT),
        )
        for index in range(SURVEY_STATION_COUNT)
    )


def survey_covering_radius_m(
    target_radius_m: float = TARGET_RADIUS_M,
    ring_radius_m: float = SURVEY_RING_RADIUS_M,
) -> float:
    """Worst distance to the seven survey stations over the target disk.

    The only worst candidates are the target centre and the outer boundary
    midway between two adjacent ring stations.
    """

    inner_gap = ring_radius_m
    boundary_gap = sqrt(
        target_radius_m**2
        + ring_radius_m**2
        - 2.0
        * target_radius_m
        * ring_radius_m
        * cos(pi / SURVEY_STATION_COUNT)
    )
    return max(inner_gap, boundary_gap)


def _point_route_length(start: Point, route: list[Point]) -> float:
    return sum(
        first.distance_to(second)
        for first, second in zip([start] + route, route)
    )


def _plan_point_route(start: Point, points: tuple[Point, ...]) -> list[Point]:
    """Return a short open route through a moderate number of action points."""

    if len(points) < 2:
        return list(points)
    best_route: list[Point] | None = None
    best_length = float("inf")
    for first in points:
        remaining = [point for point in points if point is not first]
        route = [first]
        while remaining:
            following = min(remaining, key=route[-1].distance_to)
            route.append(following)
            remaining.remove(following)
        improved = True
        while improved:
            improved = False
            for left in range(len(route) - 1):
                for right in range(left + 1, len(route)):
                    candidate = (
                        route[:left]
                        + list(reversed(route[left : right + 1]))
                        + route[right + 1 :]
                    )
                    length = _point_route_length(start, candidate)
                    if length + 1e-7 < _point_route_length(start, route):
                        route = candidate
                        improved = True
        length = _point_route_length(start, route)
        if length < best_length:
            best_route = route
            best_length = length
    return best_route or []


def _bearing_point(origin: Point, bearing_deg: float, distance_m: float) -> Point:
    angle = (bearing_deg % 360.0) * pi / 180.0
    return Point(origin.x + distance_m * cos(angle), origin.y + distance_m * sin(angle))


def robust_second_station(
    first_observation: Observation,
    current_position: Point,
    offset_m: float = 1000.0,
    deflection_deg: float = 45.0,
) -> Point:
    """Choose one of two guaranteed-reception follow-up stations.

    The source lies at distance ``d <= 1500`` from the first station and its
    true bearing differs by at most one degree.  For either 1000 m candidate at
    45 degrees from the measured ray, its distance to every compatible source
    is no greater than ``max(1000, d)``.  The nearer symmetric candidate is
    selected to reduce travel.
    """

    candidates = (
        _bearing_point(
            first_observation.station,
            first_observation.bearing_deg + deflection_deg,
            offset_m,
        ),
        _bearing_point(
            first_observation.station,
            first_observation.bearing_deg - deflection_deg,
            offset_m,
        ),
    )
    return min(candidates, key=current_position.distance_to)


def fast_second_station(
    observation: Observation,
    current_position: Point,
    step_m: float = 700.0,
    deflection_deg: float = 7.0,
) -> Point:
    """Choose a short, guaranteed-reception probe with useful parallax."""

    candidates = (
        _bearing_point(
            observation.station,
            observation.bearing_deg + deflection_deg,
            step_m,
        ),
        _bearing_point(
            observation.station,
            observation.bearing_deg - deflection_deg,
            step_m,
        ),
    )
    return min(candidates, key=current_position.distance_to)


def _unused_robust_station(
    observations: list[Observation], current_position: Point
) -> Point:
    """Return a new station whose reception is certified by an earlier hit."""

    candidates: list[Point] = []
    # Every deflection below 59 degrees remains safe under the one-degree
    # bearing error; several choices prevent a repeated same-point observation.
    for observation in observations:
        for deflection_deg in (35.0, 45.0, 55.0):
            candidates.extend(
                (
                    _bearing_point(
                        observation.station,
                        observation.bearing_deg + deflection_deg,
                        1000.0,
                    ),
                    _bearing_point(
                        observation.station,
                        observation.bearing_deg - deflection_deg,
                        1000.0,
                    ),
                )
            )
    unused = [
        candidate
        for candidate in candidates
        if all(candidate.distance_to(item.station) > 1.0 for item in observations)
    ]
    if not unused:
        raise RuntimeError("No unused guaranteed-reception station is available.")
    return min(unused, key=current_position.distance_to)


def _update_state_with_bearing(
    state: ChannelState,
    station: Point,
    bearing_deg: float,
    circle_sides: int,
) -> None:
    observation = Observation(station, bearing_deg)
    if not state.observations:
        state.feasible_region = first_source_region(
            observation,
            error_deg=BEARING_ERROR_DEG,
            target_radius_m=TARGET_RADIUS_M,
            reception_upper_m=MAX_RECEPTION_RADIUS_M,
            near_threshold_m=NEAR_RADIUS_M,
            circle_sides=circle_sides,
        )
    else:
        constraints = list(observation_halfplanes(observation, BEARING_ERROR_DEG))
        constraints.extend(
            circumscribed_circle_halfplanes(
                station, MAX_RECEPTION_RADIUS_M, sides=circle_sides
            )
        )
        state.feasible_region = clip_polygon(
            state.feasible_region, constraints, tol=1e-7
        )
        if not state.feasible_region:
            raise RuntimeError(f"Channel {state.channel} has inconsistent bearings.")
    state.observations.append(observation)


def _region_radius_and_center(state: ChannelState) -> tuple[float, Point]:
    circle = minimum_enclosing_circle(state.feasible_region, tol=1e-7)
    return circle.radius, circle.center


def _nominal_bearing_intersection(state: ChannelState) -> Point | None:
    """Return an angular least-squares point estimate for cheap trial clears.

    This estimate is never used as a worst-case certificate.  A failed trial is
    followed by another measurement and the deterministic feasible-region
    policy remains the final fallback.
    """

    a00 = a01 = a11 = b0 = b1 = 0.0
    for observation in state.observations:
        angle = observation.bearing_deg * pi / 180.0
        ux, uy = cos(angle), sin(angle)
        m00, m01, m11 = 1.0 - ux * ux, -ux * uy, 1.0 - uy * uy
        a00 += m00
        a01 += m01
        a11 += m11
        b0 += m00 * observation.station.x + m01 * observation.station.y
        b1 += m01 * observation.station.x + m11 * observation.station.y
    determinant = a00 * a11 - a01 * a01
    if determinant <= 1e-4:
        return None
    point = Point(
        (b0 * a11 - b1 * a01) / determinant,
        (a00 * b1 - a01 * b0) / determinant,
    )

    def project(candidate: Point) -> Point:
        radial_distance = hypot(candidate.x, candidate.y)
        if radial_distance <= TARGET_RADIUS_M:
            return candidate
        scale = TARGET_RADIUS_M / radial_distance
        return Point(candidate.x * scale, candidate.y * scale)

    def angular_cost(candidate: Point) -> float:
        total = 0.0
        for observation in state.observations:
            predicted = atan2(
                candidate.y - observation.station.y,
                candidate.x - observation.station.x,
            )
            measured = observation.bearing_deg * pi / 180.0
            residual = atan2(sin(predicted - measured), cos(predicted - measured))
            total += residual * residual
        return total

    point = project(point)
    # Refine the algebraic line intersection with angular least squares because
    # the simulator perturbs angles, not perpendicular distances.
    for _ in range(12):
        h00 = h01 = h11 = g0 = g1 = 0.0
        for observation in state.observations:
            dx = point.x - observation.station.x
            dy = point.y - observation.station.y
            squared_distance = max(dx * dx + dy * dy, 1.0)
            jx, jy = -dy / squared_distance, dx / squared_distance
            predicted = atan2(dy, dx)
            measured = observation.bearing_deg * pi / 180.0
            residual = atan2(sin(predicted - measured), cos(predicted - measured))
            h00 += jx * jx
            h01 += jx * jy
            h11 += jy * jy
            g0 += jx * residual
            g1 += jy * residual
        damping = 1e-10
        h00 += damping
        h11 += damping
        normal_determinant = h00 * h11 - h01 * h01
        if normal_determinant <= 1e-18:
            break
        step_x = (-g0 * h11 + g1 * h01) / normal_determinant
        step_y = (-h00 * g1 + h01 * g0) / normal_determinant
        step_norm = hypot(step_x, step_y)
        if step_norm < 1e-4:
            break
        if step_norm > 500.0:
            scale = 500.0 / step_norm
            step_x *= scale
            step_y *= scale
        old_cost = angular_cost(point)
        accepted = False
        for line_search in range(8):
            fraction = 0.5**line_search
            candidate = project(
                Point(point.x + fraction * step_x, point.y + fraction * step_y)
            )
            if angular_cost(candidate) < old_cost:
                point = candidate
                accepted = True
                break
        if not accepted:
            break

    # Use the posterior centroid inside the deterministic polygon.  Each source
    # has one shared reception radius R ~ U[1000, 1500], so positive readings
    # impose R >= max(d_positive) and negative readings impose
    # R < min(d_negative).  The compatible interval length is the exact
    # marginal reception likelihood under that rehearsal prior.  It is used
    # only for trial clears; the deterministic feasible region remains the
    # correctness fallback.
    if state.feasible_region:
        polygon = state.feasible_region
        weighted_x = weighted_y = total_weight = 0.0
        anchor = polygon[0]
        for second, third in zip(polygon[1:-1], polygon[2:]):
            area = abs(
                (second.x - anchor.x) * (third.y - anchor.y)
                - (second.y - anchor.y) * (third.x - anchor.x)
            ) / 2.0
            if area <= 1e-12:
                continue
            samples = (
                Point(
                    (4.0 * anchor.x + second.x + third.x) / 6.0,
                    (4.0 * anchor.y + second.y + third.y) / 6.0,
                ),
                Point(
                    (anchor.x + 4.0 * second.x + third.x) / 6.0,
                    (anchor.y + 4.0 * second.y + third.y) / 6.0,
                ),
                Point(
                    (anchor.x + second.x + 4.0 * third.x) / 6.0,
                    (anchor.y + second.y + 4.0 * third.y) / 6.0,
                ),
            )
            for sample in samples:
                lower_radius = max(
                    MIN_RECEPTION_RADIUS_M,
                    *(sample.distance_to(item.station) for item in state.observations),
                )
                upper_radius = (
                    min(
                        MAX_RECEPTION_RADIUS_M,
                        *(sample.distance_to(point) for point in state.no_signal_stations),
                    )
                    if state.no_signal_stations
                    else MAX_RECEPTION_RADIUS_M
                )
                likelihood = max(0.0, upper_radius - lower_radius) / (
                    MAX_RECEPTION_RADIUS_M - MIN_RECEPTION_RADIUS_M
                )
                weight = area * likelihood / 3.0
                weighted_x += weight * sample.x
                weighted_y += weight * sample.y
                total_weight += weight
        if total_weight > 1e-12:
            return Point(weighted_x / total_weight, weighted_y / total_weight)
    return point


def _route_target(state: ChannelState) -> Point:
    estimate = _nominal_bearing_intersection(state)
    return estimate if estimate is not None else _region_radius_and_center(state)[1]


def _open_route_length(start: Point, route: list[ChannelState]) -> float:
    points = [start] + [_route_target(state) for state in route]
    return sum(first.distance_to(second) for first, second in zip(points, points[1:]))


def _two_opt_open_route(start: Point, route: list[ChannelState]) -> list[ChannelState]:
    """Shorten an open route while keeping its start fixed and end free."""

    best = list(route)
    best_length = _open_route_length(start, best)
    improved = True
    while improved:
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 1, len(best)):
                candidate = best[:left] + list(reversed(best[left : right + 1])) + best[right + 1 :]
                candidate_length = _open_route_length(start, candidate)
                if candidate_length + 1e-7 < best_length:
                    best = candidate
                    best_length = candidate_length
                    improved = True
    return best


def _plan_open_route(start: Point, states: list[ChannelState]) -> list[ChannelState]:
    """Plan a deterministic multi-start 2-opt route through feasible centres."""

    if len(states) < 2:
        return list(states)
    centres = {state.channel: _route_target(state) for state in states}
    candidates: list[list[ChannelState]] = []

    # Nearest-neighbour seeds from every possible first task avoid committing to
    # the locally nearest region when it produces an expensive final detour.
    for first in states:
        remaining = [state for state in states if state is not first]
        route = [first]
        point = centres[first.channel]
        while remaining:
            following = min(
                remaining,
                key=lambda state: point.distance_to(centres[state.channel]),
            )
            route.append(following)
            remaining.remove(following)
            point = centres[following.channel]
        candidates.append(_two_opt_open_route(start, route))

    return min(candidates, key=lambda route: _open_route_length(start, route))


class AdaptiveOmniSearch:
    """Complete seven-station discovery with immediate active localization."""

    def __init__(
        self,
        *,
        channels: range = range(1, 21),
        circle_sides: int = 180,
        max_sources: int = 16,
        max_localization_steps: int = 12,
        speculative_clear: bool = False,
        fast_second_step_m: float = 700.0,
        fast_second_deflection_deg: float = 7.0,
    ) -> None:
        self.channels = tuple(channels)
        self.circle_sides = circle_sides
        self.max_sources = max_sources
        self.max_localization_steps = max_localization_steps
        self.speculative_clear = speculative_clear
        self.fast_second_step_m = fast_second_step_m
        self.fast_second_deflection_deg = fast_second_deflection_deg
        self.states = {channel: ChannelState(channel) for channel in self.channels}
        self.measure_count = 0
        self.clear_attempt_count = 0

    def _measure(
        self, robot: RobotInterface, point: Point, channel: int
    ) -> MeasureReply:
        reply = robot.measure(point, channel)
        self.measure_count += 1
        if reply.result not in {"no_signal", "near", "direction"}:
            raise RuntimeError(f"Unexpected measure result: {reply.result}")
        if reply.result == "direction" and reply.bearing_deg is None:
            raise RuntimeError("A direction result must contain bearing_deg.")
        return reply

    def _clear(
        self, robot: RobotInterface, point: Point, state: ChannelState
    ) -> bool:
        reply = robot.clear(point, state.channel)
        self.clear_attempt_count += 1
        if reply.success:
            state.cleared = True
            state.clear_time_s = reply.virtual_time_s
        return reply.success

    def _resolve_channel(self, robot: RobotInterface, state: ChannelState) -> None:
        speculative_attempts = 0
        for _ in range(self.max_localization_steps):
            if (
                self.speculative_clear
                and len(state.observations) >= 2
                and speculative_attempts < 2
            ):
                estimate = _nominal_bearing_intersection(state)
                if estimate is not None:
                    speculative_attempts += 1
                    if self._clear(robot, estimate, state):
                        return
                    # Measure at the failed trial point: it costs no extra travel
                    # and usually supplies a high-parallax correction bearing.
                    reply = self._measure(robot, estimate, state.channel)
                    if reply.result == "direction":
                        _update_state_with_bearing(
                            state,
                            estimate,
                            float(reply.bearing_deg),
                            self.circle_sides,
                        )
                        continue
                    if reply.result == "near":
                        raise RuntimeError(
                            f"Clear/measure inconsistency for channel {state.channel}."
                        )
                    speculative_attempts = 2

            radius_m, center = _region_radius_and_center(state)
            if radius_m <= CLEAR_RADIUS_M + 1e-7:
                if not self._clear(robot, center, state):
                    raise RuntimeError(
                        f"Guaranteed clear failed for channel {state.channel}."
                    )
                return

            # The feasible-region centre is a guaranteed reception point only
            # after its radius has fallen below the minimum reception radius.
            if len(state.observations) == 1:
                next_point = fast_second_station(
                    state.observations[0],
                    robot.position,
                    self.fast_second_step_m,
                    self.fast_second_deflection_deg,
                )
            elif radius_m > MIN_RECEPTION_RADIUS_M + 1e-7:
                next_point = _unused_robust_station(
                    state.observations, robot.position
                )
            else:
                next_point = center

            reply = self._measure(robot, next_point, state.channel)
            if reply.result == "near":
                if not self._clear(robot, next_point, state):
                    raise RuntimeError(f"Near clear failed for channel {state.channel}.")
                return
            if reply.result == "no_signal":
                raise RuntimeError(
                    f"Guaranteed follow-up lost channel {state.channel} signal."
                )
            _update_state_with_bearing(
                state,
                next_point,
                float(reply.bearing_deg),
                self.circle_sides,
            )

        raise RuntimeError(f"Channel {state.channel} did not converge for clearing.")

    def _scan_station(self, robot: RobotInterface, station: Point) -> None:
        unknown = [
            channel
            for channel, state in self.states.items()
            if not state.detected and not state.cleared
        ]
        if not unknown:
            return

        # Starting with the current channel saves one switch whenever possible.
        current_channel = getattr(robot, "current_channel", None)
        if current_channel in unknown:
            unknown.remove(current_channel)
            unknown.insert(0, current_channel)

        newly_detected: list[ChannelState] = []
        for channel in unknown:
            reply = self._measure(robot, station, channel)
            if reply.result == "near":
                state = self.states[channel]
                if not self._clear(robot, station, state):
                    raise RuntimeError(f"Near clear failed for channel {channel}.")
            elif reply.result == "direction":
                state = self.states[channel]
                _update_state_with_bearing(
                    state,
                    station,
                    float(reply.bearing_deg),
                    self.circle_sides,
                )
                newly_detected.append(state)
            else:
                self.states[channel].no_signal_stations.append(station)

        # Resolve the nearest current feasible region first.
        while newly_detected:
            state = min(
                newly_detected,
                key=lambda item: robot.position.distance_to(
                    _region_radius_and_center(item)[1]
                ),
            )
            newly_detected.remove(state)
            self._resolve_channel(robot, state)

    def run(self, robot: RobotInterface) -> StrategyResult:
        if survey_covering_radius_m() > MIN_RECEPTION_RADIUS_M + 1e-9:
            raise RuntimeError("Survey stations do not guarantee full reception coverage.")

        unvisited = list(survey_stations())
        visited: list[Point] = []
        first_station = unvisited.pop(0)
        self._scan_station(robot, first_station)
        visited.append(first_station)

        while unvisited and sum(
            state.detected or state.cleared for state in self.states.values()
        ) < self.max_sources:
            station = min(unvisited, key=robot.position.distance_to)
            unvisited.remove(station)
            self._scan_station(robot, station)
            visited.append(station)

        cleared = tuple(
            channel for channel, state in self.states.items() if state.cleared
        )
        # If fewer than the known maximum of 16 sources were cleared, all seven
        # stations have been scanned and every never-detected channel is certified
        # absent by the 1000 m covering guarantee.
        certified_absent = tuple(
            channel
            for channel, state in self.states.items()
            if not state.detected and not state.cleared
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


class BatchOmniSearch(AdaptiveOmniSearch):
    """Survey first, then clear sources in a nearest-region batch.

    This is the travel-efficient policy used for the competition metric.  Shared
    survey stations often provide two or more bearings for many channels at the
    cost of one physical visit, after which the remaining active localization
    tasks are ordered by distance from the robot's current position.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.minimum_surveys_before_enroute = int(
            kwargs.pop("minimum_surveys_before_enroute", 1)
        )
        self.enroute_detour_limit_m = float(
            kwargs.pop("enroute_detour_limit_m", 500.0)
        )
        self.nominal_enroute_detour_limit_m = float(
            kwargs.pop("nominal_enroute_detour_limit_m", 350.0)
        )
        kwargs.setdefault("speculative_clear", True)
        super().__init__(*args, **kwargs)

    def _clear_certified_en_route(
        self, robot: RobotInterface, unvisited: list[Point]
    ) -> None:
        """Insert cheap certified clears into the still-required survey route."""

        while unvisited:
            direct_to_survey = min(robot.position.distance_to(point) for point in unvisited)
            choices: list[tuple[float, Point, ChannelState]] = []
            for state in self.states.values():
                if not state.detected or state.cleared:
                    continue
                radius_m, centre = _region_radius_and_center(state)
                if radius_m > CLEAR_RADIUS_M + 1e-7:
                    continue
                detour = (
                    robot.position.distance_to(centre)
                    + min(centre.distance_to(point) for point in unvisited)
                    - direct_to_survey
                )
                choices.append((detour, centre, state))
            if not choices:
                return
            detour, centre, state = min(choices, key=lambda item: item[0])
            if detour > self.enroute_detour_limit_m:
                return
            if not self._clear(robot, centre, state):
                raise RuntimeError(
                    f"Guaranteed en-route clear failed for channel {state.channel}."
                )

    def _scan_station_deferred(self, robot: RobotInterface, station: Point) -> int:
        detected_before = sum(state.detected for state in self.states.values())
        channels = [
            channel
            for channel, state in self.states.items()
            if not state.cleared and len(state.observations) < 2
        ]
        channels.sort(
            key=lambda channel: (
                not self.states[channel].detected,
                channel,
            )
        )
        current_channel = getattr(robot, "current_channel", None)
        if current_channel in channels:
            channels.remove(current_channel)
            channels.insert(0, current_channel)

        for channel in channels:
            state = self.states[channel]
            found_count = sum(
                item.detected or item.cleared for item in self.states.values()
            )
            if not state.detected and found_count >= self.max_sources:
                continue
            reply = self._measure(robot, station, channel)
            if reply.result == "near":
                if not self._clear(robot, station, state):
                    raise RuntimeError(f"Near clear failed for channel {channel}.")
            elif reply.result == "direction":
                _update_state_with_bearing(
                    state,
                    station,
                    float(reply.bearing_deg),
                    self.circle_sides,
                )
            else:
                state.no_signal_stations.append(station)
        return sum(state.detected for state in self.states.values()) - detected_before

    def _clear_nominal_en_route(
        self, robot: RobotInterface, unvisited: list[Point]
    ) -> None:
        """Resolve sources whose estimate is a small detour to the next survey stop."""

        while unvisited:
            next_survey = unvisited[0]
            direct = robot.position.distance_to(next_survey)
            choices: list[tuple[float, ChannelState]] = []
            for state in self.states.values():
                if state.cleared or len(state.observations) < 2:
                    continue
                target = _route_target(state)
                detour = (
                    robot.position.distance_to(target)
                    + target.distance_to(next_survey)
                    - direct
                )
                choices.append((detour, state))
            if not choices:
                return
            detour, state = min(choices, key=lambda item: (item[0], item[1].channel))
            if detour > self.nominal_enroute_detour_limit_m:
                return
            self._resolve_channel(robot, state)

    def run(self, robot: RobotInterface) -> StrategyResult:
        if survey_covering_radius_m() > MIN_RECEPTION_RADIUS_M + 1e-9:
            raise RuntimeError("Survey stations do not guarantee full reception coverage.")

        unvisited = list(survey_stations())
        visited: list[Point] = []
        first_station = unvisited.pop(0)
        self._scan_station_deferred(robot, first_station)
        visited.append(first_station)
        while unvisited and sum(
            state.detected or state.cleared for state in self.states.values()
        ) < self.max_sources:
            if len(visited) >= self.minimum_surveys_before_enroute:
                self._clear_certified_en_route(robot, unvisited)
                self._clear_nominal_en_route(robot, unvisited)
            station = unvisited.pop(0)
            self._scan_station_deferred(robot, station)
            visited.append(station)

        unresolved = [
            state for state in self.states.values() if state.detected and not state.cleared
        ]
        while unresolved:
            action_targets = [
                (
                    fast_second_station(
                        state.observations[0],
                        robot.position,
                        self.fast_second_step_m,
                        self.fast_second_deflection_deg,
                    )
                    if len(state.observations) == 1
                    else _route_target(state),
                    state,
                )
                for state in unresolved
            ]
            action_route = _plan_point_route(
                robot.position,
                tuple(point for point, _ in action_targets),
            )
            state = min(
                action_targets,
                key=lambda item: item[0].distance_to(action_route[0]),
            )[1]
            unresolved.remove(state)
            self._resolve_channel(robot, state)

        cleared = tuple(
            channel for channel, state in self.states.items() if state.cleared
        )
        certified_absent = tuple(
            channel
            for channel, state in self.states.items()
            if not state.detected and not state.cleared
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


def bearing_deg(origin: Point, target: Point) -> float:
    """Return the problem's east-zero, counter-clockwise bearing."""

    return degrees(atan2(target.y - origin.y, target.x - origin.x)) % 360.0


def distance_m(first: Point, second: Point) -> float:
    return hypot(first.x - second.x, first.y - second.y)

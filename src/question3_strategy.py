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
SURVEY_RING_RADIUS_M = 1200.0


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
    """Return the origin and six equally spaced stations on a survey ring."""

    if ring_radius_m <= 0.0:
        raise ValueError("ring_radius_m must be positive.")
    return (Point(0.0, 0.0),) + tuple(
        Point(
            ring_radius_m * cos(index * pi / 3.0),
            ring_radius_m * sin(index * pi / 3.0),
        )
        for index in range(6)
    )


def survey_covering_radius_m(
    target_radius_m: float = TARGET_RADIUS_M,
    ring_radius_m: float = SURVEY_RING_RADIUS_M,
) -> float:
    """Worst distance to the seven survey stations over the target disk.

    In a 60-degree sector the only candidates are the Voronoi junction between
    the centre and two adjacent ring stations, and the outer boundary midway
    between those stations.
    """

    inner_gap = ring_radius_m / sqrt(3.0)
    boundary_gap = sqrt(
        target_radius_m**2
        + ring_radius_m**2
        - 2.0 * target_radius_m * ring_radius_m * cos(pi / 6.0)
    )
    return max(inner_gap, boundary_gap)


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


class AdaptiveOmniSearch:
    """Complete seven-station discovery with immediate active localization."""

    def __init__(
        self,
        *,
        channels: range = range(1, 21),
        circle_sides: int = 180,
        max_sources: int = 16,
        max_localization_steps: int = 12,
    ) -> None:
        self.channels = tuple(channels)
        self.circle_sides = circle_sides
        self.max_sources = max_sources
        self.max_localization_steps = max_localization_steps
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
        for _ in range(self.max_localization_steps):
            radius_m, center = _region_radius_and_center(state)
            if radius_m <= CLEAR_RADIUS_M + 1e-7:
                if not self._clear(robot, center, state):
                    raise RuntimeError(
                        f"Guaranteed clear failed for channel {state.channel}."
                    )
                return

            # The feasible-region centre is a guaranteed reception point only
            # after its radius has fallen below the minimum reception radius.
            if radius_m > MIN_RECEPTION_RADIUS_M + 1e-7:
                next_point = _unused_robust_station(
                    state.observations, robot.position
                )
            elif len(state.observations) == 1:
                next_point = robust_second_station(
                    state.observations[0], robot.position
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
        # The origin must be scanned first; subsequent ring stations are selected
        # greedily from the robot's post-clear position.
        origin = unvisited.pop(0)
        self._scan_station(robot, origin)
        visited.append(origin)

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

    def _scan_station_deferred(self, robot: RobotInterface, station: Point) -> None:
        channels = [
            channel for channel, state in self.states.items() if not state.cleared
        ]
        current_channel = getattr(robot, "current_channel", None)
        if current_channel in channels:
            channels.remove(current_channel)
            channels.insert(0, current_channel)

        for channel in channels:
            reply = self._measure(robot, station, channel)
            state = self.states[channel]
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

    def run(self, robot: RobotInterface) -> StrategyResult:
        if survey_covering_radius_m() > MIN_RECEPTION_RADIUS_M + 1e-9:
            raise RuntimeError("Survey stations do not guarantee full reception coverage.")

        unvisited = list(survey_stations())
        visited: list[Point] = []
        origin = unvisited.pop(0)
        self._scan_station_deferred(robot, origin)
        visited.append(origin)
        while unvisited and sum(
            state.detected or state.cleared for state in self.states.values()
        ) < self.max_sources:
            station = min(unvisited, key=robot.position.distance_to)
            unvisited.remove(station)
            self._scan_station_deferred(robot, station)
            visited.append(station)

        unresolved = [
            state for state in self.states.values() if state.detected and not state.cleared
        ]
        while unresolved:
            state = min(
                unresolved,
                key=lambda item: robot.position.distance_to(
                    _region_radius_and_center(item)[1]
                ),
            )
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

"""Deterministic local simulator for Question 3 strategy validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from random import Random

from question1_geometry import Point
from question3_strategy import (
    ClearReply,
    MeasureReply,
    bearing_deg,
    distance_m,
)


@dataclass
class OmniSource:
    channel: int
    position: Point
    reception_radius_m: float
    cleared: bool = False


class LocalOmniSimulator:
    """Implements the official motion, channel, sensing, and clearing costs."""

    def __init__(self, sources: list[OmniSource], seed: int = 0) -> None:
        channels = [source.channel for source in sources]
        if len(channels) != len(set(channels)):
            raise ValueError("Each source must use a different channel.")
        self.sources = {source.channel: source for source in sources}
        self._position = Point(0.0, 0.0)
        self.current_channel = 1
        self._virtual_time_s = 0.0
        self._rng = Random(seed)
        self._error_cache: dict[tuple[int, int, int], float] = {}
        self.actions: list[dict[str, object]] = []

    @property
    def position(self) -> Point:
        return self._position

    @property
    def virtual_time_s(self) -> float:
        return self._virtual_time_s

    def _move(self, point: Point) -> float:
        movement_s = distance_m(self._position, point) / 5.0
        self._virtual_time_s += movement_s
        self._position = point
        return movement_s

    def _fixed_error(self, channel: int, point: Point) -> float:
        # The same electromagnetic environment at the same point gives the same
        # reading error.  Millimetre rounding avoids floating representation noise.
        key = (channel, round(point.x * 1000.0), round(point.y * 1000.0))
        if key not in self._error_cache:
            self._error_cache[key] = self._rng.uniform(-1.0, 1.0)
        return self._error_cache[key]

    def measure(self, position: Point, channel: int) -> MeasureReply:
        start = self._virtual_time_s
        movement_s = self._move(position)
        switch_s = 0.0 if channel == self.current_channel else 1.0
        self._virtual_time_s += switch_s + 5.0
        self.current_channel = channel

        source = self.sources.get(channel)
        measured_bearing: float | None = None
        if source is None or source.cleared:
            result = "no_signal"
        else:
            separation = distance_m(position, source.position)
            if separation > source.reception_radius_m + 1e-9:
                result = "no_signal"
            elif separation <= 5.0 + 1e-9:
                result = "near"
            else:
                result = "direction"
                measured_bearing = (
                    bearing_deg(position, source.position)
                    + self._fixed_error(channel, position)
                ) % 360.0

        self.actions.append(
            {
                "action": "measure",
                "channel": channel,
                "x": position.x,
                "y": position.y,
                "result": result,
                "bearing_deg": measured_bearing,
                "movement_s": movement_s,
                "switch_s": switch_s,
                "action_s": 5.0,
                "start_s": start,
                "end_s": self._virtual_time_s,
            }
        )
        return MeasureReply(result, self._virtual_time_s, measured_bearing)

    def clear(self, position: Point, channel: int) -> ClearReply:
        start = self._virtual_time_s
        movement_s = self._move(position)
        source = self.sources.get(channel)
        success = bool(
            source is not None
            and not source.cleared
            and distance_m(position, source.position) <= 20.0 + 1e-9
        )
        action_s = 5.0 if success else 3.0
        self._virtual_time_s += action_s
        if success and source is not None:
            source.cleared = True
        self.actions.append(
            {
                "action": "clear",
                "channel": channel,
                "x": position.x,
                "y": position.y,
                "result": "success" if success else "no_target_in_range",
                "movement_s": movement_s,
                "switch_s": 0.0,
                "action_s": action_s,
                "start_s": start,
                "end_s": self._virtual_time_s,
            }
        )
        return ClearReply(success, self._virtual_time_s)


def random_case(seed: int, minimum_sources: int = 10, maximum_sources: int = 16) -> list[OmniSource]:
    """Generate one reproducible all-omnidirectional training case."""

    rng = Random(seed)
    count = rng.randint(minimum_sources, maximum_sources)
    channels = rng.sample(range(1, 21), count)
    sources = []
    for channel in channels:
        radius = 1800.0 * sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * pi)
        sources.append(
            OmniSource(
                channel=channel,
                position=Point(radius * cos(angle), radius * sin(angle)),
                reception_radius_m=rng.uniform(1000.0, 1500.0),
            )
        )
    return sources

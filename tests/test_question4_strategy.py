from __future__ import annotations

import sys
import unittest
from math import cos, pi, sin, sqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from question1_geometry import Observation, Point  # noqa: E402
from question3_strategy import MIN_RECEPTION_RADIUS_M  # noqa: E402
from question4_simulator import LocalMixedSimulator, MixedSource, random_case  # noqa: E402
from question4_strategy import (  # noqa: E402
    LATTICE_EDGE_M,
    MixedDirectionalSearch,
    bracketing_candidates,
    plan_station_route,
    survey_stations_for_profile,
    triangular_lattice_stations,
)


class Question4StrategyTests(unittest.TestCase):
    def test_lattice_intercepts_every_sampled_semicircle(self) -> None:
        stations = triangular_lattice_stations()
        self.assertGreater(len(stations), 20)
        for radial_index in range(13):
            radius = 1800.0 * radial_index / 12.0
            for angle_index in range(36):
                angle = 2.0 * pi * angle_index / 36.0
                source = Point(radius * cos(angle), radius * sin(angle))
                nearby = [
                    station
                    for station in stations
                    if station.distance_to(source) <= MIN_RECEPTION_RADIUS_M + 1e-7
                ]
                for direction_index in range(36):
                    direction = 2.0 * pi * direction_index / 36.0
                    self.assertTrue(
                        any(
                            (station.x - source.x) * cos(direction)
                            + (station.y - source.y) * sin(direction)
                            >= -1e-7
                            for station in nearby
                        )
                    )

    def test_bracketing_step_preserves_range_and_one_emission_side(self) -> None:
        source = Point(0.0, 0.0)
        for distance in (20.0001, 50.0, 500.0, 1500.0):
            station = Point(-distance, 0.0)
            # True bearing is zero; exhaust the two reading-error endpoints.
            for measured in (-1.0, 1.0):
                candidates = bracketing_candidates(Observation(station, measured))
                self.assertTrue(all(point.distance_to(source) <= distance + 1e-7 for point in candidates))
                # The worst emission boundary is vertical through the source.
                for sign in (-1.0, 1.0):
                    self.assertTrue(any(sign * point.y >= -1e-7 for point in candidates))

    def test_boundary_directional_sources_are_cleared(self) -> None:
        sources: list[MixedSource] = []
        for index, channel in enumerate(range(1, 11)):
            angle = 2.0 * pi * index / 10.0
            radius = 1799.0 if index % 2 == 0 else 900.0
            position = Point(radius * cos(angle), radius * sin(angle))
            # Tangential directions create difficult half-plane boundaries.
            direction = (angle * 180.0 / pi + 90.0) % 360.0 if index < 5 else None
            sources.append(MixedSource(channel, position, 1000.0, direction))
        simulator = LocalMixedSimulator(sources, seed=91)
        result = MixedDirectionalSearch(circle_sides=120).run(simulator)
        self.assertEqual(len(result.cleared_channels), len(sources))
        self.assertTrue(all(source.cleared for source in sources))

    def test_source_point_is_inside_every_directional_halfplane(self) -> None:
        source = MixedSource(1, Point(0.0, 0.0), 1000.0, 180.0)
        simulator = LocalMixedSimulator([source], seed=5)
        reply = simulator.measure(Point(0.0, 0.0), 1)
        self.assertEqual(reply.result, "near")

    def test_random_mixed_cases_clear_every_source(self) -> None:
        for seed in range(10):
            sources = random_case(seed)
            simulator = LocalMixedSimulator(sources, seed=10_000 + seed)
            result = MixedDirectionalSearch(circle_sides=120).run(simulator)
            self.assertEqual(len(result.cleared_channels), len(sources), msg=f"seed={seed}")
            self.assertLessEqual(len(result.survey_stations_visited), len(triangular_lattice_stations()))
            self.assertGreater(result.average_clear_time_s, 0.0)

    def test_lattice_edge_has_reception_margin(self) -> None:
        self.assertLess(LATTICE_EDGE_M, MIN_RECEPTION_RADIUS_M)
        stations = triangular_lattice_stations()
        self.assertEqual(len(stations), 25)
        route = plan_station_route(stations)
        route_length = sum(
            first.distance_to(second)
            for first, second in zip((Point(0.0, 0.0),) + route, route)
        )
        self.assertLess(route_length, 24_200.0)

    def test_survey_profile_sizes(self) -> None:
        self.assertEqual(len(survey_stations_for_profile("certified")), 25)
        self.assertEqual(len(survey_stations_for_profile("balanced")), 23)
        self.assertEqual(len(survey_stations_for_profile("fast")), 20)
        with self.assertRaises(ValueError):
            survey_stations_for_profile("unknown")


if __name__ == "__main__":
    unittest.main()

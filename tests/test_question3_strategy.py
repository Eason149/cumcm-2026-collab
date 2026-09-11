from __future__ import annotations

import sys
import unittest
from math import cos, pi, sin
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from question1_geometry import Observation, Point  # noqa: E402
from question3_simulator import (  # noqa: E402
    LocalOmniSimulator,
    OmniSource,
    random_case,
)
from question3_strategy import (  # noqa: E402
    AdaptiveOmniSearch,
    BatchOmniSearch,
    MIN_RECEPTION_RADIUS_M,
    fast_second_station,
    robust_second_station,
    survey_covering_radius_m,
    survey_stations,
)


class Question3StrategyTests(unittest.TestCase):
    def test_seven_stations_cover_target_with_margin(self) -> None:
        stations = survey_stations()
        self.assertEqual(len(stations), 7)
        self.assertTrue(
            all(abs(station.distance_to(Point(0.0, 0.0)) - 999.0) < 1e-7 for station in stations)
        )
        self.assertLess(survey_covering_radius_m(), MIN_RECEPTION_RADIUS_M)
        patrol_distance = Point(0.0, 0.0).distance_to(stations[0]) + sum(
            first.distance_to(second)
            for first, second in zip(stations, stations[1:])
        )
        self.assertLess(patrol_distance, 6201.0)

        # Direct deterministic audit of the full disk.
        worst = 0.0
        for radial_index in range(101):
            radius = 1800.0 * radial_index / 100.0
            for angle_index in range(720):
                angle = 2.0 * pi * angle_index / 720.0
                point = Point(radius * cos(angle), radius * sin(angle))
                nearest = min(point.distance_to(station) for station in stations)
                worst = max(worst, nearest)
        self.assertLessEqual(worst, survey_covering_radius_m() + 1e-6)

    def test_robust_second_station_preserves_minimum_reception(self) -> None:
        observation = Observation(Point(0.0, 0.0), 0.0)
        candidate = robust_second_station(observation, Point(0.0, 0.0))
        for distance in range(5, 1501, 5):
            for error_deg in (-1.0, 0.0, 1.0):
                angle = error_deg * pi / 180.0
                source = Point(distance * cos(angle), distance * sin(angle))
                allowed = max(1000.0, float(distance))
                self.assertLessEqual(candidate.distance_to(source), allowed + 1e-7)

    def test_fast_second_station_preserves_minimum_reception(self) -> None:
        observation = Observation(Point(0.0, 0.0), 0.0)
        candidate = fast_second_station(observation, Point(0.0, 0.0))
        for distance in range(5, 1501, 5):
            for error_deg in (-1.0, 0.0, 1.0):
                angle = error_deg * pi / 180.0
                source = Point(distance * cos(angle), distance * sin(angle))
                self.assertLessEqual(candidate.distance_to(source), 1000.0 + 1e-7)

    def test_local_simulator_reuses_same_point_error(self) -> None:
        source = OmniSource(1, Point(500.0, 100.0), 1000.0)
        simulator = LocalOmniSimulator([source], seed=7)
        first = simulator.measure(Point(0.0, 0.0), 1)
        second = simulator.measure(Point(0.0, 0.0), 1)
        self.assertEqual(first.result, "direction")
        self.assertEqual(first.bearing_deg, second.bearing_deg)

    def test_strategy_clears_boundary_and_low_radius_sources(self) -> None:
        sources = []
        for index, channel in enumerate(range(1, 11)):
            angle = 2.0 * pi * index / 10.0 + pi / 10.0
            radius = 1799.0 if index % 2 == 0 else 1100.0
            sources.append(
                OmniSource(
                    channel,
                    Point(radius * cos(angle), radius * sin(angle)),
                    1000.0,
                )
            )
        simulator = LocalOmniSimulator(sources, seed=23)
        result = AdaptiveOmniSearch().run(simulator)
        self.assertEqual(len(result.cleared_channels), len(sources))
        self.assertTrue(all(source.cleared for source in sources))
        self.assertLessEqual(len(result.survey_stations_visited), 7)
        self.assertTrue(all(action["result"] != "no_target_in_range" for action in simulator.actions if action["action"] == "clear"))

    def test_random_cases_clear_every_source(self) -> None:
        for seed in range(10):
            sources = random_case(seed)
            simulator = LocalOmniSimulator(sources, seed=1000 + seed)
            result = AdaptiveOmniSearch(circle_sides=120).run(simulator)
            self.assertEqual(len(result.cleared_channels), len(sources), msg=f"seed={seed}")
            self.assertGreater(result.average_clear_time_s, 0.0)
            # At most two cheap nominal-position attempts precede the certified
            # feasible-region fallback for each source.
            self.assertLessEqual(result.clear_attempt_count, 3 * len(sources))

    def test_batch_policy_clears_every_source(self) -> None:
        for seed in range(5):
            sources = random_case(seed)
            simulator = LocalOmniSimulator(sources, seed=2000 + seed)
            result = BatchOmniSearch(circle_sides=120).run(simulator)
            self.assertEqual(len(result.cleared_channels), len(sources), msg=f"seed={seed}")
            self.assertTrue(all(source.cleared for source in sources))
            # At most two cheap nominal-position attempts precede the certified
            # feasible-region fallback for each source.
            self.assertLessEqual(result.clear_attempt_count, 3 * len(sources))


if __name__ == "__main__":
    unittest.main()

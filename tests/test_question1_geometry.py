"""Unit tests for the Question 1 geometry kernel."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from question1_geometry import (  # noqa: E402
    HalfPlane,
    Observation,
    Point,
    check_diametral_circles,
    intersect_halfplanes,
    localization_region,
    minimum_enclosing_circle,
    observation_halfplanes,
    polygon_diameter,
    solve_question1,
)


class Question1GeometryTests(unittest.TestCase):
    def test_forward_wedge_rejects_backward_extension(self) -> None:
        planes = observation_halfplanes(Observation(Point(0.0, 0.0), 0.0), 1.0)
        self.assertTrue(all(plane.contains(Point(100.0, 0.0)) for plane in planes))
        self.assertFalse(all(plane.contains(Point(-100.0, 0.0)) for plane in planes))

    def test_angle_wraparound(self) -> None:
        planes = observation_halfplanes(Observation(Point(0.0, 0.0), 359.5), 1.0)
        self.assertTrue(all(plane.contains(Point(100.0, 0.0)) for plane in planes))

    def test_single_observation_is_unbounded(self) -> None:
        result = localization_region([Observation(Point(0.0, 0.0), 30.0)])
        self.assertEqual(result.status, "unbounded")

    def test_inconsistent_halfplanes_are_empty(self) -> None:
        result = intersect_halfplanes(
            [HalfPlane(1.0, 0.0, 0.0), HalfPlane(-1.0, 0.0, -1.0)]
        )
        self.assertEqual(result.status, "empty")

    def test_symmetric_stations_produce_bounded_region(self) -> None:
        observations = [
            Observation(Point(-800.0, 0.0), 0.0),
            Observation(Point(800.0, 0.0), 180.0),
            Observation(Point(0.0, -900.0), 90.0),
            Observation(Point(0.0, 900.0), 270.0),
        ]
        result = solve_question1(observations)
        self.assertEqual(result.intersection.status, "bounded")
        self.assertIsNotNone(result.diameter)
        self.assertIsNotNone(result.minimum_enclosing_circle)
        self.assertGreaterEqual(len(result.intersection.vertices), 4)
        for observation in observations:
            planes = observation_halfplanes(observation)
            self.assertTrue(all(plane.contains(Point(0.0, 0.0)) for plane in planes))

    def test_equilateral_triangle_disproves_universal_diameter_circle(self) -> None:
        side = 10.0
        vertices = (
            Point(0.0, 0.0),
            Point(side, 0.0),
            Point(side / 2.0, side * math.sqrt(3.0) / 2.0),
        )
        diameter, pairs = polygon_diameter(vertices)
        checks = check_diametral_circles(vertices, pairs)
        enclosing = minimum_enclosing_circle(vertices)
        self.assertAlmostEqual(diameter, side, places=8)
        self.assertTrue(all(not check.covers for check in checks))
        self.assertAlmostEqual(enclosing.radius, side / math.sqrt(3.0), places=8)
        self.assertGreater(enclosing.radius, diameter / 2.0)

    def test_rectangle_is_covered_by_a_diametral_circle(self) -> None:
        vertices = (
            Point(-2.0, -1.0),
            Point(2.0, -1.0),
            Point(2.0, 1.0),
            Point(-2.0, 1.0),
        )
        _, pairs = polygon_diameter(vertices)
        checks = check_diametral_circles(vertices, pairs)
        self.assertTrue(all(check.covers for check in checks))

    def test_randomized_consistent_observations_retain_true_source(self) -> None:
        rng = random.Random(20260910)
        stations = (
            Point(-1200.0, 0.0),
            Point(1200.0, 0.0),
            Point(0.0, -1200.0),
            Point(0.0, 1200.0),
        )
        for _ in range(50):
            source = Point(rng.uniform(-250.0, 250.0), rng.uniform(-250.0, 250.0))
            observations = []
            for station in stations:
                true_bearing = math.degrees(
                    math.atan2(source.y - station.y, source.x - station.x)
                ) % 360.0
                observations.append(
                    Observation(station, true_bearing + rng.uniform(-1.0, 1.0))
                )

            result = solve_question1(observations)
            self.assertEqual(result.intersection.status, "bounded")
            for observation in observations:
                self.assertTrue(
                    all(
                        plane.contains(source, 1e-6)
                        for plane in observation_halfplanes(observation)
                    )
                )
            circle = result.minimum_enclosing_circle
            self.assertIsNotNone(circle)
            assert circle is not None
            self.assertTrue(
                all(
                    circle.center.distance_to(vertex) <= circle.radius + 1e-5
                    for vertex in result.intersection.vertices
                )
            )


if __name__ == "__main__":
    unittest.main()

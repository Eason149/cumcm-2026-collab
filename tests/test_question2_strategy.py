"""Regression tests for the robust second-station strategy."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from question1_geometry import Observation, Point, polygon_diameter  # noqa: E402
from question2_strategy import (  # noqa: E402
    centerline_baseline,
    circumscribed_circle_polygon,
    conditional_reception_evaluation,
    first_source_region,
    posterior_polygon,
    sample_convex_polygon,
    search_posterior_optimal_candidates,
    search_second_station_candidates,
    worst_posterior_diameter,
)


def _inside_convex_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    signed_area = sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    )
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    return all(
        orientation
        * (
            (second.x - first.x) * (point.y - first.y)
            - (second.y - first.y) * (point.x - first.x)
        )
        >= -1e-6
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    )


class Question2StrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = Observation(Point(0.0, 0.0), 35.0)
        self.region = first_source_region(self.observation, circle_sides=180)

    def test_circle_approximation_is_outer_and_tight(self) -> None:
        radius = 1500.0
        polygon = circumscribed_circle_polygon(Point(0.0, 0.0), radius, 180)
        distances = [point.distance_to(Point(0.0, 0.0)) for point in polygon]
        self.assertTrue(all(distance >= radius for distance in distances))
        self.assertLess(max(distances) - radius, 0.23)

    def test_first_region_contains_physical_sources(self) -> None:
        for radial_distance in (5.01, 750.0, 1499.0):
            for bearing in (34.0, 35.0, 36.0):
                angle = math.radians(bearing)
                source = Point(
                    radial_distance * math.cos(angle),
                    radial_distance * math.sin(angle),
                )
                self.assertTrue(_inside_convex_polygon(source, self.region))

    def test_recommended_points_guarantee_reception_and_improve_angle(self) -> None:
        grid = search_second_station_candidates(
            self.observation,
            self.region,
            grid_size=61,
            edge_subdivisions=8,
            radial_levels=5,
        )
        samples = sample_convex_polygon(self.region, 8, 5)
        baseline = centerline_baseline(self.observation, self.region, samples)
        for candidate in (grid.positive_candidate, grid.negative_candidate):
            self.assertTrue(candidate.guaranteed_visible)
            self.assertLessEqual(candidate.maximum_source_distance_m, 1000.0 + 1e-6)
            self.assertGreater(candidate.minimum_intersection_angle_deg, 35.0)
        self.assertLess(baseline.minimum_intersection_angle_deg, 1e-6)

    def test_second_wedge_keeps_the_consistent_source(self) -> None:
        angle = math.radians(35.0)
        source = Point(900.0 * math.cos(angle), 900.0 * math.sin(angle))
        second_station = Point(252.4, 971.4)
        for error in (-1.0, 0.0, 1.0):
            posterior = posterior_polygon(
                self.region, second_station, source, measurement_error_deg=error
            )
            self.assertTrue(posterior)
            self.assertTrue(_inside_convex_polygon(source, posterior))

    def test_conditional_reception_accepts_valid_expanded_domain_point(self) -> None:
        angle = math.radians(35.0)
        along = (math.cos(angle), math.sin(angle))
        normal = (-math.sin(angle), math.cos(angle))
        candidate = Point(
            650.0 * along[0] + 750.0 * normal[0],
            650.0 * along[1] + 750.0 * normal[1],
        )
        audit = conditional_reception_evaluation(
            candidate,
            self.observation.station,
            self.region,
        )
        self.assertTrue(audit.guaranteed_visible)
        self.assertGreater(audit.minimum_margin_m, 9.0)
        for source in sample_convex_polygon(self.region, 40, 10):
            sampled_margin = max(
                1000.0, source.distance_to(self.observation.station)
            ) - source.distance_to(candidate)
            self.assertGreaterEqual(sampled_margin + 1e-6, audit.minimum_margin_m)

    def test_posterior_search_uses_loss_then_near_optimal_travel(self) -> None:
        result = search_posterior_optimal_candidates(
            self.observation,
            self.region,
            grid_size=21,
            source_edge_subdivisions=4,
            source_radial_levels=3,
            measurement_errors_deg=(-1.0, 0.0, 1.0),
        )
        self.assertTrue(result.optimum.guaranteed_visible)
        self.assertTrue(result.recommended.guaranteed_visible)
        self.assertLessEqual(
            result.recommended_posterior.worst_diameter_m,
            result.near_optimal_threshold_m + 1e-7,
        )
        self.assertLessEqual(
            result.recommended.travel_distance_m,
            result.optimum.travel_distance_m + 1e-7,
        )

    def test_robust_candidate_reduces_sampled_worst_posterior(self) -> None:
        grid = search_second_station_candidates(
            self.observation,
            self.region,
            grid_size=61,
            edge_subdivisions=8,
            radial_levels=5,
        )
        samples = sample_convex_polygon(self.region, 5, 4)
        baseline = centerline_baseline(self.observation, self.region, samples)
        selected_worst = worst_posterior_diameter(
            self.region,
            grid.positive_candidate.point,
            samples,
            measurement_errors_deg=(-1.0, 0.0, 1.0),
        )
        baseline_worst = worst_posterior_diameter(
            self.region,
            baseline.point,
            samples,
            measurement_errors_deg=(-1.0, 0.0, 1.0),
        )
        self.assertLess(selected_worst.worst_diameter_m, 0.3 * baseline_worst.worst_diameter_m)
        diameter, _ = polygon_diameter(selected_worst.worst_polygon)
        self.assertAlmostEqual(diameter, selected_worst.worst_diameter_m, places=7)


if __name__ == "__main__":
    unittest.main()

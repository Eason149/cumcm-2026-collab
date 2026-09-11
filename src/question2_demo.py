"""Reproduce the Question 2 candidate-region and posterior figures."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "cumcm2026-matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Polygon

from plot_style import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    save_publication_figure,
    style_axis,
)
from question1_geometry import Observation, Point, polygon_diameter
from question2_strategy import (
    CandidateEvaluation,
    centerline_baseline,
    first_source_region,
    representative_posterior,
    sample_convex_polygon,
    search_posterior_optimal_candidates,
    search_second_station_candidates,
    worst_posterior_diameter,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "tables" / "question2_demo.json"
CANDIDATE_FIGURE_PATH = ROOT / "results" / "figures" / "question2_candidate_region.png"
POSTERIOR_FIGURE_PATH = ROOT / "results" / "figures" / "question2_posterior_comparison.png"


def _candidate_payload(candidate: CandidateEvaluation) -> dict:
    return {
        "point": asdict(candidate.point),
        "guaranteed_visible": candidate.guaranteed_visible,
        "maximum_source_distance_m": candidate.maximum_source_distance_m,
        "minimum_intersection_angle_deg": candidate.minimum_intersection_angle_deg,
        "conditional_reception_margin_m": (
            candidate.reception_margin_m
            if np.isfinite(candidate.reception_margin_m)
            else None
        ),
        "travel_distance_m": candidate.travel_distance_m,
        "travel_time_s": candidate.travel_distance_m / 5.0,
    }


def _polygon_patch(vertices, facecolor, edgecolor, alpha, label, linewidth=1.8):
    return Polygon(
        [(point.x, point.y) for point in vertices],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )


def _local_basis(bearing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return along-bearing and left-normal unit vectors."""

    angle = np.deg2rad(bearing_deg)
    along = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    return along, normal


def _point_to_local(point: Point, origin: Point, along, normal) -> tuple[float, float]:
    displacement = np.array([point.x - origin.x, point.y - origin.y])
    return float(displacement @ along), float(displacement @ normal)


def _points_to_local(points, origin: Point, along, normal) -> np.ndarray:
    return np.asarray(
        [_point_to_local(point, origin, along, normal) for point in points],
        dtype=float,
    )


def _mirror_candidate(
    candidate: CandidateEvaluation,
    origin: Point,
    along: np.ndarray,
    normal: np.ndarray,
) -> CandidateEvaluation:
    """Reflect a candidate across the first measured bearing line."""

    local_along, local_lateral = _point_to_local(candidate.point, origin, along, normal)
    reflected = np.array([origin.x, origin.y]) + local_along * along - local_lateral * normal
    return CandidateEvaluation(
        point=Point(float(reflected[0]), float(reflected[1])),
        guaranteed_visible=candidate.guaranteed_visible,
        maximum_source_distance_m=candidate.maximum_source_distance_m,
        minimum_intersection_angle_deg=candidate.minimum_intersection_angle_deg,
        travel_distance_m=candidate.travel_distance_m,
        reception_margin_m=candidate.reception_margin_m,
    )


def main() -> None:
    apply_publication_style()
    first_observation = Observation(Point(0.0, 0.0), 35.0)
    first_region = first_source_region(first_observation, circle_sides=180)
    first_diameter, _ = polygon_diameter(first_region)
    source_samples = sample_convex_polygon(
        first_region, edge_subdivisions=10, radial_levels=5
    )

    # Retain the fixed-1000 m / angle-maximization solution only as a baseline.
    fixed_grid = search_second_station_candidates(
        first_observation,
        first_region,
        grid_size=61,
        edge_subdivisions=10,
        radial_levels=5,
    )
    fixed_positive = fixed_grid.positive_candidate
    fixed_negative = fixed_grid.negative_candidate
    baseline = centerline_baseline(
        first_observation, first_region, source_samples
    )

    convergence = []
    posterior_search = None
    for grid_size in (21, 41, 61):
        posterior_search = search_posterior_optimal_candidates(
            first_observation,
            first_region,
            grid_size=grid_size,
            source_edge_subdivisions=6,
            source_radial_levels=4,
            measurement_errors_deg=(-1.0, -0.5, 0.0, 0.5, 1.0),
            reception_safety_margin_m=0.5,
            near_optimal_fraction=0.10,
        )
        convergence.append(
            {
                "grid_size": grid_size,
                "evaluated_feasible_count": posterior_search.evaluated_count,
                "optimum_point": asdict(posterior_search.optimum.point),
                "optimum_sampled_worst_diameter_m": (
                    posterior_search.optimum_posterior.worst_diameter_m
                ),
                "recommended_point": asdict(posterior_search.recommended.point),
                "recommended_sampled_worst_diameter_m": (
                    posterior_search.recommended_posterior.worst_diameter_m
                ),
            }
        )
    assert posterior_search is not None
    optimum = posterior_search.optimum
    selected = posterior_search.recommended
    along, normal = _local_basis(first_observation.bearing_deg)
    optimum_mirror = _mirror_candidate(
        optimum, first_observation.station, along, normal
    )
    selected_mirror = _mirror_candidate(
        selected, first_observation.station, along, normal
    )
    selected_local = _point_to_local(
        selected.point, first_observation.station, along, normal
    )
    optimum_local = _point_to_local(
        optimum.point, first_observation.station, along, normal
    )

    posterior_samples = sample_convex_polygon(
        first_region, edge_subdivisions=20, radial_levels=7
    )
    posterior_error_grid = tuple(np.linspace(-1.0, 1.0, 21))
    selected_worst = worst_posterior_diameter(
        first_region,
        selected.point,
        posterior_samples,
        measurement_errors_deg=posterior_error_grid,
    )
    baseline_worst = worst_posterior_diameter(
        first_region,
        baseline.point,
        posterior_samples,
        measurement_errors_deg=posterior_error_grid,
    )

    (
        representative_source,
        representative_error,
        representative_polygon,
        representative_diameter,
        representative_circle,
    ) = representative_posterior(
        first_observation,
        first_region,
        selected.point,
        source_distance_m=900.0,
        measurement_errors_deg=tuple(np.linspace(-1.0, 1.0, 41)),
    )

    discretization_excess_1500_m = 1500.0 * (
        1.0 / np.cos(np.pi / 180.0) - 1.0
    )
    result = {
        "scenario": {
            "first_station": asdict(first_observation.station),
            "first_bearing_deg": first_observation.bearing_deg,
            "bearing_error_bound_deg": 1.0,
            "target_radius_m": 1800.0,
            "unknown_reception_radius_range_m": [1000.0, 1500.0],
            "circle_approximation_sides": 180,
            "maximum_radial_excess_for_1500_m_circle_m": discretization_excess_1500_m,
        },
        "first_region": {
            "vertex_count": len(first_region),
            "diameter_m": first_diameter,
            "vertices": [asdict(point) for point in first_region],
        },
        "candidate_rule": {
            "conditional_reception_rule": "d2(G) <= max(1000, d1(G))",
            "continuous_reception_audit": True,
            "reception_safety_margin_m": 0.5,
            "station_domain_rule": "S2 must lie in the radius-1800 m dog-motion disk when that operational boundary applies",
            "immediate_clearance_rule": "posterior minimum enclosing circle radius <= 20 m",
            "selection_order": "domain and reception feasibility; 20 m clearance test; posterior loss; travel distance",
            "primary_objective": "minimize sampled worst posterior diameter",
            "secondary_rule": "minimum travel distance within the 10% near-optimal set",
            "posterior_grid_sizes": [21, 41, 61],
            "posterior_source_sampling": {
                "edge_subdivisions": 6,
                "radial_levels": 4,
                "measurement_errors_deg": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
            "disclaimer": "posterior optimum is deterministic-grid approximate, not a continuous global bound",
        },
        "posterior_grid_optimum": {
            **_candidate_payload(optimum),
            "sampled_worst_diameter_m": posterior_search.optimum_posterior.worst_diameter_m,
            "sampled_worst_minimum_enclosing_radius_lower_bound_m": posterior_search.optimum_posterior.worst_diameter_m / 2.0,
            "guarantees_immediate_clearance_in_sampled_scenarios": posterior_search.optimum_posterior.worst_diameter_m <= 40.0,
            "relative_position": {
                "forward_along_first_bearing_m": optimum_local[0],
                "left_lateral_m": optimum_local[1],
                "absolute_offset_angle_from_first_bearing_deg": abs(float(np.degrees(np.arctan2(optimum_local[1], optimum_local[0])))),
            },
            "symmetric_alternative_point": asdict(optimum_mirror.point),
        },
        "recommended_near_optimal_candidate": {
            **_candidate_payload(selected),
            "sampled_worst_diameter_m": posterior_search.recommended_posterior.worst_diameter_m,
            "sampled_worst_minimum_enclosing_radius_lower_bound_m": posterior_search.recommended_posterior.worst_diameter_m / 2.0,
            "guarantees_immediate_clearance_in_sampled_scenarios": posterior_search.recommended_posterior.worst_diameter_m <= 40.0,
            "relative_position": {
                "forward_along_first_bearing_m": selected_local[0],
                "left_lateral_m": selected_local[1],
                "absolute_offset_angle_from_first_bearing_deg": abs(float(np.degrees(np.arctan2(selected_local[1], selected_local[0])))),
            },
            "near_optimal_threshold_m": posterior_search.near_optimal_threshold_m,
            "symmetric_alternative_point": asdict(selected_mirror.point),
        },
        "near_optimal_grid_point_count": int(np.count_nonzero(posterior_search.near_optimal_mask)),
        "fixed_1000m_angle_baseline": {
            "positive_candidate": _candidate_payload(fixed_positive),
            "negative_candidate": _candidate_payload(fixed_negative),
        },
        "centerline_baseline": _candidate_payload(baseline),
        "sampled_worst_posterior": {
            "source_sample_count": len(posterior_samples),
            "measurement_error_grid_count": len(posterior_error_grid),
            "selected_candidate_diameter_m": selected_worst.worst_diameter_m,
            "selected_worst_source": asdict(selected_worst.worst_source),
            "selected_worst_measurement_error_deg": selected_worst.worst_measurement_error_deg,
            "centerline_baseline_diameter_m": baseline_worst.worst_diameter_m,
            "relative_reduction": 1.0
            - selected_worst.worst_diameter_m / baseline_worst.worst_diameter_m,
        },
        "representative_source": {
            "point": asdict(representative_source),
            "distance_from_first_station_m": 900.0,
            "worst_tested_second_measurement_error_deg": representative_error,
            "posterior_diameter_m": representative_diameter,
            "posterior_minimum_enclosing_circle": {
                "center": asdict(representative_circle.center),
                "radius_m": representative_circle.radius,
            },
            "clearance_radius_m": 20.0,
            "clearance_radius_excess_m": representative_circle.radius - 20.0,
            "guarantees_immediate_clearance": representative_circle.radius <= 20.0,
        },
        "grid_convergence": convergence,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    direction, normal = along, normal
    grid_along, grid_lateral = np.meshgrid(
        posterior_search.along_values_m,
        posterior_search.lateral_values_m,
    )
    grid_x = (
        first_observation.station.x
        + grid_along * direction[0]
        + grid_lateral * normal[0]
    )
    grid_y = (
        first_observation.station.y
        + grid_along * direction[1]
        + grid_lateral * normal[1]
    )
    masked_scores = np.ma.masked_invalid(posterior_search.losses_m)
    candidate_cmap = LinearSegmentedColormap.from_list(
        "candidate_quality",
        [PALETTE["pale_yellow"], PALETTE["coral"], PALETTE["teal"]],
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), constrained_layout=True)
    axis = axes[0]
    axis.add_patch(
        CirclePatch(
            (0.0, 0.0),
            1800.0,
            facecolor=PALETTE["pale_yellow"],
            edgecolor=PALETTE["gold"],
            linewidth=1.5,
            alpha=0.25,
            label="Target disk",
        )
    )
    axis.add_patch(
        _polygon_patch(
            first_region,
            PALETTE["light_blue"],
            PALETTE["teal"],
            0.72,
            "First feasible region",
        )
    )
    ray_end = first_observation.station.x + 1600.0 * direction[0], first_observation.station.y + 1600.0 * direction[1]
    axis.plot(
        [first_observation.station.x, ray_end[0]],
        [first_observation.station.y, ray_end[1]],
        color=PALETTE["ink"],
        linestyle="--",
        linewidth=1.2,
        label="Measured center bearing",
    )
    axis.scatter(
        [first_observation.station.x],
        [first_observation.station.y],
        s=100,
        marker="o",
        color=PALETTE["red"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=5,
        label="First station",
    )
    axis.set_xlim(-1900, 1900)
    axis.set_ylim(-1900, 1900)
    axis.set_aspect("equal", adjustable="box")
    axis.margins(x=0.04, y=0.08)
    axis.set_xlabel("East x (m)")
    axis.set_ylabel("North y (m)")
    axis.set_title("First-observation uncertainty")
    style_axis(axis)
    axis.legend(loc="lower right", fontsize=8)
    add_panel_label(axis, "A")

    axis = axes[1]
    levels = np.linspace(float(masked_scores.min()), float(masked_scores.max()), 17)
    quality = axis.contourf(
        grid_x,
        grid_y,
        masked_scores,
        levels=levels,
        cmap=candidate_cmap,
        alpha=0.92,
        antialiased=True,
    )
    axis.contour(
        grid_x,
        grid_y,
        posterior_search.feasible_mask.astype(float),
        levels=[0.5],
        colors=[PALETTE["ink"]],
        linewidths=1.0,
    )
    axis.contour(
        grid_x,
        grid_y,
        posterior_search.near_optimal_mask.astype(float),
        levels=[0.5],
        colors=[PALETTE["red"]],
        linewidths=1.8,
    )
    axis.add_patch(
        _polygon_patch(
            first_region,
            PALETTE["light_blue"],
            PALETTE["teal"],
            0.28,
            "Possible source region",
            linewidth=1.2,
        )
    )
    axis.scatter(
        [
            optimum.point.x,
            optimum_mirror.point.x,
            selected.point.x,
            selected_mirror.point.x,
        ],
        [
            optimum.point.y,
            optimum_mirror.point.y,
            selected.point.y,
            selected_mirror.point.y,
        ],
        marker="*",
        s=190,
        color=PALETTE["red"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=6,
        label="Grid optimum / travel-selected",
    )
    axis.scatter(
        [baseline.point.x],
        [baseline.point.y],
        marker="D",
        s=60,
        color=PALETTE["lavender"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=6,
        label="Centerline baseline",
    )
    axis.scatter(
        [first_observation.station.x],
        [first_observation.station.y],
        s=75,
        color=PALETTE["coral"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=6,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("East x (m)")
    axis.set_ylabel("North y (m)")
    axis.set_title("Guaranteed-visible candidate region")
    style_axis(axis)
    axis.legend(loc="lower left", fontsize=8)
    add_panel_label(axis, "B")
    colorbar = figure.colorbar(quality, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Worst sampled posterior diameter (m)")
    figure.suptitle("Question 2 robust second-station selection", fontsize=15)
    save_publication_figure(figure, CANDIDATE_FIGURE_PATH)
    plt.close(figure)

    posterior_figure, posterior_axes = plt.subplots(
        1, 2, figsize=(12.2, 5.5), constrained_layout=True
    )
    axis = posterior_axes[0]
    axis.add_patch(
        _polygon_patch(
            first_region,
            PALETTE["light_blue"],
            PALETTE["teal"],
            0.62,
            f"First region (D={first_diameter:.1f} m)",
        )
    )
    axis.plot(
        [first_observation.station.x, representative_source.x],
        [first_observation.station.y, representative_source.y],
        color=PALETTE["gray"],
        linestyle="--",
        linewidth=1.0,
    )
    axis.plot(
        [selected.point.x, representative_source.x],
        [selected.point.y, representative_source.y],
        color=PALETTE["gray"],
        linestyle="--",
        linewidth=1.0,
    )
    axis.scatter(
        [first_observation.station.x, selected.point.x],
        [first_observation.station.y, selected.point.y],
        s=85,
        color=[PALETTE["coral"], PALETTE["red"]],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        label="Detection stations",
        zorder=5,
    )
    axis.scatter(
        [representative_source.x],
        [representative_source.y],
        marker="*",
        s=175,
        color=PALETTE["gold"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        label="Representative source",
        zorder=6,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("East x (m)")
    axis.set_ylabel("North y (m)")
    axis.set_title("Before the second observation")
    style_axis(axis)
    axis.legend(loc="upper left", fontsize=8)
    add_panel_label(axis, "A")

    axis = posterior_axes[1]
    axis.add_patch(
        _polygon_patch(
            representative_polygon,
            PALETTE["coral"],
            PALETTE["red"],
            0.62,
            f"Posterior region (D={representative_diameter:.1f} m)",
        )
    )
    axis.add_patch(
        CirclePatch(
            (representative_circle.center.x, representative_circle.center.y),
            representative_circle.radius,
            fill=False,
            edgecolor="#66A977",
            linestyle="--",
            linewidth=2.0,
            label=f"Minimum enclosing radius={representative_circle.radius:.1f} m",
        )
    )
    posterior_diameter, posterior_pairs = polygon_diameter(representative_polygon)
    diameter_pair = posterior_pairs[0]
    axis.plot(
        [diameter_pair[0].x, diameter_pair[1].x],
        [diameter_pair[0].y, diameter_pair[1].y],
        color="#8479C7",
        linewidth=2.0,
        label=f"Diameter={posterior_diameter:.1f} m",
    )
    axis.scatter(
        [representative_source.x],
        [representative_source.y],
        marker="*",
        s=175,
        color=PALETTE["gold"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=6,
        label="True source",
    )
    xs = [point.x for point in representative_polygon]
    ys = [point.y for point in representative_polygon]
    padding = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.3
    axis.set_xlim(min(xs) - padding, max(xs) + padding)
    axis.set_ylim(min(ys) - padding, max(ys) + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("East x (m)")
    axis.set_ylabel("North y (m)")
    axis.set_title(
        f"After second observation (tested error={representative_error:+.1f}°)"
    )
    style_axis(axis)
    axis.legend(loc="upper left", fontsize=8)
    add_panel_label(axis, "B")
    posterior_figure.suptitle("Uncertainty reduction by the second station", fontsize=15)
    save_publication_figure(posterior_figure, POSTERIOR_FIGURE_PATH)
    plt.close(posterior_figure)

    print(f"Wrote {RESULT_PATH}")
    print(f"Wrote {CANDIDATE_FIGURE_PATH}")
    print(f"Wrote {POSTERIOR_FIGURE_PATH}")


if __name__ == "__main__":
    main()

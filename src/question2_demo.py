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

from plot_style import PALETTE, add_panel_label, apply_publication_style
from question1_geometry import Observation, Point, polygon_diameter
from question2_strategy import (
    CandidateEvaluation,
    centerline_baseline,
    first_source_region,
    minimum_intersection_angle,
    refine_candidate,
    representative_posterior,
    sample_convex_polygon,
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


def main() -> None:
    apply_publication_style()
    operational_reception_m = 999.0
    first_observation = Observation(Point(0.0, 0.0), 35.0)
    first_region = first_source_region(first_observation, circle_sides=180)
    first_diameter, _ = polygon_diameter(first_region)
    source_samples = sample_convex_polygon(
        first_region, edge_subdivisions=10, radial_levels=5
    )

    convergence = []
    final_grid = None
    for grid_size in (61, 101, 141, 181):
        grid = search_second_station_candidates(
            first_observation,
            first_region,
            grid_size=grid_size,
            edge_subdivisions=10,
            radial_levels=5,
        )
        convergence.append(
            {
                "grid_size": grid_size,
                "positive_point": asdict(grid.positive_candidate.point),
                "positive_minimum_angle_deg": grid.positive_candidate.minimum_intersection_angle_deg,
                "negative_point": asdict(grid.negative_candidate.point),
                "negative_minimum_angle_deg": grid.negative_candidate.minimum_intersection_angle_deg,
            }
        )
        final_grid = grid
    assert final_grid is not None

    coarse_dx = float(final_grid.x_values[1] - final_grid.x_values[0])
    coarse_dy = float(final_grid.y_values[1] - final_grid.y_values[0])
    positive = refine_candidate(
        final_grid.positive_candidate,
        first_observation,
        first_region,
        side=1,
        x_halfwidth_m=2.0 * coarse_dx,
        y_halfwidth_m=2.0 * coarse_dy,
        grid_size=81,
        guaranteed_reception_m=operational_reception_m,
    )
    negative = refine_candidate(
        final_grid.negative_candidate,
        first_observation,
        first_region,
        side=-1,
        x_halfwidth_m=2.0 * coarse_dx,
        y_halfwidth_m=2.0 * coarse_dy,
        grid_size=81,
        guaranteed_reception_m=operational_reception_m,
    )
    local_dx = 4.0 * coarse_dx / 80.0
    local_dy = 4.0 * coarse_dy / 80.0
    positive = refine_candidate(
        positive,
        first_observation,
        first_region,
        side=1,
        x_halfwidth_m=2.0 * local_dx,
        y_halfwidth_m=2.0 * local_dy,
        grid_size=61,
        edge_subdivisions=60,
        radial_levels=11,
        guaranteed_reception_m=operational_reception_m,
    )
    negative = refine_candidate(
        negative,
        first_observation,
        first_region,
        side=-1,
        x_halfwidth_m=2.0 * local_dx,
        y_halfwidth_m=2.0 * local_dy,
        grid_size=61,
        edge_subdivisions=60,
        radial_levels=11,
        guaranteed_reception_m=operational_reception_m,
    )
    dense_audit_samples = sample_convex_polygon(
        first_region, edge_subdivisions=200, radial_levels=15
    )
    positive_dense_score = minimum_intersection_angle(
        positive.point, first_observation.station, dense_audit_samples
    )
    negative_dense_score = minimum_intersection_angle(
        negative.point, first_observation.station, dense_audit_samples
    )
    baseline = centerline_baseline(
        first_observation, first_region, source_samples
    )

    posterior_samples = sample_convex_polygon(
        first_region, edge_subdivisions=20, radial_levels=7
    )
    posterior_error_grid = tuple(np.linspace(-1.0, 1.0, 21))
    positive_worst = worst_posterior_diameter(
        first_region,
        positive.point,
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
        positive.point,
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
            "guaranteed_reception_distance_m": 1000.0,
            "operational_design_distance_m": operational_reception_m,
            "reception_safety_margin_m": 1000.0 - operational_reception_m,
            "primary_objective": "maximize sampled worst-case acute intersection angle",
            "tie_breaker": "minimize travel distance",
            "coarse_grid_size": 181,
            "local_refinement_grid_sizes": [81, 61],
            "dense_angle_audit_source_count": len(dense_audit_samples),
        },
        "positive_candidate": _candidate_payload(positive),
        "negative_candidate": _candidate_payload(negative),
        "dense_angle_audit": {
            "positive_minimum_angle_deg": positive_dense_score,
            "negative_minimum_angle_deg": negative_dense_score,
        },
        "centerline_baseline": _candidate_payload(baseline),
        "sampled_worst_posterior": {
            "source_sample_count": len(posterior_samples),
            "measurement_error_grid_count": len(posterior_error_grid),
            "selected_candidate_diameter_m": positive_worst.worst_diameter_m,
            "selected_worst_source": asdict(positive_worst.worst_source),
            "selected_worst_measurement_error_deg": positive_worst.worst_measurement_error_deg,
            "centerline_baseline_diameter_m": baseline_worst.worst_diameter_m,
            "relative_reduction": 1.0
            - positive_worst.worst_diameter_m / baseline_worst.worst_diameter_m,
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
        },
        "grid_convergence": convergence,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    direction_angle = np.deg2rad(first_observation.bearing_deg)
    direction = np.array([np.cos(direction_angle), np.sin(direction_angle)])
    grid_x, grid_y = np.meshgrid(final_grid.x_values, final_grid.y_values)
    masked_scores = np.ma.masked_invalid(final_grid.angle_scores_deg)
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
        final_grid.feasible_mask.astype(float),
        levels=[0.5],
        colors=[PALETTE["ink"]],
        linewidths=1.0,
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
        [positive.point.x, negative.point.x],
        [positive.point.y, negative.point.y],
        marker="*",
        s=190,
        color=PALETTE["red"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=6,
        label="Recommended candidates",
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
    axis.legend(loc="lower left", fontsize=8)
    add_panel_label(axis, "B")
    colorbar = figure.colorbar(quality, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Worst sampled intersection angle (degrees)")
    figure.suptitle("Question 2 robust second-station selection", fontsize=15)
    CANDIDATE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CANDIDATE_FIGURE_PATH, dpi=240)
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
        [positive.point.x, representative_source.x],
        [positive.point.y, representative_source.y],
        color=PALETTE["gray"],
        linestyle="--",
        linewidth=1.0,
    )
    axis.scatter(
        [first_observation.station.x, positive.point.x],
        [first_observation.station.y, positive.point.y],
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
    axis.legend(loc="upper left", fontsize=8)
    add_panel_label(axis, "B")
    posterior_figure.suptitle("Uncertainty reduction by the second station", fontsize=15)
    posterior_figure.savefig(POSTERIOR_FIGURE_PATH, dpi=240)
    plt.close(posterior_figure)

    print(f"Wrote {RESULT_PATH}")
    print(f"Wrote {CANDIDATE_FIGURE_PATH}")
    print(f"Wrote {POSTERIOR_FIGURE_PATH}")


if __name__ == "__main__":
    main()

"""Reproduce the Question 2 candidate-region and posterior figures."""

from __future__ import annotations

import csv
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
from matplotlib.colors import LogNorm
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
CONVERGENCE_FIGURE_PATH = ROOT / "results" / "figures" / "question2_convergence.png"
PARETO_FIGURE_PATH = ROOT / "results" / "figures" / "question2_pareto_front.png"
GRID_TABLE_PATH = ROOT / "results" / "tables" / "question2_candidate_grid.csv"
PARETO_TABLE_PATH = ROOT / "results" / "tables" / "question2_pareto_front.csv"
POSTERIOR_GRID_SIZES = (21, 41, 61, 81, 101)
GRID_CONVERGENCE_TOLERANCE = 0.001


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


def _posterior_payload(posterior) -> dict:
    return {
        "sampled_worst_diameter_m": posterior.worst_diameter_m,
        "clearance_guaranteed_on_grid": posterior.clearance_guaranteed,
        "clearance_radius_lower_bound_m": posterior.clearance_radius_lower_bound_m,
        "worst_exact_mec_radius_m": posterior.worst_exact_mec_radius_m,
        "exact_mec_evaluation_count": posterior.exact_mec_evaluation_count,
        "diameter_pruned_evaluation_count": posterior.diameter_pruned_evaluation_count,
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
    for grid_size in POSTERIOR_GRID_SIZES:
        posterior_search = search_posterior_optimal_candidates(
            first_observation,
            first_region,
            grid_size=grid_size,
            source_edge_subdivisions=6,
            source_radial_levels=4,
            measurement_errors_deg=(-1.0, -0.5, 0.0, 0.5, 1.0),
            reception_safety_margin_m=0.5,
            clearance_radius_m=20.0,
            station_domain_center=Point(0.0, 0.0),
            station_domain_radius_m=1800.0,
        )
        optimum_loss = posterior_search.optimum_posterior.worst_diameter_m
        previous_loss = (
            convergence[-1]["optimum_sampled_worst_diameter_m"]
            if convergence
            else None
        )
        relative_change = (
            abs(optimum_loss - previous_loss) / previous_loss
            if previous_loss is not None
            else None
        )
        convergence.append(
            {
                "grid_size": grid_size,
                "evaluated_feasible_count": posterior_search.evaluated_count,
                "optimum_point": asdict(posterior_search.optimum.point),
                "optimum_sampled_worst_diameter_m": optimum_loss,
                "relative_change_from_previous": relative_change,
                "converged": (
                    relative_change is not None
                    and relative_change < GRID_CONVERGENCE_TOLERANCE
                ),
                "recommended_point": asdict(posterior_search.recommended.point),
                "recommended_sampled_worst_diameter_m": (
                    posterior_search.recommended_posterior.worst_diameter_m
                ),
                "clearance_feasible_grid_count": int(
                    np.count_nonzero(posterior_search.clearance_feasible_mask)
                ),
                "selection_mode": posterior_search.selection_mode,
            }
        )
    assert posterior_search is not None
    assert convergence[-1]["converged"]
    optimum = posterior_search.optimum
    selected = posterior_search.recommended
    along, normal = _local_basis(first_observation.bearing_deg)
    optimum_mirror = _mirror_candidate(
        optimum, first_observation.station, along, normal
    )
    selected_mirror = _mirror_candidate(
        selected, first_observation.station, along, normal
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
            "objectives": ["minimize travel distance", "minimize sampled worst posterior diameter"],
            "first_priority": "guarantee one-step clearance when any grid point passes exact scenario classification",
            "secondary_rule": "maximum-distance-to-chord knee on the travel/log-loss Pareto frontier",
            "posterior_grid_sizes": list(POSTERIOR_GRID_SIZES),
            "grid_convergence_tolerance": GRID_CONVERGENCE_TOLERANCE,
            "grid_convergence_rule": "adjacent optimum relative change below 0.1%",
            "posterior_source_sampling": {
                "edge_subdivisions": 6,
                "radial_levels": 4,
                "measurement_errors_deg": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
            "disclaimer": "posterior optimum is deterministic-grid approximate, not a continuous global bound",
            "station_search_domain": "complete conditional-reception bounds intersected with the radius-1800 m activity disk",
            "clearance_rule": "diameter/2 lower-bound pruning followed by exact MEC when not pruned",
            "pareto_loss_scale": "natural logarithm",
        },
        "posterior_grid_optimum": {
            **_candidate_payload(optimum),
            **_posterior_payload(posterior_search.optimum_posterior),
            "symmetric_alternative_point": asdict(optimum_mirror.point),
        },
        "recommended_pareto_knee_candidate": {
            **_candidate_payload(selected),
            **_posterior_payload(posterior_search.recommended_posterior),
            "pareto_knee_score": posterior_search.pareto_knee_score,
            "symmetric_alternative_point": asdict(selected_mirror.point),
        },
        "pareto_front_grid_point_count": posterior_search.pareto_front_count,
        "clearance_feasible_grid_point_count": int(
            np.count_nonzero(posterior_search.clearance_feasible_mask)
        ),
        "selection_mode": posterior_search.selection_mode,
        "fixed_1000m_angle_baseline": {
            "positive_candidate": _candidate_payload(fixed_positive),
            "negative_candidate": _candidate_payload(fixed_negative),
        },
        "centerline_baseline": _candidate_payload(baseline),
        "sampled_worst_posterior": {
            "source_sample_count": len(posterior_samples),
            "measurement_error_grid_count": len(posterior_error_grid),
            "selected_candidate_diameter_m": selected_worst.worst_diameter_m,
            "selected_clearance_guaranteed_on_grid": selected_worst.clearance_guaranteed,
            "selected_clearance_radius_lower_bound_m": (
                selected_worst.clearance_radius_lower_bound_m
            ),
            "selected_exact_mec_evaluation_count": (
                selected_worst.exact_mec_evaluation_count
            ),
            "selected_diameter_pruned_evaluation_count": (
                selected_worst.diameter_pruned_evaluation_count
            ),
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
        },
        "grid_convergence": convergence,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    GRID_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRID_TABLE_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "along_m",
                "lateral_m",
                "x_m",
                "y_m",
                "conditionally_feasible",
                "worst_posterior_diameter_m",
                "pareto_optimal",
                "guaranteed_clearance",
            ]
        )
        for row, lateral in enumerate(posterior_search.lateral_values_m):
            for column, along_value in enumerate(posterior_search.along_values_m):
                x_value = (
                    first_observation.station.x
                    + along_value * along[0]
                    + lateral * normal[0]
                )
                y_value = (
                    first_observation.station.y
                    + along_value * along[1]
                    + lateral * normal[1]
                )
                loss = posterior_search.losses_m[row, column]
                writer.writerow(
                    [
                        float(along_value),
                        float(lateral),
                        float(x_value),
                        float(y_value),
                        bool(posterior_search.feasible_mask[row, column]),
                        float(loss) if np.isfinite(loss) else "",
                        bool(posterior_search.pareto_mask[row, column]),
                        bool(posterior_search.clearance_feasible_mask[row, column]),
                    ]
                )

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
    candidate_cmap = plt.get_cmap("cividis_r")

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
            label="目标与活动圆域",
        )
    )
    axis.add_patch(
        _polygon_patch(
            first_region,
            PALETTE["light_blue"],
            PALETTE["teal"],
            0.72,
            "首次观测可行域",
        )
    )
    ray_end = first_observation.station.x + 1600.0 * direction[0], first_observation.station.y + 1600.0 * direction[1]
    axis.plot(
        [first_observation.station.x, ray_end[0]],
        [first_observation.station.y, ray_end[1]],
        color=PALETTE["ink"],
        linestyle="--",
        linewidth=1.2,
        label="首次示向中心线",
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
        label="第一次检测点",
    )
    axis.set_xlim(-1900, 1900)
    axis.set_ylim(-1900, 1900)
    axis.set_aspect("equal", adjustable="box")
    axis.margins(x=0.04, y=0.08)
    axis.set_xlabel("东向坐标 x（m）")
    axis.set_ylabel("北向坐标 y（m）")
    axis.set_title("首次观测后的源位置不确定域")
    style_axis(axis)
    axis.legend(loc="lower right", fontsize=8)
    add_panel_label(axis, "A")

    axis = axes[1]
    score_min = float(masked_scores.min())
    score_max = float(masked_scores.max())
    levels = np.geomspace(score_min, score_max, 17)
    quality = axis.contourf(
        grid_x,
        grid_y,
        masked_scores,
        levels=levels,
        norm=LogNorm(vmin=score_min, vmax=score_max),
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
    pareto_rows, pareto_columns = np.nonzero(posterior_search.pareto_mask)
    axis.scatter(
        grid_x[pareto_rows, pareto_columns],
        grid_y[pareto_rows, pareto_columns],
        s=18,
        facecolors="none",
        edgecolors=PALETTE["orange"],
        linewidths=0.8,
        zorder=5,
        label="Pareto前沿候选点",
    )
    axis.add_patch(
        CirclePatch(
            (0.0, 0.0),
            1800.0,
            fill=False,
            edgecolor=PALETTE["gray"],
            linewidth=1.0,
            linestyle="--",
            label="活动圆域边界",
        )
    )
    axis.add_patch(
        _polygon_patch(
            first_region,
            PALETTE["light_blue"],
            PALETTE["teal"],
            0.28,
            "可能源区域",
            linewidth=1.2,
        )
    )
    axis.scatter(
        [optimum.point.x, optimum_mirror.point.x],
        [optimum.point.y, optimum_mirror.point.y],
        marker="*",
        s=190,
        color=PALETTE["red"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=6,
        label="网格最小后验直径点",
    )
    axis.scatter(
        [selected.point.x, selected_mirror.point.x],
        [selected.point.y, selected_mirror.point.y],
        marker="P",
        s=115,
        color=PALETTE["cyan"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=7,
        label="Pareto膝点推荐方案",
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
        label="中心线基准点",
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
    axis.set_xlabel("东向坐标 x（m）")
    axis.set_ylabel("北向坐标 y（m）")
    axis.set_title("条件接收可行域与后验直径")
    style_axis(axis)
    axis.legend(loc="lower left", fontsize=8)
    add_panel_label(axis, "B")
    colorbar = figure.colorbar(quality, ax=axis, fraction=0.046, pad=0.03)
    colorbar_ticks = [score_min, 150.0, 250.0, 500.0, 1000.0, score_max]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([f"{value:.0f}" for value in colorbar_ticks])
    colorbar.set_label("最坏抽样后验直径（对数色阶，m）")
    figure.suptitle("问题二 第二检测点的条件鲁棒选址", fontsize=15)
    save_publication_figure(figure, CANDIDATE_FIGURE_PATH)
    plt.close(figure)

    feasible_rows, feasible_columns = np.nonzero(posterior_search.feasible_mask)
    feasible_distances = np.hypot(
        grid_x[feasible_rows, feasible_columns] - first_observation.station.x,
        grid_y[feasible_rows, feasible_columns] - first_observation.station.y,
    )
    feasible_losses = posterior_search.losses_m[feasible_rows, feasible_columns]
    pareto_distances = np.hypot(
        grid_x[pareto_rows, pareto_columns] - first_observation.station.x,
        grid_y[pareto_rows, pareto_columns] - first_observation.station.y,
    )
    pareto_losses = posterior_search.losses_m[pareto_rows, pareto_columns]
    pareto_order = np.argsort(pareto_distances)
    pareto_distances = pareto_distances[pareto_order]
    pareto_losses = pareto_losses[pareto_order]
    pareto_rows = pareto_rows[pareto_order]
    pareto_columns = pareto_columns[pareto_order]
    normalized_distance = (
        (pareto_distances - pareto_distances.min()) / np.ptp(pareto_distances)
    )
    pareto_log_losses = np.log(pareto_losses)
    normalized_log_loss = (
        (pareto_log_losses - pareto_log_losses.min()) / np.ptp(pareto_log_losses)
    )
    knee_scores = 1.0 - normalized_distance - normalized_log_loss

    with PARETO_TABLE_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "travel_distance_m",
                "worst_posterior_diameter_m",
                "normalized_distance",
                "normalized_log_loss",
                "knee_score",
                "selected_knee",
                "x_m",
                "y_m",
            ]
        )
        for position, (row, column) in enumerate(zip(pareto_rows, pareto_columns)):
            writer.writerow(
                [
                    float(pareto_distances[position]),
                    float(pareto_losses[position]),
                    float(normalized_distance[position]),
                    float(normalized_log_loss[position]),
                    float(knee_scores[position]),
                    bool(
                        abs(pareto_distances[position] - selected.travel_distance_m) < 1e-7
                        and abs(pareto_losses[position] - posterior_search.recommended_posterior.worst_diameter_m) < 1e-7
                    ),
                    float(grid_x[row, column]),
                    float(grid_y[row, column]),
                ]
            )

    pareto_figure, pareto_axis = plt.subplots(
        figsize=(7.3, 5.2), constrained_layout=True
    )
    pareto_axis.scatter(
        feasible_distances,
        feasible_losses,
        s=13,
        color=PALETTE["light_blue"],
        alpha=0.42,
        edgecolors="none",
        label="可行候选点",
    )
    pareto_axis.plot(
        pareto_distances,
        pareto_losses,
        color=PALETTE["teal"],
        linewidth=2.0,
        marker="o",
        markersize=2.5,
        label="Pareto前沿",
    )
    shortest_position = int(np.argmin(pareto_distances))
    minimum_loss_position = int(np.argmin(pareto_losses))
    pareto_axis.scatter(
        [pareto_distances[shortest_position]],
        [pareto_losses[shortest_position]],
        marker="s",
        s=80,
        color="#66A977",
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=5,
        label="最短移动方案",
    )
    pareto_axis.scatter(
        [pareto_distances[minimum_loss_position]],
        [pareto_losses[minimum_loss_position]],
        marker="D",
        s=76,
        color=PALETTE["red"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=5,
        label="最小定位损失方案",
    )
    pareto_axis.scatter(
        [selected.travel_distance_m],
        [posterior_search.recommended_posterior.worst_diameter_m],
        marker="*",
        s=190,
        color=PALETTE["orange"],
        edgecolor=PALETTE["ink"],
        linewidth=0.8,
        zorder=6,
        label="Pareto膝点",
    )
    pareto_axis.annotate(
        "膝点",
        (
            selected.travel_distance_m,
            posterior_search.recommended_posterior.worst_diameter_m,
        ),
        xytext=(12, 13),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": PALETTE["ink"], "lw": 0.8},
        fontsize=9,
    )
    pareto_axis.set_yscale("log")
    pareto_axis.set_xlabel("移动距离（m）")
    pareto_axis.set_ylabel("最坏抽样后验直径（m，对数刻度）")
    pareto_axis.set_title("移动代价与定位损失的 Pareto 前沿")
    style_axis(pareto_axis)
    pareto_axis.legend(loc="upper right", fontsize=8)
    save_publication_figure(pareto_figure, PARETO_FIGURE_PATH)
    plt.close(pareto_figure)

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
            f"首次可行域（D={first_diameter:.1f} m）",
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
        label="两个检测点",
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
        label="代表性真实源",
        zorder=6,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("东向坐标 x（m）")
    axis.set_ylabel("北向坐标 y（m）")
    axis.set_title("第二次观测前")
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
            f"后验区域（D={representative_diameter:.1f} m）",
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
            label=f"最小包围圆半径={representative_circle.radius:.1f} m",
        )
    )
    posterior_diameter, posterior_pairs = polygon_diameter(representative_polygon)
    diameter_pair = posterior_pairs[0]
    axis.plot(
        [diameter_pair[0].x, diameter_pair[1].x],
        [diameter_pair[0].y, diameter_pair[1].y],
        color="#8479C7",
        linewidth=2.0,
        label=f"区域直径={posterior_diameter:.1f} m",
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
        label="真实源",
    )
    xs = [point.x for point in representative_polygon]
    ys = [point.y for point in representative_polygon]
    padding = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.3
    axis.set_xlim(min(xs) - padding, max(xs) + padding)
    axis.set_ylim(min(ys) - padding, max(ys) + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("东向坐标 x（m）")
    axis.set_ylabel("北向坐标 y（m）")
    axis.set_title(
        f"第二次观测后（测试误差={representative_error:+.1f}°）"
    )
    style_axis(axis)
    axis.legend(loc="upper left", fontsize=8)
    add_panel_label(axis, "B")
    posterior_figure.suptitle("第二检测点对定位不确定域的压缩", fontsize=15)
    save_publication_figure(posterior_figure, POSTERIOR_FIGURE_PATH)
    plt.close(posterior_figure)

    convergence_figure, convergence_axes = plt.subplots(
        2,
        1,
        figsize=(6.7, 6.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    loss_axis, change_axis = convergence_axes
    grid_sizes = np.asarray([item["grid_size"] for item in convergence], dtype=float)
    optimum_losses = np.asarray(
        [item["optimum_sampled_worst_diameter_m"] for item in convergence],
        dtype=float,
    )
    recommended_losses = np.asarray(
        [item["recommended_sampled_worst_diameter_m"] for item in convergence],
        dtype=float,
    )
    loss_axis.plot(
        grid_sizes,
        optimum_losses,
        color=PALETTE["teal"],
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        label="网格最小后验直径",
    )
    loss_axis.plot(
        grid_sizes,
        recommended_losses,
        color=PALETTE["orange"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=5.0,
        label="Pareto膝点",
    )
    for x_value, y_value in zip(grid_sizes, optimum_losses):
        loss_axis.annotate(
            f"{y_value:.1f}",
            (x_value, y_value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    loss_axis.set_ylabel("最坏抽样后验直径（m）")
    loss_axis.set_title("候选网格加密与目标值稳定性")
    loss_axis.margins(y=0.18)
    style_axis(loss_axis)
    loss_axis.legend(loc="upper right")
    add_panel_label(loss_axis, "A")

    relative_changes_percent = np.asarray(
        [
            np.nan
            if item["relative_change_from_previous"] is None
            else 100.0 * item["relative_change_from_previous"]
            for item in convergence
        ],
        dtype=float,
    )
    change_axis.plot(
        grid_sizes[1:],
        relative_changes_percent[1:],
        color=PALETTE["blue"],
        marker="D",
        linewidth=1.8,
        markersize=5.0,
        label="相邻网格最优值变化",
    )
    change_axis.axhline(
        100.0 * GRID_CONVERGENCE_TOLERANCE,
        color=PALETTE["red"],
        linestyle=":",
        linewidth=1.5,
        label="0.1%收敛判据",
    )
    for x_value, y_value in zip(grid_sizes[1:], relative_changes_percent[1:]):
        label = f"{y_value:.4f}%" if y_value < 0.01 else f"{y_value:.3f}%"
        change_axis.annotate(
            label,
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    change_axis.set_yscale("log")
    change_axis.set_xlabel("候选网格阶数")
    change_axis.set_ylabel("相邻最优值相对变化（%）")
    change_axis.set_xticks(grid_sizes)
    style_axis(change_axis)
    change_axis.legend(loc="lower left")
    add_panel_label(change_axis, "B")
    save_publication_figure(convergence_figure, CONVERGENCE_FIGURE_PATH)
    plt.close(convergence_figure)

    print(f"Wrote {RESULT_PATH}")
    print(f"Wrote {GRID_TABLE_PATH}")
    print(f"Wrote {CANDIDATE_FIGURE_PATH}")
    print(f"Wrote {POSTERIOR_FIGURE_PATH}")
    print(f"Wrote {CONVERGENCE_FIGURE_PATH}")
    print(f"Wrote {PARETO_FIGURE_PATH}")
    print(f"Wrote {PARETO_TABLE_PATH}")


if __name__ == "__main__":
    main()

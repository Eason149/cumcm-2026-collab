"""Independent and scenario-based validation for the Question 2 strategy."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict
from math import cos, hypot, pi, sin
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cumcm-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from plot_style import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    save_publication_figure,
    style_axis,
)
from question1_geometry import Observation, Point, polygon_diameter
from question2_strategy import (
    conditional_reception_evaluation,
    first_source_region,
    sample_convex_polygon,
    search_posterior_optimal_candidates,
    worst_posterior_diameter,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_RESULT_PATH = ROOT / "results" / "tables" / "question2_demo.json"
VALIDATION_RESULT_PATH = ROOT / "results" / "tables" / "question2_validation.json"
VALIDATION_TABLE_PATH = ROOT / "results" / "tables" / "question2_validation.csv"
VALIDATION_FIGURE_PATH = ROOT / "results" / "figures" / "question2_validation.png"
PARETO_TABLE_PATH = ROOT / "results" / "tables" / "question2_pareto_front.csv"


def _continuous_objective_data():
    observation = Observation(Point(0.0, 0.0), 35.0)
    region = first_source_region(observation, circle_sides=180)
    samples = sample_convex_polygon(region, edge_subdivisions=6, radial_levels=4)
    errors = (-1.0, -0.5, 0.0, 0.5, 1.0)
    anchor = min(region, key=lambda point: max(1000.0, point.distance_to(observation.station)))
    radius = max(1000.0, anchor.distance_to(observation.station))
    bounds = [(anchor.x - radius, anchor.x + radius), (anchor.y - radius, anchor.y + radius)]
    return observation, region, samples, errors, bounds


def _continuous_optimizer_audit() -> dict:
    observation, region, samples, errors, bounds = _continuous_objective_data()
    cache: dict[tuple[float, float], tuple[float, float]] = {}

    def objective(values) -> float:
        key = (round(float(values[0]), 7), round(float(values[1]), 7))
        if key not in cache:
            point = Point(*key)
            reception = conditional_reception_evaluation(
                point, observation.station, region, safety_margin_m=0.5
            )
            if reception.guaranteed_visible:
                loss = worst_posterior_diameter(
                    region, point, samples, measurement_errors_deg=errors
                ).worst_diameter_m
            else:
                loss = 10000.0 + 1000.0 * (0.5 - reception.minimum_margin_m)
            cache[key] = (loss, reception.minimum_margin_m)
        return cache[key][0]

    runs = []
    for seed in (2026, 149):
        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            popsize=10,
            maxiter=60,
            tol=5e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        point = Point(float(result.x[0]), float(result.x[1]))
        reception = conditional_reception_evaluation(
            point, observation.station, region, safety_margin_m=0.5
        )
        runs.append(
            {
                "seed": seed,
                "point": asdict(point),
                "sampled_worst_diameter_m": float(result.fun),
                "reception_margin_m": reception.minimum_margin_m,
                "function_evaluations": int(result.nfev),
            }
        )
    best = min(runs, key=lambda item: item["sampled_worst_diameter_m"])
    dense_samples = sample_convex_polygon(region, edge_subdivisions=20, radial_levels=7)
    dense_errors = tuple(float(value) for value in np.linspace(-1.0, 1.0, 21))
    dense = worst_posterior_diameter(
        region, Point(**best["point"]), dense_samples, measurement_errors_deg=dense_errors
    )
    best["dense_worst_diameter_m"] = dense.worst_diameter_m
    best["dense_scenario_count"] = len(dense_samples) * len(dense_errors)
    spread = max(item["sampled_worst_diameter_m"] for item in runs) - min(
        item["sampled_worst_diameter_m"] for item in runs
    )
    return {
        "method": "differential_evolution",
        "runs": runs,
        "best": best,
        "two_seed_objective_spread_m": spread,
    }


def _pareto_knee(records: list[dict]) -> tuple[list[dict], dict]:
    """Extract nondominated records and select the log-loss chord knee."""

    ordered = sorted(records, key=lambda item: (item["distance_m"], item["loss_m"]))
    frontier = []
    best_loss = float("inf")
    for item in ordered:
        if item["loss_m"] < best_loss - 1e-7:
            frontier.append(item)
            best_loss = item["loss_m"]
    distances = np.asarray([item["distance_m"] for item in frontier], dtype=float)
    log_losses = np.log([item["loss_m"] for item in frontier])
    normalized_distance = (distances - distances.min()) / np.ptp(distances)
    normalized_log_loss = (log_losses - log_losses.min()) / np.ptp(log_losses)
    scores = 1.0 - normalized_distance - normalized_log_loss
    for item, score in zip(frontier, scores):
        item["knee_score"] = float(score)
    return frontier, frontier[int(np.argmax(scores))]


def _continuous_pareto_audit(demo: dict) -> dict:
    """Independently approximate the Pareto front with epsilon constraints."""

    observation, region, samples, errors, _ = _continuous_objective_data()
    maximum_budget = demo["posterior_grid_optimum"]["travel_distance_m"]

    def solve_budget(
        budget_m: float,
        seed: int,
        maxiter: int = 18,
        popsize: int = 5,
    ) -> dict:
        if budget_m <= 1e-12:
            point = observation.station
            loss = worst_posterior_diameter(
                region, point, samples, measurement_errors_deg=errors
            ).worst_diameter_m
            reception = conditional_reception_evaluation(
                point, observation.station, region, safety_margin_m=0.5
            )
            return {
                "movement_budget_m": 0.0,
                "seed": seed,
                "point": asdict(point),
                "distance_m": 0.0,
                "loss_m": loss,
                "reception_margin_m": reception.minimum_margin_m,
                "function_evaluations": 1,
            }

        cache: dict[tuple[float, float], tuple[float, float]] = {}

        def objective(values) -> float:
            radius = budget_m * float(values[0])
            angle = -pi + 2.0 * pi * float(values[1])
            point = Point(radius * cos(angle), radius * sin(angle))
            key = (round(point.x, 6), round(point.y, 6))
            if key not in cache:
                rounded_point = Point(*key)
                reception = conditional_reception_evaluation(
                    rounded_point,
                    observation.station,
                    region,
                    safety_margin_m=0.5,
                )
                if reception.guaranteed_visible:
                    loss = worst_posterior_diameter(
                        region,
                        rounded_point,
                        samples,
                        measurement_errors_deg=errors,
                    ).worst_diameter_m
                else:
                    loss = 10000.0 + 1000.0 * max(0.0, -reception.minimum_margin_m)
                cache[key] = (loss, reception.minimum_margin_m)
            return cache[key][0]

        result = differential_evolution(
            objective,
            [(0.0, 1.0), (0.0, 1.0)],
            seed=seed,
            popsize=popsize,
            maxiter=maxiter,
            tol=2e-3,
            polish=False,
            workers=1,
            updating="immediate",
        )
        radius = budget_m * float(result.x[0])
        angle = -pi + 2.0 * pi * float(result.x[1])
        point = Point(radius * cos(angle), radius * sin(angle))
        reception = conditional_reception_evaluation(
            point, observation.station, region, safety_margin_m=0.5
        )
        return {
            "movement_budget_m": budget_m,
            "seed": seed,
            "point": asdict(point),
            "distance_m": radius,
            "loss_m": float(result.fun),
            "reception_margin_m": reception.minimum_margin_m,
            "function_evaluations": int(result.nfev),
        }

    coarse_budgets = np.linspace(0.0, maximum_budget, 13)
    records = [
        solve_budget(float(budget), 6100 + index)
        for index, budget in enumerate(coarse_budgets)
    ]
    _, coarse_knee = _pareto_knee(records)
    knee_budget = coarse_knee["movement_budget_m"]
    knee_index = int(np.argmin(np.abs(coarse_budgets - knee_budget)))
    left = coarse_budgets[max(0, knee_index - 1)]
    right = coarse_budgets[min(len(coarse_budgets) - 1, knee_index + 1)]
    adaptive_budgets = [
        float(value)
        for value in np.linspace(left, right, 9)[1:-1]
        if all(abs(value - old) > 1e-7 for old in coarse_budgets)
    ]
    records.extend(
        solve_budget(budget, 7100 + index)
        for index, budget in enumerate(adaptive_budgets)
    )
    frontier, knee = _pareto_knee(records)

    refined = [
        solve_budget(knee["movement_budget_m"], seed, maxiter=45, popsize=8)
        for seed in (2026, 149)
    ]
    best_refined = min(refined, key=lambda item: item["loss_m"])
    records = [
        item
        for item in records
        if abs(item["movement_budget_m"] - knee["movement_budget_m"]) > 1e-7
    ]
    records.append(best_refined)
    frontier, knee = _pareto_knee(records)

    grid = demo["recommended_pareto_knee_candidate"]
    grid_points = [grid["point"], grid["symmetric_alternative_point"]]
    coordinate_gap = min(
        hypot(knee["point"]["x"] - point["x"], knee["point"]["y"] - point["y"])
        for point in grid_points
    )
    return {
        "method": "epsilon-constrained differential evolution in continuous polar coordinates",
        "coarse_budget_count": len(coarse_budgets),
        "adaptive_budget_count": len(adaptive_budgets),
        "frontier": frontier,
        "continuous_knee": knee,
        "grid_knee": {
            "point": grid["point"],
            "symmetric_alternative_point": grid["symmetric_alternative_point"],
            "distance_m": grid["travel_distance_m"],
            "loss_m": grid["sampled_worst_diameter_m"],
        },
        "coordinate_gap_to_nearest_symmetric_grid_knee_m": coordinate_gap,
        "relative_distance_gap": abs(knee["distance_m"] - grid["travel_distance_m"])
        / grid["travel_distance_m"],
        "relative_loss_gap": abs(knee["loss_m"] - grid["sampled_worst_diameter_m"])
        / grid["sampled_worst_diameter_m"],
        "refined_seed_losses_m": [item["loss_m"] for item in refined],
    }


def _sampling_convergence(demo: dict) -> list[dict]:
    """Audit source-position and bearing-error discretization at the final points."""

    observation = Observation(Point(0.0, 0.0), 35.0)
    region = first_source_region(observation, circle_sides=180)
    optimum_point = Point(**demo["posterior_grid_optimum"]["point"])
    recommended_point = Point(**demo["recommended_pareto_knee_candidate"]["point"])
    levels = ((6, 4, 5), (10, 5, 11), (20, 7, 21))
    output = []
    previous_optimum = None
    previous_recommended = None
    for edge_subdivisions, radial_levels, error_count in levels:
        samples = sample_convex_polygon(
            region,
            edge_subdivisions=edge_subdivisions,
            radial_levels=radial_levels,
        )
        errors = tuple(float(value) for value in np.linspace(-1.0, 1.0, error_count))
        optimum_loss = worst_posterior_diameter(
            region,
            optimum_point,
            samples,
            measurement_errors_deg=errors,
        ).worst_diameter_m
        recommended_loss = worst_posterior_diameter(
            region,
            recommended_point,
            samples,
            measurement_errors_deg=errors,
        ).worst_diameter_m
        output.append(
            {
                "edge_subdivisions": edge_subdivisions,
                "radial_levels": radial_levels,
                "source_sample_count": len(samples),
                "measurement_error_grid_count": error_count,
                "scenario_count_per_candidate": len(samples) * error_count,
                "optimum_worst_diameter_m": optimum_loss,
                "recommended_worst_diameter_m": recommended_loss,
                "optimum_relative_change_from_previous": (
                    None
                    if previous_optimum is None
                    else abs(optimum_loss - previous_optimum) / previous_optimum
                ),
                "recommended_relative_change_from_previous": (
                    None
                    if previous_recommended is None
                    else abs(recommended_loss - previous_recommended) / previous_recommended
                ),
            }
        )
        previous_optimum = optimum_loss
        previous_recommended = recommended_loss
    return output


def _scenario_audits(demo: dict) -> list[dict]:
    central_grid = demo["grid_convergence"][1]
    scenarios = [
        {
            "name": "中心长角域",
            "station": {"x": 0.0, "y": 0.0},
            "bearing_deg": 35.0,
            "first_region_diameter_m": demo["first_region"]["diameter_m"],
            "grid_optimum_diameter_m": central_grid["optimum_sampled_worst_diameter_m"],
            "recommended_point": central_grid["recommended_point"],
            "recommended_distance_m": hypot(
                central_grid["recommended_point"]["x"], central_grid["recommended_point"]["y"]
            ),
            "clearance_guaranteed": False,
        }
    ]
    for name, station, bearing in (
        ("边界向外", Point(1600.0, 0.0), 0.0),
        ("边界斜切", Point(1600.0, 0.0), 80.0),
    ):
        observation = Observation(station, bearing)
        region = first_source_region(observation, circle_sides=180)
        region_diameter, _ = polygon_diameter(region)
        search = search_posterior_optimal_candidates(
            observation,
            region,
            grid_size=41,
            source_edge_subdivisions=6,
            source_radial_levels=4,
            measurement_errors_deg=(-1.0, -0.5, 0.0, 0.5, 1.0),
            station_domain_center=None,
            station_domain_radius_m=None,
        )
        scenarios.append(
            {
                "name": name,
                "station": asdict(station),
                "bearing_deg": bearing,
                "first_region_diameter_m": region_diameter,
                "grid_optimum_diameter_m": search.optimum_posterior.worst_diameter_m,
                "recommended_point": asdict(search.recommended.point),
                "recommended_distance_m": search.recommended.travel_distance_m,
                "clearance_guaranteed": search.recommended_posterior.clearance_guaranteed,
            }
        )
    return scenarios


def _write_validation_csv(scenarios: list[dict], continuous_pareto: dict) -> None:
    VALIDATION_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATION_TABLE_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "case", "x", "y", "value_1", "value_2", "flag"])
        for item in scenarios:
            writer.writerow(
                [
                    "scenario",
                    item["name"],
                    item["station"]["x"],
                    item["station"]["y"],
                    item["first_region_diameter_m"],
                    item["grid_optimum_diameter_m"],
                    item["clearance_guaranteed"],
                ]
            )
        for item in continuous_pareto["frontier"]:
            writer.writerow(
                [
                    "continuous_pareto",
                    f"budget_{item['movement_budget_m']:.3f}",
                    item["point"]["x"],
                    item["point"]["y"],
                    item["distance_m"],
                    item["loss_m"],
                    item is continuous_pareto["continuous_knee"],
                ]
            )


def _plot_validation(
    demo: dict,
    continuous: dict,
    continuous_pareto: dict,
    scenarios: list[dict],
) -> None:
    apply_publication_style()
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), constrained_layout=True)
    grid_sizes = np.asarray([row["grid_size"] for row in demo["grid_convergence"]])
    grid_losses = np.asarray(
        [row["optimum_sampled_worst_diameter_m"] for row in demo["grid_convergence"]]
    )
    axes[0].plot(
        grid_sizes, grid_losses, marker="o", color=PALETTE["teal"], linewidth=2.0,
        label="确定性网格",
    )
    axes[0].axhline(
        continuous["best"]["sampled_worst_diameter_m"], color=PALETTE["orange"],
        linestyle="--", linewidth=1.8, label="差分进化连续坐标复核",
    )
    axes[0].set_xlabel("候选网格阶数")
    axes[0].set_ylabel("最坏后验直径（m）")
    axes[0].set_title("网格解与独立优化复核")
    axes[0].set_xticks(grid_sizes)
    axes[0].set_ylim(109.5, 118.0)
    style_axis(axes[0])
    axes[0].legend(loc="upper right")
    add_panel_label(axes[0], "A")

    labels = [item["name"] for item in scenarios]
    values = [item["grid_optimum_diameter_m"] for item in scenarios]
    for index, item in enumerate(scenarios):
        marker = "o" if item["clearance_guaranteed"] else "X"
        color = PALETTE["teal"] if item["clearance_guaranteed"] else PALETTE["red"]
        axes[1].scatter(index, values[index], s=85, marker=marker, color=color, zorder=3)
        axes[1].annotate(
            f"{values[index]:.1f}", (index, values[index]), xytext=(0, 7),
            textcoords="offset points", ha="center", fontsize=8,
        )
    axes[1].axhline(
        40.0, color=PALETTE["gray"], linestyle=":", linewidth=1.5,
        label="清除必要直径阈值 40 m",
    )
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_ylabel("41阶网格最小后验直径（m）")
    axes[1].set_title("不同首次观测构型")
    axes[1].set_ylim(0.0, max(values) * 1.18)
    style_axis(axes[1])
    axes[1].legend(loc="center right")
    add_panel_label(axes[1], "B")

    with PARETO_TABLE_PATH.open(encoding="utf-8-sig", newline="") as stream:
        grid_frontier = list(csv.DictReader(stream))
    grid_distances = np.asarray(
        [float(item["travel_distance_m"]) for item in grid_frontier]
    )
    grid_losses = np.asarray(
        [float(item["worst_posterior_diameter_m"]) for item in grid_frontier]
    )
    continuous_frontier = continuous_pareto["frontier"]
    continuous_distances = [item["distance_m"] for item in continuous_frontier]
    continuous_losses = [item["loss_m"] for item in continuous_frontier]
    axes[2].plot(
        grid_distances,
        grid_losses,
        color=PALETTE["teal"],
        linewidth=2.0,
        label="确定性网格前沿",
    )
    axes[2].scatter(
        continuous_distances,
        continuous_losses,
        s=34,
        marker="o",
        facecolor="white",
        edgecolor=PALETTE["orange"],
        linewidth=1.3,
        zorder=4,
        label="连续ε约束复核",
    )
    grid_knee = continuous_pareto["grid_knee"]
    continuous_knee = continuous_pareto["continuous_knee"]
    axes[2].scatter(
        [grid_knee["distance_m"]],
        [grid_knee["loss_m"]],
        marker="P",
        s=90,
        color=PALETTE["teal"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=5,
        label="网格膝点",
    )
    axes[2].scatter(
        [continuous_knee["distance_m"]],
        [continuous_knee["loss_m"]],
        marker="*",
        s=140,
        color=PALETTE["orange"],
        edgecolor=PALETTE["ink"],
        linewidth=0.7,
        zorder=6,
        label="连续复核膝点",
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("移动距离（m）")
    axes[2].set_ylabel("最坏后验直径（m，对数刻度）")
    axes[2].set_title("Pareto前沿与膝点复核")
    style_axis(axes[2])
    axes[2].legend(loc="upper right", fontsize=7.5)
    add_panel_label(axes[2], "C")
    figure.suptitle("问题二选址模型的独立与跨构型验证", fontsize=13, fontweight="bold")
    save_publication_figure(figure, VALIDATION_FIGURE_PATH)
    plt.close(figure)


def main() -> None:
    demo = json.loads(DEMO_RESULT_PATH.read_text(encoding="utf-8"))
    grid_optimum = demo["posterior_grid_optimum"]["sampled_worst_diameter_m"]
    continuous = _continuous_optimizer_audit()
    continuous_pareto = _continuous_pareto_audit(demo)
    scenarios = _scenario_audits(demo)
    sampling_convergence = _sampling_convergence(demo)
    payload = {
        "continuous_optimizer_audit": continuous,
        "grid_vs_continuous_relative_gap": (
            grid_optimum / continuous["best"]["sampled_worst_diameter_m"] - 1.0
        ),
        "continuous_pareto_audit": continuous_pareto,
        "source_error_sampling_convergence": sampling_convergence,
        "scenario_audits": scenarios,
    }
    VALIDATION_RESULT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_validation_csv(scenarios, continuous_pareto)
    _plot_validation(demo, continuous, continuous_pareto, scenarios)
    print(f"Wrote {VALIDATION_RESULT_PATH}")
    print(f"Wrote {VALIDATION_TABLE_PATH}")
    print(f"Wrote {VALIDATION_FIGURE_PATH}")


if __name__ == "__main__":
    main()

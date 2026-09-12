"""Monte Carlo validation and publication figures for Question 4."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch

from plot_style import PALETTE, apply_publication_style, save_publication_figure, style_axis
from question1_geometry import Point
from question4_simulator import LocalMixedSimulator, random_case
from question4_strategy import (
    MixedDirectionalSearch,
    TARGET_RADIUS_M,
    plan_station_route,
    survey_stations_for_profile,
    triangular_lattice_stations,
)


TABLE_PATH = ROOT / "results" / "tables" / "question4_demo.json"
SUMMARY_PATH = ROOT / "results" / "tables" / "question4_summary.md"
FIGURE_DIR = ROOT / "results" / "figures"
SURVEY_PROFILE = "balanced"
PURSUIT_DEFLECTION_DEG = 8.0
ENROUTE_DETOUR_LIMIT_M = 500.0
ENROUTE_SPECULATIVE_LIMIT_M = 400.0


def run_cases(case_count: int = 30) -> tuple[list[dict[str, object]], LocalMixedSimulator]:
    records: list[dict[str, object]] = []
    example: LocalMixedSimulator | None = None
    for seed in range(case_count):
        sources = random_case(seed)
        simulator = LocalMixedSimulator(sources, seed=10_000 + seed)
        result = MixedDirectionalSearch(
            circle_sides=120,
            survey_profile=SURVEY_PROFILE,
            pursuit_deflection_deg=PURSUIT_DEFLECTION_DEG,
            enroute_detour_limit_m=ENROUTE_DETOUR_LIMIT_M,
            enroute_speculative_limit_m=ENROUTE_SPECULATIVE_LIMIT_M,
        ).run(simulator)
        actual = len(sources)
        directional = sum(source.is_directional for source in sources)
        movement_distance_m = 5.0 * sum(float(action["movement_s"]) for action in simulator.actions)
        records.append(
            {
                "seed": seed,
                "source_count": actual,
                "directional_count": directional,
                "omnidirectional_count": actual - directional,
                "cleared_count": len(result.cleared_channels),
                "cleared_ratio": len(result.cleared_channels) / actual,
                "completion_time_s": result.completion_time_s,
                "average_clear_time_s": result.average_clear_time_s,
                "measure_count": result.measure_count,
                "clear_attempt_count": result.clear_attempt_count,
                "survey_station_count": len(result.survey_stations_visited),
                "movement_distance_m": movement_distance_m,
            }
        )
        if seed == 3:
            example = simulator
    if example is None:
        raise RuntimeError("No example case was generated.")
    return records, example


def estimate_profile_reliability(sample_count: int = 1_000_000) -> dict[str, object]:
    """Estimate directional discovery reliability under the local case model."""

    rng = np.random.default_rng(260_913)
    radius = TARGET_RADIUS_M * np.sqrt(rng.random(sample_count))
    position_angle = 2.0 * np.pi * rng.random(sample_count)
    x = radius * np.cos(position_angle)
    y = radius * np.sin(position_angle)
    direction = 2.0 * np.pi * rng.random(sample_count)
    ux, uy = np.cos(direction), np.sin(direction)
    reception_radius = rng.uniform(1000.0, 1500.0, sample_count)
    results: dict[str, object] = {}
    for profile in ("fast", "balanced", "certified"):
        stations = survey_stations_for_profile(profile)
        detected = np.zeros(sample_count, dtype=bool)
        for station in stations:
            detected |= (
                ((x - station.x) ** 2 + (y - station.y) ** 2 <= reception_radius**2)
                & ((station.x - x) * ux + (station.y - y) * uy >= 0.0)
            )
        miss_rate = float((~detected).mean())
        route = plan_station_route(stations)
        route_length = sum(
            first.distance_to(second)
            for first, second in zip((Point(0.0, 0.0),) + route, route)
        )
        results[profile] = {
            "station_count": len(stations),
            "route_length_m": route_length,
            "directional_miss_rate": miss_rate,
            "estimated_all_detected_for_15_directional": (1.0 - miss_rate) ** 15,
        }
    return results


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    fields = (
        "cleared_ratio",
        "completion_time_s",
        "average_clear_time_s",
        "measure_count",
        "clear_attempt_count",
        "survey_station_count",
        "movement_distance_m",
    )
    summary: dict[str, object] = {
        "case_count": len(records),
        "survey_profile": SURVEY_PROFILE,
        "pursuit_deflection_deg": PURSUIT_DEFLECTION_DEG,
        "enroute_detour_limit_m": ENROUTE_DETOUR_LIMIT_M,
        "enroute_speculative_limit_m": ENROUTE_SPECULATIVE_LIMIT_M,
    }
    for field in fields:
        values = np.array([float(record[field]) for record in records])
        summary[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "p95": float(np.quantile(values, 0.95)),
        }
    summary["all_sources_cleared_in_every_case"] = all(
        record["cleared_count"] == record["source_count"] for record in records
    )
    active_stations = survey_stations_for_profile(SURVEY_PROFILE)
    summary["lattice_station_count"] = len(active_stations)
    route = plan_station_route(active_stations)
    summary["full_survey_route_m"] = sum(
        first.distance_to(second)
        for first, second in zip((Point(0.0, 0.0),) + route, route)
    )
    summary["profile_detection_reliability"] = estimate_profile_reliability()
    return summary


def write_summary_markdown(summary: dict[str, object]) -> None:
    labels = (
        ("cleared_ratio", "被清除比例", 3),
        ("completion_time_s", "总定位清除时间（s）", 1),
        ("average_clear_time_s", "平均定位清除时间（s/个）", 1),
        ("measure_count", "检测次数", 1),
        ("clear_attempt_count", "清除尝试次数", 1),
    )
    lines = [
        "<!-- Generated by src/question4_demo.py; do not edit values manually. -->",
        "| 指标 | 均值 | 中位数 | 最小值 | 最大值 | 95% 分位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label, digits in labels:
        values = summary[key]
        row = [label]
        for statistic in ("mean", "median", "minimum", "maximum", "p95"):
            row.append(f"{float(values[statistic]):.{digits}f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        (
            "",
            f"三角格点共 {summary['lattice_station_count']} 个；完整巡检开放路径长 "
            f"{float(summary['full_survey_route_m']):.1f} m。",
            "",
            "| 巡检模式 | 测站数 | 路线长度（m） | 单个定向源估计漏检率 | 15 个定向源全部发现估计概率 |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for profile in ("fast", "balanced", "certified"):
        values = summary["profile_detection_reliability"][profile]
        lines.append(
            f"| {profile} | {values['station_count']} | "
            f"{values['route_length_m']:.1f} | "
            f"{values['directional_miss_rate']:.6f} | "
            f"{values['estimated_all_detected_for_15_directional']:.6f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_lattice() -> None:
    apply_publication_style()
    figure, axis = plt.subplots(figsize=(5.3, 5.0))
    axis.add_patch(CirclePatch((0.0, 0.0), TARGET_RADIUS_M, fill=False, color=PALETTE["ink"], linewidth=1.4))
    all_stations = triangular_lattice_stations()
    active_stations = survey_stations_for_profile(SURVEY_PROFILE)
    route = plan_station_route(active_stations)
    route_x = [0.0] + [point.x for point in route]
    route_y = [0.0] + [point.y for point in route]
    axis.plot(route_x, route_y, color=PALETTE["gray"], linewidth=0.8, alpha=0.75)
    axis.scatter([point.x for point in route], [point.y for point in route], color=PALETTE["red"], s=22, zorder=3)
    omitted = [point for point in all_stations if point not in active_stations]
    axis.scatter(
        [point.x for point in omitted],
        [point.y for point in omitted],
        marker="x",
        color=PALETTE["gray"],
        s=24,
        label="Certified-only stations",
        zorder=3,
    )
    axis.set_aspect("equal")
    axis.set_xlim(-3000, 3000)
    axis.set_ylim(-3000, 3000)
    axis.set_xlabel("East coordinate (m)")
    axis.set_ylabel("North coordinate (m)")
    axis.set_title("Balanced directional discovery route")
    axis.text(0.02, 0.02, f"{len(route)} active stations; edge = 999 m", transform=axis.transAxes, fontsize=8)
    axis.legend(loc="upper right")
    style_axis(axis)
    save_publication_figure(figure, FIGURE_DIR / "question4_lattice_coverage.png")
    plt.close(figure)


def plot_metrics(records: list[dict[str, object]]) -> None:
    apply_publication_style()
    completion = np.array([float(record["completion_time_s"]) / 60.0 for record in records])
    average = np.array([float(record["average_clear_time_s"]) for record in records])
    directional = np.array([float(record["directional_count"]) for record in records])
    measures = np.array([float(record["measure_count"]) for record in records])
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.75))
    axes[0].hist(average, bins=8, color=PALETTE["orange"], alpha=0.85)
    axes[0].set_xlabel("Average time per source (s)")
    axes[0].set_title("Competition metric")
    scatter = axes[1].scatter(directional, completion, c=measures, cmap="viridis", s=24)
    axes[1].set_xlabel("Directional-source count")
    axes[1].set_ylabel("Completion time (min)")
    axes[1].set_title("Directional burden")
    figure.colorbar(scatter, ax=axes[1], label="Measurements")
    axes[2].scatter(measures, completion, color=PALETTE["teal"], s=22)
    axes[2].set_xlabel("Number of measurements")
    axes[2].set_ylabel("Completion time (min)")
    axes[2].set_title("Sensing-time tradeoff")
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.18, 1.03, label, transform=axis.transAxes, fontsize=12, fontweight="bold", color=PALETTE["ink"], va="bottom", ha="left")
        style_axis(axis)
    figure.tight_layout(w_pad=1.8)
    save_publication_figure(figure, FIGURE_DIR / "question4_monte_carlo.png")
    plt.close(figure)


def plot_example(simulator: LocalMixedSimulator) -> None:
    apply_publication_style()
    figure, axis = plt.subplots(figsize=(5.3, 5.0))
    axis.add_patch(CirclePatch((0.0, 0.0), TARGET_RADIUS_M, fill=False, color=PALETTE["ink"], linewidth=1.3))
    route_x = [0.0] + [float(action["x"]) for action in simulator.actions]
    route_y = [0.0] + [float(action["y"]) for action in simulator.actions]
    axis.plot(route_x, route_y, color=PALETTE["gray"], linewidth=0.65, alpha=0.65)
    omni = [source for source in simulator.sources.values() if not source.is_directional]
    directional = [source for source in simulator.sources.values() if source.is_directional]
    axis.scatter([source.position.x for source in omni], [source.position.y for source in omni], marker="o", s=28, color=PALETTE["teal"], label="Omnidirectional")
    axis.scatter([source.position.x for source in directional], [source.position.y for source in directional], marker="^", s=34, color=PALETTE["red"], label="Directional")
    for source in directional:
        angle = float(source.direction_deg) * np.pi / 180.0
        axis.arrow(source.position.x, source.position.y, 130 * np.cos(angle), 130 * np.sin(angle), color=PALETTE["red"], width=8, head_width=55, length_includes_head=True)
    axis.set_aspect("equal")
    axis.set_xlim(-3000, 3000)
    axis.set_ylim(-3000, 3000)
    axis.set_xlabel("East coordinate (m)")
    axis.set_ylabel("North coordinate (m)")
    axis.set_title("Example mixed-source trajectory")
    axis.legend(loc="upper right")
    style_axis(axis)
    save_publication_figure(figure, FIGURE_DIR / "question4_example_route.png")
    plt.close(figure)


def main() -> None:
    case_count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    records, example = run_cases(case_count)
    summary = summarize(records)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(
        json.dumps({"summary": summary, "cases": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_markdown(summary)
    plot_lattice()
    plot_metrics(records)
    plot_example(example)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Profile comparison tables and figures for Question 4."""

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
from matplotlib.patches import Circle as CirclePatch, Wedge

from plot_style import PALETTE, apply_publication_style, save_publication_figure, style_axis
from question1_geometry import Point
from question4_simulator import LocalMixedSimulator, random_case
from question4_strategy import (
    MixedDirectionalSearch,
    TARGET_RADIUS_M,
    plan_station_route,
    survey_stations_for_profile,
)


PROFILE_ORDER = ("certified", "balanced", "rapid", "turbo", "fast", "sprint")
PROFILE_NOTES = {
    "certified": "25-point triangular lattice; deterministic discovery proof.",
    "balanced": "23-point robust empirical set; zero misses in the reliability sample.",
    "rapid": "21-point speed/reliability compromise.",
    "turbo": "20-point default competition profile.",
    "fast": "20-point debug-speed profile with higher discovery risk.",
    "sprint": "18-point experimental speed profile; not recommended for formal tests.",
}
JSON_PATH = ROOT / "results" / "tables" / "question4_model_profiles.json"
MD_PATH = ROOT / "results" / "tables" / "question4_model_profiles.md"
FIGURE_DIR = ROOT / "results" / "figures"


def route_length_m(profile: str) -> float:
    route = plan_station_route(survey_stations_for_profile(profile))
    return sum(first.distance_to(second) for first, second in zip((Point(0.0, 0.0),) + route, route))


def run_profile_cases(profile: str, case_count: int) -> dict[str, object]:
    records: list[dict[str, float | int | bool]] = []
    for seed in range(case_count):
        sources = random_case(seed)
        simulator = LocalMixedSimulator(sources, seed=10_000 + seed)
        result = MixedDirectionalSearch(circle_sides=120, survey_profile=profile).run(simulator)
        records.append(
            {
                "seed": seed,
                "source_count": len(sources),
                "completion_time_s": result.completion_time_s,
                "average_clear_time_s": result.average_clear_time_s,
                "measure_count": result.measure_count,
                "survey_station_count": len(result.survey_stations_visited),
                "all_cleared": all(source.cleared for source in simulator.sources.values()),
            }
        )
    totals = np.array([float(record["completion_time_s"]) for record in records])
    averages = np.array([float(record["average_clear_time_s"]) for record in records])
    measures = np.array([float(record["measure_count"]) for record in records])
    stations = np.array([float(record["survey_station_count"]) for record in records])
    return {
        "records": records,
        "mean_total_time_s": float(totals.mean()),
        "median_total_time_s": float(np.median(totals)),
        "max_total_time_s": float(totals.max()),
        "mean_time_per_source_s": float(averages.mean()),
        "median_time_per_source_s": float(np.median(averages)),
        "mean_measure_count": float(measures.mean()),
        "mean_visited_stations": float(stations.mean()),
        "all_cases_cleared": bool(all(bool(record["all_cleared"]) for record in records)),
    }


def estimate_risks(sample_count: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(260_913)
    radius = TARGET_RADIUS_M * np.sqrt(rng.random(sample_count))
    position_angle = 2.0 * np.pi * rng.random(sample_count)
    x = radius * np.cos(position_angle)
    y = radius * np.sin(position_angle)
    direction = 2.0 * np.pi * rng.random(sample_count)
    ux, uy = np.cos(direction), np.sin(direction)
    reception_radius = rng.uniform(1000.0, 1500.0, sample_count)
    risks: dict[str, dict[str, float]] = {}
    for profile in PROFILE_ORDER:
        detected = np.zeros(sample_count, dtype=bool)
        for station in survey_stations_for_profile(profile):
            detected |= (
                ((x - station.x) ** 2 + (y - station.y) ** 2 <= reception_radius**2)
                & ((station.x - x) * ux + (station.y - y) * uy >= 0.0)
            )
        miss_rate = float((~detected).mean())
        risks[profile] = {
            "directional_miss_rate": miss_rate,
            "estimated_all_detected_for_15_directional": float((1.0 - miss_rate) ** 15),
            "estimated_any_miss_for_15_directional": float(1.0 - (1.0 - miss_rate) ** 15),
        }
    return risks


def build_report(case_count: int = 30, risk_samples: int = 500_000) -> dict[str, object]:
    risks = estimate_risks(risk_samples)
    profiles: dict[str, dict[str, object]] = {}
    for profile in PROFILE_ORDER:
        print(f"running profile: {profile}", flush=True)
        case_summary = run_profile_cases(profile, case_count)
        profiles[profile] = {
            "station_count": len(survey_stations_for_profile(profile)),
            "route_length_m": route_length_m(profile),
            "note": PROFILE_NOTES[profile],
            **risks[profile],
            **case_summary,
        }
    return {
        "case_count": case_count,
        "risk_samples": risk_samples,
        "recommended_formal_profile": "turbo",
        "profiles": profiles,
    }


def write_markdown(report: dict[str, object]) -> None:
    profiles = report["profiles"]
    case_count = int(report["case_count"])
    risk_samples = int(report["risk_samples"])
    lines = [
        "<!-- Generated by src/question4_profiles.py; do not edit values manually. -->",
        "# 问题四模型档位对比",
        "",
        "推荐正式测试默认使用 `turbo`。若演练出现漏检迹象，切换到 `rapid` 或 `balanced`；若需要理论保底，切换到 `certified`。",
        "",
        f"速度口径为固定种子 0--{case_count - 1} 的本地演练均值；风险口径为 {risk_samples:,} 次随机几何抽样。"
        "对数风险图中，抽样漏检为 0 的档位以 $10^{-7}$ 作为显示占位，表中仍记录为 0。",
        "",
        "| 模型 | 测站数 | 路线长度(m) | 平均总时长(s) | 平均每源(s) | 最大总时长(s) | 单定向漏检率 | 15定向任一漏检风险 | 定位 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile in PROFILE_ORDER:
        values = profiles[profile]
        lines.append(
            f"| `{profile}` | {values['station_count']} | {values['route_length_m']:.1f} | "
            f"{values['mean_total_time_s']:.1f} | {values['mean_time_per_source_s']:.1f} | "
            f"{values['max_total_time_s']:.1f} | {values['directional_miss_rate']:.6f} | "
            f"{values['estimated_any_miss_for_15_directional']:.6f} | {values['note']} |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_principle_diagram(report: dict[str, object]) -> None:
    apply_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))

    source = Point(0.0, 0.0)
    axes[0].add_patch(Wedge((source.x, source.y), 1.0, -50, 130, color=PALETTE["orange"], alpha=0.2))
    axes[0].plot([0.0, 0.0], [-1.15, 1.15], color=PALETTE["gray"], linewidth=1.0)
    axes[0].scatter([0.0], [0.0], color=PALETTE["red"], s=35, label="source")
    axes[0].scatter([0.65, -0.65], [0.35, 0.35], color=[PALETTE["teal"], PALETTE["gray"]], s=30)
    axes[0].annotate("signal side", (0.65, 0.35), xytext=(0.25, 0.9), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
    axes[0].annotate("shadow side", (-0.65, 0.35), xytext=(-1.05, 0.85), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"]})
    axes[0].set_title("Directional half-plane")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-1.25, 1.25)
    axes[0].set_ylim(-1.15, 1.15)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    colors = {
        "certified": PALETTE["ink"],
        "balanced": PALETTE["teal"],
        "rapid": PALETTE["blue"],
        "turbo": PALETTE["orange"],
        "fast": PALETTE["red"],
        "sprint": PALETTE["gray"],
    }
    axes[1].add_patch(CirclePatch((0.0, 0.0), TARGET_RADIUS_M, fill=False, color=PALETTE["ink"], linewidth=1.0))
    for profile in PROFILE_ORDER:
        stations = survey_stations_for_profile(profile)
        axes[1].scatter(
            [point.x for point in stations],
            [point.y for point in stations],
            s=10 if profile != "certified" else 8,
            alpha=0.55,
            label=profile,
            color=colors[profile],
        )
    axes[1].set_title("Survey profile nesting")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-2800, 2800)
    axes[1].set_ylim(-2800, 2800)
    axes[1].set_xlabel("East (m)")
    axes[1].set_ylabel("North (m)")
    axes[1].legend(fontsize=7, loc="upper right", ncols=2)
    for axis in axes:
        style_axis(axis)
    figure.tight_layout()
    save_publication_figure(figure, FIGURE_DIR / "question4_principle_diagram.png")
    plt.close(figure)


def plot_speed_comparison(report: dict[str, object]) -> None:
    apply_publication_style()
    profiles = list(PROFILE_ORDER)
    values = report["profiles"]
    total = np.array([values[profile]["mean_total_time_s"] for profile in profiles])
    per_source = np.array([values[profile]["mean_time_per_source_s"] for profile in profiles])
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    x = np.arange(len(profiles))
    axes[0].bar(x, total / 60.0, color=PALETTE["teal"], alpha=0.85)
    axes[0].set_ylabel("Mean completion time (min)")
    axes[0].set_xticks(x, profiles, rotation=30, ha="right")
    axes[0].set_title("Total time by profile")
    axes[1].bar(x, per_source, color=PALETTE["orange"], alpha=0.85)
    axes[1].axhline(500.0, color=PALETTE["red"], linestyle="--", linewidth=1.0, label="500 s/source target")
    axes[1].set_ylabel("Mean time per source (s)")
    axes[1].set_xticks(x, profiles, rotation=30, ha="right")
    axes[1].set_title("Competition metric")
    axes[1].legend(fontsize=7)
    for axis in axes:
        style_axis(axis)
    figure.tight_layout()
    save_publication_figure(figure, FIGURE_DIR / "question4_profile_speed.png")
    plt.close(figure)


def plot_risk_comparison(report: dict[str, object]) -> None:
    apply_publication_style()
    profiles = list(PROFILE_ORDER)
    values = report["profiles"]
    miss = np.array([max(values[profile]["directional_miss_rate"], 1e-7) for profile in profiles])
    any_miss = np.array([max(values[profile]["estimated_any_miss_for_15_directional"], 1e-7) for profile in profiles])
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    x = np.arange(len(profiles))
    bars0 = axes[0].bar(x, miss, color=PALETTE["red"], alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Single directional-source miss rate")
    axes[0].set_xticks(x, profiles, rotation=30, ha="right")
    axes[0].set_title("Discovery risk")
    bars1 = axes[1].bar(x, any_miss, color=PALETTE["lavender"], alpha=0.9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Any miss risk for 15 directional sources")
    axes[1].set_xticks(x, profiles, rotation=30, ha="right")
    axes[1].set_title("Worst-composition risk")
    for axis, bars, metric in (
        (axes[0], bars0, "directional_miss_rate"),
        (axes[1], bars1, "estimated_any_miss_for_15_directional"),
    ):
        for index, profile in enumerate(profiles):
            if values[profile][metric] == 0.0:
                bars[index].set_alpha(0.35)
                axis.text(index, 1.35e-7, "0 in\nsample", ha="center", va="bottom", fontsize=6)
    for axis in axes:
        style_axis(axis)
    figure.tight_layout()
    save_publication_figure(figure, FIGURE_DIR / "question4_profile_risk.png")
    plt.close(figure)


def main() -> None:
    case_count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    report = build_report(case_count=case_count)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report)
    plot_principle_diagram(report)
    plot_speed_comparison(report)
    plot_risk_comparison(report)
    print(json.dumps({k: v for k, v in report.items() if k != "profiles"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

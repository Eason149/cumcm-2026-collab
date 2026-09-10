"""Generate the reproducible synthetic example used in the Question 1 note."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "cumcm2026-matplotlib"),
)

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Polygon
from PIL import Image

from plot_style import apply_publication_style
from question1_geometry import Observation, Point, solve_question1


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "results" / "figures" / "question1_demo.png"
SENSITIVITY_FIGURE_PATH = ROOT / "results" / "figures" / "question1_sensitivity.png"
TABLE_PATH = ROOT / "results" / "tables" / "question1_demo.json"


def configure_matplotlib() -> None:
    """Use an available Chinese font and publication-friendly vector settings."""
    available = {font.name for font in fm.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    selected = next((font for font in candidates if font in available), None)
    if selected:
        mpl.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
        }
    )


def save_figure_bundle(figure: plt.Figure, png_path: Path) -> list[Path]:
    """Save PNG, PDF, SVG, and a grayscale PNG for reproducibility and print QA."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [png_path, png_path.with_suffix(".pdf"), png_path.with_suffix(".svg")]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        if path.suffix == ".svg":
            # Matplotlib emits spaces before many SVG newlines; normalize the
            # generated text so repository whitespace checks remain useful.
            svg_text = path.read_text(encoding="utf-8")
            clean_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
            path.write_text(clean_svg, encoding="utf-8", newline="\n")

    grayscale_path = png_path.with_name(f"{png_path.stem}_grayscale.png")
    with Image.open(png_path) as image:
        image.convert("L").save(grayscale_path, dpi=(300, 300))
    paths.append(grayscale_path)
    return paths


def ray_endpoint(observation: Observation, bearing_deg: float, length: float) -> Point:
    angle = math.radians(bearing_deg)
    return Point(
        observation.station.x + length * math.cos(angle),
        observation.station.y + length * math.sin(angle),
    )


def main() -> None:
    apply_publication_style()
    configure_matplotlib()
    observations = [
        Observation(Point(-800.0, 0.0), 0.0),
        Observation(Point(800.0, 0.0), 180.0),
        Observation(Point(0.0, -900.0), 90.0),
        Observation(Point(0.0, 900.0), 270.0),
    ]
    result = solve_question1(observations, error_deg=1.0)
    if result.intersection.status != "bounded":
        raise RuntimeError(f"Unexpected intersection status: {result.intersection.status}")

    vertices = result.intersection.vertices
    diameter = result.diameter
    enclosing = result.minimum_enclosing_circle
    assert diameter is not None and enclosing is not None

    sensitivity = []
    for error_deg in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        sensitivity_result = solve_question1(observations, error_deg=error_deg)
        sensitivity_circle = sensitivity_result.minimum_enclosing_circle
        assert sensitivity_result.diameter is not None and sensitivity_circle is not None
        sensitivity.append(
            {
                "angle_error_deg": error_deg,
                "diameter_m": sensitivity_result.diameter,
                "minimum_enclosing_radius_m": sensitivity_circle.radius,
                "guaranteed_clear_from_mec_center": sensitivity_circle.radius <= 20.0,
            }
        )

    payload = {
        "case": "four symmetric stations, true source at the origin",
        "angle_error_deg": 1.0,
        "observations": [asdict(observation) for observation in observations],
        "intersection_status": result.intersection.status,
        "vertices": [asdict(vertex) for vertex in vertices],
        "diameter_m": diameter,
        "diametral_pairs": [
            [asdict(first), asdict(second)] for first, second in result.diametral_pairs
        ],
        "diametral_circle_checks": [
            {
                "center": asdict(check.circle.center),
                "radius_m": check.circle.radius,
                "farthest_vertex_distance_m": check.farthest_vertex_distance,
                "covers": check.covers,
            }
            for check in result.diametral_circle_checks
        ],
        "minimum_enclosing_circle": {
            "center": asdict(enclosing.center),
            "radius_m": enclosing.radius,
            "support": [asdict(point) for point in enclosing.support],
        },
        "guaranteed_clear_from_mec_center": enclosing.radius <= 20.0,
        "sensitivity": sensitivity,
    }
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    palette = {
        "region": "#56B4E9", "region_edge": "#0072B2", "diameter": "#6A51A3",
        "circle": "#009E73", "source": "#D55E00", "station": "#222222",
        "ray": "#666666", "boundary": "#999999", "threshold": "#E69F00",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.05))
    polygon_xy = [(vertex.x, vertex.y) for vertex in vertices]
    first_pair = result.diametral_pairs[0]

    for index, axis in enumerate(axes):
        axis.add_patch(Polygon(
            polygon_xy, closed=True, facecolor=palette["region"],
            edgecolor=palette["region_edge"], linewidth=1.5, alpha=0.55,
            label="定位可行域", zorder=3,
        ))
        axis.scatter(
            [0.0], [0.0], marker="*", s=85, color=palette["source"],
            edgecolor="white", linewidth=0.5, label="真实位置（算例）", zorder=6,
        )
        axis.plot(
            [first_pair[0].x, first_pair[1].x],
            [first_pair[0].y, first_pair[1].y],
            color=palette["diameter"], linewidth=1.8,
            label=f"区域直径 {diameter:.2f} m", zorder=5,
        )
        axis.add_patch(CirclePatch(
            (enclosing.center.x, enclosing.center.y), enclosing.radius,
            fill=False, edgecolor=palette["circle"], linestyle="--", linewidth=1.6,
            label=f"最小包围圆半径 {enclosing.radius:.2f} m", zorder=4,
        ))
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("东向坐标 x（m）")
        axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.text(-0.12, 1.03, chr(ord("A") + index), transform=axis.transAxes,
                  fontsize=11, fontweight="bold")

        if index == 0:
            for obs_index, observation in enumerate(observations):
                center_end = ray_endpoint(observation, observation.bearing_deg, 1000.0)
                axis.plot(
                    [observation.station.x, center_end.x],
                    [observation.station.y, center_end.y],
                    color=palette["ray"], linewidth=0.9, alpha=0.85,
                    label="示向中心线" if obs_index == 0 else None, zorder=1,
                )
                for sign in (-1.0, 1.0):
                    boundary_end = ray_endpoint(
                        observation, observation.bearing_deg + sign, 1000.0
                    )
                    axis.plot(
                        [observation.station.x, boundary_end.x],
                        [observation.station.y, boundary_end.y],
                        color=palette["boundary"], linestyle=":", linewidth=0.8,
                        label="±1°角域边界" if obs_index == 0 and sign < 0 else None,
                        zorder=1,
                    )
            axis.scatter(
                [observation.station.x for observation in observations],
                [observation.station.y for observation in observations],
                marker="^", s=35, color=palette["station"], label="检测点", zorder=5,
            )
            axis.set_xlim(-980, 980)
            axis.set_ylim(-1080, 1080)
            axis.set_ylabel("北向坐标 y（m）")
            axis.set_title("全局测站与示向角域")
        else:
            margin = enclosing.radius * 1.35
            axis.set_xlim(enclosing.center.x - margin, enclosing.center.x + margin)
            axis.set_ylim(enclosing.center.y - margin, enclosing.center.y + margin)
            axis.set_title("定位区域局部放大")

    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=4, frameon=True, framealpha=0.92,
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.24, wspace=0.30)

    figure_paths = save_figure_bundle(figure, FIGURE_PATH)
    plt.close(figure)

    sensitivity_figure, sensitivity_axis = plt.subplots(
        figsize=(5.8, 3.6), constrained_layout=True
    )
    errors = [row["angle_error_deg"] for row in sensitivity]
    diameters = [row["diameter_m"] for row in sensitivity]
    radii = [row["minimum_enclosing_radius_m"] for row in sensitivity]
    sensitivity_axis.plot(
        errors, diameters, "o-", color=palette["diameter"], linewidth=1.6,
        markersize=4, label="定位区域直径",
    )
    sensitivity_axis.plot(
        errors, radii, "s--", color=palette["circle"], linewidth=1.6,
        markersize=4, label="最小包围圆半径",
    )
    sensitivity_axis.axhline(
        20.0, color=palette["threshold"], linestyle="-.", linewidth=1.4,
        label="20 m 清除半径",
    )
    sensitivity_axis.set_xlabel("示向误差上界（°）")
    sensitivity_axis.set_ylabel("距离（m）")
    sensitivity_axis.set_title("定位结果对示向误差上界的敏感性")
    sensitivity_axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
    sensitivity_axis.legend(frameon=True, framealpha=0.92)
    sensitivity_paths = save_figure_bundle(sensitivity_figure, SENSITIVITY_FIGURE_PATH)
    plt.close(sensitivity_figure)

    print(f"Wrote {TABLE_PATH}")
    for path in [*figure_paths, *sensitivity_paths]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

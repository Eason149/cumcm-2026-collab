"""Generate the reproducible synthetic example used in the Question 1 note."""

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
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Polygon

from question1_geometry import Observation, Point, solve_question1


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "results" / "figures" / "question1_demo.png"
SENSITIVITY_FIGURE_PATH = ROOT / "results" / "figures" / "question1_sensitivity.png"
TABLE_PATH = ROOT / "results" / "tables" / "question1_demo.json"


def main() -> None:
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

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), constrained_layout=True)
    polygon_xy = [(vertex.x, vertex.y) for vertex in vertices]
    stations_x = [observation.station.x for observation in observations]
    stations_y = [observation.station.y for observation in observations]
    first_pair = result.diametral_pairs[0]

    for index, axis in enumerate(axes):
        axis.add_patch(
            Polygon(
                polygon_xy,
                closed=True,
                facecolor="#9ecae1",
                edgecolor="#08519c",
                linewidth=2,
                alpha=0.65,
                label="Feasible polygon",
            )
        )
        axis.scatter(
            [0.0], [0.0], marker="*", s=160, color="#d7301f", label="True source"
        )
        axis.plot(
            [first_pair[0].x, first_pair[1].x],
            [first_pair[0].y, first_pair[1].y],
            color="#756bb1",
            linewidth=2,
            label=f"Diameter = {diameter:.3f} m",
        )
        axis.add_patch(
            CirclePatch(
                (enclosing.center.x, enclosing.center.y),
                enclosing.radius,
                fill=False,
                edgecolor="#31a354",
                linestyle="--",
                linewidth=2,
                label=f"Minimum enclosing radius = {enclosing.radius:.3f} m",
            )
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("East x (m)")
        axis.grid(alpha=0.25)

        if index == 0:
            axis.scatter(
                stations_x,
                stations_y,
                marker="^",
                s=80,
                color="#252525",
                label="Stations",
            )
            for observation in observations:
                axis.plot(
                    [observation.station.x, 0.0],
                    [observation.station.y, 0.0],
                    color="#969696",
                    linewidth=0.9,
                    alpha=0.7,
                )
            axis.set_ylabel("North y (m)")
            axis.set_title("Full station geometry")
            axis.legend(loc="upper right", fontsize=8)
        else:
            margin = enclosing.radius * 1.35
            axis.set_xlim(enclosing.center.x - margin, enclosing.center.x + margin)
            axis.set_ylim(enclosing.center.y - margin, enclosing.center.y + margin)
            axis.set_title("Localization region (zoomed)")
            axis.legend(loc="upper right", fontsize=8)

    figure.suptitle("Question 1 bounded-bearing localization", fontsize=15)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)

    sensitivity_figure, sensitivity_axis = plt.subplots(
        figsize=(7.2, 4.8), constrained_layout=True
    )
    errors = [row["angle_error_deg"] for row in sensitivity]
    diameters = [row["diameter_m"] for row in sensitivity]
    radii = [row["minimum_enclosing_radius_m"] for row in sensitivity]
    sensitivity_axis.plot(errors, diameters, "o-", color="#756bb1", label="Diameter")
    sensitivity_axis.plot(
        errors,
        radii,
        "s-",
        color="#31a354",
        label="Minimum enclosing radius",
    )
    sensitivity_axis.axhline(
        20.0,
        color="#d7301f",
        linestyle="--",
        linewidth=1.5,
        label="20 m clearing radius",
    )
    sensitivity_axis.set_xlabel("Bearing error bound (degrees)")
    sensitivity_axis.set_ylabel("Distance (m)")
    sensitivity_axis.set_title("Sensitivity to the bearing error bound")
    sensitivity_axis.grid(alpha=0.25)
    sensitivity_axis.legend()
    sensitivity_figure.savefig(SENSITIVITY_FIGURE_PATH, dpi=220)
    plt.close(sensitivity_figure)

    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {SENSITIVITY_FIGURE_PATH}")


if __name__ == "__main__":
    main()

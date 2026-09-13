"""Adapter for the 21-station backbone strategy used in Q4 speed trials.

The implementation keeps our repository interface (`RobotInterface` and
`StrategyResult`) while calling the locally supplied v4 backbone solver when
that reproducibility package is available in the workspace.  This lets our
tables compare the previous lattice profiles with the faster 21-station
backbone under the same local simulator and seeds.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from math import cos, pi, sin

from question1_geometry import Point
from question3_strategy import RobotInterface, StrategyResult


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "B题_Q3_Q4_本地复现包_20260912"
PACKAGE = ROOT / PACKAGE_NAME
if not PACKAGE.exists() and ROOT.parent.exists():
    PACKAGE = ROOT.parent / PACKAGE_NAME
V4_DIR = PACKAGE / "workstreams" / "q4_deep_optimization_v4_20260912"
V3_DIR = PACKAGE / "workstreams" / "q4_uniform_optimization_v3_20260912"
V2_DIR = PACKAGE / "workstreams" / "q4_local_optimization_v2_20260912"
V1_DIR = PACKAGE / "workstreams" / "q4_local_optimization_20260912"


V4_PARAMETERS: dict[str, Any] = {
    "route": "joint",
    "share": True,
    "directional": False,
    "angle_bin": 4.0,
    "trial_radius": 40.0,
    "probe_offset": 0.25,
    "radio_limit": 12,
    "search_first": False,
    "source_bias": 0.0,
    "crossbar": True,
    "crossbar_offset": 0.03,
    "crossbar_after_dark": False,
    "station_layout": "8_16",
    "station_inner": 990.0,
    "crossbar_fraction": 0.25,
    "conditional_range": True,
    "route_mode": "backbone",
    "route_estimate": 0.5,
    "opportunistic": False,
    "drop_sites": False,
    "share_ratio": 0.7,
    "negative_regions": "disk",
    "crossbar_reference": "latest",
    "stop_when_found16": True,
    "scan_current_first": True,
    "near_clear_distance": 50.0,
    "near_clear_radius": 20.0,
    "station_spec": [8, 12, 995, 1864, 0],
    "opportunistic_distance": 650.0,
    "opportunistic_candidates": 3,
    "exact_insert": True,
    "cumulative_points": 0.0,
    "cumulative_cap": 2,
    "initial_rotation_steps": 48,
    "initial_reflection": False,
    "initial_rotation_span": 1.5707963267948966,
}


def backbone21_stations() -> tuple[Point, ...]:
    """Return the public 21-station coverage net used by the backbone solver."""

    stations = [Point(0.0, 0.0)]
    stations.extend(
        Point(995.0 * cos(2.0 * pi * index / 8.0), 995.0 * sin(2.0 * pi * index / 8.0))
        for index in range(8)
    )
    stations.extend(
        Point(1864.0 * cos(2.0 * pi * index / 12.0), 1864.0 * sin(2.0 * pi * index / 12.0))
        for index in range(12)
    )
    return tuple(stations)


class _BackboneClient:
    def __init__(self, robot: RobotInterface) -> None:
        self.robot = robot
        self.rows: list[dict[str, object]] = []

    @property
    def position(self) -> np.ndarray:
        return np.array([self.robot.position.x, self.robot.position.y], dtype=float)

    @property
    def channel(self) -> int:
        return int(getattr(self.robot, "current_channel", 1))

    @property
    def virtual(self) -> float:
        return self.robot.virtual_time_s

    def act(self, endpoint: str, position: np.ndarray, channel: int) -> dict[str, object]:
        point = Point(float(position[0]), float(position[1]))
        if endpoint == "/measure":
            reply = self.robot.measure(point, channel)
            result = {
                "measure_result": reply.result,
                "svd_deg": reply.bearing_deg,
                "virtual_time_s": reply.virtual_time_s,
            }
        elif endpoint == "/clear":
            reply = self.robot.clear(point, channel)
            result = {
                "clear_result": "success" if reply.success else "no_target_in_range",
                "virtual_time_s": reply.virtual_time_s,
            }
        else:
            raise ValueError(f"unknown endpoint: {endpoint}")
        self.rows.append(
            {
                "endpoint": endpoint,
                "channel": channel,
                "x": point.x,
                "y": point.y,
                **result,
            }
        )
        return result


def _load_v4_strategy():
    if not V4_DIR.exists():
        raise FileNotFoundError(f"Q4 v4 reproducibility package not found: {V4_DIR}")
    for directory in (V1_DIR, V2_DIR, V3_DIR, V4_DIR):
        text = str(directory)
        if text not in sys.path:
            sys.path.insert(0, text)
    if "numba" not in sys.modules:
        numba_stub = types.ModuleType("numba")

        def njit(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorate(function):
                return function

            return decorate

        numba_stub.njit = njit
        sys.modules["numba"] = numba_stub
    module_path = V4_DIR / "strategy_v4.py"
    spec = importlib.util.spec_from_file_location("question4_external_strategy_v4", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_backbone21(robot: RobotInterface) -> StrategyResult:
    strategy = _load_v4_strategy()
    client = _BackboneClient(robot)
    result = strategy.solve(client, **V4_PARAMETERS)
    cleared = tuple(item["channel"] for item in result["cleared_sources"])
    absent = tuple(result.get("absent_channels", ()))
    stations = tuple(Point(float(x), float(y)) for x, y in result.get("stations", ()))
    measure_count = sum(1 for row in client.rows if row["endpoint"] == "/measure")
    clear_count = sum(1 for row in client.rows if row["endpoint"] == "/clear")
    return StrategyResult(
        cleared_channels=cleared,
        certified_absent_channels=absent,
        completion_time_s=robot.virtual_time_s,
        average_clear_time_s=robot.virtual_time_s / len(cleared) if cleared else float("inf"),
        survey_stations_visited=stations,
        measure_count=measure_count,
        clear_attempt_count=clear_count,
    )

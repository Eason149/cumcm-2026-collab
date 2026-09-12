"""开放路线与可移动搜索站优化。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from .config import Q3Config
from .geometry_backend import project_to_disks


def open_tour(
    position: Sequence[float],
    points: Sequence[Sequence[float]],
) -> tuple[list[int], float]:
    """多首点最近邻构造，再以开放路径 2-opt 改善。"""

    stacked = np.vstack((np.asarray(position), np.asarray(points)))
    task_count = len(points)
    distances = np.linalg.norm(
        stacked[:, None, :] - stacked[None, :, :], axis=2
    ).tolist()

    starts: list[tuple[float, list[int]]] = []
    for first in range(1, task_count + 1):
        route = [0, first]
        remaining = set(range(1, task_count + 1)) - {first}
        while remaining:
            next_point = min(
                remaining,
                key=lambda index: (distances[route[-1]][index], index),
            )
            route.append(next_point)
            remaining.remove(next_point)
        cost = sum(
            distances[start][end]
            for start, end in zip(route, route[1:])
        )
        starts.append((cost, route))

    candidates: list[tuple[float, list[int]]] = []
    for cost, route in sorted(starts)[:4]:
        for _ in range(2 * task_count):
            best_delta = 0.0
            best_segment: tuple[int, int] | None = None
            for start in range(1, task_count):
                for end in range(start + 1, task_count + 1):
                    delta = (
                        distances[route[start - 1]][route[end]]
                        - distances[route[start - 1]][route[start]]
                    )
                    if end < task_count:
                        delta += (
                            distances[route[start]][route[end + 1]]
                            - distances[route[end]][route[end + 1]]
                        )
                    if delta < best_delta - 1.0e-8:
                        best_delta = delta
                        best_segment = (start, end)
            if best_segment is None:
                break
            start, end = best_segment
            route[start : end + 1] = reversed(route[start : end + 1])
            cost += best_delta
        candidates.append((cost, route))

    cost, route = min(candidates)
    return [index - 1 for index in route[1:]], cost


def site_on_route(
    before: Sequence[float],
    after: Sequence[float],
    corners: np.ndarray,
    config: Q3Config,
) -> np.ndarray:
    """在四圆盘交集内最小化搜索站前后两段距离。"""

    previous = np.asarray(before)
    following = np.asarray(after)
    initial = project_to_disks(previous, corners)

    def objective(point: np.ndarray) -> float:
        return float(
            np.linalg.norm(point - previous) + np.linalg.norm(point - following)
        )

    def gradient(point: np.ndarray) -> np.ndarray:
        incoming = point - previous
        outgoing = point - following
        return (
            incoming / max(np.linalg.norm(incoming), 1.0e-10)
            + outgoing / max(np.linalg.norm(outgoing), 1.0e-10)
        )

    radius = config.search_inner_radius
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": lambda point: radius**2
            - np.sum((corners - point) ** 2, axis=1),
            "jac": lambda point: 2 * (corners - point),
        },
        options={"maxiter": 30, "ftol": 1.0e-7},
    )
    candidate = project_to_disks(result.x, corners)
    return candidate if objective(candidate) < objective(initial) else initial


"""共享探测和搜索—清除联合任务规划。"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .config import Q3Config
from .geometry_backend import (
    ERROR,
    bearing_halfplanes,
    clip_polygon,
    minimum_circle,
    project_to_disks,
)
from .models import SolverState
from .route import open_tour, site_on_route


TaskKind = Literal["source", "site"]
Task = tuple[TaskKind, int]


class JointPlanner:
    """v5 采用版的无副作用规划器。"""

    def __init__(
        self,
        config: Q3Config,
        sector_corners: dict[int, np.ndarray],
    ) -> None:
        self.config = config
        self.sector_corners = sector_corners

    def choose_initial_probe(
        self,
        state: SolverState,
        current_position: np.ndarray,
    ) -> np.ndarray:
        """从半径固定的 12 个候选中选择预计总收缩量最大的点。"""

        candidates = [
            self.config.probe_radius * np.array([math.cos(angle), math.sin(angle)])
            for angle in np.arange(0, 2 * math.pi, math.pi / 6)
        ]
        candidates = [
            point
            for point in candidates
            if math.dist(current_position, point) > self.config.probe_radius * 0.75
        ]
        return max(candidates, key=lambda point: self._probe_gain(state, point))

    def _probe_gain(self, state: SolverState, point: np.ndarray) -> float:
        total = 0.0
        for belief in state.known_sources.values():
            if belief.radius <= self.config.clearing_radius:
                continue
            delta = belief.center - point
            angle = math.degrees(math.atan2(delta[1], delta[0])) % 360
            halfplanes, bounds = bearing_halfplanes(point, angle, ERROR)
            polygon = clip_polygon(belief.polygon, halfplanes, bounds)
            if len(polygon):
                total += belief.radius - minimum_circle(polygon)["radius"]
        return total

    def choose_next_task(
        self,
        state: SolverState,
        current_position: np.ndarray,
    ) -> tuple[Task, dict[int, np.ndarray]]:
        """联合排列已发现源和未完成搜索扇区，只返回下一项任务。"""

        active_sectors = (
            sorted(state.pending_sectors) if state.unknown_channels else []
        )
        choices = {
            sector: project_to_disks(current_position, self.sector_corners[sector])
            for sector in active_sectors
        }

        if not state.known_sources:
            if not choices:
                raise RuntimeError("No pending action can resolve channels")
            sector = min(
                choices,
                key=lambda index: (math.dist(current_position, choices[index]), index),
            )
            return ("site", sector), choices

        labels: list[Task] = [
            ("source", channel) for channel in sorted(state.known_sources)
        ] + [("site", sector) for sector in active_sectors]
        planned_sites = [
            project_to_disks(
                self.config.planning_radius
                * np.array(
                    [
                        math.cos(
                            self.config.rotation + (sector - 1) * math.pi / 3
                        ),
                        math.sin(
                            self.config.rotation + (sector - 1) * math.pi / 3
                        ),
                    ]
                ),
                self.sector_corners[sector],
            )
            for sector in active_sectors
        ]
        points = [
            state.known_sources[channel].center
            for channel in sorted(state.known_sources)
        ] + planned_sites

        order, cost = open_tour(current_position, points)
        for _ in range(self.config.site_passes):
            for route_position, point_index in enumerate(order):
                kind, identifier = labels[point_index]
                if kind != "site":
                    continue
                before = (
                    current_position
                    if route_position == 0
                    else points[order[route_position - 1]]
                )
                if route_position == len(order) - 1:
                    points[point_index] = project_to_disks(
                        before, self.sector_corners[identifier]
                    )
                else:
                    points[point_index] = site_on_route(
                        before,
                        points[order[route_position + 1]],
                        self.sector_corners[identifier],
                        self.config,
                    )
            order, cost = open_tour(current_position, points)

        for point_index, (kind, identifier) in enumerate(labels):
            if kind == "site":
                choices[identifier] = points[point_index]
        state.route_decisions.append(
            {
                "sequence_before": None,
                "labels": labels,
                "order": order,
                "estimated_distance_m": cost,
            }
        )
        return labels[order[0]], choices


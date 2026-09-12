"""测量、扫描、共享测向和清除动作及其状态更新。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import Q3Config
from .geometry_backend import (
    ERROR,
    bearing_halfplanes,
    clip_polygon,
    continuous_cover,
    exclude_disk,
    initial_polygon,
    minimum_circle,
    negative_refine,
    project_to_disks,
)
from .models import ProgressCallback, SolverState, SourceBelief


class ActionExecutor:
    """执行真实动作，并把反馈写回 ``SolverState``。"""

    def __init__(
        self,
        client: Any,
        state: SolverState,
        config: Q3Config,
        sector_corners: dict[int, np.ndarray],
        progress: ProgressCallback | None = None,
    ) -> None:
        self.client = client
        self.state = state
        self.config = config
        self.sector_corners = sector_corners
        self.progress = progress

    def observe(self, channel: int, position: Any, kind: str) -> dict[str, Any]:
        point = np.array(position, dtype=float)
        reply = self.client.act("/measure", point, channel)
        record: dict[str, Any] = {
            "position": point.tolist(),
            "response": reply,
            "kind": kind,
            "sequence": len(self.client.rows),
        }
        self.state.observations[channel].append(record)

        if reply["measure_result"] == "no_signal":
            self._update_no_signal(channel, point)
        else:
            self._update_positive_signal(channel, point, reply)

        if channel in self.state.known_sources:
            belief = self.state.known_sources[channel]
            record.update(
                posterior_vertices=(
                    None if belief.polygon is None else belief.polygon.tolist()
                ),
                circle_center=belief.center.tolist(),
                bound_m=belief.radius,
            )
        return reply

    def _update_no_signal(self, channel: int, point: np.ndarray) -> None:
        self.state.negative_observations[channel].append(point.tolist())
        belief = self.state.known_sources.get(channel)
        if belief is not None:
            polygon = negative_refine(
                belief.polygon,
                self.state.negative_observations[channel],
                self.state.positive_observations[channel],
                "both",
            )
            circle = minimum_circle(polygon)
            belief.polygon = polygon
            belief.center = circle["center"]
            belief.radius = circle["radius"]
            return

        if continuous_cover(self.state.negative_observations[channel]):
            self.state.absent_channels.add(channel)
            self.state.remaining_channels.remove(channel)

    def _update_positive_signal(
        self,
        channel: int,
        point: np.ndarray,
        reply: dict[str, Any],
    ) -> None:
        self.state.positive_observations[channel].append(point.tolist())
        if reply["measure_result"] == "near":
            self.state.known_sources[channel] = SourceBelief(
                polygon=None,
                center=point,
                radius=self.config.near_bound_radius,
                last_signal_position=point,
            )
            return

        belief = self.state.known_sources.get(channel)
        if belief is None:
            polygon = initial_polygon(point, reply["svd_deg"])
        else:
            halfplanes, bounds = bearing_halfplanes(point, reply["svd_deg"], ERROR)
            polygon = clip_polygon(belief.polygon, halfplanes, bounds)

        polygon = negative_refine(
            polygon,
            self.state.negative_observations[channel],
            self.state.positive_observations[channel],
            "both",
        )
        circle = minimum_circle(polygon)
        localization_count = 0 if belief is None else belief.localization_count
        trial_count = 0 if belief is None else belief.trial_count
        self.state.known_sources[channel] = SourceBelief(
            polygon=polygon,
            center=circle["center"],
            radius=circle["radius"],
            last_signal_position=point,
            localization_count=localization_count,
            trial_count=trial_count,
        )

    def share(self) -> None:
        """在当前位置仅执行预计能显著收缩区域的共享测向。"""

        position = np.array(self.client.position)
        threshold = self.config.clearing_radius - self.config.numeric_epsilon
        for channel in sorted(self.state.known_sources):
            belief = self.state.known_sources[channel]
            if belief.radius <= threshold:
                continue
            if math.dist(position, belief.last_signal_position) < self.config.share_min_baseline:
                continue
            if math.dist(position, belief.center) > self.config.share_max_center_distance:
                continue
            delta = belief.center - position
            if np.linalg.norm(delta) < 1.0e-5:
                continue
            angle = math.degrees(math.atan2(delta[1], delta[0])) % 360
            halfplanes, bounds = bearing_halfplanes(position, angle, ERROR)
            polygon = clip_polygon(belief.polygon, halfplanes, bounds)
            if (
                len(polygon)
                and minimum_circle(polygon)["radius"]
                < self.config.share_ratio * belief.radius
            ):
                self.observe(channel, position, "shared_bearing")

    def scan(self, position: Any, sector_index: int | None, kind: str = "search") -> None:
        point = np.asarray(position)
        visit: dict[str, Any] = {
            "position": point.tolist(),
            "fixed_site_index": sector_index,
        }
        if sector_index is not None and sector_index != 0:
            visit["sector_corners"] = self.sector_corners[sector_index].tolist()
        self.state.scan_visits.append(visit)

        for channel in sorted(self.state.unknown_channels):
            self.observe(channel, point, kind)
        self.share()

    def clear(self, channel: int, *, trial: bool = False) -> bool:
        """执行保证清除或允许失败的试探清除。"""

        belief = self.state.known_sources[channel]
        if trial:
            point = self._trial_clear_point(belief)
        elif belief.polygon is None:
            point = belief.center
        else:
            point = project_to_disks(
                self.client.position,
                belief.polygon,
                self.config.clearing_radius - self.config.numeric_epsilon,
            )

        response = self.client.act("/clear", point, channel)
        bound = (
            float(np.max(np.linalg.norm(belief.polygon - point, axis=1)))
            if belief.polygon is not None
            else self.config.near_bound_radius
        )
        record: dict[str, Any] = {
            "sequence": len(self.client.rows),
            "channel": channel,
            "position": point.tolist(),
            "response": response,
            "trial": trial,
            "prior_vertices": (
                None if belief.polygon is None else belief.polygon.tolist()
            ),
            "prior_bound_m": bound,
        }
        self.state.clear_attempts.append(record)

        if response["clear_result"] != "success":
            return self._handle_clear_failure(channel, point, response, record, trial)

        self.state.cleared_sources.append(
            {
                "channel": channel,
                "clear_point": point.tolist(),
                "bound_m": min(bound, self.config.clearing_radius),
                "bound_source": (
                    "successful_clear_response" if trial else "prior_geometry"
                ),
                "observations": self.state.observations[channel],
                "virtual_time_after_clear_s": self.client.virtual,
            }
        )
        self.state.known_sources.pop(channel)
        self.state.remaining_channels.remove(channel)
        if self.progress:
            self.progress(
                f"已清除 {len(self.state.cleared_sources)} 个源；"
                f"最近频道 {channel}；虚拟时间 {self.client.virtual:.2f} 秒"
            )
        return True

    def _trial_clear_point(self, belief: SourceBelief) -> np.ndarray:
        if belief.polygon is None:
            return belief.center
        vertex = min(
            belief.polygon,
            key=lambda point: math.dist(self.client.position, point),
        )
        delta = belief.center - vertex
        length = np.linalg.norm(delta)
        return vertex + delta * min(
            1.0,
            self.config.trial_inset / max(length, 1.0e-10),
        )

    def _handle_clear_failure(
        self,
        channel: int,
        point: np.ndarray,
        response: dict[str, Any],
        record: dict[str, Any],
        trial: bool,
    ) -> bool:
        if not trial or response["clear_result"] != "no_target_in_range":
            raise RuntimeError("Certified clear failed")

        belief = self.state.known_sources[channel]
        belief.trial_count += 1
        polygon = exclude_disk(
            belief.polygon,
            point,
            self.config.clearing_radius - 1.0e-5,
        )
        if not len(polygon):
            raise RuntimeError("Clear-negative region became empty")
        circle = minimum_circle(polygon)
        belief.polygon = polygon
        belief.center = circle["center"]
        belief.radius = circle["radius"]
        record.update(
            posterior_vertices=polygon.tolist(),
            circle_center=belief.center.tolist(),
            bound_m=belief.radius,
        )
        return False


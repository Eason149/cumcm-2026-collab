"""第三题在线状态与数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


ProgressCallback = Callable[[str], None]


@dataclass
class SourceBelief:
    """单个已发现源的保守位置估计。"""

    polygon: np.ndarray | None
    center: np.ndarray
    radius: float
    last_signal_position: np.ndarray
    localization_count: int = 0
    trial_count: int = 0


@dataclass
class SolverState:
    """求解器在每次真实反馈后更新的全部公开状态。"""

    remaining_channels: set[int]
    pending_sectors: set[int]
    known_sources: dict[int, SourceBelief] = field(default_factory=dict)
    cleared_sources: list[dict[str, Any]] = field(default_factory=list)
    absent_channels: set[int] = field(default_factory=set)
    observations: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    negative_observations: dict[int, list[list[float]]] = field(default_factory=dict)
    positive_observations: dict[int, list[list[float]]] = field(default_factory=dict)
    scan_visits: list[dict[str, Any]] = field(default_factory=list)
    route_decisions: list[dict[str, Any]] = field(default_factory=list)
    clear_attempts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, channel_count: int, sector_count: int) -> "SolverState":
        channels = set(range(1, channel_count + 1))
        return cls(
            remaining_channels=channels,
            pending_sectors=set(range(1, sector_count + 1)),
            observations={channel: [] for channel in channels},
            negative_observations={channel: [] for channel in channels},
            positive_observations={channel: [] for channel in channels},
        )

    @property
    def unknown_channels(self) -> set[int]:
        """尚未发现、清除或判空的频道。"""

        return self.remaining_channels - set(self.known_sources)


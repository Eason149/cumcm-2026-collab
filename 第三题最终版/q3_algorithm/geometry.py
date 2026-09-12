"""v5 主流程使用的搜索扇区与定位点几何。"""

from __future__ import annotations

import math

import numpy as np

from .config import Q3Config
from .models import SourceBelief


def build_sector_corners(config: Q3Config) -> dict[int, np.ndarray]:
    """构造六个连续搜索扇区的四角点，返回值单位为米。"""

    return {
        sector + 1: np.array(
            [
                [
                    radius * math.cos(config.rotation + sector * math.pi / 3 + angle),
                    radius * math.sin(config.rotation + sector * math.pi / 3 + angle),
                ]
                for radius in (config.search_inner_radius, config.source_domain_radius)
                for angle in (-math.pi / 6, math.pi / 6)
            ]
        )
        for sector in range(config.sector_count)
    }


def choose_localization_target(
    belief: SourceBelief,
    current_position: np.ndarray,
    config: Q3Config,
) -> np.ndarray:
    """选择圆心或保证接收的偏置定位点。"""

    target = belief.center.copy()
    polygon = belief.polygon
    if not config.localization_offset or belief.localization_count % 2 != 0:
        return target
    if polygon is None:
        return target

    squared_distances = np.sum(
        (polygon[:, None, :] - polygon[None, :, :]) ** 2,
        axis=2,
    )
    first, second = np.unravel_index(
        np.argmax(squared_distances), squared_distances.shape
    )
    direction = polygon[first] - polygon[second]
    direction = direction / np.linalg.norm(direction)
    if np.dot(direction, current_position - target) < 0:
        direction = -direction
    transverse = np.array([-direction[1], direction[0]])

    candidates = [
        target
        + config.localization_advance * belief.radius * direction
        + sign * config.localization_offset * belief.radius * transverse
        for sign in (-1, 1)
    ]
    limit = config.reception_guard_radius - config.numeric_epsilon
    candidates = [
        point
        for point in candidates
        if np.max(np.linalg.norm(polygon - point, axis=1)) < limit
    ]
    if candidates:
        return min(
            candidates,
            key=lambda point: math.dist(current_position, point),
        )
    return target


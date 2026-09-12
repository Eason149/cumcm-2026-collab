"""包内几何内核的单一适配入口。

历史 ``q3_optimizer_v4.py`` 已整理为同包的 ``geometry_kernel.py``。其余模块
只依赖本适配层，后续替换几何实现时无需修改规划器和求解器。
"""

from __future__ import annotations

from .geometry_kernel import (
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


__all__ = [
    "ERROR",
    "bearing_halfplanes",
    "clip_polygon",
    "continuous_cover",
    "exclude_disk",
    "initial_polygon",
    "minimum_circle",
    "negative_refine",
    "project_to_disks",
]

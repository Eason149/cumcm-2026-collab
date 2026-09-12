"""第三题采用版参数与题设常量。"""

from __future__ import annotations

from dataclasses import dataclass


VERSION = "q3_joint_search_clear_route_v5"

# 保留原附件中的采用参数，便于结果文件与历史实验继续对齐。
SELECTED_PARAMETERS = {
    "route_mode": "joint",
    "probe": 250,
    "share_at_search": True,
    "share_during": False,
    "share_ratio": 0.7,
    "offset": 0.1,
    "advance": 0.25,
    "unknown_scan": False,
    "rotation": 0.0,
    "exact_positive": False,
    "info_points": False,
    "arc_scan": 0,
    "estimate": "circle",
    "trial_radius": 40,
    "trial_limit": 12,
    "probe_count": 1,
    "site_passes": 1,
    "site_bonus": 0.0,
    "planning_radius": 1500,
    "commit": False,
    "sweep_advance": 0.0,
}


@dataclass(frozen=True)
class Q3Config:
    """v5 正式路径实际使用的参数。

    单位：距离均为米，角度均为弧度（另有说明时除外）。
    """

    channel_count: int = 20
    minimum_source_count: int = 10
    maximum_source_count: int = 16
    sector_count: int = 6

    search_inner_radius: float = 990.0
    source_domain_radius: float = 1800.0
    reception_guard_radius: float = 1000.0
    clearing_radius: float = 20.0
    near_bound_radius: float = 5.0
    numeric_epsilon: float = 1.0e-6

    probe_radius: float = 250.0
    probe_count: int = 1
    share_ratio: float = 0.7
    share_min_baseline: float = 150.0
    share_max_center_distance: float = 1500.0

    localization_offset: float = 0.1
    localization_advance: float = 0.25
    localization_limit: int = 12

    trial_radius: float = 40.0
    trial_limit: int = 12
    trial_inset: float = 18.0

    site_passes: int = 1
    planning_radius: float = 1500.0
    rotation: float = 0.0
    max_action_rows: int = 550


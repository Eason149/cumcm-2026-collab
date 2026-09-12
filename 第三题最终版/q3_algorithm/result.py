"""第三题结果字典的统一组装。"""

from __future__ import annotations

from typing import Any

from .config import Q3Config
from .models import SolverState


def build_result(client: Any, state: SolverState, config: Q3Config) -> dict[str, Any]:
    cleared_count = len(state.cleared_sources)
    return {
        "strategy": "joint_tour",
        "cleared_count": cleared_count,
        "cleared_sources": state.cleared_sources,
        "cleared_channels": sorted(item["channel"] for item in state.cleared_sources),
        "observations": {
            str(channel): state.observations[channel]
            for channel in state.observations
        },
        "negative_observations": {
            str(channel): state.negative_observations[channel]
            for channel in sorted(state.absent_channels)
        },
        "scan_visits": state.scan_visits,
        "source_upper_bound_stop": cleared_count == config.maximum_source_count,
        "count_inferred_absent_channels": (
            sorted(state.remaining_channels)
            if cleared_count == config.maximum_source_count
            else []
        ),
        "average_virtual_seconds_per_cleared_source": (
            client.virtual / cleared_count
        ),
        "route_decisions": state.route_decisions,
        "clear_attempts": state.clear_attempts,
    }


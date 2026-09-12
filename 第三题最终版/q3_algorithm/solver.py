"""第三题 v5 采用版的顶层编排流程。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .actions import ActionExecutor
from .config import Q3Config, SELECTED_PARAMETERS, VERSION
from .geometry import build_sector_corners, choose_localization_target
from .models import ProgressCallback, SolverState
from .planner import JointPlanner
from .result import build_result


def solve(
    client: Any,
    *,
    config: Q3Config | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """执行第三题 v5 正式策略，不读取仿真世界内部状态。"""

    adopted = config or Q3Config()
    state = SolverState.create(adopted.channel_count, adopted.sector_count)
    sector_corners = build_sector_corners(adopted)
    actions = ActionExecutor(client, state, adopted, sector_corners, progress)
    planner = JointPlanner(adopted, sector_corners)

    actions.scan([0.0, 0.0], 0)
    _run_initial_probes(client, state, actions, planner, adopted)

    while state.remaining_channels:
        if len(client.rows) >= adopted.max_action_rows:
            raise RuntimeError("Exploration action cap")
        if len(state.cleared_sources) == adopted.maximum_source_count:
            break

        ready_channel = _nearest_ready_source(client, state, adopted)
        if ready_channel is not None:
            actions.clear(ready_channel)
            actions.share()
            continue

        route_record_count = len(state.route_decisions)
        task, site_choices = planner.choose_next_task(
            state, np.asarray(client.position)
        )
        if len(state.route_decisions) > route_record_count:
            state.route_decisions[-1]["sequence_before"] = len(client.rows)
        kind, identifier = task
        if kind == "site":
            state.pending_sectors.remove(identifier)
            actions.scan(site_choices[identifier], identifier)
            continue

        _execute_source_task(
            client,
            state,
            actions,
            identifier,
            adopted,
        )

    cleared_count = len(state.cleared_sources)
    if not adopted.minimum_source_count <= cleared_count <= adopted.maximum_source_count:
        raise RuntimeError("Source count outside Q3 bounds")
    return build_result(client, state, adopted)


def _run_initial_probes(
    client: Any,
    state: SolverState,
    actions: ActionExecutor,
    planner: JointPlanner,
    config: Q3Config,
) -> None:
    for _ in range(config.probe_count if config.probe_radius else 0):
        if not state.known_sources:
            break
        target = planner.choose_initial_probe(state, np.asarray(client.position))
        for channel in sorted(state.known_sources):
            if state.known_sources[channel].radius > config.clearing_radius:
                actions.observe(channel, target, "joint_probe")


def _nearest_ready_source(
    client: Any,
    state: SolverState,
    config: Q3Config,
) -> int | None:
    threshold = config.clearing_radius - config.numeric_epsilon
    ready = [
        channel
        for channel, belief in state.known_sources.items()
        if belief.radius <= threshold
        and math.dist(client.position, belief.center) <= 2 * config.clearing_radius
    ]
    if not ready:
        return None
    return min(
        ready,
        key=lambda channel: (
            math.dist(client.position, state.known_sources[channel].center),
            channel,
        ),
    )


def _execute_source_task(
    client: Any,
    state: SolverState,
    actions: ActionExecutor,
    channel: int,
    config: Q3Config,
) -> None:
    belief = state.known_sources[channel]
    clear_threshold = config.clearing_radius - config.numeric_epsilon
    if belief.radius <= clear_threshold:
        actions.clear(channel)
        actions.share()
        return
    if belief.radius <= config.trial_radius and belief.trial_count < config.trial_limit:
        actions.clear(channel, trial=True)
        actions.share()
        return

    prior_radius = belief.radius
    if prior_radius >= config.reception_guard_radius - config.numeric_epsilon:
        raise RuntimeError("Reception bound violated")
    target = choose_localization_target(
        belief,
        np.asarray(client.position),
        config,
    )
    max_vertex_distance = float(
        np.max(np.linalg.norm(belief.polygon - target, axis=1))
    )
    reply = actions.observe(channel, target, "localization")
    belief = state.known_sources[channel]
    belief.localization_count += 1
    state.observations[channel][-1].update(
        prior_radius_m=prior_radius,
        max_vertex_distance_m=max_vertex_distance,
        conditional_probe=False,
    )
    if (
        reply["measure_result"] == "no_signal"
        or belief.localization_count > config.localization_limit
    ):
        raise RuntimeError("Guaranteed localization failed")


def solve_optimized_v5(
    client: Any,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """与原附件同名的采用版入口。"""

    result = solve(client, progress=progress)
    result.update(version=VERSION, parameters=SELECTED_PARAMETERS.copy())
    return result

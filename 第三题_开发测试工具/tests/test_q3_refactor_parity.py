"""分层版与桌面原版在同一冻结案例上的动作序列回归测试。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest


WORKSPACE = Path(
    os.environ.get("Q3_ALGORITHM_ROOT", "D:/Cumcm/第三题最终版")
)
REFERENCE_REPO = Path(os.environ.get("Q3_REFERENCE_REPO", "D:/Cumcm"))
BACKEND_DIRECTORY = REFERENCE_REPO / "B题_算法分析与核心代码"
LEGACY_FILE = Path(
    os.environ.get(
        "Q3_LEGACY_FILE",
        "C:/Users/ASUS/Desktop/B题_算法分析与核心代码/第三问_核心算法.py",
    )
)

if not (BACKEND_DIRECTORY / "q3_optimizer_v4.py").exists():
    pytest.skip("缺少 q3_optimizer_v4.py，跳过跨工作区回归测试", allow_module_level=True)
if not LEGACY_FILE.exists():
    pytest.skip("缺少桌面原版第三问核心算法，跳过回归测试", allow_module_level=True)

sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(BACKEND_DIRECTORY))
sys.path.insert(0, str(REFERENCE_REPO / "src"))

# 桌面附件保留了一个未使用的历史 import；测试时提供空模块即可。
sys.modules.setdefault("practice_single_source", types.ModuleType("practice_single_source"))

from q3_algorithm import solve_optimized_v5 as solve_refactored  # noqa: E402
from q3_optimized.adapters.local_simulator import (  # noqa: E402
    DeterministicLocalSimulator,
)
from q3_optimized.cases import load_frozen_cases  # noqa: E402
from question1_geometry import Point  # noqa: E402


class LocalClient:
    """只实现原版与分层版共同使用的 client 协议。"""

    def __init__(self, simulator: DeterministicLocalSimulator) -> None:
        self.simulator = simulator
        self.position = np.zeros(2, dtype=float)
        self.channel = 1
        self.virtual = 0.0
        self.rows: list[dict[str, object]] = []

    def act(self, path: str, position: object, channel: int) -> dict[str, object]:
        coordinates = np.asarray(position, dtype=float)
        point = Point(float(coordinates[0]), float(coordinates[1]))
        if path == "/measure":
            reply = self.simulator.measure(point, channel)
            response = {"measure_result": reply.result, "svd_deg": reply.bearing_deg}
            self.channel = channel
        elif path == "/clear":
            reply = self.simulator.clear(point, channel)
            response = {"clear_result": reply.result}
        else:
            raise ValueError(path)
        self.position = coordinates.copy()
        self.virtual = float(reply.virtual_time_s)
        self.rows.append(
            {
                "path": path,
                "position": coordinates.copy(),
                "channel": channel,
                "response": response,
            }
        )
        return response


def _load_legacy_solver():
    specification = importlib.util.spec_from_file_location("desktop_q3_legacy", LEGACY_FILE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.solve_optimized_v5


@pytest.mark.parametrize("case_index", range(5))
def test_refactor_preserves_frozen_case_action_sequence(case_index: int) -> None:
    case_file = (
        REFERENCE_REPO
        / "data"
        / "processed"
        / "question3_optimized_frozen_cases_v1.json"
    )
    case = load_frozen_cases(case_file, split="tuning")[case_index]

    legacy_client = LocalClient(
        DeterministicLocalSimulator(case.fresh_sources(), error_seed=case.error_seed)
    )
    refactored_client = LocalClient(
        DeterministicLocalSimulator(case.fresh_sources(), error_seed=case.error_seed)
    )

    legacy_result = _load_legacy_solver()(legacy_client)
    refactored_result = solve_refactored(refactored_client)

    assert refactored_result["version"] == legacy_result["version"]
    assert refactored_result["cleared_channels"] == legacy_result["cleared_channels"]
    assert refactored_client.virtual == pytest.approx(legacy_client.virtual, abs=1.0e-9)
    assert len(refactored_client.rows) == len(legacy_client.rows)
    assert refactored_result == legacy_result

    for legacy_action, refactored_action in zip(
        legacy_client.rows, refactored_client.rows
    ):
        assert refactored_action["path"] == legacy_action["path"]
        assert refactored_action["channel"] == legacy_action["channel"]
        assert refactored_action["response"] == legacy_action["response"]
        np.testing.assert_allclose(
            refactored_action["position"],
            legacy_action["position"],
            rtol=0.0,
            atol=1.0e-9,
        )

"""历史第三问 v5 算法的官方模拟器客户端适配层。

历史算法使用 ``client.act(path, numpy_position, channel)`` 以及
``position``、``channel``、``rows``、``virtual`` 状态；当前仓库的官方 HTTP
客户端则使用 ``measure``/``clear`` 和强类型回复。本模块只做这两种
契约之间的转换，不改动 v5 算法。

安全性由仓库的通用 ``q3`` HTTP 传输层和本文件的严格响应校验保证：四个官方
端点、同一 request_id 重试、逐请求 JSONL 日志、响应字段校验和单调虚拟时间。
本适配层另外要求显式运行确认，并在会话期间安装现实时间截止保护，任何退出
路径都尽力调用 ``/exit``。这里没有导入或运行 ``src/q3_optimized`` 的策略。
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q3.adapters.official_http import (  # noqa: E402
    OfficialSimulatorClient as _BaseOfficialSimulatorClient,
    SimulatorProtocolError,
)
from q3.models import EnterReply, ExitReply  # noqa: E402
from q3.runtime.deadline import DeadlineGuard  # noqa: E402
from q3.runtime.exit_guard import ExitGuard  # noqa: E402
from q4.protocol import (  # noqa: E402
    ProtocolError as WireProtocolError,
    endpoint,
    identifier,
    validate_response,
)
from question1_geometry import Point  # noqa: E402


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class VerifiedExitReply(ExitReply):
    """已由严格协议校验确认是用户主动退出的响应。"""

    exit_reason: str = "user_exit"


def _finite_float(response: dict[str, Any], key: str) -> float:
    try:
        value = float(response[key])
    except (KeyError, TypeError, ValueError) as error:
        raise SimulatorProtocolError(f"Missing or invalid {key}") from error
    if not math.isfinite(value):
        raise SimulatorProtocolError(f"{key} must be finite")
    return value


def _validate_timestamp(response: dict[str, Any]) -> None:
    if _finite_float(response, "real_timestamp_ms") < 0.0:
        raise SimulatorProtocolError("real_timestamp_ms must be non-negative")


def _validate_virtual_time(response: dict[str, Any], previous: float) -> float:
    value = _finite_float(response, "virtual_time_s")
    if value < 0.0:
        raise SimulatorProtocolError("virtual_time_s must be non-negative")
    if value + 1.0e-9 < previous:
        raise SimulatorProtocolError("virtual_time_s must not decrease")
    return value


def _optional_float(response: dict[str, Any], key: str) -> float | None:
    if key not in response or response[key] is None:
        return None
    return _finite_float(response, key)


class OfficialSimulatorClient(_BaseOfficialSimulatorClient):
    """通用 q3 传输层加附件入口需要的严格响应验证。"""

    def _post(
        self,
        path: str,
        payload: dict[str, object],
        *,
        enforce_deadline: bool = True,
    ) -> dict[str, Any]:
        previous = self._virtual_time_s
        response = super()._post(path, payload, enforce_deadline=enforce_deadline)
        try:
            _validate_timestamp(response)
            self._virtual_time_s = _validate_virtual_time(response, previous)
        except SimulatorProtocolError:
            self._virtual_time_s = previous
            raise
        return response

    def enter(self) -> EnterReply:
        previous = self._virtual_time_s
        response = super()._post(
            "/enter",
            self._base_payload(self._request_id("enter")),
            enforce_deadline=False,
        )
        # 服务端可能已接受 /enter；后续字段无效时仍必须让清理逻辑尝试 /exit。
        self.entered = True
        try:
            validate_response(200, response, "/enter")
            self._virtual_time_s = _validate_virtual_time(response, previous)
        except (WireProtocolError, SimulatorProtocolError) as error:
            self._virtual_time_s = previous
            if isinstance(error, SimulatorProtocolError):
                raise
            raise SimulatorProtocolError(f"Invalid official /enter response: {error}") from error
        return EnterReply(
            _finite_float(response, "remaining_real_duration_s"),
            self._virtual_time_s,
        )

    def exit(self) -> ExitReply:
        previous = self._virtual_time_s
        response = super()._post(
            "/exit",
            self._base_payload(self._request_id("exit")),
            enforce_deadline=False,
        )
        self.entered = False
        try:
            validate_response(200, response, "/exit")
            self._virtual_time_s = _validate_virtual_time(response, previous)
        except (WireProtocolError, SimulatorProtocolError) as error:
            self._virtual_time_s = previous
            if isinstance(error, SimulatorProtocolError):
                raise
            raise SimulatorProtocolError(f"Invalid official /exit response: {error}") from error
        return VerifiedExitReply(
            test_duration=_optional_float(response, "test_duration"),
            program_runtime=_optional_float(response, "program_runtime"),
            virtual_time_s=_optional_float(response, "virtual_time_s"),
        )


@dataclass(frozen=True)
class LegacySessionResult:
    """一次历史算法官方会话的结果。"""

    strategy_result: Any
    program_runtime_s: float
    exit_reply: ExitReply | None


class _LegacyRawOfficialTransport(OfficialSimulatorClient):
    """严格传输层的极薄扩展：保留服务器原始回复字典。"""

    def act_raw(self, path: str, point: Point, channel: int) -> dict[str, Any]:
        payload = self._base_payload(self._request_id(path.removeprefix("/")))
        payload.update(
            {"position": {"x": point.x, "y": point.y}, "channel": channel}
        )
        response = self._post(path, payload)
        try:
            validate_response(200, response, path)
        except WireProtocolError as error:
            raise SimulatorProtocolError(
                f"Invalid official response for {path}: {error}"
            ) from error

        # 只有通过完整路径响应校验的动作才改变本地状态。
        self._position = point
        if path == "/measure":
            self.current_channel = channel
        return response


class LegacyV5OfficialClient:
    """把严格官方 HTTP 客户端适配成历史 v5 的 client 契约。"""

    def __init__(
        self,
        robot_id: str,
        *,
        base_url: str = "http://127.0.0.1:2026",
        log_path: Path,
        timeout_s: float = 5.0,
        retry_count: int = 3,
        retry_backoff_s: float = 0.25,
        confirm_official_run: bool = False,
    ) -> None:
        # 在构造传输层之前校验，确保该适配器永远不会连到非本机地址。
        endpoint(base_url)
        identifier(robot_id, 64)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be a finite positive number")
        if type(retry_count) is not int or retry_count < 1:
            raise ValueError("retry_count must be a positive integer")
        if not math.isfinite(retry_backoff_s) or retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s must be finite and non-negative")

        self.log_path = Path(log_path)
        self.timeout_s = float(timeout_s)
        self.retry_count = retry_count
        self.retry_backoff_s = float(retry_backoff_s)
        self._confirmed = confirm_official_run is True
        self._transport = _LegacyRawOfficialTransport(
            robot_id,
            base_url=base_url,
            log_path=self.log_path,
            timeout_s=self.timeout_s,
            retry_count=self.retry_count,
            retry_backoff_s=self.retry_backoff_s,
        )

        # 这些名称是历史 solve() 直接读取的契约。
        self.position = np.array([0.0, 0.0], dtype=float)
        self.channel = 1
        self.virtual = 0.0
        self.rows: list[dict[str, Any]] = []

    @property
    def entered(self) -> bool:
        return self._transport.entered

    @property
    def minimum_exit_margin_s(self) -> float:
        """覆盖 ``/exit`` 全部超时和重试退避的保守下界。"""

        backoff = self.retry_backoff_s * self.retry_count * (self.retry_count - 1) / 2
        return self.retry_count * self.timeout_s + backoff

    def validate_safety_margin(self, safety_margin_s: float) -> None:
        if not math.isfinite(safety_margin_s) or safety_margin_s <= 0.0:
            raise ValueError("safety_margin_s must be a finite positive number")
        if safety_margin_s + 1e-12 < self.minimum_exit_margin_s:
            raise ValueError(
                "safety_margin_s is too small for worst-case /exit retries: "
                f"need at least {self.minimum_exit_margin_s:.2f}s"
            )

    def _sync_state(self) -> None:
        point = self._transport.position
        self.position = np.array([point.x, point.y], dtype=float)
        self.virtual = float(self._transport.virtual_time_s)

    def set_deadline(self, deadline: DeadlineGuard | None) -> None:
        self._transport.set_deadline(deadline)

    def enter(self) -> EnterReply:
        if not self._confirmed:
            raise PermissionError(
                "Refusing /enter without explicit confirm_official_run=True"
            )
        reply = self._transport.enter()
        self._sync_state()
        return reply

    def act(self, path: str, position: Any, channel: int) -> dict[str, Any]:
        """执行一次历史 v5 动作，返回其期待的字典响应。"""

        if not self.entered:
            raise RuntimeError("Official simulator session has not entered")
        if path not in {"/measure", "/clear"}:
            raise ValueError("Legacy act path must be /measure or /clear")
        if type(channel) is not int or not 1 <= channel <= 20:
            raise ValueError("channel must be an integer in 1..20")
        coordinates = np.asarray(position, dtype=float)
        if coordinates.shape != (2,) or not np.all(np.isfinite(coordinates)):
            raise ValueError("position must contain exactly two finite coordinates")
        if np.any(np.abs(coordinates) > 10000.0):
            raise ValueError("position is outside the protocol coordinate guard")
        point = Point(float(coordinates[0]), float(coordinates[1]))

        response = self._transport.act_raw(path, point, channel)

        # 仅在服务器接受且严格校验通过后更新历史状态。
        self._sync_state()
        # 按官方计时规则，只有测向会切换接收频道；
        # 清除指令携带的 channel 不改变当前测向频道。
        if path == "/measure":
            self.channel = channel
        self.rows.append(
            {
                "sequence": len(self.rows) + 1,
                "path": path,
                "position": self.position.tolist(),
                "channel": channel,
                "response": dict(response),
                "virtual_time_s": self.virtual,
            }
        )
        return response

    def exit(self) -> ExitReply:
        reply = self._transport.exit()
        self._sync_state()
        return reply

    def try_exit(self) -> ExitReply | None:
        reply = self._transport.try_exit()
        self._sync_state()
        return reply

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "LegacyV5OfficialClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.try_exit()
        self.close()


def run_legacy_session(
    client: LegacyV5OfficialClient,
    solver: Callable[[LegacyV5OfficialClient], ResultT],
    *,
    safety_margin_s: float = 30.0,
) -> LegacySessionResult:
    """以官方生命周期运行 v5，包括截止保护和异常时 ``/exit``。"""

    client.validate_safety_margin(safety_margin_s)
    started = time.monotonic()
    exit_reply: ExitReply | None = None
    with ExitGuard(client) as guard:
        entered = client.enter()
        client.set_deadline(
            DeadlineGuard.from_remaining(
                entered.remaining_real_seconds,
                safety_margin_s=safety_margin_s,
            )
        )
        strategy_result = solver(client)
        exit_reply = guard.finish()
    return LegacySessionResult(
        strategy_result=strategy_result,
        program_runtime_s=time.monotonic() - started,
        exit_reply=exit_reply,
    )


__all__ = [
    "LegacySessionResult",
    "LegacyV5OfficialClient",
    "VerifiedExitReply",
    "run_legacy_session",
]

"""第三题分层版的官方模拟器专用入口。

``--check-only`` 只检查代码、依赖和参数，不联网；``--probe-port`` 只检测
本机模拟器端口，不调用 ``/enter``。正式运行必须显式提供机器人编号和
``--confirm-official-run``，因为 ``/enter`` 可能消耗一次演练或正式测试机会。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from datetime import datetime
from pathlib import Path
import socket
import sys
from typing import Any
from urllib.parse import urlsplit
import uuid


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
SOURCE_DIRECTORY = REPOSITORY / "src"
FINAL_DIRECTORY = REPOSITORY / "第三题最终版"
for path in (FINAL_DIRECTORY, HERE, SOURCE_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", help="与模拟器当前登录队号逐字一致")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:2026",
        help="官方模拟器本机地址",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="单次 HTTP 超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="同一 request_id 的最多请求次数")
    parser.add_argument(
        "--exit-margin",
        type=float,
        default=30.0,
        help="为 /exit 预留的现实时间",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "logs")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查分层算法、几何后端和官方客户端依赖，不联网",
    )
    parser.add_argument(
        "--probe-port",
        action="store_true",
        help="只建立 TCP 连接检查本机端口，不调用 /enter",
    )
    parser.add_argument(
        "--confirm-official-run",
        action="store_true",
        help="确认本次 /enter 可能消耗一次演练或正式测试机会",
    )
    return parser


def _validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from q4.protocol import endpoint, identifier

    try:
        endpoint(args.url)
        if args.robot_id is not None:
            identifier(args.robot_id, 64)
    except ValueError as exc:
        parser.error(str(exc))
    if args.timeout <= 0:
        parser.error("--timeout 必须为正数")
    if args.retries < 1:
        parser.error("--retries 必须至少为 1")
    worst_case_exit = (
        args.retries * args.timeout
        + 0.25 * args.retries * (args.retries - 1) / 2
    )
    if args.exit_margin < worst_case_exit:
        parser.error(
            f"--exit-margin 至少应为 {worst_case_exit:.2f} 秒，以覆盖最坏 /exit 重试"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def resolve_solver():
    import q3_algorithm
    from q3_algorithm import solve_optimized_v5
    from legacy_v5_official import LegacyV5OfficialClient, run_legacy_session

    algorithm_directory = FINAL_DIRECTORY / "q3_algorithm"
    helper = algorithm_directory / "geometry_kernel.py"
    if not helper.is_file():
        raise RuntimeError("缺少 q3_algorithm/geometry_kernel.py")
    source_files = sorted(algorithm_directory.glob("*.py"))
    if not source_files:
        raise RuntimeError("缺少 q3_algorithm 分层代码")

    provenance = {
        "strategy_entrypoint": "q3_algorithm.solve_optimized_v5",
        "strategy_version": q3_algorithm.VERSION,
        "selected_parameters": dict(q3_algorithm.SELECTED_PARAMETERS),
        "strategy_sources": {
            path.name: _sha256(path) for path in source_files
        },
        "geometry_source": str(helper.resolve()),
        "geometry_sha256": _sha256(helper),
        "transport_adapter": "legacy_v5_official.LegacyV5OfficialClient",
    }
    return solve_optimized_v5, LegacyV5OfficialClient, run_legacy_session, provenance


def probe_port(url: str, timeout: float) -> dict[str, Any]:
    parsed = urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=min(timeout, 5.0)):
            pass
    except OSError as exc:
        return {
            "tcp_reachable": False,
            "host": host,
            "port": port,
            "error": str(exc),
            "enter_called": False,
        }
    return {
        "tcp_reachable": True,
        "host": host,
        "port": port,
        "enter_called": False,
    }


def run_official(args: argparse.Namespace) -> int:
    solver, client_class, run_session, provenance = resolve_solver()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = args.output_dir.resolve() / f"q3_v5_{stamp}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    action_log = folder / "actions.jsonl"
    metadata_path = folder / "run_metadata.json"
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "question": 3,
        "execution_mode": "official_http",
        "algorithm": provenance,
        "client_log": str(action_log),
    }
    _write_json(metadata_path, metadata)

    try:
        with client_class(
            args.robot_id,
            base_url=args.url,
            log_path=action_log,
            timeout_s=args.timeout,
            retry_count=args.retries,
            confirm_official_run=args.confirm_official_run,
        ) as client:
            outcome = run_session(
                client,
                lambda current: solver(
                    current,
                    progress=lambda message: print(message, flush=True),
                ),
                safety_margin_s=args.exit_margin,
            )
    except BaseException as exc:
        _write_json(
            folder / "failure.json",
            {
                **metadata,
                "failed_at": datetime.now().astimezone().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    result = asdict(outcome)
    result.update(
        question=3,
        execution_mode="official_http",
        algorithm=provenance,
        client_log=str(action_log),
    )
    result_path = folder / "result.json"
    _write_json(result_path, result)
    print(
        json.dumps(
            {
                "ok": True,
                "strategy_version": provenance["strategy_version"],
                "cleared_count": outcome.strategy_result.get("cleared_count"),
                "average_virtual_seconds_per_cleared_source": (
                    outcome.strategy_result.get(
                        "average_virtual_seconds_per_cleared_source"
                    )
                ),
                "exit_reason": getattr(outcome.exit_reply, "exit_reason", None),
                "result": str(result_path),
                "request_log": str(action_log),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(args, parser)

    check_report: dict[str, Any] = {}
    if args.check_only:
        _, _, _, provenance = resolve_solver()
        check_report = {
            "dependencies_ok": True,
            "algorithm": provenance,
            "network_called": False,
            "enter_called": False,
        }
    if args.probe_port:
        check_report.update(probe_port(args.url, args.timeout))
    if args.check_only or args.probe_port:
        print(json.dumps(check_report, ensure_ascii=False, indent=2))
        return 0 if check_report.get("tcp_reachable", True) else 1

    if not args.robot_id:
        parser.error("正式连接必须提供 --robot-id")
    if not args.confirm_official_run:
        parser.error("拒绝调用 /enter：请确认后添加 --confirm-official-run")
    print(
        "即将连接官方模拟器，请确认界面已选择第三题且接口已就绪。",
        flush=True,
    )
    return run_official(args)


if __name__ == "__main__":
    raise SystemExit(main())

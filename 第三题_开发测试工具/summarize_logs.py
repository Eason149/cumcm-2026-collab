"""汇总第三题官方模拟器日志；只读取，不修改日志。"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parent / "logs"


def accepted_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        response = record.get("response") or {}
        request = record.get("request") or {}
        request_id = request.get("request_id")
        if response.get("accepted") is not True or not request_id:
            continue
        key = (record["path"], request_id)
        if key not in seen:
            accepted.append(record)
            seen.add(key)
    errors = sum(bool(record.get("error")) for record in records)
    return accepted, errors


def summarize_run(folder: Path) -> dict[str, Any]:
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    strategy = result["strategy_result"]
    records, errors = accepted_records(folder / "actions.jsonl")
    enters = [record for record in records if record["path"] == "/enter"]
    exits = [record for record in records if record["path"] == "/exit"]
    if len(enters) != 1 or len(exits) != 1:
        raise ValueError("入场或退场记录数量不是 1")

    actions = [
        record for record in records if record["path"] in {"/measure", "/clear"}
    ]
    measure_records = [record for record in actions if record["path"] == "/measure"]
    clear_records = [record for record in actions if record["path"] == "/clear"]
    successful_clears = [
        record
        for record in clear_records
        if record["response"].get("clear_result") == "success"
    ]
    failed_clears = [
        record
        for record in clear_records
        if record["response"].get("clear_result") != "success"
    ]
    cleared_channels = {record["request"]["channel"] for record in successful_clears}

    distance = 0.0
    position = (0.0, 0.0)
    receiver_channel = 1
    switches = 0
    signal_counts: Counter[str] = Counter()
    for record in actions:
        requested = record["request"]["position"]
        next_position = (float(requested["x"]), float(requested["y"]))
        distance += math.dist(position, next_position)
        position = next_position
        if record["path"] == "/measure":
            channel = int(record["request"]["channel"])
            if channel != receiver_channel:
                switches += 1
            receiver_channel = channel
            signal_counts[record["response"]["measure_result"]] += 1

    total_s = (
        float(exits[0]["response"]["virtual_time_s"])
        - float(enters[0]["response"]["virtual_time_s"])
    )
    source_count = len(cleared_channels)
    movement_s = distance / 5.0
    measure_s = 5.0 * len(measure_records)
    switch_s = float(switches)
    successful_clear_s = 5.0 * len(successful_clears)
    failed_clear_s = 3.0 * len(failed_clears)
    reconstructed_s = (
        movement_s + measure_s + switch_s + successful_clear_s + failed_clear_s
    )

    observation_kinds: Counter[str] = Counter()
    for channel_records in strategy["observations"].values():
        observation_kinds.update(record["kind"] for record in channel_records)

    if strategy["cleared_count"] != source_count:
        raise ValueError("结果文件清除数与动作日志不一致")

    return {
        "run": folder.name,
        "sources": source_count,
        "total_virtual_s": total_s,
        "per_source_s": total_s / source_count,
        "program_runtime_s": float(result["program_runtime_s"]),
        "exit_reason": result["exit_reply"].get("exit_reason"),
        "actions": len(actions),
        "measures": len(measure_records),
        "clear_attempts": len(clear_records),
        "successful_clears": len(successful_clears),
        "failed_clears": len(failed_clears),
        "channel_switches": switches,
        "movement_m": distance,
        "movement_s": movement_s,
        "measure_s": measure_s,
        "switch_s": switch_s,
        "successful_clear_s": successful_clear_s,
        "failed_clear_s": failed_clear_s,
        "reconstructed_s": reconstructed_s,
        "time_residual_s": total_s - reconstructed_s,
        "signal_counts": dict(signal_counts),
        "observation_kinds": dict(observation_kinds),
        "scan_visits": len(strategy["scan_visits"]),
        "route_decisions": len(strategy["route_decisions"]),
        "upper_bound_stop": bool(strategy["source_upper_bound_stop"]),
        "request_errors": errors,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_source = [row["per_source_s"] for row in rows]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["sources"]].append(row)

    total_sources = sum(row["sources"] for row in rows)
    total_virtual = sum(row["total_virtual_s"] for row in rows)
    total_costs = {
        key: sum(row[key] for row in rows)
        for key in (
            "movement_s",
            "measure_s",
            "switch_s",
            "successful_clear_s",
            "failed_clear_s",
        )
    }
    counters = {
        key: sum(row[key] for row in rows)
        for key in (
            "actions",
            "measures",
            "clear_attempts",
            "successful_clears",
            "failed_clears",
            "channel_switches",
            "movement_m",
            "program_runtime_s",
            "scan_visits",
            "route_decisions",
            "request_errors",
        )
    }
    signal_totals: Counter[str] = Counter()
    observation_kind_totals: Counter[str] = Counter()
    for row in rows:
        signal_totals.update(row["signal_counts"])
        observation_kind_totals.update(row["observation_kinds"])
    return {
        "runs": len(rows),
        "normal_exits": sum(row["exit_reason"] == "user_exit" for row in rows),
        "upper_bound_stops": sum(row["upper_bound_stop"] for row in rows),
        "total_sources": total_sources,
        "total_virtual_s": total_virtual,
        "weighted_per_source_s": total_virtual / total_sources,
        "case_mean_per_source_s": statistics.mean(per_source),
        "median_per_source_s": statistics.median(per_source),
        "population_std_per_source_s": statistics.pstdev(per_source),
        "min_per_source_s": min(per_source),
        "max_per_source_s": max(per_source),
        "runs_at_most_200_s_per_source": sum(value <= 200 for value in per_source),
        "groups": {
            source_count: {
                "runs": len(items),
                "sources": sum(item["sources"] for item in items),
                "total_virtual_s": sum(item["total_virtual_s"] for item in items),
                "weighted_per_source_s": (
                    sum(item["total_virtual_s"] for item in items)
                    / sum(item["sources"] for item in items)
                ),
                "case_mean_per_source_s": statistics.mean(
                    item["per_source_s"] for item in items
                ),
                "min_per_source_s": min(item["per_source_s"] for item in items),
                "max_per_source_s": max(item["per_source_s"] for item in items),
            }
            for source_count, items in sorted(groups.items())
        },
        "totals": counters,
        "signal_totals": dict(signal_totals),
        "observation_kind_totals": dict(observation_kind_totals),
        "runs_with_failed_clear": sum(row["failed_clears"] > 0 for row in rows),
        "time_costs": {
            key: {
                "seconds": value,
                "percent": 100.0 * value / total_virtual,
            }
            for key, value in total_costs.items()
        },
        "maximum_absolute_time_residual_s": max(
            abs(row["time_residual_s"]) for row in rows
        ),
        "cases": rows,
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for folder in sorted(ROOT.glob("q3_v5_*")):
        if not folder.is_dir():
            continue
        if not (folder / "result.json").exists():
            excluded.append({"run": folder.name, "reason": "缺少 result.json"})
            continue
        try:
            rows.append(summarize_run(folder))
        except Exception as exc:
            excluded.append({"run": folder.name, "reason": str(exc)})
    output = aggregate(rows)
    output["excluded"] = excluded
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

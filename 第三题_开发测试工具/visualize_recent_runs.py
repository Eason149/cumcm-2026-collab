"""从最近三次完整官方日志生成第三题论文型可视化。"""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

import summarize_logs


HERE = Path(__file__).resolve().parent
LOG_ROOT = HERE / "logs"
OUTPUT_DIR = HERE.parent / "results" / "figures" / "q3_official_recent"
RUNS = [
    ("测试1", "q3_v5_20260913_010525_13abbda6"),
    ("测试2", "q3_v5_20260913_010657_58b3448c"),
    ("测试3", "q3_v5_20260913_010746_8a62cda6"),
]

BLUE = "#3B6FB6"
TEAL = "#4C9F91"
ORANGE = "#D98C4A"
PURPLE = "#7D789A"
RED = "#C65A5A"
GREEN = "#318A67"
GRID = "#D9E1EA"
DOMAIN_FILL = "#F3F7FA"
TEXT = "#243447"


plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Source Han Serif SC"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlecolor": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    }
)


def _accepted_actions(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        response = row.get("response") or {}
        request = row.get("request") or {}
        request_id = request.get("request_id")
        if response.get("accepted") is not True or not request_id:
            continue
        key = (row["path"], request_id)
        if key in seen:
            continue
        seen.add(key)
        if row["path"] in {"/measure", "/clear"}:
            accepted.append(row)
    return accepted


def _position(record: dict[str, Any]) -> np.ndarray:
    value = record["request"]["position"]
    return np.array([float(value["x"]), float(value["y"])])


def _unique_points(points: list[np.ndarray], tolerance: float = 1.0e-7) -> np.ndarray:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - prior) <= tolerance for prior in unique):
            unique.append(point)
    return np.asarray(unique) if unique else np.empty((0, 2))


def _compress_route(points: list[np.ndarray]) -> np.ndarray:
    compressed = [np.array([0.0, 0.0])]
    for point in points:
        if np.linalg.norm(point - compressed[-1]) > 1.0e-7:
            compressed.append(point)
    return np.asarray(compressed)


def load_run(label: str, name: str) -> dict[str, Any]:
    folder = LOG_ROOT / name
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    strategy = result["strategy_result"]
    actions = _accepted_actions(folder / "actions.jsonl")
    summary = summarize_logs.summarize_run(folder)

    kind_points: dict[str, list[np.ndarray]] = {}
    for channel_rows in strategy["observations"].values():
        for row in channel_rows:
            kind_points.setdefault(row["kind"], []).append(
                np.asarray(row["position"], dtype=float)
            )

    success_clears = [
        _position(row)
        for row in actions
        if row["path"] == "/clear"
        and row["response"].get("clear_result") == "success"
    ]
    failed_clears = [
        _position(row)
        for row in actions
        if row["path"] == "/clear"
        and row["response"].get("clear_result") != "success"
    ]
    return {
        "label": label,
        "name": name,
        "short_id": name.rsplit("_", 1)[-1],
        "strategy": strategy,
        "summary": summary,
        "route": _compress_route([_position(row) for row in actions]),
        "search_points": _unique_points(
            [np.asarray(row["position"], dtype=float) for row in strategy["scan_visits"]]
        ),
        "probe_points": _unique_points(kind_points.get("joint_probe", [])),
        "localization_points": _unique_points(kind_points.get("localization", [])),
        "success_clears": _unique_points(success_clears),
        "failed_clears": _unique_points(failed_clears),
    }


def _scatter_if_any(ax: plt.Axes, points: np.ndarray, **kwargs: Any) -> None:
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], **kwargs)


def draw_route(ax: plt.Axes, run: dict[str, Any], *, compact: bool) -> None:
    summary = run["summary"]
    route = run["route"]
    domain = Circle(
        (0, 0),
        1800,
        facecolor=DOMAIN_FILL,
        edgecolor="#98A9BC",
        linewidth=1.1 if compact else 1.5,
        zorder=0,
    )
    ax.add_patch(domain)
    ax.plot(
        route[:, 0],
        route[:, 1],
        color=TEAL,
        linewidth=1.05 if compact else 1.45,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        route[:, 0],
        route[:, 1],
        s=5 if compact else 9,
        color=TEAL,
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    _scatter_if_any(
        ax,
        run["search_points"],
        s=25 if compact else 45,
        marker="o",
        facecolors="white",
        edgecolors=BLUE,
        linewidths=1.2,
        zorder=4,
    )
    _scatter_if_any(
        ax,
        run["probe_points"],
        s=28 if compact else 55,
        marker="D",
        color=PURPLE,
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    _scatter_if_any(
        ax,
        run["localization_points"],
        s=24 if compact else 45,
        marker="^",
        color=ORANGE,
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
    )
    _scatter_if_any(
        ax,
        run["success_clears"],
        s=42 if compact else 75,
        marker="*",
        color=GREEN,
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
    )
    _scatter_if_any(
        ax,
        run["failed_clears"],
        s=32 if compact else 65,
        marker="x",
        color=RED,
        linewidths=1.4,
        zorder=7,
    )
    ax.scatter([0], [0], marker="P", s=32 if compact else 55, color=TEXT, zorder=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2050, 2050)
    ax.set_ylim(-2050, 2050)
    ax.grid(color=GRID, linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("x / m", fontsize=8 if compact else 10)
    ax.set_ylabel("y / m", fontsize=8 if compact else 10)
    ax.tick_params(labelsize=7 if compact else 9)
    ax.set_title(
        f"{run['label']}｜{summary['sources']} 源｜{summary['per_source_s']:.2f} s/源",
        fontsize=10 if compact else 13,
        fontweight="bold",
        pad=7,
    )
    ax.text(
        0.02,
        0.02,
        f"移动 {summary['movement_m']:.0f} m  ·  动作 {summary['actions']} 次  ·  失败清除 {summary['failed_clears']} 次",
        transform=ax.transAxes,
        fontsize=6.8 if compact else 9,
        color="#52677D",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
        zorder=10,
    )


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=TEAL, lw=1.5, label="机器人移动轨迹"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=BLUE, label="搜索站", markersize=6),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PURPLE, label="联合探测点", markersize=6),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE, label="专门定位点", markersize=6),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=GREEN, label="成功清除位置", markersize=8),
        Line2D([0], [0], marker="x", color=RED, lw=0, label="失败清除位置", markersize=6),
        Line2D([0], [0], marker="P", color=TEXT, lw=0, label="起点", markersize=6),
    ]


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_overview(runs: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(15.5, 10.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.055, right=0.97, bottom=0.13, top=0.88, wspace=0.22, hspace=0.28)
    bar_ax = fig.add_subplot(grid[0, 0])
    route_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    values = [run["summary"]["per_source_s"] for run in runs]
    labels = [f"{run['label']}\n{run['summary']['sources']} 源" for run in runs]
    bars = bar_ax.barh(labels, values, color=[BLUE, TEAL, PURPLE], height=0.58)
    bar_ax.axvline(200, color=RED, linewidth=1.2, linestyle="--", label="200 s/源参考线")
    bar_ax.set_xlim(0, max(values) * 1.25)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("平均定位清除时间 / (s/源)")
    bar_ax.set_title("三次正式测试效率对比", fontsize=13, fontweight="bold", pad=10)
    bar_ax.grid(axis="x", color=GRID, linewidth=0.7)
    bar_ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        bar_ax.text(value + 3, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=10, color=TEXT, fontweight="bold")
    bar_ax.text(
        0.02,
        -0.18,
        "三次合并平均：218.82 s/源；39/39 个干扰源全部清除",
        transform=bar_ax.transAxes,
        fontsize=10,
        color="#52677D",
    )

    for ax, run in zip(route_axes, runs):
        draw_route(ax, run, compact=True)

    fig.suptitle("问题三｜V5 搜索—定位—清除联合规划：正式测试轨迹", fontsize=19, fontweight="bold", color=TEXT, y=0.955)
    fig.text(0.055, 0.915, "数据来源：最近三次完整官方模拟器日志；圆周表示半径 1800 m 的源域", fontsize=10.5, color="#52677D")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.014, "注：星形点为成功清除位置，并非模拟器未公开的真实干扰源坐标；本图展示轨迹与动作位置，不展示接收覆盖带。", ha="center", fontsize=9, color="#66788A")
    save_figure(fig, "01_最近三次正式测试总览")


def create_time_breakdown(runs: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor="white")
    categories = OrderedDict(
        [
            ("移动", ("movement_s", TEAL)),
            ("测向", ("measure_s", BLUE)),
            ("频道切换", ("switch_s", PURPLE)),
            ("成功清除", ("successful_clear_s", GREEN)),
            ("失败清除", ("failed_clear_s", RED)),
        ]
    )
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))
    for label, (key, color) in categories.items():
        values = np.array([run["summary"][key] / run["summary"]["sources"] for run in runs])
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.62)
        bottom += values
    ax.axhline(200, color=RED, linewidth=1.2, linestyle="--", alpha=0.8)
    for index, total in enumerate(bottom):
        ax.text(index, total + 4, f"{total:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT)
    ax.set_xticks(x, [f"{run['label']}\n{run['summary']['sources']} 源" for run in runs])
    ax.set_ylabel("每源虚拟时间 / (s/源)")
    ax.set_title("最近三次正式测试的每源耗时构成", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.text(0.5, 0.015, "三次累计耗时中，移动占 74.02%，测向占 19.80%；虚线为 200 s/源参考值。", ha="center", fontsize=9.5, color="#52677D")
    fig.tight_layout(rect=(0.03, 0.08, 0.98, 0.96))
    save_figure(fig, "02_最近三次耗时构成")


def create_individual_routes(runs: list[dict[str, Any]]) -> None:
    for index, run in enumerate(runs, start=1):
        fig, ax = plt.subplots(figsize=(8.2, 8.2), facecolor="white")
        draw_route(ax, run, compact=False)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8.7, bbox_to_anchor=(0.5, 0.015))
        fig.text(0.5, 0.965, f"日志编号：{run['short_id']}｜成功清除位置仅为清除动作坐标", ha="center", fontsize=9, color="#52677D")
        fig.tight_layout(rect=(0.03, 0.075, 0.98, 0.95))
        save_figure(fig, f"{index + 2:02d}_{run['label']}_机器人轨迹")


def main() -> None:
    runs = [load_run(label, name) for label, name in RUNS]
    create_overview(runs)
    create_time_breakdown(runs)
    create_individual_routes(runs)
    print(f"已生成 {len(list(OUTPUT_DIR.glob('*')))} 个文件：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

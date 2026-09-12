"""从正式测试前的 19 次完整模拟日志生成论文型可视化。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import visualize_recent_runs as base


RESULT_DIR = base.HERE.parent / "results" / "figures" / "q3_simulation_history"
FINAL_DIR = base.HERE.parent / "第三题最终版" / "可视化图片"

SIM_RUNS = [
    "q3_v5_20260913_002956_fa35d4f0",
    "q3_v5_20260913_003016_9e24c34e",
    "q3_v5_20260913_003029_6b5c4d64",
    "q3_v5_20260913_003041_b4e33a4b",
    "q3_v5_20260913_003053_1fd0127d",
    "q3_v5_20260913_003117_9043180c",
    "q3_v5_20260913_003131_251cf0bf",
    "q3_v5_20260913_003144_eb332b34",
    "q3_v5_20260913_003157_eccf35d1",
    "q3_v5_20260913_003209_2a7f4f48",
    "q3_v5_20260913_003220_eae1ec60",
    "q3_v5_20260913_003233_f6d5798c",
    "q3_v5_20260913_003246_d75dec1e",
    "q3_v5_20260913_003259_987f0c0f",
    "q3_v5_20260913_003310_9dbc06f3",
    "q3_v5_20260913_003320_1b3e328c",
    "q3_v5_20260913_003331_b13059a9",
    "q3_v5_20260913_003342_45142f84",
    "q3_v5_20260913_003406_f0319547",
]


def save_figure(fig: plt.Figure, stem: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    png = RESULT_DIR / f"{stem}.png"
    pdf = RESULT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(png, FINAL_DIR / png.name)
    shutil.copy2(pdf, FINAL_DIR / pdf.name)


def load_runs() -> list[dict]:
    return [base.load_run(f"模拟{i:02d}", name) for i, name in enumerate(SIM_RUNS, 1)]


def representative_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["summary"]["sources"]].append(run)
    selected = []
    for sources in sorted(grouped):
        run = min(grouped[sources], key=lambda item: item["summary"]["per_source_s"])
        copied = dict(run)
        copied["label"] = f"{sources}源代表案例"
        selected.append(copied)
    return selected


def create_dashboard(selected: list[dict]) -> None:
    fig = plt.figure(figsize=(22, 11.8), facecolor="white")
    grid = fig.add_gridspec(
        2, 4, left=0.035, right=0.985, bottom=0.125, top=0.865,
        wspace=0.22, hspace=0.30,
    )
    bar_ax = fig.add_subplot(grid[0, 0])
    route_axes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(1, 8)]

    values = [run["summary"]["per_source_s"] for run in selected]
    labels = [f"{run['summary']['sources']} 源" for run in selected]
    colors = [base.BLUE if value > 200 else base.TEAL for value in values]
    bars = bar_ax.barh(labels, values, color=colors, height=0.60)
    bar_ax.axvline(200, color=base.RED, linewidth=1.2, linestyle="--")
    bar_ax.set_xlim(0, max(values) * 1.24)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("平均定位清除时间 / (s/源)")
    bar_ax.set_title("各源数量的最佳模拟案例", fontsize=14, fontweight="bold", pad=10)
    bar_ax.grid(axis="x", color=base.GRID, linewidth=0.7)
    bar_ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        bar_ax.text(
            value + 2.5, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
            va="center", fontsize=10, color=base.TEXT, fontweight="bold",
        )
    bar_ax.text(
        0.02, -0.18, "选取规则：同一源数量下平均耗时最短的完整模拟记录",
        transform=bar_ax.transAxes, fontsize=9.5, color="#52677D",
    )

    for ax, run in zip(route_axes, selected):
        base.draw_route(ax, run, compact=True)

    total_sources = sum(run["summary"]["sources"] for run in selected)
    cleared = sum(run["summary"]["successful_clears"] for run in selected)
    weighted = sum(run["summary"]["total_virtual_s"] for run in selected) / total_sources
    fig.suptitle(
        "问题三｜V5 多源搜索、定位与清除：代表性模拟案例",
        fontsize=21, fontweight="bold", color=base.TEXT, y=0.955,
    )
    fig.text(
        0.035, 0.905,
        f"正式测试前 19 次完整模拟日志中按源数量分组取最佳案例｜代表案例合计 {cleared}/{total_sources} 全部清除｜加权平均 {weighted:.2f} s/源",
        fontsize=11, color="#52677D",
    )
    fig.legend(
        handles=base.legend_handles(), loc="lower center", ncol=7,
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.046),
    )
    fig.text(
        0.5, 0.015,
        "注：星形点为成功清除指令的位置，并非模拟器未公开的真实干扰源坐标；轨迹图未展示接收覆盖带。",
        ha="center", fontsize=9.5, color="#66788A",
    )
    save_figure(fig, "06_模拟测试代表案例总览")


def create_efficiency_scatter(runs: list[dict], selected: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0), facecolor="white")
    rng = np.random.default_rng(20260913)
    xs = np.array([run["summary"]["sources"] for run in runs], dtype=float)
    ys = np.array([run["summary"]["per_source_s"] for run in runs])
    jitter = rng.uniform(-0.11, 0.11, len(runs))
    ax.scatter(xs + jitter, ys, s=65, color=base.BLUE, alpha=0.82, edgecolor="white", linewidth=0.8, label="完整模拟案例")
    sx = np.array([run["summary"]["sources"] for run in selected])
    sy = np.array([run["summary"]["per_source_s"] for run in selected])
    ax.plot(sx, sy, color=base.TEAL, marker="*", markersize=12, linewidth=1.8, label="各源数量最佳案例")
    ax.axhline(200, color=base.RED, linewidth=1.3, linestyle="--", label="200 s/源参考线")
    weighted = sum(run["summary"]["total_virtual_s"] for run in runs) / sum(run["summary"]["sources"] for run in runs)
    ax.axhline(weighted, color=base.PURPLE, linewidth=1.2, linestyle=":", label=f"19次加权平均 {weighted:.2f} s/源")
    ax.set_xticks(range(10, 17))
    ax.set_xlabel("干扰源数量 / 个")
    ax.set_ylabel("平均定位清除时间 / (s/源)")
    ax.set_title("正式测试前 19 次模拟测试的清除效率分布", fontsize=17, fontweight="bold", pad=14)
    ax.grid(color=base.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(0.5, 0.018, "每个圆点对应一次完整模拟测试；横向轻微错位仅用于避免同源数量样本重叠，不改变数据。", ha="center", fontsize=9.5, color="#52677D")
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.97))
    save_figure(fig, "07_模拟测试清除效率分布")


def create_group_time_breakdown(runs: list[dict]) -> None:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["summary"]["sources"]].append(run)
    source_counts = sorted(grouped)
    categories = [
        ("移动", "movement_s", base.TEAL),
        ("测向", "measure_s", base.BLUE),
        ("频道切换", "switch_s", base.PURPLE),
        ("成功清除", "successful_clear_s", base.GREEN),
        ("失败清除", "failed_clear_s", base.RED),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 7.0), facecolor="white")
    x = np.arange(len(source_counts))
    bottom = np.zeros(len(source_counts))
    for label, key, color in categories:
        values = np.array([
            np.mean([run["summary"][key] / run["summary"]["sources"] for run in grouped[n]])
            for n in source_counts
        ])
        ax.bar(x, values, bottom=bottom, width=0.67, color=color, label=label)
        bottom += values
    ax.axhline(200, color=base.RED, linewidth=1.25, linestyle="--")
    for i, total in enumerate(bottom):
        ax.text(i, total + 3, f"{total:.1f}", ha="center", fontsize=9.5, fontweight="bold", color=base.TEXT)
    ax.set_xticks(x, [f"{n}源\n(n={len(grouped[n])})" for n in source_counts])
    ax.set_ylabel("组内案例平均每源耗时 / (s/源)")
    ax.set_title("不同干扰源数量下的平均耗时构成", fontsize=17, fontweight="bold", pad=14)
    ax.grid(axis="y", color=base.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, 0.018, "各分量先按单次测试折算为每源耗时，再在相同源数量组内取算术平均；n 为该组完整测试次数。", ha="center", fontsize=9.5, color="#52677D")
    fig.tight_layout(rect=(0.03, 0.08, 0.98, 0.97))
    save_figure(fig, "08_模拟测试分组耗时构成")


def create_response_composition(runs: list[dict]) -> None:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["summary"]["sources"]].append(run)
    source_counts = sorted(grouped)
    categories = [
        ("无信号", "no_signal", "#AAB8C6"),
        ("方向响应", "direction", base.BLUE),
        ("近距响应", "near", base.ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 7.0), facecolor="white")
    x = np.arange(len(source_counts))
    bottom = np.zeros(len(source_counts))
    for label, key, color in categories:
        values = np.array([
            np.mean([
                run["summary"]["signal_counts"].get(key, 0) / run["summary"]["sources"]
                for run in grouped[n]
            ])
            for n in source_counts
        ])
        ax.bar(x, values, bottom=bottom, width=0.67, color=color, label=label)
        bottom += values
    ax.set_ylim(0, max(bottom) * 1.12)
    ax.set_xticks(x, [f"{n}源\n(n={len(grouped[n])})" for n in source_counts])
    ax.set_ylabel("平均响应次数 / (次/源)")
    ax.set_title("不同干扰源数量下的测向响应构成", fontsize=17, fontweight="bold", pad=14)
    ax.grid(axis="y", color=base.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, 0.018, "响应类别直接来自 result.json 的信号记录；各次数均先除以该案例干扰源数量，再按组取平均。", ha="center", fontsize=9.5, color="#52677D")
    fig.tight_layout(rect=(0.03, 0.08, 0.98, 0.97))
    save_figure(fig, "09_模拟测试测向响应构成")


def create_distance_relation(runs: list[dict], selected: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0), facecolor="white")
    sources = np.array([run["summary"]["sources"] for run in runs])
    movement = np.array([run["summary"]["movement_m"] / 1000 for run in runs])
    efficiency = np.array([run["summary"]["per_source_s"] for run in runs])
    scatter = ax.scatter(movement, efficiency, c=sources, cmap="viridis", s=78, edgecolor="white", linewidth=0.8)
    selected_names = {run["name"] for run in selected}
    for run in runs:
        if run["name"] in selected_names:
            ax.scatter(run["summary"]["movement_m"] / 1000, run["summary"]["per_source_s"], marker="*", s=180, facecolors="none", edgecolors=base.RED, linewidths=1.4)
    ax.axhline(200, color=base.RED, linewidth=1.2, linestyle="--")
    ax.set_xlabel("机器人累计移动距离 / km")
    ax.set_ylabel("平均定位清除时间 / (s/源)")
    ax.set_title("移动距离与平均清除效率的关系", fontsize=17, fontweight="bold", pad=14)
    ax.grid(color=base.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("干扰源数量 / 个")
    ax.legend(handles=[Line2D([0], [0], marker="*", color="none", markerfacecolor="none", markeredgecolor=base.RED, markersize=11, label="各源数量最佳案例")], frameon=False, loc="upper right")
    fig.text(0.5, 0.018, "每个点对应一次完整模拟测试；红色空心星标出 10–16 源各组中平均耗时最短的案例。", ha="center", fontsize=9.5, color="#52677D")
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.97))
    save_figure(fig, "10_移动距离与清除效率关系")


def create_individual_routes(selected: list[dict]) -> None:
    for index, run in enumerate(selected, start=11):
        sources = run["summary"]["sources"]
        fig, ax = plt.subplots(figsize=(8.2, 8.2), facecolor="white")
        base.draw_route(ax, run, compact=False)
        fig.legend(handles=base.legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8.7, bbox_to_anchor=(0.5, 0.015))
        fig.text(0.5, 0.965, f"正式测试前模拟日志｜日志编号：{run['short_id']}｜该源数量组内平均耗时最低", ha="center", fontsize=9, color="#52677D")
        fig.tight_layout(rect=(0.03, 0.075, 0.98, 0.95))
        save_figure(fig, f"{index:02d}_{sources}源代表案例轨迹")


def main() -> None:
    runs = load_runs()
    selected = representative_runs(runs)
    create_dashboard(selected)
    create_efficiency_scatter(runs, selected)
    create_group_time_breakdown(runs)
    create_response_composition(runs)
    create_distance_relation(runs, selected)
    create_individual_routes(selected)
    print(f"19 次模拟测试，选取 {len(selected)} 个代表案例。")
    print(f"科研结果目录：{RESULT_DIR}")
    print(f"最终版图片目录：{FINAL_DIR}")


if __name__ == "__main__":
    main()

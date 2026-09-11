"""Shared publication-grade colors and Matplotlib defaults for the paper."""

from __future__ import annotations

from matplotlib import pyplot as plt
from PIL import Image


PALETTE = {
    "coral": "#F39DA0",
    "red": "#E84445",
    "orange": "#FD763F",
    "gold": "#EECA40",
    "pale_yellow": "#EEF0A7",
    "teal": "#1999B2",
    "cyan": "#23BAC5",
    "blue": "#95BCE5",
    "light_blue": "#B8E5FA",
    "lavender": "#CAC8EF",
    "pink": "#F7B7D2",
    "green": "#B2DBB9",
    "ink": "#2B2B2B",
    "gray": "#8C8C8C",
    "grid": "#D9D9D9",
    "paper": "#FCFCFA",
}


def apply_publication_style() -> None:
    """Apply a restrained journal theme shared by all scientific figures."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def add_panel_label(axis, label: str) -> None:
    """Add a bold panel label in the upper-left margin."""

    axis.text(
        -0.10,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        va="bottom",
        ha="left",
    )


def style_axis(axis, *, grid: bool = True) -> None:
    """Use subtle major-grid and inward ticks without visual clutter."""

    axis.tick_params(direction="in", length=3.2, width=0.8)
    if grid:
        axis.grid(True, which="major", color=PALETTE["grid"], alpha=0.42, linewidth=0.6)
        axis.set_axisbelow(True)


def save_publication_figure(figure, png_path, *, dpi: int = 400) -> None:
    """Save high-resolution, vector, and grayscale-review figure files."""

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=dpi)
    figure.savefig(png_path.with_suffix(".pdf"))
    figure.savefig(png_path.with_suffix(".svg"))
    grayscale_path = png_path.with_name(f"{png_path.stem}_grayscale.png")
    with Image.open(png_path) as rendered:
        rendered.convert("L").save(grayscale_path)

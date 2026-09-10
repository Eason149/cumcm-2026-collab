"""Shared publication-style colors and Matplotlib defaults for the paper."""

from __future__ import annotations

from matplotlib import pyplot as plt


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
    """Apply a restrained, journal-style theme shared by all figures."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def add_panel_label(axis, label: str) -> None:
    """Add a bold panel label in the upper-left margin."""

    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        color=PALETTE["ink"],
        va="bottom",
        ha="left",
    )

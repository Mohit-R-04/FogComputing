"""
One visual system for every figure in the project.

Colours follow the entity, never its rank: the proposed BBN is always slot-1 blue, each
baseline keeps its own slot in every chart, and when the story is "the BBN against the
rest" the rest recede to neutral grey instead of competing in eight hues. The eight
categorical slots are used in their validated order (adjacent CVD separation and
normal-vision floors pass; three light slots sit below 3:1 contrast on the surface, so
every chart that uses them carries direct labels and a table twin in its report).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RECEDE = "#c3c2b7"

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

#: Entity colours, fixed across every chart.
MODEL_COLORS = {
    "BBN (proposed)": SLOTS[0],
    "Gradient boosting": SLOTS[1],
    "Random forest": SLOTS[2],
    "Neural network (MLP)": SLOTS[3],
    "Logistic regression": SLOTS[4],
    "Naive Bayes": SLOTS[5],
    "Health-index threshold (FTAPA)": SLOTS[6],
    "CPU threshold": SLOTS[7],
    # Variants of the proposed model share its hue family; the ablation recedes.
    "BBN (elicited, no failure data)": "#86b6ef",
    "BBN + RF fusion (ablation)": MUTED,
}

POLICY_COLORS = {
    "bbn": SLOTS[0],
    "ml": SLOTS[1],
    "threshold": SLOTS[6],
    "reactive": SLOTS[2],
    "no-ft": SLOTS[7],
    "cloud-only": SLOTS[3],
    "oracle": MUTED,
}


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def emphasis(names, highlight="BBN (proposed)"):
    """Bar colours for a one-metric chart: the highlighted entity in its hue, others grey."""
    return [MODEL_COLORS.get(n, SLOTS[0]) if n == highlight else RECEDE for n in names]


def caption(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=8, color=MUTED, ha="left", va="bottom")

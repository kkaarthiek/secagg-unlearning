"""Plain matplotlib helpers for the ablation figures: error bars over seeds, a fixed colour per series."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOURS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"]    # colour-blind safe
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED = "#1a1a1a", "#6b6b6b"


def style(ax, xlabel, ylabel, title=""):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=11, pad=10)
    ax.grid(True, color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def line(ax, x, mean, spread, label, index):
    ax.errorbar(x, mean, yerr=spread, label=label, color=COLOURS[index % 5], marker=MARKERS[index % 5],
                markersize=5, linewidth=2, capsize=3, elinewidth=1)


def reference(ax, y, text):
    ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(text, (ax.get_xlim()[0], y), textcoords="offset points", xytext=(4, 4), color=MUTED, fontsize=8)


def save(fig, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(Path(folder) / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def mean_and_spread(values):
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0), values.std(axis=0)

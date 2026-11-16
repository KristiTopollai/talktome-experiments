"""Colors and small helpers shared by the plot scripts. Figures go to runs/figures/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import matplotlib

matplotlib.use("Agg")                                         # draw into files; no screen needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from common import FIGURE_DIR  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # blue, orange, green, yellow
DARK, GRAY, LIGHT = "#222222", "#888888", "#dddddd"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": LIGHT, "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10,
                     "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.8,
                     "lines.markersize": 3})


def epoch_axis(ax):
    """ImageNet epoch on the x-axis. Most change happens early, so the axis is linear up to 0.01 epochs and
    logarithmic above ("symlog"); the first checkpoint (epoch 0) is the untrained network."""
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.25)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 1, 10, 90]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("ImageNet training epoch")


def select(rows: list[dict], **conditions) -> list[dict]:
    """Rows whose columns equal the given values, e.g. select(rows, label_set="verb_with_noun")."""
    return [r for r in rows if all(r[k] == v for k, v in conditions.items())]


def column(rows: list[dict], key: str) -> list:
    return [r[key] for r in rows]


def save(fig, name: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURE_DIR / name}")

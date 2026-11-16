"""Task 3: sessions to mastery along ImageNet training, and example learning curves at the last checkpoint.
Outputs: runs/figures/task3_mastery.png, runs/figures/task3_example_curves.png
Usage: python plots/plot_task3.py
"""
import numpy as np
from plot_style import COLORS, DARK, column, epoch_axis, plt, save, select

from common import LABEL_SETS, RESULTS_DIR, list_checkpoints, read_csv

rows = read_csv(RESULTS_DIR / "task3_mastery.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, key, title in zip(axes, ("mastered_within_10", "mastered_within_50"),
                          ("targets mastered within 10 sessions", "targets mastered within 50 sessions")):
    for color, label_set in zip(COLORS, LABEL_SETS):
        r = select(rows, label_set=label_set)
        ax.plot(column(r, "epoch"), column(r, key), marker="o", color=color, label=label_set)
    epoch_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
axes[0].set_ylabel("fraction of targets")
save(fig, "task3_mastery.png")

# Example curves: test accuracy after each session for 4 mastered targets (fastest, two in between, slowest).
last = list_checkpoints()[-1]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, label_set in zip(axes, LABEL_SETS):
    z = np.load(RESULTS_DIR / "task3" / label_set / f"{last['name']}.npz")
    stm, curves, programs, classes = z["sessions_to_mastery"], z["test_accuracy"], z["programs"], z["classes"]
    mastered = sorted((stm[p, k], p, k) for p, k in zip(*np.where(stm > 0)))
    picks = [mastered[i] for i in (0, len(mastered) // 3, 2 * len(mastered) // 3, -1)] if len(mastered) >= 4 else mastered
    for color, (s, p, k) in zip(COLORS, picks):
        y = curves[p, k][~np.isnan(curves[p, k])]
        ax.plot(np.arange(1, len(y) + 1), y, marker="o", color=color, label=f"{classes[programs[p, k]]} (S = {s})")
    ax.axhline(0.8, color=DARK, linestyle="--", linewidth=1, label="mastery criterion 0.8")
    ax.set_xlabel("session")
    ax.set_ylim(0, 1)
    ax.set_title(f"{label_set}, epoch {last['epoch']:g}")
    ax.legend(loc="lower right")
axes[0].set_ylabel("test accuracy (unseen exemplars)")
save(fig, "task3_example_curves.png")

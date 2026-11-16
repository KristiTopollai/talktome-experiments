"""Task 2: action, object and both correct, for seen and for never-taught (novel) combinations.
Output: runs/figures/task2_new_combinations.png
Usage: python plots/plot_task2.py
"""
from plot_style import COLORS, DARK, column, epoch_axis, plt, save

from common import RESULTS_DIR, read_csv

rows = read_csv(RESULTS_DIR / "task2_new_combinations.csv")
epochs = column(rows, "epoch")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, kind, title in zip(axes, ("seen", "novel"),
                           ("seen combinations (new exemplars)", "novel combinations (never taught together)")):
    ax.plot(epochs, column(rows, f"{kind}_object"), marker="o", color=COLORS[0], label="object correct")
    ax.plot(epochs, column(rows, f"{kind}_action"), marker="o", color=COLORS[1], label="action correct")
    ax.plot(epochs, column(rows, f"{kind}_both"), marker="o", color=DARK, label="both correct")
    epoch_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
axes[0].set_ylabel(f"accuracy (mean over {int(rows[0]['n_cells'])} combinations)")
save(fig, "task2_new_combinations.png")

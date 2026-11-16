"""Task 4: taught with photos, tested on videos, along ImageNet training.
Output: runs/figures/task4_photo_to_video.png
Usage: python plots/plot_task4.py
"""
from plot_style import COLORS, DARK, column, epoch_axis, plt, save, select

from common import LABEL_SETS, RESULTS_DIR, read_csv

rows = read_csv(RESULTS_DIR / "task4_photo_to_video.csv")
lines = [("photo_to_photo", DARK, "photo -> photo (reference)"), ("photo_to_video_k=all", COLORS[0], "all photos -> video"),
         ("photo_to_video_k=4", COLORS[3], "4 photos per class -> video"),
         ("photo_to_video_k=2", COLORS[1], "2 photos per class -> video"),
         ("photo_to_video_k=1", COLORS[2], "1 photo per class -> video")]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, label_set in zip(axes, LABEL_SETS):
    r = select(rows, label_set=label_set)
    for key, color, label in lines:
        ax.plot(column(r, "epoch"), column(r, key), marker="o", color=color, label=label)
    epoch_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_title(label_set)
    ax.legend(loc="upper left")
axes[0].set_ylabel("macro accuracy (mean over classes)")
save(fig, "task4_photo_to_video.png")

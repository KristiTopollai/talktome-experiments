"""Task 0: per-class accuracy along ImageNet training (one figure per read-out and label set).
Left : heatmap, one row per class (sorted by acquisition epoch), one column per checkpoint.
Right: every class's accuracy (gray), the mean over classes (black) and three example classes (colors).
Output: runs/figures/task0_<method>_<label_set>.png
Usage: python plots/plot_task0.py
"""
import numpy as np
from plot_style import COLORS, DARK, LIGHT, epoch_axis, plt, save

from common import LABEL_SETS, RESULTS_DIR, read_csv

for method in ("probe", "finetune"):
    for label_set in LABEL_SETS:
        path = RESULTS_DIR / f"task0_{method}" / f"per_class_accuracy_{label_set}.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        order = [r["class"] for r in read_csv(RESULTS_DIR / f"task0_{method}" / f"acquisition_{label_set}.csv")]
        epochs = np.array([r["epoch"] for r in rows])
        acc = np.array([[r[c] for c in order] for r in rows])          # [checkpoints, classes]
        chance = 1 / len(order)

        fig, (left, right) = plt.subplots(1, 2, figsize=(12, max(4, 0.09 * len(order) + 1.5)),
                                          gridspec_kw={"width_ratios": [1.3, 1]})
        image = left.imshow(acc.T, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ticks = sorted({int(np.argmin(np.abs(epochs - e))) for e in (0, 0.1, 1, 10, 90)})
        left.set_xticks(ticks, [f"{epochs[t]:.2g}" for t in ticks])
        left.set_xlabel("checkpoint (labelled by ImageNet epoch)")
        left.set_yticks(range(len(order)), order, fontsize=5 if len(order) > 60 else 7)
        left.grid(False)
        left.set_title("classes sorted by acquisition epoch (accuracy >= 0.5)")
        fig.colorbar(image, ax=left, fraction=0.04, label="accuracy")

        for j in range(len(order)):
            right.plot(epochs, acc[:, j], color=LIGHT, linewidth=0.8)
        examples = [0, len(order) // 2, len(order) - 1]                  # earliest, middle, latest acquired
        for color, j in zip(COLORS, examples):
            right.plot(epochs, acc[:, j], color=color, label=order[j])
        right.plot(epochs, acc.mean(1), color=DARK, linewidth=2.5, label="mean over classes")
        right.axhline(chance, color=DARK, linestyle=":", linewidth=1, label=f"chance (1/{len(order)})")
        epoch_axis(right)
        right.set_ylim(0, 1)
        right.set_ylabel("held-out accuracy of the class")
        right.legend(loc="upper left")
        right.set_title(f"{label_set}, {'linear probe' if method == 'probe' else 'fine-tuned'}")
        save(fig, f"task0_{method}_{label_set}.png")

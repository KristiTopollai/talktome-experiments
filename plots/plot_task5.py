"""Task 5: receptive ("touch <c>") and expressive ("what is this?") accuracy in a field of 3 (verb_with_noun).
Solid = condition; dashed = the random condition on the same targets (what the similar foil is compared with).
Output: runs/figures/task5_receptive_expressive_<method>.png
Usage: python plots/plot_task5.py
"""
from plot_style import COLORS, DARK, column, epoch_axis, plt, save, select

from common import RESULTS_DIR, read_csv

for method in ("probe", "finetune"):
    path = RESULTS_DIR / f"task5_receptive_expressive_{method}.csv"
    if not path.exists():
        continue
    rows = select(read_csv(path), label_set="verb_with_noun")
    if not rows:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, kind, title in zip(axes, ("receptive", "expressive"),
                               ("receptive: pick the photo for a label", "expressive: pick the label for a photo")):
        r = select(rows, condition="random")
        ax.plot(column(r, "epoch"), column(r, kind), marker="o", color=DARK, label="two unrelated foils")
        for color, cond in zip(COLORS, ("same object", "same action")):
            r = select(rows, condition=cond)
            ax.plot(column(r, "epoch"), column(r, kind), marker="o", color=color, label=f"one {cond} foil")
            r = select(rows, condition=f"random | {cond} targets")
            ax.plot(column(r, "epoch"), column(r, kind), color=color, linestyle="--", linewidth=1,
                    label=f"(dashed) two unrelated foils, same targets as '{cond}'")
        ax.axhline(1 / 3, color=DARK, linestyle=":", linewidth=1, label="chance 1/3")
        epoch_axis(ax)
        ax.set_ylim(0.2, 1)
        ax.set_title(title)
        ax.legend(loc="upper left")
    axes[0].set_ylabel(f"accuracy, field of 3 ({method})")
    save(fig, f"task5_receptive_expressive_{method}.png")

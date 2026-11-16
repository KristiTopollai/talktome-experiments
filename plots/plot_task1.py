"""Task 1: do wrong answers keep the object or the action? Enrichment over chance along ImageNet training.
Output: runs/figures/task1_error_structure_<method>.png
Usage: python plots/plot_task1.py
"""
from plot_style import COLORS, DARK, column, epoch_axis, plt, save

from common import RESULTS_DIR, read_csv

for method in ("probe", "finetune"):
    path = RESULTS_DIR / f"task1_error_structure_{method}.csv"
    if not path.exists():
        continue
    rows = read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(column(rows, "epoch"), column(rows, "E_same_object"), marker="o", color=COLORS[0],
            label='E_O: same object, other action ("pour milk")')
    ax.plot(column(rows, "epoch"), column(rows, "E_same_action"), marker="o", color=COLORS[1],
            label='E_V: same action, other object ("drink juice")')
    ax.axhline(1, color=DARK, linestyle=":", linewidth=1, label="1 = no preference")
    epoch_axis(ax)
    ax.set_ylabel("share of wrong-answer probability / chance")
    ax.set_title(f"Which wrong answers? verb_with_noun, {'linear probe' if method == 'probe' else 'fine-tuned'}")
    ax.legend()
    save(fig, f"task1_error_structure_{method}.png")

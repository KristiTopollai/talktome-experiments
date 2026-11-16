"""Task 6: retention of mastered targets A while new targets B are taught.
Left : retention rho = A after 10 B sessions / A at the end of phase A, along ImageNet training (verb_with_noun).
Right: A's test accuracy through phase B at the last checkpoint (session 0 = end of phase A).
Output: runs/figures/task6_maintenance.png
Usage: python plots/plot_task6.py
"""
import json

import numpy as np
from plot_style import COLORS, column, epoch_axis, plt, save, select

from common import RESULTS_DIR, list_checkpoints, read_csv

CONDITIONS = [("unrelated", "none"), ("unrelated", "with"), ("similar", "none"), ("similar", "with")]
STYLE = {c: dict(color=COLORS[i // 2], linestyle="-" if c[1] == "with" else "--") for i, c in enumerate(CONDITIONS)}
LABEL = {c: f"B {c[0]}, {'with' if c[1] == 'with' else 'no'} maintenance" for c in CONDITIONS}

rows = select(read_csv(RESULTS_DIR / "task6_retention.csv"), label_set="verb_with_noun")
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
for c in CONDITIONS:
    r = select(rows, B=c[0], maintenance=c[1])
    left.plot(column(r, "epoch"), column(r, "retention"), marker="o", label=LABEL[c], **STYLE[c])
epoch_axis(left)
left.set_ylim(0, 1)
left.set_ylabel("retention of A")
left.set_title("retention after 10 sessions of B")
left.legend(loc="lower right")

last = list_checkpoints()[-1]
z = np.load(RESULTS_DIR / "task6" / "verb_with_noun" / f"{last['name']}.npz")
conditions = json.loads(str(z["conditions"]))
for c in CONDITIONS:
    sel = [j for j, (b, m, _) in enumerate(conditions) if b == c[0] and m == (c[1] == "with")]
    curve = np.concatenate([[z["A_end"][sel].mean()], z["A_curve"][sel].mean((0, 1))])
    right.plot(np.arange(len(curve)), curve, marker="o", label=LABEL[c], **STYLE[c])
right.set_xlabel("session of phase B (0 = end of phase A)")
right.set_ylim(0, 1)
right.set_ylabel("test accuracy of the A targets")
right.set_title(f"through phase B, epoch {last['epoch']:g}")
save(fig, "task6_maintenance.png")

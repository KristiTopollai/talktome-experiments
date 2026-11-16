"""ImageNet validation top-1 accuracy of every checkpoint (read from the feature files of 2_extract_features.py).
Output: runs/figures/imagenet_accuracy.png
"""
from plot_style import COLORS, epoch_axis, plt, save

from common import list_checkpoints

checkpoints = list_checkpoints()
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot([c["epoch"] for c in checkpoints], [c["val_top1"] for c in checkpoints], marker="o", color=COLORS[0])
epoch_axis(ax)
ax.set_ylim(0, 1)
ax.set_ylabel("top-1 accuracy (50,000 validation images)")
ax.set_title(f"ResNet-50 on ImageNet: {len(checkpoints)} checkpoints")
save(fig, "imagenet_accuracy.png")

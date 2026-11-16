"""Task 4 - taught with photos, tested on videos (generalization across materials; frozen features).

In therapy a concept is often taught with flash cards and should then be recognized in the real world. Here:
  classes   : classes that have at least one photo and at least one video
  read-out  : one linear score per class, fit by ridge regression on one-hot targets
                s(x) = W [x, 1],   W = argmin ||[X 1] W - Y||^2 + lam ||W||^2,   answer = argmax_c s_c(x)
              trained on k photos per class (k = 1, 2, 4 or all; each photo also mirrored, as a second row)
  test      : every video (mean feature of its frames) is labeled
  score     : macro accuracy = mean over classes of the fraction of the class's videos labeled correctly;
              for k = 1, 2, 4 averaged over 10 random draws of the k photos
  reference : photo -> photo, i.e. all photos with a 5-fold split (train on 4 folds, label the photos of the 5th)
lam = alpha x (number of exemplars); alpha in {0.01, 0.1, 1, 10} is chosen by the photo -> photo score on the last
checkpoint and then used for every checkpoint.
Output: runs/results/task4_photo_to_video.csv (one row per label set and checkpoint).
Usage: python tasks/task4_photo_to_video.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np

from common import (LABEL_SETS, RESULTS_DIR, list_checkpoints, load_exemplars, one_hot, ridge_predict,
                    ridge_regression, stratified_folds, write_csv)

ALPHAS = [0.01, 0.1, 1.0, 10.0]
K_PHOTOS = [1, 2, 4, "all"]
REPEATS = 10


def macro_accuracy(pred, true) -> float:
    return float(np.mean([np.mean(pred[true == c] == c) for c in np.unique(true)]))


class PhotoToVideo:
    """Everything that stays the same across checkpoints: which classes, which photo draws, which folds."""

    def __init__(self, ex, seed=0):
        both = sorted(set(ex.label[~ex.is_video]) & set(ex.label[ex.is_video]))
        new_label = {c: i for i, c in enumerate(both)}
        keep = np.isin(ex.label, both)
        self.y = np.array([new_label.get(c, -1) for c in ex.label])    # class index among `both` (-1 = not used)
        self.photos = np.where(keep & ~ex.is_video)[0]
        self.videos = np.where(keep & ex.is_video)[0]
        self.n_classes = len(both)
        rng = np.random.default_rng(seed)
        photos_of = [self.photos[self.y[self.photos] == c] for c in range(self.n_classes)]
        self.draws = {k: [np.concatenate([p if k == "all" else rng.permutation(p)[:k] for p in photos_of])
                          for _ in range(1 if k == "all" else REPEATS)] for k in K_PHOTOS}
        self.fold = stratified_folds(self.y[self.photos], 5, seed)

    def fit_predict(self, ex, train, test, lam):
        X = np.concatenate([ex.X[train], ex.X_flipped[train]])        # photos + their mirror images
        y = np.concatenate([self.y[train], self.y[train]])
        W = ridge_regression(X, one_hot(y, self.n_classes), lam)
        return ridge_predict(W, ex.X[test]).argmax(1)

    def evaluate(self, ex, alpha) -> dict:
        lam = alpha * len(ex.X)
        out = {}
        for k in K_PHOTOS:
            out[f"photo_to_video_k={k}"] = float(np.mean(
                [macro_accuracy(self.fit_predict(ex, train, self.videos, lam), self.y[self.videos])
                 for train in self.draws[k]]))
        pred = np.zeros(len(self.photos), int)
        for f in range(5):
            train, test = self.photos[self.fold != f], self.fold == f
            pred[test] = self.fit_predict(ex, train, self.photos[test], lam)
        out["photo_to_photo"] = macro_accuracy(pred, self.y[self.photos])
        return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    checkpoints = list_checkpoints()
    rows = []
    for label_set in LABEL_SETS:
        last = load_exemplars(checkpoints[-1], label_set)
        task = PhotoToVideo(last)
        alpha = max(ALPHAS, key=lambda a: task.evaluate(last, a)["photo_to_photo"])
        print(f"{label_set}: {task.n_classes} classes, {len(task.photos)} photos, {len(task.videos)} videos; "
              f"alpha {alpha} (chosen by photo -> photo on {checkpoints[-1]['name']})", flush=True)
        for ckpt in checkpoints:
            scores = task.evaluate(load_exemplars(ckpt, label_set), alpha)
            rows.append({"label_set": label_set, "checkpoint": ckpt["name"], "epoch": ckpt["epoch"], "alpha": alpha,
                         **scores})
            print(f"  epoch {ckpt['epoch']:7.3f}  " + "  ".join(f"{k} {v:.3f}" for k, v in scores.items()), flush=True)
    write_csv(RESULTS_DIR / "task4_photo_to_video.csv", rows)


if __name__ == "__main__":
    main()

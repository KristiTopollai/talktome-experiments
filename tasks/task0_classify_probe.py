"""Task 0 (base task) - classify TALKtoME photos with a linear read-out on frozen features.

Task: C-way classification of photos (C = 50 actions, or 111 action+object classes; classes with < 3 photos are
left out). For every checkpoint:
  read-out  : multinomial logistic regression on the 2048-d feature f(x):  p(c | x) = softmax(W f(x) + b)_c
              trained by minimizing  mean cross-entropy + (lam / 2) ||W||^2  on the training photos and their
              mirror images; features are standardized with the training mean and std.
  held out  : the photos are split into 5 folds (every class spread evenly); a read-out trained on 4 folds
              predicts the 5th, so every photo gets a prediction from a read-out that never saw it.
              Repeated for 3 random splits (seeds 0, 1, 2).
  lam       : chosen once per label set from {1e-4, 1e-3, 1e-2, 0.1, 1}: the value with the best held-out
              accuracy (seed 0) on the last checkpoint; then used for every checkpoint. (Accuracy changes little
              with lam, and choosing it once makes the script ~15x faster.)
  score     : per-class accuracy acc_c = fraction of class-c photos whose most probable class is c (seeds pooled);
              acquisition epoch of a class = first epoch where acc_c (averaged over 3 neighbouring checkpoints)
              reaches 0.5. Chance = 1 / C.
Outputs (runs/results/task0_probe/):
  <label_set>/<checkpoint>.npz   prob [seeds, photos, classes] = held-out p(c | photo), labels, classes
  per_class_accuracy_<label_set>.csv, acquisition_<label_set>.csv
Time per checkpoint: under a minute on a laptop CPU, a few seconds on a GPU (used automatically if available).
Usage: python tasks/task0_classify_probe.py [--label-sets verb_with_noun]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np
import torch
import torch.nn.functional as F

from common import (LABEL_SETS, RESULTS_DIR, classification_photos, get_device, list_checkpoints,
                    load_classification_results, load_photo_features, photo_folds, save_npz, write_csv)

SEEDS = [0, 1, 2]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def train_logistic_regression(X, y, n_classes, lam):
    """Minimize mean cross-entropy(softmax(X W + b), y) + lam/2 ||W||^2 with L-BFGS. Returns (W, b)."""
    W = torch.zeros(X.shape[1], n_classes, device=X.device, requires_grad=True)
    b = torch.zeros(n_classes, device=X.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([W, b], max_iter=300, tolerance_grad=1e-6, tolerance_change=1e-9,
                                  history_size=20, line_search_fn="strong_wolfe")

    def loss_fn():
        optimizer.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + 0.5 * lam * (W * W).sum()
        loss.backward()
        return loss

    optimizer.step(loss_fn)
    return W.detach(), b.detach()


def fit_and_predict(feats, flipped, y, train, test, n_classes, lam):
    """Train on photos `train` (+ their mirror images), return p(c | x) for photos `test`."""
    X = torch.cat([feats[train], flipped[train]])
    mean, std = X.mean(0), X.std(0) + 1e-6
    W, b = train_logistic_regression((X - mean) / std, torch.cat([y[train], y[train]]), n_classes, lam)
    return torch.softmax((feats[test] - mean) / std @ W + b, dim=1)


def heldout_probabilities(feats, flipped, y, n_classes, seed, lam):
    """[photos, classes]: every photo's p(c | x) from a read-out trained on the other 4 folds."""
    fold = photo_folds(y.cpu().numpy(), seed)
    prob = torch.zeros(len(y), n_classes, device=feats.device)
    for k in range(fold.max() + 1):
        train, test = np.where(fold != k)[0], np.where(fold == k)[0]
        prob[test] = fit_and_predict(feats, flipped, y, train, test, n_classes, lam)
    return prob.cpu().numpy()


def choose_lambda(feats, flipped, y, n_classes):
    """The lam with the best held-out accuracy (seed 0); ties go to the larger lam."""
    labels = y.cpu().numpy()
    accuracy = [(heldout_probabilities(feats, flipped, y, n_classes, 0, lam).argmax(1) == labels).mean()
                for lam in LAMBDAS]
    print("    held-out accuracy per lam: " + ", ".join(f"{lam:g}: {a:.3f}" for lam, a in zip(LAMBDAS, accuracy)))
    return LAMBDAS[len(LAMBDAS) - 1 - int(np.argmax(accuracy[::-1]))]


def write_tables(method: str, label_set: str):
    """Per-class accuracy at every checkpoint, and the epoch at which each class is acquired (acc >= 0.5)."""
    results = load_classification_results(method, label_set)
    if not results:
        return
    classes, labels = results[0]["classes"], results[0]["labels"]
    rows = []
    for r in results:
        correct = r["prob"].argmax(-1) == labels                  # [seeds, photos]
        per_class = [correct[:, labels == c].mean() for c in range(len(classes))]
        rows.append({"checkpoint": r["name"], "epoch": r["epoch"], "mean_accuracy": float(np.mean(per_class)),
                     **{name: float(a) for name, a in zip(classes, per_class)}})
    out = RESULTS_DIR / f"task0_{method}"
    write_csv(out / f"per_class_accuracy_{label_set}.csv", rows)

    acc = np.array([[row[c] for c in classes] for row in rows])   # [checkpoints, classes]
    smooth = np.array([acc[max(0, t - 1):t + 2].mean(0) for t in range(len(acc))])   # 3-checkpoint mean
    epochs = [row["epoch"] for row in rows]
    acquisition = []
    for c, name in enumerate(classes):
        hit = np.where(smooth[:, c] >= 0.5)[0]
        acquisition.append({"class": name, "n_photos": int((labels == c).sum()),
                            "acquisition_epoch": epochs[hit[0]] if len(hit) else "", "final_accuracy": acc[-1, c]})
    acquisition.sort(key=lambda r: (r["acquisition_epoch"] == "", r["acquisition_epoch"] or 0))
    write_csv(out / f"acquisition_{label_set}.csv", acquisition)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-sets", nargs="+", default=LABEL_SETS)
    args = ap.parse_args()
    device = get_device()
    torch.set_flush_denormal(True)   # treat tiny numbers (< 1e-38) as 0: otherwise L-BFGS on a CPU can be ~30x slower
    checkpoints = list_checkpoints()

    for label_set in args.label_sets:
        photos, labels, classes = classification_photos(label_set)
        y = torch.as_tensor(labels, device=device)
        out_dir = RESULTS_DIR / "task0_probe" / label_set
        todo = [c for c in checkpoints if not (out_dir / f"{c['name']}.npz").exists()]
        print(f"{label_set}: {len(classes)} classes, {len(photos)} photos, {len(todo)} checkpoints to do", flush=True)
        if not todo:
            continue

        def photo_features(ckpt):
            return (torch.as_tensor(a[photos], device=device) for a in load_photo_features(ckpt, label_set))

        print(f"  choosing lam on the last checkpoint ({checkpoints[-1]['name']})", flush=True)
        lam = choose_lambda(*photo_features(checkpoints[-1]), y, len(classes))
        print(f"  lam = {lam:g}", flush=True)
        for ckpt in todo:
            t0 = time.time()
            feats, flipped = photo_features(ckpt)
            prob = np.stack([heldout_probabilities(feats, flipped, y, len(classes), seed, lam) for seed in SEEDS])
            save_npz(out_dir / f"{ckpt['name']}.npz", prob=prob.astype(np.float16), labels=labels,
                     classes=np.array(classes), epoch=ckpt["epoch"], lam=lam)
            accuracy = (prob.argmax(-1) == labels).mean()
            print(f"  {ckpt['name']} epoch {ckpt['epoch']:7.3f}: accuracy {accuracy:.3f} ({time.time() - t0:.0f} s)",
                  flush=True)
        write_tables("probe", label_set)


if __name__ == "__main__":
    main()

"""Task 2 - new combinations of known parts ("matrix training"; verb_with_noun, frozen features).

Matrix training teaches some action-object combinations ("open door", "close box") and tests whether a child can
label a combination never taught together ("open box"). Here:
  exemplars : every photo and video (a video = mean feature of its frames), standardized features x
  read-out  : two linear heads on the same features, both fit by ridge regression on one-hot targets
                action head  s_A(x) = W_A [x, 1]   (one score per action, 49 actions)
                object head  s_O(x) = W_O [x, 1]   (one score per object, 100 objects)
              W = argmin ||[X 1] W - Y||^2 + lam ||W||^2 ;  answer = (argmax s_A, argmax s_O)
  cells     : combinations (a, o) whose action also occurs with another object AND whose object also occurs with
              another action, so both parts can be learned elsewhere (49 cells)
  seen      : 5-fold split of the exemplars; the cell's other exemplars are in training
  novel     : leave-one-cell-out: ALL exemplars of the cell are removed from training, then the cell is tested
  score     : per cell, fraction of its exemplars with action correct, object correct, both correct;
              then averaged over cells (each cell counts equally)
lam = alpha x (number of exemplars); alpha in {0.01, 0.1, 1, 10} is chosen by the "seen, both correct" score on the
last checkpoint (never on the novel test) and then used for every checkpoint.
Outputs: runs/results/task2_new_combinations.csv (one row per checkpoint), task2_cells.csv (per cell).
Usage: python tasks/task2_new_combinations.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np

from common import RESULTS_DIR, list_checkpoints, load_exemplars, one_hot, stratified_folds, write_csv

ALPHAS = [0.01, 0.1, 1.0, 10.0]


def evaluate(ex, alpha) -> dict:
    """Scores of every eligible cell: {cell: {'seen_action': .., ..., 'novel_both': ..}}."""
    actions, objects = sorted(set(ex.actions)), sorted(set(ex.objects))
    action_of = np.array([actions.index(ex.actions[c]) for c in ex.label])   # action index of every exemplar
    object_of = np.array([objects.index(ex.objects[c]) for c in ex.label])
    Y = np.hstack([one_hot(action_of, len(actions)), one_hot(object_of, len(objects))])
    lam = alpha * len(ex.X)
    Xb = np.hstack([ex.X, np.ones((len(ex.X), 1))]).astype(np.float64)    # features + a constant 1 for the bias
    G, B = Xb.T @ Xb, Xb.T @ Y                                            # over ALL exemplars, computed once

    def train_and_test(test):
        # Ridge regression trained on every exemplar except those in `test`:
        #   W = (X_train^T X_train + lam I)^-1 X_train^T Y_train,  with  X_train^T X_train = X^T X - X_test^T X_test,
        # which is much cheaper than recomputing X_train^T X_train for each of the 54 fits.
        # One solve with both one-hot blocks as targets = the two heads fit separately (same lam).
        W = np.linalg.solve(G - Xb[test].T @ Xb[test] + lam * np.eye(len(G)), B - Xb[test].T @ Y[test])
        scores = Xb[test] @ W
        action_ok = scores[:, :len(actions)].argmax(1) == action_of[test]
        object_ok = scores[:, len(actions):].argmax(1) == object_of[test]
        return action_ok, object_ok

    C = len(ex.classes)
    cells = [c for c in range(C)
             if any(ex.actions[d] == ex.actions[c] for d in range(C) if d != c)
             and any(ex.objects[d] == ex.objects[c] for d in range(C) if d != c)]
    results = {c: {} for c in cells}

    # seen: 5-fold split over exemplars
    fold = stratified_folds(ex.label, 5, seed=0)
    action_ok, object_ok = np.zeros(len(ex.X), bool), np.zeros(len(ex.X), bool)
    for k in range(5):
        test = np.where(fold == k)[0]
        action_ok[test], object_ok[test] = train_and_test(test)
    for c in cells:
        mine = ex.label == c
        results[c].update(seen_action=action_ok[mine].mean(), seen_object=object_ok[mine].mean(),
                          seen_both=(action_ok & object_ok)[mine].mean())

    # novel: leave the whole cell out of training
    for c in cells:
        test = np.where(ex.label == c)[0]                 # all exemplars of the cell; train on all the others
        a_ok, o_ok = train_and_test(test)
        results[c].update(novel_action=a_ok.mean(), novel_object=o_ok.mean(), novel_both=(a_ok & o_ok).mean(),
                          n_exemplars=len(test))
    return results


def average(results: dict, key: str) -> float:
    return float(np.mean([r[key] for r in results.values()]))


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    checkpoints = list_checkpoints()
    last = load_exemplars(checkpoints[-1], "verb_with_noun")
    seen_score = {a: average(evaluate(last, a), "seen_both") for a in ALPHAS}
    alpha = max(ALPHAS, key=seen_score.get)
    print(f"alpha chosen on {checkpoints[-1]['name']}: {alpha}  (seen-both accuracy per alpha: {seen_score})", flush=True)

    keys = ["seen_action", "seen_object", "seen_both", "novel_action", "novel_object", "novel_both"]
    rows, cell_rows = [], []
    for ckpt in checkpoints:
        ex = load_exemplars(ckpt, "verb_with_noun")
        res = evaluate(ex, alpha)
        rows.append({"checkpoint": ckpt["name"], "epoch": ckpt["epoch"], "alpha": alpha, "n_cells": len(res),
                     **{k: average(res, k) for k in keys}})
        for c, r in res.items():
            cell_rows.append({"checkpoint": ckpt["name"], "epoch": ckpt["epoch"], "cell": ex.classes[c],
                              "action": ex.actions[c], "object": ex.objects[c], **r})
        print(f"epoch {ckpt['epoch']:7.3f}  " + "  ".join(f"{k} {rows[-1][k]:.3f}" for k in keys), flush=True)
    write_csv(RESULTS_DIR / "task2_new_combinations.csv", rows)
    write_csv(RESULTS_DIR / "task2_cells.csv", cell_rows)


if __name__ == "__main__":
    main()

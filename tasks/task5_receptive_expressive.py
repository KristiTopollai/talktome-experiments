"""Task 5 - receptive vs expressive identification in a field of 3, with similar distractors.

Uses the held-out probabilities p(c | x) of Task 0 (every photo, every seed). One trial has a target photo x of
class c and two distractor classes d1, d2:
  receptive  ("touch <c>")    : three photos - x and one photo each of d1 and d2. Correct if p(c | x) is larger
                                than p(c | x1) and p(c | x2): the learner picks the photo that best fits the label.
  expressive ("what is this?"): one photo x and three labels c, d1, d2. Correct if p(c | x) > p(d1 | x), p(d2 | x).
  (Ties share the credit: a 3-way tie at the top counts 1/3.) Chance = 1/3.
Distractor conditions (verb_with_noun; action_verbs has only "random"):
  random      : d1, d2 both unrelated to c (share neither its action nor its object)
  same object : d1 has c's object with another action ("pour milk" for "drink milk"), d2 unrelated
  same action : d1 has c's action with another object ("drink juice"), d2 unrelated
A related condition can only use targets that have such a class, so it is compared with
"random | <condition> targets" = the random condition on exactly the same targets.
Each photo gets 10 random trials per seed. Score = mean over trials (and seeds).
Output: runs/results/task5_receptive_expressive_<method>.csv (one row per label set, checkpoint and condition).
Usage: python tasks/task5_receptive_expressive.py [--method probe|finetune]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np

from common import (LABEL_SETS, RESULTS_DIR, class_relations, load_classification_results, split_class_name,
                    write_csv)

TRIALS_PER_PHOTO = 10


def best_of_three(t, d1, d2):
    """1 if t is the largest of (t, d1, d2); ties at the top share the credit."""
    top = np.maximum(np.maximum(t, d1), d2)
    n_top = (t == top).astype(float) + (d1 == top) + (d2 == top)
    return (t == top) / n_top


def make_trials(labels, unrelated, related=None, targets=None, seed=0):
    """Target photo and two distractor classes for every trial: (unrelated, unrelated), or (related, unrelated)."""
    rng = np.random.default_rng(seed)
    photo, d1, d2 = [], [], []
    for i, c in enumerate(labels):
        if (targets is not None and c not in targets) or (related is not None and len(related[c]) == 0):
            continue
        for _ in range(TRIALS_PER_PHOTO):
            if related is None:
                a, b = rng.choice(unrelated[c], 2, replace=False)
            else:
                a, b = rng.choice(related[c]), rng.choice(unrelated[c])
            photo.append(i)
            d1.append(a)
            d2.append(b)
    return np.array(photo), np.array(d1), np.array(d2)


def score(prob, labels, trials, seed=1):
    """prob [seeds, photos, C]. Returns (receptive, expressive) accuracy."""
    rng = np.random.default_rng(seed)
    photos_of = [np.where(labels == c)[0] for c in range(prob.shape[-1])]
    photo, d1, d2 = trials
    c = labels[photo]
    receptive, expressive = [], []
    for P in prob:                                         # one seed at a time
        x1 = np.array([rng.choice(photos_of[d]) for d in d1])   # a random photo of each distractor class
        x2 = np.array([rng.choice(photos_of[d]) for d in d2])
        receptive.append(best_of_three(P[photo, c], P[x1, c], P[x2, c]).mean())
        expressive.append(best_of_three(P[photo, c], P[photo, d1], P[photo, d2]).mean())
    return float(np.mean(receptive)), float(np.mean(expressive))


def conditions(labels, classes, label_set) -> dict:
    C = len(classes)
    if label_set != "verb_with_noun":
        return {"random": make_trials(labels, [np.array([d for d in range(C) if d != c]) for c in range(C)])}
    same_object, same_action = class_relations(classes)
    parts = [split_class_name(c) for c in classes]                      # (action, object) of every class
    unrelated = [np.array([d for d in range(C) if parts[d][0] != parts[c][0] and parts[d][1] != parts[c][1]])
                 for c in range(C)]
    trials = {"random": make_trials(labels, unrelated)}
    for name, relation in (("same object", same_object), ("same action", same_action)):
        related = [np.where(relation[c])[0] for c in range(C)]
        targets = {c for c in range(C) if len(related[c])}
        trials[name] = make_trials(labels, unrelated, related)
        trials[f"random | {name} targets"] = make_trials(labels, unrelated, targets=targets)
    return trials


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="probe", choices=["probe", "finetune"])
    args = ap.parse_args()
    rows = []
    for label_set in LABEL_SETS:
        results = load_classification_results(args.method, label_set)
        if not results:
            continue
        trials = conditions(results[0]["labels"], results[0]["classes"], label_set)
        for r in results:
            for name, t in trials.items():
                receptive, expressive = score(r["prob"], r["labels"], t)
                rows.append({"label_set": label_set, "checkpoint": r["name"], "epoch": r["epoch"], "condition": name,
                             "trials": len(t[0]), "receptive": receptive, "expressive": expressive})
                print(f"{label_set:15s} epoch {r['epoch']:7.3f}  {name:28s} receptive {receptive:.3f}  "
                      f"expressive {expressive:.3f}")
    write_csv(RESULTS_DIR / f"task5_receptive_expressive_{args.method}.csv", rows)


if __name__ == "__main__":
    main()

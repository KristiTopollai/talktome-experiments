"""Task 3 - discrete-trial teaching: how many sessions until a concept is mastered? (frozen features)

A simulated learner is taught the way targets are taught in therapy, on top of the frozen features of one checkpoint.
  exemplars : every photo and video; each class's exemplars are shuffled once and split into a teaching half
              (shown during sessions) and a test half (never shown, used only to test). Classes need >= 2 exemplars.
  learner   : p(. | x) = softmax(W x) over the labels taught so far. W starts at 0; teaching a new label adds a
              zero row. x = standardized feature / sqrt(2048), so the step size does not depend on the feature size.
  trial     : show a teaching exemplar x of target c; the learner answers (drawn at random from p), then is told
              the correct label: one gradient step on the cross-entropy  W <- W - eta (p - e_c) x^T,  eta = 3
              (e_c = one-hot vector of c; the step is the same whether the answer was right or wrong).
  session   : every target of the program gets M = 3 trials, all in random order (a new teaching exemplar each time).
  test      : after each session, without feedback: a_c = mean over the test exemplars x of c of p(c | x)
              (= the expected accuracy if the learner answers by drawing from p).
  mastery   : a_c >= 0.8 in two consecutive sessions; sessions-to-mastery S_c = number of the second one (>= 2).
              Teaching stops when all targets are mastered, or after 50 sessions ("not mastered").
  programs  : ceil(2.5 x classes / 5) random sets of K = 5 targets (each class is taught in ~2.5 programs);
              the split and the programs are the same at every checkpoint, so checkpoints are compared on equal terms.
Outputs: runs/results/task3/<label_set>/<checkpoint>.npz (all curves) and task3_mastery.csv
  (per checkpoint: fraction of targets mastered within 10 and within 50 sessions, median sessions to mastery
   where "not mastered" counts as 51).
Usage: python tasks/task3_teaching_to_mastery.py [--workers 4]     (one checkpoint per worker process; default: all CPUs)
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np

from common import LABEL_SETS, RESULTS_DIR, list_checkpoints, load_exemplars, save_npz, softmax, write_csv

K = 5                 # targets per program
M = 3                 # trials per target per session
ETA = 3.0             # step size of the learner
CRITERION = 0.8       # mastery: test accuracy >= 0.8 ...
MAX_SESSIONS = 50     # ... in two consecutive sessions, within 50 sessions


class Learner:
    """Linear softmax read-out over the labels taught so far, trained one trial at a time."""

    def __init__(self, n_features: int):
        self.W = np.zeros((0, n_features))
        self.labels: list[int] = []

    def add_labels(self, classes):
        new = [c for c in classes if c not in self.labels]
        self.labels += new
        self.W = np.vstack([self.W, np.zeros((len(new), self.W.shape[1]))])     # new labels start at score 0

    def trial(self, x: np.ndarray, c: int, rng) -> bool:
        """One teaching trial on exemplar x of class c. Returns whether the (random) answer was correct."""
        p = softmax(self.W @ x)
        k = self.labels.index(c)
        answer = rng.choice(len(p), p=p / p.sum())
        error = p.copy()
        error[k] -= 1                                    # p - e_c
        self.W -= ETA * np.outer(error, x)
        return answer == k

    def test(self, X_test: np.ndarray, c: int) -> float:
        """Expected accuracy on unseen exemplars of c: mean p(c | x) (no feedback, W unchanged)."""
        return float(softmax(X_test @ self.W.T)[:, self.labels.index(c)].mean())


def teach_test_split(label: np.ndarray, rng) -> tuple[dict, dict]:
    """teach[c], test[c]: exemplar indices of class c (first half of a random order is the test half)."""
    teach, test = {}, {}
    for c in np.unique(label):
        idx = rng.permutation(np.where(label == c)[0])
        if len(idx) >= 2:
            test[int(c)], teach[int(c)] = idx[:len(idx) // 2], idx[len(idx) // 2:]
    return teach, test


def run_session(learner, targets, X, teach, rng, extra=()) -> dict:
    """M trials per target plus one trial per class in `extra`, in random order. Returns % correct per class."""
    order = [c for c in targets for _ in range(M)] + list(extra)
    rng.shuffle(order)
    correct = {c: [] for c in order}
    for c in order:
        x = X[rng.choice(teach[c])]                      # a random teaching exemplar of c
        correct[c].append(learner.trial(x, c, rng))
    return {c: float(np.mean(v)) for c, v in correct.items()}


def teach_to_mastery(learner, targets, X, teach, test, rng):
    """Sessions until every target is mastered (or MAX_SESSIONS). Returns
    sessions_to_mastery [K] (-1 = not mastered), test_accuracy [K, MAX_SESSIONS], in_session [K, MAX_SESSIONS]
    (NaN after teaching stopped)."""
    learner.add_labels(targets)
    sessions_to_mastery = np.full(len(targets), -1)
    test_accuracy = np.full((len(targets), MAX_SESSIONS), np.nan)
    in_session = np.full((len(targets), MAX_SESSIONS), np.nan)
    for s in range(MAX_SESSIONS):
        correct = run_session(learner, targets, X, teach, rng)
        for j, c in enumerate(targets):
            in_session[j, s] = correct[c]
            test_accuracy[j, s] = learner.test(X[test[c]], c)
            if sessions_to_mastery[j] < 0 and s >= 1 and min(test_accuracy[j, s - 1:s + 1]) >= CRITERION:
                sessions_to_mastery[j] = s + 1           # sessions are counted from 1
        if (sessions_to_mastery > 0).all():
            break
    return sessions_to_mastery, test_accuracy, in_session


def simulate_checkpoint(job):
    ckpt, label_set = job
    out = RESULTS_DIR / "task3" / label_set / f"{ckpt['name']}.npz"
    if out.exists():
        return
    ex = load_exemplars(ckpt, label_set)
    X = ex.X / np.sqrt(ex.X.shape[1])
    rng = np.random.default_rng(12345)                   # fixed seed: the same split and programs at every checkpoint
    teach, test = teach_test_split(ex.label, rng)
    usable = sorted(teach)
    programs = [rng.choice(usable, K, replace=False) for _ in range(int(np.ceil(2.5 * len(usable) / K)))]
    results = [teach_to_mastery(Learner(X.shape[1]), list(targets), X, teach, test, np.random.default_rng(100 + i))
               for i, targets in enumerate(programs)]
    save_npz(out, programs=np.array(programs), classes=np.array(ex.classes), epoch=ckpt["epoch"],
             sessions_to_mastery=np.array([r[0] for r in results]),       # [programs, K]
             test_accuracy=np.array([r[1] for r in results]),             # [programs, K, MAX_SESSIONS]
             in_session=np.array([r[2] for r in results]))
    print(f"{label_set} {ckpt['name']}: done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    checkpoints = list_checkpoints()
    with Pool(args.workers) as pool:
        pool.map(simulate_checkpoint, [(c, s) for s in LABEL_SETS for c in checkpoints], chunksize=1)

    rows = []
    for label_set in LABEL_SETS:
        for ckpt in checkpoints:
            stm = np.load(RESULTS_DIR / "task3" / label_set / f"{ckpt['name']}.npz")["sessions_to_mastery"]
            rows.append({"label_set": label_set, "checkpoint": ckpt["name"], "epoch": ckpt["epoch"],
                         "mastered_within_10": float(np.mean((stm > 0) & (stm <= 10))),
                         "mastered_within_50": float(np.mean(stm > 0)),
                         "median_sessions": float(np.median(np.where(stm > 0, stm, MAX_SESSIONS + 1)))})
            print(f"{label_set:15s} epoch {ckpt['epoch']:7.3f}  mastered within 50: {rows[-1]['mastered_within_50']:.2f}"
                  f"  median sessions {rows[-1]['median_sessions']:g}")
    write_csv(RESULTS_DIR / "task3_mastery.csv", rows)


if __name__ == "__main__":
    main()

"""Task 6 - are mastered targets kept while new ones are taught? (maintenance and interference; frozen features)

Same learner, trials, sessions and test as Task 3 (imported from task3_teaching_to_mastery.py).
One run:
  phase A : teach 5 random targets A to mastery (or 50 sessions). a_A(end) = each A target's last test score.
  phase B : add 5 new targets B (5 new zero rows in W, so the learner now chooses among 10 labels) and teach
            them for 10 sessions of M = 3 trials per B target. After every B session, test the A targets.
Two factors, 2 x 2 conditions:
  B unrelated : no B target shares its action or its object with any A target
  B similar   : every B target shares its action or its object with some A target (verb_with_noun only)
  maintenance : every B session also has 1 trial per A target (mixed in at random); "none" = only B trials
24 random A sets; each A set is used in all conditions with the same random seed (paired comparison).
Score (retention):
  rho = mean a_A(after the 10th B session) / mean a_A(end of phase A)      (means over the 5 A targets and 24 runs)
  rho = 1: nothing lost. Part of the drop is mechanical: the 5 new labels already take some probability from the
  A labels before any B trial; the column A_after_adding_B_labels (test right after adding them) shows how much.
Outputs: runs/results/task6/<label_set>/<checkpoint>.npz (curves) and task6_retention.csv (one row per
checkpoint and condition).
Usage: python tasks/task6_maintenance.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np

from common import LABEL_SETS, RESULTS_DIR, list_checkpoints, load_exemplars, save_npz, write_csv
from task3_teaching_to_mastery import K, Learner, run_session, teach_test_split, teach_to_mastery

N_RUNS = 24          # random A sets
B_SESSIONS = 10      # sessions of phase B


def make_runs(ex, usable, seed=2) -> list[dict]:
    """For every run: the A targets and the unrelated / similar B targets (None if impossible)."""
    def related(a, b):
        return ex.actions[a] == ex.actions[b] or (ex.objects[a] is not None and ex.objects[a] == ex.objects[b])

    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(N_RUNS):
        A = list(rng.choice(usable, K, replace=False))
        rest = [c for c in usable if c not in A]
        unrelated = [c for c in rest if not any(related(a, c) for a in A)]
        similar = [c for c in rest if any(related(a, c) for a in A)]
        runs.append({"A": A, "unrelated": list(rng.choice(unrelated, K, replace=False)),
                     "similar": list(rng.choice(similar, K, replace=False)) if len(similar) >= K else None})
    return runs


def run_condition(X, teach, test, A, B, maintenance, seed):
    """Phase A to mastery, then phase B. Returns A at the end of phase A [K], A right after adding the B labels [K],
    A after every B session [K, B_SESSIONS], B after every B session [K, B_SESSIONS]."""
    rng = np.random.default_rng(seed)
    learner = Learner(X.shape[1])
    _, test_A, _ = teach_to_mastery(learner, A, X, teach, test, rng)
    A_end = np.array([row[~np.isnan(row)][-1] for row in test_A])        # last test score of each A target
    learner.add_labels(B)
    A_after_adding = np.array([learner.test(X[test[c]], c) for c in A])
    A_curve, B_curve = np.zeros((K, B_SESSIONS)), np.zeros((K, B_SESSIONS))
    for s in range(B_SESSIONS):
        run_session(learner, B, X, teach, rng, extra=A if maintenance else ())
        A_curve[:, s] = [learner.test(X[test[c]], c) for c in A]
        B_curve[:, s] = [learner.test(X[test[c]], c) for c in B]
    return A_end, A_after_adding, A_curve, B_curve


def simulate_checkpoint(job):
    ckpt, label_set = job
    out = RESULTS_DIR / "task6" / label_set / f"{ckpt['name']}.npz"
    if out.exists():
        return
    ex = load_exemplars(ckpt, label_set)
    X = ex.X / np.sqrt(ex.X.shape[1])
    teach, test = teach_test_split(ex.label, np.random.default_rng(12345))   # the same split as Task 3
    runs = make_runs(ex, sorted(teach))
    conditions, results = [], []
    for i, run in enumerate(runs):
        for b_kind in ("unrelated", "similar"):
            if run[b_kind] is None or (b_kind == "similar" and label_set != "verb_with_noun"):
                continue
            for maintenance in (False, True):
                conditions.append([b_kind, maintenance, i])
                results.append(run_condition(X, teach, test, run["A"], run[b_kind], maintenance, seed=1000 + i))
    save_npz(out, conditions=json.dumps(conditions), epoch=ckpt["epoch"],
             A_end=np.array([r[0] for r in results]), A_after_adding=np.array([r[1] for r in results]),
             A_curve=np.array([r[2] for r in results]), B_curve=np.array([r[3] for r in results]))
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
            z = np.load(RESULTS_DIR / "task6" / label_set / f"{ckpt['name']}.npz")
            conditions = json.loads(str(z["conditions"]))
            for b_kind in ("unrelated", "similar"):
                for maintenance in (False, True):
                    sel = [j for j, (b, m, _) in enumerate(conditions) if b == b_kind and m == maintenance]
                    if not sel:
                        continue
                    A_end, A_after_B = z["A_end"][sel].mean(), z["A_curve"][sel][:, :, -1].mean()
                    rows.append({"label_set": label_set, "checkpoint": ckpt["name"], "epoch": ckpt["epoch"],
                                 "B": b_kind, "maintenance": "with" if maintenance else "none", "runs": len(sel),
                                 "A_end": A_end, "A_after_adding_B_labels": z["A_after_adding"][sel].mean(),
                                 "A_after_B": A_after_B, "retention": A_after_B / A_end})
                    print(f"{label_set:15s} epoch {ckpt['epoch']:7.3f}  B {b_kind:9s} maintenance "
                          f"{rows[-1]['maintenance']:4s}  retention {rows[-1]['retention']:.2f}")
    write_csv(RESULTS_DIR / "task6_retention.csv", rows)


if __name__ == "__main__":
    main()

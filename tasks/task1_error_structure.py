"""Task 1 - which wrong answers does the network prefer? (verb_with_noun)

In therapy one notes whether a child who mislabels "drink milk" says "pour milk" (right object, wrong action)
or "drink juice" (right action, wrong object). We ask the same of the held-out probabilities from Task 0.
For a held-out photo x of class c (C classes in total):
  O(c) = the other classes with c's object and another action      ("drink milk" -> "pour milk")
  V(c) = the other classes with c's action and another object      ("drink milk" -> "drink juice")
  m_O(x) = sum_{d in O(c)} p(d|x) / sum_{d != c} p(d|x)   share of the wrong-answer probability on same-object classes
  m_V(x) = the same with V(c)
  E_O = mean_x m_O(x) / mean_x |O(c)| / (C - 1)            enrichment over chance (1 = no preference, > 1 = preferred)
  E_V = mean_x m_V(x) / mean_x |V(c)| / (C - 1)
(|O(c)| / (C - 1) is the share m_O would have if the wrong-answer probability were spread evenly.)
Means run over all photos of all 3 seeds. Output: runs/results/task1_error_structure_<method>.csv
Usage: python tasks/task1_error_structure.py [--method probe|finetune]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np

from common import RESULTS_DIR, class_relations, load_classification_results, write_csv


def enrichment(prob: np.ndarray, labels: np.ndarray, classes: list) -> dict:
    """prob [seeds, photos, C] held-out probabilities."""
    same_object, same_action = class_relations(classes)    # [C, C] boolean
    C = prob.shape[-1]
    P = prob.reshape(-1, C)                                 # one row per (seed, photo)
    y = np.tile(labels, prob.shape[0])
    wrong = P.copy()
    wrong[np.arange(len(y)), y] = 0                         # p(d|x) for d != c
    total_wrong = wrong.sum(1)
    keep = total_wrong > 1e-6                               # skip photos with (numerically) no wrong-answer mass
    O, V = same_object[y][keep], same_action[y][keep]       # [rows, C]: relation of every class d to the true c
    m_O = (wrong[keep] * O).sum(1) / total_wrong[keep]
    m_V = (wrong[keep] * V).sum(1) / total_wrong[keep]
    return {"accuracy": float((P.argmax(1) == y).mean()),
            "E_same_object": float(m_O.mean() / (O.sum(1) / (C - 1)).mean()),
            "E_same_action": float(m_V.mean() / (V.sum(1) / (C - 1)).mean()),
            "share_same_object": float(m_O.mean()), "share_same_action": float(m_V.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="probe", choices=["probe", "finetune"])
    args = ap.parse_args()
    rows = []
    for r in load_classification_results(args.method, "verb_with_noun"):
        rows.append({"checkpoint": r["name"], "epoch": r["epoch"], **enrichment(r["prob"], r["labels"], r["classes"])})
        print(f"epoch {r['epoch']:7.3f}  accuracy {rows[-1]['accuracy']:.3f}  E_same_object {rows[-1]['E_same_object']:.2f}"
              f"  E_same_action {rows[-1]['E_same_action']:.2f}")
    write_csv(RESULTS_DIR / f"task1_error_structure_{args.method}.csv", rows)


if __name__ == "__main__":
    main()

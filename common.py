"""Shared helpers used by every script: paths, checkpoints, TALKtoME class names, cached features, small read-outs.

Run all scripts from the repository root, e.g.  python tasks/task2_new_combinations.py
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ============================================================================ paths
# Defaults live inside the repository; override any of them with an environment variable.
REPO = Path(__file__).resolve().parent
TALKTOME_DIR = Path(os.environ.get("TALKTOME_DIR", REPO / "data" / "TALKtoME"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", REPO / "runs"))

CHECKPOINT_DIR = RUNS_DIR / "checkpoints"   # ResNet-50 checkpoints, downloaded when needed
PREPARED_DIR = RUNS_DIR / "talktome"        # step 1: square photos + video frames
FEATURE_DIR = RUNS_DIR / "features"         # step 2: one feature file per checkpoint
RESULTS_DIR = RUNS_DIR / "results"          # tasks 0-6
FIGURE_DIR = RUNS_DIR / "figures"           # plots

TALKTOME_REPO = "LSL-datasets/TALKtoME"                        # Hugging Face dataset
CHECKPOINT_REPO = "LSL-datasets/resnet50-imagenet-trajectory"  # Hugging Face model repo with the 103 checkpoints

# The two TALKtoME label sets: bare actions ("drink") and action + object ("drink milk").
LABEL_SETS = ["action_verbs", "verb_with_noun"]


# ============================================================================ checkpoints
STEPS_PER_EPOCH = 1251          # 1,281,167 ImageNet training images // batch size 1024
EPOCHS = 90
# Default subset: 16 of the 103 checkpoints, spread over training (dense early, where most change happens).
DEFAULT_EPOCHS = [0, 0.05, 0.2, 0.6, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 75, 90]


def checkpoint_steps() -> list[int]:
    """Training steps at which a checkpoint was saved: step 0 (untrained), steps 16 ... 1000,
    epochs 1.5, 2.5, 3.5, 4.5, and the end of every epoch (103 checkpoints)."""
    early = [0, 16, 32, 64, 125, 250, 500, 750, 1000]
    half_epochs = [round(STEPS_PER_EPOCH * e) for e in (1.5, 2.5, 3.5, 4.5)]
    epoch_ends = [STEPS_PER_EPOCH * e for e in range(1, EPOCHS + 1)]
    return sorted(set(early + half_epochs + epoch_ends))


def select_checkpoints(which: list[str]) -> list[str]:
    """Checkpoint names (e.g. 'step0112590'): ['default'] = 16 spread over training, ['all'] = all 103,
    or an explicit list of names."""
    names = [f"step{s:07d}" for s in checkpoint_steps()]
    if which == ["all"]:
        return names
    if which == ["default"]:
        epochs = np.array(checkpoint_steps()) / STEPS_PER_EPOCH
        return sorted({names[int(np.argmin(np.abs(epochs - e)))] for e in DEFAULT_EPOCHS})
    return which


def download_checkpoint(name: str) -> Path:
    """runs/checkpoints/<name>.pt, downloaded from Hugging Face the first time (~100 MB each)."""
    path = CHECKPOINT_DIR / f"{name}.pt"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(CHECKPOINT_REPO, f"checkpoints/{name}.pt", local_dir=RUNS_DIR)   # -> RUNS_DIR/checkpoints/
    return path


def get_device():
    """The fastest device available: an NVIDIA GPU, an Apple-silicon GPU, or the CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================ class names
PARTICLES = {"on", "off", "up", "out", "down", "over"}    # belong to the action: "put on", "blow out"
ARTICLES = {"the", "a", "an"}                             # dropped: "pack the bag" -> object "bag"
IRREGULAR = {"takeoff glasses": ("take off", "glasses"),  # folder name without a space
             "play with toys": ("play", "toys")}


def split_class_name(name: str) -> tuple[str, str]:
    """'put on shoes' -> ('put on', 'shoes'),  'pack the bag' -> ('pack', 'bag')."""
    if name in IRREGULAR:
        return IRREGULAR[name]
    words = name.split()
    n = 1
    while n < len(words) and words[n] in PARTICLES:
        n += 1
    action = " ".join(words[:n])
    obj = " ".join(w for w in words[n:] if w not in ARTICLES)
    return action, obj


def load_metadata(label_set: str) -> dict:
    """Written by 1_prepare_talktome.py: classes, actions[c], objects[c], photo_files/photo_labels, video_files/video_labels."""
    return json.loads((PREPARED_DIR / f"{label_set}.json").read_text())


def class_relations(classes: list) -> tuple[np.ndarray, np.ndarray]:
    """Boolean [C, C] matrices for verb_with_noun classes c (row) and d (column):
    same_object[c, d] = d has c's object but another action   ("drink milk" -> "pour milk")
    same_action[c, d] = d has c's action but another object   ("drink milk" -> "drink juice")"""
    actions, objects = zip(*[split_class_name(c) for c in classes])
    a, o = np.array(actions, dtype=object), np.array(objects, dtype=object)
    same_object = (o[:, None] == o[None, :]) & (a[:, None] != a[None, :])
    same_action = (a[:, None] == a[None, :]) & (o[:, None] != o[None, :])
    return same_object, same_action


# ============================================================================ checkpoints and features
def list_checkpoints() -> list[dict]:
    """All checkpoints that have a feature file (2_extract_features.py), sorted by ImageNet epoch."""
    out = []
    for path in FEATURE_DIR.glob("step*.npz"):
        with np.load(path) as z:
            out.append({"name": path.stem, "path": path, "epoch": float(z["epoch"]), "val_top1": float(z["val_top1"])})
    return sorted(out, key=lambda c: c["epoch"])


def load_photo_features(ckpt: dict, label_set: str) -> tuple[np.ndarray, np.ndarray]:
    """photo [N_photos, 2048] and photo_flipped [N_photos, 2048] (mirror images) of one checkpoint."""
    with np.load(ckpt["path"]) as z:
        return z[f"{label_set}_photo"].astype(np.float32), z[f"{label_set}_photo_flipped"].astype(np.float32)


@dataclass
class Exemplars:
    """Every photo and every video of a label set, one feature vector each (used by tasks 2, 3, 4, 6)."""
    X: np.ndarray          # [E, D] features, standardized: every dimension has mean 0, std 1 over all exemplars
    X_flipped: np.ndarray  # [E, D] the same, with photos mirrored (videos unchanged)
    label: np.ndarray      # [E] class index
    is_video: np.ndarray   # [E] True for videos, False for photos
    classes: list          # class names
    actions: list          # action of each class
    objects: list          # object of each class (None for action_verbs)


def load_exemplars(ckpt: dict, label_set: str) -> Exemplars:
    meta = load_metadata(label_set)
    with np.load(ckpt["path"]) as z:
        photo = z[f"{label_set}_photo"].astype(np.float32)
        flipped = z[f"{label_set}_photo_flipped"].astype(np.float32)
        video = z[f"{label_set}_video"].astype(np.float32)
    X = np.concatenate([photo, video])
    mean, std = X.mean(0), X.std(0) + 1e-6
    return Exemplars(
        X=(X - mean) / std,
        X_flipped=(np.concatenate([flipped, video]) - mean) / std,
        label=np.array(meta["photo_labels"] + meta["video_labels"]),
        is_video=np.array([False] * len(photo) + [True] * len(video)),
        classes=meta["classes"], actions=meta["actions"], objects=meta["objects"])


def load_resnet50(checkpoint_path, n_outputs: int | None = None):
    """ResNet-50 with the weights of a checkpoint file (see download_checkpoint). The ImageNet classifier is replaced by
    nothing (the model then outputs its 2048-d feature) or by a new linear layer with n_outputs classes."""
    import torch            # imported here so the CPU-only tasks do not need torch
    import torchvision
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = torchvision.models.resnet50()
    model.load_state_dict(state["model"])
    model.fc = torch.nn.Identity() if n_outputs is None else torch.nn.Linear(2048, n_outputs)
    return model, {k: state[k] for k in ("step", "epoch", "val_top1")}


def normalize(images):
    """uint8 images [B, 3, H, W] (torch, any device) -> float, standardized with the ImageNet pixel mean/std."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1) * 255
    return (images.float() - mean) / std


# ============================================================================ base-task photos and results
def classification_photos(label_set: str, min_photos: int = 3) -> tuple[np.ndarray, np.ndarray, list]:
    """Photos used by the base task (Task 0): classes with at least `min_photos` photos (5-fold splits need a few).
    Returns photo indices (into the list of photos of 1_prepare_talktome.py), labels 0..C-1, and the kept class names."""
    meta = load_metadata(label_set)
    labels = np.array(meta["photo_labels"])
    kept = [c for c in range(len(meta["classes"])) if (labels == c).sum() >= min_photos]
    photos = np.where(np.isin(labels, kept))[0]
    new_label = {c: i for i, c in enumerate(kept)}
    return photos, np.array([new_label[c] for c in labels[photos]]), [meta["classes"][c] for c in kept]


def photo_folds(labels: np.ndarray, seed: int, n_folds: int = 5) -> np.ndarray:
    """Fold id (0..n_folds-1) of every photo; each class is spread evenly over the folds."""
    import warnings
    from sklearn.model_selection import StratifiedKFold
    fold = np.zeros(len(labels), int)
    with warnings.catch_warnings():               # sklearn warns that some classes have fewer photos than folds
        warnings.simplefilter("ignore")
        splits = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(labels, labels))
    for k, (_, test) in enumerate(splits):
        fold[test] = k
    return fold


def load_classification_results(method: str, label_set: str) -> list[dict]:
    """Task-0 outputs (method 'probe' or 'finetune'), sorted by epoch. prob[s, i, c] = held-out p(c | photo i), seed s."""
    out = []
    for path in (RESULTS_DIR / f"task0_{method}" / label_set).glob("step*.npz"):
        with np.load(path) as z:
            out.append({"name": path.stem, "epoch": float(z["epoch"]), "prob": z["prob"].astype(np.float32),
                        "labels": z["labels"], "classes": list(z["classes"])})
    return sorted(out, key=lambda r: r["epoch"])


# ============================================================================ small math helpers
def one_hot(labels: np.ndarray, n: int) -> np.ndarray:
    Y = np.zeros((len(labels), n))
    Y[np.arange(len(labels)), labels] = 1
    return Y


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ridge_regression(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """W minimizing ||[X 1] W - Y||^2 + lam ||W||^2  (a constant 1 is appended to x for the bias)."""
    Xb = np.hstack([X, np.ones((len(X), 1))]).astype(np.float64)
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)


def ridge_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id of every item: each class's items are shuffled and dealt round-robin to the folds."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(labels), int)
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


# ============================================================================ files
def save_npz(path: Path, **arrays):
    """np.savez to a temporary name, then rename: a killed job never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        print(f"nothing to write to {path} (are the inputs of this task there?)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def read_csv(path: Path) -> list[dict]:
    """Rows as dicts; values that look like numbers are converted to float."""
    def convert(v):
        try:
            return float(v)
        except ValueError:
            return v
    with open(path, newline="") as f:
        return [{k: convert(v) for k, v in row.items()} for row in csv.DictReader(f)]

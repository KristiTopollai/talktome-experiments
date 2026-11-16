"""Step 2 - frozen ResNet-50 features of every TALKtoME photo and video, for every checkpoint.

The checkpoints (snapshots of a ResNet-50 during 90 epochs of ImageNet training; the training recipe is on their
Hugging Face page) are downloaded one at a time (~100 MB each) and deleted again once their features are saved.
A feature vector is the 2048 numbers the network computes just before its ImageNet classifier (the output of
the final average-pooling layer), on the central 224 x 224 crop of a 256 x 256 image. Per checkpoint and
label set we store
  <label_set>_photo          [N_photos, 2048]
  <label_set>_photo_flipped  [N_photos, 2048]   the same photos mirrored left-right (extra training data)
  <label_set>_video          [N_videos, 2048]   mean over the video's frames
plus the checkpoint's step, epoch and ImageNet top-1 accuracy, in runs/features/<checkpoint>.npz (~11 MB).
Every task except fine-tuning starts from these files, so the network is run only once per checkpoint.
Time per checkpoint: ~3 s on a GPU, ~6 min on a laptop CPU.
Usage: python 2_extract_features.py                       16 checkpoints spread over training (default)
       python 2_extract_features.py --checkpoints all     all 103
       python 2_extract_features.py --checkpoints step0000000 step0112590
Checkpoints that already have a feature file are skipped, so you can stop and restart at any time.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import (FEATURE_DIR, LABEL_SETS, PREPARED_DIR, download_checkpoint, get_device, load_resnet50, normalize,
                    save_npz, select_checkpoints)


@torch.no_grad()
def features(model, images: np.ndarray, device, flip: bool = False, batch_size: int = 64) -> np.ndarray:
    """uint8 images [N, 256, 256, 3] -> features [N, 2048] of their central 224 x 224 crop."""
    out = []
    for s in range(0, len(images), batch_size):
        x = torch.from_numpy(images[s:s + batch_size, 16:240, 16:240]).to(device).permute(0, 3, 1, 2)
        x = normalize(x)
        if flip:
            x = x.flip(-1)                                 # mirror left-right
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", default=["default"], help="'default' (16), 'all' (103), or names")
    ap.add_argument("--keep-checkpoints", action="store_true", help="keep the downloaded checkpoint files")
    args = ap.parse_args()
    device = get_device()
    todo = [n for n in select_checkpoints(args.checkpoints) if not (FEATURE_DIR / f"{n}.npz").exists()]
    print(f"{len(todo)} checkpoints to do, on {device}", flush=True)

    pixels = {s: dict(np.load(PREPARED_DIR / f"{s}_pixels.npz")) for s in LABEL_SETS}   # read into memory once
    for name in todo:
        t0 = time.time()
        path = download_checkpoint(name)
        model, info = load_resnet50(path)
        model = model.to(device).eval()
        result = dict(info)
        for label_set in LABEL_SETS:
            photos, frames, frame_video = (pixels[label_set][k] for k in ("photos", "frames", "frame_video"))
            frame_features = features(model, frames, device)
            n_videos = frame_video.max() + 1
            video = np.zeros((n_videos, frame_features.shape[1]), np.float32)
            np.add.at(video, frame_video, frame_features)              # sum the frames of each video ...
            video /= np.bincount(frame_video, minlength=n_videos)[:, None]   # ... and divide by their number
            result[f"{label_set}_photo"] = features(model, photos, device).astype(np.float16)
            result[f"{label_set}_photo_flipped"] = features(model, photos, device, flip=True).astype(np.float16)
            result[f"{label_set}_video"] = video.astype(np.float16)
        save_npz(FEATURE_DIR / f"{name}.npz", **result)
        if not args.keep_checkpoints:
            path.unlink()
        print(f"{name}: epoch {info['epoch']:.3f}, ImageNet top-1 {info['val_top1']:.3f}, {time.time() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()

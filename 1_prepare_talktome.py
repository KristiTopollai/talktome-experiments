"""Step 1 - read the TALKtoME photos and videos once and store them as small square images (~10 min on a laptop).

TALKtoME has two label sets, each with one folder per class:
  action_verbs/{images,videos}/<action>/             e.g. "drink", "jump"
  verb_with_noun/{images,videos}/<action object>/    e.g. "drink milk", "put on shoes"
For every class folder we keep
  photos : resized so the short side is 256 pixels, then the central 256 x 256 square
  videos : 8 frames per video, at 1/16, 3/16, ..., 15/16 of its duration, resized and cropped the same way
and split each verb_with_noun class name into an action and an object ("put on shoes" -> "put on" + "shoes").
Outputs per label set, in runs/talktome/:
  <label_set>.json        classes, actions, objects, and the file name + class of every photo and video
  <label_set>_pixels.npz  photos [N, 256, 256, 3], frames [F, 256, 256, 3] (uint8 RGB), frame_video [F]
                          (frame_video[f] = which video frame f belongs to)
Video frames are read with ffmpeg; the pip package imageio-ffmpeg ships a ready binary.
Usage: python 1_prepare_talktome.py            (downloads TALKtoME into data/TALKtoME first if it is not there)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

from common import LABEL_SETS, PREPARED_DIR, TALKTOME_DIR, TALKTOME_REPO, save_npz, split_class_name

SIZE = 256
FRAMES_PER_VIDEO = 8
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def download_talktome():
    """Photos and videos of both label sets from Hugging Face (~16 GB). Files already downloaded are skipped."""
    from huggingface_hub import snapshot_download
    print(f"downloading {TALKTOME_REPO} into {TALKTOME_DIR} (~16 GB) ...", flush=True)
    snapshot_download(TALKTOME_REPO, repo_type="dataset", local_dir=TALKTOME_DIR,
                      allow_patterns=[f"{s}/{kind}/*" for s in LABEL_SETS for kind in ("images", "videos")])


def list_files(folder, photos_only=False):
    """Non-empty, non-hidden files of a folder (sorted)."""
    if not folder.is_dir():
        return []
    files = [p for p in sorted(folder.iterdir())
             if p.is_file() and not p.name.startswith(".") and p.stat().st_size > 0]
    return [p for p in files if p.suffix.lower() in PHOTO_TYPES] if photos_only else files


def load_photo(path) -> np.ndarray:
    """Short side -> 256 pixels, then the central 256 x 256 square (uint8 RGB)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")       # respect the camera's orientation tag
        w, h = im.size
        scale = SIZE / min(w, h)
        im = im.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.BICUBIC, reducing_gap=3.0)
        w, h = im.size
        left, top = (w - SIZE) // 2, (h - SIZE) // 2
        return np.asarray(im.crop((left, top, left + SIZE, top + SIZE)))


def video_duration(path) -> float:
    """Length in seconds, read from ffmpeg's description of the file."""
    info = subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def video_frame(path, seconds: float) -> np.ndarray | None:
    """The frame at `seconds`, resized and cropped like the photos (ffmpeg also applies the rotation tag)."""
    resize_and_crop = f"scale=w='if(gt(iw,ih),-2,{SIZE})':h='if(gt(iw,ih),{SIZE},-2)':flags=area,crop={SIZE}:{SIZE}"
    for t in (seconds, 0.9 * seconds, max(0.0, seconds - 1)):     # retry earlier if the seek lands past the end
        cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-threads", "2", "-ss", f"{t:.3f}", "-i", str(path),
               "-frames:v", "1", "-vf", resize_and_crop, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
        if len(raw) == SIZE * SIZE * 3:
            return np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3)
    return None


def video_frames(path) -> list[np.ndarray]:
    duration = video_duration(path)
    frames = [video_frame(path, (k + 0.5) / FRAMES_PER_VIDEO * duration) for k in range(FRAMES_PER_VIDEO)]
    return [f for f in frames if f is not None]


def prepare(label_set: str, workers: int):
    root = TALKTOME_DIR / label_set
    classes = sorted({p.name for kind in ("images", "videos") for p in (root / kind).iterdir() if p.is_dir()})
    photo_files, photo_labels, video_files, video_labels = [], [], [], []
    for c, name in enumerate(classes):
        for path in list_files(root / "images" / name, photos_only=True):
            photo_files.append(path)
            photo_labels.append(c)
        for path in list_files(root / "videos" / name):
            video_files.append(path)
            video_labels.append(c)

    # threads are enough: PIL and ffmpeg (a separate process) do the work outside Python
    with ThreadPoolExecutor(workers) as pool:
        photos = list(pool.map(load_photo, photo_files))
        frames_of_video = list(pool.map(video_frames, video_files))

    unreadable = [str(p) for p, fr in zip(video_files, frames_of_video) if not fr]
    keep = [v for v, fr in enumerate(frames_of_video) if fr]
    frames = [f for v in keep for f in frames_of_video[v]]
    frame_video = [i for i, v in enumerate(keep) for _ in frames_of_video[v]]

    names = [split_class_name(c) if label_set == "verb_with_noun" else (c, None) for c in classes]
    meta = {"label_set": label_set, "classes": classes,
            "actions": [a for a, _ in names], "objects": [o for _, o in names],
            "photo_files": [str(p.relative_to(TALKTOME_DIR)) for p in photo_files], "photo_labels": photo_labels,
            "video_files": [str(video_files[v].relative_to(TALKTOME_DIR)) for v in keep],
            "video_labels": [video_labels[v] for v in keep]}
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    (PREPARED_DIR / f"{label_set}.json").write_text(json.dumps(meta, indent=1))
    save_npz(PREPARED_DIR / f"{label_set}_pixels.npz", photos=np.stack(photos), frames=np.stack(frames),
             frame_video=np.array(frame_video))
    print(f"{label_set}: {len(classes)} classes, {len(photos)} photos, {len(keep)} videos ({len(frames)} frames); "
          f"unreadable videos: {unreadable}", flush=True)
    if label_set == "verb_with_noun":
        print(f"  {len(set(meta['actions']))} actions, {len(set(meta['objects']))} objects; e.g. "
              + ", ".join(f"'{c}' = {a} + {o}" for c, a, o in list(zip(classes, meta['actions'], meta['objects']))[:4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--download", action="store_true", help="download again (e.g. to finish an interrupted download)")
    args = ap.parse_args()
    if args.download or not all((TALKTOME_DIR / s / kind).is_dir() for s in LABEL_SETS for kind in ("images", "videos")):
        download_talktome()
    if all((PREPARED_DIR / f"{s}_pixels.npz").exists() for s in LABEL_SETS) and not args.download:
        print(f"already prepared ({PREPARED_DIR}); delete that folder to prepare again")
        return
    for label_set in LABEL_SETS:
        prepare(label_set, args.workers)


if __name__ == "__main__":
    main()

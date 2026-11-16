"""Task 0 (base task), second read-out - fine-tune the whole network on TALKtoME photos (OPTIONAL, needs a GPU).

Same task, folds and outputs as task0_classify_probe.py, but instead of a linear read-out on frozen features the
whole ResNet-50 is trained further: the ImageNet classifier is replaced by a new linear layer with C outputs and
all weights are updated.
  training  : 30 epochs over the 4 training folds, batch 32, SGD (momentum 0.9, Nesterov, weight decay 1e-4),
              learning rate 0.003 for the pretrained layers and 0.03 for the new layer, 1 warm-up epoch then
              cosine decay; augmentation = random crop of 35-100% of the image resized to 224 + random flip.
  held out  : after the last epoch the 5th fold is classified (central 224 crop); 5 folds x 3 seeds as for the probe.
Training a ResNet-50 needs an NVIDIA GPU (e.g. a free Google Colab GPU): ~10 GPU-minutes per checkpoint for both
label sets and 3 seeds (~3 min with --seeds 0). Start with a few checkpoints, e.g. the first and the last.
The checkpoints are downloaded from Hugging Face when needed (~100 MB each; kept in runs/checkpoints/).
Outputs (runs/results/task0_finetune/): <label_set>/<checkpoint>.npz with prob [seeds, photos, classes] and
accuracy_by_epoch [seeds, epochs] (held-out accuracy after each fine-tuning epoch), plus the same tables as the probe.
Tasks 1 and 5 can then use these probabilities instead of the probe's (--method finetune).
Usage: python tasks/task0_classify_finetune.py --checkpoints step0000000 step0112590 [--seeds 0]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so that `import common` finds the repository root
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from common import (LABEL_SETS, PREPARED_DIR, RESULTS_DIR, classification_photos, download_checkpoint, load_resnet50,
                    normalize, photo_folds, save_npz, select_checkpoints)
from task0_classify_probe import write_tables

AUGMENT = T.Compose([T.RandomResizedCrop(224, scale=(0.35, 1.0), antialias=True), T.RandomHorizontalFlip()])


def finetune_one_fold(checkpoint, images, y, train, test, n_classes, args, seed):
    """Fine-tune on photos `train`; return held-out p(c|x) on `test` and the test predictions after every epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, _ = load_resnet50(checkpoint, n_outputs=n_classes)
    model = model.cuda().to(memory_format=torch.channels_last)
    new_layer = list(model.fc.parameters())
    pretrained = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.SGD([{"params": pretrained, "lr": args.lr_pretrained},
                                 {"params": new_layer, "lr": args.lr_new}],
                                momentum=0.9, nesterov=True, weight_decay=1e-4)
    base_lrs = [args.lr_pretrained, args.lr_new]
    steps_per_epoch = max(1, len(train) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    test_images = normalize(images[test][:, :, 16:240, 16:240]).contiguous(memory_format=torch.channels_last)

    predictions, step = [], 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train)
        for s in range(steps_per_epoch):
            # learning-rate multiplier: linear warm-up during epoch 1, then cosine decay to 0
            progress = (step - steps_per_epoch) / max(1, total_steps - steps_per_epoch)
            mult = (step + 1) / steps_per_epoch if step < steps_per_epoch else 0.5 * (1 + math.cos(math.pi * progress))
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * mult
            batch = order[s * args.batch_size:(s + 1) * args.batch_size]
            x = torch.stack([AUGMENT(images[i].float()) for i in batch])
            x = normalize(x).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(test_images).float()
        predictions.append(logits.argmax(1).cpu().numpy())
    return torch.softmax(logits, 1).cpu().numpy(), np.stack(predictions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="checkpoint names (e.g. step0000000 step0112590), 'default' (16) or 'all' (103)")
    ap.add_argument("--label-sets", nargs="+", default=LABEL_SETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-pretrained", type=float, default=0.003)
    ap.add_argument("--lr-new", type=float, default=0.03)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("Fine-tuning needs an NVIDIA GPU (e.g. Google Colab). The linear probe (task0_classify_probe.py) "
                 "runs on any computer.")
    torch.backends.cudnn.benchmark = True

    for label_set in args.label_sets:
        photos, labels, classes = classification_photos(label_set)
        images = torch.from_numpy(np.load(PREPARED_DIR / f"{label_set}_pixels.npz")["photos"][photos])
        images = images.permute(0, 3, 1, 2).contiguous().cuda()          # uint8 [N, 3, 256, 256] on the GPU
        y = torch.as_tensor(labels).cuda()
        for name in select_checkpoints(args.checkpoints):
            out = RESULTS_DIR / "task0_finetune" / label_set / f"{name}.npz"
            if out.exists():
                continue
            t0 = time.time()
            checkpoint = download_checkpoint(name)
            prob = np.zeros((len(args.seeds), len(labels), len(classes)), np.float32)
            accuracy_by_epoch = np.zeros((len(args.seeds), args.epochs))
            for s, seed in enumerate(args.seeds):
                fold = photo_folds(labels, seed)
                correct = np.zeros((args.epochs, len(labels)), bool)
                for k in range(fold.max() + 1):
                    train, test = np.where(fold != k)[0], np.where(fold == k)[0]
                    prob[s, test], pred = finetune_one_fold(checkpoint, images, y, train, test, len(classes), args,
                                                            seed=1000 * seed + k)
                    correct[:, test] = pred == labels[test]
                accuracy_by_epoch[s] = correct.mean(1)
            epoch = torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"]
            save_npz(out, prob=prob.astype(np.float16), labels=labels, classes=np.array(classes), epoch=epoch,
                     accuracy_by_epoch=accuracy_by_epoch)
            print(f"{label_set} {name} epoch {epoch:.3f}: accuracy {accuracy_by_epoch[:, -1].mean():.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        write_tables("finetune", label_set)


if __name__ == "__main__":
    main()

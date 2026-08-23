"""Train NanoStream-OD on the combined shapes+faces benchmark dataset.

Wires the project's own YOLO-grade augmentation pipeline (mosaic, mixup,
multi-scale, geometric, photometric, cutout) into training. The augmenters
already lived in ``nanostream/augment.py`` but no training path called them;
this is the wiring that makes the config's ``augment_*`` flags real.
"""

import argparse
import pathlib

import numpy as np
import torch

from nanostream.augment import (cutout, geometric_augment, mixup, mosaic_4,
                                multi_scale_resize, photometric_distort)
from nanostream.config import NanoStreamConfig
from nanostream.head import detection_loss
from nanostream.model import NanoStreamOD

from .combined_data import (CLASS_NAMES, NUM_CLASSES, collate, generate_sample,
                            to_target)

PROFILES = {
    "mcu": NanoStreamConfig.mcu,
    "pro": NanoStreamConfig.pro,
    "gpu": NanoStreamConfig.gpu,
}


def augment_sample(cfg, rng, data_len):
    """Build one augmented training sample (uint8 img, pixel boxes, labels).

    Mirrors the YOLOv8 recipe at a scale appropriate to the config profile:
    mosaic -> mixup -> multi-scale -> geometric -> photometric -> cutout.
    All ops run in pixel space; boxes stay [x1, y1, x2, y2] lists.
    """
    size = cfg.input_size
    if cfg.augment_mosaic and rng.random() < 0.5:
        idxs = rng.integers(0, data_len, size=4)
        imgs, box_l, lab_l = [], [], []
        for i in idxs:
            im, bx, lb = generate_sample(int(i), size)
            imgs.append(im)
            box_l.append(bx)
            lab_l.append(lb)
        img, boxes, labels = mosaic_4(imgs, box_l, lab_l, size, rng)
    else:
        idx = int(rng.integers(0, data_len))
        img, boxes, labels = generate_sample(idx, size)
        boxes, labels = list(boxes), list(labels)

    if cfg.augment_mixup and rng.random() < 0.3 and len(boxes) > 0:
        idx2 = int(rng.integers(0, data_len))
        img2, bx2, lb2 = generate_sample(idx2, size)
        img, boxes, labels = mixup(img, boxes, labels, img2, bx2, lb2, rng=rng)

    if cfg.multi_scale_train and rng.random() < 0.5:
        lo, hi = cfg.multi_scale_range
        img, boxes, labels = multi_scale_resize(img, boxes, size, rng, lo, hi,
                                                labels=labels)

    img, boxes, labels = geometric_augment(img, boxes, size, rng,
                                           labels=labels)

    img = photometric_distort(img, rng)

    if cfg.augment_cutout and rng.random() < 0.3:
        img = cutout(img, boxes, rng=rng)

    return img, boxes, labels


def _sample_to_tensor(img, boxes, labels, size):
    x = torch.from_numpy(img.copy()).float() / 127.5 - 1.0
    x = x.unsqueeze(0)  # (1, H, W)
    return x, to_target(boxes, labels, size)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg = PROFILES[args.profile](num_classes=NUM_CLASSES)
    input_size = getattr(args, "input_size", 0)
    if input_size:
        cfg.input_size = input_size
    model = NanoStreamOD(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    warmup = min(50, args.steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        prog = float(step - warmup) / float(max(1, args.steps - warmup))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = np.random.default_rng(args.seed + 42)
    model.train()
    for step in range(args.steps):
        batch = []
        for _ in range(args.batch):
            img, boxes, labels = augment_sample(cfg, rng, args.data_len)
            batch.append(_sample_to_tensor(img, boxes, labels, cfg.input_size))
        xs, ts = collate(batch)
        xs = xs.to(device)
        ts = [{k: v.to(device) for k, v in t.items()} for t in ts]
        preds = model(xs)
        losses = detection_loss(preds, ts, cfg)
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:4d}/{args.steps} loss {float(losses['total']):.3f} "
                  f"obj {float(losses['obj']):.3f} box {float(losses['box']):.3f} "
                  f"cls {float(losses['cls']):.3f} pos {losses['num_pos']:.0f}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict(), "config": vars(cfg),
            "classes": list(CLASS_NAMES), "profile": args.profile}
    torch.save(ckpt, out / f"nanostream_{args.profile}.pt")
    print(f"saved -> {out / f'nanostream_{args.profile}.pt'}")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=list(PROFILES), default="mcu")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input_size", type=int, default=0)
    p.add_argument("--data_len", type=int, default=1200)
    p.add_argument("--out", type=str, default="benchmarks/runs/ckpt")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

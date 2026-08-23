"""Train the FOMO-style baseline on the combined shapes+faces dataset."""

import argparse
import pathlib
from types import SimpleNamespace

import numpy as np
import torch

from .combined_data import NUM_CLASSES, collate, to_target
from .fomo_model import FomoDetector, focal_loss, gaussian_targets
from .train_nanostream import augment_sample


# FOMO trains at 160 px (MCU-tier baseline) but with the same augmentation
# recipe as NanoStream so the comparison isolates architecture, not data diet.
FOMO_AUG = SimpleNamespace(input_size=160, augment_mosaic=True,
                           augment_mixup=True, augment_cutout=True,
                           multi_scale_train=True, multi_scale_range=(128, 192))


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = FomoDetector(num_classes=NUM_CLASSES, width_mult=args.width,
                         img_size=args.input_size).to(device)
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
            img, boxes, labels = augment_sample(FOMO_AUG, rng, args.data_len)
            x = torch.from_numpy(img.copy()).float() / 127.5 - 1.0
            batch.append((x.unsqueeze(0), to_target(boxes, labels, args.input_size)))
        xs, ts = collate(batch)
        xs = xs.to(device)
        targets = torch.zeros(args.batch, NUM_CLASSES, model.grid, model.grid)
        for bi, t in enumerate(ts):
            if len(t["labels"]):
                g = gaussian_targets(t["boxes_norm"], t["labels"], model.grid)
                for gi, lab in enumerate(t["labels"].tolist()):
                    targets[bi, lab] = torch.maximum(targets[bi, lab], g[gi])
        targets = targets.to(device)
        logits = model(xs)
        loss = focal_loss(logits, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:4d}/{args.steps} loss {float(loss):.4f}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": list(range(NUM_CLASSES))},
               out / "fomo_bench.pt")
    print(f"saved -> {out / 'fomo_bench.pt'}")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=float, default=0.75)
    p.add_argument("--input_size", type=int, default=160)
    p.add_argument("--data_len", type=int, default=1200)
    p.add_argument("--out", type=str, default="benchmarks/runs/ckpt")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

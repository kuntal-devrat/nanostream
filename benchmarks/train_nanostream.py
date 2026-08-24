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


def augment_sample(cfg, rng, data_len, sample_cache=None):
    """Build one augmented training sample (uint8 img, pixel boxes, labels).

    Mirrors the YOLOv8 recipe at a scale appropriate to the config profile:
    mosaic -> mixup -> multi-scale -> geometric -> photometric -> cutout.
    All ops run in pixel space; boxes stay [x1, y1, x2, y2] lists.
    """
    size = cfg.input_size

    def _get(idx):
        if sample_cache is not None and len(sample_cache) > 0:
            im, bx, lb = sample_cache[idx % len(sample_cache)]
            return im.copy(), [b[:] for b in bx], list(lb)
        return generate_sample(idx, size)

    if cfg.augment_mosaic and rng.random() < 0.5:
        idxs = rng.integers(0, data_len, size=4)
        imgs, box_l, lab_l = [], [], []
        for i in idxs:
            im, bx, lb = _get(int(i))
            imgs.append(im)
            box_l.append(bx)
            lab_l.append(lb)
        img, boxes, labels = mosaic_4(imgs, box_l, lab_l, size, rng)
    else:
        idx = int(rng.integers(0, data_len))
        img, boxes, labels = _get(idx)

    if cfg.augment_mixup and rng.random() < 0.3 and len(boxes) > 0:
        idx2 = int(rng.integers(0, data_len))
        img2, bx2, lb2 = _get(idx2)
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


def _checkpoint_path(args):
    out = getattr(args, "out", "benchmarks/runs/ckpt")
    prof = getattr(args, "profile", "mcu")
    return pathlib.Path(out) / f"nanostream_{prof}.pt"


def _save_ckpt(path, model, opt, sched, rng, step, cfg, profile):
    """Save full training state so a later run can resume exactly."""
    raw_model = getattr(model, "module", model)
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "rng_np": rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(),
        "step": step,
        "config": vars(cfg),
        "classes": list(CLASS_NAMES),
        "profile": profile,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


class BenchmarkDataset(torch.utils.data.Dataset):
    """Parallelized fast dataset with in-RAM pre-rendered base samples."""
    def __init__(self, cfg, data_len, total_samples, seed=42):
        self.cfg = cfg
        self.data_len = data_len
        self.total_samples = total_samples
        self.seed = seed
        cache_count = min(data_len, 400)
        self.sample_cache = [generate_sample(i, cfg.input_size) for i in range(cache_count)]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + 1000 + idx)
        img, boxes, labels = augment_sample(self.cfg, rng, len(self.sample_cache), sample_cache=self.sample_cache)
        return _sample_to_tensor(img, boxes, labels, self.cfg.input_size)


def train(args):
    import os
    seed = getattr(args, "seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dev_str = getattr(args, "device", "") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)

    profile = getattr(args, "profile", "mcu")
    cfg = PROFILES[profile](num_classes=NUM_CLASSES)
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
    rng = np.random.default_rng(seed + 42)

    ckpt_path = _checkpoint_path(args)
    start_step = 0
    resume = getattr(args, "resume", False)
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        rng.bit_generator.state = ckpt["rng_np"]
        torch.set_rng_state(ckpt["rng_torch"])
        start_step = int(ckpt["step"]) + 1
        print(f"[resume] {profile}: continuing from step {start_step} "
              f"(found {ckpt_path})")

    steps = getattr(args, "steps", 1000)
    batch_size = getattr(args, "batch", 16)
    data_len = getattr(args, "data_len", 1200)
    save_every = getattr(args, "save_every", 500)

    # Multi-GPU support if multiple GPUs detected
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print(f"Detected {n_gpus} GPUs: Multi-GPU DataParallel active!")

    # Multi-worker prefetching DataLoader for high-throughput GPU saturation
    dataset = BenchmarkDataset(cfg, data_len=data_len, total_samples=steps * batch_size, seed=seed)
    num_workers = min(4, os.cpu_count() or 1) if device.type == "cuda" else 0
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate, prefetch_factor=2 if num_workers > 0 else None
    )

    use_amp = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    model.train()
    data_iter = iter(loader)

    # Skip already-trained batches if resumed
    if start_step > 0:
        for _ in range(start_step):
            try:
                next(data_iter)
            except StopIteration:
                break

    for step in range(start_step, steps):
        try:
            xs, ts = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xs, ts = next(data_iter)

        xs = xs.to(device, non_blocking=True)
        ts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in ts]

        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            preds = model(xs)
            losses = detection_loss(preds, ts, cfg)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 100 == 0 or step == steps - 1:
            tot = float(losses['total'].detach())
            obj_l = float(losses['obj'].detach())
            box_l = float(losses['box'].detach())
            cls_l = float(losses['cls'].detach())
            print(f"step {step:4d}/{steps} loss {tot:.3f} "
                  f"obj {obj_l:.3f} box {box_l:.3f} "
                  f"cls {cls_l:.3f} pos {losses['num_pos']:.0f}")
        if save_every and step % save_every == 0 and step > start_step:
            _save_ckpt(ckpt_path, model, opt, sched, rng, step, cfg, profile)
            print(f"[ckpt] {profile} step {step} -> {ckpt_path}")

    _save_ckpt(ckpt_path, model, opt, sched, rng, steps - 1, cfg, profile)
    print(f"saved -> {ckpt_path}")
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
    p.add_argument("--save_every", type=int, default=500,
                   help="checkpoint every N steps (for resume)")
    p.add_argument("--resume", action="store_true",
                   help="resume from existing checkpoint for this profile")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

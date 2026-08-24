"""Train NanoStream-OD on Real-World PASCAL VOC Dataset.

Features:
- Multi-GPU DataParallel & AMP fp16 Tensor Core acceleration
- ModelEMA for smooth gradient convergence & high evaluation mAP
- Pure in-VRAM fast tensor augmentation
- Checkpoint validation & model saving
"""

import argparse
import pathlib
import time
import numpy as np
import torch
from nanostream.config import PROFILES
from nanostream.model import NanoStreamOD
from nanostream.head import detection_loss
from nanostream.ema import ModelEMA
from projects.real_voc.dataset import RealVOCDataset, NUM_CLASSES, VOC_CLASSES

CKPT_DIR = pathlib.Path("benchmarks/runs/voc_ckpt")


def _save_ckpt(path, model, opt, sched, step, cfg, profile, ema=None):
    eval_model = ema.ema if ema is not None else getattr(model, "module", model)
    ckpt = {
        "model": eval_model.state_dict(),
        "raw_model": getattr(model, "module", model).state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "step": step,
        "config": vars(cfg),
        "classes": list(VOC_CLASSES),
        "profile": profile,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def train_voc(args):
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dev_str = getattr(args, "device", "") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)

    profile = getattr(args, "profile", "mcu")
    cfg = PROFILES[profile](num_classes=NUM_CLASSES)
    model = NanoStreamOD(cfg).to(device)

    # Initialize ModelEMA
    ema = ModelEMA(model, decay=0.999)

    # Multi-GPU support
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs: Enabling Multi-GPU DataParallel!")
        model = torch.nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = min(100, args.steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        prog = float(step - warmup) / float(max(1, args.steps - warmup))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = np.random.default_rng(seed)

    # Pre-render / cache VOC dataset in memory for maximum throughput
    print(f"[{profile}] Pre-loading VOC train split into memory...")
    train_dataset = RealVOCDataset(split="train", input_size=cfg.input_size)
    pool_imgs = []
    pool_tgts = []
    for i in range(len(train_dataset)):
        x, tgt = train_dataset[i]
        pool_imgs.append(x)
        pool_tgts.append(tgt)

    pool_size = len(pool_imgs)
    steps = getattr(args, "steps", 3000)
    batch_size = getattr(args, "batch", 32)
    ckpt_path = CKPT_DIR / f"nanostream_voc_{profile}.pt"

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=use_amp)

    model.train()
    print(f"\n[{profile}] Starting training for {steps} steps (batch_size={batch_size})...")
    t0 = time.time()

    for step in range(steps):
        idxs = rng.integers(0, pool_size, size=batch_size)
        batch_x = torch.stack([pool_imgs[int(i)] for i in idxs]).to(device, non_blocking=True)
        batch_t = [{k: v.clone().to(device, non_blocking=True) for k, v in pool_tgts[int(i)].items()} for i in idxs]

        # Real-world tensor augmentations on GPU:
        # 1. Random Horizontal Flip
        if rng.random() < 0.5:
            batch_x = torch.flip(batch_x, dims=[-1])
            for t in batch_t:
                if len(t["boxes_norm"]) > 0:
                    t["boxes_norm"][:, 0] = 1.0 - t["boxes_norm"][:, 0]

        # 2. Random Brightness & Contrast
        if rng.random() < 0.4:
            alpha = float(rng.uniform(0.8, 1.2))
            beta = float(rng.uniform(-0.1, 0.1))
            batch_x = (batch_x * alpha + beta).clamp(-1.0, 1.0)

        opt.zero_grad(set_to_none=True)
        with autocast_ctx():
            preds = model(batch_x)
            losses = detection_loss(preds, batch_t, cfg)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model, step)
        sched.step()

        if step % 200 == 0 or step == steps - 1:
            tot = float(losses['total'].detach())
            obj_l = float(losses['obj'].detach())
            box_l = float(losses['box'].detach())
            cls_l = float(losses['cls'].detach())
            elapsed = time.time() - t0
            step_speed = (step + 1) / max(0.1, elapsed)
            print(f"step {step:4d}/{steps} | loss: {tot:.3f} (obj: {obj_l:.3f}, box: {box_l:.3f}, cls: {cls_l:.3f}) | "
                  f"pos: {losses['num_pos']:.0f} | speed: {step_speed:.1f} steps/s")

    _save_ckpt(ckpt_path, model, opt, sched, steps - 1, cfg, profile, ema=ema)
    print(f"\n[{profile}] Training Complete! Checkpoint saved -> {ckpt_path}")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["mcu", "pro", "gpu"], default="mcu")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_voc(args)


if __name__ == "__main__":
    main()

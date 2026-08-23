"""Train NanoStream-OD on synthetic shapes (CPU-friendly, minutes not hours)."""

import argparse
import pathlib

import numpy as np
import torch

from .data import SyntheticShapes, collate, make_sample
from .head import decode_detections, detection_loss
from .model import NanoStreamOD


def evaluate_recall(model, n=64, seed=999, conf=0.3, iou_thr=0.5):
    rng = np.random.default_rng(seed)
    hits = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            img, boxes_px, labels = make_sample(160, 3, rng)
            x = (torch.from_numpy(img).float() / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
            preds = model(x)
            dets = decode_detections(preds, conf)
            matched = set()
            for gi, gb in enumerate(boxes_px):
                gx1, gy1, gx2, gy2 = [v / 160.0 for v in gb]
                best = -1
                best_iou = 0
                for di in range(dets.shape[0]):
                    if int(dets[di, 5]) != labels[gi] or di in matched:
                        continue
                    bx1, by1, bx2, by2 = dets[di, :4].tolist()
                    ix = max(0, min(bx2, gx2) - max(bx1, gx1))
                    iy = max(0, min(by2, gy2) - max(by1, gy1))
                    inter = ix * iy
                    union = ((bx2 - bx1) * (by2 - by1) +
                             (gx2 - gx1) * (gy2 - gy1) - inter)
                    iou = inter / max(union, 1e-9)
                    if iou > best_iou:
                        best_iou = iou
                        best = di
                total += 1
                if best_iou >= iou_thr:
                    hits += 1
                    matched.add(best)
    return hits / max(1, total), hits, total


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # FIX: seed numpy too (SyntheticShapes uses it)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = NanoStreamOD().to(device)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SyntheticShapes(length=args.steps * args.batch, seed=args.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=5e-4)
    warmup_steps = min(50, args.steps // 10)

    # FIX: use the same LambdaLR warmup+cosine as train_faces. The old manual
    # pg['lr'] = ... assignment was immediately overwritten by CosineAnnealingLR's
    # step(), so the warmup ramp never actually happened.
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, args.steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model.train()
    running = []
    rng = np.random.default_rng(args.seed + 42)
    for step in range(args.steps):
        # BUG-11 FIX: Random sampling instead of always using indices 0..batch-1
        indices = rng.integers(0, len(ds), size=args.batch)
        xs, ts = collate([ds[int(i)] for i in indices])
        xs = xs.to(device)
        ts = [{k: v.to(device) for k, v in t.items()} for t in ts]
        preds = model(xs)
        losses = detection_loss(preds, ts, model.cfg)
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        running.append(float(losses["total"].detach()))
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:4d}/{args.steps} "
                  f"loss {losses['total']:.4f} "
                  f"obj {float(losses['obj']):.4f} "
                  f"box {float(losses['box']):.4f} "
                  f"l1 {float(losses.get('l1', 0)):.4f} "
                  f"cls {float(losses['cls']):.4f} "
                  f"pos {losses['num_pos']:.0f}")
    recall, hits, total = evaluate_recall(model, n=64, seed=999, conf=0.3)
    print(f"val recall@{args.steps}: {recall:.2%} ({hits}/{total})")
    ckpt = {"model": model.state_dict(), "config": vars(model.cfg)}
    torch.save(ckpt, out_dir / "nanostream_shapes.pt")
    print(f"saved -> {out_dir / 'nanostream_shapes.pt'}")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/shapes")
    p.add_argument("--device", type=str, default="", help="torch device (cuda/cpu)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

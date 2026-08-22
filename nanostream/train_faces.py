"""Train NanoStream-OD on REAL face data captured from webcam.

This training pipeline uses real images auto-labeled by OpenCV's YuNet
face detector. It produces a face detector that works on real webcam input
— not synthetic drawn ellipses.

Usage:
    # Step 1: Capture real training data from webcam (once)
    python -m nanostream.dataset --capture 400

    # Step 2: Train on captured data
    python -m nanostream.train_faces --steps 2000

    # Or do both in one command (auto-captures if no data exists)
    python -m nanostream.train_faces --auto-capture 400 --steps 2000
"""

import argparse
import pathlib
import time
import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from .config import NanoStreamConfig
from .data import collate
from .dataset import DATA_DIR, FACE_CLASSES, WebcamFaceDataset, capture_webcam_training_data
from .head import decode_detections, detection_loss
from .model import NanoStreamOD


def evaluate_real_recall(model, dataset, n=100, conf=0.25, iou_thr=0.40):
    """Evaluate detection recall on held-out real face data."""
    model.eval()
    hits = 0
    total_gt = 0
    false_pos = 0
    n_eval = min(n, len(dataset))
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(n_eval):
            x, tgt = dataset[i]
            x = x.unsqueeze(0).to(device)  # (1, 1, H, W)
            preds = model(x)
            dets = decode_detections(preds, conf)

            gt_boxes = tgt["boxes_norm"].to(device)
            if len(gt_boxes) == 0:
                false_pos += dets.shape[0]
                continue

            # Convert GT from cxcywh to xyxy for IoU
            gt_xyxy = torch.stack([
                gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
                gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
                gt_boxes[:, 0] + gt_boxes[:, 2] / 2,
                gt_boxes[:, 1] + gt_boxes[:, 3] / 2,
            ], dim=1)

            matched_det = set()
            for gi in range(gt_xyxy.shape[0]):
                gx1, gy1, gx2, gy2 = gt_xyxy[gi].tolist()
                best_iou = 0
                best_di = -1
                for di in range(dets.shape[0]):
                    if di in matched_det:
                        continue
                    dx1, dy1, dx2, dy2 = dets[di, :4].tolist()
                    ix = max(0, min(dx2, gx2) - max(dx1, gx1))
                    iy = max(0, min(dy2, gy2) - max(dy1, gy1))
                    inter = ix * iy
                    union = ((dx2 - dx1) * (dy2 - dy1) +
                             (gx2 - gx1) * (gy2 - gy1) - inter)
                    iou = inter / max(union, 1e-9)
                    if iou > best_iou:
                        best_iou = iou
                        best_di = di
                total_gt += 1
                if best_iou >= iou_thr:
                    hits += 1
                    matched_det.add(best_di)

            false_pos += max(0, dets.shape[0] - len(matched_det))

    recall = hits / max(1, total_gt)
    precision = hits / max(1, hits + false_pos)
    return recall, precision, hits, total_gt


def train_faces(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = NanoStreamConfig(input_size=160, num_classes=1)
    model = NanoStreamOD(cfg)

    # Check if real data exists, otherwise auto-capture
    data_dir = pathlib.Path(args.data_dir)
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        if args.auto_capture > 0:
            print("No training data found. Capturing from webcam...")
            n_captured = capture_webcam_training_data(
                n_frames=args.auto_capture,
                camera_id=args.camera,
                output_dir=data_dir,
            )
            if n_captured < 20:
                print(f"ERROR: Only captured {n_captured} frames with faces. "
                      f"Need at least 20. Try moving your face closer to the camera.")
                return None
        else:
            print(f"ERROR: No training data found at {data_dir}")
            print(f"Run: python -m nanostream.dataset --capture 400")
            print(f"Or: python -m nanostream.train_faces --auto-capture 400")
            return None

    device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    # Load real dataset
    train_ds = WebcamFaceDataset(data_dir, augment=True, cache_in_ram=True)
    eval_ds = WebcamFaceDataset(data_dir, augment=False, cache_in_ram=True)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_params = model.param_count()
    n_samples = len(train_ds)

    print("=" * 62)
    print("  NanoStream-OD Face Detector Training (REAL DATA)")
    print(f"  Device     : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Resolution : {cfg.input_size}x{cfg.input_size} Grayscale")
    print(f"  Parameters : {total_params:,} ({total_params * 2 / 1024:.1f} KB @ int16)")
    print(f"  Real Frames: {n_samples}")
    print(f"  Steps      : {args.steps} | Batch: {args.batch} | LR: {args.lr}")
    print(f"  Grid       : {cfg.grid_size}x{cfg.grid_size} ({cfg.grid_size**2} cells)")
    print("=" * 62)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup + Cosine schedule
    warmup_steps = min(100, args.steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, args.steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    t0 = time.perf_counter()
    best_recall = 0.0

    for step in range(args.steps):
        # Sample random batch from real data
        batch_items = []
        for _ in range(args.batch):
            idx = np.random.randint(0, n_samples)
            batch_items.append(train_ds[idx])
        xs, ts = collate(batch_items)
        xs = xs.to(device)
        ts = [{k: v.to(device) for k, v in t.items()} for t in ts]

        opt.zero_grad()
        preds = model(xs)
        losses = detection_loss(preds, ts, cfg, w_obj=3.0, w_box=3.0, w_l1=1.0, w_cls=0.5)

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == args.steps - 1:
            # Quick diagnostic on a training sample
            model.eval()
            with torch.no_grad():
                test_x, test_tgt = eval_ds[0]
                test_preds = model(test_x.unsqueeze(0).to(device))
                obj_sig = torch.sigmoid(test_preds["obj"][0, 0])
                max_obj = obj_sig.max().item()
                n_above = (obj_sig > 0.25).sum().item()
            model.train()

            lr_now = opt.param_groups[0]["lr"]
            print(f"step {step:5d}/{args.steps}  "
                  f"loss {float(losses['total'].detach()):.3f}  "
                  f"obj {float(losses['obj']):.4f}  "
                  f"box {float(losses['box']):.4f}  "
                  f"pos {losses['num_pos']:.1f}  "
                  f"lr {lr_now:.5f}  "
                  f"max_sig {max_obj:.3f}  "
                  f"cells>{n_above}")

        # Periodic full evaluation
        if (step + 1) % 250 == 0 or step == args.steps - 1:
            recall, prec, h, t = evaluate_real_recall(model, eval_ds, n=80, conf=0.25)
            print(f"  >>> Eval @ step {step+1}: "
                  f"Recall={recall:.1%} Precision={prec:.1%} ({h}/{t})")
            if recall > best_recall:
                best_recall = recall
                ckpt = {
                    "model": model.state_dict(),
                    "config": vars(model.cfg),
                    "classes": list(FACE_CLASSES),
                    "step": step + 1,
                    "recall": recall,
                    "precision": prec,
                    "data_source": "webcam_real",
                }
                torch.save(ckpt, out_dir / "nanostream_faces_best.pt")
            model.train()

    t1 = time.perf_counter()
    print(f"\nTraining completed in {t1 - t0:.1f}s.")

    # Final evaluation
    model.eval()
    recall, prec, hits, total = evaluate_real_recall(model, eval_ds, n=120, conf=0.25)
    print(f"Final Recall @ IoU 0.40: {recall:.1%} ({hits}/{total})")
    print(f"Final Precision: {prec:.1%}")

    # Save final checkpoint
    ckpt_path = out_dir / "nanostream_faces.pt"
    ckpt = {
        "model": model.state_dict(),
        "config": vars(model.cfg),
        "classes": list(FACE_CLASSES),
        "recall": recall,
        "precision": prec,
        "data_source": "webcam_real",
    }
    torch.save(ckpt, ckpt_path)
    print(f"Checkpoint -> {ckpt_path.resolve()}")

    # Use best if it's better
    best_path = out_dir / "nanostream_faces_best.pt"
    if best_path.exists() and best_recall > recall:
        import shutil
        shutil.copy2(best_path, ckpt_path)
        print(f"Restored best checkpoint (recall {best_recall:.1%})")

    return model


def main():
    p = argparse.ArgumentParser(description="Train NanoStream-OD Face Detector on Real Data")
    p.add_argument("--steps", type=int, default=2000,
                   help="Training steps (default: 2000)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--lr", type=float, default=3e-3,
                   help="Peak learning rate (default: 3e-3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--out", type=str, default="runs/faces",
                   help="Output directory for checkpoints")
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                   help="Path to webcam face data directory")
    p.add_argument("--auto-capture", type=int, default=400,
                   help="Auto-capture N webcam frames if no data exists (0 to disable)")
    p.add_argument("--camera", type=int, default=0,
                   help="Webcam device index for auto-capture")
    args = p.parse_args()
    train_faces(args)


if __name__ == "__main__":
    main()

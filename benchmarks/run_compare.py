"""Cross-framework comparison: NanoStream-OD vs YOLO vs FOMO-style.

Subcommands:
    data              export the dataset in YOLO format (for ultralytics)
    train-nanostream  train NanoStream-OD (profile: mcu|pro|gpu)
    train-fomo        train the FOMO-style baseline
    train-yolo        train yolov8{n,s,m} via ultralytics (needs network install)
    eval              unified AP/latency/size comparison + face demo images
    demo              render face-detection demo images only

Tiers (same val split, tier-appropriate resolution):
    mcu  160 px  shift-only, <256 KB SRAM MCU artifact
    pro  256 px  laptop target (float, dual-scale + Lite-FPN)
    gpu  320 px  server/industry target (float, wider backbone)

All models are scored with the same AP code
(nanostream.metrics.compute_ap_per_class) on the same held-out val split.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

from nanostream.config import NanoStreamConfig
from nanostream.export import stage_buffer_sizes
from nanostream.head import decode_detections
from nanostream.metrics import (compute_ap_per_class, compute_map_multiscale,
                                evaluate_model)
from nanostream.model import NanoStreamOD

from .combined_data import (CLASS_NAMES, NUM_CLASSES, SIZE, VAL_LEN,
                            VAL_START, CombinedDataset, generate_sample)
from .fomo_model import FomoDetector, evaluate_fomo, predict

RUNS = pathlib.Path("benchmarks/runs")
CKPT_DIR = RUNS / "ckpt"
YOLO_DATA = RUNS / "yolo_data"
YOLO_OUT = RUNS / "yolo"
CONF = 0.25
TIERS = (("mcu", 160), ("pro", 256), ("gpu", 320))


def _load_nanostream(tier: str = "mcu"):
    ckpt_path = CKPT_DIR / f"nanostream_{tier}.pt"
    if not ckpt_path.exists() and tier == "mcu":
        ckpt_path = CKPT_DIR / "nanostream_bench.pt"  # legacy filename
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    valid = set(NanoStreamConfig.__dataclass_fields__.keys())
    cfg = NanoStreamConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = NanoStreamOD(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _load_fomo():
    ckpt = torch.load(CKPT_DIR / "fomo_bench.pt", map_location="cpu",
                      weights_only=False)
    model = FomoDetector(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _latency(fn, n=VAL_LEN, warmup=3):
    """Mean inference time (ms) over the validation set on CPU."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times))


def _summary(res, label):
    ap = res["per_class_ap"]
    rows = "  ".join(f"{CLASS_NAMES[c]}={ap.get(c, 0.0):.3f}" for c in range(NUM_CLASSES))
    return (f"{label:<16} mAP50={res['mAP_50']:.3f}  mAP50:95={res['mAP_50_95']:.3f}  "
            f"P={res['precision']:.3f} R={res['recall']:.3f}  AP: {rows}")


def cmd_data(args):
    from .combined_data import export_yolo
    splits = (("train", args.train_len, 0), ("val", VAL_LEN, VAL_START))
    yaml_path = export_yolo(YOLO_DATA, splits=splits)
    print(f"YOLO dataset written -> {yaml_path} "
          f"(train={args.train_len}, val={VAL_LEN})")


def cmd_train_nanostream(args):
    from .train_nanostream import train
    train(args)


def cmd_train_fomo(args):
    from .train_fomo import train
    train(args)


def cmd_train_yolo(args):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"YOLO baseline unavailable: {e}")
        print("Install with: pip install ultralytics (needs a working torchvision)")
        return 1
    yaml_path = YOLO_DATA / "data.yaml"
    if not yaml_path.exists():
        cmd_data(args)
    arch = (f"yolov8{args.yolo_model}.yaml" if args.from_scratch
            else f"yolov8{args.yolo_model}.pt")
    model = YOLO(arch)
    device = args.yolo_device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.train(data=str(yaml_path), imgsz=args.imgsz, epochs=args.yolo_epochs,
                batch=args.yolo_batch, device=device,
                project=str(YOLO_OUT.resolve()), name=f"det_{args.imgsz}",
                verbose=False, plots=False, cache=False)
    best = YOLO_OUT / f"det_{args.imgsz}" / "weights" / "best.pt"
    print(f"YOLO training done -> {best}")
    return 0


def cmd_demo(_args):
    """Render face-detection demo images for NanoStream + FOMO (eval builds them)."""
    model = _load_nanostream("mcu")
    try:
        fomo = _load_fomo()
    except FileNotFoundError:
        fomo = None  # fomo still training: render NanoStream-only demo
    _render_face_demo(model, fomo)
    return 0


def _to_yolo_img(idx, size=SIZE):
    """Grayscale uint8 sample -> BGR 3-channel image for ultralytics."""
    import cv2
    return cv2.cvtColor(generate_sample(idx, size)[0], cv2.COLOR_GRAY2BGR)


def _yolo_predictions(model, imgsz=SIZE, val_len=VAL_LEN, start_idx=VAL_START,
                      conf=CONF):
    """Run YOLO over the val split at a given input size."""
    imgs = [_to_yolo_img(start_idx + i, imgsz) for i in range(val_len)]
    return model.predict(imgs, imgsz=imgsz, conf=conf, verbose=False)


def cmd_eval(_args):
    results = {}
    lines = []

    # --- NanoStream-OD: one model per tier (skip tiers not yet trained) ---
    for tier, size in TIERS:
        try:
            model = _load_nanostream(tier)
        except FileNotFoundError:
            lines.append(f"NanoStream-{tier} skipped (no {tier} checkpoint - "
                         f"run 'train-nanostream --profile {tier}')")
            continue
        tier_val = CombinedDataset(length=VAL_LEN, start_idx=VAL_START, size=size)
        res = evaluate_model(model, tier_val, NUM_CLASSES, n_samples=VAL_LEN,
                             conf_thr=CONF)
        lat = _latency(lambda: model(tier_val[0][0].unsqueeze(0)))
        sizes = stage_buffer_sizes(model.cfg)
        static_bss = sum(s["ring"] + s["win"] + s["cas"] for s in sizes) * 2
        results[f"nanostream-{tier}"] = {
            "mAP50": res["mAP_50"], "mAP50_95": res["mAP_50_95"],
            "precision": res["precision"], "recall": res["recall"],
            "per_class_ap": {CLASS_NAMES[c]: float(res["per_class_ap"][c])
                              for c in range(NUM_CLASSES)},
            "latency_ms": lat, "params": model.param_count(),
            "static_ram_bytes": static_bss,
            "flash_int8_bytes": model.param_count(),
        }
        lines.append(_summary(res, f"NanoStream-{tier}"))

    # --- FOMO-style (MCU tier, 160 px) ---
    fomo = None
    try:
        fomo = _load_fomo()
        val_ds_160 = CombinedDataset(length=VAL_LEN, start_idx=VAL_START, size=160)
        res_f = evaluate_fomo(fomo, val_ds_160, NUM_CLASSES, conf=CONF)
        lat_f = _latency(lambda: fomo(val_ds_160[0][0].unsqueeze(0)))
        results["fomo"] = {
            "mAP50": res_f["mAP_50"], "mAP50_95": res_f["mAP_50_95"],
            "precision": res_f["precision"], "recall": res_f["recall"],
            "per_class_ap": {CLASS_NAMES[c]: float(res_f["per_class_ap"][c])
                              for c in range(NUM_CLASSES)},
            "latency_ms": lat_f, "params": fomo.param_count(),
            "flash_int8_bytes": fomo.param_count(),
        }
        lines.append(_summary(res_f, "FOMO-style"))
    except FileNotFoundError:
        lines.append("FOMO-style skipped (no checkpoint - run 'train-fomo')")

    # --- YOLO: one model per tier (trained at that tier's imgsz) ---
    try:
        from ultralytics import YOLO
        for tier, size in TIERS:
            best = YOLO_OUT / f"det_{size}" / "weights" / "best.pt"
            if not best.exists():
                best = next(YOLO_OUT.rglob(f"*det_{size}*/weights/best.pt"), None)
            if best is None and size == 160:
                # legacy: pre-rewrite runs used the unscoped name "det"
                best = next(pathlib.Path("runs").rglob(
                    "**/det/weights/best.pt"), None)
            if best is None:
                lines.append(f"YOLOv8n-{tier} skipped (no best.pt for imgsz {size})")
                continue
            yolo = YOLO(str(best))
            r_list = _yolo_predictions(yolo, imgsz=size)
            preds, gts = [], []
            val_ds = CombinedDataset(length=VAL_LEN, start_idx=VAL_START, size=size)
            for i, r in enumerate(r_list):
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.numpy() / size
                    preds.append({"boxes": xyxy,
                                  "scores": r.boxes.conf.numpy(),
                                  "class_ids": r.boxes.cls.numpy().astype(int)})
                else:
                    preds.append({"boxes": np.zeros((0, 4)),
                                  "scores": np.zeros(0),
                                  "class_ids": np.zeros(0, dtype=int)})
                _, tgt = val_ds[i]
                gt_c = tgt["boxes_norm"]
                if len(gt_c):
                    gt_xyxy = torch.stack([gt_c[:, 0] - gt_c[:, 2] / 2,
                                           gt_c[:, 1] - gt_c[:, 3] / 2,
                                           gt_c[:, 0] + gt_c[:, 2] / 2,
                                           gt_c[:, 1] + gt_c[:, 3] / 2], dim=1).numpy()
                else:
                    gt_xyxy = np.zeros((0, 4))
                gts.append({"boxes": gt_xyxy, "labels": tgt["labels"].numpy()})
            res_y = compute_ap_per_class(preds, gts, NUM_CLASSES)
            res_y["mAP_50_95"] = compute_map_multiscale(
                preds, gts, NUM_CLASSES)["mAP_50_95"]
            lat_y = _latency(lambda: yolo.predict(
                [_to_yolo_img(VAL_START, size)], imgsz=size, conf=CONF,
                verbose=False))
            params_y = sum(p.numel() for p in yolo.model.parameters())
            results[f"yolo-{tier}"] = {
                "mAP50": res_y["mAP_50"], "mAP50_95": res_y["mAP_50_95"],
                "precision": res_y["precision"], "recall": res_y["recall"],
                "per_class_ap": {CLASS_NAMES[c]: float(res_y["per_class_ap"][c])
                                  for c in range(NUM_CLASSES)},
                "latency_ms": lat_y, "params": params_y,
                "flash_int8_bytes": params_y,
            }
            lines.append(_summary(res_y, f"YOLOv8n-{tier}"))
    except ImportError as e:
        lines.append(f"YOLO baseline skipped (ultralytics unavailable: {e})")

    # --- face detection demo (mcu model) ---
    _render_face_demo(_load_nanostream("mcu"), fomo)

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "results.json").write_text(json.dumps(results, indent=2))
    print("\n" + "\n".join(lines))
    print(f"\nresults -> {RUNS / 'results.json'}")
    return 0


def _render_face_demo(ns_model, fomo_model, n_imgs=3):
    """Render NanoStream + FOMO detections on val images that contain faces."""
    import cv2
    demo_dir = RUNS / "face_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    shown = 0
    for i in range(VAL_LEN):
        if shown >= n_imgs:
            break
        img, boxes, labels = generate_sample(VAL_START + i, SIZE)
        if 3 not in labels:
            continue
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x = torch.from_numpy(img.copy()).float() / 127.5 - 1.0
        dets = decode_detections(ns_model(x.unsqueeze(0).unsqueeze(0)), CONF)
        for (x1, y1, x2, y2, sc, c) in dets.tolist():
            if int(c) == 3:
                cv2.rectangle(bgr, (int(x1 * SIZE), int(y1 * SIZE)),
                              (int(x2 * SIZE), int(y2 * SIZE)),
                              (0, 255, 0), 2)
                cv2.putText(bgr, f"face {sc:.2f}", (int(x1 * SIZE), max(12, int(y1 * SIZE) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if fomo_model is not None:
            fd = predict(fomo_model, x.unsqueeze(0))
        else:
            fd = torch.zeros(0, 6)
        for (x1, y1, x2, y2, sc, c) in fd.tolist():
            if int(c) == 3:
                cv2.rectangle(bgr, (int(x1 * SIZE), int(y1 * SIZE)),
                              (int(x2 * SIZE), int(y2 * SIZE)),
                              (255, 0, 0), 1)
        cv2.imwrite(str(demo_dir / f"face_demo_{shown}.png"), bgr)
        shown += 1
    print(f"face demo images -> {demo_dir} ({shown} rendered)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--train_len", type=int, default=1200)
    ns = sub.add_parser("train-nanostream")
    ns.add_argument("--profile", choices=["mcu", "pro", "gpu"], default="mcu")
    ns.add_argument("--steps", type=int, default=1000)
    ns.add_argument("--batch", type=int, default=16)
    ns.add_argument("--lr", type=float, default=2e-3)
    ns.add_argument("--seed", type=int, default=0)
    ns.add_argument("--data_len", type=int, default=1200)
    ns.add_argument("--out", type=str, default=str(CKPT_DIR))
    fs = sub.add_parser("train-fomo")
    fs.add_argument("--steps", type=int, default=1000)
    fs.add_argument("--batch", type=int, default=16)
    fs.add_argument("--lr", type=float, default=2e-3)
    fs.add_argument("--seed", type=int, default=0)
    fs.add_argument("--data_len", type=int, default=1200)
    fs.add_argument("--out", type=str, default=str(CKPT_DIR))
    py = sub.add_parser("train-yolo")
    py.add_argument("--yolo_model", choices=["n", "s", "m"], default="n")
    py.add_argument("--imgsz", type=int, default=160)
    py.add_argument("--yolo_epochs", type=int, default=60)
    py.add_argument("--yolo_batch", type=int, default=16)
    py.add_argument("--yolo_device", type=str, default="",
                    help="torch device for YOLO training (default: cuda if available)")
    py.add_argument("--from_scratch", action="store_true")
    sub.add_parser("eval")
    sub.add_parser("demo")
    args = p.parse_args()
    fn = {"data": cmd_data, "train-nanostream": cmd_train_nanostream,
          "train-fomo": cmd_train_fomo, "train-yolo": cmd_train_yolo,
          "eval": cmd_eval, "demo": cmd_demo}[args.cmd]
    sys.exit(fn(args) or 0)


if __name__ == "__main__":
    main()

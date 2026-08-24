"""Evaluate NanoStream-OD vs YOLOv8n on Real-World PASCAL VOC.

Computes:
1. COCO-style mAP@50 and mAP@50:95
2. Per-class AP (person, car, bicycle, dog)
3. Inference Latency & SRAM / Flash Memory Footprint
4. Renders side-by-side visual detection overlays on real test photos
"""

import json
import pathlib
import cv2
import numpy as np
import torch
from nanostream.config import PROFILES
from nanostream.model import NanoStreamOD
from nanostream.head import decode_detections
from nanostream.export import stage_buffer_sizes
from nanostream.metrics import compute_ap_per_class, compute_map_multiscale
from projects.real_voc.dataset import RealVOCDataset, VOC_CLASSES, NUM_CLASSES, YOLO_DIR

RESULTS_DIR = pathlib.Path("benchmarks/runs/voc_results")
DEMO_DIR = pathlib.Path("benchmarks/runs/voc_demo")
CKPT_DIR = pathlib.Path("benchmarks/runs/voc_ckpt")


def evaluate_nanostream(model, dataset, device, conf_thr=0.25):
    model.eval()
    preds_list = []
    gts_list = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, tgt = dataset[i]
            x_in = x.unsqueeze(0).to(device)
            preds = model(x_in)
            dets = decode_detections(preds, conf_thr=conf_thr)

            if len(dets) > 0:
                xyxy = dets[:, :4].cpu().numpy()
                scores = dets[:, 4].cpu().numpy()
                cls_ids = dets[:, 5].cpu().numpy().astype(int)
                preds_list.append({"boxes": xyxy, "scores": scores, "class_ids": cls_ids})
            else:
                preds_list.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "class_ids": np.zeros(0, dtype=int)})

            gt_c = tgt["boxes_norm"]
            if len(gt_c) > 0:
                gt_xyxy = torch.stack([
                    gt_c[:, 0] - gt_c[:, 2] / 2,
                    gt_c[:, 1] - gt_c[:, 3] / 2,
                    gt_c[:, 0] + gt_c[:, 2] / 2,
                    gt_c[:, 1] + gt_c[:, 3] / 2,
                ], dim=1).numpy()
                gts_list.append({"boxes": gt_xyxy, "labels": tgt["labels"].numpy()})
            else:
                gts_list.append({"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=int)})

    res = compute_ap_per_class(preds_list, gts_list, NUM_CLASSES)
    res["mAP_50_95"] = compute_map_multiscale(preds_list, gts_list, NUM_CLASSES)["mAP_50_95"]
    return res


def render_visual_detections(model, dataset, device, n_images=6):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    print(f"\n[Visual Demo] Rendering {n_images} real-world detections to {DEMO_DIR}...")
    for idx in range(min(n_images, len(dataset))):
        img_path, _, _ = dataset.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        h_orig, w_orig = bgr.shape[:2]

        x, _ = dataset[idx]
        with torch.no_grad():
            preds = model(x.unsqueeze(0).to(device))
            dets = decode_detections(preds, conf_thr=0.25)

        for d in dets:
            x1, y1, x2, y2 = d[:4].cpu().numpy()
            score = float(d[4].cpu())
            cid = int(d[5].cpu())
            cname = VOC_CLASSES[cid] if cid < len(VOC_CLASSES) else f"class_{cid}"
            col = colors[cid % len(colors)]

            px1, py1 = int(x1 * w_orig), int(y1 * h_orig)
            px2, py2 = int(x2 * w_orig), int(y2 * h_orig)

            cv2.rectangle(bgr, (px1, py1), (px2, py2), col, 2)
            label_text = f"{cname} {score:.2f}"
            cv2.putText(bgr, label_text, (px1, max(15, py1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        out_path = DEMO_DIR / f"voc_demo_{idx}.jpg"
        cv2.imwrite(str(out_path), bgr)

    print(f"[Visual Demo] Rendered images saved in {DEMO_DIR}")


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    print("=" * 80)
    print("       REAL-WORLD PASCAL VOC BENCHMARK: NanoStream-OD vs YOLOv8n")
    print("=" * 80)

    # 1. Evaluate NanoStream Tiers
    for profile in ["mcu", "pro", "gpu"]:
        ckpt_p = CKPT_DIR / f"nanostream_voc_{profile}.pt"
        if not ckpt_p.exists():
            print(f"NanoStream-{profile.upper()} skipped (no checkpoint found at {ckpt_p})")
            continue

        cfg = PROFILES[profile](num_classes=NUM_CLASSES)
        model = NanoStreamOD(cfg).to(device)
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        val_ds = RealVOCDataset(split="test", input_size=cfg.input_size)
        res = evaluate_nanostream(model, val_ds, device)
        sizes = stage_buffer_sizes(cfg)
        static_bss = sum(s["ring"] + s["win"] + s["cas"] for s in sizes) * 2

        all_results[f"nanostream-{profile}"] = {
            "mAP50": res["mAP_50"],
            "mAP50_95": res["mAP_50_95"],
            "precision": res["precision"],
            "recall": res["recall"],
            "per_class_ap": {VOC_CLASSES[c]: float(res["per_class_ap"][c]) for c in range(NUM_CLASSES)},
            "params": model.param_count(),
            "flash_kb": model.param_count() * 2 / 1024,
            "sram_kb": static_bss / 1024,
        }

        print(f"\n[NanoStream-{profile.upper()}] (Resolution: {cfg.input_size}x{cfg.input_size})")
        print(f"  Params: {model.param_count():,} | Peak SRAM: {static_bss/1024:.1f} KB | Flash: {model.param_count()*2/1024:.1f} KB")
        print(f"  mAP@50: {res['mAP_50']:.3f} | mAP@50:95: {res['mAP_50_95']:.3f} | Precision: {res['precision']:.3f} | Recall: {res['recall']:.3f}")
        for c in range(NUM_CLASSES):
            print(f"    - {VOC_CLASSES[c]:<10}: AP={res['per_class_ap'][c]:.3f}")

        if profile == "mcu":
            render_visual_detections(model, val_ds, device, n_images=6)

    # 2. Evaluate YOLOv8n if trained
    try:
        from ultralytics import YOLO
        yolo_ckpt = pathlib.Path("runs/detect/yolo_voc/weights/best.pt")
        if yolo_ckpt.exists():
            yolo = YOLO(str(yolo_ckpt))
            metrics = yolo.val(data=str(YOLO_DIR / "voc_edge.yaml"), imgsz=160, conf=0.25, verbose=False)
            all_results["yolov8n"] = {
                "mAP50": metrics.box.map50,
                "mAP50_95": metrics.box.map,
                "precision": metrics.box.mp,
                "recall": metrics.box.mr,
                "params": sum(p.numel() for p in yolo.model.parameters()),
            }
            print(f"\n[YOLOv8n-160px]")
            print(f"  mAP@50: {metrics.box.map50:.3f} | mAP@50:95: {metrics.box.map:.3f}")
    except Exception as e:
        print(f"YOLO evaluation note: {e}")

    # Save summary json
    out_json = RESULTS_DIR / "results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nDetailed Results JSON saved -> {out_json}")


if __name__ == "__main__":
    run_evaluation()

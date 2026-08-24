"""Master Orchestration CLI for Real-World VOC Object Detection Project.

Usage:
  python -m projects.real_voc.run_all data                 # Download & prepare dataset + YOLO format
  python -m projects.real_voc.run_all train-nanostream     # Train NanoStream-OD MCU/Pro/GPU
  python -m projects.real_voc.run_all train-yolo           # Train YOLOv8n baseline
  python -m projects.real_voc.run_all eval                 # Run full mAP benchmark & render visual demo
"""

import argparse
import sys


def cmd_data(args):
    from projects.real_voc.dataset import export_voc_to_yolo
    export_voc_to_yolo()


def cmd_train_nanostream(args):
    from projects.real_voc.train_nanostream import train_voc
    train_voc(args)


def cmd_train_yolo(args):
    import pathlib
    from ultralytics import YOLO
    from projects.real_voc.dataset import export_voc_to_yolo, YOLO_DIR

    yaml_p = YOLO_DIR / "voc_edge.yaml"
    if not yaml_p.exists():
        yaml_p = export_voc_to_yolo()

    print(f"[YOLOv8n] Training on Real VOC ({args.epochs} epochs, imgsz={args.imgsz})...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_p),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name="yolo_voc",
        exist_ok=True,
        device=args.device or "",
        plots=True,
    )


def cmd_eval(args):
    from projects.real_voc.evaluate_and_compare import run_evaluation
    run_evaluation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. Data Subcommand
    d = sub.add_parser("data", help="Download and prepare real VOC dataset")

    # 2. Train NanoStream Subcommand
    ns = sub.add_parser("train-nanostream", help="Train NanoStream-OD on real VOC")
    ns.add_argument("--profile", choices=["mcu", "pro", "gpu"], default="mcu")
    ns.add_argument("--steps", type=int, default=3000)
    ns.add_argument("--batch", type=int, default=32)
    ns.add_argument("--lr", type=float, default=2e-3)
    ns.add_argument("--device", type=str, default="")
    ns.add_argument("--seed", type=int, default=42)

    # 3. Train YOLO Subcommand
    yolo_p = sub.add_parser("train-yolo", help="Train YOLOv8n on real VOC")
    yolo_p.add_argument("--epochs", type=int, default=40)
    yolo_p.add_argument("--imgsz", type=int, default=160)
    yolo_p.add_argument("--batch", type=int, default=32)
    yolo_p.add_argument("--device", type=str, default="")

    # 4. Evaluation Subcommand
    ev = sub.add_parser("eval", help="Evaluate all models and render visual overlays")

    args = parser.parse_args()
    if args.cmd == "data":
        cmd_data(args)
    elif args.cmd == "train-nanostream":
        cmd_train_nanostream(args)
    elif args.cmd == "train-yolo":
        cmd_train_yolo(args)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()

"""Interactive webcam & simulation demo for NanoStream-OD.
Features a live on-screen HUD (SRAM gauge, FPS, MCU latency) and terminal diagnostic dashboard.
"""

import argparse
import pathlib
import sys
import time
import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from .config import NanoStreamConfig
from .data import CLASS_NAMES, calibration_images, render_frame_webcam_like
from .faces import make_face_sample
from .fixedpoint import calibrate_fixed_point
from .model import NanoStreamOD
from .tracker import ResourceTracker, enable_windows_ansi, print_dashboard


CLASS_COLORS = [
    (0, 230, 255),   # 0 (Face / Circle): Bright Cyan
    (50, 240, 100),  # 1 (Square): Emerald Green
    (0, 140, 255),   # 2 (Triangle): Amber / Orange
    (255, 100, 200), # 3: Pink / Magenta
]


def draw_hud(frame_bgr, dets, tracker_summary, fps, conf_thr=0.20,
             class_names=CLASS_NAMES, mode="stream", width=640, height=480):
    """Render rich on-screen HUD overlay with SRAM meter, FPS, and MCU cycle badge."""
    overlay = frame_bgr.copy()

    # Top banner bar
    cv2.rectangle(overlay, (0, 0), (width, 42), (20, 20, 24), -1)
    cv2.line(overlay, (0, 42), (width, 42), (60, 60, 70), 1)

    cv2.putText(overlay, "NanoStream-OD", (14, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"| Zero-NMS Detector (conf: {conf_thr:.2f})", (170, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    fps_txt = f"{fps:4.1f} FPS" if fps > 0 else "-- FPS"
    cv2.putText(overlay, fps_txt, (width - 105, 27),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 128), 1, cv2.LINE_AA)

    # Bottom diagnostic panel
    panel_h = 105
    panel_y0 = height - panel_h
    cv2.rectangle(overlay, (0, panel_y0), (width, height), (15, 16, 20), -1)
    cv2.line(overlay, (0, panel_y0), (width, panel_y0), (50, 55, 65), 1)

    # SRAM meter
    sram_kb = tracker_summary["peak_sram_kb_mcu"] if mode == "stream" else tracker_summary["peak_sram_kb_sim"]
    sram_max = 256.0
    sram_pct = min(1.0, sram_kb / sram_max)

    bar_x, bar_y, bar_w, bar_h = 14, panel_y0 + 26, 200, 14
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 48), -1)

    fill_w = int(bar_w * sram_pct)
    fill_color = (0, 220, 100) if sram_pct < 0.85 else (0, 140, 255)
    if fill_w > 0:
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 85, 95), 1)

    cv2.putText(overlay, f"Peak SRAM: {sram_kb:.1f} KB / {sram_max:.0f} KB ({sram_pct*100:.1f}%)",
                (bar_x, panel_y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Simulated MCU Latency Box
    m4_ms = tracker_summary.get("ms_cortex_m4_168mhz", 0)
    s3_ms = tracker_summary.get("ms_esp32_s3_240mhz", 0)
    cv2.putText(overlay, "MCU Latency Est:", (235, panel_y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Cortex-M4 @168MHz: {m4_ms:4.1f} ms", (235, panel_y0 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"ESP32-S3  @240MHz: {s3_ms:4.1f} ms", (235, panel_y0 + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 180), 1, cv2.LINE_AA)

    # Ops count
    macs_m = tracker_summary.get("macs", 0) / 1e6
    shifts_m = (tracker_summary.get("shifts", 0) + tracker_summary.get("adds", 0)) / 1e6
    cv2.putText(overlay, f"Ops: {macs_m:.2f}M MACs | {shifts_m:.2f}M Shifts", (bar_x, panel_y0 + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Mode: {mode.upper()} | [+/-] conf  [S] synth  [Q] quit", (bar_x, panel_y0 + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 180, 255), 1, cv2.LINE_AA)

    # Status / Detections count
    cv2.putText(overlay, f"Objects: {len(dets)} (Zero-NMS)", (width - 200, panel_y0 + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 100), 1, cv2.LINE_AA)

    # Blend overlay with original for sleek transparency
    cv2.addWeighted(overlay, 0.92, frame_bgr, 0.08, 0, frame_bgr)

    # Draw bounding boxes directly on top
    for det in dets:
        x1_n, y1_n, x2_n, y2_n, score, cls_id = det
        cls_id = int(cls_id)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"

        # Map to display coordinates
        disp_x1 = int(x1_n * width)
        disp_y1 = int(y1_n * height)
        disp_x2 = int(x2_n * width)
        disp_y2 = int(y2_n * height)

        # Skip tiny boxes (noise)
        if (disp_x2 - disp_x1) < 8 or (disp_y2 - disp_y1) < 8:
            continue

        # Draw bounding box with thickness
        cv2.rectangle(frame_bgr, (disp_x1, disp_y1), (disp_x2, disp_y2), color, 2, cv2.LINE_AA)

        # Draw corner markers for modern look
        cw = min(15, (disp_x2 - disp_x1) // 3)
        ch = min(15, (disp_y2 - disp_y1) // 3)
        cv2.line(frame_bgr, (disp_x1, disp_y1), (disp_x1 + cw, disp_y1), (255, 255, 255), 2)
        cv2.line(frame_bgr, (disp_x1, disp_y1), (disp_x1, disp_y1 + ch), (255, 255, 255), 2)
        cv2.line(frame_bgr, (disp_x2, disp_y2), (disp_x2 - cw, disp_y2), (255, 255, 255), 2)
        cv2.line(frame_bgr, (disp_x2, disp_y2), (disp_x2, disp_y2 - ch), (255, 255, 255), 2)

        # Draw label tag
        lbl = f"{name.upper()} {score*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(42, disp_y1 - lh - 6)
        cv2.rectangle(frame_bgr, (disp_x1, tag_y1),
                      (disp_x1 + lw + 8, tag_y1 + lh + 6), color, -1)
        cv2.putText(frame_bgr, lbl, (disp_x1 + 4, tag_y1 + lh + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

    return frame_bgr


def run_demo(args):
    enable_windows_ansi()
    if cv2 is None:
        print("ERROR: opencv-python is required to run the demo: pip install opencv-python")
        sys.exit(1)

    # Check for face model or shapes model
    model_path = pathlib.Path(args.model)
    if not model_path.exists() and pathlib.Path("runs/faces/nanostream_faces.pt").exists():
        model_path = pathlib.Path("runs/faces/nanostream_faces.pt")
    if not model_path.exists() and pathlib.Path("runs/shapes/nanostream_shapes.pt").exists():
        model_path = pathlib.Path("runs/shapes/nanostream_shapes.pt")

    classes = CLASS_NAMES
    if model_path.exists():
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            cfg_dict = ckpt["config"]
            valid_fields = set(NanoStreamConfig.__dataclass_fields__.keys())
            cfg = NanoStreamConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})
            model = NanoStreamOD(cfg)
        else:
            model = NanoStreamOD()
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if "classes" in ckpt:
            classes = tuple(ckpt["classes"])
        print(f"Loaded checkpoint: {model_path} (Classes: {classes})")
    else:
        model = NanoStreamOD()
        print("No checkpoint found. Using random weights (train first!).")

    model.eval()

    # CRITICAL: Do NOT call freeze_all() for float streaming mode!
    # freeze_all() converts weights to pow2 quantized form which destroys
    # float-precision objectness signal. Only freeze for int-mode export.
    if args.int_mode:
        model.freeze_all()

    tr = ResourceTracker.get()
    tr.set_flash(model.param_count())

    # Calibration for integer execution mode if requested
    calib_fracs = None
    if args.int_mode:
        print("Calibrating fixed-point dynamic ranges for integer execution...")
        calib_imgs = calibration_images(n=24, size=model.cfg.input_size)
        calib_fracs = calibrate_fixed_point(model, calib_imgs, frac_bits=model.cfg.frac_bits)
    # Initialize video capture
    cap = None
    use_synthetic = args.synthetic

    if not use_synthetic and args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Could not open video {args.video}, falling back to synthetic generator.")
            use_synthetic = True

    if not use_synthetic and cap is None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"No webcam detected on index {args.camera}. Switching to synthetic mode.")
            use_synthetic = True
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    conf_thr = args.conf
    print("\n" + "=" * 62)
    print("  NanoStream-OD Live Demo")
    print(f"  Input: {model.cfg.input_size}x{model.cfg.input_size} | "
          f"Pipeline: {'Int16 Bit-Exact' if args.int_mode else 'Float Streaming'}")
    print(f"  Classes: {', '.join(classes)} | Conf: {conf_thr:.2f}")
    print("  Controls: [+/-] threshold, [S] toggle synth, [Q/ESC] exit")
    print("=" * 62 + "\n")

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 30.0, (640, 480))
        print(f"Recording to: {args.record}")

    frame_idx = 0
    fps_smooth = 0.0
    rng = np.random.default_rng(42)

    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break

            if use_synthetic:
                if "face" in classes:
                    synth_img, _, _ = make_face_sample(
                        size=model.cfg.input_size, max_faces=2, rng=rng)
                else:
                    synth_img, _, _ = render_frame_webcam_like(
                        size=model.cfg.input_size, t=frame_idx * 0.05, rng=rng)
                display_frame = cv2.resize(
                    cv2.cvtColor(synth_img, cv2.COLOR_GRAY2BGR), (640, 480),
                    interpolation=cv2.INTER_NEAREST)
                gray_input = synth_img
            else:
                ret, bgr_frame = cap.read()
                if not ret:
                    if args.video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                display_frame = cv2.resize(bgr_frame, (640, 480))
                raw_gray = cv2.cvtColor(
                    cv2.resize(bgr_frame,
                               (model.cfg.input_size, model.cfg.input_size)),
                    cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_input = clahe.apply(raw_gray)

            # Inference
            t0 = time.perf_counter()
            if args.int_mode:
                img_u8 = torch.from_numpy(gray_input)
                dets_t = model.stream_forward_int(
                    img_u8, calib_fracs, conf_thr=conf_thr)
            else:
                img_f = (torch.from_numpy(gray_input).float() / 127.5 - 1.0
                         ).unsqueeze(0)
                dets_t, _ = model.stream_forward(img_f, conf_thr=conf_thr)

            t1 = time.perf_counter()
            dt = t1 - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smooth = (inst_fps if fps_smooth == 0.0
                          else 0.9 * fps_smooth + 0.1 * inst_fps)

            # Parse detections for HUD & dashboard
            dets_list = []
            if dets_t.numel() > 0:
                for d in dets_t:
                    x1, y1, x2, y2, score, cid = d.tolist()[:6]
                    dets_list.append((x1, y1, x2, y2, score, int(cid)))

            summary = tr.summary()

            # Terminal Dashboard (FIX: scale by model.cfg.input_size, not 160)
            inp = model.cfg.input_size
            dash_dets = [
                (d[0] * inp, d[1] * inp, d[2] * inp, d[3] * inp, d[4],
                 classes[d[5]] if d[5] < len(classes) else str(d[5]))
                for d in dets_list
            ]
            dash_txt = tr.dashboard(
                frame_idx=frame_idx, fps=fps_smooth,
                detections=dash_dets,
                mode="int" if args.int_mode else "stream")
            print_dashboard(dash_txt, first=(frame_idx == 0))

            # On-Screen HUD Render
            out_vis = draw_hud(
                display_frame, dets_list, summary, fps_smooth,
                conf_thr=conf_thr, class_names=classes,
                mode="int" if args.int_mode else "stream",
                width=640, height=480)

            if writer is not None:
                writer.write(out_vis)

            if not args.no_gui:
                cv2.imshow("NanoStream-OD | Ultra-Efficient Object Detector",
                           out_vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('+') or key == ord('='):
                    conf_thr = min(0.95, conf_thr + 0.05)
                elif key == ord('-') or key == ord('_'):
                    conf_thr = max(0.05, conf_thr - 0.05)
                elif key == ord('s') or key == ord('S'):
                    use_synthetic = not use_synthetic

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print("\nNanoStream-OD demo closed.")


def main():
    p = argparse.ArgumentParser(
        description="NanoStream-OD Live Webcam & Simulation Demo")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--synthetic", action="store_true",
                   help="Force synthetic animation test scene")
    p.add_argument("--video", type=str, default="",
                   help="Path to video file instead of webcam")
    p.add_argument("--conf", type=float, default=0.30,
                   help="Confidence threshold (default: 0.30)")
    p.add_argument("--model", type=str,
                   default="runs/faces/nanostream_faces.pt",
                   help="Path to trained PyTorch weights (.pt)")
    p.add_argument("--int-mode", action="store_true",
                   help="Run bit-exact integer simulation mode")
    p.add_argument("--no-gui", action="store_true",
                   help="Run without graphical window (headless)")
    p.add_argument("--record", type=str, default="",
                   help="Output video file path (.mp4)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max frames to process (0 = infinite)")
    args = p.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()

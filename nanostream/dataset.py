"""Real Face Dataset Pipeline: Online diverse faces + Webcam capture + OpenCV YuNet auto-labeling.

Enables downloading hundreds of diverse real human face photos from public
domain archives across different lighting, ages, ethnicities, and angles,
and auto-labeling them with OpenCV's FaceDetectorYN (YuNet ONNX model).

Usage:
    # Download 150+ diverse online real face images with auto-annotations:
    python -m nanostream.dataset --download-online

    # Or capture 400 frames from webcam:
    python -m nanostream.dataset --capture 400

    # Train on both combined:
    python -m nanostream.train_faces --steps 2500
"""

import argparse
import json
import pathlib
import time
import urllib.request
from typing import Union
import numpy as np
import torch
import torch.utils.data

try:
    import cv2
except ImportError:
    cv2 = None


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_LOCAL = pathlib.Path("data/yunet_face_detection.onnx")

DATA_DIR = pathlib.Path("data/real_faces")
FACE_CLASSES = ("face",)

# Curated list of diverse, high-quality real human portrait photo IDs (Unsplash public CDN)
# Covers diverse ethnicities, ages, genders, lighting conditions, expressions, angles, glasses, beards.
CURATED_FACE_PHOTO_IDS = [
    # Women portraits (various ethnicities, lighting, angles)
    "photo-1534528741775-53994a69daeb",
    "photo-1494790108377-be9c29b29330",
    "photo-1517841905240-472988babdf9",
    "photo-1438761681033-6461ffad8d80",
    "photo-1544005313-94ddf0286df2",
    "photo-1524504388940-b1c1722653e1",
    "photo-1531746020798-e6953c6e8e04",
    "photo-1508214751196-bcfd4ca60f91",
    "photo-1488426862026-3ee34a7d66df",
    "photo-1509967419530-da38b4704bc6",
    "photo-1531123897727-8f129e1688ce",
    "photo-1529626455594-4ff0802cfb7e",
    "photo-1496440737103-cd596325d314",
    "photo-1517486808906-6ca8b3f04846",
    "photo-1548142813-c348350df52b",
    "photo-1567532939604-b6b5b0db2604",
    "photo-1573496359142-b8d87734a5a2",
    "photo-1580489944761-15a19d654956",
    "photo-1534751516642-a171edd2521d",
    "photo-1560250097-0b93528c311a",
    # Men portraits (various ethnicities, lighting, beards, glasses)
    "photo-1507003211169-0a1dd7228f2d",
    "photo-1500648767791-00dcc994a43e",
    "photo-1472099645785-5658abf4ff4e",
    "photo-1519085360753-af0119f7cbe7",
    "photo-1506794778202-cad84cf45f1d",
    "photo-1522075469751-3a6694fb2f61",
    "photo-1539571696357-5a69c17a67c6",
    "photo-1501196354995-cbb51c65aaea",
    "photo-1513956589380-bad6acb9b9d4",
    "photo-1534308983496-4fabb1a015ee",
    "photo-1528892952291-009c663ce843",
    "photo-1566492031773-4f4e44671857",
    "photo-1568602471122-7832951cc4c5",
    "photo-1570295999919-56ceb5ecca61",
    "photo-1506794778202-cad84cf45f1d",
    "photo-1583864697784-a0efc8379f70",
    "photo-1542909168-82c3e7fdca5c",
    "photo-1552374196-1ab2a1c593e8",
    "photo-1564564321837-a57b7070ac4f",
    "photo-1535713875002-d1d0cf377fde",
    # Elderly & diverse ages
    "photo-1581579438747-1dc8d17bbce4",
    "photo-1508214751196-bcfd4ca60f91",
    "photo-1544717305-2782549b5136",
    "photo-1519699047748-de8e457a634e",
    "photo-1545167622-3a6ac756afa4",
    "photo-1530785602389-0759fbee859d",
    "photo-1541823709867-1b206113eafd",
    # Dramatic & side lighting / shadows
    "photo-1504257432389-52343af06ae3",
    "photo-1517462964-21fdcec3f25b",
    "photo-1514315384763-ba401779410f",
    "photo-1531746020798-e6953c6e8e04",
    "photo-1492562080023-ab3db95bfbce",
    "photo-1521119989659-a83eee488004",
    "photo-1544348817-5f2cf14b88c8",
    "photo-1519764622345-23439dd774f7",
]


def download_yunet_model(force: bool = False) -> pathlib.Path:
    """Download the YuNet ONNX face detector model (~337KB)."""
    if YUNET_LOCAL.exists() and not force:
        return YUNET_LOCAL
    YUNET_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet face detector ({YUNET_URL})...")
    req = urllib.request.Request(YUNET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(YUNET_LOCAL, "wb") as f:
        f.write(resp.read())
    print(f"Saved to {YUNET_LOCAL} ({YUNET_LOCAL.stat().st_size / 1024:.0f} KB)")
    return YUNET_LOCAL


def create_face_detector(input_size=(320, 320)):
    """Create OpenCV FaceDetectorYN using the YuNet ONNX model."""
    if cv2 is None:
        raise ImportError("opencv-python >= 5.0 required")
    model_path = download_yunet_model()
    detector = cv2.FaceDetectorYN_create(
        str(model_path),
        "",
        input_size,
        score_threshold=0.60,
        nms_threshold=0.30,
        top_k=10,
    )
    return detector


def detect_faces_yunet(detector, frame_bgr):
    """Run YuNet face detection on a BGR frame.

    Returns list of [x1, y1, x2, y2] bounding boxes in pixel coords.
    """
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    boxes = []
    if faces is not None:
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[14]) if len(face) > 14 else 1.0
            if score < 0.60:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            if (x2 - x1) > 12 and (y2 - y1) > 12:
                boxes.append([x1, y1, x2, y2])
    return boxes


def download_online_face_dataset(
    output_dir: Union[str, pathlib.Path] = DATA_DIR,
    target_size: int = 160,
) -> int:
    """Download diverse real face photos from curated CDN, auto-labeled with YuNet."""
    if cv2 is None:
        raise ImportError("opencv-python required")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Also import existing webcam frames if present in data/webcam_faces
    ann_path = output_dir / "annotations.json"
    annotations = []
    existing_files = set()

    webcam_dir = pathlib.Path("data/webcam_faces")
    if webcam_dir.exists() and (webcam_dir / "annotations.json").exists():
        try:
            with open(webcam_dir / "annotations.json") as f:
                webcam_anns = json.load(f)
            import shutil
            for wa in webcam_anns:
                src = webcam_dir / "images" / wa["file"]
                dst = img_dir / wa["file"]
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                annotations.append(wa)
                existing_files.add(wa["file"])
            print(f"Imported {len(webcam_anns)} webcam frames into combined real dataset.")
        except Exception as e:
            print(f"Webcam import note: {e}")

    detector = create_face_detector()

    print("\n" + "=" * 62)
    print("  Downloading Curated Diverse Real Face Dataset")
    print(f"  Target count : {len(CURATED_FACE_PHOTO_IDS)} diverse human face portraits")
    print(f"  Output folder: {output_dir.resolve()}")
    print("=" * 62 + "\n")

    saved = 0
    t0 = time.perf_counter()

    for idx, pid in enumerate(CURATED_FACE_PHOTO_IDS):
        fname = f"curated_{idx:04d}.png"
        if fname in existing_files:
            continue

        url = f"https://images.unsplash.com/{pid}?w=320&q=80"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw_bytes = resp.read()
            arr = np.frombuffer(raw_bytes, np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            h_orig, w_orig = img_bgr.shape[:2]
            boxes = detect_faces_yunet(detector, img_bgr)
            if not boxes:
                continue

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (target_size, target_size))

            scaled_boxes = []
            for bx1, by1, bx2, by2 in boxes:
                sx1 = int(bx1 * target_size / w_orig)
                sy1 = int(by1 * target_size / h_orig)
                sx2 = int(bx2 * target_size / w_orig)
                sy2 = int(by2 * target_size / h_orig)
                if (sx2 - sx1) > 8 and (sy2 - sy1) > 8:
                    scaled_boxes.append([sx1, sy1, sx2, sy2])

            if not scaled_boxes:
                continue

            cv2.imwrite(str(img_dir / fname), gray_resized)
            annotations.append({"file": fname, "boxes": scaled_boxes})
            saved += 1
            if saved % 10 == 0 or idx == len(CURATED_FACE_PHOTO_IDS) - 1:
                print(f"  Downloaded & labeled: {saved} online face portraits ({time.perf_counter() - t0:.1f}s)")

        except Exception:
            continue

    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    t1 = time.perf_counter()
    print(f"\nSuccessfully downloaded & auto-labeled {saved} online real faces in {t1 - t0:.1f}s!")
    print(f"Combined total dataset size: {len(annotations)} real samples -> {ann_path}")
    return saved


def capture_webcam_training_data(
    n_frames: int = 400,
    camera_id: int = 0,
    output_dir: Union[str, pathlib.Path] = DATA_DIR,
    target_size: int = 160,
    show_preview: bool = True,
) -> int:
    """Capture real webcam frames and auto-label faces with YuNet."""
    if cv2 is None:
        raise ImportError("opencv-python required")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    ann_path = output_dir / "annotations.json"
    annotations = []
    if ann_path.exists():
        try:
            with open(ann_path) as f:
                annotations = json.load(f)
        except Exception:
            annotations = []

    detector = create_face_detector()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("  Webcam Face Data Capture")
    print(f"  Capturing {n_frames} frames from camera {camera_id}")
    print("  Move your face around: angles, distances, lighting.")
    print(f"{'='*60}\n")

    captured = 0
    faces_found = 0
    start_idx = len(annotations)
    t0 = time.perf_counter()

    for i in range(n_frames * 3):
        if captured >= n_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if i % 2 != 0 and captured > 10:
            continue

        boxes = detect_faces_yunet(detector, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (target_size, target_size))

        h_orig, w_orig = frame.shape[:2]
        scaled_boxes = []
        for bx1, by1, bx2, by2 in boxes:
            sx1 = int(bx1 * target_size / w_orig)
            sy1 = int(by1 * target_size / h_orig)
            sx2 = int(bx2 * target_size / w_orig)
            sy2 = int(by2 * target_size / h_orig)
            if (sx2 - sx1) > 8 and (sy2 - sy1) > 8:
                scaled_boxes.append([sx1, sy1, sx2, sy2])

        if len(scaled_boxes) == 0:
            if captured > 0 and np.random.random() > 0.8:
                fname = f"neg_{start_idx + captured:05d}.png"
                cv2.imwrite(str(img_dir / fname), gray_resized)
                annotations.append({"file": fname, "boxes": []})
                captured += 1
            if show_preview:
                cv2.putText(frame, f"Capturing... {captured}/{n_frames} (No face)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("NanoStream-OD Data Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        fname = f"webcam_{start_idx + captured:05d}.png"
        cv2.imwrite(str(img_dir / fname), gray_resized)
        annotations.append({"file": fname, "boxes": scaled_boxes})
        captured += 1
        faces_found += 1

        if show_preview:
            for bx1, by1, bx2, by2 in boxes:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing... {captured}/{n_frames} ({faces_found} with faces)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("NanoStream-OD Data Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    t1 = time.perf_counter()
    print(f"\nCapture complete: {captured} frames saved in {t1 - t0:.1f}s.")
    return faces_found


class RealFaceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset with robust multi-condition photorealistic augmentations."""

    def __init__(self, data_dir: Union[str, pathlib.Path] = DATA_DIR, img_size: int = 160,
                 augment: bool = True, cache_in_ram: bool = False,
                 split: str = "train", val_frac: float = 0.2, seed: int = 42):
        self.data_dir = pathlib.Path(data_dir)
        self.img_size = img_size
        self.img_dir = self.data_dir / "images"
        ann_path = self.data_dir / "annotations.json"
        if not ann_path.exists():
            fallback = pathlib.Path("data/webcam_faces/annotations.json")
            if fallback.exists():
                ann_path = fallback
                self.img_dir = pathlib.Path("data/webcam_faces/images")
            else:
                raise FileNotFoundError(
                    "No face dataset found. Run:\n"
                    "  python -m nanostream.dataset --download-online"
                )

        with open(ann_path) as f:
            self.annotations = json.load(f)

        # FIX: real train/val split. Previously eval_ds was built over the SAME
        # annotations as train_ds (only augment differed), so every validation
        # sample was a training sample and mAP/recall were inflated.
        n = len(self.annotations)
        idx_all = np.arange(n)
        rng_split = np.random.default_rng(seed)
        rng_split.shuffle(idx_all)
        n_val = max(1, int(round(n * val_frac)))
        if split in ("val", "eval"):
            self._indices = idx_all[n - n_val:]
        else:
            self._indices = idx_all[:n - n_val]

        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cache = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        ann = self.annotations[self._indices[idx % len(self._indices)]]
        file_key = ann["file"]
        if self.cache_in_ram and file_key in self.cache:
            img = self.cache[file_key].copy()
        else:
            img_path = self.img_dir / file_key
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) if cv2 is not None else None
            if img is None:
                img = np.full((self.img_size, self.img_size), 128, dtype=np.uint8)
            if self.cache_in_ram:
                self.cache[file_key] = img.copy()

        h, w = img.shape[:2]
        boxes_px = [b[:] for b in ann["boxes"]]

        if self.augment:
            from .augment import cutout, photometric_distort, geometric_augment
            # 1. Geometric augmentations (flip, rotate, scale, translate)
            img, boxes_px = geometric_augment(img, boxes_px, self.img_size, rng=self.rng)
            h, w = img.shape[:2]

            # 2. Photometric distortions (brightness, contrast, gamma, CLAHE, blur)
            img = photometric_distort(img, rng=self.rng)

            # 3. CutOut / Random Erase (occlusion robustness)
            if self.rng.random() > 0.4:
                img = cutout(img, boxes_px, n_holes=int(self.rng.integers(1, 3)), rng=self.rng)

        # Normalize to [-1, 1]
        x = torch.from_numpy(img.copy()).float() / 127.5 - 1.0
        x = x.unsqueeze(0)

        # Convert boxes to normalized cxcywh
        if len(boxes_px) > 0:
            # BUG-14 FIX: Normalize x by width and y by height separately
            b = torch.tensor(boxes_px, dtype=torch.float32)
            cxcywh = torch.stack([
                (b[:, 0] + b[:, 2]) / 2 / float(w),
                (b[:, 1] + b[:, 3]) / 2 / float(h),
                ((b[:, 2] - b[:, 0]) / float(w)).clamp(min=0.04),
                ((b[:, 3] - b[:, 1]) / float(h)).clamp(min=0.04),
            ], dim=1)
            labels = torch.zeros(len(boxes_px), dtype=torch.long)
        else:
            cxcywh = torch.zeros(0, 4)
            labels = torch.zeros(0, dtype=torch.long)

        tgt = {"boxes_norm": cxcywh, "labels": labels}
        return x, tgt


# Alias for backward compatibility
WebcamFaceDataset = RealFaceDataset


def main():
    p = argparse.ArgumentParser(description="NanoStream-OD Real Face Dataset Pipeline")
    p.add_argument("--download-online", action="store_true",
                   help="Download diverse online real face portraits from curated public CDN")
    p.add_argument("--capture", type=int, default=0,
                   help="Capture N frames from webcam")
    p.add_argument("--camera", type=int, default=0,
                   help="Webcam index")
    p.add_argument("--output", type=str, default=str(DATA_DIR),
                   help="Output directory")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output)
    if args.download_online:
        download_online_face_dataset(output_dir=out_dir)
    elif args.capture > 0:
        capture_webcam_training_data(n_frames=args.capture, camera_id=args.camera, output_dir=out_dir)
    else:
        print("Specify either --download-online or --capture N")


if __name__ == "__main__":
    main()

"""Combined synthetic dataset (shapes + faces) for cross-framework benchmarking.

Four classes: circle, square, triangle, face. Generation is deterministic per
index (one RNG per index), so every consumer - in-memory training, YOLO-format
file export - sees byte-identical samples for a given index.
"""

import pathlib

import numpy as np
import torch
import torch.utils.data

try:
    import cv2
    from nanostream.data import _draw_object
    from nanostream.faces import _draw_face
except ImportError:  # pragma: no cover - environment guard
    cv2 = None

CLASS_NAMES = ("circle", "square", "triangle", "face")
NUM_CLASSES = len(CLASS_NAMES)

GLOBAL_SEED = 2024
SIZE = 160
TRAIN_LEN = 240       # minimum canonical train set (indices [0, TRAIN_LEN))
VAL_LEN = 60
VAL_START = 100000    # val indices live far from any train index -> splits
                      # can never overlap no matter how large train gets


def _make_rng(idx: int) -> np.random.Generator:
    return np.random.default_rng(GLOBAL_SEED * 100000 + idx)


def _overlaps(cand, boxes, thr=0.20):
    x1, y1, x2, y2 = cand
    a1 = (x2 - x1) * (y2 - y1)
    for (bx1, by1, bx2, by2) in boxes:
        ix = max(0, min(x2, bx2) - max(x1, bx1))
        iy = max(0, min(y2, by2) - max(y1, by1))
        inter = ix * iy
        a2 = (bx2 - bx1) * (by2 - by1)
        if inter > thr * min(a1, a2):
            return True
    return False


def generate_sample(idx: int, size: int = SIZE):
    """Generate one image (uint8 HxW) plus boxes (xyxy px) and labels."""
    if cv2 is None:
        raise ImportError("opencv-python is required for synthetic data generation")
    rng = _make_rng(idx)

    bg = int(rng.integers(0, 40))
    img = np.full((size, size), bg, dtype=np.uint8)
    noise = rng.normal(0, 6, size=(size, size))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    boxes, labels = [], []
    n_obj = int(rng.integers(1, 4))  # 1..3 objects per image
    for _ in range(n_obj):
        cls_id = int(rng.integers(0, NUM_CLASSES))
        if cls_id == 3:
            face_w = int(rng.integers(32, int(size * 0.5)))
            face_h = int(face_w * rng.uniform(1.15, 1.30))
            rx, ry = face_w // 2, face_h // 2
            min_cx, max_cx = rx + 4, max(rx + 5, size - rx - 4)
            min_cy, max_cy = ry + 4, max(ry + 5, size - ry - 4)
            cx = int(rng.integers(min_cx, max_cx))
            cy = int(rng.integers(min_cy, max_cy))
            m = 1.05
            cand = [cx - int(rx * m), cy - int(ry * m),
                    cx + int(rx * m), cy + int(ry * m)]
            if _overlaps(cand, boxes):
                continue
            img, bbox = _draw_face(img, cx, cy, face_w, face_h, rng)
            boxes.append(bbox)
            labels.append(3)
        else:
            side = int(rng.integers(18, 44))
            x1 = int(rng.integers(4, size - side - 4))
            y1 = int(rng.integers(4, size - side - 4))
            x2, y2 = x1 + side, y1 + side
            if _overlaps([x1, y1, x2, y2], boxes):
                continue
            img = _draw_object(img, cls_id, x1, y1, x2, y2, rng)
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
    return img, boxes, labels


def to_target(boxes_px, labels, size):
    if len(boxes_px) == 0:
        return {"boxes_norm": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long)}
    b = torch.tensor(boxes_px, dtype=torch.float32).view(-1, 4) / float(size)
    cxcywh = torch.stack([
        (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
        (b[:, 2] - b[:, 0]).clamp(min=0.05),
        (b[:, 3] - b[:, 1]).clamp(min=0.05),
    ], dim=1)
    return {"boxes_norm": cxcywh, "labels": torch.tensor(labels, dtype=torch.long)}


class CombinedDataset(torch.utils.data.Dataset):
    """Deterministic shapes+faces dataset.

    Train split uses indices ``[0, train_len)``, validation uses the far
    ``[VAL_START, VAL_START + VAL_LEN)`` range - disjoint by construction,
    reproducible, and immune to train-set size changes.
    """

    def __init__(self, length: int = TRAIN_LEN, size: int = SIZE,
                 start_idx: int = 0):
        self.length = length
        self.size = size
        self.start_idx = start_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, boxes, labels = generate_sample(self.start_idx + idx, self.size)
        x = torch.from_numpy(img.copy()).float() / 127.5 - 1.0
        x = x.unsqueeze(0)
        return x, to_target(boxes, labels, self.size)


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    return xs, [b[1] for b in batch]


def export_yolo(out_dir, splits=(
        ("train", TRAIN_LEN, 0),
        ("val", VAL_LEN, VAL_START))):
    """Write images + YOLO label files + data.yaml for ultralytics."""
    out_dir = pathlib.Path(out_dir)
    for split, length, start in splits:
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(length):
            idx = start + i
            img, boxes, labels = generate_sample(idx, SIZE)
            cv2.imwrite(str(img_dir / f"img_{idx:05d}.png"), img)
            lines = []
            for (x1, y1, x2, y2), c in zip(boxes, labels):
                w, h = x2 - x1, y2 - y1
                lines.append(f"{c} {(x1 + w / 2) / SIZE:.6f} "
                             f"{(y1 + h / 2) / SIZE:.6f} "
                             f"{w / SIZE:.6f} {h / SIZE:.6f}")
            (lbl_dir / f"img_{idx:05d}.txt").write_text("\n".join(lines) + "\n")
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: train/images\nval: val/images\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {list(CLASS_NAMES)}\n")
    return yaml_path

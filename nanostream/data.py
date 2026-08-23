"""Synthetic shapes dataset: self-contained training data, zero downloads."""

import numpy as np
import torch
import torch.utils.data

try:
    import cv2
except ImportError:
    cv2 = None

CLASS_NAMES = ("circle", "square", "triangle")


def _draw_object(img, cls_id, x1, y1, x2, y2, rng):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    r = int(max(3, min(x2 - x1, y2 - y1) / 2))
    color = int(rng.integers(140, 255))
    if cls_id == 0:
        cv2.circle(img, (cx, cy), r, (color,), -1)
    elif cls_id == 1:
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r),
                      (color,), -1)
    else:
        pts = np.array([
            [cx, cy - r],
            [cx - r, cy + r],
            [cx + r, cy + r],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (color,))
    return img


def make_sample(size=160, max_objects=3, rng=None):
    if cv2 is None:
        raise ImportError("opencv-python is required for data generation: "
                          "pip install opencv-python")
    if rng is None:
        rng = np.random.default_rng()
    bg = int(rng.integers(0, 40))
    img = np.full((size, size), bg, dtype=np.uint8)
    noise = rng.normal(0, 6, size=(size, size))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    boxes = []
    labels = []
    n_obj = int(rng.integers(1, max_objects + 1))
    for _ in range(n_obj):
        for _attempt in range(12):
            side = int(rng.integers(18, 44))
            x1 = int(rng.integers(4, size - side - 4))
            y1 = int(rng.integers(4, size - side - 4))
            x2, y2 = x1 + side, y1 + side
            ok = True
            for (bx1, by1, bx2, by2) in boxes:
                ix = max(0, min(x2, bx2) - max(x1, bx1))
                iy = max(0, min(y2, by2) - max(y1, by1))
                inter = ix * iy
                a1 = (x2 - x1) * (y2 - y1)
                a2 = (bx2 - bx1) * (by2 - by1)
                if inter > 0.25 * min(a1, a2):
                    ok = False
                    break
            if ok:
                break
        if not ok:
            continue
        cls_id = int(rng.integers(0, len(CLASS_NAMES)))
        img = _draw_object(img, cls_id, x1, y1, x2, y2, rng)
        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id)
    return img, boxes, labels


def to_target(boxes_px, labels, size):
    b = torch.tensor(boxes_px, dtype=torch.float32).view(-1, 4) / float(size)
    cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2,
                          (b[:, 1] + b[:, 3]) / 2,
                          b[:, 2] - b[:, 0],
                          b[:, 3] - b[:, 1]], dim=1)
    return {"boxes_norm": cxcywh,
            "labels": torch.tensor(labels, dtype=torch.long)}


class SyntheticShapes(torch.utils.data.Dataset):
    def __init__(self, length=512, size=160, seed=0, max_objects=3):
        self.length = length
        self.size = size
        self.seed = seed
        self.max_objects = max_objects
        # FIX: one RNG per sample index, so __getitem__ is deterministic and
        # reproducible across runs/epochs (previously it re-rolled a shared RNG
        # on every access and ignored idx entirely).
        self._rngs = [np.random.default_rng(seed + 1000 + i) for i in range(length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = self._rngs[idx % self.length]
        img, boxes, labels = make_sample(self.size, self.max_objects, rng)
        x = torch.from_numpy(img.copy()).float() / 127.5 - 1.0
        x = x.unsqueeze(0)
        tgt = to_target(boxes, labels, self.size)
        return x, tgt


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    return xs, [b[1] for b in batch]


def calibration_images(n=48, size=160, seed=1234):
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        img, _, _ = make_sample(size, 3, rng)
        # FIX: return the SAME uint8 range the real pipeline feeds the model
        # (0-255 pixels); calibrate_fixed_point quantizes via quantize_input_u8.
        imgs.append(torch.from_numpy(img))
    return imgs


def render_frame_webcam_like(size=160, t=0.0, rng=None):
    """Moving-shapes scene used when no webcam is available."""
    import cv2
    if rng is None:
        rng = np.random.default_rng(7)
    img = np.zeros((size, size), dtype=np.uint8)
    scene = [
        (0, 0.30, 0.35, 0.10),
        (1, 0.62, 0.60, 0.13),
        (2, 0.45, 0.75, 0.09),
    ]
    boxes, labels = [], []
    for i, (cls_id, sx, sy, s) in enumerate(scene):
        phase = t * (0.25 + 0.11 * i) + i * 2.1
        cx = (sx + 0.22 * np.sin(phase)) * size
        cy = (sy + 0.16 * np.cos(phase * 1.31 + i)) * size
        r = s * size
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r
        color = int(150 + 90 * np.sin(phase * 2.0) ** 2)
        tmp = np.zeros((size, size), dtype=np.uint8)
        if cls_id == 0:
            cv2.circle(tmp, (int(cx), int(cy)), int(r), (color,), -1)
        elif cls_id == 1:
            cv2.rectangle(tmp, (int(x1), int(y1)), (int(x2), int(y2)),
                          (color,), -1)
        else:
            pts = np.array([[int(cx), int(cy - r)], [int(cx - r), int(cy + r)],
                            [int(cx + r), int(cy + r)]], np.int32)
            cv2.fillPoly(tmp, [pts], (color,))
        mask = tmp > 0
        img[mask] = tmp[mask]
        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id)
    noise = rng.normal(0, 5, size=(size, size))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, boxes, labels

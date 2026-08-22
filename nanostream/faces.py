"""Synthetic & Procedural Face dataset generator for NanoStream-OD face detection."""

import math
import numpy as np
import torch
import torch.utils.data

try:
    import cv2
except ImportError:
    cv2 = None

FACE_CLASSES = ("face",)


def _generate_background(size, rng):
    """Generate realistic indoor/ambient background patterns."""
    bg_type = rng.integers(0, 4)
    if bg_type == 0:
        # Smooth spatial gradient (like a room wall with lighting)
        x_grad = np.linspace(rng.uniform(30, 80), rng.uniform(80, 200), size).reshape(1, size)
        y_grad = np.linspace(rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2), size).reshape(size, 1)
        bg = np.clip(x_grad * y_grad, 0, 255).astype(np.float32)
    elif bg_type == 1:
        # Striped / shelf-like background with vertical & horizontal edges
        base = float(rng.integers(40, 140))
        bg = np.full((size, size), base, dtype=np.float32)
        n_lines = rng.integers(2, 6)
        for _ in range(n_lines):
            pos = int(rng.integers(10, size - 10))
            thick = int(rng.integers(2, 8))
            val = float(rng.integers(20, 220))
            if rng.random() > 0.5:
                bg[:, max(0, pos - thick):min(size, pos + thick)] = val
            else:
                bg[max(0, pos - thick):min(size, pos + thick), :] = val
    elif bg_type == 2:
        # Radial spotlight gradient
        cx, cy = rng.integers(20, size - 20), rng.integers(20, size - 20)
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_d = math.sqrt(size ** 2 + size ** 2)
        bright = float(rng.integers(120, 210))
        dark = float(rng.integers(30, 90))
        bg = np.clip(bright - (bright - dark) * (dist / max_d), 0, 255).astype(np.float32)
    else:
        # Multi-patch textured ambient
        base = float(rng.integers(50, 160))
        bg = np.full((size, size), base, dtype=np.float32)
        for _ in range(rng.integers(3, 8)):
            x1, y1 = rng.integers(0, size - 30), rng.integers(0, size - 30)
            w, h = rng.integers(20, 70), rng.integers(20, 70)
            c = float(rng.integers(30, 220))
            bg[y1:y1 + h, x1:x1 + w] = (bg[y1:y1 + h, x1:x1 + w] + c) / 2.0

    # Add camera noise
    noise = rng.normal(0, rng.uniform(2.0, 7.0), size=(size, size))
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def _draw_face(img, cx, cy, face_w, face_h, rng):
    """Draw realistic facial features (head, hair, eyes, nose, mouth, neck)."""
    # 1. Skin tone & brightness
    skin_tone = int(rng.integers(130, 230))
    skin_color = (skin_tone,)
    shadow_tone = max(20, skin_tone - int(rng.integers(25, 55)))
    shadow_color = (shadow_tone,)
    highlight_tone = min(255, skin_tone + int(rng.integers(15, 35)))
    hair_color = int(rng.integers(15, 75))

    rx = int(face_w / 2)
    ry = int(face_h / 2)

    # 2. Neck
    neck_w = int(face_w * rng.uniform(0.38, 0.48))
    neck_h = int(face_h * rng.uniform(0.35, 0.55))
    neck_x1, neck_y1 = cx - neck_w // 2, cy + int(ry * 0.6)
    neck_x2, neck_y2 = cx + neck_w // 2, min(img.shape[0] - 1, neck_y1 + neck_h)
    cv2.rectangle(img, (neck_x1, neck_y1), (neck_x2, neck_y2), shadow_color, -1)

    # 3. Head Oval
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, skin_color, -1)

    # Shading across cheek/jaw
    light_dir = 1 if rng.random() > 0.5 else -1
    cv2.ellipse(img, (cx + light_dir * rx // 4, cy), (int(rx * 0.7), int(ry * 0.9)),
                0, 0, 360, (highlight_tone,), -1)

    # 4. Hair (top & sides)
    hair_type = rng.integers(0, 4)
    if hair_type == 0:
        # Short / cropped hair
        cv2.ellipse(img, (cx, cy - int(ry * 0.2)), (int(rx * 1.05), int(ry * 0.95)),
                    0, 180, 360, (hair_color,), -1)
    elif hair_type == 1:
        # Full hair / bangs
        cv2.ellipse(img, (cx, cy - int(ry * 0.25)), (int(rx * 1.1), int(ry * 0.95)),
                    0, 160, 380, (hair_color,), -1)
        # Bangs over forehead
        cv2.ellipse(img, (cx, cy - int(ry * 0.5)), (int(rx * 0.8), int(ry * 0.35)),
                    0, 0, 180, (hair_color,), -1)
    elif hair_type == 2:
        # Side parted hair
        pts = np.array([
            [cx - int(rx * 1.1), cy],
            [cx - int(rx * 1.1), cy - ry],
            [cx + int(rx * 1.1), cy - ry],
            [cx + int(rx * 1.1), cy + int(ry * 0.3)],
            [cx + int(rx * 0.8), cy - int(ry * 0.4)],
            [cx - int(rx * 0.6), cy - int(ry * 0.4)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (hair_color,))

    # 5. Eyes & Eyebrows
    eye_y = cy - int(ry * 0.12)
    eye_dx = int(rx * rng.uniform(0.38, 0.46))
    eye_r = max(2, int(rx * rng.uniform(0.10, 0.15)))

    # Eyebrows
    brow_y = eye_y - int(eye_r * rng.uniform(1.3, 1.8))
    brow_w = int(eye_r * rng.uniform(1.4, 2.0))
    cv2.ellipse(img, (cx - eye_dx, brow_y), (brow_w, max(1, eye_r // 3)), -5, 0, 180, (hair_color,), -1)
    cv2.ellipse(img, (cx + eye_dx, brow_y), (brow_w, max(1, eye_r // 3)), 5, 0, 180, (hair_color,), -1)

    # Eyes: Sclera + Iris
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        # Eye socket shading
        cv2.circle(img, (ex, eye_y), eye_r + 2, shadow_color, -1)
        # Eye white
        cv2.ellipse(img, (ex, eye_y), (eye_r, max(2, int(eye_r * 0.65))), 0, 0, 360, (245,), -1)
        # Pupil / Iris
        cv2.circle(img, (ex, eye_y), max(1, int(eye_r * 0.45)), (hair_color,), -1)

    # Glasses (30% chance)
    if rng.random() < 0.30:
        g_r = int(eye_r * 1.6)
        cv2.circle(img, (cx - eye_dx, eye_y), g_r, (30,), 1)
        cv2.circle(img, (cx + eye_dx, eye_y), g_r, (30,), 1)
        cv2.line(img, (cx - eye_dx + g_r, eye_y), (cx + eye_dx - g_r, eye_y), (30,), 1)

    # 6. Nose
    nose_y = cy + int(ry * 0.18)
    nose_w = max(2, int(rx * 0.18))
    cv2.line(img, (cx, eye_y + eye_r), (cx, nose_y), shadow_color, max(1, int(face_w * 0.02)))
    cv2.ellipse(img, (cx, nose_y), (nose_w, max(1, nose_w // 2)), 0, 0, 180, shadow_color, -1)

    # 7. Mouth & Lips
    mouth_y = cy + int(ry * 0.50)
    mouth_w = max(4, int(rx * rng.uniform(0.35, 0.48)))
    lip_tone = max(40, skin_tone - int(rng.integers(20, 45)))
    cv2.ellipse(img, (cx, mouth_y), (mouth_w, max(2, int(mouth_w * 0.35))), 0, 0, 180, (lip_tone,), -1)
    cv2.line(img, (cx - mouth_w, mouth_y), (cx + mouth_w, mouth_y), (shadow_tone,), max(1, int(face_w * 0.02)))

    # Compute tight bounding box
    top_y = cy - int(ry * 1.1)
    bot_y = cy + int(ry * 0.95)
    left_x = cx - int(rx * 1.05)
    right_x = cx + int(rx * 1.05)

    bx1 = max(0, left_x)
    by1 = max(0, top_y)
    bx2 = min(img.shape[1] - 1, right_x)
    by2 = min(img.shape[0] - 1, bot_y)

    return img, [bx1, by1, bx2, by2]


def make_face_sample(size=160, max_faces=2, rng=None):
    if cv2 is None:
        raise ImportError("opencv-python required for face dataset generation")
    if rng is None:
        rng = np.random.default_rng()

    img = _generate_background(size, rng)
    boxes = []
    labels = []

    n_faces = int(rng.integers(1, max_faces + 1))
    for _ in range(n_faces):
        for _attempt in range(10):
            # Scale variations: 32px to 80px
            face_w = int(rng.integers(32, int(size * 0.52)))
            face_h = int(face_w * rng.uniform(1.15, 1.30))
            rx = face_w // 2
            ry = face_h // 2

            min_cx, max_cx = rx + 4, max(rx + 5, size - rx - 4)
            min_cy, max_cy = ry + 4, max(ry + 5, size - ry - 4)
            cx = int(rng.integers(min_cx, max_cx))
            cy = int(rng.integers(min_cy, max_cy))

            # Non-overlap check
            ok = True
            for (bx1, by1, bx2, by2) in boxes:
                cand_x1, cand_y1 = cx - rx, cy - ry
                cand_x2, cand_y2 = cx + rx, cy + ry
                ix = max(0, min(cand_x2, bx2) - max(cand_x1, bx1))
                iy = max(0, min(cand_y2, by2) - max(cand_y1, by1))
                inter = ix * iy
                a1 = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)
                a2 = (bx2 - bx1) * (by2 - by1)
                if inter > 0.20 * min(a1, a2):
                    ok = False
                    break
            if ok:
                break

        if not ok:
            continue

        img, bbox = _draw_face(img, cx, cy, face_w, face_h, rng)
        boxes.append(bbox)
        labels.append(0)  # Class 0: Face

    return img, boxes, labels


class SyntheticFaces(torch.utils.data.Dataset):
    """Synthetic human faces dataset."""

    def __init__(self, length=1024, size=160, seed=42, max_faces=2):
        self.length = length
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.max_faces = max_faces

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, boxes, labels = make_face_sample(self.size, self.max_faces, self.rng)
        # Normalize [-1, 1]
        x = torch.from_numpy(img.copy()).float() / 127.5 - 1.0
        x = x.unsqueeze(0)

        # Boxes normalized cxcywh
        b = torch.tensor(boxes, dtype=torch.float32).view(-1, 4) / float(self.size)
        cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2,
                              (b[:, 1] + b[:, 3]) / 2,
                              (b[:, 2] - b[:, 0]).clamp(min=0.05),
                              (b[:, 3] - b[:, 1]).clamp(min=0.05)], dim=1) if len(boxes) else torch.zeros(0, 4)
        tgt = {
            "boxes_norm": cxcywh,
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return x, tgt

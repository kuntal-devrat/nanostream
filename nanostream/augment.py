"""NanoStream-OD v3.0 YOLO-Grade Data Augmentation Pipeline.

Mosaic, MixUp, CutOut, multi-scale resize, and advanced photometric
transforms that match or exceed YOLOv8's augmentation recipe.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def mosaic_4(images: list, boxes_list: list, labels_list: list,
             target_size: int = 160, rng=None) -> tuple:
    """Mosaic augmentation: tile 4 images into a single training sample.

    This is YOLOv5/v8's most impactful augmentation — it exposes the model
    to 4× scene diversity per sample and naturally handles small objects
    placed at random positions.

    Args:
        images: List of 4 grayscale numpy arrays (H, W)
        boxes_list: List of 4 box arrays, each (N, 4) in pixel coords [x1,y1,x2,y2]
        labels_list: List of 4 label arrays, each (N,)
        target_size: Output image size
        rng: numpy random generator

    Returns:
        mosaic_img: (target_size, target_size) uint8 array
        mosaic_boxes: (M, 4) combined boxes in pixel coords
        mosaic_labels: (M,) combined labels
    """
    if rng is None:
        rng = np.random.default_rng()

    s = target_size
    # Random center point for the mosaic grid
    cx = int(rng.integers(s // 4, 3 * s // 4))
    cy = int(rng.integers(s // 4, 3 * s // 4))

    canvas = np.full((s, s), int(rng.integers(40, 140)), dtype=np.uint8)
    all_boxes = []
    all_labels = []

    # Quadrant placements: top-left, top-right, bottom-left, bottom-right
    placements = [
        (0, 0, cx, cy),         # top-left
        (cx, 0, s, cy),         # top-right
        (0, cy, cx, s),         # bottom-left
        (cx, cy, s, s),         # bottom-right
    ]

    for i, (px1, py1, px2, py2) in enumerate(placements):
        if i >= len(images):
            break

        img = images[i]
        h_orig, w_orig = img.shape[:2]
        qw, qh = px2 - px1, py2 - py1

        if qw < 4 or qh < 4:
            continue

        # Resize image to fit quadrant
        resized = cv2.resize(img, (qw, qh)) if cv2 is not None else img[:qh, :qw]
        canvas[py1:py2, px1:px2] = resized[:qh, :qw]

        # Transform boxes
        if len(boxes_list[i]) > 0:
            bx = np.array(boxes_list[i], dtype=np.float32)
            # Scale boxes from original to quadrant
            scale_x = qw / max(w_orig, 1)
            scale_y = qh / max(h_orig, 1)
            bx[:, 0] = bx[:, 0] * scale_x + px1
            bx[:, 1] = bx[:, 1] * scale_y + py1
            bx[:, 2] = bx[:, 2] * scale_x + px1
            bx[:, 3] = bx[:, 3] * scale_y + py1

            # Clip to canvas
            bx[:, 0] = np.clip(bx[:, 0], 0, s)
            bx[:, 1] = np.clip(bx[:, 1], 0, s)
            bx[:, 2] = np.clip(bx[:, 2], 0, s)
            bx[:, 3] = np.clip(bx[:, 3], 0, s)

            # Filter out too-small boxes
            valid = ((bx[:, 2] - bx[:, 0]) > 6) & ((bx[:, 3] - bx[:, 1]) > 6)
            bx = bx[valid]
            lbl = np.array(labels_list[i])[valid] if len(labels_list[i]) > 0 else np.array([])

            if len(bx) > 0:
                all_boxes.append(bx)
                all_labels.append(lbl)

    if all_boxes:
        mosaic_boxes = np.concatenate(all_boxes, axis=0)
        mosaic_labels = np.concatenate(all_labels, axis=0)
    else:
        mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
        mosaic_labels = np.zeros(0, dtype=np.int64)

    return canvas, mosaic_boxes.tolist(), mosaic_labels.tolist()


def mixup(img1, boxes1, labels1, img2, boxes2, labels2,
          alpha: float = 0.5, rng=None) -> tuple:
    """MixUp augmentation: alpha-blend two images and merge annotations.

    Args:
        img1, img2: Grayscale uint8 images of same size
        boxes1, boxes2: Box lists in pixel coords [x1,y1,x2,y2]
        labels1, labels2: Label lists
        alpha: Blend ratio (0.5 = equal mix)
        rng: numpy random generator

    Returns:
        mixed_img, merged_boxes, merged_labels
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = float(rng.beta(alpha, alpha))
    lam = max(0.35, min(0.65, lam))  # Keep blend visible

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Resize img2 to match img1
    if (h1, w1) != (h2, w2):
        if cv2 is not None:
            img2 = cv2.resize(img2, (w1, h1))
        else:
            img2 = img2[:h1, :w1]
        # Scale boxes2
        scale_x = w1 / max(w2, 1)
        scale_y = h1 / max(h2, 1)
        boxes2 = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y]
                   for b in boxes2]

    mixed = np.clip(
        img1.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam),
        0, 255
    ).astype(np.uint8)

    merged_boxes = list(boxes1) + list(boxes2)
    merged_labels = list(labels1) + list(labels2)

    return mixed, merged_boxes, merged_labels


def cutout(img, boxes, n_holes: int = 2, max_size: float = 0.25, rng=None):
    """CutOut / Random Erase: occlude random rectangular patches.

    Forces the detector to learn from partial object views,
    improving robustness to occlusion.

    Args:
        img: Grayscale uint8 image (H, W)
        boxes: Box list (not modified, only image is altered)
        n_holes: Number of erased patches
        max_size: Maximum hole size as fraction of image
        rng: numpy random generator

    Returns:
        img_with_holes: Modified image
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    result = img.copy()

    for _ in range(n_holes):
        hole_w = int(rng.integers(8, max(9, int(w * max_size))))
        hole_h = int(rng.integers(8, max(9, int(h * max_size))))
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))

        x1 = max(0, cx - hole_w // 2)
        y1 = max(0, cy - hole_h // 2)
        x2 = min(w, cx + hole_w // 2)
        y2 = min(h, cy + hole_h // 2)

        fill = int(rng.integers(0, 255))
        result[y1:y2, x1:x2] = fill

    return result


def photometric_distort(img, rng=None):
    """Advanced photometric augmentation for grayscale images.

    Includes brightness, contrast, gamma, histogram equalization,
    and Gaussian blur — more aggressive than v2.0's simple jitter.

    Args:
        img: Grayscale uint8 image (H, W)
        rng: numpy random generator

    Returns:
        Augmented grayscale uint8 image
    """
    if rng is None:
        rng = np.random.default_rng()

    result = img.astype(np.float32)

    # 1. Brightness shift
    if rng.random() > 0.3:
        result += rng.uniform(-35, 35)

    # 2. Contrast
    if rng.random() > 0.3:
        alpha = rng.uniform(0.6, 1.5)
        mean = result.mean()
        result = (result - mean) * alpha + mean

    # 3. Gamma correction
    if rng.random() > 0.5:
        gamma = rng.uniform(0.7, 1.4)
        result = np.clip(result, 0, 255)
        result = 255.0 * (result / 255.0) ** gamma

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 4. Random CLAHE
    if cv2 is not None and rng.random() > 0.5:
        clip = rng.uniform(1.0, 4.0)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        result = clahe.apply(result)

    # 5. Gaussian blur
    if cv2 is not None and rng.random() > 0.6:
        k = int(rng.choice([3, 5]))
        result = cv2.GaussianBlur(result, (k, k), 0)

    return result


def multi_scale_resize(img, boxes, target_size: int, rng=None,
                       min_size: int = 128, max_size: int = 192):
    """Multi-scale training: random resize then crop/pad to target.

    YOLO trains at random resolutions each batch for scale robustness.
    We resize the image and boxes to a random intermediate size, then
    center-pad or crop to the final target_size.

    Args:
        img: Grayscale uint8 (H, W)
        boxes: List of [x1,y1,x2,y2] in pixel coords of current image
        target_size: Final output size
        rng: numpy random generator
        min_size, max_size: Random size range

    Returns:
        resized_img: (target_size, target_size) uint8
        resized_boxes: Transformed boxes in new pixel coords
    """
    if rng is None:
        rng = np.random.default_rng()

    if cv2 is None:
        return img, boxes

    h, w = img.shape[:2]
    rand_size = int(rng.integers(min_size, max_size + 1))

    # Resize to random intermediate size
    img_r = cv2.resize(img, (rand_size, rand_size))
    scale_x = rand_size / max(w, 1)
    scale_y = rand_size / max(h, 1)

    new_boxes = []
    for bx in boxes:
        nb = [bx[0] * scale_x, bx[1] * scale_y,
              bx[2] * scale_x, bx[3] * scale_y]
        new_boxes.append(nb)

    # Pad or crop to target_size
    if rand_size < target_size:
        # Center pad
        pad = (target_size - rand_size) // 2
        canvas = np.full((target_size, target_size),
                         int(img_r.mean()), dtype=np.uint8)
        canvas[pad:pad+rand_size, pad:pad+rand_size] = img_r
        # Shift boxes
        new_boxes = [[b[0]+pad, b[1]+pad, b[2]+pad, b[3]+pad]
                     for b in new_boxes]
        img_r = canvas
    elif rand_size > target_size:
        # Random crop
        off = int(rng.integers(0, rand_size - target_size + 1))
        img_r = img_r[off:off+target_size, off:off+target_size]
        new_boxes = [[b[0]-off, b[1]-off, b[2]-off, b[3]-off]
                     for b in new_boxes]
    # else: exact match, no change

    # Clip and filter
    final_boxes = []
    for b in new_boxes:
        bx = [max(0, b[0]), max(0, b[1]),
              min(target_size, b[2]), min(target_size, b[3])]
        if (bx[2] - bx[0]) > 6 and (bx[3] - bx[1]) > 6:
            final_boxes.append(bx)

    return img_r, final_boxes


def geometric_augment(img, boxes, size, rng=None):
    """Combined geometric augmentations: flip, rotate, scale, translate.

    Args:
        img: Grayscale uint8 (H, W)
        boxes: List of [x1,y1,x2,y2] pixel coords
        size: Image size
        rng: numpy random generator

    Returns:
        img_aug, boxes_aug
    """
    if rng is None:
        rng = np.random.default_rng()
    if cv2 is None:
        return img, boxes

    h, w = img.shape[:2]

    # 1. Horizontal flip (50%)
    if rng.random() > 0.5:
        img = np.flip(img, axis=1).copy()
        boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]

    # 2. Small rotation (-10 to +10 degrees)
    if rng.random() > 0.5 and len(boxes) > 0:
        angle = float(rng.uniform(-10, 10))
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                              borderValue=int(img.mean()))
        new_boxes = []
        for bx1, by1, bx2, by2 in boxes:
            corners = np.array([
                [bx1, by1, 1], [bx2, by1, 1],
                [bx1, by2, 1], [bx2, by2, 1]
            ]).T
            trans = np.dot(M, corners)
            nx1 = max(0, int(trans[0].min()))
            ny1 = max(0, int(trans[1].min()))
            nx2 = min(w, int(trans[0].max()))
            ny2 = min(h, int(trans[1].max()))
            if (nx2 - nx1) > 6 and (ny2 - ny1) > 6:
                new_boxes.append([nx1, ny1, nx2, ny2])
        boxes = new_boxes

    # 3. Random scale + translate
    if rng.random() > 0.4 and len(boxes) > 0:
        scale = float(rng.uniform(0.8, 1.2))
        dx = int(rng.integers(-12, 13))
        dy = int(rng.integers(-12, 13))
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        img = cv2.warpAffine(img, M, (w, h),
                              borderValue=int(img.mean()))
        new_boxes = []
        for bx1, by1, bx2, by2 in boxes:
            pts = np.array([[bx1, by1, 1], [bx2, by2, 1]]).T
            trans = np.dot(M, pts)
            nx1 = max(0, int(min(trans[0])))
            ny1 = max(0, int(min(trans[1])))
            nx2 = min(w, int(max(trans[0])))
            ny2 = min(h, int(max(trans[1])))
            if (nx2 - nx1) > 6 and (ny2 - ny1) > 6:
                new_boxes.append([nx1, ny1, nx2, ny2])
        boxes = new_boxes

    return img, boxes

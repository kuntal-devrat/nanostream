"""FOMO-style detector: MobileNetV2 backbone + per-cell classification head.

Faithful to Edge Impulse FOMO's architecture: no anchors, no box regression.
A 1x1 conv head classifies each grid cell; the model is trained with focal
loss on gaussian-smoothed center targets, and boxes are fixed cell
rectangles (expanded by ``box_scale`` to better cover objects, as EI does).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        hidden = _make_divisible(cin * expand)
        self.use_res = stride == 1 and cin == cout
        self.conv = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out


class FomoBackbone(nn.Module):
    """MobileNetV2 truncated at stride 8 (three stride-2 layers)."""

    def __init__(self, width_mult=0.75, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, _make_divisible(32 * width_mult), 3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_make_divisible(32 * width_mult)),
            nn.ReLU6(inplace=True),
        )
        cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2)]
        blocks = []
        cin = _make_divisible(32 * width_mult)
        for t, c, n, s in cfg:
            cout = _make_divisible(c * width_mult)
            for i in range(n):
                blocks.append(_InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(self.stem(x))


class FomoDetector(nn.Module):
    """Per-cell classifier: backbone -> 1x1 conv -> class logits per grid cell."""

    def __init__(self, num_classes=4, width_mult=0.75, img_size=160,
                 stride=8, box_scale=2.5):
        """Per-cell classifier.

        ``box_scale`` expands the fixed cell rectangle at decode time (cells are
        8px at 160x160/stride-8). 2.5 cells (~20px) is a fair middle ground for
        the 18-44px shapes and 32-80px faces in the benchmark dataset; 1.5 was
        too small and would artificially depress IoU-based AP for large objects.
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.stride = stride
        self.grid = img_size // stride
        self.box_scale = box_scale
        self.backbone = FomoBackbone(width_mult)
        in_ch = _make_divisible(32 * width_mult)
        self.head = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.head(self.backbone(x))  # (B, C, G, G)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    ce = F.binary_cross_entropy(p, targets, reduction="none")
    pt = torch.where(targets > 0.5, p, 1 - p)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def gaussian_targets(boxes_norm, labels, grid, sigma=0.9):
    """(grid, grid) per-class gaussian heatmaps centered on GT boxes."""
    G = grid
    targets = torch.zeros(len(labels), G, G)
    for i, (cx, cy, w, h) in enumerate(boxes_norm):
        gx = float(cx) * G
        gy = float(cy) * G
        ys, xs = torch.meshgrid(torch.arange(G, dtype=torch.float32),
                                torch.arange(G, dtype=torch.float32),
                                indexing="ij")
        d2 = (xs - gx) ** 2 + (ys - gy) ** 2
        targets[i] = torch.exp(-d2 / (2 * sigma ** 2))
    return targets


@torch.no_grad()
def predict(model, x, conf=0.25):
    """Decode detections: thresholded local-max cells -> cell boxes (xyxy norm).

    Returns (N, 6): x1, y1, x2, y2, score, class_id.
    """
    model.eval()
    if x.dim() == 3:  # accept a single (C, H, W) image
        x = x.unsqueeze(0)
    logits = model(x)
    prob = torch.sigmoid(logits)
    pooled = F.max_pool2d(prob, 3, 1, 1)
    keep = (prob >= pooled - 1e-6) & (prob >= conf)
    B, C, G, _ = prob.shape
    cell = model.stride / model.img_size
    half = cell * model.box_scale / 2
    dets = []
    for b in range(B):
        idx = torch.nonzero(keep[b])
        for (c, gy, gx) in idx.tolist():
            cx = (gx + 0.5) * cell
            cy = (gy + 0.5) * cell
            dets.append([cx - half, cy - half, cx + half, cy + half,
                         float(prob[b, c, gy, gx]), c])
    return torch.tensor(dets).reshape(-1, 6) if dets else torch.zeros(0, 6)


@torch.no_grad()
def evaluate_fomo(model, dataset, num_classes, conf=0.25, device="cpu"):
    """Unified AP evaluation in the same format as nanostream.metrics."""
    from nanostream.metrics import compute_ap_per_class, compute_map_multiscale
    import numpy as np
    model.eval()
    all_preds, all_targets = [], []
    for i in range(len(dataset)):
        x, tgt = dataset[i]
        dets = predict(model, x.unsqueeze(0).to(device), conf=conf)
        if dets.numel():
            pred = {"boxes": dets[:, :4].numpy(),
                    "scores": dets[:, 4].numpy(),
                    "class_ids": dets[:, 5].numpy().astype(int)}
        else:
            pred = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                    "class_ids": np.zeros(0, dtype=int)}
        gt_c = tgt["boxes_norm"]
        if len(gt_c):
            gt_xyxy = torch.stack([
                gt_c[:, 0] - gt_c[:, 2] / 2, gt_c[:, 1] - gt_c[:, 3] / 2,
                gt_c[:, 0] + gt_c[:, 2] / 2, gt_c[:, 1] + gt_c[:, 3] / 2],
                dim=1).numpy()
        else:
            gt_xyxy = np.zeros((0, 4))
        all_preds.append(pred)
        all_targets.append({"boxes": gt_xyxy, "labels": tgt["labels"].numpy()})
    res = compute_ap_per_class(all_preds, all_targets, num_classes)
    res["mAP_50_95"] = compute_map_multiscale(
        all_preds, all_targets, num_classes)["mAP_50_95"]
    return res

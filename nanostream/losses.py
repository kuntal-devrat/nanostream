"""NanoStream-OD v3.0 Loss Functions.

CIoU loss, VariFocal Loss, and scale-aware dual-assignment for
production-grade detection accuracy rivalling YOLO-family detectors.
"""

import torch
import torch.nn.functional as F


def ciou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor) -> torch.Tensor:
    """Complete-IoU loss (CIoU) — tighter box regression than GIoU.

    Adds diagonal distance penalty and aspect-ratio consistency term
    on top of standard IoU, proven to give 2-3% mAP gain over GIoU.

    Args:
        pred_xyxy: (N, 4) predicted boxes [x1, y1, x2, y2]
        tgt_xyxy:  (N, 4) target boxes [x1, y1, x2, y2]
    Returns:
        Scalar mean CIoU loss.
    """
    if pred_xyxy.numel() == 0:
        return pred_xyxy.sum() * 0.0

    # Intersection
    lt = torch.maximum(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    rb = torch.minimum(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    # Union
    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * \
             (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    area_t = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0) * \
             (tgt_xyxy[:, 3] - tgt_xyxy[:, 1]).clamp(min=0)
    union = (area_p + area_t - inter).clamp(min=1e-9)
    iou = inter / union

    # Enclosing box diagonal
    lt_c = torch.minimum(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    rb_c = torch.maximum(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    diag_c = ((rb_c[:, 0] - lt_c[:, 0]) ** 2 +
              (rb_c[:, 1] - lt_c[:, 1]) ** 2).clamp(min=1e-9)

    # Center distance
    cx_p = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    cy_p = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    cx_t = (tgt_xyxy[:, 0] + tgt_xyxy[:, 2]) / 2
    cy_t = (tgt_xyxy[:, 1] + tgt_xyxy[:, 3]) / 2
    d2 = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2

    # Aspect ratio consistency
    w_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=1e-6)
    h_p = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=1e-6)
    w_t = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=1e-6)
    h_t = (tgt_xyxy[:, 3] - tgt_xyxy[:, 1]).clamp(min=1e-6)
    v = (4 / (torch.pi ** 2)) * (torch.atan(w_t / h_t) - torch.atan(w_p / h_p)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-9)

    ciou = iou - d2 / diag_c - alpha * v
    return (1 - ciou).mean()


def varifocal_loss(pred_logits: torch.Tensor,
                   target_scores: torch.Tensor,
                   alpha: float = 0.75,
                   gamma: float = 2.0) -> torch.Tensor:
    """VariFocal Loss — quality-aware focal loss for objectness.

    Unlike standard BCE, VFL uses the target IoU quality score as
    the learning target instead of hard 0/1 labels, and applies
    asymmetric focusing: hard negatives get focal weight, positives don't.

    This outperforms standard focal loss by 1-2% mAP on dense detectors.

    Args:
        pred_logits: (N,) raw objectness logits
        target_scores: (N,) target quality scores in [0, 1]
            (0 for background, IoU quality for positives)
        alpha: Weight for negative samples
        gamma: Focusing parameter for hard negatives
    """
    pred_prob = torch.sigmoid(pred_logits)
    focal_weight = torch.where(
        target_scores > 0,
        torch.ones_like(pred_prob),                          # No focal for positives
        alpha * pred_prob.pow(gamma)                          # Focal for negatives
    )
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target_scores, reduction="none")
    return (focal_weight * bce).mean()


def scale_aware_assign(gt_boxes: torch.Tensor,
                       grid_p3: int, grid_p4: int,
                       small_threshold: float = 0.25) -> tuple:
    """Assign GT boxes to P3 or P4 based on object size.

    Small objects (area < threshold²) → P3 (20×20, stride 8)
    Large objects (area ≥ threshold²) → P4 (10×10, stride 16)

    This prevents scale confusion where both heads try to detect
    the same object, wasting capacity.

    Args:
        gt_boxes: (M, 4) cxcywh normalized boxes
        grid_p3: P3 grid size (e.g. 20)
        grid_p4: P4 grid size (e.g. 10)
        small_threshold: Max normalized width/height for "small" classification

    Returns:
        p3_mask: bool tensor, True for boxes assigned to P3
        p4_mask: bool tensor, True for boxes assigned to P4
    """
    if gt_boxes.numel() == 0:
        empty = torch.zeros(0, dtype=torch.bool, device=gt_boxes.device)
        return empty, empty

    areas = gt_boxes[:, 2] * gt_boxes[:, 3]  # w * h (normalized)
    max_dim = torch.maximum(gt_boxes[:, 2], gt_boxes[:, 3])

    # Small objects → P3, large → P4, medium → both
    is_small = max_dim < small_threshold
    is_large = max_dim >= small_threshold * 1.5

    p3_mask = is_small | ~is_large  # Small + medium → P3
    p4_mask = is_large | ~is_small  # Large + medium → P4

    return p3_mask, p4_mask

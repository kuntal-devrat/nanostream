"""NanoStream-OD v3.0 Evaluation Metrics.

Proper mAP@50, mAP@50:95, per-class AP, and F1 computation
for rigorous comparison against YOLO/FOMO/MCUNet benchmarks.
"""

import torch
import numpy as np
from typing import List, Dict


def compute_iou_matrix(pred_boxes: np.ndarray,
                       gt_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between prediction and GT boxes.

    Args:
        pred_boxes: (N, 4) [x1, y1, x2, y2]
        gt_boxes:   (M, 4) [x1, y1, x2, y2]

    Returns:
        iou_matrix: (N, M)
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    x1 = np.maximum(pred_boxes[:, 0:1], gt_boxes[:, 0:1].T)
    y1 = np.maximum(pred_boxes[:, 1:2], gt_boxes[:, 1:2].T)
    x2 = np.minimum(pred_boxes[:, 2:3], gt_boxes[:, 2:3].T)
    y2 = np.minimum(pred_boxes[:, 3:4], gt_boxes[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
             (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
             (gt_boxes[:, 3] - gt_boxes[:, 1])

    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP from recall-precision curve using 101-point interpolation.

    This matches the COCO evaluation protocol.
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        # Find precision at this recall threshold
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            ap += mpre[idx[0]]
    return ap / 101.0


def compute_ap_per_class(all_predictions: List[Dict],
                         all_targets: List[Dict],
                         num_classes: int,
                         iou_threshold: float = 0.50) -> Dict:
    """Compute per-class Average Precision.

    Args:
        all_predictions: List of dicts with keys:
            - 'boxes': (N, 4) xyxy normalized
            - 'scores': (N,) confidence scores
            - 'class_ids': (N,) predicted class indices
        all_targets: List of dicts with keys:
            - 'boxes': (M, 4) xyxy normalized
            - 'labels': (M,) class indices
        num_classes: Total number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        Dict with 'per_class_ap', 'mAP', 'precision', 'recall', 'f1'
    """
    # Collect all predictions and GT per class
    class_preds = {c: [] for c in range(num_classes)}
    class_n_gt = {c: 0 for c in range(num_classes)}
    gt_matched = []

    for img_idx, (pred, gt) in enumerate(zip(all_predictions, all_targets)):
        gt_boxes = np.array(gt['boxes']) if len(gt['boxes']) > 0 else np.zeros((0, 4))
        gt_labels = np.array(gt['labels']) if len(gt['labels']) > 0 else np.zeros(0, dtype=int)

        for c in range(num_classes):
            class_n_gt[c] += int((gt_labels == c).sum())

        pred_boxes = np.array(pred['boxes']) if len(pred['boxes']) > 0 else np.zeros((0, 4))
        pred_scores = np.array(pred['scores']) if len(pred['scores']) > 0 else np.zeros(0)
        pred_classes = np.array(pred['class_ids']) if len(pred['class_ids']) > 0 else np.zeros(0, dtype=int)

        for c in range(num_classes):
            c_mask = pred_classes == c
            for i in np.where(c_mask)[0]:
                class_preds[c].append({
                    'img_idx': img_idx,
                    'score': pred_scores[i],
                    'box': pred_boxes[i],
                })

        gt_matched.append(np.zeros(len(gt_labels), dtype=bool))

    # Compute AP per class
    per_class_ap = {}
    per_class_precision = {}
    per_class_recall = {}

    for c in range(num_classes):
        preds_c = class_preds[c]
        n_gt = class_n_gt[c]

        if n_gt == 0:
            per_class_ap[c] = 0.0
            per_class_precision[c] = 0.0
            per_class_recall[c] = 0.0
            continue

        # Sort by score descending
        preds_c.sort(key=lambda x: -x['score'])

        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))

        # Reset GT matched flags for this class
        img_gt_matched = {}

        for pi, pred in enumerate(preds_c):
            img_idx = pred['img_idx']
            gt = all_targets[img_idx]
            gt_boxes = np.array(gt['boxes']) if len(gt['boxes']) > 0 else np.zeros((0, 4))
            gt_labels = np.array(gt['labels']) if len(gt['labels']) > 0 else np.zeros(0, dtype=int)

            # Filter GT to this class
            c_gt_mask = gt_labels == c
            c_gt_boxes = gt_boxes[c_gt_mask]

            if len(c_gt_boxes) == 0:
                fp[pi] = 1
                continue

            iou = compute_iou_matrix(pred['box'].reshape(1, 4), c_gt_boxes)
            best_iou_idx = iou[0].argmax()
            best_iou = iou[0, best_iou_idx]

            # Track which GT boxes are already matched per image
            if img_idx not in img_gt_matched:
                img_gt_matched[img_idx] = np.zeros(c_gt_mask.sum(), dtype=bool)

            if best_iou >= iou_threshold and not img_gt_matched[img_idx][best_iou_idx]:
                tp[pi] = 1
                img_gt_matched[img_idx][best_iou_idx] = True
            else:
                fp[pi] = 1

        # Cumulative TP/FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / max(n_gt, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        per_class_ap[c] = _compute_ap(recalls, precisions)
        per_class_precision[c] = float(precisions[-1]) if len(precisions) > 0 else 0.0
        per_class_recall[c] = float(recalls[-1]) if len(recalls) > 0 else 0.0

    # mAP
    valid_aps = [ap for c, ap in per_class_ap.items() if class_n_gt[c] > 0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0

    # Overall precision/recall/F1 — FIX: average over ALL classes including
    # zeros. The old code dropped zero-precision classes, inflating aggregates.
    per_class_precision_list = [per_class_precision[c] for c in range(num_classes)]
    per_class_recall_list = [per_class_recall[c] for c in range(num_classes)]
    total_precision = float(np.mean(per_class_precision_list)) if num_classes > 0 else 0.0
    total_recall = float(np.mean(per_class_recall_list)) if num_classes > 0 else 0.0
    denom = total_precision + total_recall
    f1 = float(2 * total_precision * total_recall / denom) if denom > 1e-9 else 0.0

    return {
        'per_class_ap': per_class_ap,
        'mAP': float(mAP),
        'mAP_50': float(mAP),
        'precision': float(total_precision),
        'recall': float(total_recall),
        'f1': float(f1),
    }


def compute_map_multiscale(all_predictions: List[Dict],
                           all_targets: List[Dict],
                           num_classes: int,
                           iou_thresholds: List[float] = None) -> Dict:
    """Compute mAP at multiple IoU thresholds (COCO-style mAP@50:95).

    Args:
        all_predictions, all_targets: Same format as compute_ap_per_class
        num_classes: Total classes
        iou_thresholds: List of IoU thresholds (default: 0.50 to 0.95 step 0.05)

    Returns:
        Dict with 'mAP_50', 'mAP_50_95', per-threshold mAPs
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    results = {}
    all_maps = []

    for iou_thr in iou_thresholds:
        r = compute_ap_per_class(all_predictions, all_targets,
                                 num_classes, iou_threshold=iou_thr)
        key = f"mAP_{int(iou_thr*100)}"
        results[key] = r['mAP']
        all_maps.append(r['mAP'])

    results['mAP_50'] = results.get('mAP_50', 0.0)
    results['mAP_50_95'] = float(np.mean(all_maps))

    return results


@torch.no_grad()
def evaluate_model(model, dataset, num_classes: int,
                   n_samples: int = 100, conf_thr: float = 0.25,
                   device: str = "cpu") -> Dict:
    """End-to-end model evaluation with mAP metrics.

    Args:
        model: NanoStreamOD model (eval mode)
        dataset: Dataset returning (tensor, target_dict)
        num_classes: Number of classes
        n_samples: Number of samples to evaluate
        conf_thr: Detection confidence threshold
        device: torch device string

    Returns:
        Dict with mAP@50, mAP@50:95, per-class AP, etc.
    """
    from .head import decode_detections

    model.eval()
    all_preds = []
    all_targets = []

    n_eval = min(n_samples, len(dataset))

    for i in range(n_eval):
        x, tgt = dataset[i]
        x_batch = x.unsqueeze(0).to(device)

        preds = model(x_batch)
        dets = decode_detections(preds, conf_thr=conf_thr)

        # Convert to evaluation format
        if dets.numel() > 0:
            pred_dict = {
                'boxes': dets[:, :4].cpu().numpy(),
                'scores': dets[:, 4].cpu().numpy(),
                'class_ids': dets[:, 5].cpu().numpy().astype(int),
            }
        else:
            pred_dict = {
                'boxes': np.zeros((0, 4)),
                'scores': np.zeros(0),
                'class_ids': np.zeros(0, dtype=int),
            }

        # Convert GT boxes from cxcywh to xyxy
        gt_cxcywh = tgt['boxes_norm']
        if len(gt_cxcywh) > 0:
            gt_xyxy = torch.stack([
                gt_cxcywh[:, 0] - gt_cxcywh[:, 2] / 2,
                gt_cxcywh[:, 1] - gt_cxcywh[:, 3] / 2,
                gt_cxcywh[:, 0] + gt_cxcywh[:, 2] / 2,
                gt_cxcywh[:, 1] + gt_cxcywh[:, 3] / 2,
            ], dim=1).numpy()
            gt_labels = tgt['labels'].numpy()
        else:
            gt_xyxy = np.zeros((0, 4))
            gt_labels = np.zeros(0, dtype=int)

        gt_dict = {
            'boxes': gt_xyxy,
            'labels': gt_labels,
        }

        all_preds.append(pred_dict)
        all_targets.append(gt_dict)

    # Compute metrics
    result = compute_ap_per_class(all_preds, all_targets, num_classes,
                                  iou_threshold=0.50)

    # Also compute mAP@50:95
    ms = compute_map_multiscale(all_preds, all_targets, num_classes)
    result['mAP_50'] = result['mAP']
    result['mAP_50_95'] = ms['mAP_50_95']

    return result

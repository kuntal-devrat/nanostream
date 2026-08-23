"""Tests for NanoStream-OD v3.0 modules: losses, augmentations, metrics, and Lite-FPN."""

import numpy as np
import pytest
import torch

from nanostream.config import NanoStreamConfig
from nanostream.losses import ciou_loss, varifocal_loss, scale_aware_assign
from nanostream.augment import mosaic_4, mixup, cutout, photometric_distort, multi_scale_resize, geometric_augment
from nanostream.metrics import compute_ap_per_class, compute_map_multiscale, evaluate_model
from nanostream.neck import LiteFPN
from nanostream.model import NanoStreamOD
from nanostream.data import SyntheticShapes


def test_ciou_loss_perfect_and_disjoint():
    # Perfect match: IoU = 1, distance = 0, aspect ratio penalty = 0 => loss = 0
    boxes = torch.tensor([[0.2, 0.2, 0.6, 0.6], [0.1, 0.1, 0.4, 0.4]])
    loss_perfect = ciou_loss(boxes, boxes)
    assert loss_perfect.item() < 1e-4

    # Disjoint boxes: loss should be > 1.0 (due to distance penalty)
    box_a = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
    box_b = torch.tensor([[0.8, 0.8, 0.9, 0.9]])
    loss_disjoint = ciou_loss(box_a, box_b)
    assert loss_disjoint.item() > 1.0


def test_varifocal_loss():
    logits = torch.randn(100)
    # Binary targets (standard)
    targets_hard = torch.randint(0, 2, (100,)).float()
    vfl_hard = varifocal_loss(logits, targets_hard)
    assert vfl_hard.item() > 0.0

    # Continuous IoU targets (quality-aware)
    targets_soft = torch.rand(100)
    vfl_soft = varifocal_loss(logits, targets_soft)
    assert vfl_soft.item() > 0.0


def test_scale_aware_assignment():
    # 3 boxes: 1 small, 1 medium, 1 large
    gt_boxes = torch.tensor([
        [0.5, 0.5, 0.10, 0.10],  # small
        [0.5, 0.5, 0.30, 0.30],  # medium
        [0.5, 0.5, 0.60, 0.60],  # large
    ])
    p3_mask, p4_mask = scale_aware_assign(gt_boxes, grid_p3=20, grid_p4=10, small_threshold=0.25)

    assert p3_mask[0].item() is True   # Small goes to P3
    assert p4_mask[2].item() is True   # Large goes to P4


def test_mosaic_augmentation():
    rng = np.random.default_rng(42)
    imgs = [np.full((160, 160), i * 50, dtype=np.uint8) for i in range(4)]
    boxes = [
        [[20, 20, 60, 60]],
        [[30, 30, 70, 70]],
        [[10, 10, 50, 50]],
        [[40, 40, 80, 80]],
    ]
    labels = [[0], [1], [2], [0]]

    m_img, m_boxes, m_labels = mosaic_4(imgs, boxes, labels, target_size=160, rng=rng)

    assert m_img.shape == (160, 160)
    assert len(m_boxes) > 0
    assert len(m_boxes) == len(m_labels)
    for b in m_boxes:
        assert 0 <= b[0] <= 160
        assert 0 <= b[1] <= 160
        assert b[2] >= b[0]
        assert b[3] >= b[1]


def test_mixup_and_cutout():
    rng = np.random.default_rng(42)
    img1 = np.full((160, 160), 50, dtype=np.uint8)
    img2 = np.full((160, 160), 200, dtype=np.uint8)
    b1 = [[20, 20, 50, 50]]
    b2 = [[80, 80, 120, 120]]

    mixed, m_boxes, m_labels = mixup(img1, b1, [0], img2, b2, [1], alpha=0.5, rng=rng)
    assert mixed.shape == (160, 160)
    assert len(m_boxes) == 2
    assert len(m_labels) == 2

    # CutOut
    cut_img = cutout(mixed, m_boxes, n_holes=2, rng=rng)
    assert cut_img.shape == (160, 160)
    assert not np.array_equal(cut_img, mixed)


def test_metrics_ap_and_map():
    # Synthetic predictions and targets
    targets = [
        {"boxes": [[0.1, 0.1, 0.3, 0.3]], "labels": [0]},
        {"boxes": [[0.5, 0.5, 0.8, 0.8]], "labels": [1]},
    ]
    # Perfect predictions
    preds_perfect = [
        {"boxes": [[0.1, 0.1, 0.3, 0.3]], "scores": [0.95], "class_ids": [0]},
        {"boxes": [[0.5, 0.5, 0.8, 0.8]], "scores": [0.90], "class_ids": [1]},
    ]

    res = compute_ap_per_class(preds_perfect, targets, num_classes=2, iou_threshold=0.5)
    assert res["mAP"] > 0.90
    assert res["recall"] > 0.90
    assert res["precision"] > 0.90

    ms = compute_map_multiscale(preds_perfect, targets, num_classes=2)
    assert "mAP_50" in ms
    assert "mAP_50_95" in ms
    assert ms["mAP_50"] > 0.90


def test_lite_fpn_neck():
    cfg = NanoStreamConfig()
    neck = LiteFPN(cfg)
    neck.eval()

    p3 = torch.randn(2, 32, 20, 20)
    p4 = torch.randn(2, 48, 10, 10)

    fused = neck(p3, p4)
    assert fused.shape == (2, 32, 20, 20)

    # Integer forward test
    p3_int = torch.randint(-2048, 2048, (1, 32, 20, 20), dtype=torch.int32)
    p4_int = torch.randint(-2048, 2048, (1, 48, 10, 10), dtype=torch.int32)
    neck.lateral_p4.freeze_pow2()
    neck.smooth_p3.freeze_pow2()

    fused_int, out_frac = neck.forward_int(p3_int, p4_int, p3_frac=12, p4_frac=12)
    assert fused_int.shape == (1, 32, 20, 20)
    assert fused_int.dtype == torch.int32


def test_evaluate_model_end_to_end():
    model = NanoStreamOD()
    model.eval()
    ds = SyntheticShapes(length=10, seed=42)

    metrics = evaluate_model(model, ds, num_classes=3, n_samples=5, conf_thr=0.05)
    assert "mAP" in metrics
    assert "mAP_50" in metrics
    assert "mAP_50_95" in metrics
    assert "f1" in metrics

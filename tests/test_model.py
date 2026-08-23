"""Tests for NanoStream-OD core architecture and Zero-NMS head."""

import torch

from nanostream.config import NanoStreamConfig
from nanostream.data import make_sample, to_target
from nanostream.head import decode_detections, detection_loss
from nanostream.model import NanoStreamOD


def test_model_forward_shapes():
    cfg = NanoStreamConfig(input_size=160, in_channels=1, num_classes=3)
    model = NanoStreamOD(cfg)
    model.eval()

    x = torch.randn(2, 1, 160, 160)
    with torch.no_grad():
        preds = model(x)

    assert "obj" in preds
    assert "box" in preds
    assert "cls" in preds
    assert preds["obj"].shape == (2, 1, 10, 10)
    assert preds["box"].shape == (2, 4, 10, 10)
    assert preds["cls"].shape == (2, 3, 10, 10)
    assert preds["G"] == 10


def test_detection_loss_computes():
    cfg = NanoStreamConfig()
    model = NanoStreamOD(cfg)

    # Generate synthetic targets
    targets = []
    for _ in range(2):
        _, boxes, labels = make_sample(size=160, max_objects=2)
        tgt = to_target(boxes, labels, size=160)
        targets.append(tgt)

    x = torch.randn(2, 1, 160, 160)
    preds = model(x)
    losses = detection_loss(preds, targets, cfg)

    assert "total" in losses
    assert "obj" in losses
    assert "box" in losses
    assert "cls" in losses
    assert losses["total"].item() > 0.0

    # FIX: gradients + optimizer step must work (backprop was untested before)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients produced by detection_loss"
    opt.step()


def test_zero_nms_decode():
    cfg = NanoStreamConfig()
    model = NanoStreamOD(cfg)
    model.eval()

    # FIX: use a known-positive input so an empty decode FAILS the test
    x = torch.zeros(1, 1, 160, 160)
    x[0, 0, 55:105, 55:105] = 1.0  # bright square
    with torch.no_grad():
        preds = model(x)
        dets = decode_detections(preds, conf_thr=0.01, max_det=10)

    # Tensor format: [x1, y1, x2, y2, score, cls_id]
    assert dets.shape[1] == 6
    assert dets.shape[0] > 0, "decode returned nothing on a known-positive input"
    assert (dets[:, :4] >= 0.0).all()
    assert (dets[:, :4] <= 1.0).all()
    assert (dets[:, 4] >= 0.0).all() and (dets[:, 4] <= 1.0).all()


def test_cross_scale_dedup_is_class_aware():
    """Two overlapping detections of DIFFERENT classes must BOTH survive."""
    from nanostream.head import _class_aware_dedup
    dets = torch.tensor([
        [0.2, 0.2, 0.6, 0.6, 0.90, 0.0],  # class 0, high score
        [0.3, 0.3, 0.7, 0.7, 0.85, 1.0],  # class 1, overlaps class 0
        [0.2, 0.2, 0.6, 0.6, 0.80, 0.0],  # class 0, identical box (IoU 1.0)
    ])
    kept = _class_aware_dedup(dets, iou_thr=0.5)
    # class-0 dupe suppressed (IoU=1.0, same class); class-1 kept despite
    # overlapping class 0 (different class)
    assert kept.shape[0] == 2
    assert set(kept[:, 5].tolist()) == {0.0, 1.0}

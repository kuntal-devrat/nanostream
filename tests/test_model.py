"""Tests for NanoStream-OD core architecture and Zero-NMS head."""

import pytest
import torch

from nanostream.config import NanoStreamConfig
from nanostream.data import make_sample, to_target
from nanostream.head import decode_cells, decode_detections, detection_loss
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


def test_zero_nms_decode():
    cfg = NanoStreamConfig()
    model = NanoStreamOD(cfg)
    model.eval()

    x = torch.randn(1, 1, 160, 160)
    with torch.no_grad():
        preds = model(x)
        dets = decode_detections(preds, conf_thr=0.01, max_det=10)

    # Tensor format: [x1, y1, x2, y2, score, cls_id]
    if dets.numel() > 0:
        assert dets.shape[1] == 6
        assert (dets[:, :4] >= 0.0).all()
        assert (dets[:, :4] <= 1.0).all()
        assert (dets[:, 4] >= 0.0).all() and (dets[:, 4] <= 1.0).all()

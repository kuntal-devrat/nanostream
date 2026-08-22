"""Tests for Patch-Streaming Backbone and Ring Buffer execution."""

import pytest
import torch

from nanostream.config import NanoStreamConfig
from nanostream.model import NanoStreamOD


def test_streaming_backbone_equivalence():
    """Verify that patch-streaming produces the same grid features as full-frame forward."""
    cfg = NanoStreamConfig(input_size=160, in_channels=1, strip_rows=16)
    model = NanoStreamOD(cfg)
    model.eval()

    # Freeze batch norm running stats to ensure deterministic behavior
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    x = torch.randn(1, 1, 160, 160)

    with torch.no_grad():
        full_feats = model.backbone(x)["p4"][0]
        streamer = model.backbone.make_streamer()
        stream_feats, ctx_sum, ctx_n = streamer.infer_grid(x[0])

    assert stream_feats.shape == full_feats.shape
    assert stream_feats.shape == (cfg.stage_widths[-1], 10, 10)
    assert ctx_n == 100

    # Max difference between full frame conv and ring-buffered streaming conv
    diff = (full_feats - stream_feats).abs().max().item()
    assert diff < 1e-4, f"Streaming feature mismatch: max diff = {diff}"


def test_stream_forward_api():
    model = NanoStreamOD()
    model.eval()

    img = torch.randn(1, 160, 160)
    dets, preds = model.stream_forward(img, conf_thr=0.1)

    assert "obj" in preds
    assert "box" in preds
    assert "cls" in preds
    assert isinstance(dets, torch.Tensor)

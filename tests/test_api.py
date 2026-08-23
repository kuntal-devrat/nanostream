"""Test high-level NanoStream framework API."""

import numpy as np
import torch
import nanostream as ns


def test_api_load_model_and_config():
    model = ns.load_model()
    assert isinstance(model, ns.NanoStreamOD)
    assert isinstance(model.cfg, ns.NanoStreamConfig)
    assert model.param_count() > 0


def test_api_detect_numpy_image():
    model = ns.load_model()
    # FIX: use a KNOWN-POSITIVE input (bright square on dark background) so an
    # empty result FAILS the test — previously a uniform-128 image with conf 0.05
    # passed vacuously even if detect() always returned [].
    dummy_img = np.full((160, 160, 3), 15, dtype=np.uint8)
    dummy_img[55:105, 55:105] = 235
    dets = ns.detect(model, dummy_img, conf_thr=0.10)
    assert isinstance(dets, list)
    assert len(dets) > 0, "detect() returned no detections on a known-positive input"
    for d in dets:
        assert isinstance(d, ns.Detection)
        assert 0.0 <= d.score <= 1.0
        assert 0.0 <= d.x1 <= 1.0
        assert 0.0 <= d.y1 <= 1.0
        assert d.x2 >= d.x1
        assert d.y2 >= d.y1


def test_api_draw_detections():
    dummy_img = np.full((200, 200, 3), 50, dtype=np.uint8)
    dummy_det = ns.Detection(x1=0.2, y1=0.2, x2=0.6, y2=0.6, score=0.85, class_id=0, class_name="face")
    vis = ns.draw_detections(dummy_img, [dummy_det])
    assert vis.shape == dummy_img.shape
    # Check that drawing actually modified canvas
    assert not np.array_equal(vis, dummy_img)


def test_api_preprocess_image():
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    tensor, orig = ns.preprocess_image(dummy_img, target_size=160)
    assert tensor.shape == (1, 1, 160, 160)
    assert tensor.dtype == torch.float32
    assert -1.0 <= tensor.min() <= tensor.max() <= 1.0

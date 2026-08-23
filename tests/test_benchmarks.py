"""Tests for the cross-framework benchmark infrastructure."""

import numpy as np
import torch

from benchmarks.combined_data import (CLASS_NAMES, NUM_CLASSES, SIZE,
                                      TRAIN_LEN, VAL_START, CombinedDataset,
                                      generate_sample, to_target)
from benchmarks.fomo_model import (FomoDetector, focal_loss,
                                   gaussian_targets, predict)


def test_generate_sample_deterministic():
    img1, boxes1, labels1 = generate_sample(123)
    img2, boxes2, labels2 = generate_sample(123)
    assert np.array_equal(img1, img2)
    assert boxes1 == boxes2
    assert labels1 == labels2


def test_combined_dataset_deterministic_across_instances():
    a = CombinedDataset(length=8, start_idx=VAL_START)
    b = CombinedDataset(length=8, start_idx=VAL_START)
    xa, ta = a[3]
    xb, tb = b[3]
    assert torch.equal(xa, xb)
    assert torch.equal(ta["boxes_norm"], tb["boxes_norm"])
    assert torch.equal(ta["labels"], tb["labels"])


def test_train_val_indices_never_overlap():
    # Train samples come from [0, train_len); val from [VAL_START, ...].
    # The two ranges are disjoint for any reasonable train_len.
    assert VAL_START > 1000
    assert TRAIN_LEN < VAL_START


def test_combined_dataset_contains_faces():
    n_face = 0
    ds = CombinedDataset(length=60, start_idx=VAL_START)
    for i in range(len(ds)):
        _, tgt = ds[i]
        if (tgt["labels"] == 3).any():
            n_face += 1
    # Faces must be well represented or the benchmark says nothing about them.
    assert n_face >= 10, f"only {n_face}/60 val images contain a face"


def test_classes_include_face():
    assert CLASS_NAMES[-1] == "face"
    assert NUM_CLASSES == 4


def test_to_target_empty():
    tgt = to_target([], [], SIZE)
    assert tgt["boxes_norm"].shape == (0, 4)
    assert tgt["labels"].shape == (0,)


def test_fomo_gaussian_targets_shape_and_peak():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    labels = torch.tensor([0])
    t = gaussian_targets(boxes, labels, grid=20)
    assert t.shape == (1, 20, 20)
    assert t[0, 10, 10] > 0.99  # peak at the center cell


def test_fomo_focal_loss_smoke():
    logits = torch.zeros(2, 4, 20, 20)
    targets = torch.zeros(2, 4, 20, 20)
    targets[0, 1, 10, 10] = 1.0
    loss = focal_loss(logits, targets)
    assert loss.ndim == 0
    assert 0.0 < float(loss) < 1.0


def test_fomo_predict_columns():
    model = FomoDetector(num_classes=NUM_CLASSES, width_mult=0.5)
    x = torch.randn(1, 1, SIZE, SIZE)
    dets = predict(model, x, conf=0.9)  # high conf -> likely empty, shape only
    assert dets.shape[1] == 6


def test_fomo_predict_detects_obvious_object():
    model = FomoDetector(num_classes=NUM_CLASSES, width_mult=0.5)
    img = np.full((SIZE, SIZE), 20, dtype=np.uint8)
    img[50:110, 50:110] = 220  # bright square in the middle
    x = torch.from_numpy(img).float() / 127.5 - 1.0
    dets = predict(model, x.unsqueeze(0), conf=0.01)
    assert dets.shape[1] == 6
    assert dets.numel() > 0  # some cell must fire on a big bright square


def test_yolo_export(tmp_path):
    from benchmarks.combined_data import export_yolo
    yaml_path = export_yolo(tmp_path)
    assert yaml_path.exists()
    assert (tmp_path / "train" / "images").is_dir()
    assert (tmp_path / "val" / "labels").is_dir()
    txt = (tmp_path / "train" / "labels").glob("*.txt")
    labels = [p.read_text().strip().split("\n")[0].split()[0] for p in txt][:5]
    assert all(c in "0123" for c in labels)
    yaml_text = yaml_path.read_text()
    assert "nc: 4" in yaml_text
    assert "face" in yaml_text


def test_nanostream_trains_on_combined():
    from nanostream.config import NanoStreamConfig
    from nanostream.head import detection_loss
    from nanostream.model import NanoStreamOD
    from benchmarks.combined_data import collate

    torch.manual_seed(0)
    cfg = NanoStreamConfig(num_classes=NUM_CLASSES, input_size=SIZE,
                           dual_scale=True, neck_type="lite_fpn")
    model = NanoStreamOD(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    ds = CombinedDataset(length=16, start_idx=0)
    losses = []
    for step in range(8):
        idxs = np.arange(0, 16, dtype=int)
        xs, ts = collate([ds[int(i)] for i in idxs])
        preds = model(xs)
        loss = detection_loss(preds, ts, cfg)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        losses.append(float(loss["total"]))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_nanostream_resume_continues():
    from types import SimpleNamespace
    from benchmarks.train_nanostream import train
    import pathlib
    import tempfile

    out = tempfile.mkdtemp()
    base = dict(profile="mcu", batch=4, lr=2e-3, seed=0, input_size=0,
                data_len=200, device="cpu", out=out, save_every=6)
    train(SimpleNamespace(steps=12, resume=False, **base))
    ckpt_path = pathlib.Path(out) / "nanostream_mcu.pt"
    assert ckpt_path.exists()
    first = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert first["step"] == 11  # final checkpoint records the last trained step
    train(SimpleNamespace(steps=20, resume=True, **base))
    second = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert second["step"] == 19  # resumed run continued, not restarted


def test_config_profiles_scale_capacity():
    from nanostream.config import NanoStreamConfig
    mcu = NanoStreamConfig.mcu(num_classes=4)
    pro = NanoStreamConfig.pro(num_classes=4)
    gpu = NanoStreamConfig.gpu(num_classes=4)
    assert mcu.input_size == 160
    assert pro.input_size == 256
    assert gpu.input_size == 320
    # widths strictly increase across tiers
    assert mcu.stage_widths[-1] < pro.stage_widths[-1] < gpu.stage_widths[-1]
    assert mcu.head_hidden < pro.head_hidden < gpu.head_hidden
    # all tiers keep dual-scale + neck (the float accuracy path)
    for c in (mcu, pro, gpu):
        assert c.dual_scale and c.neck_type == "lite_fpn"


def test_augment_sample_produces_valid_sample():
    from benchmarks.train_nanostream import augment_sample
    from nanostream.config import NanoStreamConfig
    cfg = NanoStreamConfig.pro(num_classes=4)
    rng = np.random.default_rng(0)
    img, boxes, labels = augment_sample(cfg, rng, data_len=200)
    assert img.shape == (cfg.input_size, cfg.input_size)
    assert img.dtype == np.uint8
    assert len(boxes) == len(labels)
    size = cfg.input_size
    for (x1, y1, x2, y2) in boxes:
        assert 0 <= x1 < x2 <= size
        assert 0 <= y1 < y2 <= size


def test_augment_sample_deterministic_per_seed():
    from benchmarks.train_nanostream import augment_sample
    from nanostream.config import NanoStreamConfig
    cfg = NanoStreamConfig.mcu(num_classes=4)
    a = augment_sample(cfg, np.random.default_rng(7), data_len=100)
    b = augment_sample(cfg, np.random.default_rng(7), data_len=100)
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]

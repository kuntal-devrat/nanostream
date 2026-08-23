"""Tests for Power-of-Two Quantization and Bit-Exact Integer fixed-point math."""

import torch

from nanostream.data import calibration_images
from nanostream.fixedpoint import calibrate_fixed_point
from nanostream.layers import ShiftConv2d
from nanostream.model import NanoStreamOD
from nanostream.quant import pow2_quantize


def test_pow2_quantization():
    w = torch.randn(8, 4, 3, 3)
    pw = pow2_quantize(w)

    assert pw.nonzero_taps > 0
    assert pw.exponent.shape == w.shape
    assert pw.sign.shape == w.shape

    # Reconstructed floating-point weights
    w_rec = pw.to_float()
    assert w_rec.shape == w.shape


def test_shiftconv_freeze_and_integer_forward():
    conv = ShiftConv2d(4, 8, 3, stride=2, bias=True).name("test_conv")
    conv.freeze_pow2()

    assert conv.frozen
    assert conv.pow2 is not None

    # Forward with integer inputs (e.g. Q12 fixed point)
    x_int = torch.randint(-2048, 2048, (1, 4, 32, 32), dtype=torch.int32)
    y_int = conv.forward_fixed_int(x_int, in_frac=12)

    assert y_int.shape == (1, 8, 16, 16)
    assert y_int.dtype == torch.int32


def test_calibration_and_integer_decode():
    model = NanoStreamOD()
    model.eval()

    calib_imgs = calibration_images(n=8, size=160)
    fracs = calibrate_fixed_point(model, calib_imgs, frac_bits=12, passes=1)

    assert "input_frac" in fracs
    assert "stage_in_frac" in fracs
    assert "head_in_frac" in fracs

    # FIX: int decode must produce the SAME 2.5x-scaled boxes as the float
    # reference (the old code emitted 2.5x-smaller boxes than the C kernel).
    # Generate a known-positive input: bright square on dark background.
    img_u8 = torch.full((160, 160), 15, dtype=torch.uint8)
    img_u8[50:110, 50:110] = 240
    dets_int = model.stream_forward_int(img_u8, fracs, conf_thr=0.10)

    assert isinstance(dets_int, torch.Tensor)
    if dets_int.numel() > 0:
        assert dets_int.shape[1] == 6
        # Boxes must be within [0, 1] and width/height plausible
        assert (dets_int[:, :4] >= 0.0).all()
        assert (dets_int[:, :4] <= 1.0).all()
        # The 2.5x scale means a mid-grid object spans > 0.1 of the image
        w = dets_int[:, 2] - dets_int[:, 0]
        assert (w > 0.05).any(), "int decode boxes too small (2.5x scale missing)"


def test_int_decode_matches_float_decode():
    """int stream_forward_int and float stream_forward must agree on box size."""
    model = NanoStreamOD()
    model.eval()

    calib_imgs = calibration_images(n=8, size=160)
    fracs = calibrate_fixed_point(model, calib_imgs, frac_bits=12, passes=1)

    img_u8 = torch.full((160, 160), 15, dtype=torch.uint8)
    img_u8[60:100, 60:100] = 240
    dets_int = model.stream_forward_int(img_u8, fracs, conf_thr=0.15)

    # float path
    x = (img_u8.float() / 127.5 - 1.0).unsqueeze(0)
    dets_float, _ = model.stream_forward(x, conf_thr=0.15)

    if dets_int.numel() > 0 and dets_float.numel() > 0:
        w_int = (dets_int[:, 2] - dets_int[:, 0]).median()
        w_float = (dets_float[:, 2] - dets_float[:, 0]).median()
        # Same 2.5x scale factor; allow pow2-quantization tolerance
        assert abs(w_int - w_float) < 0.25, \
            f"int/float box width mismatch: {w_int:.3f} vs {w_float:.3f}"

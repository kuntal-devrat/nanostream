"""Tests for Power-of-Two Quantization and Bit-Exact Integer fixed-point math."""

import pytest
import torch

from nanostream.data import calibration_images
from nanostream.fixedpoint import calibrate_fixed_point, decode_int_detections, sig_lut_q
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

    # Test int stream forward
    test_img_u8 = torch.randint(0, 255, (160, 160), dtype=torch.uint8)
    dets = model.stream_forward_int(test_img_u8, fracs, conf_thr=0.01)

    assert isinstance(dets, torch.Tensor)
    if dets.numel() > 0:
        assert dets.shape[1] == 6

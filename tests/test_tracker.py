"""Tests for MCU Resource & SRAM Tracker."""

import pytest
import torch

from nanostream.model import NanoStreamOD
from nanostream.tracker import ResourceTracker


def test_tracker_sram_and_flash_budgets():
    tr = ResourceTracker.get()
    tr.reset()

    model = NanoStreamOD()
    model.eval()
    tr.set_flash(model.param_count())

    img = torch.randn(1, 160, 160)
    model.stream_forward(img, conf_thr=0.45)

    summary = tr.summary()

    # Budget specifications from PRD:
    # Peak SRAM < 256 KB
    # Flash < 512 KB
    assert summary["peak_sram_kb_mcu"] < 256.0, f"Peak SRAM exceeded: {summary['peak_sram_kb_mcu']} KB"
    assert summary["flash_kb"] < 512.0, f"Flash exceeded: {summary['flash_kb']} KB"
    assert summary["macs"] > 0
    assert summary["cycles"] > 0

    dashboard_txt = tr.dashboard(frame_idx=1, fps=35.0, mode="stream")
    assert "NanoStream-OD" in dashboard_txt
    assert "Peak SRAM" in dashboard_txt

"""Tests for C Header Model Weights Export."""

import pathlib
import pytest

from nanostream.data import calibration_images
from nanostream.export import export_c_header
from nanostream.fixedpoint import calibrate_fixed_point
from nanostream.model import NanoStreamOD


def test_export_c_header(tmp_path):
    model = NanoStreamOD()
    model.eval()

    calib_imgs = calibration_images(n=4, size=160)
    fracs = calibrate_fixed_point(model, calib_imgs, frac_bits=12, passes=1)

    out_file = tmp_path / "model_weights.h"
    export_c_header(model, fracs, out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")

    assert "#define NS_INPUT_SIZE" in content
    assert "#define NS_STAGES" in content
    assert "#define NS_GRID" in content
    assert "NS_SIG_LUT[512]" in content
    assert "ns_layers[" in content
    assert "ns_head1" in content
    assert "ns_head_obj" in content
    assert "ns_head_box" in content
    assert "ns_head_cls" in content


def test_c_kernel_compilation_and_execution(tmp_path):
    import shutil
    import subprocess

    gcc = shutil.which("gcc")
    if not gcc:
        pytest.skip("GCC not available on system")

    mcu_dir = pathlib.Path(__file__).parent.parent / "nanostream" / "mcu"
    c_src = mcu_dir / "nanostream_mcu.c"
    runner_src = mcu_dir / "mcu_test_runner.c"

    # FIX: compile the IN-TEST generated header (the old test used the
    # pre-committed model_weights.h, so export and C-run were never coupled).
    weights_h = tmp_path / "model_weights.h"
    model = NanoStreamOD()
    model.eval()
    calib_imgs = calibration_images(n=4, size=160)
    fracs = calibrate_fixed_point(model, calib_imgs, frac_bits=12, passes=1)
    export_c_header(model, fracs, weights_h)

    # The C source includes "model_weights.h" from its own directory; copy the
    # generated header there so the compile uses it.
    import shutil as _sh
    _sh.copy2(weights_h, mcu_dir / "model_weights.h")

    bin_path = tmp_path / "mcu_test.exe"
    cmd_compile = [
        gcc, "-O2", str(runner_src), str(c_src),
        f"-I{mcu_dir}", "-o", str(bin_path)
    ]
    res = subprocess.run(cmd_compile, capture_output=True, text=True)
    assert res.returncode == 0, f"GCC Compilation failed: {res.stderr}"

    cmd_run = [str(bin_path)]
    res_run = subprocess.run(cmd_run, capture_output=True, text=True)
    assert res_run.returncode == 0, f"C execution failed: {res_run.stderr}"
    assert "Inference finished successfully!" in res_run.stdout
    # FIX: the runner now prints the ACTUAL static BSS (computed from the
    # generated per-stage buffer sizes), not a hardcoded "<256 KB" string.
    assert "Static BSS:" in res_run.stdout
    assert "within 256 KB budget: YES" in res_run.stdout

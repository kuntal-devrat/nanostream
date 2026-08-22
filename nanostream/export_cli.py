"""CLI command for exporting trained NanoStream-OD weights to C header."""

import argparse
import pathlib
import torch

from .data import calibration_images
from .export import export_c_header
from .fixedpoint import calibrate_fixed_point
from .model import NanoStreamOD


def main():
    p = argparse.ArgumentParser(description="Export NanoStream-OD model to static C header (model_weights.h)")
    p.add_argument("--model", type=str, default="runs/shapes/nanostream_shapes.pt",
                   help="Path to trained PyTorch checkpoint (.pt)")
    p.add_argument("--out", type=str, default="nanostream/mcu/model_weights.h",
                   help="Output path for generated C header")
    p.add_argument("--calib-samples", type=int, default=8,
                   help="Number of synthetic/calibration samples for fixed-point shift calibration")
    p.add_argument("--frac-bits", type=int, default=12,
                   help="Input fractional bits (default 12 for Q12)")
    args = p.parse_args()

    model_path = pathlib.Path(args.model)
    if model_path.exists():
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            from .config import DEFAULT_CONFIG, NanoStreamConfig
            cfg_dict = ckpt["config"]
            cfg = NanoStreamConfig(**{k: v for k, v in cfg_dict.items() if hasattr(DEFAULT_CONFIG, k)})
            model = NanoStreamOD(cfg)
        else:
            model = NanoStreamOD()
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {model_path}")
    else:
        model = NanoStreamOD()
        print(f"Model checkpoint not found at {model_path}, using randomly initialized weights for export demo.")

    model.eval()
    print("Calibrating fixed-point dynamic ranges...")
    calib_imgs = calibration_images(n=args.calib_samples, size=model.cfg.input_size)
    calib_fracs = calibrate_fixed_point(model, calib_imgs, frac_bits=args.frac_bits, verbose=True)

    out_file = export_c_header(model, calib_fracs, args.out)
    print(f"Successfully exported C header -> {out_file.resolve()}")
    print("Zero dynamic allocation (no malloc/free). Ready for ARM Cortex-M / ESP32-S3 deployment.")


if __name__ == "__main__":
    main()

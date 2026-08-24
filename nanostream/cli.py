"""NanoStream-OD: Production CLI Interface.

Usage:
  nanostream info                  Print model architecture & MCU SRAM budget breakdown
  nanostream demo                  Run live interactive webcam/synthetic detector demo
  nanostream export                Export quantized fixed-point weights to C header or ONNX
  nanostream benchmark             Run streaming latency and memory benchmarks
"""

import argparse
import sys
from nanostream.config import NanoStreamConfig, PROFILES
from nanostream.export import stage_buffer_sizes


def cmd_info(args):
    print("=" * 65)
    print("        NanoStream-OD: Sub-256KB Streaming Object Detector")
    print("=" * 65)
    for profile_name in ["mcu", "pro", "gpu"]:
        cfg = PROFILES[profile_name](num_classes=3)
        sizes = stage_buffer_sizes(cfg)
        static_bss = sum(s["ring"] + s["win"] + s["cas"] for s in sizes) * 2
        print(f"\n[{profile_name.upper()} Tier]")
        print(f"  Input Resolution : {cfg.input_size}x{cfg.input_size} (1 channel)")
        print(f"  Strip Rows       : {cfg.strip_rows} rows/strip")
        print(f"  Channel Widths   : {cfg.stage_widths}")
        print(f"  Estimated SRAM   : {static_bss / 1024:.1f} KB ({'PASS <256KB' if static_bss <= 256*1024 else 'High-capacity'})")
        print(f"  Zero-NMS Decoding: Active (O(1) direct decode)")
    print("\n" + "=" * 65)


def cmd_demo(args):
    from nanostream.demo import main as demo_main
    demo_main()


def cmd_export(args):
    from nanostream.export_cli import main as export_main
    export_main()


def cmd_benchmark(args):
    from nanostream.benchmark import main as bench_main
    bench_main()


def main():
    parser = argparse.ArgumentParser(
        prog="nanostream",
        description="NanoStream-OD: Ultra-efficient streaming object detector for edge devices & MCUs",
    )
    sub = parser.add_subparsers(dest="command", help="Available subcommands")

    p_info = sub.add_parser("info", help="Print model profiles and SRAM memory budget")
    p_info.set_defaults(func=cmd_info)

    p_demo = sub.add_parser("demo", help="Run live detection demo")
    p_demo.set_defaults(func=cmd_demo)

    p_exp = sub.add_parser("export", help="Export model to C header (model_weights.h) or ONNX")
    p_exp.set_defaults(func=cmd_export)

    p_bm = sub.add_parser("benchmark", help="Run latency & memory tracker benchmarks")
    p_bm.set_defaults(func=cmd_benchmark)

    args, extra = parser.parse_known_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()

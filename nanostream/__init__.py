"""NanoStream-OD: Ultra-Efficient NMS-Free Patch-Streaming Object Detector.

Zero-NMS Detection | <256KB Peak SRAM | Multiplier-Free Bit-Shift Execution

Quickstart:
    import nanostream as ns

    # Load model
    model = ns.load_model()

    # Detect objects
    dets = ns.detect(model, "frame.jpg", conf_thr=0.20)
    for d in dets:
        print(f"Found {d.class_name} at {d.box} (score: {d.score:.2f})")

    # Export to bare-metal C header
    ns.export_to_c(model, "model_weights.h")
"""

from .api import Detection, detect, draw_detections, export_to_c, load_model, preprocess_image
from .config import DEFAULT_CONFIG, NanoStreamConfig
from .model import NanoStreamOD

__all__ = [
    "NanoStreamConfig",
    "DEFAULT_CONFIG",
    "NanoStreamOD",
    "Detection",
    "load_model",
    "detect",
    "draw_detections",
    "export_to_c",
    "preprocess_image",
]
__version__ = "0.1.0"

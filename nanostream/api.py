"""NanoStream-OD High-Level Python Framework API.

Enables developers to load models, run detection on images/videos,
and export to C headers in 3 lines of Python code:

    import nanostream as ns
    model = ns.load_model()
    detections = ns.detect(model, "photo.jpg")
"""

from dataclasses import dataclass
import pathlib
from typing import List, Tuple, Union
import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from .config import DEFAULT_CONFIG, NanoStreamConfig
from .data import CLASS_NAMES
from .export import export_c_header as _export_c_header
from .fixedpoint import calibrate_fixed_point
from .head import decode_detections
from .model import NanoStreamOD


@dataclass
class Detection:
    """Represents a single detected bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str

    @property
    def box(self) -> Tuple[float, float, float, float]:
        """(x1, y1, x2, y2) in normalized [0, 1] coordinates."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_pixel_box(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to integer pixel coordinates (px1, py1, px2, py2)."""
        return (
            int(self.x1 * img_w),
            int(self.y1 * img_h),
            int(self.x2 * img_w),
            int(self.y2 * img_h)
        )

    def __repr__(self):
        return f"<Detection {self.class_name} {self.score*100:.1f}% [{self.x1:.2f},{self.y1:.2f},{self.x2:.2f},{self.y2:.2f}]>"


def load_model(checkpoint_path: Union[str, pathlib.Path] = None,
               device: str = "cpu") -> NanoStreamOD:
    """Load a trained NanoStream-OD model from disk or create default.

    Args:
        checkpoint_path: Path to `.pt` checkpoint. If None, checks default face/shapes checkpoints.
        device: 'cpu' or 'cuda'.
    """
    if checkpoint_path is None:
        candidates = [
            pathlib.Path("runs/faces/nanostream_faces.pt"),
            pathlib.Path("runs/shapes/nanostream_shapes.pt"),
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = c
                break

    if checkpoint_path is not None and pathlib.Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "config" in ckpt:
            cfg_dict = ckpt["config"]
            valid_fields = set(NanoStreamConfig.__dataclass_fields__.keys())
            cfg = NanoStreamConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})
            model = NanoStreamOD(cfg)
        else:
            model = NanoStreamOD()
        try:
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state_dict)
        except Exception:
            model = NanoStreamOD(DEFAULT_CONFIG)
        if "classes" in ckpt:
            model.class_names = tuple(ckpt["classes"])
        else:
            model.class_names = CLASS_NAMES
    else:
        model = NanoStreamOD()
        model.class_names = CLASS_NAMES

    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Union[str, pathlib.Path, np.ndarray],
                     target_size: int = 160) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocess image (file path or numpy BGR/gray array) for NanoStream-OD.

    Returns:
        tensor: (1, 1, target_size, target_size) normalized to [-1, 1]
        orig_bgr: original BGR image for drawing
    """
    if isinstance(image, (str, pathlib.Path)):
        if cv2 is None:
            raise ImportError("opencv-python is required to load images from file path")
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image}")
    elif isinstance(image, np.ndarray):
        img_bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    gray_resized = cv2.resize(gray, (target_size, target_size))

    # Apply CLAHE for robust contrast handling
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_norm = clahe.apply(gray_resized)

    x = (torch.from_numpy(gray_norm).float() / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
    return x, img_bgr


def detect(model: NanoStreamOD,
           image: Union[str, pathlib.Path, np.ndarray],
           conf_thr: float = 0.20,
           use_streaming: bool = True) -> List[Detection]:
    """Run object detection on an image.

    Args:
        model: Loaded NanoStreamOD model.
        image: File path or numpy array (BGR or Grayscale).
        conf_thr: Detection confidence threshold (default: 0.20).
        use_streaming: If True, uses the microcontroller-matching strip streaming engine.

    Returns:
        List of Detection objects.
    """
    x, _ = preprocess_image(image, target_size=model.cfg.input_size)
    class_names = getattr(model, "class_names", CLASS_NAMES)
    device = next(model.parameters()).device
    x = x.to(device)

    if use_streaming:
        dets_t, _ = model.stream_forward(x.squeeze(0), conf_thr=conf_thr)
    else:
        with torch.no_grad():
            preds = model(x)
        dets_t = decode_detections(preds, conf_thr=conf_thr)

    results = []
    if dets_t.numel() > 0:
        for row in dets_t:
            x1, y1, x2, y2, score, cid = row.tolist()[:6]
            cid_int = int(cid)
            c_name = class_names[cid_int] if cid_int < len(class_names) else f"cls_{cid_int}"
            results.append(Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(score),
                class_id=cid_int,
                class_name=c_name
            ))
    return results


def draw_detections(image_bgr: np.ndarray,
                    detections: List[Detection],
                    color: Tuple[int, int, int] = (0, 230, 255)) -> np.ndarray:
    """Draw bounding boxes and confidence badges onto a BGR image."""
    if cv2 is None:
        raise ImportError("opencv-python required for draw_detections")

    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.to_pixel_box(w, h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"{det.class_name.upper()} {det.score*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, y1 - lh - 6)
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + lw + 8, tag_y1 + lh + 6), color, -1)
        cv2.putText(canvas, label, (x1 + 4, tag_y1 + lh + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

    return canvas


def export_to_c(model: NanoStreamOD,
                output_path: Union[str, pathlib.Path] = "nanostream/mcu/model_weights.h",
                calib_samples: int = 48) -> pathlib.Path:
    """Calibrate fixed-point shifts and export static C header for bare-metal MCU."""
    from .data import calibration_images
    calib_imgs = calibration_images(n=calib_samples, size=model.cfg.input_size)
    calib_fracs = calibrate_fixed_point(model, calib_imgs, frac_bits=model.cfg.frac_bits)
    out_file = _export_c_header(model, calib_fracs, str(output_path))
    return out_file

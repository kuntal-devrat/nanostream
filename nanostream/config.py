"""Configuration definition for NanoStream-OD v3.0.

All architectural dimensions, training options, and MCU deployment
parameters are defined here as a single source of truth.
"""

from dataclasses import dataclass


@dataclass
class NanoStreamConfig:
    # --- Input / Resolution ---
    input_size: int = 160
    in_channels: int = 1
    strip_rows: int = 16

    # --- Backbone widths (stem, stage1, stage2=P3, stage3=P4) ---
    stage_widths: tuple = (16, 24, 32, 48)

    # --- Detection head ---
    num_classes: int = 3
    head_hidden: int = 48

    # --- Backbone style ---
    # Note: v3.0 builds ShiftConv StageBlocks (no depthwise/SE); these fields
    # were removed to stop advertising an architecture that is not built.

    # --- Feature fusion (Lite-FPN neck) ---
    neck_type: str = "lite_fpn"      # "none" or "lite_fpn"

    # --- Dual-scale detection ---
    dual_scale: bool = True
    p3_loss_weight: float = 0.5      # Weight for P3 auxiliary loss

    # --- Quantization / MCU ---
    frac_bits: int = 12
    pure_shift: bool = True

    # --- Training augmentation ---
    augment_mosaic: bool = True
    augment_mixup: bool = True
    augment_cutout: bool = True
    multi_scale_train: bool = True   # Random resize 128-192 per step
    multi_scale_range: tuple = (128, 192)

    @property
    def context_dim(self) -> int:
        """Context dimension equals P4 channel width (no separate field needed)."""
        return self.stage_widths[-1]

    @property
    def grid_size(self) -> int:
        """P4 grid size (stride 16) = 10 for 160x160 input."""
        return self.input_size // 16

    @property
    def grid_size_p3(self) -> int:
        """P3 grid size (stride 8) = 20 for 160x160 input."""
        return self.input_size // 8

    @property
    def total_stride(self) -> int:
        return 16

    # ------------------------------------------------------------------
    # Deployment tiers. ``mcu`` is the shift-only <256 KB SRAM artifact;
    # ``pro`` (laptop) and ``gpu`` (server/industry) scale width + input
    # resolution and keep the full dual-scale + Lite-FPN pipeline. They are
    # float-only targets - the C export path is for ``mcu``.
    # ------------------------------------------------------------------
    @classmethod
    def mcu(cls, num_classes: int = 3) -> "NanoStreamConfig":
        return cls(num_classes=num_classes, input_size=160, strip_rows=16,
                   stage_widths=(16, 24, 32, 48), head_hidden=48,
                   neck_type="lite_fpn", dual_scale=True, p3_loss_weight=0.5,
                   augment_mosaic=True, augment_mixup=True, augment_cutout=True,
                   multi_scale_train=True, multi_scale_range=(128, 192))

    @classmethod
    def pro(cls, num_classes: int = 3) -> "NanoStreamConfig":
        return cls(num_classes=num_classes, input_size=256, strip_rows=16,
                   stage_widths=(24, 32, 48, 64), head_hidden=64,
                   neck_type="lite_fpn", dual_scale=True, p3_loss_weight=0.75,
                   augment_mosaic=True, augment_mixup=True, augment_cutout=True,
                   multi_scale_train=True, multi_scale_range=(224, 288))

    @classmethod
    def gpu(cls, num_classes: int = 3) -> "NanoStreamConfig":
        return cls(num_classes=num_classes, input_size=320, strip_rows=16,
                   stage_widths=(32, 48, 64, 96), head_hidden=96,
                   neck_type="lite_fpn", dual_scale=True, p3_loss_weight=1.0,
                   augment_mosaic=True, augment_mixup=True, augment_cutout=True,
                   multi_scale_train=True, multi_scale_range=(288, 352))


DEFAULT_CONFIG = NanoStreamConfig()

PROFILES = {
    "mcu": NanoStreamConfig.mcu,
    "pro": NanoStreamConfig.pro,
    "gpu": NanoStreamConfig.gpu,
}

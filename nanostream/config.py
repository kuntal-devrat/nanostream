"""Configuration definition for NanoStream-OD v3.0.

All architectural dimensions, training options, and MCU deployment
parameters are defined here as a single source of truth.
"""

from dataclasses import dataclass, field


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
    use_depthwise: bool = True       # Depthwise separable convolutions
    use_se: bool = True              # Squeeze-and-Excitation channel attention
    expansion_factor: int = 2        # Inverted residual expansion ratio

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


DEFAULT_CONFIG = NanoStreamConfig()

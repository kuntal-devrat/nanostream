"""Configuration definition for NanoStream-OD 2.0."""

from dataclasses import dataclass, field


@dataclass
class NanoStreamConfig:
    input_size: int = 160
    in_channels: int = 1
    stem_width: int = 16
    stage_widths: tuple = (16, 24, 32, 48)
    num_classes: int = 3
    strip_rows: int = 16
    context_dim: int = 32
    head_hidden: int = 48
    frac_bits: int = 12
    pure_shift: bool = True
    dual_scale: bool = True
    expansion_factor: int = 2

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

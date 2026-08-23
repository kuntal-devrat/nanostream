"""NanoStream-OD v3.0 Lite-FPN Neck: top-down multi-scale feature fusion.

Fuses high-level semantic features (P4, stride 16) back into high-resolution
spatial features (P3, stride 8) using only shift-add 1×1 convolutions.

This gives the P3 head access to semantic context (what the object IS) while
retaining fine spatial resolution (where the object IS) — critical for
detecting small/distant objects that FOMO and single-scale detectors miss.

Added cost: ~800 parameters, ~0.5M MACs — negligible vs the accuracy gain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NanoStreamConfig
from .layers import ShiftConv2d


class LiteFPN(nn.Module):
    """Lightweight Feature Pyramid Network for P3+P4 fusion.

    Architecture:
        P4 (10×10×C4) → Upsample 2× → 1×1 conv → 20×20×C3
        P3 (20×20×C3) + P4_up → element-wise add → 20×20×C3 [Fused P3]

    P4 is passed through unchanged. Only P3 gets enriched.
    """

    def __init__(self, cfg: NanoStreamConfig):
        super().__init__()
        c_p3 = cfg.stage_widths[-2]  # 32
        c_p4 = cfg.stage_widths[-1]  # 48

        # Lateral connection: project P4 channels down to P3 channel count
        self.lateral_p4 = ShiftConv2d(c_p4, c_p3, 1, stride=1, bias=False).name("fpn_lat")
        self.lateral_bn = nn.BatchNorm2d(c_p3)

        # Smoothing conv after fusion (prevents aliasing from upsampling)
        self.smooth_p3 = ShiftConv2d(c_p3, c_p3, 1, stride=1, bias=False).name("fpn_smooth")
        self.smooth_bn = nn.BatchNorm2d(c_p3)

    def frozen_pairs(self):
        return [(self.lateral_p4, self.lateral_bn),
                (self.smooth_p3, self.smooth_bn)]

    def forward(self, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Fuse P4 into P3 via top-down pathway.

        Args:
            p3: (B, C3, H3, W3) stride-8 features
            p4: (B, C4, H4, W4) stride-16 features

        Returns:
            p3_fused: (B, C3, H3, W3) enriched P3 features
        """
        # Project P4 to P3 channel count
        p4_lateral = F.relu(self.lateral_bn(self.lateral_p4(p4)))

        # Upsample P4 to P3 spatial size
        p4_up = F.interpolate(p4_lateral, size=p3.shape[2:],
                               mode='nearest')

        # Fuse: element-wise addition
        fused = p3 + p4_up

        # Smooth to reduce upsampling artifacts
        fused = F.relu(self.smooth_bn(self.smooth_p3(fused)))

        return fused

    @torch.no_grad()
    def forward_int(self, p3_int: torch.Tensor, p4_int: torch.Tensor,
                    p3_frac: int, p4_frac: int) -> tuple:
        """Integer forward for MCU parity.

        Args:
            p3_int: (B, C3, H3, W3) int32 P3 features
            p4_int: (B, C4, H4, W4) int32 P4 features
            p3_frac: Fractional bits of P3
            p4_frac: Fractional bits of P4

        Returns:
            fused_int: (B, C3, H3, W3) int32
            out_frac: Output fractional bits
        """
        # Lateral conv on P4
        lat = self.lateral_p4.forward_fixed_int(p4_int, p4_frac)
        lat = torch.relu(lat)
        lat_frac = p4_frac - self.lateral_p4.fixed_out_shift

        # Nearest-neighbor upsample (integer safe — just duplicate pixels)
        lat_up = F.interpolate(lat.float(), size=p3_int.shape[2:],
                                mode='nearest').to(torch.int32)

        # Align Q-formats for addition
        if lat_frac > p3_frac:
            lat_up = lat_up >> (lat_frac - p3_frac)
            out_frac = p3_frac
        elif p3_frac > lat_frac:
            p3_int = p3_int >> (p3_frac - lat_frac)
            out_frac = lat_frac
        else:
            out_frac = p3_frac

        fused = p3_int + lat_up

        # Smooth conv
        smooth = self.smooth_p3.forward_fixed_int(fused, out_frac)
        smooth = torch.relu(smooth)
        out_frac = out_frac - self.smooth_p3.fixed_out_shift

        return smooth, out_frac

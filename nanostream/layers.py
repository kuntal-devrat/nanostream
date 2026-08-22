"""ShiftConv2d & InvertedResidualBlock: convolutions whose weights are signed powers of two.

Frozen mode reconstructs weights as +/-2^e, so inference is pure shift-add.
`forward_fixed_int` implements exact integer semantics (arithmetic shifts,
int16 saturation) that mirror the generated MCU C kernel 1:1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tracker
from .quant import Pow2Weight, pow2_quantize


def _ishift(t: torch.Tensor, e: int) -> torch.Tensor:
    if e >= 0:
        return t << e
    return t >> (-e)


class ShiftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = self.kernel_size[0] // 2
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.pow2: Pow2Weight | None = None
        self.frozen_bias: torch.Tensor | None = None
        self.frozen = False
        self.fixed_in_frac = None
        self.fixed_out_shift = 0
        self._name_hint = "conv"

    def name(self, hint):
        self._name_hint = hint
        return self

    def freeze_pow2(self, bn=None, pure_shift=True):
        from .quant import fold_bn_into_conv
        w, b = fold_bn_into_conv(self.weight, bn, pure_shift=pure_shift) \
            if bn is not None else (self.weight.detach().float(),
                                    self.bias.detach().float().clone()
                                    if self.bias is not None
                                    else torch.zeros(self.out_channels))
        self.frozen_bias = b
        self.pow2 = pow2_quantize(w)
        self.frozen = True
        return self

    def unfreeze(self):
        self.frozen = False
        self.pow2 = None
        self.frozen_bias = None

    def effective_weight(self):
        if self.frozen:
            return self.pow2.to_float().to(self.weight.device)
        return self.weight

    def effective_bias(self):
        if self.frozen:
            return self.frozen_bias.to(self.weight.device)
        return None if self.bias is None else self.bias

    @torch.no_grad()
    def forward_fixed_int(self, x_int: torch.Tensor, in_frac: int) -> torch.Tensor:
        if x_int.dim() == 3:
            x_int = x_int.unsqueeze(0)
        B, C, H, W = x_int.shape
        Kh, Kw = self.kernel_size
        s = self.stride
        if H == Kh and s == 2:
            p_y, p_x = 0, self.padding
        else:
            p_y, p_x = self.padding, self.padding
        H_out = (H + 2 * p_y - Kh) // s + 1
        W_out = (W + 2 * p_x - Kw) // s + 1
        xp = F.pad(x_int.to(torch.int64), (p_x, p_x, p_y, p_y))
        E = self.pow2.exponent.to(torch.int32).to(x_int.device)
        S = self.pow2.sign.to(torch.int64).to(x_int.device)
        acc = torch.zeros(B, self.out_channels, H_out, W_out,
                          dtype=torch.int64, device=x_int.device)

        if self.groups == 1:
            for ky in range(Kh):
                for kx in range(Kw):
                    patch = xp[:, :, ky:ky + H_out * s:s, kx:kx + W_out * s:s]
                    for oc in range(self.out_channels):
                        eo = E[oc, :, ky, kx]
                        so = S[oc, :, ky, kx]
                        for ev in torch.unique(eo).tolist():
                            if ev <= -99:
                                continue
                            sel = eo == ev
                            signed = _ishift(patch[:, sel], int(ev)) * \
                                so[sel].view(1, -1, 1, 1)
                            acc[:, oc] += signed.sum(1)
        else:
            # Depthwise grouped convolution
            c_per_g = self.in_channels // self.groups
            out_per_g = self.out_channels // self.groups
            for g in range(self.groups):
                in_slice = slice(g * c_per_g, (g + 1) * c_per_g)
                out_slice = slice(g * out_per_g, (g + 1) * out_per_g)
                for ky in range(Kh):
                    for kx in range(Kw):
                        patch = xp[:, in_slice, ky:ky + H_out * s:s, kx:kx + W_out * s:s]
                        for oc_local in range(out_per_g):
                            oc = g * out_per_g + oc_local
                            eo = E[oc, :, ky, kx]
                            so = S[oc, :, ky, kx]
                            for ev in torch.unique(eo).tolist():
                                if ev <= -99:
                                    continue
                                sel = eo == ev
                                signed = _ishift(patch[:, sel], int(ev)) * \
                                    so[sel].view(1, -1, 1, 1)
                                acc[:, oc] += signed.sum(1)

        bq = (self.frozen_bias.to(x_int.device) * float(2.0 ** in_frac)) \
            .round().to(torch.int64).view(1, -1, 1, 1)
        acc = acc + bq
        self.calib_max_acc = max(getattr(self, "calib_max_acc", 0),
                                 int(acc.abs().max().item()))
        acc = acc >> self.fixed_out_shift
        acc = acc.clamp(-32768, 32767)
        return acc.to(torch.int32)

    def forward(self, x):
        w = self.effective_weight()
        b = self.effective_bias()
        if x.shape[-2] == self.kernel_size[0] and self.stride == 2:
            pad = (0, self.padding)
        else:
            pad = (self.padding, self.padding)
        y = F.conv2d(x, w, b, self.stride, pad, groups=self.groups)
        tr = tracker.ResourceTracker.get()
        if tr.enabled:
            taps = self.pow2.nonzero_taps if self.frozen else (
                (self.in_channels // self.groups) * self.out_channels *
                self.kernel_size[0] * self.kernel_size[1])
            tr.log_conv(y.numel() // max(1, y.shape[0]),
                        self.in_channels, self.out_channels,
                        self.kernel_size[0], taps)
        return y


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block: 1x1 Expand -> 3x3 Depthwise -> 1x1 Project + Residual Skip.

    Dramatically expands the network receptive field with minimal parameter count.
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        self.expand_ratio = expand_ratio

        layers = []
        if expand_ratio != 1:
            # 1x1 Pointwise Expansion
            self.expand_conv = ShiftConv2d(in_channels, hidden_dim, 1, bias=False).name("exp")
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
        else:
            self.expand_conv = None
            self.expand_bn = None

        # 3x3 Depthwise Convolution
        self.dw_conv = ShiftConv2d(hidden_dim, hidden_dim, 3, stride=stride,
                                   groups=hidden_dim, bias=False).name("dw")
        self.dw_bn = nn.BatchNorm2d(hidden_dim)

        # 1x1 Linear Pointwise Projection
        self.proj_conv = ShiftConv2d(hidden_dim, out_channels, 1, bias=False).name("proj")
        self.proj_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = x
        if self.expand_conv is not None:
            out = F.relu6(self.expand_bn(self.expand_conv(out)))
        out = F.relu6(self.dw_bn(self.dw_conv(out)))
        out = self.proj_bn(self.proj_conv(out))

        if self.use_res_connect:
            return identity + out
        return out

"""ShiftConv2d, DepthwiseSeparableConv, LiteSqueezeExcite, InvertedResidualBlock.

v3.0: All conv weights are signed powers of two — inference is pure shift-add.
`forward_fixed_int` implements exact integer semantics that mirror the C MCU kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tracker
from .quant import Pow2Weight, ZERO_EXP, pow2_quantize


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
                            if ev <= ZERO_EXP:
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
                                if ev <= ZERO_EXP:
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


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: DW 3×3 + PW 1×1.

    Uses ~9× fewer parameters and MACs than regular 3×3 convolution.
    All operations use ShiftConv2d for pure bit-shift execution on MCU.
    """

    def __init__(self, cin, cout, stride=1, hint="dws"):
        super().__init__()
        # 3×3 depthwise (groups=cin)
        self.dw = ShiftConv2d(cin, cin, 3, stride=stride,
                               groups=cin, bias=False).name(f"{hint}_dw")
        self.dw_bn = nn.BatchNorm2d(cin)
        # 1×1 pointwise
        self.pw = ShiftConv2d(cin, cout, 1, stride=1,
                               bias=False).name(f"{hint}_pw")
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

    def frozen_pairs(self):
        return [(self.dw, self.dw_bn), (self.pw, self.pw_bn)]

    @torch.no_grad()
    def forward_int(self, x_int, in_frac):
        y = self.dw.forward_fixed_int(x_int, in_frac)
        y = torch.relu(y)
        dw_frac = in_frac - self.dw.fixed_out_shift
        y = self.pw.forward_fixed_int(y, dw_frac)
        y = torch.relu(y)
        return y


class LiteSqueezeExcite(nn.Module):
    """Lightweight Squeeze-and-Excitation block using only 1×1 shifts.

    Channel attention: global avg pool → 1×1 reduce → ReLU → 1×1 expand → sigmoid
    Costs <1% extra params/MACs for ~2-5% mAP improvement.
    """

    def __init__(self, channels, reduction=4, hint="se"):
        super().__init__()
        mid = max(4, channels // reduction)
        self.squeeze = ShiftConv2d(channels, mid, 1, bias=True).name(f"{hint}_sq")
        self.excite = ShiftConv2d(mid, channels, 1, bias=True).name(f"{hint}_ex")

    def forward(self, x):
        # Global average pooling
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = F.relu(self.squeeze(scale))
        scale = torch.sigmoid(self.excite(scale))
        return x * scale

    def frozen_pairs(self):
        return [(self.squeeze, None), (self.excite, None)]


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block: 1×1 Expand → 3×3 DW → 1×1 Project + Skip.

    The core building block of MobileNetV2/V3 and EfficientNet.
    Uses ShiftConv2d throughout for pure bit-shift MCU execution.
    Optionally includes Squeeze-and-Excite channel attention.
    """

    def __init__(self, cin, cout, stride=1, expand_ratio=2,
                 use_se=False, hint="irb"):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.stride = stride
        self.use_res = (stride == 1 and cin == cout)
        hidden = int(round(cin * expand_ratio))

        layers = []

        # 1×1 Pointwise Expansion (skip if ratio == 1)
        if expand_ratio != 1:
            self.expand = ShiftConv2d(cin, hidden, 1, bias=False).name(f"{hint}_exp")
            self.expand_bn = nn.BatchNorm2d(hidden)
        else:
            self.expand = None
            self.expand_bn = None

        # 3×3 Depthwise
        self.dw = ShiftConv2d(hidden, hidden, 3, stride=stride,
                               groups=hidden, bias=False).name(f"{hint}_dw")
        self.dw_bn = nn.BatchNorm2d(hidden)

        # Squeeze-and-Excite (optional)
        if use_se:
            self.se = LiteSqueezeExcite(hidden, hint=f"{hint}_se")
        else:
            self.se = None

        # 1×1 Pointwise Projection (linear, no activation)
        self.proj = ShiftConv2d(hidden, cout, 1, bias=False).name(f"{hint}_proj")
        self.proj_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        identity = x
        out = x

        if self.expand is not None:
            out = F.relu(self.expand_bn(self.expand(out)))

        out = F.relu(self.dw_bn(self.dw(out)))

        if self.se is not None:
            out = self.se(out)

        out = self.proj_bn(self.proj(out))

        if self.use_res:
            return identity + out
        return out

    def frozen_pairs(self):
        pairs = []
        if self.expand is not None:
            pairs.append((self.expand, self.expand_bn))
        pairs.append((self.dw, self.dw_bn))
        if self.se is not None:
            pairs.extend(self.se.frozen_pairs())
        pairs.append((self.proj, self.proj_bn))
        return pairs

    @torch.no_grad()
    def forward_int(self, x_int, in_frac):
        identity = x_int
        out = x_int
        cur_frac = in_frac

        if self.expand is not None:
            out = self.expand.forward_fixed_int(out, cur_frac)
            out = torch.relu(out)
            cur_frac = cur_frac - self.expand.fixed_out_shift

        out = self.dw.forward_fixed_int(out, cur_frac)
        out = torch.relu(out)
        cur_frac = cur_frac - self.dw.fixed_out_shift

        # SE block skipped in integer mode (would need sigmoid LUT per channel)

        out = self.proj.forward_fixed_int(out, cur_frac)
        proj_frac = cur_frac - self.proj.fixed_out_shift

        if self.use_res:
            # Align Q-formats before addition (BUG-3 fix)
            if proj_frac < in_frac:
                identity = identity >> (in_frac - proj_frac)
            elif proj_frac > in_frac:
                out = out >> (proj_frac - in_frac)
            out = identity + out

        return out

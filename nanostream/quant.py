"""Power-of-two weight quantization: every multiply becomes a bit shift.

After folding BatchNorm (with its per-channel scale snapped to a power of
two), every effective weight is exactly +/-2^k. A convolution degenerates to
"shift the input by k, add/subtract" -- no multipliers required.
"""

import torch


ZERO_EXP = -99


class Pow2Weight:
    """Signed power-of-two weights: sign + combined integer exponent per tap."""

    def __init__(self, sign: torch.Tensor, exponent: torch.Tensor):
        self.sign = sign
        self.exponent = exponent

    @property
    def nonzero_taps(self) -> int:
        return int((self.exponent > ZERO_EXP).sum().item())

    def taps_per_out_channel(self):
        return (self.exponent > ZERO_EXP).flatten(1).sum(1)

    def to_float(self):
        e = self.exponent.to(torch.float32)
        mag = torch.pow(2.0, e.clamp(min=-30.0))
        wq = self.sign * torch.where(e <= float(ZERO_EXP), torch.zeros_like(mag), mag)
        return wq


def pow2_quantize(weight: torch.Tensor, zero_thr: float = 2.0 ** -14,
                  min_e: int = -22, max_e: int = 9) -> Pow2Weight:
    """Quantize weights so each tap equals sign(w)*2^e with integer e."""
    if weight.dim() == 4:
        reduce_dims = (1, 2, 3)
    else:
        reduce_dims = tuple(d for d in range(weight.dim()) if d != 0)
    amax = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
    wn = weight / amax
    mag = wn.abs()
    a_exp = torch.round(torch.log2(amax))
    e_rel = torch.round(torch.log2(mag.clamp_min(1e-16))).clamp(float(min_e), float(max_e))
    e_comb = (e_rel + a_exp).to(torch.int32)
    keep = mag >= zero_thr
    e_comb = torch.where(keep, e_comb, torch.full_like(e_comb, ZERO_EXP))
    sign = torch.where(weight >= 0,
                       torch.ones_like(weight),
                       -torch.ones_like(weight)).to(torch.float32)
    return Pow2Weight(sign=sign, exponent=e_comb)


def snap_to_pow2(scale: torch.Tensor, max_abs_exp: int = 4) -> torch.Tensor:
    e = torch.round(torch.log2(scale.clamp_min(1e-12))).clamp(-max_abs_exp, max_abs_exp)
    return torch.pow(2.0, e)


def fold_bn_into_conv(conv_weight: torch.Tensor, bn,
                      pure_shift: bool = True):
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    inv_std = torch.rsqrt(var + 1e-5)
    chan_scale = gamma * inv_std
    conv_weight = conv_weight.detach().float()
    if pure_shift:
        snapped = snap_to_pow2(chan_scale.abs()) * torch.sign(chan_scale + 1e-12)
        ratio = snapped / chan_scale.clamp(min=1e-12)
        conv_weight = conv_weight * ratio.view(-1, 1, 1, 1)
    else:
        conv_weight = conv_weight * chan_scale.view(-1, 1, 1, 1)
    b_fold = beta - gamma * mean * inv_std
    return conv_weight, b_fold


def quant_error(weight: torch.Tensor, pw: Pow2Weight) -> float:
    return float((weight.detach().float() - pw.to_float()).abs().mean().item())

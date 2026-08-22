"""Fixed-point runtime semantics shared 1:1 by the Python simulator and the
generated MCU C kernel: Q-format chains, sigmoid LUT, integer decoding,
and per-layer output-shift calibration."""

import math
import numpy as np
import torch

LUT_BITS = 7
LUT_SIZE = 1 << (16 - LUT_BITS)


def build_sig_lut() -> np.ndarray:
    """512-entry table: idx -> round(32767*sigmoid((idx*128 - 32768)/32768))."""
    i = np.arange(LUT_SIZE)
    x = (i.astype(np.float64) * 128.0 - 32768.0) / 32768.0
    return np.round(32767.0 / (1.0 + np.exp(-x))).astype(np.uint16)


_SIG_LUT = None


def sig_lut():
    global _SIG_LUT
    if _SIG_LUT is None:
        _SIG_LUT = build_sig_lut()
    return _SIG_LUT


def sig_lut_q(x_q15) -> torch.Tensor:
    """Python mirror of the C lookup: sigmoid(q15) -> probability q15."""
    lut = torch.from_numpy(sig_lut().astype(np.int64))
    idx = (x_q15.to(torch.int64) + 32768) >> LUT_BITS
    idx = idx.clamp(0, LUT_SIZE - 1)
    return lut[idx].to(torch.int32)


def quantize_input_u8(img_u8: torch.Tensor, frac_bits: int = 12):
    """u8 pixel -> normalized [-1,1]-ish fixed point, exact C-parity formula."""
    u = img_u8.to(torch.int64)
    q = (u * 8192 + 127) // 255 - (1 << frac_bits)
    return q.clamp(-32768, 32767).to(torch.int32)


def magic_recip(n: int, bits: int = 24):
    return round((1 << bits) / n), bits


def calibrate_fixed_point(model, images_u8, frac_bits: int = 12,
                          passes: int = 2, verbose=False):
    """Choose per-layer output right-shifts so int16 outputs never clip."""
    model.freeze_all()
    from .layers import ShiftConv2d
    convs = [m for m in model.modules() if isinstance(m, ShiftConv2d)]
    for c in convs:
        c.fixed_out_shift = 0
        c.calib_max_acc = 0

    def run_one(img):
        x = quantize_input_u8(img, frac_bits)
        if x.dim() == 2:
            feats = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            feats = x.unsqueeze(0)
        else:
            feats = x
        cur = frac_bits

        # Stem
        feats = model.backbone.stem.forward_fixed_int(feats, cur)
        feats = torch.relu(feats)
        cur = cur - model.backbone.stem.fixed_out_shift

        for blk in model.backbone.stages:
            feats = blk.forward_int(feats, cur)
            cur = cur - blk.conv.fixed_out_shift

        grid = feats[0]
        cnt = grid.shape[-1] * grid.shape[-2]
        ctx_sum = grid.sum(dim=(1, 2))
        model.head.forward_int(grid, ctx_sum.view(-1), cnt, cur)

    for p in range(passes):
        for c in convs:
            c.calib_max_acc = 0
        for img in images_u8:
            run_one(img)
        changed = False
        for c in convs:
            m = int(c.calib_max_acc)
            r = max(0, math.ceil(math.log2(max(m, 1) / 30000.0)))
            if r != c.fixed_out_shift:
                changed = True
                c.fixed_out_shift = r
        if verbose:
            print(f"calib pass {p}: " +
                  " ".join(f"{getattr(c, '_name_hint', 'c')}={c.fixed_out_shift}" for c in convs))
        if not changed and p > 0:
            break

    fracs = {
        "stem": frac_bits,
    }
    cur = frac_bits - model.backbone.stem.fixed_out_shift
    for blk in model.backbone.stages:
        fracs[blk.conv._name_hint] = cur
        cur -= blk.conv.fixed_out_shift

    head_fracs = {}
    hf = cur
    head_map = dict((getattr(m, "_name_hint", ""), m) for m in convs)
    for cname in ("head1", "head_obj", "head_box", "head_cls"):
        if cname in head_map:
            conv = head_map[cname]
            head_fracs[cname] = hf
            hf -= conv.fixed_out_shift
        else:
            head_fracs[cname] = hf

    return {"input_frac": frac_bits, "stage_in_frac": fracs,
            "head_in_frac": head_fracs}


@torch.no_grad()
def decode_int_detections(obj_q: torch.Tensor, box_q: torch.Tensor,
                          cls_q: torch.Tensor, out_frac: int,
                          G: int, conf_thr: float = 0.30):
    """Integer decoding for microcontroller: table lookups and integer math."""
    q15_scale = 15 - out_frac
    if q15_scale >= 0:
        obj_q15 = (obj_q.to(torch.int64) << q15_scale).clamp(-32768, 32767).to(torch.int32)
    else:
        obj_q15 = (obj_q.to(torch.int64) >> (-q15_scale)).clamp(-32768, 32767).to(torch.int32)

    prob_q15 = sig_lut_q(obj_q15)
    thr_q15 = int(round(conf_thr * 32767.0))
    mask = prob_q15 > thr_q15
    ys, xs = mask.nonzero(as_tuple=True)
    if ys.numel() == 0:
        return torch.zeros(0, 6, dtype=torch.float32)

    scores = prob_q15[ys, xs].float() / 32767.0

    bx_q = box_q[:, ys, xs]
    if q15_scale >= 0:
        bx_q15 = (bx_q.to(torch.int64) << q15_scale).clamp(-32768, 32767).to(torch.int32)
    else:
        bx_q15 = (bx_q.to(torch.int64) >> (-q15_scale)).clamp(-32768, 32767).to(torch.int32)

    cx_sig = sig_lut_q(bx_q15[0]).float() / 32767.0
    cy_sig = sig_lut_q(bx_q15[1]).float() / 32767.0
    w_sig = sig_lut_q(bx_q15[2]).float() / 32767.0
    h_sig = sig_lut_q(bx_q15[3]).float() / 32767.0

    cx = (xs.float() + cx_sig) / G
    cy = (ys.float() + cy_sig) / G
    bw = w_sig
    bh = h_sig

    cls_ids = torch.zeros(ys.shape[0], dtype=torch.float32)

    dets = torch.stack([
        (cx - bw / 2).clamp(0.0, 1.0),
        (cy - bh / 2).clamp(0.0, 1.0),
        (cx + bw / 2).clamp(0.0, 1.0),
        (cy + bh / 2).clamp(0.0, 1.0),
        scores,
        cls_ids,
    ], dim=1)
    return dets

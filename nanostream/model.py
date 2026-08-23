"""NanoStream-OD v3.0: patch-streaming backbone + Lite-FPN neck + zero-NMS dual-scale head.

v3.0 changes:
  - stream_forward now passes BOTH P3 and P4 features to head (BUG-1 fix)
  - Lite-FPN neck fuses P4 semantics into P3
  - Clean context computation
"""

import torch
import torch.nn as nn

from . import tracker
from .backbone import StreamingBackbone
from .config import DEFAULT_CONFIG, NanoStreamConfig
from .head import DualAssignHead, decode_detections
from .neck import LiteFPN


class NanoStreamOD(nn.Module):

    def __init__(self, cfg: NanoStreamConfig | None = None):
        super().__init__()
        self.cfg = cfg or DEFAULT_CONFIG
        self.backbone = StreamingBackbone(self.cfg)
        self.head = DualAssignHead(self.cfg)

        # Lite-FPN neck for multi-scale feature fusion
        neck_type = getattr(self.cfg, 'neck_type', 'lite_fpn')
        if neck_type == "lite_fpn" and getattr(self.cfg, 'dual_scale', True):
            self.neck = LiteFPN(self.cfg)
        else:
            self.neck = None

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone(x)
        p3 = feats.get("p3", None)
        p4 = feats["p4"]
        ctx = feats["ctx"]

        # Apply FPN neck to fuse P4 semantics into P3
        if self.neck is not None and p3 is not None:
            p3 = self.neck(p3, p4)

        preds = self.head({"p3": p3, "p4": p4}, ctx)
        preds["G"] = self.cfg.grid_size
        return preds

    @torch.no_grad()
    def stream_forward(self, img: torch.Tensor, conf_thr: float = 0.30):
        """Streaming execution: MCU-matching strip pipeline.

        BUG-1 FIX: Now captures and forwards P3 features to the head.
        """
        tr = tracker.ResourceTracker.get()
        tr.start_frame()

        was_training = self.training
        if was_training:
            self.eval()

        if getattr(self, "_streamer", None) is None:
            self._streamer = self.backbone.streamer()
        else:
            self._streamer.reset()

        device = next(self.parameters()).device
        img = img.to(device)

        if img.dim() == 2:
            img = img.unsqueeze(0)  # (1, H, W)
        elif img.dim() == 3 and img.shape[0] != 1 and img.shape[0] != self.cfg.in_channels:
            img = img.unsqueeze(0)

        H = img.shape[-2]
        strip_h = self.cfg.strip_rows
        for r in range(0, H, strip_h):
            strip = img[:, r:r + strip_h]
            self._streamer.feed_strip(strip)

        # BUG-1 FIX: finish() now returns (p4_grid, ctx_avg, p3_grid)
        grid, ctx_avg, p3_grid = self._streamer.finish()

        # Apply FPN neck in streaming mode
        feats = {"p4": grid}
        if self.neck is not None and p3_grid is not None:
            p3_fused = self.neck(p3_grid, grid)
            feats["p3"] = p3_fused
        elif p3_grid is not None:
            feats["p3"] = p3_grid

        preds = self.head(feats, ctx_avg.unsqueeze(0))
        preds["G"] = self.cfg.grid_size
        dets = decode_detections(preds, conf_thr)

        if was_training:
            self.train()

        return dets, preds

    @torch.no_grad()
    def stream_forward_int(self, img_u8: torch.Tensor, fracs: dict,
                           conf_thr: float = 0.30):
        """Bit-exact integer streaming path (mirrors the C MCU kernel)."""
        from .fixedpoint import decode_int_detections, quantize_input_u8
        tr = tracker.ResourceTracker.get()
        tr.start_frame()

        if getattr(self, "_int_streamer", None) is None:
            self._int_streamer = self.backbone.streamer()
            stage_fracs = [fracs["input_frac"]]
            for blk in self.backbone.stages:
                stage_fracs.append(fracs["stage_in_frac"][blk.conv._name_hint])
            self._int_streamer.set_int_mode(stage_fracs)
        else:
            self._int_streamer.reset()

        xq = quantize_input_u8(img_u8, fracs["input_frac"])
        if xq.dim() == 2:
            xq = xq.unsqueeze(0)

        H = xq.shape[-2]
        strip_h = self.cfg.strip_rows
        for r in range(0, H, strip_h):
            strip = xq[:, r:r + strip_h]
            self._int_streamer.feed_strip(strip)

        grid, ctx_avg, p3_grid = self._int_streamer.finish()
        cnt = self._int_streamer.ctx_n
        ctx_sum = self._int_streamer.ctx_acc

        head_frac = fracs["head_in_frac"].get("head1", fracs["head_in_frac"].get("head_p41", 12))
        obj_q, box_q, cls_q, out_frac = self.head.forward_int(
            grid[0], ctx_sum.view(-1).to(torch.int64), cnt, head_frac)

        dets = decode_int_detections(obj_q, box_q, cls_q, out_frac,
                                     self.cfg.grid_size, conf_thr)
        return dets

    @property
    def streamer(self):
        return getattr(self, "_streamer", None)

    @streamer.setter
    def streamer(self, v):
        self._streamer = v

    def unfreeze_all(self):
        for m in self.modules():
            if hasattr(m, "unfreeze"):
                m.unfreeze()

    def freeze_all(self, pure_shift: bool = True):
        """Quantize all weights to signed pow2 for integer/MCU execution."""
        for m in self.modules():
            pairs = getattr(m, "frozen_pairs", None)
            if callable(pairs):
                for conv, bn in pairs():
                    conv.freeze_pow2(bn=bn, pure_shift=pure_shift)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

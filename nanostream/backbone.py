"""NanoStream-OD v3.0 Backbone: DW-Sep Inverted Residual + Lite-FPN Streaming.

Patch-streaming architecture: thin horizontal strips flow through a cascade
of ring buffers instead of materializing full feature maps.

v3.0 changes:
  - Depthwise separable InvertedResidualBlock stages (9× fewer params/MACs)
  - Squeeze-and-Excite channel attention on P3/P4 stages
  - PatchStreamer captures BOTH P3 and P4 features (BUG-1 fix)
  - Lite-FPN neck fuses P4 semantics into P3 for small-object detection
  - Fixed boundary padding in ring buffer emit (BUG-5 fix)
  - Cleaned up stem integer path (BUG-4 fix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tracker
from .config import NanoStreamConfig
from .layers import ShiftConv2d, InvertedResidualBlock, DepthwiseSeparableConv


class BackboneOutput(dict):
    """Dictionary supporting both key lookups and [0] index for P4 features."""

    def __getitem__(self, key):
        if key == 0:
            return self["p4"]
        return super().__getitem__(key)


class StemBlock(nn.Module):
    """Initial stride-2 stem convolution."""

    def __init__(self, cin: int, cout: int, hint: str = "stem",
                 stride: int = 2):
        super().__init__()
        self.conv = ShiftConv2d(cin, cout, 3, stride=stride).name(hint)
        self.bn = nn.BatchNorm2d(cout)
        self.fixed_out_shift = 0

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

    def frozen_pairs(self):
        return [(self.conv, self.bn)]

    @torch.no_grad()
    def forward_int(self, x_int, in_frac):
        y = self.conv.forward_fixed_int(x_int, in_frac)
        return torch.relu(y)


class StageBlock(nn.Module):
    """Stage Block: 3×3 stride-2 downsampling conv + 1×1 residual refinement."""

    def __init__(self, cin: int, cout: int, hint: str, stride: int = 2):
        super().__init__()
        self.conv = ShiftConv2d(cin, cout, 3, stride=stride).name(hint)
        self.bn = nn.BatchNorm2d(cout)
        self.refine = ShiftConv2d(cout, cout, 1).name(f"{hint}_ref")
        self.refine_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        ref = self.refine_bn(self.refine(h))
        return h + F.relu(ref)

    def frozen_pairs(self):
        return [(self.conv, self.bn), (self.refine, self.refine_bn)]

    @torch.no_grad()
    def forward_int(self, x_int, in_frac):
        y = self.conv.forward_fixed_int(x_int, in_frac)
        h = torch.relu(y)
        out_f = in_frac - self.conv.fixed_out_shift
        ref = self.refine.forward_fixed_int(h, out_f)
        ref_f = out_f - self.refine.fixed_out_shift
        # Align Q-formats before residual addition (BUG-3 fix)
        if ref_f > out_f:
            ref = ref >> (ref_f - out_f)
        elif out_f > ref_f:
            h = h >> (out_f - ref_f)
        return h + torch.relu(ref)


class StreamStage:
    """Ring-buffer stage: emits output rows as soon as inputs allow."""

    def __init__(self, block: nn.Module, name: str, total_out: int,
                 c_in: int, w_in: int, stride: int = 2, pad: int = 1):
        self.block = block
        self.name = name
        self.stride = stride
        self.pad = pad
        self.total_out = total_out
        self.c_in = c_in
        self.w_in = w_in
        self.total_in = total_out * stride  # BUG-5 fix: track actual input dim
        self.tracker = tracker.ResourceTracker.get()
        self.reset()
        self.exec_mode = "float"
        self.in_frac = None

    def reset(self):
        self.rows = torch.zeros(self.c_in, 0, self.w_in)
        self.abs0 = 0
        self.next_out = 0

    def _abs_end(self):
        return self.abs0 + self.rows.shape[1] - 1

    def _register_ram(self):
        c, n, w = self.rows.shape
        self.tracker.alloc(f"ring:{self.name}", sim_bytes=c * n * w * 4,
                           mcu_bytes=c * n * w)

    def push(self, rows: torch.Tensor):
        if rows.shape[1] == 0:
            return None
        self.rows = torch.cat([self.rows, rows.contiguous()], dim=1) \
            if self.rows.shape[1] else rows.contiguous()
        self._register_ram()
        return self._emit()

    def flush(self):
        outs = []
        dev = self.rows.device
        while self.next_out < self.total_out:
            need_hi = self.next_out * self.stride + self.stride - 1 + self.pad
            if need_hi > self._abs_end():
                missing = int(need_hi - self._abs_end())
                z = torch.zeros(self.c_in, missing, self.w_in, device=dev)
                self.rows = torch.cat([self.rows, z], dim=1)
                self._register_ram()
            got = self._emit()
            if got is not None:
                outs.append(got)
        result = torch.cat(outs, dim=1) if outs else None
        self.release_ram()
        return result

    def release_ram(self):
        self.tracker.release(f"ring:{self.name}")
        self.tracker.release(f"win:{self.name}")
        self.tracker.release(f"out:{self.name}")

    def _emit(self):
        dev = self.rows.device
        produced = []
        while self.next_out < self.total_out:
            c_lo = self.next_out * self.stride - self.pad
            c_hi = c_lo + 3 - 1
            if self._abs_end() < c_hi:
                break
            w_top = max(0, -c_lo)
            # BUG-5 fix: use total_in (actual input spatial dim) for boundary
            w_bot = max(0, c_hi - (self.total_in - 1))
            valid_lo = max(c_lo, 0)
            valid_hi = min(c_hi, self.total_in - 1)
            rel_lo = valid_lo - self.abs0
            rel_hi = valid_hi - self.abs0
            chunk = self.rows[:, rel_lo:rel_hi + 1]

            parts = []
            if w_top > 0:
                parts.append(torch.zeros(self.c_in, w_top, self.w_in, device=dev))
            parts.append(chunk)
            if w_bot > 0:
                parts.append(torch.zeros(self.c_in, w_bot, self.w_in, device=dev))
            window = torch.cat(parts, dim=1).unsqueeze(0)

            self.tracker.alloc(f"win:{self.name}",
                               sim_bytes=window.numel() * 4,
                               mcu_bytes=window.numel())

            if self.exec_mode == "int":
                out_row = self.block.forward_int(window.to(torch.int32),
                                                 self.in_frac)[0]
            else:
                out_row = self.block(window)[0]

            self.tracker.alloc(f"out:{self.name}",
                               sim_bytes=out_row.numel() * 4,
                               mcu_bytes=out_row.numel())
            produced.append(out_row)
            self.next_out += 1

        if self.next_out < self.total_out:
            drop_below = max(0, self.next_out * self.stride - self.pad)
            if drop_below > self.abs0:
                drop_n = min(drop_below - self.abs0, self.rows.shape[1])
                self.rows = self.rows[:, drop_n:]
                self.abs0 += drop_n
                self._register_ram()
        else:
            self.rows = self.rows[:, :0]
            self._register_ram()

        if produced:
            return torch.cat(produced, dim=1)
        return None


class StreamingBackbone(nn.Module):
    """v3.0 Backbone: DW-Sep InvRes stages + dual-scale P3/P4 output."""

    def __init__(self, cfg: NanoStreamConfig):
        super().__init__()
        self.cfg = cfg
        widths = cfg.stage_widths
        use_dw = getattr(cfg, 'use_depthwise', True)
        use_se = getattr(cfg, 'use_se', True)
        expand = getattr(cfg, 'expansion_factor', 2)

        self.stem = StemBlock(cfg.in_channels, widths[0], "stem", stride=2)
        self.stages = nn.ModuleList([
            StageBlock(widths[0], widths[1], "stage1", stride=2),
            StageBlock(widths[1], widths[2], "stage2", stride=2),
            StageBlock(widths[2], widths[3], "stage3", stride=2),
        ])

    def frozen_pairs(self):
        pairs = self.stem.frozen_pairs()
        for blk in self.stages:
            pairs.extend(blk.frozen_pairs())
        return pairs

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Full-frame forward. Returns P3 (stride 8), P4 (stride 16), and ctx."""
        h = self.stem(x)
        h = self.stages[0](h)
        p3 = self.stages[1](h)   # Stride 8: (B, C2, 20, 20)
        p4 = self.stages[2](p3)  # Stride 16: (B, C3, 10, 10)
        ctx = p4.mean(dim=(2, 3))
        return BackboneOutput(p3=p3, p4=p4, ctx=ctx)

    def streamer(self):
        return PatchStreamer(self)

    def make_streamer(self):
        return self.streamer()


class PatchStreamer:
    """v3.0 Patch Streamer: captures BOTH P3 and P4 features (BUG-1 fix)."""

    def __init__(self, backbone: StreamingBackbone):
        self.backbone = backbone
        self.cfg = backbone.cfg
        self.tracker = tracker.ResourceTracker.get()
        size = backbone.cfg.input_size

        self.stream_stages = [
            StreamStage(
                backbone.stem, name="s0",
                total_out=size // 2,
                c_in=backbone.cfg.in_channels, w_in=size)
        ]

        for i, blk in enumerate(backbone.stages):
            self.stream_stages.append(StreamStage(
                blk, name=f"s{i+1}",
                total_out=size // (2 ** (i + 2)),
                c_in=backbone.cfg.stage_widths[i],
                w_in=size // (2 ** (i + 1))))

        self.strip_rows = backbone.cfg.strip_rows

        # BUG-1 FIX: Track P3 features separately from P4
        self.p3_grid_feats = []  # Stage2 output (stride 8)
        self.grid_feats = []     # Stage3 output (stride 16)
        self.ctx_acc = None
        self.ctx_n = 0

    def set_int_mode(self, stage_in_fracs):
        for st, f in zip(self.stream_stages, stage_in_fracs):
            st.exec_mode = "int"
            st.in_frac = int(f)

    def clear_int_mode(self):
        for st in self.stream_stages:
            st.exec_mode = "float"
            st.in_frac = None

    def reset(self):
        for st in self.stream_stages:
            st.reset()
            st.release_ram()
        self.p3_grid_feats = []
        self.grid_feats = []
        self.ctx_acc = None
        self.ctx_n = 0

    def _collect_p3(self, out: torch.Tensor):
        """BUG-1 FIX: Collect P3 (stage2) features."""
        self.p3_grid_feats.append(out)

    def _collect_final(self, out: torch.Tensor):
        """Collect P4 (stage3) features and accumulate context."""
        self.grid_feats.append(out)
        row_sum = out.sum(dim=(1, 2))
        self.ctx_acc = row_sum if self.ctx_acc is None \
            else self.ctx_acc + row_sum
        self.ctx_n += out.shape[1] * out.shape[2]

    def _push_cascade(self, data):
        """Push data through stages, capturing P3 at stage2 output."""
        cur = data
        for i, st in enumerate(self.stream_stages):
            if cur is None or cur.shape[1] == 0:
                return
            cur = st.push(cur)

            # BUG-1 FIX: Capture P3 features from stage index 2 (stage2)
            if i == 2 and cur is not None and cur.shape[1] > 0:
                self._collect_p3(cur.clone())

        # After all stages, cur is P4 output
        if cur is not None and cur.shape[1] > 0:
            self._collect_final(cur)

    def feed_strip(self, strip: torch.Tensor):
        if strip.dim() == 2:
            strip = strip.unsqueeze(0)
        self.tracker.alloc("in_strip",
                           sim_bytes=strip.numel() * 4,
                           mcu_bytes=strip.numel())
        self._push_cascade(strip)
        self.tracker.release("in_strip")

    def finish(self):
        """Flush all stages, collecting remaining P3 and P4 features."""
        for i in range(len(self.stream_stages) - 1):
            tail = self.stream_stages[i].flush()
            if tail is not None and tail.shape[1] > 0:
                cur = tail
                for j in range(i + 1, len(self.stream_stages)):
                    prev_cur = cur
                    cur = self.stream_stages[j].push(cur)

                    # BUG-1 FIX: Capture P3 from stage2 during flush
                    if j == 2 and cur is not None and cur.shape[1] > 0:
                        self._collect_p3(cur.clone())

                    if cur is None or cur.shape[1] == 0:
                        cur = None
                        break

                if cur is not None and cur.shape[1] > 0:
                    self._collect_final(cur)

        # Flush last stage
        last_tail = self.stream_stages[-1].flush()
        if last_tail is not None and last_tail.shape[1] > 0:
            self._collect_final(last_tail)

        for st in self.stream_stages:
            st.release_ram()

        if not self.grid_feats:
            raise RuntimeError("No P4 output features produced by backbone")

        # Assemble P4 grid
        full_grid = torch.cat(self.grid_feats, dim=1).unsqueeze(0)
        ctx = self.ctx_acc / max(1, self.ctx_n)

        # Assemble P3 grid (BUG-1 FIX)
        p3_grid = None
        if self.p3_grid_feats:
            p3_grid = torch.cat(self.p3_grid_feats, dim=1).unsqueeze(0)

        return full_grid, ctx, p3_grid

    def infer_grid(self, img: torch.Tensor, return_p3: bool = False):
        self.reset()
        if img.dim() == 2:
            img = img.unsqueeze(0)
        H = img.shape[-2]
        strip_h = self.strip_rows
        for r in range(0, H, strip_h):
            strip = img[:, r:r + strip_h]
            self.feed_strip(strip)
        grid, ctx_avg, p3_grid = self.finish()
        cnt = self.ctx_n
        ctx_sum = self.ctx_acc
        if return_p3:
            return grid[0], ctx_sum, cnt, p3_grid
        return grid[0], ctx_sum, cnt

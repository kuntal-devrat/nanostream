"""NanoStream-OD v3.0 Detection Head: Zero-NMS Dual-Scale with CIoU + VariFocal.

v3.0 changes:
  - CIoU loss replaces GIoU for tighter box regression (head.py)
  - VariFocal Loss for quality-aware objectness
  - P3 loss now includes L1 + classification (BUG-10 fix)
  - cin_p4 uses explicit addition not multiplication (BUG-2 fix)
  - Cross-scale duplicate suppression between P3 and P4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ShiftConv2d
from .losses import ciou_loss, varifocal_loss


class ScaleHead(nn.Module):
    """Detection head branch for a single scale level."""

    def __init__(self, cin: int, hid: int, num_classes: int, name_prefix: str = "head"):
        super().__init__()
        self.conv1 = ShiftConv2d(cin, hid, 1).name(f"{name_prefix}1")
        self.bn1 = nn.BatchNorm2d(hid)
        self.obj = ShiftConv2d(hid, 1, 1).name(f"{name_prefix}_obj")
        self.box = ShiftConv2d(hid, 4, 1).name(f"{name_prefix}_box")
        self.cls = ShiftConv2d(hid, num_classes, 1).name(f"{name_prefix}_cls")

    def frozen_pairs(self):
        return [(self.conv1, self.bn1), (self.obj, None),
                (self.box, None), (self.cls, None)]

    def forward(self, x: torch.Tensor):
        h = F.relu(self.bn1(self.conv1(x)))
        return {
            "obj": self.obj(h),
            "box": self.box(h),
            "cls": self.cls(h),
        }

    @torch.no_grad()
    def forward_int(self, x: torch.Tensor, in_frac: int):
        h = self.conv1.forward_fixed_int(x, in_frac)
        h = torch.relu(h)
        out_frac = in_frac - self.conv1.fixed_out_shift
        obj_q = self.obj.forward_fixed_int(h, out_frac)
        box_q = self.box.forward_fixed_int(h, out_frac)
        cls_q = self.cls.forward_fixed_int(h, out_frac)
        return obj_q, box_q, cls_q, out_frac


class DualAssignHead(nn.Module):
    """v3.0 Multi-Scale Zero-NMS Head (P4: 10×10, P3: 20×20)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # BUG-2 FIX: Explicit addition, not multiplication
        cin_p4 = cfg.stage_widths[-1] + cfg.context_dim
        cin_p3 = cfg.stage_widths[-2]
        hid = cfg.head_hidden

        # P4 Head (Stride 16, 10×10)
        self.head_p4 = ScaleHead(cin_p4, hid, cfg.num_classes, name_prefix="head")

        # P3 Head (Stride 8, 20×20) for small/distant objects
        self.dual_scale = getattr(cfg, "dual_scale", True)
        if self.dual_scale:
            self.head_p3 = ScaleHead(cin_p3, hid // 2, cfg.num_classes, name_prefix="head_p3")
        else:
            self.head_p3 = None

        self.G = cfg.grid_size

    def frozen_pairs(self):
        pairs = self.head_p4.frozen_pairs()
        if self.head_p3 is not None:
            pairs.extend(self.head_p3.frozen_pairs())
        return pairs

    def forward(self, feats, ctx: torch.Tensor):
        if isinstance(feats, dict):
            feat_p4 = feats["p4"]
            feat_p3 = feats.get("p3", None)
        else:
            feat_p4 = feats
            feat_p3 = None

        B, _, G = feat_p4.shape[0], feat_p4.shape[1], feat_p4.shape[-1]
        ctx_map = ctx.view(ctx.shape[0], -1, 1, 1).expand(-1, -1, G, G)
        x_p4 = torch.cat([feat_p4, ctx_map], dim=1)
        out_p4 = self.head_p4(x_p4)

        result = {
            "obj": out_p4["obj"],
            "box": out_p4["box"],
            "cls": out_p4["cls"],
            "G": G,
        }

        if self.dual_scale and self.head_p3 is not None and feat_p3 is not None:
            out_p3 = self.head_p3(feat_p3)
            result["p3_obj"] = out_p3["obj"]
            result["p3_box"] = out_p3["box"]
            result["p3_cls"] = out_p3["cls"]
            result["G_p3"] = feat_p3.shape[-1]

        return result

    @torch.no_grad()
    def forward_int(self, grid_int: torch.Tensor, ctx_sum_int: torch.Tensor,
                    ctx_count: int, in_frac: int):
        G = grid_int.shape[-1]
        m, s = magic_reciprocal(ctx_count)
        ctx_q = ((ctx_sum_int.view(1, -1).to(torch.int64) * m) >> s)
        ctx_q = ctx_q.clamp(-(2 ** 30), 2 ** 30 - 1).to(torch.int32)
        ctx_map = ctx_q.view(1, -1, 1, 1).expand(-1, -1, G, G)
        x = torch.cat([grid_int.unsqueeze(0), ctx_map], dim=1)
        obj_q, box_q, cls_q, out_frac = self.head_p4.forward_int(x, in_frac)
        return obj_q[0, 0], box_q[0], cls_q[0], out_frac


def magic_reciprocal(n: int, prec_bits: int = 20):
    """Return (m, sh) with m/2^sh ~= 1/n."""
    sh = prec_bits
    m = round((1 << sh) / n)
    return m, sh


def decode_cells(box_reg: torch.Tensor, G: int):
    """Decode all G*G cells to absolute cxcywh boxes in [0,1] coords."""
    device = box_reg.device
    xs = torch.arange(G, device=device, dtype=torch.float32).view(1, G).expand(G, G)
    ys = torch.arange(G, device=device, dtype=torch.float32).view(G, 1).expand(G, G)
    cx = (xs + torch.sigmoid(box_reg[0])) / G
    cy = (ys + torch.sigmoid(box_reg[1])) / G
    w = (2.5 * torch.sigmoid(box_reg[2])).clamp(max=1.0)
    h = (2.5 * torch.sigmoid(box_reg[3])).clamp(max=1.0)
    return torch.stack([cx, cy, w, h], dim=-1).view(-1, 4)


def cxcywh_to_xyxy(b):
    return torch.stack([b[..., 0] - b[..., 2] / 2,
                        b[..., 1] - b[..., 3] / 2,
                        b[..., 0] + b[..., 2] / 2,
                        b[..., 1] + b[..., 3] / 2], dim=-1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


# Keep GIoU as fallback
def generalized_iou_loss(pred_xyxy, tgt_xyxy):
    """Element-wise GIoU loss between matched pred-target pairs."""
    if pred_xyxy.numel() == 0:
        return pred_xyxy.sum() * 0.0
    lt = torch.maximum(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    rb = torch.minimum(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area_t = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]) * (tgt_xyxy[:, 3] - tgt_xyxy[:, 1])
    union = (area_p + area_t - inter).clamp(min=1e-9)
    iou = inter / union

    lt_c = torch.minimum(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    rb_c = torch.maximum(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    c_area = (wh_c[:, 0] * wh_c[:, 1]).clamp(min=1e-9)
    giou = iou - (c_area - union) / c_area
    return (1 - giou).mean()


@torch.no_grad()
def assign_dual(box_reg: torch.Tensor, gt_boxes: torch.Tensor, G: int):
    """One primary + one secondary cell per GT."""
    device = box_reg.device
    n_cells = G * G
    all_cells = decode_cells(box_reg, G)
    centers = all_cells[:, :2]
    corners = cxcywh_to_xyxy(all_cells)

    prim_idx, prim_gt, sec_idx, sec_gt = [], [], [], []
    used = set()
    for gi in range(gt_boxes.shape[0]):
        gcx, gcy, gw, gh = gt_boxes[gi].tolist()
        inside = ((centers[:, 0] - gcx).abs() < gw / 2) & \
                 ((centers[:, 1] - gcy).abs() < gh / 2)
        if not bool(inside.any()):
            d = (centers - torch.tensor([gcx, gcy], device=device)).norm(dim=1)
            if used:
                d[torch.tensor(sorted(used), dtype=torch.long, device=device)] = float("inf")
            ranked = [(int(d.argmin()), 0.0)]
        else:
            gt_xyxy = cxcywh_to_xyxy(gt_boxes[gi:gi + 1])
            ious = pairwise_iou(corners[inside], gt_xyxy)[0]
            order = torch.argsort(ious, descending=True)
            cell_ids = inside.nonzero(as_tuple=True)[0][order]
            ranked = [(int(c), float(ious[o]))
                      for o, c in zip(order.tolist(), cell_ids.tolist())]

        picked = 0
        for c, sc in ranked:
            if c in used:
                continue
            if picked == 0:
                prim_idx.append(c)
                prim_gt.append(gi)
                used.add(c)
                picked += 1
            elif picked == 1:
                sec_idx.append(c)
                sec_gt.append(gi)
                used.add(c)
                picked += 1
            else:
                break

    return {
        "prim_idx": torch.tensor(prim_idx or [], dtype=torch.long, device=device),
        "prim_gt": torch.tensor(prim_gt or [], dtype=torch.long, device=device),
        "sec_idx": torch.tensor(sec_idx or [], dtype=torch.long, device=device),
        "sec_gt": torch.tensor(sec_gt or [], dtype=torch.long, device=device),
        "num_cells": n_cells, "G": G,
    }


def detection_loss(preds: dict, targets: list, cfg=None, device=None,
                   w_obj: float = 2.0, w_box: float = 4.0, w_l1: float = 1.0, w_cls: float = 0.5):
    """v3.0 Detection loss with CIoU + VariFocal + full P3 loss (BUG-10 fix)."""
    obj_preds = preds["obj"]
    box_preds = preds["box"]
    cls_preds = preds["cls"]
    G = preds["G"]
    B = obj_preds.shape[0]
    dev = obj_preds.device

    total_loss_obj = torch.tensor(0.0, device=dev)
    total_loss_box = torch.tensor(0.0, device=dev)
    total_loss_l1 = torch.tensor(0.0, device=dev)
    total_loss_cls = torch.tensor(0.0, device=dev)
    total_num_pos = 0.0

    for b in range(B):
        obj_logit = obj_preds[b, 0].reshape(-1)
        box_reg = box_preds[b]
        cls_logit = cls_preds[b]
        t = targets[b]
        gt_boxes = t["boxes_norm"].to(dev) if len(t["boxes_norm"]) else torch.zeros(0, 4, device=dev)
        gt_labels = t["labels"].to(dev) if len(t["labels"]) else torch.zeros(0, dtype=torch.long, device=dev)

        assign = assign_dual(box_reg, gt_boxes, G)

        # VariFocal Loss for objectness (quality-aware)
        tgt_obj = torch.zeros_like(obj_logit)
        if len(assign["prim_idx"]):
            tgt_obj[assign["prim_idx"]] = 1.0
        if len(assign["sec_idx"]):
            tgt_obj[assign["sec_idx"]] = 0.25

        loss_obj = varifocal_loss(obj_logit, tgt_obj, alpha=0.75, gamma=2.0)
        total_loss_obj = total_loss_obj + loss_obj

        if len(assign["prim_idx"]):
            pred_all = decode_cells(box_reg, G)
            for idx_key, gt_key, w in (("prim_idx", "prim_gt", 1.0),
                                       ("sec_idx", "sec_gt", 0.20)):
                idx = assign[idx_key]
                gsel = assign[gt_key]
                if len(idx) == 0:
                    continue
                p_xyxy = cxcywh_to_xyxy(pred_all[idx])
                t_xyxy = cxcywh_to_xyxy(gt_boxes[gsel])
                # CIoU loss (replaces GIoU)
                total_loss_box = total_loss_box + w * ciou_loss(p_xyxy, t_xyxy)
                total_loss_l1 = total_loss_l1 + w * F.l1_loss(
                    p_xyxy.clamp(0, 1), t_xyxy.clamp(0, 1))

            labels = gt_labels[assign["prim_gt"]]
            K = cls_logit.shape[0]
            cls_sel = cls_logit.view(K, -1).t()[assign["prim_idx"]]
            smooth = torch.full_like(cls_sel, 0.02)
            smooth.scatter_(1, labels.view(-1, 1), 0.98)
            total_loss_cls = total_loss_cls + F.binary_cross_entropy_with_logits(cls_sel, smooth)
            total_num_pos += float(len(assign["prim_idx"]))

    total_loss_obj = total_loss_obj / B
    total_loss_box = total_loss_box / B
    total_loss_l1 = total_loss_l1 / B
    total_loss_cls = total_loss_cls / B

    # BUG-10 FIX: P3 loss now includes L1 + classification (not just obj + box)
    p3_loss_weight = getattr(cfg, 'p3_loss_weight', 0.5) if cfg else 0.5
    if "p3_obj" in preds and "p3_box" in preds:
        p3_obj_preds = preds["p3_obj"]
        p3_box_preds = preds["p3_box"]
        p3_cls_preds = preds.get("p3_cls", None)
        G_p3 = preds["G_p3"]
        loss_p3_obj = torch.tensor(0.0, device=dev)
        loss_p3_box = torch.tensor(0.0, device=dev)
        loss_p3_l1 = torch.tensor(0.0, device=dev)
        loss_p3_cls = torch.tensor(0.0, device=dev)

        for b in range(B):
            obj_l = p3_obj_preds[b, 0].reshape(-1)
            b_reg = p3_box_preds[b]
            t = targets[b]
            gt_b = t["boxes_norm"].to(dev) if len(t["boxes_norm"]) else torch.zeros(0, 4, device=dev)
            gt_lbl = t["labels"].to(dev) if len(t["labels"]) else torch.zeros(0, dtype=torch.long, device=dev)

            assign_p3 = assign_dual(b_reg, gt_b, G_p3)
            tgt_p3 = torch.zeros_like(obj_l)
            if len(assign_p3["prim_idx"]):
                tgt_p3[assign_p3["prim_idx"]] = 1.0

            loss_p3_obj = loss_p3_obj + varifocal_loss(obj_l, tgt_p3)

            if len(assign_p3["prim_idx"]):
                p_all = decode_cells(b_reg, G_p3)
                p_xyxy = cxcywh_to_xyxy(p_all[assign_p3["prim_idx"]])
                t_xyxy = cxcywh_to_xyxy(gt_b[assign_p3["prim_gt"]])
                loss_p3_box = loss_p3_box + ciou_loss(p_xyxy, t_xyxy)
                loss_p3_l1 = loss_p3_l1 + F.l1_loss(
                    p_xyxy.clamp(0, 1), t_xyxy.clamp(0, 1))

                # BUG-10 FIX: Add classification loss for P3
                if p3_cls_preds is not None:
                    p3_cls_logit = p3_cls_preds[b]
                    p3_labels = gt_lbl[assign_p3["prim_gt"]]
                    K_p3 = p3_cls_logit.shape[0]
                    p3_cls_sel = p3_cls_logit.view(K_p3, -1).t()[assign_p3["prim_idx"]]
                    p3_smooth = torch.full_like(p3_cls_sel, 0.02)
                    p3_smooth.scatter_(1, p3_labels.view(-1, 1), 0.98)
                    loss_p3_cls = loss_p3_cls + F.binary_cross_entropy_with_logits(
                        p3_cls_sel, p3_smooth)

        total_loss_obj = total_loss_obj + p3_loss_weight * (loss_p3_obj / B)
        total_loss_box = total_loss_box + p3_loss_weight * (loss_p3_box / B)
        total_loss_l1 = total_loss_l1 + p3_loss_weight * (loss_p3_l1 / B)
        total_loss_cls = total_loss_cls + p3_loss_weight * (loss_p3_cls / B)

    total = w_obj * total_loss_obj + w_box * total_loss_box + w_l1 * total_loss_l1 + w_cls * total_loss_cls
    return {
        "total": total,
        "obj": total_loss_obj.detach(),
        "box": total_loss_box.detach(),
        "l1": total_loss_l1.detach(),
        "cls": total_loss_cls.detach(),
        "num_pos": total_num_pos / B
    }


@torch.no_grad()
def decode_detections(preds: dict, conf_thr: float = 0.30, max_det: int = 16):
    """Zero-NMS multi-scale decode with 3×3 local peak + cross-scale dedup."""
    all_dets = []

    # 1. Decode P4 (10×10)
    obj = torch.sigmoid(preds["obj"])[0, 0]
    obj_max = F.max_pool2d(obj.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
    mask = (obj == obj_max) & (obj > conf_thr)
    ys, xs = mask.nonzero(as_tuple=True)
    if ys.numel() > 0:
        scores = obj[ys, xs]
        G = obj.shape[-1]
        br = preds["box"][0]
        cx = (xs.float() + torch.sigmoid(br[0][ys, xs])) / G
        cy = (ys.float() + torch.sigmoid(br[1][ys, xs])) / G
        bw = (2.5 * torch.sigmoid(br[2][ys, xs])).clamp(max=1.0)
        bh = (2.5 * torch.sigmoid(br[3][ys, xs])).clamp(max=1.0)
        cls_scores = torch.sigmoid(preds["cls"][0])
        cls_ids = cls_scores[:, ys, xs].argmax(dim=0)
        p4_dets = torch.stack([
            (cx - bw / 2).clamp(0.0, 1.0),
            (cy - bh / 2).clamp(0.0, 1.0),
            (cx + bw / 2).clamp(0.0, 1.0),
            (cy + bh / 2).clamp(0.0, 1.0),
            scores,
            cls_ids.float()
        ], dim=1)
        all_dets.append(p4_dets)

    # 2. Decode P3 (20×20) with class argmax
    if "p3_obj" in preds and "p3_box" in preds:
        p3_obj = torch.sigmoid(preds["p3_obj"])[0, 0]
        p3_max = F.max_pool2d(p3_obj.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
        p3_mask = (p3_obj == p3_max) & (p3_obj > conf_thr)
        p3_ys, p3_xs = p3_mask.nonzero(as_tuple=True)
        if p3_ys.numel() > 0:
            p3_scores = p3_obj[p3_ys, p3_xs]
            G_p3 = p3_obj.shape[-1]
            p3_br = preds["p3_box"][0]
            cx_p3 = (p3_xs.float() + torch.sigmoid(p3_br[0][p3_ys, p3_xs])) / G_p3
            cy_p3 = (p3_ys.float() + torch.sigmoid(p3_br[1][p3_ys, p3_xs])) / G_p3
            bw_p3 = (2.5 * torch.sigmoid(p3_br[2][p3_ys, p3_xs])).clamp(max=1.0)
            bh_p3 = (2.5 * torch.sigmoid(p3_br[3][p3_ys, p3_xs])).clamp(max=1.0)
            # Fix: Use class argmax instead of hardcoding class 0
            if "p3_cls" in preds:
                p3_cls_scores = torch.sigmoid(preds["p3_cls"][0])
                p3_cls_ids = p3_cls_scores[:, p3_ys, p3_xs].argmax(dim=0).float()
            else:
                p3_cls_ids = torch.zeros(p3_ys.shape[0], device=obj.device)
            p3_dets = torch.stack([
                (cx_p3 - bw_p3 / 2).clamp(0.0, 1.0),
                (cy_p3 - bh_p3 / 2).clamp(0.0, 1.0),
                (cx_p3 + bw_p3 / 2).clamp(0.0, 1.0),
                (cy_p3 + bh_p3 / 2).clamp(0.0, 1.0),
                p3_scores,
                p3_cls_ids
            ], dim=1)
            all_dets.append(p3_dets)

    if not all_dets:
        return torch.zeros(0, 6, device=preds["obj"].device)

    combined = torch.cat(all_dets, dim=0)

    # Cross-scale duplicate suppression: if P3 and P4 both detect same object,
    # keep the higher-confidence one
    if len(all_dets) > 1 and combined.shape[0] > 1:
        combined = _cross_scale_suppress(combined, iou_thr=0.5)

    order = torch.argsort(combined[:, 4], descending=True)[:max_det]
    return combined[order]


def _cross_scale_suppress(dets: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
    """Remove duplicate detections across P3/P4 scales via simple IoU check."""
    if dets.shape[0] <= 1:
        return dets

    # Sort by score descending
    order = torch.argsort(dets[:, 4], descending=True)
    dets = dets[order]

    keep = []
    suppressed = set()

    for i in range(dets.shape[0]):
        if i in suppressed:
            continue
        keep.append(i)

        for j in range(i + 1, dets.shape[0]):
            if j in suppressed:
                continue
            # Compute IoU between i and j
            xi1 = max(dets[i, 0].item(), dets[j, 0].item())
            yi1 = max(dets[i, 1].item(), dets[j, 1].item())
            xi2 = min(dets[i, 2].item(), dets[j, 2].item())
            yi2 = min(dets[i, 3].item(), dets[j, 3].item())
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_i = (dets[i, 2] - dets[i, 0]).item() * (dets[i, 3] - dets[i, 1]).item()
            area_j = (dets[j, 2] - dets[j, 0]).item() * (dets[j, 3] - dets[j, 1]).item()
            union = area_i + area_j - inter
            iou = inter / max(union, 1e-9)
            if iou > iou_thr:
                suppressed.add(j)

    return dets[keep]

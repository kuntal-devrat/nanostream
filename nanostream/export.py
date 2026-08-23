"""Model weight exporter: converts trained PyTorch model to static C header."""

import math
import pathlib
import numpy as np
import torch

from .config import DEFAULT_CONFIG
from .fixedpoint import build_sig_lut, calibrate_fixed_point, magic_recip
from .layers import ShiftConv2d
from .model import NanoStreamOD
from .quant import ZERO_EXP


def _format_array_i8(arr, per_line=16, indent=4):
    lines = []
    ind = " " * indent
    flat = list(arr.flatten())
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        lines.append(ind + ", ".join(f"{int(x):4d}" for x in chunk) + ",")
    return "\n".join(lines)


def _format_array_i32(arr, per_line=8, indent=4):
    lines = []
    ind = " " * indent
    flat = list(arr.flatten())
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        lines.append(ind + ", ".join(f"{int(x):8d}" for x in chunk) + ",")
    return "\n".join(lines)


def _format_array_u16(arr, per_line=12, indent=4):
    lines = []
    ind = " " * indent
    flat = list(arr.flatten())
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        lines.append(ind + ", ".join(f"0x{int(x):04x}" for x in chunk) + ",")
    return "\n".join(lines)


def export_c_header(model: NanoStreamOD, calib_fracs: dict, out_path: str | pathlib.Path):
    """Export frozen model weights to static C header file."""
    cfg = model.cfg
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sig_table = build_sig_lut()
    recip_m, recip_s = magic_recip(cfg.grid_size * cfg.grid_size, bits=24)

    header_lines = [
        "/* ========================================================================= */",
        "/* NanoStream-OD: Static C Model Weights & Architecture Definition           */",
        "/* Auto-generated for bare-metal MCU (ARM Cortex-M, ESP32, RISC-V)          */",
        "/* Zero dynamic allocation (no malloc/free)                                 */",
        "/* ========================================================================= */",
        "#ifndef MODEL_WEIGHTS_H",
        "#define MODEL_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define NS_INPUT_SIZE       {cfg.input_size}",
        f"#define NS_INPUT_CHANNELS   {cfg.in_channels}",
        f"#define NS_STRIP_ROWS       {cfg.strip_rows}",
        f"#define NS_STAGES           {len(cfg.stage_widths)}",
        f"#define NS_GRID             {cfg.grid_size}",
        f"#define NS_CG               {cfg.stage_widths[-1]}",
        f"#define NS_HEAD1_CIN        {cfg.stage_widths[-1] + cfg.context_dim}",
        f"#define NS_HID              {cfg.head_hidden}",
        f"#define NS_NUM_CLASSES      {cfg.num_classes}",
        f"#define NS_INPUT_FRAC       {calib_fracs['input_frac']}",
        f"#define NS_RECIP_M          {recip_m}",
        f"#define NS_RECIP_S          {recip_s}",
        f"#define NS_BOX_SCALE_NUM    5",
        f"#define NS_BOX_SCALE_DEN    2",
        "",
        "/* Buffer sizing constants */",
        "#define NS_RING_ELEMS       16384",
        "#define NS_WIN_ELEMS        8192",
        "#define NS_CAS_ELEMS        8192",
        "",
        "typedef struct {",
        "    const char *name;",
        "    int cin, cout;",
        "    int k, pad, stride_sh;",
        "    int w_in, w_out;",
        "    int total_out;",
        "    int in_frac, out_shift;",
        "    const int8_t *exp;",
        "    const int8_t *sgn;",
        "    const int32_t *b;",
        "} ns_layer_t;",
        "",
        "/* Sigmoid LUT (512 entries, Q15 in / Q15 out) */",
        "static const uint16_t NS_SIG_LUT[512] = {",
        _format_array_u16(sig_table),
        "};",
        "",
    ]

    # Export backbone stages (stem + stages)
    stage_layers = []
    cin = cfg.in_channels
    cur_frac = calib_fracs["input_frac"]
    cur_w = cfg.input_size

    # Collect all backbone convs (stem, stage1, stage2, stage3)
    all_stages = [("stem", model.backbone.stem.conv)]
    for i, blk in enumerate(model.backbone.stages):
        all_stages.append((blk.conv._name_hint, blk.conv))

    for i, (name, conv) in enumerate(all_stages):
        cout = conv.out_channels
        k = conv.kernel_size[0]
        pad = conv.padding
        stride = conv.stride
        stride_sh = int(round(math.log2(stride)))
        w_in = cur_w
        w_out = cur_w // stride
        total_out = cur_w // stride
        in_frac = calib_fracs["stage_in_frac"].get(name, cur_frac)
        out_shift = conv.fixed_out_shift

        exp = conv.pow2.exponent.cpu().numpy().astype(np.int8)
        sgn = conv.pow2.sign.cpu().numpy().astype(np.int8)
        bias_f = conv.frozen_bias.detach().cpu().numpy()
        bias_q = np.round(bias_f * (2.0 ** in_frac)).astype(np.int32)

        header_lines.extend([
            f"/* Stage {i} ({name}): in={cin} out={cout} k={k} s={stride} in_frac={in_frac} out_shift={out_shift} */",
            f"static const int8_t ns_exp_{name}[{exp.size}] = {{",
            _format_array_i8(exp),
            "};",
            f"static const int8_t ns_sgn_{name}[{sgn.size}] = {{",
            _format_array_i8(sgn),
            "};",
            f"static const int32_t ns_b_{name}[{bias_q.size}] = {{",
            _format_array_i32(bias_q),
            "};",
            "",
        ])
        stage_layers.append({
            "name": f'"{name}"', "cin": cin, "cout": cout, "k": k, "pad": pad,
            "stride_sh": stride_sh, "w_in": w_in, "w_out": w_out,
            "total_out": total_out, "in_frac": in_frac, "out_shift": out_shift,
            "exp": f"ns_exp_{name}", "sgn": f"ns_sgn_{name}", "b": f"ns_b_{name}"
        })
        cin = cout
        cur_w = w_out
        cur_frac = in_frac - out_shift

    # Export head layers
    head_module = getattr(model.head, "head_p4", model.head)
    head_convs = [
        ("head1", head_module.conv1, cfg.stage_widths[-1] + cfg.context_dim, cfg.head_hidden),
        ("head_obj", head_module.obj, cfg.head_hidden, 1),
        ("head_box", head_module.box, cfg.head_hidden, 4),
        ("head_cls", head_module.cls, cfg.head_hidden, cfg.num_classes),
    ]

    head_layers = {}
    for hname, conv, h_cin, h_cout in head_convs:
        in_frac = calib_fracs["head_in_frac"].get(hname, cur_frac)
        out_shift = conv.fixed_out_shift
        exp = conv.pow2.exponent.cpu().numpy().astype(np.int8)
        sgn = conv.pow2.sign.cpu().numpy().astype(np.int8)
        bias_f = conv.frozen_bias.detach().cpu().numpy()
        bias_q = np.round(bias_f * (2.0 ** in_frac)).astype(np.int32)

        header_lines.extend([
            f"/* Head Layer ({hname}): in={h_cin} out={h_cout} 1x1 in_frac={in_frac} out_shift={out_shift} */",
            f"static const int8_t ns_exp_{hname}[{exp.size}] = {{",
            _format_array_i8(exp),
            "};",
            f"static const int8_t ns_sgn_{hname}[{sgn.size}] = {{",
            _format_array_i8(sgn),
            "};",
            f"static const int32_t ns_b_{hname}[{bias_q.size}] = {{",
            _format_array_i32(bias_q),
            "};",
            "",
        ])
        head_layers[hname] = {
            "name": f'"{hname}"', "cin": h_cin, "cout": h_cout, "k": 1, "pad": 0,
            "stride_sh": 0, "w_in": cfg.grid_size, "w_out": cfg.grid_size,
            "total_out": cfg.grid_size, "in_frac": in_frac, "out_shift": out_shift,
            "exp": f"ns_exp_{hname}", "sgn": f"ns_sgn_{hname}", "b": f"ns_b_{hname}"
        }

    head1_conv = head_module.conv1
    head_out_frac = calib_fracs["head_in_frac"].get("head1", 12) - head1_conv.fixed_out_shift
    # BUG-12 FIX: Find correct insertion point dynamically instead of hardcoded index
    head_frac_line = f"#define NS_HEAD_OUT_FRAC    {head_out_frac}"
    insert_idx = len(header_lines)
    for idx, line in enumerate(header_lines):
        if line.startswith("#define NS_") or line.startswith("#define NS_BOX"):
            insert_idx = idx + 1
    header_lines.insert(insert_idx, head_frac_line)

    # Write stage layers table
    header_lines.extend([
        "/* Backbone stages table */",
        f"static const ns_layer_t ns_layers[{len(stage_layers)}] = {{",
    ])
    for s in stage_layers:
        header_lines.append(
            f"    {{ {s['name']}, {s['cin']}, {s['cout']}, {s['k']}, {s['pad']}, "
            f"{s['stride_sh']}, {s['w_in']}, {s['w_out']}, {s['total_out']}, "
            f"{s['in_frac']}, {s['out_shift']}, {s['exp']}, {s['sgn']}, {s['b']} }},"
        )
    header_lines.append("};")
    header_lines.append("")

    # Write head layers structures
    for hname in ("head1", "head_obj", "head_box", "head_cls"):
        s = head_layers[hname]
        header_lines.append(
            f"static const ns_layer_t ns_{hname} = {{ "
            f"{s['name']}, {s['cin']}, {s['cout']}, {s['k']}, {s['pad']}, "
            f"{s['stride_sh']}, {s['w_in']}, {s['w_out']}, {s['total_out']}, "
            f"{s['in_frac']}, {s['out_shift']}, {s['exp']}, {s['sgn']}, {s['b']} }};"
        )

    header_lines.extend(["", "#endif /* MODEL_WEIGHTS_H */", ""])
    out_path.write_text("\n".join(header_lines), encoding="utf-8")
    return out_path

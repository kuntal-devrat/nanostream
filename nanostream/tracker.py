"""Cycle & SRAM tracker: live diagnostic dashboard for simulated MCU execution."""

import sys
import os
import time


_MHZ_CORTEX_M4 = 168.0
_MHZ_ESP32_S3 = 240.0
_SRAM_BUDGET_KB = 256.0
_FLASH_BUDGET_KB = 512.0

_CYCLE_WEIGHT_MAC = 1.0
_CYCLE_WEIGHT_ADD = 0.45
_CYCLE_WEIGHT_SHIFT = 0.15
_CYCLE_OVERHEAD = 1.18


class ResourceTracker:
    _instance = None

    def __init__(self):
        self.reset()

    @classmethod
    def get(cls) -> "ResourceTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        self.frame_macs = 0
        self.frame_adds = 0
        self.frame_shifts = 0
        self.live_sim_bytes = 0
        self.live_mcu_bytes = 0
        self.peak_sim_bytes = 0
        self.peak_mcu_bytes = 0
        self._buffers = {}
        self.flash_bytes = 0
        self.enabled = True

    def start_frame(self):
        self.frame_macs = 0
        self.frame_adds = 0
        self.frame_shifts = 0

    def log_conv(self, out_elems, cin, cout, k, nonzero_taps):
        if not self.enabled:
            return
        spatial_out = max(1, out_elems // max(1, cout))
        macs = spatial_out * cin * cout * k * k
        self.frame_macs += macs
        self.frame_shifts += int(spatial_out * nonzero_taps)
        self.frame_adds += int(spatial_out * (nonzero_taps + cout))

    def alloc(self, name, sim_bytes, mcu_bytes):
        prev = self._buffers.get(name)
        if prev is not None:
            self.release(name)
        self._buffers[name] = (sim_bytes, mcu_bytes)
        self.live_sim_bytes += sim_bytes
        self.live_mcu_bytes += mcu_bytes
        self.peak_sim_bytes = max(self.peak_sim_bytes, self.live_sim_bytes)
        self.peak_mcu_bytes = max(self.peak_mcu_bytes, self.live_mcu_bytes)

    def release(self, name):
        buf = self._buffers.pop(name, None)
        if buf is not None:
            self.live_sim_bytes -= buf[0]
            self.live_mcu_bytes -= buf[1]

    def set_flash(self, param_count):
        self.flash_bytes = int(param_count)

    def cycles_estimate(self):
        c = (
            self.frame_macs * _CYCLE_WEIGHT_MAC
            + self.frame_adds * _CYCLE_WEIGHT_ADD
            + self.frame_shifts * _CYCLE_WEIGHT_SHIFT
        ) * _CYCLE_OVERHEAD
        return c

    def summary(self):
        cyc = self.cycles_estimate()
        return {
            "macs": self.frame_macs,
            "adds": self.frame_adds,
            "shifts": self.frame_shifts,
            "peak_sram_kb_sim": self.peak_sim_bytes / 1024.0,
            "peak_sram_kb_mcu": self.peak_mcu_bytes / 1024.0,
            "flash_kb": self.flash_bytes / 1024.0,
            "cycles": cyc,
            "ms_cortex_m4_168mhz": cyc / (_MHZ_CORTEX_M4 * 1e6) * 1e3,
            "ms_esp32_s3_240mhz": cyc / (_MHZ_ESP32_S3 * 1e6) * 1e3,
        }

    def dashboard(
        self,
        frame_idx=None,
        fps=None,
        detections=None,
        mode="stream",
    ):
        s = self.summary()
        sram_kb = s["peak_sram_kb_mcu"] if mode == "stream" else s["peak_sram_kb_sim"]
        budget_frac = min(1.0, sram_kb / _SRAM_BUDGET_KB)
        filled = int(round(budget_frac * 20))
        bar = "#" * filled + "." * (20 - filled)

        lines = []
        lines.append("+=" + "-" * 58 + "=+")
        title = "NanoStream-OD | NMS-free patch-stream detector"
        lines.append("| " + title.ljust(56) + " |")
        lines.append("+" + "-" * 60 + "+")

        col1 = f"Frame {frame_idx}" if frame_idx is not None else "Frame -"
        col2 = f"{fps:5.1f} FPS host" if fps else "  --  FPS host"
        lat = f"lat {1000.0 / fps:5.1f} ms" if fps else ""
        lines.append("| " + f"{col1}  {col2}  {lat} [{mode}]".ljust(56) + " |")

        ram_line = (
            f"Peak SRAM {sram_kb:7.1f} KB / {_SRAM_BUDGET_KB:.0f} KB [{bar}]"
            f" {budget_frac * 100:4.1f}%"
        )
        lines.append("| " + ram_line.ljust(56) + " |")

        ops = (
            f"MACs {s['macs'] / 1e6:6.2f}M  shifts+adds "
            f"{(s['shifts'] + s['adds']) / 1e6:5.2f}M  Flash {s['flash_kb']:5.1f}/{_FLASH_BUDGET_KB:.0f}KB"
        )
        lines.append("| " + ops.ljust(56) + " |")

        mcu_line = (
            f"Est MCU  M4@168MHz {s['ms_cortex_m4_168mhz']:6.1f} ms   "
            f"ESP32-S3@240MHz {s['ms_esp32_s3_240mhz']:5.1f} ms"
        )
        lines.append("| " + mcu_line.ljust(56) + " |")

        det_txt = "none"
        if detections:
            parts = []
            for d in detections[:4]:
                x1, y1, x2, y2, score, name = d
                parts.append(f"{name} {score:.2f}")
                parts.append(f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
                if len(parts) >= 4:
                    break
            det_txt = " ".join(parts)
        lines.append("| " + ("Det: " + det_txt)[:56].ljust(56) + " |")
        lines.append("+" + "=" * 60 + "+")
        return "\n".join(lines)


def print_dashboard(text, first=False):
    if not sys.stdout.isatty():
        print(text)
        return
    if not first:
        n_lines = text.count("\n") + 1
        sys.stdout.write(f"\x1b[{n_lines}A")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def enable_windows_ansi():
    if os.name == "nt":
        os.system("")

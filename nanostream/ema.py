"""Model Exponential Moving Average (EMA) for NanoStream-OD.

Keeps a moving average of model parameters during training to produce
cleaner, higher-accuracy evaluation checkpoints with faster convergence.
"""

import copy
import math
import torch
import torch.nn as nn


class ModelEMA:
    """Maintains moving averages of parameters with warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.999, tau: float = 2000.0):
        # Extract underlying module if DataParallel / DistributedDataParallel
        raw = getattr(model, "module", model)
        self.ema = copy.deepcopy(raw).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    def update(self, model: nn.Module, step: int = None):
        """Update EMA parameters with current model parameters."""
        raw = getattr(model, "module", model)
        self.updates += 1
        curr_step = step if step is not None else self.updates
        # Smooth warmup decay
        d = self.decay * (1.0 - math.exp(-float(curr_step) / self.tau))

        with torch.no_grad():
            msd = raw.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * d + msd[k].detach().to(v.device) * (1.0 - d))
                else:
                    v.copy_(msd[k].detach().to(v.device))

    def state_dict(self):
        return {
            "ema": self.ema.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
            "tau": self.tau,
        }

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["ema"])
        self.updates = state.get("updates", 0)
        self.decay = state.get("decay", self.decay)
        self.tau = state.get("tau", self.tau)

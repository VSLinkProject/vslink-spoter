import math

import numpy as np
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        ref = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(dtype=ref[k].dtype) for k in ref}, strict=True)


def mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def warmup_cosine(opt, warmup: int, epochs: int):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warmup if ep < warmup else
        0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(epochs - warmup, 1))))


def amp_tools(device: str):
    on = device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=on)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=on)
    return scaler, (lambda: torch.autocast(device_type="cuda", dtype=torch.float16, enabled=on))

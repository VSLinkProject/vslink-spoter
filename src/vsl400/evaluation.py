import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .models.spoter import build_channels


@torch.no_grad()
def predict_multiview(model, loader, parents, device):
    model.eval()
    out, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        out.append(F.log_softmax(model(build_channels(xb, parents, model.channels)).float(), 1).cpu())
        ys.append(yb)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def predict_front_view(model, loader, parents, device):
    model.eval()
    out, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        x = build_channels(xb[:, 0].unsqueeze(1), parents, model.channels)[:, 0]
        out.append(F.log_softmax(model(x).float(), 1).cpu())
        ys.append(yb)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def topk_metrics(logp: np.ndarray, y: np.ndarray) -> dict:
    top5 = np.argsort(-logp, axis=1)[:, :5]
    return dict(top1=float((top5[:, 0] == y).mean()),
                top5=float((top5 == y[:, None]).any(1).mean()),
                macro_f1=float(f1_score(y, top5[:, 0], average="macro", zero_division=0)))


def per_signer_accuracy(correct: np.ndarray, signers: np.ndarray) -> dict:
    return {int(k): float(correct[signers == k].mean()) for k in sorted(set(signers.tolist()))}

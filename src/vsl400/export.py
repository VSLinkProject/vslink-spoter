import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data.normalization import (BODY, EYE_L, EYE_OFFSET, HANDS, HEAD_HEIGHT, HEAD_WIDTH,
                                  MARGIN, NOSE, SHOULDER_L, SHOULDER_R)


class SpoterNormalize(nn.Module):
    """Torch port of normalize_spoter using ONNX-exportable ops: (B, T, 75, 2) -> (B, T, 75, 2)."""

    def __init__(self, t_frames: int = 60):
        super().__init__()
        self.register_buffer("tri", torch.tril(torch.ones(t_frames, t_frames, dtype=torch.bool)))
        self.register_buffer("order", torch.arange(t_frames, dtype=torch.float32))

    def forward(self, x):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        present = (x[..., 0] != 0) | (x[..., 1] != 0)
        sl, sr = x[:, :, SHOULDER_L], x[:, :, SHOULDER_R]
        head = torch.sqrt(((sl - sr) ** 2).sum(-1))
        valid = present[:, :, SHOULDER_L] & present[:, :, SHOULDER_R] & (head > 1e-6)
        any_valid = valid.any(-1, keepdim=True)
        neck = (sl + sr) / 2.0
        eye_y = torch.where(present[:, :, EYE_L], x[:, :, EYE_L, 1],
                            torch.where(present[:, :, NOSE], x[:, :, NOSE, 1], neck[..., 1] - head))

        T = x.shape[1]
        tt = self.order[:T]
        last = torch.where(self.tri[:T, :T].unsqueeze(0) & valid.unsqueeze(1),
                           tt.view(1, 1, -1), torch.full_like(tt, -1.0).view(1, 1, -1)).amax(-1)
        first = torch.where(valid, tt.view(1, -1),
                            torch.full_like(tt, float(T)).view(1, -1)).amin(-1, keepdim=True)
        idx = torch.where(last < 0, first.expand_as(last), last)
        idx = torch.where(any_valid, idx, torch.zeros_like(idx)).long()

        h = torch.gather(head, 1, idx)
        h = torch.where(any_valid, h, torch.ones_like(h))
        left = torch.gather(neck[..., 0], 1, idx) - (HEAD_WIDTH / 2.0) * h
        top = torch.gather(eye_y, 1, idx) - EYE_OFFSET * h
        bx = (x[:, :, BODY, 0] - left[..., None]) / (HEAD_WIDTH * h)[..., None] - 0.5
        by = (x[:, :, BODY, 1] - top[..., None]) / (HEAD_HEIGHT * h)[..., None] - 0.5
        m = present[:, :, BODY]
        ox = [torch.where(m, bx, torch.zeros_like(bx))]
        oy = [torch.where(m, by, torch.zeros_like(by))]

        for a, b in HANDS:
            xa, ya, m = x[:, :, a:b, 0], x[:, :, a:b, 1], present[:, :, a:b]
            has = m.any(-1)
            inf = torch.full_like(xa, float("inf"))
            z = torch.zeros_like(has, dtype=xa.dtype)
            x0 = torch.where(has, torch.where(m, xa, inf).amin(-1), z)
            x1 = torch.where(has, torch.where(m, xa, -inf).amax(-1), z)
            y0 = torch.where(has, torch.where(m, ya, inf).amin(-1), z)
            y1 = torch.where(has, torch.where(m, ya, -inf).amax(-1), z)
            w, hg = x1 - x0, y1 - y0
            wide = w > hg
            dx = torch.where(wide, MARGIN * w, MARGIN * hg + (hg - w) / 2.0)
            dy = torch.where(wide, MARGIN * w + (w - hg) / 2.0, MARGIN * hg)
            side = torch.maximum(w, hg) * (1.0 + 2.0 * MARGIN)
            ok = has & (side > 1e-8)
            side = torch.where(ok, side, torch.ones_like(side))
            nx = (xa - (x0 - dx)[..., None]) / side[..., None] - 0.5
            ny = (ya - (y0 - dy)[..., None]) / side[..., None] - 0.5
            keep = m & ok[..., None]
            ox.append(torch.where(keep, nx, torch.zeros_like(nx)))
            oy.append(torch.where(keep, ny, torch.zeros_like(ny)))

        out = torch.stack([torch.cat(ox, -1), torch.cat(oy, -1)], -1)
        return torch.where(any_valid[:, :, None, None], out, torch.zeros_like(out))


class WebModel(nn.Module):
    """Raw keypoints (B, 60, 75, 2) -> class probabilities: normalization + 6 channels + student."""

    def __init__(self, student: nn.Module, parents: torch.Tensor, t_frames: int = 60):
        super().__init__()
        self.normalize = SpoterNormalize(t_frames)
        self.student = student
        self.register_buffer("parents", parents.detach().cpu().clone().long())

    def forward(self, keypoints):
        x = self.normalize(keypoints)
        bone = x - torch.index_select(x, 2, self.parents)
        motion = torch.cat([x[:, 1:] - x[:, :-1], torch.zeros_like(x[:, :1])], dim=1)
        return torch.softmax(self.student(torch.cat([x, bone, motion], dim=-1)), dim=-1)


def export_onnx(model: nn.Module, example: torch.Tensor, path: str | Path, opsets=(17, 14)) -> int:
    kw = dict(input_names=["keypoints"], output_names=["probs"], do_constant_folding=True,
              dynamic_axes={"keypoints": {0: "batch"}, "probs": {0: "batch"}})
    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    errors = []
    for op in opsets:
        try:
            k = dict(kw, opset_version=op)
            if has_dynamo:
                k["dynamo"] = False
            torch.onnx.export(model, (example,), str(path), **k)
            return op
        except Exception as e:  # noqa: BLE001
            errors.append(f"opset {op}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("ONNX export failed:\n" + "\n".join(errors))


def synthetic_keypoints(t_frames: int = 60, num_keypoints: int = 75, seed: int = 0) -> np.ndarray:
    """Synthetic keypoint sequence (t_frames, K, 2) used as a public test probe."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, t_frames)[:, None]
    x = np.zeros((t_frames, num_keypoints, 2), np.float32)
    x[:, :33] = 0.5 + rng.normal(0, 0.08, (1, 33, 2))
    x[:, NOSE] = [0.50, 0.30]
    x[:, EYE_L] = [0.48, 0.28]
    x[:, SHOULDER_L] = [0.40, 0.45]
    x[:, SHOULDER_R] = [0.60, 0.45]
    for (a, b), cx, sgn in ((HANDS[0], 0.38, 1.0), (HANDS[1], 0.62, -1.0)):
        centre = np.concatenate([cx + 0.06 * np.cos(sgn * t), 0.62 + 0.06 * np.sin(t)], axis=1)
        x[:, a:b] = centre[:, None, :] + rng.normal(0, 0.02, (1, b - a, 2))
    return x.astype(np.float32)


def run_onnx(path: str | Path, X: np.ndarray, batch: int = 256) -> np.ndarray:
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"keypoints": X[i:i + batch]})[0]
                           for i in range(0, len(X), batch)])


@torch.no_grad()
def run_torch(model: nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    return np.concatenate([model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))).numpy()
                           for i in range(0, len(X), batch)])

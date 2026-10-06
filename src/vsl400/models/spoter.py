import numpy as np
import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """Transformer encoder shared by all views; views differ only by a learned view embedding."""

    def __init__(self, num_keypoints=75, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, num_views=3, channels=6):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(num_keypoints * channels, d_model), nn.LayerNorm(d_model),
            nn.ReLU(), nn.Dropout(dropout))
        self.pos_embedding = nn.Parameter(torch.randn(1, 120, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.view_embedding = nn.Parameter(torch.randn(num_views, 1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward_views(self, x):
        B, V, T, _ = x.shape
        z = self.input_projection(x.reshape(B * V, T, -1))
        z = torch.cat((self.cls_token.expand(B * V, -1, -1), z), dim=1)
        z = z + self.pos_embedding[:, :T + 1, :]
        ve = self.view_embedding.squeeze(1).squeeze(1)
        z = z + ve.unsqueeze(0).expand(B, V, -1).reshape(B * V, 1, -1)
        return self.transformer(z)[:, 0, :].reshape(B, V, -1)


def _head(d_in, d_model, num_classes, dropout):
    return nn.Sequential(nn.Linear(d_in, d_model), nn.BatchNorm1d(d_model), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(d_model, num_classes))


class SpoterMultiView(nn.Module):
    """Teacher: shared encoder over 3 views, classifier on the concatenated CLS features."""

    def __init__(self, num_classes=400, num_keypoints=75, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.3, channels=6, num_views=3):
        super().__init__()
        self.channels = channels
        self.shared_encoder = SharedEncoder(num_keypoints, d_model, nhead, num_layers,
                                            dim_feedforward, dropout, num_views, channels)
        self.classifier = _head(d_model * num_views, d_model, num_classes, dropout)

    def forward(self, x):
        B, V, T, K, C = x.shape
        feat = self.shared_encoder.forward_views(x.reshape(B, V, T, K * C))
        return self.classifier(feat.reshape(B, -1))


class SpoterSingleView(nn.Module):
    """Student: same encoder on the front view only (view embedding 0)."""

    def __init__(self, num_classes=400, num_keypoints=75, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.3, channels=6, num_views=3):
        super().__init__()
        self.channels = channels
        self.shared_encoder = SharedEncoder(num_keypoints, d_model, nhead, num_layers,
                                            dim_feedforward, dropout, num_views, channels)
        self.classifier = _head(d_model, d_model, num_classes, dropout)

    def features(self, x):
        B, T, K, C = x.shape
        e = self.shared_encoder
        z = e.input_projection(x.reshape(B, T, K * C))
        z = torch.cat((e.cls_token.expand(B, -1, -1), z), dim=1)
        z = z + e.pos_embedding[:, :T + 1, :]
        z = z + e.view_embedding[0]
        return e.transformer(z)[:, 0, :]

    def forward(self, x):
        return self.classifier(self.features(x))


def model_kwargs(cfg: dict) -> dict:
    m = cfg["model"]
    return dict(num_classes=m["num_classes"], num_keypoints=cfg["data"]["num_keypoints"],
                d_model=m["d_model"], nhead=m["nhead"], num_layers=m["num_layers"],
                dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
                channels=m["channels"], num_views=len(cfg["data"]["views"]))


def bone_parents() -> torch.Tensor:
    parent = np.arange(75, dtype=np.int64)
    pose = {11: 11, 12: 11, 0: 11,
            13: 11, 15: 13, 17: 15, 19: 15, 21: 15,
            14: 12, 16: 14, 18: 16, 20: 16, 22: 16,
            23: 11, 24: 12, 25: 23, 26: 24, 27: 25, 28: 26,
            29: 27, 30: 28, 31: 29, 32: 30,
            1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 7: 3, 8: 6, 9: 0, 10: 0}
    for k, v in pose.items():
        parent[k] = v
    hand = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    for off in (33, 54):
        for i, c in enumerate(hand):
            parent[off + i] = off + c
    return torch.from_numpy(parent)


def build_channels(x: torch.Tensor, parents: torch.Tensor, channels: int = 6) -> torch.Tensor:
    if channels == 2:
        return x
    bone = x - x[:, :, :, parents, :]
    motion = torch.zeros_like(x)
    motion[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    return torch.cat([x, bone, motion], dim=-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

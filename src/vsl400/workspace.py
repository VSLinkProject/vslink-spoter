from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data.cache import load_cache
from .data.datasets import multiview_loader
from .models.spoter import bone_parents
from .utils import ensure_dir, get_device, load_json


@dataclass
class Workspace:
    cfg: dict
    out: Path
    df: pd.DataFrame
    cache: np.ndarray
    cache_raw: np.ndarray
    labels: np.ndarray
    signers: np.ndarray
    split: dict
    label_names: list
    device: str
    parents: object
    _loaders: dict = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: dict) -> "Workspace":
        out = Path(cfg["paths"]["output_dir"])
        ddir = out / "data"
        if not (ddir / "splits.json").exists():
            raise FileNotFoundError(f"{ddir}/splits.json not found; run scripts/prepare_data.py first")
        df = pd.read_csv(ddir / "index_split.csv")
        cache, cache_raw = load_cache(ddir)
        device = get_device()
        return cls(cfg=cfg, out=out, df=df, cache=cache, cache_raw=cache_raw,
                   labels=df["label"].to_numpy().astype(np.int64),
                   signers=df["signer_id"].to_numpy().astype(np.int64),
                   split={k: np.asarray(v, dtype=np.int64)
                          for k, v in load_json(ddir / "splits.json").items()},
                   label_names=load_json(ddir / "labels.json"),
                   device=device, parents=bone_parents().to(device))

    def dir(self, *parts) -> Path:
        return ensure_dir(self.out.joinpath(*parts))

    def eval_loader(self, name: str):
        if name not in self._loaders:
            self._loaders[name] = multiview_loader(self.cache, self.labels, self.split[name], batch=64,
                                                   workers=self.cfg["train"]["loader_workers"])
        return self._loaders[name]

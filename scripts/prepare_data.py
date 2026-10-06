import shutil
from pathlib import Path

import pandas as pd

from vsl400.data.cache import build_cache
from vsl400.data.splits import check_label_order, make_signer_split, split_indices
from vsl400.utils import ensure_dir, load_config, load_json, parse_args, save_json


def main():
    cfg = load_config(parse_args("Split by signer and build the keypoint cache.").config)
    paths, d = cfg["paths"], cfg["data"]
    for key in ("keypoint_root", "index_csv", "labels_json"):
        if not paths.get(key) or not Path(paths[key]).exists():
            raise FileNotFoundError(f"paths.{key} is missing")
    out = ensure_dir(Path(paths["output_dir"]) / "data")

    df = pd.read_csv(paths["index_csv"])
    labels = load_json(paths["labels_json"])
    assert len(labels) == cfg["model"]["num_classes"]
    check_label_order(df, labels)
    df = make_signer_split(df, d["split"]["seed"], d["split"]["ratios"], d["split"]["expected"])
    for name in ("train", "val", "test"):
        sub = df[df.split == name]
        print(f"{name:5s} {len(sub):6d} videos  {sub.signer_id.nunique():2d} signers")

    df.to_csv(out / "index_split.csv", index=False)
    shutil.copy(paths["labels_json"], out / "labels.json")
    save_json({k: v.tolist() for k, v in split_indices(df).items()}, out / "splits.json")
    build_cache(df, paths["keypoint_root"], out, d["views"], d["t_frames"], d["cache_workers"])
    print(f"cache written to {out}")


if __name__ == "__main__":
    main()

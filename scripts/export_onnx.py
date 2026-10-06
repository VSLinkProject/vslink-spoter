import numpy as np
import torch

from vsl400.data.normalization import normalize_spoter
from vsl400.export import SpoterNormalize, WebModel, export_onnx, run_onnx, run_torch, synthetic_keypoints
from vsl400.models.spoter import SpoterSingleView, bone_parents, model_kwargs
from vsl400.utils import load_config, parse_args, save_json
from vsl400.workspace import Workspace


def main():
    ws = Workspace.load(load_config(parse_args("Export the student to ONNX with normalization built in.").config))
    T, K = ws.cfg["data"]["t_frames"], ws.cfg["data"]["num_keypoints"]
    sd = torch.load(ws.out / "checkpoints" / "student.pt", map_location="cpu")["model_state_dict"]
    student = SpoterSingleView(**model_kwargs(ws.cfg))
    student.load_state_dict(sd, strict=True)
    web = WebModel(student.eval(), bone_parents(), T).eval()
    exp = ws.dir("export")

    X = np.ascontiguousarray(ws.cache_raw[ws.split["test"]])
    sub = X[np.random.default_rng(11).choice(len(X), size=min(2000, len(X)), replace=False)]
    with torch.no_grad():
        norm_err = float(np.abs(SpoterNormalize(T)(torch.from_numpy(sub)).numpy()
                                - np.stack([normalize_spoter(k, T) for k in sub])).max())
    assert norm_err < 1e-5, f"normalization mismatch: {norm_err:.2e}"

    path = exp / "model.onnx"
    opset = export_onnx(web, torch.from_numpy(X[:2]), path, ws.cfg["export"]["opsets"])
    p_ort = run_onnx(path, X)
    p_pt = run_torch(web, X)
    prob_err = float(np.abs(p_ort - p_pt).max())
    agree = float((p_ort.argmax(1) == p_pt.argmax(1)).mean())
    assert prob_err < 1e-4 and agree >= 0.999, f"ONNX mismatch: {prob_err:.2e}, top-1 agreement {agree:.4f}"

    probe = synthetic_keypoints(T, K)
    p_probe = run_onnx(path, probe[None])[0]
    save_json(dict(keypoints=np.round(probe, 6).tolist(),
                   top5=[dict(index=int(i), prob=round(float(p_probe[i]), 6))
                         for i in np.argsort(-p_probe)[:5]]), exp / "sample_check.json")
    save_json(ws.label_names, exp / "labels.json")
    save_json(dict(input=dict(name="keypoints", dtype="float32", shape=["batch", T, K, 2]),
                   output=dict(name="probs", dtype="float32", shape=["batch", len(ws.label_names)]),
                   keypoint_order={"0-32": "pose", "33-53": "left hand", "54-74": "right hand"},
                   coordinates="MediaPipe normalized image coordinates x, y in [0, 1]; no mirroring",
                   missing="0 for every point of an undetected part",
                   frame_sampling=f"idx[i] = floor(i * (T - 1) / {T - 1}), i = 0..{T - 1}",
                   opset=opset), exp / "preprocessing.json")
    print(f"exported {path} (opset {opset}, {path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

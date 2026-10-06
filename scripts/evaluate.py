import pandas as pd
import torch

from vsl400.evaluation import per_signer_accuracy, predict_front_view, predict_multiview, topk_metrics
from vsl400.models.spoter import SpoterSingleView, model_kwargs
from vsl400.training.trainer import load_teacher
from vsl400.utils import load_config, parse_args, save_json
from vsl400.workspace import Workspace


def main():
    ws = Workspace.load(load_config(parse_args("Evaluate teacher and student on the test set.").config))
    ckpt = ws.out / "checkpoints"
    loader = ws.eval_loader("test")
    signers = ws.signers[ws.split["test"]]

    teacher = load_teacher(ws, torch.load(ckpt / "teacher.pt", map_location=ws.device)["model_state_dict"])
    student = SpoterSingleView(**model_kwargs(ws.cfg)).to(ws.device)
    student.load_state_dict(torch.load(ckpt / "student.pt", map_location=ws.device)["model_state_dict"])

    results, by_signer = {}, {}
    for name, (logp, y) in {"teacher_3view": predict_multiview(teacher, loader, ws.parents, ws.device),
                            "student_1view": predict_front_view(student, loader, ws.parents, ws.device)}.items():
        results[name] = topk_metrics(logp, y)
        by_signer[name] = per_signer_accuracy(logp.argmax(1) == y, signers)

    res = ws.dir("results")
    save_json(results, res / "metrics.json")
    pd.DataFrame(by_signer).rename_axis("signer").to_csv(res / "per_signer.csv")
    print(f"{'model':<15}{'top-1':>8}{'top-5':>8}{'macro-F1':>10}")
    for name, m in results.items():
        print(f"{name:<15}{m['top1']:>8.2%}{m['top5']:>8.2%}{m['macro_f1']:>10.4f}")


if __name__ == "__main__":
    main()

import torch

from vsl400.models.warm_start import warm_start_gap
from vsl400.training.trainer import load_teacher, train_teacher, warm_student
from vsl400.utils import load_config, parse_args
from vsl400.workspace import Workspace


def main():
    ws = Workspace.load(load_config(parse_args("Train the 3-view teacher.").config))
    sd, info = train_teacher(ws)
    torch.save({"model_state_dict": sd, **info}, ws.dir("checkpoints") / "teacher.pt")

    teacher = load_teacher(ws, sd)
    xb, _ = next(iter(ws.eval_loader("val")))
    gap = warm_start_gap(teacher, warm_student(ws, teacher), xb.to(ws.device), ws.parents)
    assert gap < 1e-3, f"warm start mismatch: {gap:.2e}"
    print(f"teacher val {info['val']:.4f} (epoch {info['epoch']})")


if __name__ == "__main__":
    main()

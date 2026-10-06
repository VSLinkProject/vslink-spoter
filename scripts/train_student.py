import torch

from vsl400.training.trainer import load_teacher, train_student
from vsl400.utils import load_config, parse_args
from vsl400.workspace import Workspace


def main():
    ws = Workspace.load(load_config(parse_args("Train the front-view student from the teacher.").config))
    ck = torch.load(ws.out / "checkpoints" / "teacher.pt", map_location=ws.device)
    sd = train_student(ws, load_teacher(ws, ck["model_state_dict"]))
    torch.save({"model_state_dict": sd}, ws.dir("checkpoints") / "student.pt")


if __name__ == "__main__":
    main()

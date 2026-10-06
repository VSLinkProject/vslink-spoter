import torch

from .spoter import SpoterMultiView, SpoterSingleView, build_channels


def warm_start_student(teacher: SpoterMultiView, student: SpoterSingleView,
                       parents: torch.Tensor, t_frames: int, num_keypoints: int) -> SpoterSingleView:
    """Copy the teacher into the student; side-view contributions are folded into the bias."""
    D = teacher.shared_encoder.cls_token.shape[-1]
    dev = next(teacher.parameters()).device
    student.shared_encoder.load_state_dict(teacher.shared_encoder.state_dict())
    with torch.no_grad():
        zeros = build_channels(torch.zeros(1, 3, t_frames, num_keypoints, 2, device=dev),
                               parents, teacher.channels)
        c = teacher.shared_encoder.forward_views(zeros.reshape(1, 3, t_frames, -1))[0]
        W, b = teacher.classifier[0].weight, teacher.classifier[0].bias
        student.classifier[0].weight.copy_(W[:, :D])
        student.classifier[0].bias.copy_(b + W[:, D:2 * D] @ c[1] + W[:, 2 * D:] @ c[2])
        student.classifier[1].load_state_dict(teacher.classifier[1].state_dict())
        student.classifier[4].load_state_dict(teacher.classifier[4].state_dict())
    return student


@torch.no_grad()
def warm_start_gap(teacher, student, x_views: torch.Tensor, parents: torch.Tensor) -> float:
    teacher.eval(); student.eval()
    x3 = build_channels(x_views, parents, teacher.channels).clone()
    x3[:, 1:] = 0
    x1 = build_channels(x_views[:, :1], parents, student.channels)[:, 0]
    return float((teacher(x3) - student(x1)).abs().max())

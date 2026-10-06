import time

import pandas as pd
import torch
import torch.nn as nn

from ..data.datasets import front_view_loader, multiview_loader
from ..evaluation import predict_multiview
from ..models.spoter import SpoterMultiView, SpoterSingleView, build_channels, model_kwargs
from ..models.warm_start import warm_start_student
from ..utils import free_memory, set_seed
from .components import EMA, amp_tools, mixup, warmup_cosine


def _state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _log(name, ep, E, every, msg):
    if ep + 1 == E or (every > 0 and (ep == 0 or (ep + 1) % every == 0)):
        print(f"{name} epoch {ep + 1:3d}/{E} {msg}", flush=True)


def _step(model, opt, scaler, loss):
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()


def load_teacher(ws, state_dict) -> SpoterMultiView:
    m = SpoterMultiView(**model_kwargs(ws.cfg)).to(ws.device)
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def warm_student(ws, teacher) -> SpoterSingleView:
    student = SpoterSingleView(**model_kwargs(ws.cfg)).to(ws.device)
    return warm_start_student(teacher, student, ws.parents,
                              ws.cfg["data"]["t_frames"], ws.cfg["data"]["num_keypoints"])


def train_teacher(ws):
    cfg, tc, tr = ws.cfg, ws.cfg["teacher"], ws.cfg["train"]
    E = tc["epochs"]
    set_seed(tc["seed"])
    model = SpoterMultiView(**model_kwargs(cfg)).to(ws.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tr["weight_decay"])
    sched = warmup_cosine(opt, tc["warmup"], E)
    ce = nn.CrossEntropyLoss(label_smoothing=tr["label_smoothing"])
    ema = EMA(model, tr["ema_decay"])
    scaler, autocast = amp_tools(ws.device)
    loader = multiview_loader(ws.cache, ws.labels, ws.split["train"], tr["batch_size"],
                              tr["loader_workers"], cfg["augment"]["teacher"])
    val_loader, y_val = ws.eval_loader("val"), ws.labels[ws.split["val"]]

    best, best_sd, best_ep, hist = -1.0, None, 0, []
    for ep in range(E):
        t0 = time.time()
        model.train()
        tot, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            xb = build_channels(xb.to(ws.device, non_blocking=True), ws.parents, model.channels)
            yb = yb.to(ws.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                xm, ya, yc, lam = mixup(xb, yb, tc["mixup"])
                out = model(xm)
                loss = lam * ce(out, ya) + (1 - lam) * ce(out, yc)
            if torch.isnan(loss):
                continue
            _step(model, opt, scaler, loss)
            ema.update(model)
            tot += float(loss.detach()) * yb.size(0)
            correct += int((out.argmax(1) == yb).sum())
            seen += yb.size(0)
        sched.step()

        raw_sd = _state(model)
        acc_raw = float((predict_multiview(model, val_loader, ws.parents, ws.device)[0].argmax(1) == y_val).mean())
        ema.copy_to(model)
        acc_ema = float((predict_multiview(model, val_loader, ws.parents, ws.device)[0].argmax(1) == y_val).mean())
        if max(acc_raw, acc_ema) > best:
            best, best_ep = max(acc_raw, acc_ema), ep + 1
            best_sd = _state(model) if acc_ema >= acc_raw else raw_sd
        model.load_state_dict(raw_sd)

        hist.append(dict(epoch=ep + 1, loss=tot / max(seen, 1), train_acc=correct / max(seen, 1),
                         val_raw=acc_raw, val_ema=acc_ema))
        _log("teacher", ep, E, tr["log_every"],
             f"({time.time() - t0:.1f}s) loss {hist[-1]['loss']:.3f} val {max(acc_raw, acc_ema):.4f} "
             f"best {best:.4f} @{best_ep}")

    pd.DataFrame(hist).to_csv(ws.dir("history") / "teacher.csv", index=False)
    del loader, model, ema, opt
    free_memory()
    return best_sd, dict(val=best, epoch=best_ep)


def train_student(ws, teacher):
    cfg, sc, tr = ws.cfg, ws.cfg["student"], ws.cfg["train"]
    E = sc["epochs"]
    set_seed(sc["seed"])
    model = warm_student(ws, teacher)
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=tr["weight_decay"])
    sched = warmup_cosine(opt, sc["warmup"], E)
    ce = nn.CrossEntropyLoss(label_smoothing=tr["label_smoothing"])
    ema = EMA(model, tr["ema_decay"])
    scaler, autocast = amp_tools(ws.device)
    loader = front_view_loader(ws.cache_raw, ws.labels, ws.split["train"], tr["batch_size"],
                               tr["student_loader_workers"], cfg["augment"]["student"],
                               cfg["data"]["t_frames"])

    hist = []
    for ep in range(E):
        t0 = time.time()
        model.train()
        tot, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            x = build_channels(xb.to(ws.device, non_blocking=True).unsqueeze(1), ws.parents,
                               model.channels)[:, 0]
            yb = yb.to(ws.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(x)
                loss = ce(out, yb)
            if torch.isnan(loss):
                continue
            _step(model, opt, scaler, loss)
            ema.update(model)
            tot += float(loss.detach()) * yb.size(0)
            correct += int((out.argmax(1) == yb).sum())
            seen += yb.size(0)
        sched.step()
        hist.append(dict(epoch=ep + 1, loss=tot / max(seen, 1), train_acc=correct / max(seen, 1)))
        _log("student", ep, E, tr["log_every"],
             f"({time.time() - t0:.1f}s) loss {hist[-1]['loss']:.3f} train {hist[-1]['train_acc']:.4f}")

    ema.copy_to(model)
    sd = _state(model)
    pd.DataFrame(hist).to_csv(ws.dir("history") / "student.csv", index=False)
    del loader, model, ema, opt
    free_memory()
    return sd

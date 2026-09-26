"""Unchanged optimizer, schedules, clipping and EMA update order."""

import math
import torch
from .losses import objective, balanced_gradients
from .probe import probe_objective
from .runtime import autocast


def make_optimizer(model, c, device):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or name.endswith((".bias", ".position")) else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": c["optimizer"]["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=c["optimizer"]["lr"],
        betas=tuple(c["optimizer"]["betas"]),
        eps=1e-8,
        fused=str(device).startswith("cuda"),
    )


def learning_rate(draws, c):
    opt = c["optimizer"]
    warm = opt["warmup_epochs"] * c["train_records"]
    if draws < warm:
        return opt["start_lr"] + (opt["lr"] - opt["start_lr"]) * draws / max(1, warm)
    progress = min(
        1.0,
        max(0.0, (draws - warm) / max(1, c["total_draws"] - warm - c["batch_size"])),
    )
    return opt["min_lr"] + (opt["lr"] - opt["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


def ema_momentum(draws, c):
    fraction = min(1.0, (draws + c["batch_size"]) / c["total_draws"])
    return c["ema_end"] - (c["ema_end"] - c["ema_start"]) * (1 + math.cos(math.pi * fraction)) / 2


def one_update(model, optimizer, observed, target, c, draws, device, target_norms=None, zero_targets=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(draws, c)
    for group in optimizer.param_groups:
        group["lr"] = lr
    observed = observed.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    with autocast(device):
        objective_for_stage = probe_objective if c["stage"] == "probe" else objective
        raw, aux, per_field, per_jepa = objective_for_stage(
            model, observed, target, c, target_norms, zero_targets
        )
    if not all(bool(torch.isfinite(v)) for v in list(raw.values()) + list(aux.values())):
        raise FloatingPointError("Nonfinite loss; no optimizer update")
    metrics = {k: float(v.detach()) for k, v in raw.items()}
    guard = balanced_gradients(
        model.named_parameters(), raw["field"], aux, c["minimum_field_gradient_share"]
    )
    metrics.update(guard)
    for name, value in aux.items():
        metrics["weighted_" + name] = guard["aux_scale"] * float(value.detach())
    metrics["total"] = metrics["field"] + sum(metrics["weighted_" + k] for k in aux)
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        c["optimizer"]["grad_clip"],
        error_if_nonfinite=True,
    )
    if any(p.grad is not None for p in model.target_encoder.parameters()):
        raise AssertionError("EMA teacher received a gradient")
    optimizer.step()
    momentum = 0.0
    if c["use_jepa"] or c["stage"] == "teacher_pretrain":
        momentum = ema_momentum(draws, c)
        model.update_target(momentum)
    metrics.update(
        lr=lr,
        grad_norm_before_clip=float(norm),
        ema_momentum=momentum,
        jepa_active=float(c["use_jepa"]),
        teacher_forward_active=float(c["use_jepa"]),
    )
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite metric")
    return metrics, per_field.cpu(), per_jepa.cpu()

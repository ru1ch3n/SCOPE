"""Physical relative L2, JEPA prediction and the audited gradient-norm guard."""

import math
import torch


def physical_relative(predicted, target, normalization):
    scale = (
        torch.as_tensor(normalization["std"], device=target.device, dtype=torch.float32).view(1, 2, 1, 1) * 2
    )
    mean = torch.as_tensor(normalization["mean"], device=target.device, dtype=torch.float32).view(1, 2, 1, 1)
    physical = target.float() * scale + mean
    denominator = torch.linalg.vector_norm(physical.flatten(2), dim=2)
    if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
        raise FloatingPointError("Zero/nonfinite target norm: fail explicitly; no sample omitted")
    return (
        torch.linalg.vector_norm(((predicted.float() - target.float()) * scale).flatten(2), dim=2)
        / denominator
    )


def field_objective(predicted, target, c, target_norms, zero_targets):
    """Source-audited zeros use the approved finite training-only extension.

    Nonzero denominators are physical L2 norms audited from original float64
    fields. Zero flags come from that source audit, NEVER from a model prediction.
    """
    if c["pde"] != "ns-bounded":
        return physical_relative(predicted, target, c["normalization"])
    if target_norms is None or zero_targets is None:
        raise ValueError("Explicit source-audited norms and zero flags are required")
    scale = 2 * torch.as_tensor(c["normalization"]["std"], device=target.device, dtype=torch.float32).view(
        1, 2, 1, 1
    )
    mean = torch.as_tensor(c["normalization"]["mean"], device=target.device, dtype=torch.float32).view(
        1, 2, 1, 1
    )
    norms = target_norms.to(device=target.device, dtype=torch.float32)
    flags = zero_targets.to(device=target.device)
    if flags.dtype != torch.bool or norms.shape != target.shape[:2] or flags.shape != norms.shape:
        raise ValueError("Invalid source target metadata")
    if (
        not bool(torch.isfinite(norms).all())
        or bool((norms < 0).any())
        or bool(((norms <= 0) & ~flags).any())
    ):
        raise ValueError("Invalid nonzero source norm")
    rms = torch.as_tensor(c["zero_target_policy"]["train_rms_a_u"], device=target.device, dtype=torch.float32)
    if rms.shape != (2,) or not bool(torch.isfinite(rms).all()) or bool((rms <= 0).any()):
        raise ValueError("Invalid training-split RMS scale")
    error = torch.where(
        flags[:, :, None, None],
        predicted.float() * scale + mean,
        (predicted.float() - target.float()) * scale,
    )
    denominator = torch.where(flags, rms[None, :] * math.sqrt(target.shape[-2] * target.shape[-1]), norms)
    return torch.linalg.vector_norm(error.flatten(2), dim=2) / denominator


def objective(model, observed, target, c, target_norms=None, zero_targets=None):
    """Physical field objectives throughout; original latent MSE and variance definition.

    One body serves every objective variant. A term is present exactly when its
    published weight is nonzero, so `full` is computed as before, term by term
    and in the same order. `L_V` always acts on the online full-view latent
    `z_F = encoder(full_pair_view(target))`, never on the predicted latent.
    `grounding_gradient == "decoder_only"` decodes a stop-gradient copy of
    `z_F`: the reported grounding value is unchanged, only its gradient path is.
    A variant with no JEPA weight performs no teacher forward at all.
    """
    weights = c["loss"]
    pretrain = c["stage"] == "teacher_pretrain"
    use_jepa = bool(c["use_jepa"])
    use_grounding = not pretrain and weights["grounding_weight"] > 0
    use_variance = weights["variance_weight"] > 0
    zero = torch.zeros((), device=target.device)
    if not (pretrain or use_jepa or use_grounding or use_variance):
        # Field only: the fully visible pair never enters the graph at all.
        output = model(observed)
        per_field = field_objective(output["fields"], target, c, target_norms, zero_targets)
        raw = {
            "field": per_field.mean(),
            "jepa": zero,
            "grounding": zero,
            "variance": zero,
            "field_a_objective": per_field[:, 0].mean(),
            "field_u_objective": per_field[:, 1].mean(),
            "online_full_latent_std": zero,
            "teacher_latent_std": zero,
        }
        return raw, {}, per_field.detach(), per_field.new_zeros(len(target))
    full_view = model.full_pair_view(target)
    full_latent = model.encoder(full_view)
    variance = torch.relu(0.1 - full_latent.float().flatten(1).std(dim=0, correction=0)).square().mean()
    if pretrain:
        predicted = model.decoder(full_latent)
        per_field = field_objective(predicted, target, c, target_norms, zero_targets)
        z = per_field.new_zeros(len(target))
        raw = {"field": per_field.mean(), "variance": variance}
        auxiliaries = {"variance": weights["variance_weight"] * variance} if use_variance else {}
        teacher_std = full_latent.detach().float().std(correction=0)
    else:
        output = model(observed)
        per_field = field_objective(output["fields"], target, c, target_norms, zero_targets)
        auxiliaries, raw = {}, {"field": per_field.mean(), "variance": variance}
        if use_grounding:
            latent = full_latent.detach() if c.get("grounding_gradient") == "decoder_only" else full_latent
            grounding = field_objective(model.decoder(latent), target, c, target_norms, zero_targets).mean()
            auxiliaries["grounding"] = weights["grounding_weight"] * grounding
            raw["grounding"] = grounding
        if use_variance:
            auxiliaries["variance"] = weights["variance_weight"] * variance
        if use_jepa:
            with torch.no_grad():
                teacher = model.target_encoder(full_view)
            z = (output["predicted_latent"].float() - teacher.float()).square().mean((1, 2))
            raw["jepa"] = z.mean()
            auxiliaries = {"jepa": weights["jepa_weight"] * raw["jepa"], **auxiliaries}
            teacher_std = teacher.float().std(correction=0)
        else:
            z = per_field.new_zeros(len(target))
            teacher_std = zero
    raw.update(
        field_a_objective=per_field[:, 0].mean(),
        field_u_objective=per_field[:, 1].mean(),
        online_full_latent_std=full_latent.float().std(correction=0),
        teacher_latent_std=teacher_std,
    )
    return raw, auxiliaries, per_field.detach(), z.detach()


def gradient_norm(grads, selected=None):
    values = [g for i, g in enumerate(grads) if g is not None and (selected is None or selected[i])]
    if not values:
        return 0.0
    norms = torch._foreach_norm(values, 2)
    return float(torch.linalg.vector_norm(torch.stack(norms).double()))


def balanced_gradients(named_parameters, field, auxiliaries, minimum_field_share=0.5):
    """Measure each weighted auxiliary separately, so cancellation cannot fake compliance.

    The guarantee is about pre-AdamW Euclidean gradient norms, both globally and
    on the online encoder. It is not a fraction of scalar loss or AdamW's final
    parameter displacement. Common final gradient clipping preserves the ratio.
    """
    named = [(n, p) for n, p in named_parameters if p.requires_grad]
    params = [p for _, p in named]
    encoder = [n.startswith("encoder.") for n, _ in named]
    active = [(n, value) for n, value in auxiliaries.items() if value.requires_grad]
    field_grads = torch.autograd.grad(field, params, allow_unused=True, retain_graph=bool(active))
    field_norm = gradient_norm(field_grads)
    field_encoder = gradient_norm(field_grads, encoder)
    auxiliary_sum = [None] * len(params)
    norms, encoder_norms = {}, {}
    for j, (name, value) in enumerate(active):
        grad = torch.autograd.grad(value, params, allow_unused=True, retain_graph=j < len(active) - 1)
        norms[name] = gradient_norm(grad)
        encoder_norms[name] = gradient_norm(grad, encoder)
        for i, g in enumerate(grad):
            if g is not None:
                if auxiliary_sum[i] is None:
                    # Some autograd results are expanded views with zero strides.
                    auxiliary_sum[i] = g.clone()
                else:
                    auxiliary_sum[i].add_(g)
        del grad
    aux_norm = sum(norms.values())
    aux_encoder = sum(encoder_norms.values())
    finite = [field_norm, field_encoder, aux_norm, aux_encoder, *norms.values()]
    if not all(math.isfinite(x) and x >= 0 for x in finite):
        raise FloatingPointError("Nonfinite component gradient; no update")
    alpha = min(
        1.0,
        field_norm / aux_norm if aux_norm else 1.0,
        field_encoder / aux_encoder if aux_encoder else 1.0,
    )
    # Roundoff margin only when actually reducing auxiliary contributions.
    if alpha < 1.0:
        alpha *= 1.0 - 1e-6
    total_aux_norm = gradient_norm(auxiliary_sum)
    dot = 0.0
    if field_norm and total_aux_norm:
        products = [
            (g * a).sum().double()
            for g, a in zip(field_grads, auxiliary_sum)
            if g is not None and a is not None
        ]
        dot = float(torch.stack(products).sum()) if products else 0.0
    cosine = dot / (field_norm * total_aux_norm) if field_norm and total_aux_norm else 0.0
    for p, f, a in zip(params, field_grads, auxiliary_sum):
        if f is not None:
            p.grad = f.clone() if a is None else f.add(a, alpha=alpha)
        else:
            p.grad = None if a is None else a.mul(alpha)
    share = field_norm / (field_norm + alpha * aux_norm) if field_norm + alpha * aux_norm else 1.0
    share_enc = (
        field_encoder / (field_encoder + alpha * aux_encoder) if field_encoder + alpha * aux_encoder else 1.0
    )
    if min(share, share_enc) < minimum_field_share - 1e-9:
        raise AssertionError("Field gradient-share guarantee violated")
    return {
        "field_grad_norm": field_norm,
        "aux_grad_norm_sum_before": aux_norm,
        "field_encoder_grad_norm": field_encoder,
        "aux_encoder_grad_norm_sum_before": aux_encoder,
        "aux_scale": alpha,
        "field_gradient_share": share,
        "field_encoder_gradient_share": share_enc,
        "gradient_capped": float(alpha < 1.0),
        "field_aux_cosine": max(-1.0, min(1.0, cosine)),
        **{"grad_norm_" + k: v for k, v in norms.items()},
    }

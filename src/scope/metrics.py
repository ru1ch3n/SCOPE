"""Physical evaluation metrics, including explicit undefined zero-target cases."""

import numpy as np
import torch
import torch.nn.functional as F
from .observations import known_region


def region_masks(geometry, device):
    disk = torch.from_numpy(np.stack([known_region(g) for g in geometry])).to(device)
    fluid = ~disk
    band = (F.max_pool2d(disk[:, None].float(), kernel_size=5, stride=1, padding=2)[:, 0] > 0) & fluid
    interior = fluid & ~band
    masks = {"full": torch.ones_like(disk), "fluid": fluid, "obstacle_band": band, "fluid_interior": interior}
    if any(bool((m.flatten(1).sum(1) == 0).any()) for m in masks.values()):
        raise ValueError("Empty geometric evaluation region")
    return masks


def metric_arrays(pred, target, norm, source_norms, zeros, geometry):
    device = pred.device
    scale = 2 * torch.as_tensor(norm["std"], device=device, dtype=torch.float64).view(1, 2, 1, 1)
    mean = torch.as_tensor(norm["mean"], device=device, dtype=torch.float64).view(1, 2, 1, 1)
    flags = torch.as_tensor(np.array(zeros, copy=True), device=device, dtype=torch.bool)
    truth = torch.where(
        flags[:, :, None, None], torch.zeros_like(target, dtype=torch.float64), target.double() * scale + mean
    )
    predicted = pred.double() * scale + mean
    assert bool(torch.isfinite(predicted).all())
    error = predicted - truth
    result = {}
    for region, mask in region_masks(geometry, device).items():
        count = mask.flatten(1).sum(1).double()[:, None]
        numerator = torch.linalg.vector_norm((error * mask[:, None]).flatten(2), dim=2)
        denom = (
            torch.as_tensor(np.array(source_norms, copy=True), device=device, dtype=torch.float64)
            if region == "full"
            else torch.linalg.vector_norm((truth * mask[:, None]).flatten(2), dim=2)
        )
        valid = (~flags) & (denom > 1e-10)
        # Invalid relative errors have placeholder zero plus an obligatory valid
        # mask; summaries NEVER include them. JSON uses null if no valid rows.
        relative = torch.where(
            valid, numerator / torch.where(valid, denom, torch.ones_like(denom)), torch.zeros_like(denom)
        )
        result[region + "_relative"] = relative.cpu().numpy()
        result[region + "_relative_valid"] = valid.cpu().numpy()
        result[region + "_absolute_rmse"] = (numerator / count.sqrt()).cpu().numpy()
    return result


def summarize_arrays(arrays, zero_flags):
    out = {}
    for region in ("full", "fluid", "obstacle_band", "fluid_interior"):
        out[region] = {}
        for ch, name in enumerate(("a", "u")):
            valid = arrays[region + "_relative_valid"][:, ch]
            rel = arrays[region + "_relative"][:, ch]
            absolute = arrays[region + "_absolute_rmse"][:, ch]
            zero = zero_flags[:, ch]
            out[region][name] = {
                "relative_l2_defined_target_mean_percent": float(rel[valid].mean() * 100)
                if valid.any()
                else None,
                "relative_l2_defined_records": int(valid.sum()),
                "relative_l2_undefined_ids": np.flatnonzero(~valid).tolist(),
                "absolute_rmse_all_records_mean": float(absolute.mean()),
                "zero_target_records": int(zero.sum()),
                "zero_target_ids": np.flatnonzero(zero).tolist(),
                "zero_target_absolute_rmse_mean": float(absolute[zero].mean()) if zero.any() else None,
            }
    return out


def darcy_errors(pred, truth, norm):
    scale = 2 * norm["std"][0]
    mean = norm["mean"][0]
    pred_high = (pred[:, 0].float() * scale + mean) > 7.5
    truth_high = (truth[:, 0].float() * scale + mean) > 7.5
    return (pred_high != truth_high).float().flatten(1).mean(1)

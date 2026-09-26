from dataclasses import asdict
import numpy as np
import pytest
import torch
from scope.model import ModelSpec, SCOPE
from scope.losses import physical_relative, field_objective, balanced_gradients
from scope.metrics import darcy_errors, metric_arrays, summarize_arrays
from scope.config import base_config
from scope.runtime import initialize


def test_exact_main_parameter_count():
    with torch.device("meta"):
        model = SCOPE()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 40040834
    assert not any(p.requires_grad for p in model.target_encoder.parameters())


def test_full_model_forward_and_mask_validation():
    initialize(31)
    model = SCOPE().eval()
    x = torch.zeros(1, 4, 128, 128)
    x[:, 1, ::8, ::8] = 1
    with torch.no_grad():
        y = model(x)
    assert y["fields"].shape == (1, 2, 128, 128)
    assert y["predicted_latent"].shape == (1, 256, 128)
    assert torch.isfinite(y["fields"]).all()
    x[:, 0, 1, 1] = 1
    with pytest.raises(ValueError, match="unobserved"):
        model(x)


def test_physical_relative_not_mse():
    target = torch.ones(2, 2, 4, 4)
    norm = {"mean": [0, 0], "std": [2, 3]}
    error = physical_relative(target * 1.25, target, norm)
    torch.testing.assert_close(error, torch.full((2, 2), 0.25))
    with pytest.raises(FloatingPointError, match="Zero"):
        physical_relative(target, target * 0, norm)


def test_gradient_guard_measures_components_before_cancellation():
    a = torch.nn.Parameter(torch.tensor([1.0]))
    b = torch.nn.Parameter(torch.tensor([1.0]))
    field = a.sum() + b.sum()
    aux = {"positive": 100 * a.sum() + 50 * b.sum(), "negative": -100 * a.sum() - 50 * b.sum()}
    metrics = balanced_gradients([("encoder.weight", a), ("decoder.weight", b)], field, aux)
    assert 0 < metrics["aux_scale"] < 1
    assert metrics["field_gradient_share"] >= 0.5
    assert metrics["field_encoder_gradient_share"] >= 0.5
    torch.testing.assert_close(a.grad, torch.ones(1))
    torch.testing.assert_close(b.grad, torch.ones(1))


def test_zero_target_extension_and_no_silent_eval_omission():
    c = base_config("ns-bounded")
    scale = 2 * torch.tensor(c["normalization"]["std"]).view(1, 2, 1, 1)
    mean = torch.tensor(c["normalization"]["mean"]).view(1, 2, 1, 1)
    target = (-mean / scale).expand(2, 2, 128, 128).clone()
    predicted = (1 - mean) / scale
    predicted = predicted.expand_as(target).clone()
    flags = torch.ones(2, 2, dtype=torch.bool)
    norms = torch.zeros(2, 2, dtype=torch.float64)
    errors = field_objective(predicted, target, c, norms, flags)
    expected = 1 / torch.tensor(c["zero_target_policy"]["train_rms_a_u"])
    torch.testing.assert_close(errors, expected.expand(2, 2))
    arrays = metric_arrays(
        predicted, target, c["normalization"], norms.numpy(), flags.numpy(), np.array([[50, 50, 10]] * 2)
    )
    summary = summarize_arrays(arrays, flags.numpy())
    assert summary["full"]["u"]["relative_l2_defined_target_mean_percent"] is None
    assert summary["full"]["u"]["relative_l2_undefined_ids"] == [0, 1]
    assert summary["full"]["u"]["absolute_rmse_all_records_mean"] == pytest.approx(1, abs=1e-6)


def test_darcy_binary_error_not_continuous_relative_error():
    norm = {"mean": [7.5, 0], "std": [4.5, 1]}
    target = torch.zeros(1, 2, 4, 4)
    target[:, 0] = 0.5
    pred = target.clone()
    pred[:, 0, 0] = -0.5
    assert darcy_errors(pred, target, norm).item() == 0.25

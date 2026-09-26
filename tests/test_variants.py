"""The ablations differ in the objective and nowhere else."""

import copy
from pathlib import Path
import pytest
import torch
from scope.config import PDES, VARIANTS, base_config, load_config, stage_config, variant_config
from scope.losses import objective
from scope.model import ModelSpec, SCOPE
from scope.runtime import initialize

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
# Naming, bookkeeping and the objective itself are allowed to differ; nothing else is.
DECLARED_DIFFERENCES = {
    "schema",
    "variant",
    "loss",
    "ablation",
    "pretrain_epochs",
    "minimum_field_gradient_share",
    "ema_start",
    "ema_end",
    "pretrained_initialization",
    "grounding_gradient",
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("pde", PDES)
def test_everything_but_the_objective_matches_the_reference(variant, pde):
    reference, c = base_config(pde), variant_config(pde, variant)
    assert set(c) - set(reference) <= DECLARED_DIFFERENCES
    for key, value in reference.items():
        if key not in DECLARED_DIFFERENCES:
            assert c[key] == value, key
    assert c["main_epochs"] == 500
    assert c["model_spec"] == reference["model_spec"] and c["seed"] == reference["seed"]
    assert c["optimizer"] == reference["optimizer"] and c["batch_size"] == 32
    assert c["evaluation"] == reference["evaluation"]


def test_full_is_the_unchanged_published_configuration():
    for pde in PDES:
        assert variant_config(pde, "full") == base_config(pde)
        assert "variant" not in variant_config(pde, "full")


def test_published_variant_configurations_load():
    files = sorted(CONFIGS.glob("variants/*.json"))
    assert len(files) == len(PDES) * (len(VARIANTS) - 1)
    for path in files:
        c = load_config(path)
        assert path.name == f"{c['variant']}-{c['pde']}.json"
    for pde in PDES:
        assert load_config(CONFIGS / f"{pde}.json") == base_config(pde)


def test_jepa_and_pretraining_follow_the_declared_weights():
    for variant in VARIANTS:
        c = variant_config("poisson", variant)
        main = stage_config(c, "main", "data")
        assert main["use_jepa"] == (c["loss"]["jepa_weight"] > 0)
        assert main["optimizer"]["warmup_epochs"] == 5
        if c["pretrain_epochs"]:
            pre = stage_config(c, "teacher_pretrain", "data")
            assert pre["use_jepa"] is False and pre["optimizer"]["warmup_epochs"] == 3
    assert variant_config("poisson", "fo")["pretrain_epochs"] == 0


def tiny(variant, stage="main"):
    c = variant_config("poisson", variant)
    c["model_spec"] = dict(
        c["model_spec"],
        encoder_width=16,
        encoder_depth=1,
        encoder_heads=2,
        predictor_width=16,
        predictor_depth=1,
        predictor_heads=2,
        anchor_channels=8,
        decoder_width=8,
    )
    c.update(train_records=8, batch_size=4, stage=stage, use_jepa=stage == "main" and c["loss"]["jepa_weight"] > 0)
    return c


def batch(c, n=4):
    initialize(11)
    model = SCOPE(ModelSpec(**c["model_spec"]))
    target = torch.randn(n, 2, 128, 128) * 0.5
    observed = torch.zeros(n, 4, 128, 128)
    observed[:, 1, ::8, ::8] = 1
    observed[:, 0] = target[:, 0] * observed[:, 1]
    return model, observed, target


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("full", {"jepa", "grounding", "variance"}),
        ("fjv", {"jepa", "variance"}),
        ("fg", {"grounding"}),
        ("fjvg", {"jepa", "grounding", "variance"}),
        ("fo", set()),
    ],
)
def test_main_auxiliaries_are_exactly_the_nonzero_weights(variant, expected):
    c = tiny(variant)
    model, observed, target = batch(c)
    raw, aux, _, _ = objective(model, observed, target, c)
    assert set(aux) == expected
    for name in expected:
        torch.testing.assert_close(aux[name], c["loss"][name + "_weight"] * raw[name])


@pytest.mark.parametrize("variant", ["fg", "fo"])
def test_a_variant_without_jepa_never_reads_the_teacher(variant):
    c = tiny(variant)
    model, observed, target = batch(c)
    calls = []
    model.target_encoder.register_forward_pre_hook(lambda *a: calls.append(1))
    objective(model, observed, target, c)
    assert calls == []


def test_fjvg_reports_the_full_grounding_value_but_trains_only_the_decoder():
    reference, decoder_only = tiny("full"), tiny("fjvg")
    assert reference["loss"] == decoder_only["loss"]
    model, observed, target = batch(decoder_only)
    raw_full, _, _, _ = objective(model, observed, target, reference)
    raw_dec, _, _, _ = objective(model, observed, target, decoder_only)
    torch.testing.assert_close(raw_full["grounding"], raw_dec["grounding"])
    encoder = list(model.encoder.parameters())
    predictor = list(model.predictor.parameters()) + list(model.observation_condition.parameters())
    grads = torch.autograd.grad(raw_dec["grounding"], encoder, allow_unused=True, retain_graph=True)
    assert all(g is None or not g.any() for g in grads)
    grads = torch.autograd.grad(raw_dec["grounding"], predictor, allow_unused=True, retain_graph=True)
    assert all(g is None or not g.any() for g in grads)
    grads = torch.autograd.grad(
        raw_dec["grounding"], list(model.decoder.parameters()), allow_unused=True, retain_graph=True
    )
    assert any(g is not None and g.any() for g in grads)
    grads = torch.autograd.grad(raw_dec["field"], encoder, allow_unused=True)
    assert any(g is not None and g.any() for g in grads)


def test_grounding_decodes_the_full_view_latent_and_bypasses_the_predictor():
    c = tiny("fg")
    model, observed, target = batch(c)
    raw, _, _, _ = objective(model, observed, target, c)
    from scope.losses import physical_relative

    expected = physical_relative(
        model.decoder(model.encoder(model.full_pair_view(target))), target, c["normalization"]
    ).mean()
    torch.testing.assert_close(raw["grounding"], expected)
    predictor = list(model.predictor.parameters()) + list(model.observation_condition.parameters())
    grads = torch.autograd.grad(raw["grounding"], predictor, allow_unused=True, retain_graph=True)
    assert all(g is None or not g.any() for g in grads)


def test_variance_is_measured_on_the_online_full_view_latent():
    c = tiny("fjv")
    model, observed, target = batch(c)
    target[:] = target[:1]  # a nearly constant batch keeps the hinge active
    raw, aux, _, _ = objective(model, observed, target, c)
    latent = model.encoder(model.full_pair_view(target))
    expected = torch.relu(0.1 - latent.float().flatten(1).std(dim=0, correction=0)).square().mean()
    torch.testing.assert_close(raw["variance"], expected, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(aux["variance"], c["loss"]["variance_weight"] * expected, rtol=1e-5, atol=1e-8)
    predictor = list(model.predictor.parameters()) + list(model.observation_condition.parameters())
    grads = torch.autograd.grad(raw["variance"], predictor, allow_unused=True, retain_graph=True)
    assert all(g is None or not g.any() for g in grads)


def test_field_only_never_touches_the_fully_visible_pair():
    c = tiny("fo")
    model, observed, target = batch(c)
    calls = []
    model.encoder.register_forward_pre_hook(lambda *a: calls.append(1))
    raw, aux, _, _ = objective(model, observed, target, c)
    assert aux == {} and len(calls) == 1
    assert float(raw["jepa"]) == 0 and float(raw["grounding"]) == 0 and float(raw["variance"]) == 0
    assert c["minimum_field_gradient_share"] == 1.0


def test_pretraining_carries_variance_only_when_the_variant_declares_it():
    for variant, expected in (("full", {"variance"}), ("fjv", {"variance"}), ("fjvg", {"variance"}), ("fg", set())):
        c = tiny(variant, stage="teacher_pretrain")
        model, observed, target = batch(c)
        _, aux, _, _ = objective(model, observed, target, c)
        assert set(aux) == expected, variant


def test_the_reference_objective_is_unchanged_by_the_variant_machinery():
    """`full` must still be the exact computation this release was frozen with."""
    c = tiny("full")
    model, observed, target = batch(c)
    raw, aux, per_field, per_jepa = objective(model, observed, target, c)
    full_view = model.full_pair_view(target)
    full_latent = model.encoder(full_view)
    variance = torch.relu(0.1 - full_latent.float().flatten(1).std(dim=0, correction=0)).square().mean()
    output = model(observed)
    from scope.losses import physical_relative

    expected_field = physical_relative(output["fields"], target, c["normalization"])
    grounding = physical_relative(model.decoder(full_latent), target, c["normalization"]).mean()
    with torch.no_grad():
        teacher = model.target_encoder(full_view)
    jepa = (output["predicted_latent"].float() - teacher.float()).square().mean((1, 2)).mean()
    torch.testing.assert_close(raw["field"], expected_field.mean())
    torch.testing.assert_close(raw["grounding"], grounding)
    torch.testing.assert_close(raw["variance"], variance)
    torch.testing.assert_close(raw["jepa"], jepa)
    assert list(aux) == ["jepa", "grounding", "variance"]
    assert per_field.shape == (len(target), 2) and per_jepa.shape == (len(target),)


def test_a_changed_configuration_is_refused(tmp_path):
    from scope.io import write_json

    c = copy.deepcopy(variant_config("poisson", "fjv"))
    c["loss"]["jepa_weight"] = 0.5
    write_json(tmp_path / "tampered.json", c)
    with pytest.raises(ValueError, match="unmodified published configuration"):
        load_config(tmp_path / "tampered.json")

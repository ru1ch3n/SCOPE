"""A probe trains one decoder on a frozen base, and nothing else."""

import numpy as np
import pytest
import torch
from scope import probe
from scope.config import stage_config, variant_config
from scope.dataset import HomogeneousDataset
from scope.evaluate import conditions
from scope.losses import physical_relative
from scope.model import ModelSpec, SCOPE
from scope.runtime import initialize
from scope.train import expected_trainable, stages_for


def test_declared_parameter_counts_are_exact():
    spec = ModelSpec()
    with torch.device("meta"):
        for size, declared in probe.DECLARED.items():
            assert sum(p.numel() for p in probe.ProbeDecoder(spec, size).parameters()) == declared
            assert probe.declared_count(spec, size) == declared
    assert probe.DECLARED["orig"] == probe.ORIGINAL_PARAMETERS
    assert probe.release_decoder_table(spec) == probe.SIZES["orig"] == probe.ORIGINAL
    for size in probe.REPORTED_SIZES:
        nominal = {"5m": 5e6, "10m": 10e6, "15m": 15e6}[size]
        assert abs(probe.DECLARED[size] - nominal) / nominal <= 0.002
    with pytest.raises(ValueError, match="unknown probe size"):
        probe.ProbeDecoder(spec, "3m")


def test_attaching_freezes_the_whole_base():
    with torch.device("meta"):
        model = SCOPE()
        report = probe.attach(model, "10m")
    assert report["replaced_decoder_parameters"] == probe.ORIGINAL_PARAMETERS
    assert report["trainable_parameters"] == probe.DECLARED["10m"]
    assert report["frozen_parameters"] == probe.FROZEN_PARAMETERS == 78_027_776
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"decoder"}
    probe.assert_frozen_base(model)


def tiny_probe(observation, size="orig"):
    base = variant_config("poisson", "fjv")
    base["model_spec"] = dict(
        base["model_spec"],
        encoder_width=16,
        encoder_depth=1,
        encoder_heads=2,
        predictor_width=16,
        predictor_depth=1,
        predictor_heads=2,
        anchor_channels=8,
        decoder_width=8,
    )
    record = {
        "run": "runs/fjv-poisson",
        "variant": "fjv",
        "pde": "poisson",
        "epoch": 500.0,
        "checkpoint_sha256": "0" * 64,
        "science_sha256": "1" * 64,
    }
    c = probe.probe_config(base, record, size, observation)
    c.update(stage="probe", use_jepa=False, train_records=8, batch_size=4)
    initialize(5)
    model = SCOPE(ModelSpec(**c["model_spec"]))
    probe.attach(model, size)
    target = torch.randn(4, 2, 128, 128) * 0.5
    observed = torch.zeros(4, 4, 128, 128)
    observed[:, 1, ::8, ::8] = 1
    observed[:, 0] = target[:, 0] * observed[:, 1]
    return c, model, observed, target


def test_probe_configuration_declares_a_decoder_only_experiment():
    c, _, _, _ = tiny_probe("uniform500", "5m")
    assert c["variant"] == "probe" and c["probe_size"] == "5m"
    assert c["probe_observation"] == "uniform500" and c["probe_epochs"] == 100
    assert all(v == 0.0 for v in c["loss"].values())
    assert c["minimum_field_gradient_share"] == 1.0
    assert c["pretrained_initialization"] is False and c["ema_start"] == c["ema_end"] == 0.0
    assert c["main_epochs"] == 100 and c["optimizer"]["warmup_epochs"] == 5
    assert stages_for(c) == ("probe",)
    assert expected_trainable(c) == probe.DECLARED["5m"]
    with pytest.raises(ValueError, match="unknown probe observation"):
        probe.probe_config(variant_config("poisson", "fjv"), {}, "5m", "uniform100")


def test_full_observation_probe_decodes_the_encoder_latent():
    c, model, observed, target = tiny_probe("full")
    raw, aux, per_field, per_jepa = probe.probe_objective(model, observed, target, c)
    expected = physical_relative(
        model.decoder(model.encoder(model.full_pair_view(target))), target, c["normalization"]
    )
    torch.testing.assert_close(raw["field"], expected.mean())
    assert aux == {} and per_field.shape == (4, 2) and not per_jepa.any()


def test_sparse_probe_decodes_the_predictor_latent():
    c, model, observed, target = tiny_probe("uniform500")
    raw, aux, _, _ = probe.probe_objective(model, observed, target, c)
    with torch.no_grad():
        latent = model(observed)["predicted_latent"]
    expected = physical_relative(model.decoder(latent), target, c["normalization"])
    torch.testing.assert_close(raw["field"], expected.mean())
    assert aux == {}


@pytest.mark.parametrize("observation", probe.OBSERVATIONS)
def test_only_the_decoder_receives_a_gradient(observation):
    c, model, observed, target = tiny_probe(observation)
    raw, _, _, _ = probe.probe_objective(model, observed, target, c)
    raw["field"].backward()
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("decoder."))
    assert any(p.grad is not None and p.grad.any() for p in model.decoder.parameters())


@pytest.mark.parametrize("observation", probe.OBSERVATIONS)
def test_objective_rejects_an_unfrozen_base(observation):
    """The sparse latent is built under no_grad, so the freeze is asserted directly."""
    c, model, observed, target = tiny_probe(observation)
    model.encoder.requires_grad_(True)
    with pytest.raises(AssertionError, match="frozen"):
        probe.probe_objective(model, observed, target, c)


def test_a_probe_carries_no_auxiliary_objective():
    c, model, observed, target = tiny_probe("full")
    c["loss"]["grounding_weight"] = 0.25
    with pytest.raises(ValueError, match="no auxiliary objective"):
        probe.probe_objective(model, observed, target, c)


def write_tiny_data(root):
    folder = root / "poisson"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    fields = rng.normal(0, 0.5, (8, 2, 128, 128)).astype(np.float32)
    np.save(folder / "train-f32.npy", fields, allow_pickle=False)
    np.save(folder / "train-physical-norms.npy", np.ones((8, 2)), allow_pickle=False)
    np.save(folder / "train-zero-targets.npy", np.zeros((8, 2), np.bool_), allow_pickle=False)
    return folder


def probe_dataset(tmp_path, observation):
    write_tiny_data(tmp_path)
    base = variant_config("poisson", "fjv")
    base.update(train_records=8, batch_size=4, probe_epochs=2, variant="probe", probe_observation=observation)
    return HomogeneousDataset(stage_config(base, "probe", tmp_path))


def test_sparse_probe_trains_on_uniform_500_alternating_tasks(tmp_path):
    dataset = probe_dataset(tmp_path, "uniform500")
    for draw in range(16):
        observed, target, meta, _, _ = dataset[draw]
        task, budget, family = int(meta[2]), int(meta[3]), int(meta[4])
        assert (task, budget, family) == ((draw // 4) % 2, 500, 0)
        visible = observed[[1, 3]].sum((1, 2)).tolist()
        assert visible == ([500, 0] if task == 0 else [0, 500])


def test_full_observation_probe_trains_on_the_fully_visible_pair(tmp_path):
    dataset = probe_dataset(tmp_path, "full")
    observed, target, meta, _, _ = dataset[0]
    assert [int(meta[2]), int(meta[3]), int(meta[4])] == [2, 16384, 0]
    assert bool((observed[[1, 3]] == 1).all())
    torch.testing.assert_close(observed[[0, 2]], target)


def test_full_observation_protocol_is_one_comparable_condition():
    assert conditions("poisson", "full-observation") == [("joint", "full", 16384)]
    assert conditions("ns-bounded", "full-observation") == [("joint", "full", 16384)]

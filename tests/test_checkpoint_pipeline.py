import copy
from pathlib import Path
import torch
import pytest
from scope import checkpoint
from scope.config import stage_config
from scope.dataset import HomogeneousDataset
from scope.evaluate import evaluate
from scope.io import read_json
from scope.optimization import one_update
from scope.runtime import equal_tree, cpu_tree
from scope.train import smoke_data, initialize_stage, run, STOP


def test_strict_resume_next_update_and_mismatch(tmp_path):
    data_root = tmp_path / "data"
    base = smoke_data(data_root)
    c = stage_config(base, "teacher_pretrain", data_root)
    device = torch.device("cpu")
    model, opt, _ = initialize_stage(c, tmp_path, device)
    ds = HomogeneousDataset(c)
    batch = [torch.stack([ds[i][j] for i in range(4)]) for j in range(5)]
    obs, target, _, norms, zeros = batch
    one_update(model, opt, obs, target, c, 0, device, norms, zeros)
    directory = tmp_path / "checkpoints"
    record = checkpoint.save(directory, model, opt, c, 4, obs[:2])
    first = one_update(model, opt, obs, target, c, 4, device, norms, zeros)
    model_state, opt_state = cpu_tree(model.state_dict()), cpu_tree(opt.state_dict())
    path, receipt = checkpoint.current(directory)
    assert receipt == record
    checkpoint.load(path, model, opt, c, device)
    second = one_update(model, opt, obs, target, c, 4, device, norms, zeros)
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])
    assert equal_tree(model_state, cpu_tree(model.state_dict()))
    assert equal_tree(opt_state, cpu_tree(opt.state_dict()))
    wrong = copy.deepcopy(c)
    wrong["loss"]["jepa_weight"] = 2
    with pytest.raises(ValueError, match="configuration mismatch"):
        checkpoint.load(path, model, opt, wrong, device)


def test_smoke_pretrain_main_resume_evaluate(tmp_path):
    output = tmp_path / "run"
    data_root = output / "synthetic-data"
    base = smoke_data(data_root)
    STOP["signal"] = None
    result = run(base, data_root, output, torch.device("cpu"), workers=0, max_updates=2)
    assert result["status"] == "paused"
    result = run(base, data_root, output, torch.device("cpu"), workers=0, resume=True)
    assert result["status"] == "complete" and result["epoch"] == 2
    transfer = read_json(output / "main/initialization.json")["transfer"]
    assert transfer["predictor_condition_unchanged"] and transfer["encoder_decoder_transferred"]
    results_path = evaluate(output, data_root, torch.device("cpu"), batch_size=4)
    report = read_json(results_path)
    assert report["smoke_test"] and report["training_complete"]
    assert len(report["rows"]) == 2
    assert report["strict_same_runtime_sentinel"]
    assert not report["test_used_for_checkpoint_selection"]
    with pytest.raises(FileExistsError):
        run(base, data_root, output, torch.device("cpu"), workers=0)


def test_epoch_row_permutations_without_tail_loss(tmp_path):
    base = smoke_data(tmp_path)
    c = stage_config(base, "main", tmp_path)
    ds = HomogeneousDataset(c)
    for start in (0, base["train_records"]):
        assert sorted(ds.row_id(i) for i in range(start, start + base["train_records"])) == list(
            range(base["train_records"])
        )
    resumed = HomogeneousDataset(c, start=8)
    for i in range(4):
        for first, second in zip(ds[i + 8], resumed[i]):
            assert torch.equal(first, second)

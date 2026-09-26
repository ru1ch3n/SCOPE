"""Each epoch visits all training rows once; conditions are per optimizer batch."""

import numpy as np
import torch
from torch.utils.data import Dataset
from .observations import TASKS, FAMILIES, condition, observation_view, seed_for


class HomogeneousDataset(Dataset):
    def __init__(self, config, start=0, stop=None):
        self.c = config
        self.start = int(start)
        self.stop = int(config["total_draws"] if stop is None else stop)
        assert self.start % config["batch_size"] == 0
        self.array = None
        self.geometry = None
        self.norms = None
        self.zeros = None
        self.permutations = {}

    def __len__(self):
        return self.stop - self.start

    def row_id(self, draw):
        epoch, offset = divmod(draw, self.c["train_records"])
        if epoch not in self.permutations:
            if len(self.permutations) > 2:
                self.permutations.clear()
            self.permutations[epoch] = np.random.default_rng(
                seed_for(self.c["seed"], "row-order", epoch)
            ).permutation(self.c["train_records"])
        return int(self.permutations[epoch][offset])

    def __getitem__(self, index):
        draw = self.start + int(index)
        if not self.start <= draw < self.stop:
            raise IndexError(index)
        if self.array is None:
            self.array = np.load(self.c["train_path"], mmap_mode="r", allow_pickle=False)
            assert self.array.shape == (self.c["train_records"], 2, 128, 128)
            assert self.array.dtype == np.float32
            self.norms = np.load(self.c["train_norms_path"], mmap_mode="r", allow_pickle=False)
            self.zeros = np.load(self.c["train_zero_targets_path"], mmap_mode="r", allow_pickle=False)
            if self.c["pde"] == "ns-bounded":
                self.geometry = np.load(self.c["train_geometry_path"], allow_pickle=False)
        row = self.row_id(draw)
        target = np.array(self.array[row], dtype=np.float32, copy=True)
        stage = self.c["stage"]
        full_view_stage = stage == "teacher_pretrain" or (
            stage == "probe" and self.c["probe_observation"] == "full"
        )
        if full_view_stage:
            task, family, budget = 2, 0, 16384
            ones = np.ones((128, 128), np.float32)
            observed = np.stack((target[0], ones, target[1], ones))
        else:
            if stage == "probe":
                # A sparse probe trains on one fixed condition: uniform / 500,
                # forward and inverse alternating by optimizer batch. This is a
                # pure function of the batch index, so a resumed probe repeats
                # the same schedule exactly.
                task, family, budget = (draw // self.c["batch_size"]) % 2, 0, 500
            else:
                task, family, budget = condition(draw // self.c["batch_size"], self.c["seed"])
            observed = observation_view(
                target,
                TASKS[task],
                budget,
                FAMILIES[family],
                seed_for(self.c["seed"], "position", draw, row),
                None if self.geometry is None else self.geometry[row],
            )
        meta = np.array([draw, row, task, budget, family], dtype=np.int64)
        return (
            torch.from_numpy(observed),
            torch.from_numpy(target),
            torch.from_numpy(meta),
            torch.from_numpy(np.array(self.norms[row], copy=True)),
            torch.from_numpy(np.array(self.zeros[row], copy=True)),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["array"] = None
        state["geometry"] = None
        state["norms"] = None
        state["zeros"] = None
        state["permutations"] = {}
        return state


def assert_batch(meta, batch_size):
    assert len(meta) == batch_size
    assert torch.equal(meta[:, 0], torch.arange(int(meta[0, 0]), int(meta[0, 0]) + batch_size))
    assert torch.all(meta[:, 2:] == meta[0, 2:])
    assert int(meta[0, 0]) % batch_size == 0

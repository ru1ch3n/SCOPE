"""Deterministic runtime and checkpoint helpers; no cluster dependencies."""

import contextlib
import hashlib
import os
import random
import shutil
import time
import numpy as np
import torch


def initialize(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@torch.no_grad()
def sentinel(model, observed, device):
    was_training = model.training
    model.eval()
    with autocast(device):
        result = model(observed.to(device))["fields"].float().cpu()
    model.train(was_training)
    return result


def atomic_link(source, target):
    temporary = target.with_name(target.name + f".{time.time_ns()}.tmp")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        x = tensor.detach().cpu().contiguous()
        h.update(str((str(x.dtype), tuple(x.shape))).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_tree(v) for v in x)
    return x


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b

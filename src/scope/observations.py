"""Geometry-only, exact-budget observation families and homogeneous batch schedule."""

from functools import lru_cache
import hashlib
import numpy as np

TASKS = ("forward", "inverse", "joint")
FAMILIES = ("uniform", "low_resolution", "random_lines", "cluster", "block")
BUDGETS = (500, 1024, 2048, 4096, 8192, 12288, 16384)
GRID_SHAPES = {
    500: (20, 25),
    1024: (32, 32),
    2048: (32, 64),
    4096: (64, 64),
    8192: (64, 128),
    12288: (96, 128),
    16384: (128, 128),
}
LARGE_BUDGETS = (4915, 9830)  # round(30% and 60% of 128^2)
BUDGET_SLOTS = (500, 500, 1024, 2048, 4096, 8192, 12288, 16384)


def seed_for(seed, *parts):
    s = "|".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little")


@lru_cache(maxsize=2048)
def shuffled(seed, name, cycle, values):
    return tuple(np.random.default_rng(seed_for(seed, name, cycle)).permutation(values).tolist())


def condition(batch_index, seed):
    """Exactly one of each family per five batches; 40/40/20 tasks per family per 25."""
    group, offset = divmod(int(batch_index), 5)
    family = int(shuffled(seed, "family", group, tuple(range(5)))[offset])
    task = int(shuffled(seed, f"task-{family}", group // 5, (0, 0, 1, 1, 2))[group % 5])
    slots = LARGE_BUDGETS if FAMILIES[family] in ("block", "random_lines") else BUDGET_SLOTS
    budget = int(shuffled(seed, f"budget-{family}", group // len(slots), slots)[group % len(slots)])
    return task, family, budget


@lru_cache(maxsize=1)
def coordinates():
    y, x = np.indices((128, 128), dtype=np.float64)
    return np.stack((y.ravel(), x.ravel()), axis=1) / 127.0


def cluster_geometry(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    centers = []
    # Greedy farthest-of-64 sampling with a hard separation floor.
    centers.append(rng.uniform(0.08, 0.92, 2))
    for _ in range(n - 1):
        candidates = rng.uniform(0.08, 0.92, (64, 2))
        distance = np.linalg.norm(candidates[:, None, :] - np.asarray(centers)[None, :, :], axis=2).min(1)
        candidate = candidates[int(np.argmax(distance))]
        if distance.max() < 0.30:
            raise AssertionError("Cluster centers too concentrated")
        centers.append(candidate)
    widths = rng.uniform(0.12, 0.20, n)
    return np.asarray(centers), widths


def mask(family, budget, seed):
    if family not in FAMILIES or not 1 <= budget <= 16384:
        raise ValueError("Invalid observation condition")
    if family in ("block", "random_lines") and budget not in LARGE_BUDGETS:
        raise ValueError("Block/lines are restricted to 30% and 60%")
    rng = np.random.default_rng(seed)
    if family == "low_resolution":
        if budget not in GRID_SHAPES:
            raise ValueError("No exact regular-grid shape for this budget")
        ny, nx = GRID_SHAPES[budget]
        if rng.integers(2):
            ny, nx = nx, ny
        # Cartesian grid of original pixels, nearly equal integer spacing, no interpolation.
        yy = np.floor((np.arange(ny) + rng.uniform(0, 1)) * 128 / ny).astype(np.int64)
        xx = np.floor((np.arange(nx) + rng.uniform(0, 1)) * 128 / nx).astype(np.int64)
        selected = (yy[:, None] * 128 + xx[None, :]).ravel()
    elif family == "uniform":
        selected = rng.permutation(16384)[:budget]
    elif family == "random_lines":
        # Random whole rows OR columns; at most one final partial line for exact cardinality.
        line_order = rng.permutation(128)
        along = rng.permutation(128)
        order = (line_order[:, None] * 128 + along[None, :]).ravel()
        if rng.integers(2):
            order = (order % 128) * 128 + order // 128
        selected = order[:budget]
    else:
        xy = coordinates()
        if family == "cluster":
            centers, widths = cluster_geometry(seed_for(seed, "centers"))
            distance = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            # Broad, equal-weight components, no field-dependent point choice.
            weights = np.exp(-distance / (2 * widths[None, :] ** 2)).mean(1)
            score = -np.log(np.maximum(rng.random(16384), np.finfo(float).tiny)) / np.maximum(weights, 1e-12)
        else:
            center = rng.uniform(0.20, 0.80, 2)
            aspect = rng.uniform(2 / 3, 3 / 2)
            delta = np.abs(xy - center)
            score = np.maximum(delta[:, 0] * np.sqrt(aspect), delta[:, 1] / np.sqrt(aspect))
        selected = np.lexsort((rng.random(16384), score))[:budget]
    result = np.zeros(16384, dtype=np.float32)
    result[selected] = 1
    assert int(result.sum()) == budget
    return result.reshape(128, 128)


def plain_observation_view(fields, task, budget, family, seed):
    fields = np.asarray(fields, dtype=np.float32)
    if fields.shape != (2, 128, 128) or task not in TASKS:
        raise ValueError("Invalid fields/task")
    masks = np.zeros_like(fields)
    for ch in range(2):
        if (task == "forward" and ch == 1) or (task == "inverse" and ch == 0):
            continue
        masks[ch] = mask(family, budget, seed_for(seed, "channel", ch))
    return np.stack((fields[0] * masks[0], masks[0], fields[1] * masks[1], masks[1]))


def known_region(geometry):
    """Published geometry only; axes match the original integer-grid sampler.

    The generic binary-mask path can represent non-circular internal obstacles.
    This campaign's actual source supplies one distinct cx/cy/r triple per row.
    This is the benchmark's known disk, NOT a claim that every disk cell is zero.
    """
    g = np.asarray(geometry)
    if g.shape == (128, 128) and g.dtype == np.bool_:
        if g.all():
            raise ValueError("No fluid domain remains")
        return g.copy()
    if g.shape != (3,) or not np.isfinite(g).all():
        raise ValueError("Published per-record geometry required")
    cx, cy, r = g
    if r <= 0 or min(cx - r, cy - r) < 0 or max(cx + r, cy + r) >= 128:
        raise ValueError("Cylinder geometry outside domain")
    i, j = np.indices((128, 128))
    return (i - cx) ** 2 + (j - cy) ** 2 <= r * r


def observation_view(fields, task, budget, family, seed, geometry=None):
    if geometry is None:
        return plain_observation_view(fields, task, budget, family, seed)
    fields = np.asarray(fields, dtype=np.float32)
    if fields.shape != (2, 128, 128) or task not in TASKS:
        raise ValueError("Invalid fields/task")
    disk = known_region(geometry)
    masks = np.repeat(disk[None, :, :], 2, axis=0).astype(np.float32)
    for ch in range(2):
        if (task == "forward" and ch == 1) or (task == "inverse" and ch == 0):
            continue
        if family == "benchmark_uniform":
            if budget != 100:
                raise ValueError("Benchmark sparse probability is 1%")
            sparse = np.random.default_rng(seed_for(seed, "channel", ch)).random((128, 128)) < 0.01
        else:
            # Preserve true Cartesian grids: remove overlap with the known disk,
            # never insert off-grid replacement points. Nominal full-grid count
            # is homogeneous; actual extra-fluid count is measured per record.
            sparse = mask(family, budget, seed_for(seed, "channel", ch)) > 0
        masks[ch] = np.logical_or(disk, sparse).astype(np.float32)
    return np.stack((np.where(masks[0], fields[0], 0), masks[0], np.where(masks[1], fields[1], 0), masks[1]))


def observation_counts(observed, geometry):
    disk = known_region(geometry)
    m = np.asarray(observed)[[1, 3]] > 0
    assert np.all(m[:, disk])
    return {
        "known_disk_points": int(disk.sum()),
        "fluid_domain_points": int((~disk).sum()),
        "extra_fluid_points_a_u": m[:, ~disk].sum(1).astype(int).tolist(),
        "total_unique_points_a_u": m.sum((1, 2)).astype(int).tolist(),
    }


def evaluation_conditions():
    return [
        (task, family, budget)
        for task in TASKS
        for family in FAMILIES
        for budget in (LARGE_BUDGETS if family in ("block", "random_lines") else BUDGETS)
    ]

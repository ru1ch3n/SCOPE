import numpy as np
import pytest
from scope.observations import (
    FAMILIES,
    BUDGETS,
    LARGE_BUDGETS,
    GRID_SHAPES,
    condition,
    mask,
    observation_view,
    known_region,
    observation_counts,
    cluster_geometry,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_counts_determinism_and_binary_masks(family):
    budgets = LARGE_BUDGETS if family in ("block", "random_lines") else BUDGETS
    for budget in budgets:
        result = mask(family, budget, 173)
        assert result.shape == (128, 128)
        assert np.isin(result, [0, 1]).all()
        assert result.sum() == budget
        np.testing.assert_array_equal(result, mask(family, budget, 173))


def test_homogeneous_balanced_schedule():
    conditions = np.array([condition(i, 20260913) for i in range(200)])
    for group in conditions.reshape(-1, 5, 3):
        assert sorted(group[:, 1]) == list(range(5))
    for family in range(5):
        selected = conditions[conditions[:, 1] == family]
        assert np.bincount(selected[:, 0], minlength=3).tolist() == [16, 16, 8]
        if FAMILIES[family] in ("block", "random_lines"):
            assert np.unique(selected[:, 2], return_counts=True)[1].tolist() == [20, 20]
        else:
            assert np.unique(selected[:, 2], return_counts=True)[1].tolist() == [10, 5, 5, 5, 5, 5, 5]


@pytest.mark.parametrize("task", ("forward", "inverse", "joint"))
def test_no_hidden_field_leakage(task):
    fields = np.random.default_rng(17).normal(size=(2, 128, 128)).astype(np.float32)
    observed = observation_view(fields, task, 500, "uniform", 91)
    altered = fields.copy()
    altered[observed[[1, 3]] == 0] = 12345
    np.testing.assert_array_equal(observed, observation_view(altered, task, 500, "uniform", 91))
    assert (observed[[0, 2]] * (1 - observed[[1, 3]]) == 0).all()
    expected = [500, 0] if task == "forward" else ([0, 500] if task == "inverse" else [500, 500])
    assert observed[[1, 3]].sum((1, 2)).tolist() == expected


def test_regular_grid_and_broad_clusters():
    for budget, shape in GRID_SHAPES.items():
        selected = mask("low_resolution", budget, 92) > 0
        rows, cols = selected.any(1), selected.any(0)
        assert sorted((rows.sum(), cols.sum())) == sorted(shape)
        np.testing.assert_array_equal(selected, rows[:, None] & cols[None, :])
    for seed in range(20):
        centers, widths = cluster_geometry(seed)
        assert 2 <= len(centers) <= 4
        assert np.all((widths >= 0.12) & (widths <= 0.20))
        distances = np.linalg.norm(centers[:, None] - centers[None], axis=2)
        assert np.all(distances[np.triu_indices(len(centers), 1)] >= 0.30)


def test_cylinder_is_known_internal_region_not_forced_zero():
    geometry = np.array([45, 61, 12])
    fields = np.full((2, 128, 128), 7, dtype=np.float32)
    disk = known_region(geometry)
    observed = observation_view(fields, "forward", 500, "low_resolution", 19, geometry)
    assert np.all(observed[[1, 3]][:, disk] == 1)
    assert np.all(observed[[0, 2]][:, disk] == 7)
    assert observed[3].sum() == disk.sum()
    assert not observed[3, 0].any()
    counts = observation_counts(observed, geometry)
    assert counts["extra_fluid_points_a_u"][0] <= 500
    assert counts["extra_fluid_points_a_u"][1] == 0
    # No arbitrary off-grid replacements when a regular grid overlaps the disk.
    from scope.observations import seed_for

    grid = mask("low_resolution", 500, seed_for(19, "channel", 0)) > 0
    np.testing.assert_array_equal(observed[1] > 0, disk | grid)

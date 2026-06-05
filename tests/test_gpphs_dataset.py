"""
Tests for prepare_gpphs_data() and get_gpphs_dataloaders() in neuromancer/dataset.py
Run with:  python tests/test_gpphs_dataset.py
"""

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from neuromancer.dataset import prepare_gpphs_data, get_gpphs_dataloaders, StaticDataset


# ---------------------------------------------------------------------------
# prepare_gpphs_data — returns (train_ds, dev_ds, test_ds, batch_size)
# for use as a LitDataModule data_setup_function
# ---------------------------------------------------------------------------

def test_prepare_returns_datasets():
    """prepare_gpphs_data must return (StaticDataset x3, int) for LitDataModule."""
    T, nx = 300, 2
    x     = np.random.randn(T, nx)
    x_dot = np.random.randn(T, nx)

    train_ds, dev_ds, test_ds, bs = prepare_gpphs_data(x, x_dot, batch_size=32)

    assert isinstance(train_ds, StaticDataset), f"train_ds is {type(train_ds)}"
    assert isinstance(dev_ds,   StaticDataset), f"dev_ds is {type(dev_ds)}"
    assert isinstance(test_ds,  StaticDataset), f"test_ds is {type(test_ds)}"
    assert bs == 32,                            f"batch_size wrong: {bs}"
    assert train_ds.name == 'train'
    assert dev_ds.name   == 'dev'
    assert test_ds.name  == 'test'
    print("PASSED  test_prepare_returns_datasets")


def test_prepare_litdatamodule_unpack():
    """LitDataModule.setup() unpacks exactly 4 values — verify the signature."""
    T, nx = 300, 2
    x     = np.random.randn(T, nx)
    x_dot = np.random.randn(T, nx)

    result = prepare_gpphs_data(x, x_dot, batch_size=16)
    assert len(result) == 4, f"Expected 4-tuple, got length {len(result)}"
    print("PASSED  test_prepare_litdatamodule_unpack")


# ---------------------------------------------------------------------------
# get_gpphs_dataloaders — returns (loaders x3), dims
# for standalone use outside LitDataModule
# ---------------------------------------------------------------------------

def test_basic_with_control():
    """2D inputs: x, x_dot, u all provided."""
    T, nx, nu = 300, 2, 1
    x     = np.random.randn(T, nx)
    x_dot = np.random.randn(T, nx)
    u     = np.random.randn(T, nu)

    (train, dev, test), dims = get_gpphs_dataloaders(x, x_dot, u, batch_size=32)
    batch = next(iter(train))

    assert batch['X'].shape    == (32, nx), f"X shape wrong: {batch['X'].shape}"
    assert batch['Xdot'].shape == (32, nx), f"Xdot shape wrong: {batch['Xdot'].shape}"
    assert batch['U'].shape    == (32, nu), f"U shape wrong: {batch['U'].shape}"
    assert batch['name'] == 'train',        f"name wrong: {batch['name']}"
    print("PASSED  test_basic_with_control")


def test_1d_inputs_auto_promoted():
    """1D inputs (T,) should be silently promoted to (T, 1)."""
    T = 300
    x     = np.random.randn(T)
    x_dot = np.random.randn(T)
    u     = np.random.randn(T)

    (train, _, _), _ = get_gpphs_dataloaders(x, x_dot, u, batch_size=16)
    batch = next(iter(train))

    assert batch['X'].shape    == (16, 1), f"X shape wrong: {batch['X'].shape}"
    assert batch['Xdot'].shape == (16, 1), f"Xdot shape wrong: {batch['Xdot'].shape}"
    assert batch['U'].shape    == (16, 1), f"U shape wrong: {batch['U'].shape}"
    print("PASSED  test_1d_inputs_auto_promoted")


def test_autonomous_no_u():
    """Autonomous system: u=None means no 'U' key in batch."""
    T, nx = 300, 2
    x     = np.random.randn(T, nx)
    x_dot = np.random.randn(T, nx)

    (train, _, _), dims = get_gpphs_dataloaders(x, x_dot, u=None, batch_size=32)
    batch = next(iter(train))

    assert 'U' not in batch, "U should not be in batch for autonomous system"
    assert 'X'    in batch,  "X missing from batch"
    assert 'Xdot' in batch,  "Xdot missing from batch"
    print("PASSED  test_autonomous_no_u")


def test_custom_split_ratio():
    """split_ratio=[60, 20] → 60% train, 20% dev, 20% test."""
    T, nx = 600, 2
    x     = np.random.randn(T, nx)
    x_dot = np.random.randn(T, nx)

    (train, dev, test), _ = get_gpphs_dataloaders(x, x_dot, split_ratio=[60.0, 20.0])

    total = len(train.dataset) + len(dev.dataset) + len(test.dataset)
    assert total == T, f"Split lost samples: {total} != {T}"
    print(f"PASSED  test_custom_split_ratio  "
          f"(train={len(train.dataset)}, dev={len(dev.dataset)}, test={len(test.dataset)})")


def test_default_split_thirds():
    """Default split_ratio=None → three roughly equal thirds."""
    T, nx = 300, 2
    x     = np.random.randn(T, nx)
    x_dot = np.random.randn(T, nx)

    (train, dev, test), _ = get_gpphs_dataloaders(x, x_dot)

    total = len(train.dataset) + len(dev.dataset) + len(test.dataset)
    assert total == T, f"Split lost samples: {total} != {T}"
    for ds, name in [(train, 'train'), (dev, 'dev'), (test, 'test')]:
        assert abs(len(ds.dataset) - T // 3) <= 2, \
            f"{name} split size unexpected: {len(ds.dataset)}"
    print("PASSED  test_default_split_thirds")


def test_output_dtype_float32():
    """Output tensors should always be float32, even if inputs are float64."""
    T, nx = 300, 2
    x     = np.random.randn(T, nx)   # float64 by default
    x_dot = np.random.randn(T, nx)

    (train, _, _), _ = get_gpphs_dataloaders(x, x_dot)
    batch = next(iter(train))

    assert batch['X'].dtype    == torch.float32, f"X dtype: {batch['X'].dtype}"
    assert batch['Xdot'].dtype == torch.float32, f"Xdot dtype: {batch['Xdot'].dtype}"
    print("PASSED  test_output_dtype_float32")


# ---------------------------------------------------------------------------
# Validation — shared by both functions since prepare_gpphs_data does the work
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    """x and x_dot with different shapes should raise an AssertionError."""
    x     = np.random.randn(300, 2)
    x_dot = np.random.randn(300, 3)   # wrong nx
    try:
        prepare_gpphs_data(x, x_dot)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("PASSED  test_shape_mismatch_raises")


def test_u_length_mismatch_raises():
    """u with wrong number of rows should raise an AssertionError."""
    x     = np.random.randn(300, 2)
    x_dot = np.random.randn(300, 2)
    u     = np.random.randn(200, 1)   # wrong T
    try:
        prepare_gpphs_data(x, x_dot, u)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("PASSED  test_u_length_mismatch_raises")


if __name__ == '__main__':
    print("Running GP-PHS dataset tests...\n")
    test_prepare_returns_datasets()
    test_prepare_litdatamodule_unpack()
    test_basic_with_control()
    test_1d_inputs_auto_promoted()
    test_autonomous_no_u()
    test_custom_split_ratio()
    test_default_split_thirds()
    test_output_dtype_float32()
    test_shape_mismatch_raises()
    test_u_length_mismatch_raises()
    print("\nAll tests passed!")
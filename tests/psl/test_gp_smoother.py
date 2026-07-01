"""
Tests for gp_smooth in neuromancer.psl.gp_smoother.

Each test has one job. Comments explain *why* a test exists, not what the code does.
All tests use n_iter=5 unless the test is specifically about training behavior.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import numpy as np
import pytest
from neuromancer.psl.gp_smoother import gp_smooth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T = 15  # short enough to be fast, long enough for Cholesky not to be trivial
t = np.linspace(0.0, 1.0, T)


def noisy(nx, seed=0):
    """Return (T, nx) random observations."""
    return np.random.default_rng(seed).standard_normal((T, nx))


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

def test_returns_numpy_arrays():
    """All three outputs must be plain numpy arrays, not tensors."""
    results = gp_smooth(t, noisy(1), n_iter=5)
    for arr in results:
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nx", [1, 2, 4])
def test_output_shapes(nx):
    """Each output has shape (T, nx) regardless of the number of state dims."""
    x_s, xd_s, xd_v = gp_smooth(t, noisy(nx), n_iter=5)
    for arr in (x_s, xd_s, xd_v):
        assert arr.shape == (T, nx)


def test_1d_input_is_promoted_to_2d():
    """A flat (T,) input should be treated as (T, 1) and return (T, 1) outputs."""
    x_s, xd_s, xd_v = gp_smooth(t, noisy(1).squeeze(), n_iter=5)
    for arr in (x_s, xd_s, xd_v):
        assert arr.shape == (T, 1)


# ---------------------------------------------------------------------------
# Numerical validity
# ---------------------------------------------------------------------------

def test_no_nans():
    """No output should contain NaN under normal inputs."""
    x_s, xd_s, xd_v = gp_smooth(t, noisy(2), n_iter=5)
    for arr in (x_s, xd_s, xd_v):
        assert not np.isnan(arr).any()


def test_derivative_variance_nonnegative():
    """
    x_dot_var is the diagonal of a posterior covariance matrix.
    Negative values would mean a broken Cholesky solve or missing clamp.
    """
    _, _, xd_v = gp_smooth(t, noisy(2), n_iter=5)
    assert (xd_v >= 0).all()


# ---------------------------------------------------------------------------
# Correctness (math sanity checks)
# ---------------------------------------------------------------------------

def test_smoothed_output_closer_to_truth_than_raw_noisy():
    """
    The whole point of the smoother: it should reduce noise.
    Uses a clean sine with small additive noise so the bar is easy to clear.
    """
    t_long = np.linspace(0.0, 2.0 * np.pi, 30)
    x_clean = np.sin(t_long).reshape(-1, 1)
    x_noisy = x_clean + np.random.default_rng(42).standard_normal((30, 1)) * 0.05

    x_s, _, _ = gp_smooth(t_long, x_noisy, n_iter=20, noise_var=0.005)

    mse_smooth = np.mean((x_s - x_clean) ** 2)
    mse_noisy  = np.mean((x_noisy - x_clean) ** 2)
    assert mse_smooth < mse_noisy


def test_derivative_sign_on_monotone_signal():
    """
    On a strictly increasing signal the mean derivative must be positive.
    Tests that K_cross and alpha have the right orientation.
    """
    x_increasing = (t ** 2).reshape(-1, 1)
    _, xd_s, _ = gp_smooth(t, x_increasing, n_iter=10)
    assert xd_s.mean() > 0


def test_derivative_magnitude_on_linear_signal():
    """
    On x = t the true derivative is 1.0 everywhere.
    The GP should recover this within a loose tolerance after enough iterations.
    """
    x_linear = t.reshape(-1, 1)
    _, xd_s, _ = gp_smooth(t, x_linear, n_iter=50, noise_var=1e-4)
    assert np.allclose(xd_s, 1.0, atol=0.3)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lr", [0.01, 0.05, 0.2])
def test_various_lr_no_nan(lr):
    """Optimizer learning rate should not cause NaN for any reasonable value."""
    x_s, xd_s, xd_v = gp_smooth(t, noisy(1), n_iter=3, lr=lr)
    for arr in (x_s, xd_s, xd_v):
        assert not np.isnan(arr).any()


def test_zero_iterations_does_not_crash():
    """
    With n_iter=0 the training loop is skipped entirely.
    The function should still return valid-shaped, non-NaN arrays using default hyperparameters.
    """
    x_s, xd_s, xd_v = gp_smooth(t, noisy(1), n_iter=0)
    for arr in (x_s, xd_s, xd_v):
        assert arr.shape == (T, 1)
        assert not np.isnan(arr).any()


def test_minimum_length_timeseries():
    """
    T=3 is the smallest meaningful timeseries.
    The 3x3 Cholesky and all matrix ops should stay numerically stable.
    """
    t_short = np.linspace(0.0, 1.0, 3)
    x_short = np.random.default_rng(0).standard_normal((3, 1))
    x_s, xd_s, xd_v = gp_smooth(t_short, x_short, n_iter=5)
    for arr in (x_s, xd_s, xd_v):
        assert arr.shape == (3, 1)
        assert not np.isnan(arr).any()
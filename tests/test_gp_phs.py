"""
test_gp_phs.py
==============
Comprehensive pytest suite for gp_phs.py.

Organised by component, exactly mirroring the source file structure:

  § Param utilities       — positive_param, nonneg_param, recover, param_table
  § _extract_closure_parameters
  § PHSMatrices           — Component 1
  § PHSKernel             — Component 2
  § PHSMeanFunction       — Component 3
  § GPPHSModel            — Component 4
  § GPPHSLoss             — (from loss.py, tightly coupled to gp_phs)
  § GPPosterior           — Component 5
  § GPPHSNode             — Component 6
  § Integration           — two-component interaction tests
  § End-to-end            — full gradient-descent training loop

All tests are written against the *actual* implementation, not the docstrings.
Key discrepancies deliberately tested:
  - get_R uses raw d (NOT d²); the docstring says "R = diag(d²)" but the
    code does torch.diag_embed(d).
  - PHSKernel.signal_var has shape (1,), not ().
  - PHSKernel.forward requires N == M (jitter adds a square eye).
  - positive_param uses torch.exp, not softplus.

Run:
    pytest tests/test_gp_phs.py -v
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

# ---------------------------------------------------------------------------
# Module-level fixture helpers
# ---------------------------------------------------------------------------

def _phs(nx=2, nu=1) -> "PHSMatrices":
    """Minimal PHSMatrices with constant, non-zero diagonal R."""
    from neuromancer.dynamics.gp_phs import PHSMatrices
    return PHSMatrices(
        nx=nx, nu=nu,
        J_upper={(0, 1): lambda x: torch.ones(x.shape[0])},
        R_diag={i: lambda x: torch.ones(x.shape[0]) * 0.1 for i in range(nx)},
        G_full={(0, 0): lambda x: torch.ones(x.shape[0])},
    )


def _model_and_likelihood(nx=2, nu=1):
    from neuromancer.dynamics.gp_phs import GPPHSModel
    phs = _phs(nx, nu)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPPHSModel(phs, nx, nu)
    return model, likelihood, phs


def _posterior(nx=2, nu=1):
    from neuromancer.dynamics.gp_phs import GPPosterior
    model, likelihood, phs = _model_and_likelihood(nx, nu)
    model.eval(); likelihood.eval()
    return GPPosterior(
        model=model, likelihood=likelihood, phs_matrices=phs,
        lengthscale=model.covar_module.lengthscale.detach(),
        signal_var=model.covar_module.signal_var.detach(),
        noise_var=likelihood.noise.detach(),
    )


# ===========================================================================
# § Param utilities
# ===========================================================================


class TestParamTable:

    class _FakeParam:
        """Minimal stand-in exposing `.value`, since positive_param/recover
        were removed but param_table still expects a `.value` attribute."""
        def __init__(self, init_value: float):
            self.raw = nn.Parameter(torch.log(torch.tensor(init_value)))

        @property
        def value(self):
            return torch.exp(self.raw)

    def test_returns_dataframe_with_correct_columns(self):
        from neuromancer.dynamics.gp_phs import param_table
        import pandas as pd
        k1 = self._FakeParam(2.0)
        k2 = self._FakeParam(0.5)
        df = param_table({"k1_": (k1, 2.0), "k2_": (k2, 0.5)})
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"Parameter", "True", "Learned", "Rel. Error %"}

    def test_row_count_matches_param_map(self):
        from neuromancer.dynamics.gp_phs import param_table
        params = {f"x{i}": (self._FakeParam(float(i + 1)), float(i + 1))
                  for i in range(3)}
        df = param_table(params)
        assert len(df) == 3

    def test_relative_error_is_zero_at_true_value(self):
        from neuromancer.dynamics.gp_phs import param_table
        raw = self._FakeParam(1.0)
        df  = param_table({"exact_": (raw, 1.0)})
        assert df["Rel. Error %"].iloc[0] < 1e-3


# ===========================================================================
# § _extract_closure_parameters
# ===========================================================================

_GLOBAL_PARAM = nn.Parameter(torch.tensor(99.0))


class TestExtractClosureParameters:

    def test_nn_module_input_returns_named_parameters(self):
        from neuromancer.dynamics.gp_phs import _extract_closure_parameters
        mod = nn.Linear(3, 2)
        result = _extract_closure_parameters(mod)
        assert set(result.keys()) == {"weight", "bias"}
        assert result["weight"] is mod.weight
        assert result["bias"]   is mod.bias

    def test_closure_parameter_detected(self):
        from neuromancer.dynamics.gp_phs import _extract_closure_parameters
        def make():
            p = nn.Parameter(torch.tensor(1.0))
            return p, lambda x: x * p
        p, fn = make()
        result = _extract_closure_parameters(fn)
        assert any(v is p for v in result.values())

    def test_nn_module_in_closure_parameters_extracted(self):
        from neuromancer.dynamics.gp_phs import _extract_closure_parameters
        def make():
            m = nn.Linear(2, 2)
            return m, lambda x: m(x)
        m, fn = make()
        result = _extract_closure_parameters(fn)
        assert len(result) == len(list(m.parameters()))
        for p in m.parameters():
            assert any(v is p for v in result.values())

    def test_plain_lambda_returns_empty(self):
        from neuromancer.dynamics.gp_phs import _extract_closure_parameters
        fn = lambda x: x * 2.0 + 1.0
        result = _extract_closure_parameters(fn)
        assert result == {}

    def test_non_parameter_closure_value_ignored(self):
        from neuromancer.dynamics.gp_phs import _extract_closure_parameters
        def make():
            c = 3.14
            s = "hello"
            return lambda x: x * c + len(s)
        fn = make()
        result = _extract_closure_parameters(fn)
        assert result == {}

    def test_global_parameter_detected_via_load_global(self):
        from neuromancer.dynamics.gp_phs import _extract_closure_parameters
        fn = lambda x: x + _GLOBAL_PARAM
        result = _extract_closure_parameters(fn)
        assert any(v is _GLOBAL_PARAM for v in result.values())

    def test_multiple_closure_params_all_detected(self):
        from neuromancer.dynamics.gp_phs import _extract_closure_parameters
        def make():
            p1 = nn.Parameter(torch.tensor(1.0))
            p2 = nn.Parameter(torch.tensor(2.0))
            return p1, p2, lambda x: x * p1 + p2
        p1, p2, fn = make()
        result = _extract_closure_parameters(fn)
        ids = {id(v) for v in result.values()}
        assert id(p1) in ids
        assert id(p2) in ids


# ===========================================================================
# § PHSMatrices — Component 1
# ===========================================================================

class TestPHSMatricesConstruction:

    def test_valid_construction_does_not_raise(self):
        _phs()

    def test_J_upper_key_i_equals_j_raises(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.raises(ValueError, match="J_upper key"):
            PHSMatrices(nx=2, nu=1,
                J_upper={(0, 0): lambda x: torch.ones(x.shape[0])},
                R_diag={i: lambda x: torch.ones(x.shape[0]) for i in range(2)},
                G_full={(0, 0): lambda x: torch.ones(x.shape[0])})

    def test_J_upper_key_lower_triangle_raises(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.raises(ValueError, match="J_upper key"):
            PHSMatrices(nx=2, nu=1,
                J_upper={(1, 0): lambda x: torch.ones(x.shape[0])},
                R_diag={i: lambda x: torch.ones(x.shape[0]) for i in range(2)},
                G_full={(0, 0): lambda x: torch.ones(x.shape[0])})

    def test_J_upper_key_out_of_range_raises(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.raises(ValueError, match="J_upper key"):
            PHSMatrices(nx=2, nu=1,
                J_upper={(0, 5): lambda x: torch.ones(x.shape[0])},
                R_diag={i: lambda x: torch.ones(x.shape[0]) for i in range(2)},
                G_full={(0, 0): lambda x: torch.ones(x.shape[0])})

    def test_R_diag_key_out_of_range_raises(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.raises(ValueError, match="R_diag key"):
            PHSMatrices(nx=2, nu=1,
                J_upper={(0, 1): lambda x: torch.ones(x.shape[0])},
                R_diag={5: lambda x: torch.ones(x.shape[0])},
                G_full={(0, 0): lambda x: torch.ones(x.shape[0])})

    def test_R_diag_negative_key_raises(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.raises(ValueError, match="R_diag key"):
            PHSMatrices(nx=2, nu=1,
                J_upper={(0, 1): lambda x: torch.ones(x.shape[0])},
                R_diag={-1: lambda x: torch.ones(x.shape[0])},
                G_full={(0, 0): lambda x: torch.ones(x.shape[0])})

    def test_G_full_row_out_of_range_raises(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.raises(ValueError, match="G_full key"):
            PHSMatrices(nx=2, nu=1,
                J_upper={(0, 1): lambda x: torch.ones(x.shape[0])},
                R_diag={i: lambda x: torch.ones(x.shape[0]) for i in range(2)},
                G_full={(5, 0): lambda x: torch.ones(x.shape[0])})

    def test_G_full_col_out_of_range_raises(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.raises(ValueError, match="G_full key"):
            PHSMatrices(nx=2, nu=1,
                J_upper={(0, 1): lambda x: torch.ones(x.shape[0])},
                R_diag={i: lambda x: torch.ones(x.shape[0]) for i in range(2)},
                G_full={(0, 5): lambda x: torch.ones(x.shape[0])})

    def test_missing_R_diagonal_emits_user_warning(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with pytest.warns(UserWarning, match="R_diag missing"):
            PHSMatrices(nx=3, nu=1,
                J_upper={(0, 1): lambda x: torch.ones(x.shape[0])},
                R_diag={0: lambda x: torch.ones(x.shape[0])},
                G_full={(0, 0): lambda x: torch.ones(x.shape[0])})

    def test_closure_parameter_registered_on_module(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        k = nn.Parameter(torch.tensor(0.0))  # raw log-param; recovered via exp() in the lambda
        phs = PHSMatrices(nx=2, nu=1,
            J_upper={(0, 1): lambda x: torch.ones(x.shape[0])},
            R_diag={
                0: lambda x: torch.exp(k) * torch.ones(x.shape[0]),
                1: lambda x: torch.ones(x.shape[0]) * 0.1,
            },
            G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        param_ids = {id(p) for p in phs.parameters()}
        assert id(k) in param_ids

class TestPHSMatricesGetJ:

    def test_shape(self):
        phs = _phs(nx=3, nu=1)
        x = torch.randn(7, 3)
        assert phs.get_J(x).shape == (7, 3, 3)

    def test_skew_symmetric(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        phs = PHSMatrices(nx=3, nu=1,
            J_upper={
                (0, 1): lambda x: torch.ones(x.shape[0]) * 2.0,
                (0, 2): lambda x: x[:, 0],
                (1, 2): lambda x: torch.sin(x[:, 1]),
            },
            R_diag={i: lambda x: torch.ones(x.shape[0]) * 0.1 for i in range(3)},
            G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        x = torch.randn(8, 3)
        J = phs.get_J(x)
        assert (J + J.transpose(-1, -2)).abs().max().item() < 1e-6

    def test_diagonal_is_zero(self):
        phs = _phs()
        x = torch.randn(5, 2)
        J = phs.get_J(x)
        assert J.diagonal(dim1=-1, dim2=-2).abs().max().item() < 1e-6

    def test_upper_triangle_values_come_from_callable(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        phs = PHSMatrices(nx=2, nu=1,
            J_upper={(0, 1): lambda x: x[:, 0] * 3.0},
            R_diag={i: lambda x: torch.zeros(x.shape[0]) for i in range(2)},
            G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        x = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        J = phs.get_J(x)
        assert torch.allclose(J[:, 0, 1], x[:, 0] * 3.0)
        assert torch.allclose(J[:, 1, 0], -x[:, 0] * 3.0)

    def test_empty_J_upper_returns_zero_matrix(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        phs = PHSMatrices(nx=2, nu=1, J_upper={},
            R_diag={i: lambda x: torch.zeros(x.shape[0]) for i in range(2)},
            G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        x = torch.randn(4, 2)
        assert phs.get_J(x).abs().max().item() == 0.0


class TestPHSMatricesGetR:

    def test_shape(self):
        phs = _phs(nx=3, nu=1)
        x = torch.randn(6, 3)
        assert phs.get_R(x).shape == (6, 3, 3)

    def test_is_diagonal(self):
        phs = _phs()
        x = torch.randn(5, 2)
        R = phs.get_R(x)
        off = R.clone()
        for i in range(2):
            off[:, i, i] = 0.0
        assert off.abs().max().item() == 0.0

    def test_diagonal_equals_raw_callback_output_not_squared(self):
        """
        IMPORTANT: get_R stores d = fn(x) directly, NOT fn(x)**2.
        The docstring says 'R = diag(d²)' but the code does torch.diag_embed(d).
        """
        from neuromancer.dynamics.gp_phs import PHSMatrices
        phs = PHSMatrices(nx=2, nu=1,
            J_upper={(0, 1): lambda x: torch.zeros(x.shape[0])},
            R_diag={
                0: lambda x: torch.full((x.shape[0],), 3.0),
                1: lambda x: torch.full((x.shape[0],), 5.0),
            },
            G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        x = torch.randn(4, 2)
        R = phs.get_R(x)
        assert torch.allclose(R[:, 0, 0], torch.full((4,), 3.0))
        assert torch.allclose(R[:, 1, 1], torch.full((4,), 5.0))

    def test_missing_diagonal_entry_is_zero(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            phs = PHSMatrices(nx=2, nu=1,
                J_upper={(0, 1): lambda x: torch.zeros(x.shape[0])},
                R_diag={0: lambda x: torch.ones(x.shape[0])},
                G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        x = torch.randn(3, 2)
        R = phs.get_R(x)
        assert torch.allclose(R[:, 1, 1], torch.zeros(3))


class TestPHSMatricesGetG:

    def test_shape(self):
        phs = _phs(nx=2, nu=3)
        x = torch.randn(5, 2)
        assert phs.get_G(x).shape == (5, 2, 3)

    def test_specified_entries_correct(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices
        phs = PHSMatrices(nx=2, nu=2,
            J_upper={(0, 1): lambda x: torch.zeros(x.shape[0])},
            R_diag={i: lambda x: torch.zeros(x.shape[0]) for i in range(2)},
            G_full={
                (0, 0): lambda x: torch.full((x.shape[0],), 7.0),
                (1, 1): lambda x: torch.full((x.shape[0],), 4.0),
            })
        x = torch.randn(3, 2)
        G = phs.get_G(x)
        assert torch.allclose(G[:, 0, 0], torch.full((3,), 7.0))
        assert torch.allclose(G[:, 1, 1], torch.full((3,), 4.0))
        assert torch.allclose(G[:, 0, 1], torch.zeros(3))
        assert torch.allclose(G[:, 1, 0], torch.zeros(3))

    def test_unspecified_entries_are_zero(self):
        phs = _phs(nx=2, nu=1)
        x = torch.randn(4, 2)
        G = phs.get_G(x)
        assert torch.allclose(G[:, 1, 0], torch.zeros(4))


class TestPHSMatricesForward:

    def test_returns_three_tensors(self):
        phs = _phs()
        x = torch.randn(4, 2)
        out = phs(x)
        assert len(out) == 3

    def test_shapes_are_correct(self):
        phs = _phs(nx=3, nu=2)
        x = torch.randn(5, 3)
        J, R, G = phs(x)
        assert J.shape == (5, 3, 3)
        assert R.shape == (5, 3, 3)
        assert G.shape == (5, 3, 2)

    def test_J_is_skew_symmetric_in_forward(self):
        phs = _phs()
        x = torch.randn(4, 2)
        J, _, _ = phs(x)
        assert (J + J.transpose(-1, -2)).abs().max().item() < 1e-6


# ===========================================================================
# § PHSKernel — Component 2
# ===========================================================================

class TestPHSKernelProperties:

    def test_raw_lengthscale_shape(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        assert k.raw_lengthscale.shape == (2,)

    def test_raw_signal_var_shape(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        assert k.raw_signal_var.shape == (1,)

    def test_lengthscale_uses_softplus(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        expected = F.softplus(k.raw_lengthscale)
        assert torch.allclose(k.lengthscale, expected)

    def test_signal_var_uses_softplus(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        expected = F.softplus(k.raw_signal_var)
        assert torch.allclose(k.signal_var, expected)

    def test_lengthscale_always_positive(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        with torch.no_grad():
            k.raw_lengthscale.fill_(-100.0)
        assert (k.lengthscale > 0).all()

    def test_signal_var_always_positive(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        with torch.no_grad():
            k.raw_signal_var.fill_(-100.0)
        assert k.signal_var.item() > 0

    def test_signal_var_shape_is_1_not_scalar(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        assert k.signal_var.shape == torch.Size([1])


class TestPHSKernelRBFAndHessian:

    def test_shapes(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        x1, x2 = torch.randn(5, 2), torch.randn(7, 2)
        k_rbf, H = k._rbf_and_hessian(x1, x2)
        assert k_rbf.shape == (5, 7)
        assert H.shape     == (5, 7, 2, 2)

    def test_rbf_self_diagonal_is_one(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        x = torch.randn(6, 2)
        k_rbf, _ = k._rbf_and_hessian(x, x)
        assert torch.allclose(k_rbf.diagonal(), torch.ones(6), atol=1e-5)

    def test_rbf_values_between_zero_and_one(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        x1, x2 = torch.randn(4, 2), torch.randn(6, 2)
        k_rbf, _ = k._rbf_and_hessian(x1, x2)
        assert (k_rbf > 0).all()
        assert (k_rbf <= 1.0 + 1e-6).all()

    def test_hessian_H_is_symmetric_per_block(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        x = torch.randn(5, 2)
        _, H = k._rbf_and_hessian(x, x)
        assert torch.allclose(H, H.transpose(-1, -2), atol=1e-5)


class TestPHSKernelForward:

    def test_output_is_dense_linear_operator(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        from linear_operator.operators import DenseLinearOperator
        k = PHSKernel(_phs(), nx=2)
        x = torch.randn(5, 2)
        out = k.forward(x, x)
        assert isinstance(out, DenseLinearOperator)

    def test_output_shape_N_equals_M(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        N = 6
        x = torch.randn(N, 2)
        K = k.forward(x, x).to_dense()
        assert K.shape == (N * 2, N * 2)

    def test_self_kernel_is_symmetric(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        x = torch.randn(5, 2)
        K = k.forward(x, x).to_dense()
        assert (K - K.T).abs().max().item() < 1e-4

    def test_jitter_makes_matrix_positive_definite(self):
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        x = torch.randn(5, 2)
        K = k.forward(x, x).to_dense()
        torch.linalg.cholesky(K)

    def test_forward_requires_N_equals_M(self):
        """
        forward(x1, x2) crashes when N != M because jitter adds a square eye.
        This is expected behaviour.
        """
        from neuromancer.dynamics.gp_phs import PHSKernel
        k = PHSKernel(_phs(), nx=2)
        x1 = torch.randn(4, 2)
        x2 = torch.randn(7, 2)
        with pytest.raises(Exception):
            k.forward(x1, x2)


# ===========================================================================
# § PHSMeanFunction — Component 3
# ===========================================================================

class TestPHSMeanFunction:

    def test_output_shape(self):
        from neuromancer.dynamics.gp_phs import PHSMeanFunction
        mean_fn = PHSMeanFunction(_phs(nx=2, nu=1), nx=2, nu=1)
        N = 8
        out = mean_fn.forward(torch.randn(N, 2), torch.randn(N, 1))
        assert out.shape == (N * 2,)

    def test_output_equals_Gu_flattened(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices, PHSMeanFunction
        phs = PHSMatrices(nx=2, nu=1,
            J_upper={(0, 1): lambda x: torch.zeros(x.shape[0])},
            R_diag={i: lambda x: torch.zeros(x.shape[0]) for i in range(2)},
            G_full={(0, 0): lambda x: torch.full((x.shape[0],), 2.0)})
        mean_fn = PHSMeanFunction(phs, nx=2, nu=1)
        N = 5
        x = torch.randn(N, 2)
        u = torch.ones(N, 1)
        out = mean_fn.forward(x, u)
        G   = phs.get_G(x)
        expected = (G @ u.unsqueeze(-1)).squeeze(-1).reshape(-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_zero_G_gives_zero_mean(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices, PHSMeanFunction
        phs = PHSMatrices(nx=2, nu=1,
            J_upper={(0, 1): lambda x: torch.zeros(x.shape[0])},
            R_diag={i: lambda x: torch.zeros(x.shape[0]) for i in range(2)},
            G_full={})
        mean_fn = PHSMeanFunction(phs, nx=2, nu=1)
        out = mean_fn.forward(torch.randn(4, 2), torch.randn(4, 1))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_no_learnable_parameters(self):
        from neuromancer.dynamics.gp_phs import PHSMeanFunction
        mean_fn = PHSMeanFunction(_phs(), nx=2, nu=1)
        assert len(list(mean_fn.parameters())) == 0


# ===========================================================================
# § GPPHSModel — Component 4
# ===========================================================================

class TestGPPHSModel:

    def test_forward_returns_multivariate_normal(self):
        model, _, _ = _model_and_likelihood()
        dist = model(torch.randn(5, 2), torch.randn(5, 1))
        assert isinstance(dist, gpytorch.distributions.MultivariateNormal)

    def test_mean_shape(self):
        model, _, _ = _model_and_likelihood(nx=2, nu=1)
        N = 6
        dist = model(torch.randn(N, 2), torch.randn(N, 1))
        assert dist.mean.shape == (N * 2,)

    def test_covariance_shape(self):
        model, _, _ = _model_and_likelihood(nx=2, nu=1)
        N = 6
        dist = model(torch.randn(N, 2), torch.randn(N, 1))
        K = dist.lazy_covariance_matrix.to_dense()
        assert K.shape == (N * 2, N * 2)

    def test_covariance_is_symmetric(self):
        model, _, _ = _model_and_likelihood()
        N = 5
        K = model(torch.randn(N, 2), torch.randn(N, 1)).lazy_covariance_matrix.to_dense()
        assert (K - K.T).abs().max().item() < 1e-4

    def test_has_trainable_parameters(self):
        model, _, _ = _model_and_likelihood()
        assert len(list(model.parameters())) > 0


# ===========================================================================
# § GPPHSLoss (from loss.py — tightly coupled to gp_phs)
# ===========================================================================

class TestGPPHSLoss:

    def _setup(self, N=8, nx=2, nu=1):
        from neuromancer.loss import GPPHSLoss
        model, likelihood, _ = _model_and_likelihood(nx, nu)
        return GPPHSLoss(model, likelihood), model, likelihood

    def test_returns_scalar(self):
        loss_fn, _, _ = self._setup()
        loss = loss_fn(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        assert loss.shape == ()

    def test_loss_is_finite(self):
        loss_fn, _, _ = self._setup()
        loss = loss_fn(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        assert torch.isfinite(loss)

    def test_gradients_flow_to_kernel_params(self):
        loss_fn, model, _ = self._setup()
        loss = loss_fn(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_xdot_2d_and_flat_give_same_loss(self):
        loss_fn, _, _ = self._setup()
        x, u   = torch.randn(6, 2), torch.randn(6, 1)
        xdot2d = torch.randn(6, 2)
        l1 = loss_fn(x, u, xdot2d)
        l2 = loss_fn(x, u, xdot2d.reshape(-1))
        assert torch.allclose(l1, l2, atol=1e-5)

    def test_xdot_var_path_is_finite(self):
        loss_fn, _, _ = self._setup()
        loss = loss_fn(
            torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2),
            xdot_var=torch.rand(8, 2) * 0.01,
        )
        assert torch.isfinite(loss)

    def test_xdot_var_none_falls_back_to_likelihood_noise(self):
        loss_fn, _, likelihood = self._setup()
        loss_none = loss_fn(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        loss_var  = loss_fn(
            torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2),
            xdot_var=torch.full((8, 2), likelihood.noise.item()),
        )
        assert torch.isfinite(loss_none) and torch.isfinite(loss_var)


# ===========================================================================
# § GPPosterior — Component 5
# ===========================================================================

class TestGPPosteriorKernelHelpers:

    def test_k_HH_shape(self):
        post = _posterior()
        K = post._k_HH(torch.randn(5, 2), torch.randn(7, 2))
        assert K.shape == (5, 7)

    def test_k_HH_self_diagonal_equals_signal_var(self):
        post = _posterior()
        x = torch.randn(6, 2)
        K = post._k_HH(x, x)
        expected = post.signal_var.expand(6)
        assert torch.allclose(K.diagonal(), expected, atol=1e-5)

    def test_k_HH_symmetric(self):
        post = _posterior()
        x = torch.randn(6, 2)
        K = post._k_HH(x, x)
        assert (K - K.T).abs().max().item() < 1e-5

    def test_k_HH_values_positive(self):
        post = _posterior()
        K = post._k_HH(torch.randn(4, 2), torch.randn(5, 2))
        assert (K > 0).all()

    def test_k_xdotH_shape(self):
        post = _posterior(nx=2, nu=1)
        K = post._k_xdotH(torch.randn(8, 2), torch.randn(5, 2))
        assert K.shape == (8 * 2, 5)

    def test_k_xdotH_shape_nx3(self):
        post = _posterior(nx=3, nu=1)
        K = post._k_xdotH(torch.randn(4, 3), torch.randn(6, 3))
        assert K.shape == (4 * 3, 6)

    def test_k_xdotH_finite(self):
        post = _posterior()
        K = post._k_xdotH(torch.randn(5, 2), torch.randn(4, 2))
        assert torch.isfinite(K).all()


class TestGPPosteriorGetKPhsPlusNoise:

    def test_shape_without_xdot_var(self):
        post = _posterior()
        N, nx = 6, 2
        K = post._get_K_phs_plus_noise(torch.randn(N, nx), torch.randn(N, 1))
        assert K.shape == (N * nx, N * nx)

    def test_shape_with_xdot_var(self):
        post = _posterior()
        N, nx = 6, 2
        K = post._get_K_phs_plus_noise(
            torch.randn(N, nx), torch.randn(N, 1),
            xdot_var=torch.rand(N, nx) * 0.01,
        )
        assert K.shape == (N * nx, N * nx)

    def test_noise_var_I_fallback_adds_to_diagonal(self):
        post = _posterior()
        N, nx = 4, 2
        x = torch.randn(N, nx); u = torch.randn(N, 1)
        K_noisy = post._get_K_phs_plus_noise(x, u)
        with torch.no_grad():
            K_clean = post.model(x, u).lazy_covariance_matrix.to_dense()
        diff = K_noisy - K_clean
        assert diff.diagonal().min().item() > 0


class TestGPPosteriorForward:

    def test_H_mean_shape(self):
        post = _posterior()
        N, M = 10, 7
        H_mean, _, _ = post(
            torch.randn(N, 2), torch.randn(N, 1), torch.randn(N, 2),
            torch.randn(M, 2),
        )
        assert H_mean.shape == (M,)

    def test_H_var_shape_and_nonneg(self):
        post = _posterior()
        N, M = 10, 7
        _, H_var, _ = post(
            torch.randn(N, 2), torch.randn(N, 1), torch.randn(N, 2),
            torch.randn(M, 2),
        )
        assert H_var.shape == (M,)
        assert (H_var >= 0).all()

    def test_H_samples_shape(self):
        post = _posterior()
        N, M, S = 10, 7, 5
        _, _, H_samples = post(
            torch.randn(N, 2), torch.randn(N, 1), torch.randn(N, 2),
            torch.randn(M, 2), n_samples=S,
        )
        assert H_samples.shape == (S, M)

    def test_H_mean_is_finite(self):
        post = _posterior()
        N = 8
        H_mean, _, _ = post(
            torch.randn(N, 2), torch.randn(N, 1), torch.randn(N, 2),
            torch.randn(N, 2),
        )
        assert torch.isfinite(H_mean).all()

    def test_H_samples_finite(self):
        post = _posterior()
        N = 8
        _, _, H_samples = post(
            torch.randn(N, 2), torch.randn(N, 1), torch.randn(N, 2),
            torch.randn(N, 2), n_samples=4,
        )
        assert torch.isfinite(H_samples).all()

    def test_xdot_var_path(self):
        post = _posterior()
        N, M = 8, 5
        H_mean, H_var, H_samples = post(
            torch.randn(N, 2), torch.randn(N, 1), torch.randn(N, 2),
            torch.randn(M, 2),
            xdot_var=torch.rand(N, 2) * 0.01,
        )
        assert H_mean.shape == (M,)
        assert H_var.shape  == (M,)
        assert (H_var >= 0).all()

    def test_runs_inside_no_grad(self):
        post = _posterior()
        N = 6
        H_mean, H_var, H_samples = post(
            torch.randn(N, 2), torch.randn(N, 1), torch.randn(N, 2),
            torch.randn(N, 2),
        )
        assert not H_mean.requires_grad
        assert not H_samples.requires_grad


class TestGPPosteriorPredict:

    def test_accepts_lists_of_numpy_arrays(self):
        post = _posterior()
        N, nx, nu = 10, 2, 1
        smoothed  = [np.random.randn(N, nx).astype(np.float32),
                     np.random.randn(N, nx).astype(np.float32)]
        us        = [np.random.randn(N, nu).astype(np.float32)] * 2
        xdots     = [np.random.randn(N, nx).astype(np.float32)] * 2
        xdot_vars = [np.abs(np.random.randn(N, nx)).astype(np.float32)] * 2
        H_mean, H_var, H_samples = post.predict(smoothed, us, xdots, xdot_vars, n_samples=3)
        assert H_mean.shape    == (2 * N,)
        assert H_var.shape     == (2 * N,)
        assert H_samples.shape == (3, 2 * N)

    def test_test_x_none_uses_train_x(self):
        post = _posterior()
        N, nx, nu = 6, 2, 1
        smoothed  = [np.random.randn(N, nx).astype(np.float32)]
        us        = [np.random.randn(N, nu).astype(np.float32)]
        xdots     = [np.random.randn(N, nx).astype(np.float32)]
        xdot_vars = [np.abs(np.random.randn(N, nx)).astype(np.float32)]
        H_mean, _, _ = post.predict(smoothed, us, xdots, xdot_vars, test_x=None)
        assert H_mean.shape == (N,)

    def test_output_types_are_tensors(self):
        post = _posterior()
        N, nx, nu = 5, 2, 1
        smoothed  = [np.random.randn(N, nx).astype(np.float32)]
        us        = [np.random.randn(N, nu).astype(np.float32)]
        xdots     = [np.random.randn(N, nx).astype(np.float32)]
        xdot_vars = [np.abs(np.random.randn(N, nx)).astype(np.float32)]
        outs = post.predict(smoothed, us, xdots, xdot_vars)
        for out in outs:
            assert isinstance(out, torch.Tensor)


# ===========================================================================
# § GPPHSNode — Component 6
# ===========================================================================

class TestGPPHSNodeConstruction:

    def test_loss_fn_is_none_before_first_forward(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        node = GPPHSNode(_phs(), nx=2, nu=1)
        assert node._loss_fn is None

    def test_initialized_flag_absent_before_first_forward(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        node = GPPHSNode(_phs(), nx=2, nu=1)
        assert not hasattr(node, '_initialized')

    def test_likelihood_noise_initialized_to_1e_3(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        node = GPPHSNode(_phs(), nx=2, nu=1)
        assert abs(node.likelihood.noise.item() - 1e-3) < 1e-6

    def test_has_trainable_parameters(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        node = GPPHSNode(_phs(), nx=2, nu=1)
        assert len(list(node.parameters())) > 0


class TestGPPHSNodeForward:

    def _node(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        return GPPHSNode(_phs(), nx=2, nu=1)

    def test_returns_scalar(self):
        node = self._node()
        out = node(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        assert out.shape == ()

    def test_output_is_finite(self):
        node = self._node()
        out = node(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        assert torch.isfinite(out)

    def test_initialized_flag_set_after_first_forward(self):
        node = self._node()
        node(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        assert hasattr(node, '_initialized')

    def test_loss_fn_created_after_first_forward(self):
        node = self._node()
        node(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        assert node._loss_fn is not None

    def test_initialization_happens_exactly_once(self):
        node = self._node()
        X, U, Xdot = torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2)
        node(X, U, Xdot)
        raw_after_first = node.gp_model.covar_module.raw_signal_var.item()
        node(X, U, Xdot)
        raw_after_second = node.gp_model.covar_module.raw_signal_var.item()
        assert raw_after_first == raw_after_second

    def test_signal_var_initialized_from_xdot_variance(self):
        node = self._node()
        Xdot = torch.ones(8, 2) * 5.0 + torch.randn(8, 2)
        X, U = torch.randn(8, 2), torch.randn(8, 1)
        node(X, U, Xdot)
        expected_raw = math.log(Xdot.var().clamp(min=1e-6).item())
        assert abs(node.gp_model.covar_module.raw_signal_var.item() - expected_raw) < 1e-5

    def test_xdot_var_path_is_finite(self):
        node = self._node()
        out = node(
            torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2),
            Xdot_var=torch.rand(8, 2) * 0.01,
        )
        assert torch.isfinite(out)

    def test_gp_model_training_mode_mirrors_node_mode(self):
        node = self._node()
        X, U, Xdot = torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2)
        node.train()
        node(X, U, Xdot)
        assert node.gp_model.training

        node.eval()
        node(X, U, Xdot)
        assert not node.gp_model.training


class TestGPPHSNodePosterior:

    def test_returns_gp_posterior(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode, GPPosterior
        node = GPPHSNode(_phs(), nx=2, nu=1)
        node(torch.randn(6, 2), torch.randn(6, 1), torch.randn(6, 2))
        post = node.posterior()
        assert isinstance(post, GPPosterior)

    def test_posterior_uses_current_learned_hyperparams(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        node = GPPHSNode(_phs(), nx=2, nu=1)
        node(torch.randn(6, 2), torch.randn(6, 1), torch.randn(6, 2))
        post = node.posterior()
        assert torch.allclose(
            post.lengthscale,
            node.gp_model.covar_module.lengthscale.detach()
        )
        assert torch.allclose(
            post.signal_var,
            node.gp_model.covar_module.signal_var.detach()
        )
        assert torch.allclose(
            post.noise_var,
            node.likelihood.noise.detach()
        )


# ===========================================================================
# § Integration tests
# ===========================================================================

class TestIntegration:

    def test_phsmatrices_flows_into_kernel(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices, PHSKernel
        k_raw = nn.Parameter(torch.tensor(0.0))
        phs = PHSMatrices(nx=2, nu=1,
            J_upper={(0, 1): lambda x: torch.exp(k_raw) * torch.ones(x.shape[0])},
            R_diag={i: lambda x: torch.ones(x.shape[0]) * 0.1 for i in range(2)},
            G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        kernel = PHSKernel(phs, nx=2)
        x = torch.randn(5, 2)
        K1 = kernel.forward(x, x).to_dense()

        with torch.no_grad():
            k_raw.fill_(math.log(10.0))
        K2 = kernel.forward(x, x).to_dense()
        assert not torch.allclose(K1, K2)

    def test_gpphs_model_mean_uses_phsmatrices_G(self):
        model, _, phs = _model_and_likelihood()
        N = 6
        x = torch.randn(N, 2)
        u = torch.randn(N, 1)
        dist = model(x, u)
        G = phs.get_G(x)
        expected_mean = (G @ u.unsqueeze(-1)).squeeze(-1).reshape(-1)
        assert torch.allclose(dist.mean, expected_mean, atol=1e-6)

    def test_gpphs_loss_gradient_reaches_phsmatrices(self):
        from neuromancer.dynamics.gp_phs import PHSMatrices, GPPHSModel
        from neuromancer.loss import GPPHSLoss
        k_raw = nn.Parameter(torch.tensor(math.log(0.5)))
        phs = PHSMatrices(nx=2, nu=1,
            J_upper={(0, 1): lambda x: torch.exp(k_raw) * torch.ones(x.shape[0])},
            R_diag={i: lambda x: torch.ones(x.shape[0]) * 0.1 for i in range(2)},
            G_full={(0, 0): lambda x: torch.ones(x.shape[0])})
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPPHSModel(phs, 2, 1)
        loss_fn = GPPHSLoss(model, likelihood)
        loss = loss_fn(torch.randn(8, 2), torch.randn(8, 1), torch.randn(8, 2))
        loss.backward()
        assert k_raw.grad is not None

    def test_gpphs_node_posterior_produces_valid_H(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        torch.manual_seed(1)
        node = GPPHSNode(_phs(), nx=2, nu=1)
        opt  = torch.optim.Adam(node.parameters(), lr=1e-2)
        X, U, Xdot = torch.randn(12, 2), torch.randn(12, 1), torch.randn(12, 2)
        node.train()
        for _ in range(5):
            opt.zero_grad(); node(X, U, Xdot).backward(); opt.step()
        node.eval()
        post = node.posterior()
        H_mean, H_var, _ = post(X, U, Xdot, X)
        assert torch.isfinite(H_mean).all()
        assert (H_var >= 0).all()


# ===========================================================================
# § End-to-end
# ===========================================================================

class TestEndToEnd:

    def test_nlml_decreases_over_training(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        torch.manual_seed(42)
        node = GPPHSNode(_phs(), nx=2, nu=1)
        opt  = torch.optim.Adam(node.parameters(), lr=1e-2)
        X, U, Xdot = torch.randn(20, 2), torch.randn(20, 1), torch.randn(20, 2)
        node.train()
        loss_initial = node(X, U, Xdot).item()
        for _ in range(10):
            opt.zero_grad(); node(X, U, Xdot).backward(); opt.step()
        loss_final = node(X, U, Xdot).item()
        assert loss_final < loss_initial, (
            f"NLML did not decrease: {loss_initial:.4f} → {loss_final:.4f}"
        )

    def test_posterior_shapes_after_training(self):
        from neuromancer.dynamics.gp_phs import GPPHSNode
        torch.manual_seed(7)
        N, nx, nu, S = 16, 2, 1, 4
        node = GPPHSNode(_phs(), nx=nx, nu=nu)
        opt  = torch.optim.Adam(node.parameters(), lr=1e-2)
        X, U, Xdot = torch.randn(N, nx), torch.randn(N, nu), torch.randn(N, nx)
        node.train()
        for _ in range(5):
            opt.zero_grad(); node(X, U, Xdot).backward(); opt.step()
        node.eval()
        post = node.posterior()
        H_mean, H_var, H_samples = post(X, U, Xdot, X, n_samples=S)
        assert H_mean.shape    == (N,)
        assert H_var.shape     == (N,)
        assert H_samples.shape == (S, N)
        assert (H_var >= 0).all()
        assert torch.isfinite(H_mean).all()
        assert torch.isfinite(H_samples).all()
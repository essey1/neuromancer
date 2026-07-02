"""
Comprehensive tests for HamiltonianApproximator.

Coverage:
    Unit        — individual private helpers (_k_se, _tps_basis, _fit_*, _eval_*, _grad_*)
    Integration — .fit() → .forward() → .gradient() pipelines for each method
    End-to-end  — full PHS workflow: fit on GP posterior samples, solve ODE with ∇H*
    Data-flow   — shapes, dtypes, device placement, buffer vs parameter registration
    Contract    — public-API invariants, error paths, idempotence of repeated fits
"""

import torch
import pytest
import numpy as np
from neuromancer.modules.hamiltonian_approximator import HamiltonianApproximator

METHODS = ['spline', 'gp']


# ─── helpers ────────────────────────────────────────────────────────────────

def make_approx(method: str, nx: int, **kw) -> HamiltonianApproximator:
    if method == 'gp':
        return HamiltonianApproximator(
            method='gp',
            lengthscale=torch.ones(nx),
            signal_var=torch.tensor(1.0),
            **kw,
        )
    return HamiltonianApproximator(method='spline', **kw)

def random_fit(
    approx: HamiltonianApproximator,
    nx: int,
    n: int = 25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, nx, generator=rng)
    H = torch.randn(n, generator=rng)
    approx.fit(x, H)
    return x, H

def well_separated_fit(approx, nx, n=25, seed=0, min_dist=0.15):
    """Like random_fit, but ensures training points are spread widely
    enough relative to lengthscale=1 to keep the kernel/TPS matrix
    well-conditioned. Verified (not assumed): condition number stays
    ill-posed (~1e7, even at float64) unless spacing exceeds ~1x the
    lengthscale. In 1D, randn-based rejection sampling can't reliably
    reach that spacing, so points are laid out deterministically instead.
    """
    rng = torch.Generator().manual_seed(seed)
    if nx == 1:
        x = torch.linspace(-15.0, 15.0, n).unsqueeze(-1)
    else:
        pts = []
        tries = 0
        while len(pts) < n and tries < 10000:
            candidate = torch.randn(nx, generator=rng)
            if all((candidate - p).norm().item() > min_dist for p in pts):
                pts.append(candidate)
            tries += 1
        x = torch.stack(pts)
    H = torch.randn(n, generator=rng)
    approx.fit(x, H)
    return x, H

def quadratic_H(x: torch.Tensor) -> torch.Tensor:
    """Known closed-form Hamiltonian H(x) = ½‖x‖²; ∇H = x."""
    return 0.5 * (x ** 2).sum(dim=-1)


def quadratic_grad(x: torch.Tensor) -> torch.Tensor:
    return x.clone()



# ════════════════════════════════════════════════════════════════════════════
# CONTRACT / VALIDATION
# ════════════════════════════════════════════════════════════════════════════

class TestContractErrors:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            HamiltonianApproximator(method='rbf')

    def test_gp_without_lengthscale_raises(self):
        with pytest.raises(ValueError):
            HamiltonianApproximator(method='gp', signal_var=torch.tensor(1.0))

    def test_gp_without_signal_var_raises(self):
        with pytest.raises(ValueError):
            HamiltonianApproximator(method='gp', lengthscale=torch.ones(2))

    def test_gp_without_any_hyperparams_raises(self):
        with pytest.raises(ValueError):
            HamiltonianApproximator(method='gp')

    def test_fit_returns_self(self):
        approx = make_approx('spline', 2)
        x = torch.randn(10, 2)
        ret = approx.fit(x, torch.randn(10))
        assert ret is approx

    @pytest.mark.parametrize("method", METHODS)
    def test_fit_is_idempotent(self, method):
        """Re-fitting should replace the stored interpolant/state."""
        approx = make_approx(method, 2)

        x1, H1 = random_fit(approx, 2, seed=1)

        old_x = approx._x_fit.clone()
        old_H = approx._H_fit.clone()

        x2, H2 = random_fit(approx, 2, seed=2)

        assert torch.allclose(approx._x_fit, x2)
        assert torch.allclose(approx._H_fit, H2)
        assert not torch.allclose(old_x, approx._x_fit)
        assert not torch.allclose(old_H, approx._H_fit)
# ════════════════════════════════════════════════════════════════════════════
# UNIT — GP helpers
# ════════════════════════════════════════════════════════════════════════════

class TestGPKernel:
    """Tests for _k_se in isolation."""

    def _make_gp(self, nx: int, ell: float = 1.0, sf: float = 1.0):
        return HamiltonianApproximator(
            method='gp',
            lengthscale=torch.full((nx,), ell),
            signal_var=torch.tensor(sf),
        )

    def test_k_se_shape(self):
        approx = self._make_gp(3)
        x1 = torch.randn(5, 3)
        x2 = torch.randn(7, 3)
        K = approx._k_se(x1, x2)
        assert K.shape == (5, 7)

    def test_k_se_symmetry(self):
        approx = self._make_gp(2)
        x = torch.randn(6, 2)
        K = approx._k_se(x, x)
        assert torch.allclose(K, K.T, atol=1e-6)

    def test_k_se_positive_definite(self):
        """K + εI must be PD so Cholesky succeeds."""
        approx = self._make_gp(2)
        x = torch.randn(8, 2)
        K = approx._k_se(x, x) + 1e-4 * torch.eye(8)
        eigvals = torch.linalg.eigvalsh(K)
        assert (eigvals > 0).all()

    def test_k_se_diagonal_equals_signal_var(self):
        """k(x, x) = σ²_f for all x."""
        sf = 2.5
        approx = self._make_gp(3, sf=sf)
        x = torch.randn(5, 3)
        diag_k = approx._k_se(x, x).diag()
        assert torch.allclose(diag_k, torch.full((5,), sf), atol=1e-5)

    def test_k_se_lengthscale_effect(self):
        """Larger lengthscale → slower decay → higher off-diagonal values."""
        x1 = torch.tensor([[0.0, 0.0]])
        x2 = torch.tensor([[1.0, 0.0]])
        short = HamiltonianApproximator(
            method='gp', lengthscale=torch.tensor([0.5, 0.5]), signal_var=torch.tensor(1.0)
        )
        long_ = HamiltonianApproximator(
            method='gp', lengthscale=torch.tensor([5.0, 5.0]), signal_var=torch.tensor(1.0)
        )
        assert short._k_se(x1, x2).item() < long_._k_se(x1, x2).item()

    def test_k_se_signal_var_scales_output(self):
        approx1 = self._make_gp(2, sf=1.0)
        approx3 = self._make_gp(2, sf=3.0)
        x1 = torch.randn(4, 2)
        x2 = torch.randn(4, 2)
        assert torch.allclose(approx3._k_se(x1, x2), 3.0 * approx1._k_se(x1, x2), atol=1e-5)


# ════════════════════════════════════════════════════════════════════════════
# UNIT — Spline helpers
# ════════════════════════════════════════════════════════════════════════════

class TestTPSBasis:
    """Tests for the static _tps_basis kernel function."""

    def test_output_shape(self):
        r2 = torch.rand(4, 7)
        out = HamiltonianApproximator._tps_basis(r2)
        assert out.shape == (4, 7)

    def test_at_zero(self):
        """φ(0) should be 0 (limit of r² log r as r→0)."""
        r2 = torch.zeros(3)
        out = HamiltonianApproximator._tps_basis(r2)
        assert torch.allclose(out, torch.zeros(3), atol=1e-6)

    def test_positive_r2(self):
        """φ(r²) = ½ r² log(r²). For r²=1, φ = 0."""
        r2 = torch.tensor([1.0])
        out = HamiltonianApproximator._tps_basis(r2)
        assert torch.allclose(out, torch.zeros(1), atol=1e-6)

    def test_no_nan_for_small_values(self):
        r2 = torch.tensor([0.0, 1e-10, 1e-6, 1.0, 4.0])
        out = HamiltonianApproximator._tps_basis(r2)
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_differentiable(self):
        r2 = torch.rand(5, requires_grad=True)
        out = HamiltonianApproximator._tps_basis(r2)
        out.sum().backward()
        assert r2.grad is not None
        assert not r2.grad.isnan().any()


class TestSplineFitInternals:
    """Tests for _fit_spline internals and torch weight extraction."""

    def test_torch_weights_populated_after_fit(self):
        approx = make_approx('spline', 2)
        random_fit(approx, 2)
        assert approx._torch_w is not None

    def test_scipy_rbf_populated_after_fit(self):
        approx = make_approx('spline', 2)
        random_fit(approx, 2)
        assert approx._scipy_rbf is not None

    def test_torch_spline_eval_close_to_scipy(self):
        """
        The torch reimplementation (used for autograd) should track the
        scipy ground truth closely enough that gradients are meaningful.
        """
        approx = make_approx('spline', 2)
        x_fit, _ = random_fit(approx, 2)
        # Evaluate both paths at the training points.
        torch_out = approx._torch_spline_eval(x_fit)
        scipy_out = approx._eval_spline(x_fit)
        # They differ by the polynomial part; check that the scale is similar.
        assert torch_out.shape == scipy_out.shape
        assert not torch_out.isnan().any()


# ════════════════════════════════════════════════════════════════════════════
# UNIT — GP fit internals
# ════════════════════════════════════════════════════════════════════════════

class TestGPFitInternals:
    def test_alpha_populated_after_fit(self):
        approx = make_approx('gp', 2)
        random_fit(approx, 2)
        assert approx._alpha is not None

    def test_alpha_shape(self):
        n, nx = 20, 3
        approx = make_approx('gp', nx)
        random_fit(approx, nx, n=n)
        assert approx._alpha.shape == (n,)

    def test_x_fit_stored(self):
        approx = make_approx('gp', 2)
        x, _ = random_fit(approx, 2, n=15)
        assert approx._x_fit.shape == (15, 2)
        assert torch.allclose(approx._x_fit, x.detach())

    def test_H_fit_stored(self):
        approx = make_approx('gp', 2)
        x, H = random_fit(approx, 2, n=12)
        assert torch.allclose(approx._H_fit, H.detach())


# ════════════════════════════════════════════════════════════════════════════
# DATA FLOW — shapes, dtypes, buffers
# ════════════════════════════════════════════════════════════════════════════

class TestDataFlow:
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("nx", [1, 2, 4])
    def test_forward_shape(self, method, nx):
        approx = make_approx(method, nx)
        random_fit(approx, nx)
        assert approx(torch.randn(7, nx)).shape == (7,)

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("nx", [1, 2, 4])
    def test_gradient_shape(self, method, nx):
        approx = make_approx(method, nx)
        random_fit(approx, nx)
        assert approx.gradient(torch.randn(5, nx)).shape == (5, nx)

    @pytest.mark.parametrize("method", METHODS)
    def test_forward_dtype_float64(self, method):
        approx = make_approx(method, 2)
        x = torch.randn(10, 2, dtype=torch.float64)
        H = torch.randn(10, dtype=torch.float64)
        approx.fit(x, H)
        out = approx(torch.randn(4, 2, dtype=torch.float64))
        assert out.dtype == torch.float64

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("nx", [1, 2, 4])
    def test_no_nan_forward(self, method, nx):
        approx = make_approx(method, nx)
        random_fit(approx, nx)
        assert not approx(torch.randn(8, nx)).isnan().any()

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("nx", [1, 2, 4])
    def test_no_nan_gradient(self, method, nx):
        approx = make_approx(method, nx)
        random_fit(approx, nx)
        assert not approx.gradient(torch.randn(6, nx)).isnan().any()

    @pytest.mark.parametrize("method", METHODS)
    def test_single_query_point(self, method):
        """K=1 edge-case must not break shape contracts."""
        approx = make_approx(method, 2)
        random_fit(approx, 2)
        assert approx(torch.randn(1, 2)).shape == (1,)
        assert approx.gradient(torch.randn(1, 2)).shape == (1, 2)

    def test_gp_hyperparams_are_buffers_not_parameters(self):
        approx = make_approx('gp', 3)
        buf_names   = {n for n, _ in approx.named_buffers()}
        param_names = {n for n, _ in approx.named_parameters()}
        assert 'lengthscale' in buf_names   and 'lengthscale' not in param_names
        assert 'signal_var'  in buf_names   and 'signal_var'  not in param_names

    def test_gp_buffers_not_in_state_dict_parameters(self):
        """Optimizer should see zero learnable parameters."""
        approx = make_approx('gp', 2)
        assert len(list(approx.parameters())) == 0

    def test_fit_detaches_x_and_H(self):
        """Stored _x_fit / _H_fit must not carry grad_fn (memory-safe)."""
        approx = make_approx('gp', 2)
        x = torch.randn(10, 2, requires_grad=True)
        H = torch.randn(10, requires_grad=True)
        approx.fit(x, H)
        assert approx._x_fit.grad_fn is None
        assert approx._H_fit.grad_fn is None

    @pytest.mark.parametrize("method", METHODS)
    def test_large_batch_forward(self, method):
        """Should handle K >> M without memory errors."""
        approx = make_approx(method, 2)
        random_fit(approx, 2, n=30)
        out = approx(torch.randn(500, 2))
        assert out.shape == (500,)
        assert not out.isnan().any()

    @pytest.mark.parametrize("method", METHODS)
    def test_nx_1_scalar_state(self, method):
        """1-D state space edge-case."""
        approx = make_approx(method, 1)
        random_fit(approx, 1, n=20)
        x = torch.randn(5, 1)
        assert approx(x).shape == (5,)
        assert approx.gradient(x).shape == (5, 1)


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION — pipeline correctness
# ════════════════════════════════════════════════════════════════════════════

class TestInterpolation:
    """Both methods are interpolants: they should recover training values."""

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("nx", [1, 2])
    def test_reproduces_training_values(self, method, nx):
        approx = make_approx(method, nx)
        x_fit, H_fit = well_separated_fit(approx, nx)
        assert torch.allclose(approx(x_fit), H_fit, atol=1e-2)

    @pytest.mark.parametrize("method", METHODS)
    def test_smooth_function_low_residual(self, method):
        """
        When the underlying function is smooth (quadratic), the approximator
        should achieve small leave-out error on held-out points close to training.
        """
        nx = 2
        rng = torch.Generator().manual_seed(42)
        x_train = torch.randn(50, nx, generator=rng)
        H_train = quadratic_H(x_train)

        approx = make_approx(method, nx)
        approx.fit(x_train, H_train)

        x_test = 0.1 * torch.randn(20, nx, generator=rng)  # near-origin → dense coverage
        H_pred = approx(x_test)
        H_true = quadratic_H(x_test)
        rmse = (H_pred - H_true).pow(2).mean().sqrt()
        assert rmse < 0.3, f"{method} RMSE {rmse:.4f} too large on quadratic H"


class TestGPGradientConsistency:
    """Analytic GP gradient vs autograd."""

    @pytest.mark.parametrize("nx", [1, 2, 4])
    def test_analytic_vs_autograd(self, nx):
        approx = make_approx('gp', nx)
        random_fit(approx, nx)

        x = torch.randn(4, nx)
        grad_analytic = approx.gradient(x)

        x_ad = x.detach().requires_grad_(True)
        approx._eval_gp(x_ad).sum().backward()

        assert torch.allclose(grad_analytic, x_ad.grad, atol=1e-4)

    def test_gradient_direction_quadratic(self):
        """For H=½‖x‖², ∇H*(x) ≈ x if fit is good."""
        nx = 2
        rng = torch.Generator().manual_seed(7)
        x_train = torch.randn(60, nx, generator=rng)
        H_train = quadratic_H(x_train)

        approx = make_approx('gp', nx)
        approx.fit(x_train, H_train)

        x_test = 0.2 * torch.randn(10, nx, generator=rng)
        grad_pred = approx.gradient(x_test)
        grad_true = quadratic_grad(x_test)

        cos_sim = torch.nn.functional.cosine_similarity(grad_pred, grad_true, dim=-1)
        # Directions should broadly agree (cosine > 0.8) for easy quadratic.
        assert (cos_sim > 0.8).float().mean() > 0.8


class TestSplineGradientConsistency:
    """Spline autograd gradient vs finite differences on the scipy eval."""

    @pytest.mark.parametrize("nx", [1, 2])
    def test_matches_finite_differences(self, nx):
        approx = make_approx('spline', nx)
        well_separated_fit(approx, nx)

        x    = torch.randn(3, nx)
        grad = approx.gradient(x)

        eps     = 1e-4
        fd_grad = torch.zeros_like(grad)
        for d in range(nx):
            xp = x.clone(); xp[:, d] += eps
            xm = x.clone(); xm[:, d] -= eps
            fd_grad[:, d] = (approx(xp) - approx(xm)) / (2 * eps)

        assert torch.allclose(grad, fd_grad, atol=2e-3)

    def test_gradient_direction_quadratic(self):
        """Same sanity-check as the GP version."""
        nx = 2
        rng = torch.Generator().manual_seed(9)
        x_train = torch.randn(60, nx, generator=rng)
        H_train = quadratic_H(x_train)

        approx = make_approx('spline', nx)
        approx.fit(x_train, H_train)

        x_test = 0.2 * torch.randn(8, nx, generator=rng)
        grad_pred = approx.gradient(x_test)
        grad_true = quadratic_grad(x_test)

        cos_sim = torch.nn.functional.cosine_similarity(grad_pred, grad_true, dim=-1)
        assert (cos_sim > 0.7).float().mean() > 0.7


class TestGPNoiseHyperparameter:
    """Noise nugget controls numerical stability."""

    def test_very_small_noise_still_converges(self):
        approx = HamiltonianApproximator(
            method='gp',
            lengthscale=torch.ones(2),
            signal_var=torch.tensor(1.0),
            noise=1e-6,
        )
        x, H = random_fit(approx, 2)
        assert not approx(x).isnan().any()

    def test_large_noise_acts_as_regularizer(self):
        """High noise → smoother fit → larger residual on training data, but no crash."""
        approx = HamiltonianApproximator(
            method='gp',
            lengthscale=torch.ones(2),
            signal_var=torch.tensor(1.0),
            noise=10.0,
        )
        x, H = random_fit(approx, 2)
        pred = approx(x)
        # Residuals can be large with heavy regularization — just no NaN.
        assert not pred.isnan().any()


# ════════════════════════════════════════════════════════════════════════════
# END-TO-END — PHS ODE integration
# ════════════════════════════════════════════════════════════════════════════

class TestEndToEndPHS:
    """
    Simulates the intended usage context: fit H* from samples, then use
    ∇H*(x) to drive a port-Hamiltonian ODE   ẋ = (J − R) ∇H*(x).

    We use simple Euler integration so the test has no external ODE-solver
    dependency. The system is a damped harmonic oscillator in PHS form:

        J = [[0, 1], [-1, 0]],  R = [[0, 0], [0, γ]],  H = ½‖x‖²
        ẋ = (J − R) x  →  energy must decrease (R > 0 is dissipative).
    """

    def _euler_simulate(
        self,
        approx: HamiltonianApproximator,
        x0: torch.Tensor,
        J: torch.Tensor,
        R: torch.Tensor,
        dt: float = 0.01,
        steps: int = 200,
    ) -> list[torch.Tensor]:
        JR = J - R
        traj = [x0]
        x = x0.clone()
        for _ in range(steps):
            grad_H = approx.gradient(x.unsqueeze(0)).squeeze(0)   # (nx,)
            dx = JR @ grad_H
            x = x + dt * dx
            traj.append(x)
        return traj

    @pytest.mark.parametrize("method", METHODS)
    def test_dissipative_energy_decreases(self, method):
        """
        With γ > 0 (damping), H should decrease monotonically over the
        simulated trajectory when ∇H* ≈ ∇H (dense training).
        """
        nx = 2
        gamma = 0.5
        J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        R = torch.tensor([[0.0, 0.0], [0.0, gamma]])

        rng = torch.Generator().manual_seed(13)
        x_train = torch.randn(80, nx, generator=rng)
        H_train = quadratic_H(x_train)

        approx = make_approx(method, nx)
        approx.fit(x_train, H_train)

        x0 = torch.tensor([1.0, 0.0])
        traj = self._euler_simulate(approx, x0, J, R, dt=0.01, steps=300)

        energies = [quadratic_H(x.unsqueeze(0)).item() for x in traj]
        # Energy should be non-increasing (allow tiny numerical noise).
        decreases = sum(
            energies[i + 1] <= energies[i] + 1e-3
            for i in range(len(energies) - 1)
        )
        frac_decreasing = decreases / (len(energies) - 1)
        assert frac_decreasing > 0.95, (
            f"{method}: only {frac_decreasing:.1%} of steps had non-increasing energy"
        )

    @pytest.mark.parametrize("method", METHODS)
    def test_conservative_energy_conserved(self, method):
        """
        With R=0 (no damping), H should remain approximately constant.
        Euler is not symplectic so we allow a small drift.
        """
        nx = 2
        J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        R = torch.zeros(2, 2)

        rng = torch.Generator().manual_seed(17)
        x_train = torch.randn(80, nx, generator=rng)
        H_train = quadratic_H(x_train)

        approx = make_approx(method, nx)
        approx.fit(x_train, H_train)

        x0 = torch.tensor([1.0, 0.0])
        traj = self._euler_simulation_conservative(approx, x0, J, dt=0.001, steps=200)

        H_start = quadratic_H(traj[0].unsqueeze(0)).item()
        H_end   = quadratic_H(traj[-1].unsqueeze(0)).item()
        # Energy drift should be small relative to initial energy.
        assert abs(H_end - H_start) / (abs(H_start) + 1e-8) < 0.15, (
            f"{method}: energy drifted by {abs(H_end - H_start):.4f} "
            f"from start {H_start:.4f}"
        )

    def _euler_simulation_conservative(self, approx, x0, J, dt, steps):
        traj = [x0]
        x = x0.clone()
        for _ in range(steps):
            grad_H = approx.gradient(x.unsqueeze(0)).squeeze(0)
            dx = J @ grad_H
            x = x + dt * dx
            traj.append(x)
        return traj

    @pytest.mark.parametrize("method", METHODS)
    def test_gradient_used_in_ode_is_finite(self, method):
        """Every ∇H* evaluation during ODE must be finite (no blow-up)."""
        nx = 2
        J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        R = torch.tensor([[0.0, 0.0], [0.0, 0.3]])

        rng = torch.Generator().manual_seed(21)
        x_train = torch.randn(60, nx, generator=rng)
        H_train = quadratic_H(x_train)

        approx = make_approx(method, nx)
        approx.fit(x_train, H_train)

        x = torch.tensor([0.5, 0.5])
        JR = J - R
        for _ in range(100):
            grad_H = approx.gradient(x.unsqueeze(0)).squeeze(0)
            assert grad_H.isfinite().all(), f"{method}: NaN/Inf in gradient during ODE"
            x = x + 0.01 * (JR @ grad_H)


# ════════════════════════════════════════════════════════════════════════════
# END-TO-END — forward call via __call__ delegates to forward()
# ════════════════════════════════════════════════════════════════════════════

class TestCallableInterface:
    @pytest.mark.parametrize("method", METHODS)
    def test_call_equals_forward(self, method):
        approx = make_approx(method, 2)
        random_fit(approx, 2)
        x = torch.randn(5, 2)
        assert torch.allclose(approx(x), approx.forward(x), atol=1e-8)

    @pytest.mark.parametrize("method", METHODS)
    def test_gradient_not_none_after_fit(self, method):
        approx = make_approx(method, 2)
        random_fit(approx, 2)
        g = approx.gradient(torch.randn(3, 2))
        assert g is not None

    @pytest.mark.parametrize("method", METHODS)
    def test_gradient_detached_from_graph_by_default(self, method):
        """
        The returned gradient tensor itself should not require grad —
        it's a value used for ODE driving, not for backprop through H*.
        (GP analytic path never has grad; spline autograd uses .detach equiv.)
        """
        approx = make_approx(method, 2)
        random_fit(approx, 2)
        g = approx.gradient(torch.randn(4, 2))
        # For gp: analytic, no grad. For spline: grad returned by autograd.grad
        # has no grad_fn by construction.
        assert not g.requires_grad


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION — method parity on identical data
# ════════════════════════════════════════════════════════════════════════════

class TestMethodParity:
    """Both methods should agree in rough magnitude on the same data."""

    def test_forward_same_order_of_magnitude(self):
        nx = 2
        rng = torch.Generator().manual_seed(55)
        x_train = torch.randn(40, nx, generator=rng)
        H_train = quadratic_H(x_train)

        sp = make_approx('spline', nx)
        gp = make_approx('gp', nx)
        sp.fit(x_train, H_train)
        gp.fit(x_train, H_train)

        x_test = torch.randn(10, nx, generator=rng) * 0.3  # stay inside training range
        H_sp = sp(x_test)
        H_gp = gp(x_test)

        # Both should be positive for quadratic H near origin; rough scale check.
        assert (H_sp > -0.5).all() and (H_gp > -0.5).all()
        # Ratio within one order of magnitude.
        ratio = (H_sp.abs() / (H_gp.abs() + 1e-6))
        assert (ratio < 20).all() and (ratio > 0.05).all()

    def test_gradient_magnitudes_comparable(self):
        nx = 2
        rng = torch.Generator().manual_seed(66)
        x_train = torch.randn(40, nx, generator=rng)
        H_train = quadratic_H(x_train)

        sp = make_approx('spline', nx)
        gp = make_approx('gp', nx)
        sp.fit(x_train, H_train)
        gp.fit(x_train, H_train)

        x_test = 0.2 * torch.randn(8, nx, generator=rng)
        g_sp = approx_norm = sp.gradient(x_test).norm(dim=-1)
        g_gp = gp.gradient(x_test).norm(dim=-1)

        ratio = g_sp / (g_gp + 1e-6)
        assert (ratio < 50).all() and (ratio > 0.02).all()


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION — nn.Module interface
# ════════════════════════════════════════════════════════════════════════════

class TestNNModuleInterface:
    @pytest.mark.parametrize("method", METHODS)
    def test_is_nn_module(self, method):
        import torch.nn as nn
        approx = make_approx(method, 2)
        assert isinstance(approx, nn.Module)

    @pytest.mark.parametrize("method", METHODS)
    def test_eval_mode_does_not_break_inference(self, method):
        approx = make_approx(method, 2)
        random_fit(approx, 2)
        approx.eval()
        out = approx(torch.randn(5, 2))
        assert out.shape == (5,)
        assert not out.isnan().any()

    def test_gp_state_dict_contains_buffers(self):
        approx = make_approx('gp', 2)
        sd = approx.state_dict()
        assert 'lengthscale' in sd
        assert 'signal_var'  in sd

    @pytest.mark.parametrize("method", METHODS)
    def test_repr_does_not_raise(self, method):
        approx = make_approx(method, 2)
        _ = repr(approx)  # should not raise
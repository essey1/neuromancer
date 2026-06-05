"""
hamiltonian_approximator.py
============================
Approximates sampled H values from GPPosterior with a callable
function H*(x) that can be evaluated at arbitrary points for ODE solving.

Preserves PHS structure — we approximate H, not f, so:
    ẋ = (J(x) - R(x)) · ∇H*(x) + G(x) · u
remains a valid Port-Hamiltonian system.

Three methods:
    'spline' : Thin-plate spline (default). φ(r) = r² · log(r)
    'rbf'    : Multiquadric RBF.            φ(r) = sqrt(r² + 1)
    'gp'     : GP with SE kernel.           requires lengthscale and signal_var

∇H*(x) for 'spline'/'rbf' is computed via autograd on a differentiable
torch reimplementation of the fitted interpolant.
∇H*(x) for 'gp' is computed analytically from the SE kernel derivative.
"""

import torch
import torch.nn as nn
import numpy as np


class HamiltonianApproximator(nn.Module):
    """
    Fits a callable approximation H*(x) to sampled H values from GPPosterior.

    Args:
        method      : 'spline' (default) | 'rbf' | 'gp'
        lengthscale : (nx,) tensor — required for method='gp'
        signal_var  : scalar tensor  — required for method='gp'
        noise       : diagonal nugget for GP kernel stability (method='gp' only)

    Usage:
        # spline or rbf
        approx = HamiltonianApproximator(method='spline')
        approx.fit(x_fit, H_sample)

        # GP
        approx = HamiltonianApproximator(
                     method='gp',
                     lengthscale=learned['lengthscale'],
                     signal_var=learned['signal_var'],
                 )
        approx.fit(x_fit, H_sample)

        H_star   = approx(new_x)           # (K,)
        grad_H   = approx.gradient(new_x)  # (K, nx)
    """

    _SCIPY_KERNELS = {
        'spline': 'thin_plate_spline',
        'rbf':    'multiquadric',
    }

    def __init__(
        self,
        method:      str   = 'spline',
        lengthscale: torch.Tensor = None,
        signal_var:  torch.Tensor = None,
        noise:       float = 1e-4,
    ):
        super().__init__()

        if method not in ('spline', 'rbf', 'gp'):
            raise ValueError(f"method must be 'spline', 'rbf', or 'gp', got '{method}'")
        if method == 'gp' and (lengthscale is None or signal_var is None):
            raise ValueError("method='gp' requires lengthscale and signal_var")

        self.method = method
        self.noise  = noise

        if method == 'gp':
            self.register_buffer('lengthscale', lengthscale.detach().squeeze())
            self.register_buffer('signal_var',  signal_var.detach().squeeze())

        self._interpolant = None  # scipy RBF interpolant (spline/rbf)
        self._x_fit       = None  # fit points : (M, nx)
        self._H_fit       = None  # fit values : (M,)
        self._weights     = None  # RBF weights solved in torch (spline/rbf)
        self._alpha       = None  # GP weights: (K_fit + noise·I)⁻¹ H_vals (gp)

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, x: torch.Tensor, H_vals: torch.Tensor):
        """Fit to H values at states x."""
        self._x_fit = x.detach()
        self._H_fit = H_vals.detach()

        if self.method == 'gp':
            self._fit_gp(x, H_vals)
        else:
            self._fit_rbf(x, H_vals)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate H*(x) → (K,)"""
        if self.method == 'gp':
            return self._eval_gp(x)
        else:
            return self._eval_rbf_scipy(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∇H*(x) → (K, nx)"""
        if self.method == 'gp':
            return self._grad_gp(x)
        else:
            return self._grad_rbf_autograd(x)

    # ── GP (SE kernel) ─────────────────────────────────────────────────────────

    def _fit_gp(self, x: torch.Tensor, H_vals: torch.Tensor):
        K = self._k_se(x, x) + self.noise * torch.eye(len(x), dtype=x.dtype)
        L = torch.linalg.cholesky(K)
        self._alpha = torch.cholesky_solve(
            H_vals.detach().unsqueeze(-1), L
        ).squeeze(-1)

    def _k_se(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """SE kernel: σ²f · exp(-0.5 · ||x1-x2||²_Λ) → (N, M)"""
        inv_l = 1.0 / self.lengthscale
        delta = (x1[:, None, :] - x2[None, :, :]) * inv_l
        return self.signal_var * torch.exp(-0.5 * (delta ** 2).sum(-1))

    def _eval_gp(self, x: torch.Tensor) -> torch.Tensor:
        return self._k_se(x, self._x_fit) @ self._alpha

    def _grad_gp(self, x: torch.Tensor) -> torch.Tensor:
        """∇H*(x) = Σᵢ αᵢ · k(x, xᵢ) · Λ⁻²(xᵢ - x) → (K, nx)"""
        inv_l2  = 1.0 / self.lengthscale ** 2
        k       = self._k_se(x, self._x_fit)
        diff    = self._x_fit[None, :, :] - x[:, None, :]
        weights = k * self._alpha[None, :]
        return (weights[:, :, None] * diff * inv_l2).sum(dim=1)

    # ── RBF / spline ───────────────────────────────────────────────────────────

    def _fit_rbf(self, x: torch.Tensor, H_vals: torch.Tensor):
        from scipy.interpolate import RBFInterpolator
        self._interpolant = RBFInterpolator(
            x.detach().cpu().numpy(),
            H_vals.detach().cpu().numpy(),
            kernel=self._SCIPY_KERNELS[self.method],
        )
        self._weights = self._solve_rbf_weights(self._x_fit, self._H_fit)

    def _basis(self, r2: torch.Tensor) -> torch.Tensor:
        if self.method == 'spline':
            safe_r2 = r2.clamp(min=1e-10)
            return 0.5 * r2 * torch.log(safe_r2)
        else:
            return torch.sqrt(r2 + 1.0)

    def _solve_rbf_weights(
        self, x: torch.Tensor, H_vals: torch.Tensor
    ) -> torch.Tensor:
        diff = x[:, None, :] - x[None, :, :]
        r2   = (diff ** 2).sum(-1)
        Phi  = self._basis(r2)
        return torch.linalg.lstsq(Phi, H_vals.unsqueeze(-1)).solution.squeeze(-1)

    def _torch_rbf_eval(self, x: torch.Tensor) -> torch.Tensor:
        x_c  = self._x_fit.to(x.device)
        diff = x[:, None, :] - x_c[None, :, :]
        r2   = (diff ** 2).sum(-1)
        return self._basis(r2) @ self._weights.to(x.device)

    def _eval_rbf_scipy(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        return torch.tensor(self._interpolant(x_np), dtype=x.dtype, device=x.device)

    def _grad_rbf_autograd(self, x: torch.Tensor) -> torch.Tensor:
        x_req = x.detach().requires_grad_(True)
        H     = self._torch_rbf_eval(x_req)
        return torch.autograd.grad(H.sum(), x_req, create_graph=False)[0]


class GPHamiltonianApproximator(HamiltonianApproximator):
    """Backward-compatible alias — use HamiltonianApproximator(method='gp') instead."""

    def __init__(self, lengthscale: torch.Tensor, signal_var: torch.Tensor, noise: float = 1e-4):
        super().__init__(method='gp', lengthscale=lengthscale, signal_var=signal_var, noise=noise)

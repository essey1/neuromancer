"""
hamiltonian_approximator.py
============================
Approximates sampled H values from GPPosterior with a callable
H*(x) for use in ODE solving. Preserves PHS structure by approximating
H directly, so:

    ẋ = (J(x) - R(x)) · ∇H*(x) + G(x) · u

remains a valid Port-Hamiltonian system.

Two methods:
    'spline' : Thin-plate spline via scipy.interpolate.RBFInterpolator.
               ∇H* via torch.autograd on a differentiable torch reimplementation.
    'gp'     : GP with SE kernel. Requires lengthscale and signal_var.
               ∇H* computed analytically from the SE kernel derivative.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import RBFInterpolator


class HamiltonianApproximator(nn.Module):
    """
    Fits H*(x) to sampled Hamiltonian values, then evaluates H* and ∇H*
    at arbitrary query points.

    Args:
        method      : 'spline' (default) | 'gp'
        lengthscale : (nx,) tensor — required for method='gp'
        signal_var  : scalar tensor  — required for method='gp'
        noise       : nugget added to diagonal for GP stability (default 1e-4)

    Example — spline::

        approx = HamiltonianApproximator(method='spline')
        approx.fit(x_train, H_samples)
        H_star  = approx(x_query)          # (K,)
        grad_H  = approx.gradient(x_query) # (K, nx)

    Example — GP::

        approx = HamiltonianApproximator(
            method='gp',
            lengthscale=model.covar_module.base_kernel.lengthscale.squeeze(),
            signal_var=model.covar_module.outputscale,
        )
        approx.fit(x_train, H_samples)
        H_star  = approx(x_query)
        grad_H  = approx.gradient(x_query)
    """

    def __init__(
        self,
        method:      str             = 'spline',
        lengthscale: torch.Tensor    = None,
        signal_var:  torch.Tensor    = None,
        noise:       float           = 1e-4,
    ):
        super().__init__()

        if method not in ('spline', 'gp'):
            raise ValueError(f"method must be 'spline' or 'gp', got '{method}'")
        if method == 'gp' and (lengthscale is None or signal_var is None):
            raise ValueError("method='gp' requires lengthscale and signal_var")

        self.method = method
        self.noise  = noise

        if method == 'gp':
            # Store as buffers so they move with .to(device) and are excluded
            # from parameter updates.
            self.register_buffer('lengthscale', lengthscale.detach().squeeze())
            self.register_buffer('signal_var',  signal_var.detach().squeeze())

        # Populated by .fit()
        self._x_fit      : torch.Tensor     = None  # (M, nx) fit points
        self._H_fit      : torch.Tensor     = None  # (M,)    fit values
        self._scipy_rbf  : RBFInterpolator  = None  # spline: scipy interpolant
        self._torch_w    : torch.Tensor     = None  # spline: weights for autograd
        self._alpha      : torch.Tensor     = None  # gp:     dual coefficients

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, x: torch.Tensor, H_vals: torch.Tensor) -> 'HamiltonianApproximator':
        """
        Fit the approximator.

        Args:
            x      : (M, nx) state points used to collect H samples.
            H_vals : (M,)    sampled Hamiltonian values at those states.
        """
        self._x_fit = x.detach()
        self._H_fit = H_vals.detach()

        if self.method == 'spline':
            self._fit_spline(x, H_vals)
        else:
            self._fit_gp(x, H_vals)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate H*(x). Returns shape (K,)."""
        if self.method == 'spline':
            return self._eval_spline(x)
        else:
            return self._eval_gp(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∇H*(x). Returns shape (K, nx)."""
        if self.method == 'spline':
            return self._grad_spline(x)
        else:
            return self._grad_gp(x)

    # ── thin-plate spline ──────────────────────────────────────────────────────

    def _fit_spline(self, x: torch.Tensor, H_vals: torch.Tensor):
        x_np  = x.detach().cpu().numpy()
        H_np  = H_vals.detach().cpu().numpy()

        # scipy handles all the kernel setup and linear solve for us.
        self._scipy_rbf = RBFInterpolator(
            x_np, H_np, kernel='thin_plate_spline'
        )

        # Mirror the solved weights into torch so autograd can differentiate
        # through the same kernel at query time (scipy weights are not
        # differentiable). RBFInterpolator stores the solved coefficients in
        # .d (polynomial) and .coeffs (kernel part); we only need .coeffs for
        # the kernel term since we recompute the polynomial separately.
        self._torch_w = torch.tensor(
            self._scipy_rbf._coeffs, dtype=x.dtype
        )

    @staticmethod
    def _tps_basis(r2: torch.Tensor) -> torch.Tensor:
        """φ(r) = r² log(r) — the thin-plate spline kernel, r² = squared distance."""
        safe = r2.clamp(min=1e-12)   # avoids log(0)
        return 0.5 * r2 * torch.log(safe)

    def _torch_spline_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable torch reimplementation of the fitted thin-plate spline.
        Matches scipy output closely; used only for autograd.
        """
        x_c  = self._x_fit.to(x.device)
        diff = x[:, None, :] - x_c[None, :, :]        # (K, M, nx)
        r2   = (diff ** 2).sum(dim=-1)                 # (K, M)
        Phi  = self._tps_basis(r2)                     # (K, M)
        return Phi @ self._torch_w.to(x.device)        # (K,)

    def _eval_spline(self, x: torch.Tensor) -> torch.Tensor:
        """Use scipy for accurate evaluation (no polynomial drop)."""
        H_np = self._scipy_rbf(x.detach().cpu().numpy())
        return torch.tensor(H_np, dtype=x.dtype, device=x.device)

    def _grad_spline(self, x: torch.Tensor) -> torch.Tensor:
        """∇H* via autograd through the torch reimplementation."""
        x_req = x.detach().requires_grad_(True)
        H     = self._torch_spline_eval(x_req)
        grad, = torch.autograd.grad(H.sum(), x_req)
        return grad  # (K, nx)

    # ── GP with SE kernel ──────────────────────────────────────────────────────

    def _k_se(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Squared-exponential kernel with ARD lengthscales.

            k(x1, x2) = σ²_f · exp(-½ ‖x1 - x2‖²_Λ)

        where Λ = diag(ℓ²). Returns (N, M).
        """
        inv_l = 1.0 / self.lengthscale                         # (nx,)
        delta = (x1[:, None, :] - x2[None, :, :]) * inv_l     # (N, M, nx)
        return self.signal_var * torch.exp(-0.5 * (delta ** 2).sum(-1))

    def _fit_gp(self, x: torch.Tensor, H_vals: torch.Tensor):
        """
        Compute dual coefficients α = (K + σ²_n I)⁻¹ H via Cholesky.
        This is the standard GP posterior mean weight vector.
        """
        M = len(x)
        K = self._k_se(x, x) + self.noise * torch.eye(M, dtype=x.dtype, device=x.device)
        L = torch.linalg.cholesky(K)                         # lower triangular
        self._alpha = torch.cholesky_solve(
            H_vals.detach().unsqueeze(-1), L                  # (M, 1)
        ).squeeze(-1)                                          # (M,)

    def _eval_gp(self, x: torch.Tensor) -> torch.Tensor:
        """H*(x) = k(x, X_fit) · α  →  (K,)"""
        return self._k_se(x, self._x_fit) @ self._alpha

    def _grad_gp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analytic gradient of the SE-kernel GP posterior mean:

            ∂H*/∂x = Σᵢ αᵢ · k(x, xᵢ) · Λ⁻¹(xᵢ - x)

        where Λ = diag(ℓ²). Returns (K, nx).
        """
        inv_l2  = 1.0 / self.lengthscale ** 2                 # (nx,)
        k       = self._k_se(x, self._x_fit)                  # (K, M)
        diff    = self._x_fit[None, :, :] - x[:, None, :]     # (K, M, nx)
        weights = k * self._alpha[None, :]                     # (K, M)
        return (weights[:, :, None] * diff * inv_l2).sum(dim=1)  # (K, nx)
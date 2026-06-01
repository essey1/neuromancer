"""
hamiltonian_approximator.py
============================
Approximates sampled H values from GPPosterior with a callable
function H*(x) that can be evaluated at arbitrary points for ODE solving.

Preserves PHS structure — we approximate H, not f, so:
    ẋ = (J(x) - R(x)) · ∇H*(x) + G(x) · u
remains a valid Port-Hamiltonian system.

∇H*(x) is computed via autograd on a differentiable torch reimplementation
of the fitted interpolant.
"""

import torch
import torch.nn as nn
from scipy.interpolate import RBFInterpolator
import numpy as np


class HamiltonianApproximator(nn.Module):
    """
    Fits a callable approximation H*(x) to sampled H values from GPPosterior.

    Supports two approximation methods:
        'spline' : Thin-plate spline — default, recommended by the paper.
                   φ(r) = r² · log(r)
                   Smooth, minimises bending energy, good general choice.
        'rbf'    : Multiquadric RBF interpolation.
                   φ(r) = sqrt(r² + 1)
                   Good alternative for higher dimensional state spaces.

    ∇H*(x) is obtained via autograd on a differentiable torch
    reimplementation of the fitted interpolant — same basis function,
    same fit points, weights solved via lstsq.

    Args:
        method : 'spline' (default) or 'rbf'

    Usage:
        # after GPPosterior
        H_mean, H_var, H_samples = posterior(train_x, train_u,
                                              train_xdot, test_x)

        # fit approximator to one H sample
        approx = HamiltonianApproximator(method='spline')
        approx.fit(test_x, H_samples[0])

        # evaluate H* and ∇H* at new points
        H_star      = approx(new_x)            # (K,)
        grad_H_star = approx.gradient(new_x)   # (K, nx)

        # ensemble: one approximator per sample
        ensemble = []
        for i in range(n_samples):
            a = HamiltonianApproximator()
            a.fit(test_x, H_samples[i])
            ensemble.append(a)
    """

    # basis functions for scipy fitting
    _SCIPY_KERNELS = {
        'spline': 'thin_plate_spline',
        'rbf':    'multiquadric',
    }

    def __init__(self, method: str = 'spline'):
        super().__init__()

        if method not in self._SCIPY_KERNELS:
            raise ValueError(
                f"method must be 'spline' or 'rbf', got '{method}'"
            )

        self.method       = method
        self._interpolant = None   # scipy interpolant — set after fit()
        self._x_fit       = None   # fit points : (M, nx)
        self._H_fit       = None   # fit H values : (M,)
        self._weights     = None   # solved weights for torch eval : (M,)

    def fit(
        self,
        x:      torch.Tensor,
        H_vals: torch.Tensor,
    ):
        """
        Fit the approximator to sampled H values from GPPosterior.

        Args:
            x      : (M, nx)  states where H was sampled (test_x from GPPosterior)
            H_vals : (M,)     one row of H_samples from GPPosterior
        """
        self._x_fit = x.detach()
        self._H_fit = H_vals.detach()

        x_np = x.detach().cpu().numpy()
        H_np = H_vals.detach().cpu().numpy()

        # fit scipy interpolant
        self._interpolant = RBFInterpolator(
            x_np, H_np,
            kernel=self._SCIPY_KERNELS[self.method]
        )

        # pre-solve weights for the differentiable torch version
        # so gradient() doesn't need to solve lstsq at every call
        self._weights = self._solve_weights(self._x_fit, self._H_fit)
        return self

    def _basis(self, r2: torch.Tensor) -> torch.Tensor:
        """
        Apply the basis function to squared distances.

        thin plate spline : φ(r) = r² · log(r)  = 0.5 · r² · log(r²)
        multiquadric      : φ(r) = sqrt(r² + 1)

        Args:
            r2 : (K, M)  squared distances

        Returns:
            (K, M)
        """
        if self.method == 'spline':
            # numerically stable: 0.5 · r² · log(r²), zero where r²=0
            safe_r2 = r2.clamp(min=1e-10)
            return 0.5 * r2 * torch.log(safe_r2)
        else:
            return torch.sqrt(r2 + 1.0)

    def _solve_weights(
        self,
        x:      torch.Tensor,
        H_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve for RBF weights at the fit points so that:
            Φ · w = H_vals
        where Φ[i,j] = φ(||x[i] - x[j]||)

        Args:
            x      : (M, nx)
            H_vals : (M,)

        Returns:
            weights : (M,)
        """
        diff = x[:, None, :] - x[None, :, :]          # (M, M, nx)
        r2   = (diff ** 2).sum(-1)                     # (M, M)
        Phi  = self._basis(r2)                         # (M, M)

        # solve Φ · w = H_vals
        weights = torch.linalg.lstsq(
            Phi, H_vals.unsqueeze(-1)
        ).solution.squeeze(-1)                         # (M,)

        return weights

    def _torch_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable evaluation of H*(x) in pure torch.
        Uses pre-solved weights from fit() — autograd can flow through this.

        Args:
            x : (K, nx)

        Returns:
            H* : (K,)
        """
        x_c = self._x_fit.to(x.device)                # (M, nx)

        diff = x[:, None, :] - x_c[None, :, :]        # (K, M, nx)
        r2   = (diff ** 2).sum(-1)                     # (K, M)
        Phi  = self._basis(r2)                         # (K, M)

        return Phi @ self._weights.to(x.device)        # (K,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate H*(x) at new points using the scipy interpolant.
        More accurate than _torch_eval — use this for H* values.

        Args:
            x : (K, nx)  new states — K can be anything

        Returns:
            H* : (K,)
        """
        if self._interpolant is None:
            raise RuntimeError(
                "Call fit() before evaluating HamiltonianApproximator."
            )

        x_np = x.detach().cpu().numpy()
        H_np = self._interpolant(x_np)
        return torch.tensor(H_np, dtype=x.dtype, device=x.device)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇H*(x) via autograd on the differentiable torch evaluation.

        Args:
            x : (K, nx)  points where gradient is needed

        Returns:
            ∇H* : (K, nx)  — passed directly to ode.py
                             ẋ = (J(x)-R(x)) · ∇H*(x) + G(x) · u
        """
        if self._weights is None:
            raise RuntimeError(
                "Call fit() before computing gradient."
            )

        x_req = x.detach().requires_grad_(True)
        H     = self._torch_eval(x_req)               # (K,)
        grad  = torch.autograd.grad(
            H.sum(), x_req, create_graph=False
        )[0]                                          # (K, nx)
        return grad


class GPHamiltonianApproximator(nn.Module):
    """
    GP posterior mean as H* approximator.

    Fits H*(x) = k(x, X) @ K⁻¹ @ H_vals using the SE kernel.
    Gradient ∇H*(x) is computed analytically — no autograd, no noisy weights.

    Reuses the lengthscale learned by GPPHSProblem so the smoothing is
    physically calibrated to the system dynamics.

    Args:
        lengthscale : (nx,)  from learned['lengthscale']
        signal_var  : scalar from learned['signal_var']
        noise       : nugget added to the diagonal for numerical stability

    Usage:
        approx = GPHamiltonianApproximator(
                     lengthscale=learned['lengthscale'],
                     signal_var=learned['signal_var'],
                 ).fit(x_fit, H_sample)

        H_star   = approx(x)           # (K,)
        grad_H   = approx.gradient(x)  # (K, nx)
    """

    def __init__(
        self,
        lengthscale: torch.Tensor,
        signal_var:  torch.Tensor,
        noise:       float = 1e-4,
    ):
        super().__init__()
        self.register_buffer('lengthscale', lengthscale.detach().squeeze())
        self.register_buffer('signal_var',  signal_var.detach().squeeze())
        self.noise   = noise
        self._x_fit  = None
        self._alpha  = None   # (K⁻¹ @ H_vals) — solved once in fit()

    def _k(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """SE kernel  k(x1, x2) = σ²f · exp(-0.5 · ||x1-x2||²_Λ)  →  (N, M)"""
        inv_l = 1.0 / self.lengthscale
        delta  = (x1[:, None, :] - x2[None, :, :]) * inv_l   # (N, M, nx)
        return self.signal_var * torch.exp(-0.5 * (delta ** 2).sum(-1))

    def fit(self, x: torch.Tensor, H_vals: torch.Tensor):
        """Solve α = (K + noise·I)⁻¹ H_vals via Cholesky."""
        self._x_fit = x.detach()
        K = self._k(x, x) + self.noise * torch.eye(len(x), dtype=x.dtype)
        L = torch.linalg.cholesky(K)
        self._alpha = torch.cholesky_solve(
            H_vals.detach().unsqueeze(-1), L
        ).squeeze(-1)                                          # (M,)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """H*(x) = k(x, X) @ α  →  (K,)"""
        return self._k(x, self._x_fit) @ self._alpha

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        ∇H*(x) = Σᵢ αᵢ · k(x, xᵢ) · Λ⁻²(xᵢ - x)  →  (K, nx)

        From:  ∂/∂x k(x, xᵢ) = k(x, xᵢ) · Λ⁻²(xᵢ - x)
        """
        inv_l2  = 1.0 / self.lengthscale ** 2                  # (nx,)
        k       = self._k(x, self._x_fit)                      # (K, M)
        diff    = self._x_fit[None, :, :] - x[:, None, :]      # (K, M, nx)
        weights = k * self._alpha[None, :]                      # (K, M)
        return (weights[:, :, None] * diff * inv_l2).sum(dim=1) # (K, nx)
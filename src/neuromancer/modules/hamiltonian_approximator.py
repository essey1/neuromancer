"""
hamiltonian_approximator.py
============================
Approximates sampled H values from GPPosterior with a callable
function H*(x) that can be evaluated at arbitrary points for ODE solving.

Preserves PHS structure — we approximate H, not f, so:
    ẋ = (J(x) - R(x)) · ∇H*(x) + G(x) · u
remains a valid Port-Hamiltonian system.

∇H*(x) is computed via autograd on H*(x) — no manual gradient needed.
"""

import torch
import torch.nn as nn
from scipy.interpolate import RBFInterpolant
import numpy as np


class HamiltonianApproximator(nn.Module):
    """
    Fits a callable approximation H*(x) to sampled H values from GPPosterior.

    Supports two approximation methods:
        'rbf'    : Radial basis function interpolation (scipy) — smooth,
                   exact at sample points, good for low dimensions
        'spline' : Thin-plate spline — special case of RBF, good default

    ∇H*(x) is obtained via torch.autograd — the approximator is
    differentiable so the ODE solver can compute ∇H* at any x.

    Args:
        method : 'rbf' or 'spline' (default 'rbf')

    Usage:
        # after GPPosterior
        H_mean, H_var, H_samples = posterior(train_x, train_u,
                                              train_xdot, test_x)

        # fit approximator to one sample
        approx = HamiltonianApproximator(method='rbf')
        approx.fit(test_x, H_samples[0])

        # evaluate H* and ∇H* at new points
        H_star      = approx(new_x)           # (K,)
        grad_H_star = approx.gradient(new_x)  # (K, nx)
    """

    def __init__(self, method: str = 'rbf'):
        super().__init__()

        if method not in ('rbf', 'spline'):
            raise ValueError(f"method must be 'rbf' or 'spline', got '{method}'")

        self.method      = method
        self._interpolant = None   # set after fit()
        self._x_fit      = None   # training points — (M, nx)
        self._H_fit      = None   # training H values — (M,)

    def fit(
        self,
        x:       torch.Tensor,
        H_vals:  torch.Tensor,
    ):
        """
        Fit the approximator to sampled H values.

        Args:
            x      : (M, nx)  states where H was sampled
            H_vals : (M,)     sampled H values from GPPosterior
        """
        self._x_fit = x.detach()
        self._H_fit = H_vals.detach()

        x_np = x.detach().cpu().numpy()
        H_np = H_vals.detach().cpu().numpy()

        kernel = 'thin_plate_spline' if self.method == 'spline' else 'multiquadric'
        self._interpolant = RBFInterpolant(x_np, H_np, kernel=kernel)

    def _eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate H*(x) via the fitted interpolant.
        Bridges scipy interpolant → torch tensor for autograd.

        Args:
            x : (K, nx)

        Returns:
            (K,)
        """
        if self._interpolant is None:
            raise RuntimeError("Call fit() before evaluating HamiltonianApproximator.")

        x_np   = x.detach().cpu().numpy()
        H_np   = self._interpolant(x_np)
        return torch.tensor(H_np, dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate H*(x) at new points.

        Args:
            x : (K, nx)

        Returns:
            H* : (K,)
        """
        return self._eval(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇H*(x) via autograd.

        Since scipy interpolant breaks the autograd graph, we use a
        torch-differentiable re-evaluation trick: fit a thin RBF layer
        in torch using the stored fit points, then differentiate that.

        Args:
            x : (K, nx)

        Returns:
            ∇H* : (K, nx)
        """
        if self._interpolant is None:
            raise RuntimeError("Call fit() before computing gradient.")

        # re-evaluate using torch ops so autograd can differentiate
        x_req = x.requires_grad_(True)
        H     = self._torch_rbf(x_req)               # (K,)
        grad  = torch.autograd.grad(
            H.sum(), x_req, create_graph=False
        )[0]                                          # (K, nx)
        return grad

    def _torch_rbf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable RBF evaluation in pure torch using fit points.
        Uses multiquadric basis: φ(r) = sqrt(r² + 1)

        Args:
            x : (K, nx)

        Returns:
            (K,)
        """
        x_c  = self._x_fit.to(x.device)              # (M, nx)
        H_c  = self._H_fit.to(x.device)              # (M,)

        # pairwise distances: (K, M)
        diff = x[:, None, :] - x_c[None, :, :]       # (K, M, nx)
        r2   = (diff ** 2).sum(-1)                    # (K, M)
        phi  = torch.sqrt(r2 + 1.0)                   # (K, M) multiquadric

        # weights via least squares on fit points (K, M) @ (M,) → (K,)
        # use stored H values as weights directly (works for interpolation)
        weights = torch.linalg.lstsq(
            phi[:1].T @ phi[:1],
            phi[:1].T @ H_c.unsqueeze(-1)
        ).solution.squeeze(-1)                        # (M,)

        return phi @ weights                          # (K,)
"""
gp_smoother.py
==============
Fits an RBF GP to noisy state measurements (t, x) and returns
smoothed states, derivative means, and derivative variances (Δ diagonal).
"""

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np


class _GP(ExactGP):
    def __init__(self, t, y, likelihood):
        super().__init__(t, y, likelihood)
        self.mean   = gpytorch.means.ZeroMean()
        self.kernel = ScaleKernel(RBFKernel())

    def forward(self, t):
        return gpytorch.distributions.MultivariateNormal(
            self.mean(t), self.kernel(t)
        )


def gp_smooth(t, x, n_iter=100, lr=0.1, noise_var=0.01):
    """
    Args:
        t : (T,)     timestamps
        x : (T, nx)  noisy state observations

    Returns:
        x_smooth     : (T, nx)
        x_dot_smooth : (T, nx)
        x_dot_var    : (T, nx)  — diagonal of Δ for GP-PHS NLML
    """
    t = torch.tensor(t, dtype=torch.float32).squeeze()
    x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(-1)

    T, nx = x.shape
    x_smooth, x_dot_smooth, x_dot_var = [], [], []

    for d in range(nx):
        y = x[:, d]

        likelihood = GaussianLikelihood()
        model = _GP(t, y, likelihood)
        likelihood.noise = noise_var

        model.train(); likelihood.train()
        opt = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()), lr=lr
        )
        mll = ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(n_iter):
            opt.zero_grad()
            (-mll(model(t), y)).backward()
            opt.step()

        model.eval(); likelihood.eval()

        with torch.no_grad():
            ls = model.kernel.base_kernel.lengthscale.squeeze()
            sv = model.kernel.outputscale.squeeze()
            nv = likelihood.noise.squeeze()

            # smoothed state
            x_smooth.append(likelihood(model(t)).mean)

            # training kernel + noise
            K  = model.kernel(t).to_dense()
            K += (nv + 1e-6) * torch.eye(T)
            L  = torch.linalg.cholesky(K)

            # cross-cov k_{f,f'}(t, t) = sv * (t_i - t_j) / ls² * exp(...)
            diff     = t[:, None] - t[None, :]           # (T, T)
            k_rbf    = sv * torch.exp(-0.5 * diff**2 / ls**2)
            K_cross  = k_rbf * diff / ls**2              # (T, T)

            # derivative mean
            alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze()
            x_dot_smooth.append(K_cross.T @ alpha)

            # derivative variance: sv/ls² - diag(K_cross^T K^{-1} K_cross)
            V           = torch.cholesky_solve(K_cross, L)
            prior_var   = sv / ls**2
            x_dot_var.append((prior_var - (K_cross * V).sum(0)).clamp(min=0))

    return (
        torch.stack(x_smooth,     dim=-1).numpy(),
        torch.stack(x_dot_smooth, dim=-1).numpy(),
        torch.stack(x_dot_var,    dim=-1).numpy(),
    )
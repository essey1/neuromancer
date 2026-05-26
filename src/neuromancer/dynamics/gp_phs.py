"""
gp_phs.py
=========
Component 1 — PHSMatrices : structural definition of J, R, G
Component 2 — PHSKernel   : physics-structured GP covariance

Port-Hamiltonian System:
    ẋ = (J(x) - R(x)) · ∇H  +  G(x) · u

Kernel:
    k_phs(x, x') = σ²f · (J(x)-R(x)) · H_rbf(x,x') · (J(x')-R(x'))ᵀ

    H_rbf(x,x')ᵢⱼ = k_rbf(x,x') · [Λ⁻¹ᵢⱼ - (Λ⁻¹δ)ᵢ · (Λ⁻¹δ)ⱼ]
    δ = x - x'

Entry signature for all matrices (J, R, G):
    callable(x: Tensor) -> Tensor of shape (batch,)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import Kernel
from linear_operator.operators import DenseLinearOperator
from typing import Callable, Dict, Optional, Tuple

Entry = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Helper: extract nn.Parameters from a callable's closure
# ---------------------------------------------------------------------------

def _extract_closure_parameters(fn) -> Dict[str, nn.Parameter]:
    """
    Walk a callable's __closure__ and collect any nn.Parameter or
    nn.Module objects so they can be registered with PyTorch.

    Works for plain lambdas, nested defs, and nn.Module instances.
    """
    if isinstance(fn, nn.Module):
        return dict(fn.named_parameters())

    params = {}
    for i, cell in enumerate(getattr(fn, "__closure__", None) or []):
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, nn.Parameter):
            params[f"closure_{i}"] = val
        elif isinstance(val, nn.Module):
            for name, p in val.named_parameters():
                params[f"closure_{i}_{name}"] = p
    return params


# ---------------------------------------------------------------------------
# Component 1 — PHSMatrices
# ---------------------------------------------------------------------------

class PHSMatrices(nn.Module):
    """
    Structural definition of the PHS matrices J, R, G.

    All three matrices are functions of state x only.
    Control input u is applied externally: ẋ = (J-R)∇H + G·u

    Args:
        nx          : state dimension
        nu          : input dimension (needed to size G)
        J_upper     : upper-triangle entries of J (i < j only)
                      dict[(i,j)] -> callable(x) -> (batch,)
                      skew-symmetry enforced: J[j,i] = -J[i,j]
        R_diag      : diagonal entries of R
                      dict[i] -> callable(x) -> (batch,)
                      PSD enforced: R = diag(d²)
        G_full      : full (nx, nu) entries of G
                      dict[(i,j)] -> callable(x) -> (batch,)
        extra_parameters : explicitly register params the closure scan misses

    Example
    -------
        w = nn.Parameter(torch.tensor(1.0))

        phs = PHSMatrices(
            nx=3, nu=2,
            J_upper={
                (0, 1): lambda x: torch.ones(x.shape[0]),
                (0, 2): lambda x: x[:, 1],
                (1, 2): lambda x: w * torch.sin(x[:, 0]),  # w is auto-detected
            },
            R_diag={
                0: lambda x: torch.ones(x.shape[0]) * 0.5,
                1: lambda x: x[:, 2].abs(),
                2: lambda x: torch.ones(x.shape[0]) * 0.1,
            },
            G_full={
                (0, 0): lambda x: torch.ones(x.shape[0]),
                (2, 1): lambda x: x[:, 0],
            },
        )

        J, R, G = phs(x)
    """

    def __init__(
        self,
        nx: int,
        nu: int,
        J_upper: Dict[Tuple[int, int], Entry],
        R_diag:  Dict[int, Entry],
        G_full:  Dict[Tuple[int, int], Entry],
        extra_parameters: Optional[Dict[str, nn.Parameter]] = None,
    ):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self._J_upper = J_upper
        self._R_diag  = R_diag
        self._G_full  = G_full

        # ── validate indices ───────────────────────────────────────────────
        for (i, j) in J_upper:
            if not (0 <= i < j < nx):
                raise ValueError(
                    f"J_upper key ({i},{j}) invalid — need 0 <= i < j < nx={nx}"
                )
        for i in R_diag:
            if not (0 <= i < nx):
                raise ValueError(
                    f"R_diag key {i} invalid — need 0 <= i < nx={nx}"
                )
        for (i, j) in G_full:
            if not (0 <= i < nx and 0 <= j < nu):
                raise ValueError(
                    f"G_full key ({i},{j}) invalid — need i < nx={nx}, j < nu={nu}"
                )

        # ── warn on incomplete R diagonal ──────────────────────────────────
        missing = [i for i in range(nx) if i not in R_diag]
        if missing:
            import warnings
            warnings.warn(
                f"R_diag missing entries for row(s) {missing} — R will be degenerate.",
                UserWarning,
                stacklevel=2,
            )

        # ── register parameters found in closures ──────────────────────────
        for tag, entries in [
            ("J", {f"{i}_{j}": fn for (i, j), fn in J_upper.items()}),
            ("R", {f"{i}":     fn for i,       fn in R_diag.items()}),
            ("G", {f"{i}_{j}": fn for (i, j), fn in G_full.items()}),
        ]:
            for key, fn in entries.items():
                for pname, param in _extract_closure_parameters(fn).items():
                    reg = f"{tag}_{key}_{pname}"
                    if reg not in dict(self.named_parameters()):
                        self.register_parameter(reg, param)

        if extra_parameters:
            for name, param in extra_parameters.items():
                self.register_parameter(f"extra_{name}", param)

    # ── matrix constructors ────────────────────────────────────────────────

    def get_J(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns skew-symmetric J of shape (batch, nx, nx).
        User supplies upper triangle; lower is filled as J[j,i] = -J[i,j].
        """
        batch = x.shape[0]
        J = torch.zeros(batch, self.nx, self.nx, dtype=x.dtype, device=x.device)
        for (i, j), fn in self._J_upper.items():
            J[:, i, j] = fn(x)
        return J - J.transpose(-1, -2)

    def get_R(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns PSD R of shape (batch, nx, nx).
        User supplies diagonal d; R = diag(d²) guarantees R >= 0.
        """
        batch = x.shape[0]
        d = torch.zeros(batch, self.nx, dtype=x.dtype, device=x.device)
        for i, fn in self._R_diag.items():
            d[:, i] = fn(x)
        return torch.diag_embed(d ** 2)

    def get_G(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns unconstrained G of shape (batch, nx, nu).
        User supplies the full structure; absent entries are zero.
        u is applied externally: ẋ = (J-R)∇H + G·u
        """
        batch = x.shape[0]
        G = torch.zeros(batch, self.nx, self.nu, dtype=x.dtype, device=x.device)
        for (i, j), fn in self._G_full.items():
            G[:, i, j] = fn(x)
        return G

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (J, R, G) evaluated at x. Shape: (batch, nx, nx) each."""
        return self.get_J(x), self.get_R(x), self.get_G(x)


# ---------------------------------------------------------------------------
# Component 2 — PHSKernel
# ---------------------------------------------------------------------------

class PHSKernel(Kernel):
    """
    Port-Hamiltonian System kernel for GPyTorch.

    k_phs(x, x') = σ²f · (J(x)-R(x)) · H_rbf(x,x') · (J(x')-R(x'))ᵀ

    Args:
        phs_matrices : PHSMatrices instance — provides get_J(x), get_R(x)
        nx           : state dimension

    Learnable parameters:
        raw_lengthscale : (nx,)  diagonal of Λ, constrained positive via softplus
        raw_signal_var  : ()     σ²f, constrained positive via softplus
    """

    is_stationary = False

    def __init__(self, phs_matrices: PHSMatrices, nx: int, **kwargs):
        super().__init__(**kwargs)
        self.phs = phs_matrices
        self.nx  = nx

        self.register_parameter(
            "raw_lengthscale",
            nn.Parameter(torch.zeros(nx))
        )
        self.register_parameter(
            "raw_signal_var",
            nn.Parameter(torch.zeros(1))
        )

        self.softplus = nn.Softplus()

    @property
    def lengthscale(self) -> torch.Tensor:
        """Λ diagonal — (nx,) strictly positive."""
        return self.softplus(self.raw_lengthscale)

    @property
    def signal_var(self) -> torch.Tensor:
        """σ²f — strictly positive scalar."""
        return self.softplus(self.raw_signal_var)

    def _rbf_and_hessian(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scalar RBF values and their nx×nx Hessian blocks
        for all (N, M) pairs. Fully vectorised — no loops.

        Args:
            x1 : (N, nx)
            x2 : (M, nx)

        Returns:
            k_rbf : (N, M)
            H     : (N, M, nx, nx)
        """
        inv_lambda  = 1.0 / self.lengthscale           # (nx,)
        inv_lambda2 = inv_lambda ** 2                  # (nx,)

        # weighted difference δ = Λ⁻¹(x1 - x2) : (N, M, nx)
        delta = (x1[:, None, :] - x2[None, :, :]) * inv_lambda

        # scalar RBF : (N, M)
        k_rbf = torch.exp(-0.5 * (delta ** 2).sum(dim=-1))

        # outer product δδᵀ per pair : (N, M, nx, nx)
        outer = delta[:, :, :, None] * delta[:, :, None, :]

        # Λ⁻¹ diagonal matrix : (nx, nx)
        lambda_inv_diag = torch.diag(inv_lambda2)

        # Hessian H[n,m,i,j] = k_rbf[n,m] · (Λ⁻¹ᵢⱼ - δᵢδⱼ) : (N, M, nx, nx)
        H = k_rbf[:, :, None, None] * (lambda_inv_diag - outer)

        return k_rbf, H

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, **params
    ) -> DenseLinearOperator:
        """
        Compute the full PHS kernel matrix.

        Args:
            x1 : (N, nx)
            x2 : (M, nx)

        Returns:
            DenseLinearOperator of shape (N·nx, M·nx)
        """
        N = x1.shape[0]
        M = x2.shape[0]

        # step 1 & 2: RBF scalar values + Hessian blocks
        _, H = self._rbf_and_hessian(x1, x2)            # (N, M, nx, nx)

        # step 3: (J-R) evaluated at each point — state dependent only
        JR1 = self.phs.get_J(x1) - self.phs.get_R(x1)  # (N, nx, nx)
        JR2 = self.phs.get_J(x2) - self.phs.get_R(x2)  # (M, nx, nx)

        # step 4: sandwich  σ²f · JR1 · H · JR2ᵀ
        # broadcast over the missing dimension before matmul
        JR1 = JR1[:, None, :, :]                         # (N, 1, nx, nx)
        JR2 = JR2[None, :, :, :]                         # (1, M, nx, nx)

        K_phs = self.signal_var * (JR1 @ H @ JR2.transpose(-1, -2))
        #                                                   (N, M, nx, nx)

        # step 5: block reshape (N, M, nx, nx) → (N·nx, M·nx)
        K_out = K_phs.permute(0, 2, 1, 3).reshape(N * self.nx, M * self.nx)

        return DenseLinearOperator(K_out)
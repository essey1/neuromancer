"""
gp_phs.py
=========
Component 1 — PHSMatrices : structural definition of J, R, G
Component 2 — PHSKernel   : physics-structured GP covariance
Component 3 — PHSMeanFunction : GP prior mean m(x, u) = G(x) · u
Component 4 — GPPHSModel      : full GP model combining mean + kernel

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

        # weighted difference δₖ = (x1ₖ - x2ₖ)/λₖ : (N, M, nx)
        delta = (x1[:, None, :] - x2[None, :, :]) * inv_lambda

        # scalar RBF : (N, M)
        k_rbf = torch.exp(-0.5 * (delta ** 2).sum(dim=-1))

        # Mixed Hessian: ∂²k/∂xᵢ∂x'ⱼ = k·[δᵢⱼ/λᵢ² − (Δᵢ/λᵢ²)(Δⱼ/λⱼ²)]
        # outer product needs Δ/λ² not Δ/λ; divide delta by λ again.
        delta2 = delta * inv_lambda                             # (N, M, nx): Δ/λ²
        outer  = delta2[:, :, :, None] * delta2[:, :, None, :]  # (N, M, nx, nx)

        # Λ⁻² diagonal matrix : (nx, nx)
        lambda_inv_diag = torch.diag(inv_lambda2)

        # H[n,m,i,j] = k_rbf[n,m] · (δᵢⱼ/λᵢ² − (Δᵢ/λᵢ²)(Δⱼ/λⱼ²)) : (N, M, nx, nx)
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

# ---------------------------------------------------------------------------
# Component 3 — PHSMeanFunction
# ---------------------------------------------------------------------------

class PHSMeanFunction(gpytorch.means.Mean):
    """
    Prior mean function for the PHS GP model.

    Computes m(x, u) = G(x) · u for each data point.

    This class expects x and u to be passed separately — concatenation
    and splitting is handled by GPPHSModel.forward, not here.

    Args:
        phs_matrices : PHSMatrices instance — provides get_G(x)
        nx           : state dimension
        nu           : input dimension

    Input:
        x : (N, nx)
        u : (N, nu)

    Output:
        (N·nx,) — flattened mean vector
    """

    def __init__(self, phs_matrices, nx, nu):
        super().__init__()
        self.phs = phs_matrices
        self.nx  = nx
        self.nu  = nu

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # G(x) : (N, nx, nu)
        G = self.phs.get_G(x)

        # batched matrix-vector product: G(xₙ) · uₙ for each n
        # u.unsqueeze(-1) : (N, nu, 1)
        # G @ u           : (N, nx, 1) → squeeze → (N, nx)
        mean = (G @ u.unsqueeze(-1)).squeeze(-1)    # (N, nx)

        # flatten to (N·nx,) to match the (N·nx, M·nx) kernel output
        return mean.reshape(-1)


# ---------------------------------------------------------------------------
# Component 4 — GPPHSModel
# ---------------------------------------------------------------------------

class GPPHSModel(nn.Module):
    """
    Full GP model for Port-Hamiltonian dynamics.

    GP prior:
        ẋ ~ GP(G(x)u, k_phs(x, x'))

    x and u are always passed separately.
    The kernel receives x only (k_phs is a function of x alone).
    The mean receives x and u separately (m = G(x)·u needs both).

    Note: does NOT inherit from ExactGP. GPyTorch's ExactGP assumes
    N inputs → N scalar outputs, but PHS has N inputs → N·nx vector
    outputs, so the noise shape conflicts. NLML is computed in GPPHSLoss.

    Args:
        train_x    : (N, nx)  — kept for API compatibility, not stored
        train_u    : (N, nu)  — kept for API compatibility, not stored
        train_xdot : (N·nx,) — kept for API compatibility, not stored
        likelihood : gpytorch.likelihoods.GaussianLikelihood instance
        phs_matrices : PHSMatrices instance
        nx           : state dimension
        nu           : input dimension

    Usage:
        model = GPPHSModel(train_x, train_u, train_xdot,
                           likelihood, phs_matrices, nx, nu)

        pred = model(x, u)   # MultivariateNormal, mean (N·nx,), covar (N·nx, N·nx)
    """

    def __init__(
        self,
        train_x:    torch.Tensor,
        train_u:    torch.Tensor,
        train_xdot: torch.Tensor,
        likelihood,
        phs_matrices,
        nx: int,
        nu: int,
    ):
        super().__init__()

        self.nx = nx
        self.nu = nu

        self.mean_module  = PHSMeanFunction(phs_matrices, nx, nu)
        self.covar_module = PHSKernel(phs_matrices, nx)

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x : (N, nx) — state
            u : (N, nu) — control input

        Returns:
            MultivariateNormal with
                mean  : (N·nx,)
                covar : (N·nx, N·nx)
        """
        mean  = self.mean_module.forward(x, u)    # (N·nx,)
        # Call .forward() directly: Kernel.__call__ wraps the result in
        # LazyEvaluatedKernelTensor and validates shape as (N, N), but our
        # kernel intentionally returns (N·nx, N·nx).
        covar = self.covar_module.forward(x, x)  # (N·nx, N·nx)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# Component 5 — GPPosterior
# ---------------------------------------------------------------------------

class GPPosterior(nn.Module):
    """
    GP posterior for Port-Hamiltonian dynamics.

    Forms the joint distribution over [Ẋ, H(x*)] from equation (14):

        [Ẋ       ]       [K_phs            k_ẋH(X, x*)  ]
        [H(x*)]  ~ N(0,  [k_ẋH(X,x*)ᵀ     k_HH(x*, x*) ])

    Conditions on training Ẋ to obtain posterior over H(x*):

        μ_H = k_ẋH(X,x*)ᵀ · (K_phs + Δ)⁻¹ · Ẋ
        Σ_H = k_HH(x*,x*) - k_ẋH(X,x*)ᵀ · (K_phs + Δ)⁻¹ · k_ẋH(X,x*)

    Samples from this posterior are then passed to
    HamiltonianApproximator to get a callable H*(x).

    Args:
        model        : trained GPPHSModel
        likelihood   : trained GaussianLikelihood
        phs_matrices : trained PHSMatrices — carries learned J, R, G
        lengthscale  : (nx,)  learned Λ diagonal
        signal_var   : ()     learned σ_f
        noise_var    : ()     learned noise variance

    Usage:
        learned  = problem.train(train_x, train_u, train_xdot)
        posterior = GPPosterior(**learned)

        # get posterior over H at test points
        H_mean, H_var, H_samples = posterior(train_x, train_xdot, test_x)
    """

    def __init__(
        self,
        model,
        likelihood,
        phs_matrices,
        lengthscale:  torch.Tensor,
        signal_var:   torch.Tensor,
        noise_var:    torch.Tensor,
        **kwargs,     # absorb extra keys from problem.train() dict (e.g. nlml_history)
    ):
        super().__init__()
        self.model        = model
        self.likelihood   = likelihood
        self.phs          = phs_matrices

        self.register_buffer("lengthscale", lengthscale)   # (nx,)
        self.register_buffer("signal_var",  signal_var)    # scalar
        self.register_buffer("noise_var",   noise_var)     # scalar

        # enforce eval mode — no gradients needed
        self.model.eval()
        self.likelihood.eval()

    # ── kernel helpers ─────────────────────────────────────────────────────

    def _k_HH(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scalar RBF kernel between H values.

            k_HH(x, x') = σ²f · exp(-0.5·||x-x'||²_Λ)

        Args:
            x1 : (N, nx)
            x2 : (M, nx)

        Returns:
            (N, M)
        """
        inv_lambda = 1.0 / self.lengthscale              # (nx,)
        delta      = (x1[:, None, :] - x2[None, :, :])  # (N, M, nx)
        sq_dist    = ((delta * inv_lambda) ** 2).sum(-1)        # (N, M)
        return self.signal_var * torch.exp(-0.5 * sq_dist)      # (N, M)

    def _k_xdotH(
        self,
        x_train: torch.Tensor,
        x_test:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-kernel between ẋ at training points and H at test points.

            k_ẋH(x, x') = (J(x)-R(x)) · ∇_x k_HH(x, x')

        where:
            ∇_x k_HH(x,x') = k_HH(x,x') · (-Λ⁻¹(x-x'))

        so:
            k_ẋH(x,x') = σ²f · (J-R)(x) · (-Λ⁻¹(x-x')) · exp(-||x-x'||²_Λ)

        Args:
            x_train : (N, nx)  training states
            x_test  : (M, nx)  test states

        Returns:
            (N·nx, M)  — one nx-vector per training point per test point
        """
        N  = x_train.shape[0]
        M  = x_test.shape[0]
        nx = self.phs.nx

        inv_lambda = 1.0 / self.lengthscale                        # (nx,)

        # scalar k_HH values : (N, M)
        k_hh = self._k_HH(x_train, x_test)

        # weighted difference : (N, M, nx)
        delta = (x_train[:, None, :] - x_test[None, :, :])        # (N, M, nx)

        # ∇_x k_HH : (N, M, nx)  — gradient of k_HH w.r.t. x_train
        grad_k = -k_hh[:, :, None] * delta * (inv_lambda ** 2)    # (N, M, nx)

        # (J-R) at training points : (N, nx, nx)
        JR = self.phs.get_J(x_train) - self.phs.get_R(x_train)

        # k_ẋH[n, m] = JR[n] @ grad_k[n, m]  : (N, M, nx)
        # JR       : (N, 1, nx, nx)
        # grad_k   : (N, M, nx, 1)
        k_xdotH = (JR[:, None, :, :] @
                   grad_k[:, :, :, None]).squeeze(-1)              # (N, M, nx)

        # reshape to (N·nx, M)
        return k_xdotH.permute(0, 2, 1).reshape(N * nx, M)

    # ── posterior over H ───────────────────────────────────────────────────

    def _get_K_phs_plus_noise(
        self,
        x_train: torch.Tensor,
        u_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the training kernel matrix (K_phs + Δ) from the trained model.
        GPyTorch has already computed and cached this via the Cholesky.

        Returns:
            (N·nx, N·nx)
        """
        with torch.no_grad():
            # get the lazy covariance from the trained model
            train_dist = self.model(x_train, u_train)
            K = train_dist.lazy_covariance_matrix

            # add noise (Δ) — GaussianLikelihood adds noise_var * I
            noise = self.noise_var * torch.eye(
                K.shape[-1], dtype=x_train.dtype, device=x_train.device
            )
            return K.to_dense() + noise                            # (N·nx, N·nx)

    def forward(
        self,
        train_x:    torch.Tensor,
        train_u:    torch.Tensor,
        train_xdot: torch.Tensor,
        test_x:     torch.Tensor,
        n_samples:  int = 10,
    ):
        """
        Compute posterior over H(x*) conditioned on training Ẋ.

        Args:
            train_x    : (N, nx)   training states
            train_u    : (N, nu)   training control inputs
            train_xdot : (N, nx)   training state derivatives
            test_x     : (M, nx)   test states to sample H at
            n_samples  : number of H samples to draw

        Returns:
            H_mean    : (M,)         posterior mean of H at test points
            H_var     : (M,)         posterior variance of H at test points
            H_samples : (n_samples, M)  samples from posterior over H
        """
        N  = train_x.shape[0]
        M  = test_x.shape[0]
        nx = self.phs.nx

        with torch.no_grad():

            # ── build cross and self kernels ───────────────────────────────
            # k_ẋH(X, x*) : (N·nx, M)
            K_xdotH = self._k_xdotH(train_x, test_x)

            # k_HH(x*, x*) : (M, M)
            K_HH = self._k_HH(test_x, test_x)

            # (K_phs + Δ) : (N·nx, N·nx)
            K_noise = self._get_K_phs_plus_noise(train_x, train_u)

            # ── solve (K_phs + Δ)⁻¹ · k_ẋH via Cholesky ──────────────────
            n_k = K_noise.shape[0]
            jitter = 1e-6 * torch.eye(n_k, dtype=K_noise.dtype, device=K_noise.device)
            L = torch.linalg.cholesky(K_noise + jitter)           # (N·nx, N·nx)

            # subtract prior mean G(x)·u — posterior conditions on the
            # Hamiltonian residual (J-R)∇H, not the full ẋ which includes G·u
            G_train = self.phs.get_G(train_x)                     # (N, nx, nu)
            Gu = (G_train @ train_u.unsqueeze(-1)).squeeze(-1)    # (N, nx)
            xdot_residual = train_xdot - Gu                       # (N, nx)
            xdot_flat = xdot_residual.reshape(-1)                 # (N·nx,)

            # α = (K_phs + Δ)⁻¹ · (Ẋ - G·u) : (N·nx,)
            alpha = torch.cholesky_solve(
                xdot_flat.unsqueeze(-1), L
            ).squeeze(-1)                                          # (N·nx,)

            # V = (K_phs + Δ)⁻¹ · k_ẋH : (N·nx, M)
            V = torch.cholesky_solve(K_xdotH, L)                  # (N·nx, M)

            # ── posterior mean and covariance ──────────────────────────────
            # μ_H = k_ẋH(X,x*)ᵀ · α : (M,)
            H_mean = K_xdotH.T @ alpha                             # (M,)

            # Σ_H = k_HH - k_ẋH(X,x*)ᵀ · V : (M, M)
            H_cov  = K_HH - K_xdotH.T @ V                         # (M, M)

            # clamp diagonal for numerical stability
            H_var  = H_cov.diagonal().clamp(min=0.0)               # (M,)

            # ── sample from posterior ──────────────────────────────────────
            # H_cov is theoretically PSD but numerically indefinite due to
            # floating-point accumulation in the Schur complement. We are
            # inside torch.no_grad() so eigendecomposition is safe and cheap.
            eigvals, eigvecs = torch.linalg.eigh(H_cov)   # ascending
            eigvals_pos = eigvals.clamp(min=0.0)           # nearest PSD
            L_H = eigvecs @ torch.diag(eigvals_pos.sqrt())  # (M, M)
            eps = torch.randn(n_samples, M,
                              dtype=test_x.dtype, device=test_x.device)
            H_samples = H_mean[None, :] + (eps @ L_H.T)   # (n_samples, M)

        return H_mean, H_var, H_samples
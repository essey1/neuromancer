from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Tuple

Entry = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _extract_closure_parameters(fn: Entry) -> Dict[str, nn.Parameter]:
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


class PHSMatrices(nn.Module):
    """
    Holds the structure of J, R, G for a Port-Hamiltonian System.

    Args:
        nx: state dimension
        nu: input dimension
        J_upper: upper-triangle entries only — dict[(i,j)] -> callable(x,u) -> (batch,)
                 skew-symmetry enforced automatically: J[j,i] = -J[i,j]
        R_diag:  diagonal entries only — dict[i] -> callable(x,u) -> (batch,)
                 PSD enforced via R = diag(d^2)
        G_full:  full (nx, nu) structure — dict[(i,j)] -> callable(x,u) -> (batch,)
        extra_parameters: manually register any params the closure scan misses
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

        # validate J: upper triangle only
        for (i, j) in J_upper:
            if not (0 <= i < j < nx):
                raise ValueError(f"J_upper key ({i},{j}) invalid — need 0 <= i < j < nx={nx}")

        # validate R: diagonal only
        for i in R_diag:
            if not (0 <= i < nx):
                raise ValueError(f"R_diag key {i} invalid — need 0 <= i < nx={nx}")

        # validate G: full matrix
        for (i, j) in G_full:
            if not (0 <= i < nx and 0 <= j < nu):
                raise ValueError(f"G_full key ({i},{j}) invalid — need i < nx={nx}, j < nu={nu}")

        # warn if R diagonal is incomplete
        missing = [i for i in range(nx) if i not in R_diag]
        if missing:
            import warnings
            warnings.warn(
                f"R_diag missing entries for row(s) {missing} — R will be degenerate.",
                UserWarning, stacklevel=2,
            )

        # register parameters found in closures
        for tag, entries in [("J", {f"{i}_{j}": fn for (i,j),fn in J_upper.items()}),
                              ("R", {f"{i}":     fn for i,    fn in R_diag.items()}),
                              ("G", {f"{i}_{j}": fn for (i,j),fn in G_full.items()})]:
            for key, fn in entries.items():
                for pname, param in _extract_closure_parameters(fn).items():
                    reg = f"{tag}_{key}_{pname}"
                    if reg not in dict(self.named_parameters()):
                        self.register_parameter(reg, param)

        if extra_parameters:
            for name, param in extra_parameters.items():
                self.register_parameter(f"extra_{name}", param)

    def get_J(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        J = torch.zeros(batch, self.nx, self.nx, dtype=x.dtype, device=x.device)
        for (i, j), fn in self._J_upper.items():
            J[:, i, j] = fn(x, u)
        return J - J.transpose(-1, -2)  # enforces skew-symmetry

    def get_R(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        d = torch.zeros(batch, self.nx, dtype=x.dtype, device=x.device)
        for i, fn in self._R_diag.items():
            d[:, i] = fn(x, u)
        return torch.diag_embed(d ** 2)  # enforces PSD

    def get_G(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        G = torch.zeros(batch, self.nx, self.nu, dtype=x.dtype, device=x.device)
        for (i, j), fn in self._G_full.items():
            G[:, i, j] = fn(x, u)
        return G

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_J(x, u), self.get_R(x, u), self.get_G(x, u)



import numpy as np
import torch

from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss, BarrierLoss

from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays


# parameterized quadratic programming
def lossQuadratic(aggLoss, weight):
    # parameters
    p = variable("p")
    p1, p2 = p[:, 0], p[:, 1]
    # variables
    x = variable("x")
    x1, x2 = x[:, 0], x[:, 1]
    # objective function
    f = x1 ** 2 + x2 ** 2
    obj = [f.minimize(weight=1.0, name="obj")]
    # constraints
    Q_con = weight
    con_1 = Q_con * (x1 + x2 - p1 >= 0)
    con_1.name = "c1"
    con_2 = Q_con * (x1 + x2 - p1 <= 5)
    con_2.name = "c2"
    con_3 = Q_con * (x1 - x2 + p2 <= 5)
    con_3.name = "c3"
    con_4 = Q_con * (x1 - x2 + p2 >= 0)
    con_4.name = "c4"
    cons = [con_1, con_2, con_3, con_4]
    # loss
    loss = aggLoss(obj, cons)
    return loss


# parameterized Rosenbrock problem
def lossRosenbrock(aggLoss, weight):
    # parameters
    p = variable("p")
    p1, p2 = p[:, 0], p[:, 1]
    y = variable("y")
    y1, y2 = y[:, 0], y[:, 1]
    # objective function
    f = (1 - y1) ** 2 + p1 * (y2 - y1 ** 2) ** 2
    obj = [f.minimize(weight=1.0, name="obj")]
    # constraints
    Q_con = weight
    con_1 = Q_con * (y1 >= y2)
    con_1.name = "c1"
    con_2 = Q_con * (y1 ** 2 + y2 ** 2 >= p2 / 2)
    con_2.name = "c2"
    con_3 = Q_con * (y1 ** 2 + y2 ** 2 <= p2)
    con_3.name = "c3"
    cons = [con_1, con_2, con_3]
    # loss
    loss = aggLoss(obj, cons)
    return loss


import pytest
import gpytorch
from neuromancer.loss import GPPHSLoss
from neuromancer.dynamics.gp_phs import GPPHSModel, PHSMatrices

agg_losses = [PenaltyLoss, BarrierLoss]
problems = [lossQuadratic, lossRosenbrock]


@given(arrays(np.float64, (3,2), elements=st.floats(0.5, 1.2)),
       arrays(np.float64, (3,2), elements=st.floats(0.0, 3.0)),
       arrays(np.float64, (3,2), elements=st.floats(0.0, 1.0)),
       st.sampled_from(agg_losses),
       st.integers(0, 200))
@settings(max_examples=200, deadline=None)
def test_add(p, x, y, aggLoss, weight):
    # data points
    datapoints = {"p": torch.from_numpy(p),
                  "x": torch.from_numpy(x),
                  "y": torch.from_numpy(y),
                  "name": "test"}
    # loss for quadratic
    loss1 = lossQuadratic(aggLoss, weight)
    output1 = loss1(datapoints)
    # loss for Rosenbrock
    loss2 = lossRosenbrock(aggLoss, weight)
    output2 = loss2(datapoints)
    # add
    loss = loss1 + loss2
    output = loss(datapoints)
    # test
    assert torch.isclose(output1["objective_loss"] + output2["objective_loss"], output["objective_loss"])
    assert torch.isclose(output1["penalty_loss"] + output2["penalty_loss"], output["penalty_loss"])
    assert torch.isclose(output1["loss"] + output2["loss"], output["loss"])


@given(st.floats(0.1, 10),
       arrays(np.float64, (3,2), elements=st.floats(0.5, 1.2)),
       arrays(np.float64, (3,2), elements=st.floats(0.0, 3.0)),
       arrays(np.float64, (3,2), elements=st.floats(0.0, 1.0)),
       st.sampled_from(agg_losses),
       st.sampled_from(problems),
       st.integers(1, 200))
@settings(max_examples=200, deadline=None)
def test_mul(multiplier, p, x, y, aggLoss, problems, weight):  # noqa: F811
    # data points
    datapoints = {"p": torch.from_numpy(p),
                  "x": torch.from_numpy(x),
                  "y": torch.from_numpy(y),
                  "name": "test"}
    # loss
    loss = problems(aggLoss, weight)
    output = loss(datapoints)
    # mul
    weighted_loss = multiplier * loss
    weighted_output = weighted_loss(datapoints)
    # test
    assert torch.isclose(multiplier * output["objective_loss"], weighted_output["objective_loss"])
    assert torch.isclose(multiplier * output["penalty_loss"], weighted_output["penalty_loss"])
    assert torch.isclose(multiplier * output["loss"], weighted_output["loss"])


# ── GP-PHS Loss ────────────────────────────────────────────────────────────────

def _make_gpphs_loss(nx=2, nu=1, N=5):
    J_upper = {(0, 1): lambda x: torch.ones(x.shape[0])}
    R_diag  = {0: lambda x: torch.ones(x.shape[0]),
               1: lambda x: torch.ones(x.shape[0])}
    G_full  = {(0, 0): lambda x: torch.ones(x.shape[0])}
    phs = PHSMatrices(nx, nu, J_upper, R_diag, G_full)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPPHSModel(phs, nx, nu)
    return GPPHSLoss(model, likelihood), nx, nu, N


class TestGPPHSLoss:
    def setup_method(self):
        self.loss_fn, nx, nu, N = _make_gpphs_loss()
        self.nx, self.nu, self.N = nx, nu, N
        self.x    = torch.randn(N, nx)
        self.u    = torch.randn(N, nu)
        self.xdot = torch.randn(N, nx)

    def test_output_is_scalar(self):
        loss = self.loss_fn(self.x, self.u, self.xdot)
        assert loss.shape == torch.Size([])

    def test_output_is_differentiable(self):
        loss = self.loss_fn(self.x, self.u, self.xdot)
        assert loss.grad_fn is not None

    def test_accepts_flat_xdot(self):
        loss = self.loss_fn(self.x, self.u, self.xdot.reshape(-1))
        assert loss.shape == torch.Size([])
        assert not loss.isnan()

    def test_without_xdot_var_is_finite(self):
        loss = self.loss_fn(self.x, self.u, self.xdot)
        assert not loss.isnan()
        assert torch.isfinite(loss)

    def test_with_xdot_var_is_finite(self):
        xdot_var = torch.ones(self.N, self.nx) * 0.1
        loss = self.loss_fn(self.x, self.u, self.xdot, xdot_var=xdot_var)
        assert not loss.isnan()
        assert torch.isfinite(loss)

    def test_larger_xdot_var_changes_loss(self):
        small_var = torch.ones(self.N, self.nx) * 0.01
        large_var = torch.ones(self.N, self.nx) * 10.0
        loss_small = self.loss_fn(self.x, self.u, self.xdot, xdot_var=small_var).item()
        loss_large = self.loss_fn(self.x, self.u, self.xdot, xdot_var=large_var).item()
        assert loss_small != loss_large

    def test_jitter_prevents_nan_on_degenerate_inputs(self):
        x_degen = torch.zeros(self.N, self.nx)
        loss = self.loss_fn(x_degen, self.u, self.xdot)
        assert not loss.isnan()
        assert torch.isfinite(loss)

    def test_residual_depends_on_u(self):
        u_zero  = torch.zeros(self.N, self.nu)
        u_large = torch.ones(self.N, self.nu) * 5.0
        loss_zero  = self.loss_fn(self.x, u_zero,  self.xdot).item()
        loss_large = self.loss_fn(self.x, u_large, self.xdot).item()
        assert loss_zero != loss_large

    def test_backward(self):
        loss = self.loss_fn(self.x, self.u, self.xdot)
        loss.backward()

    def test_parameters_receive_gradients(self):
        loss = self.loss_fn(self.x, self.u, self.xdot)
        loss.backward()
        grads = [p.grad for p in self.loss_fn.model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads)

    def test_xdot_var_changes_result(self):
        loss1 = self.loss_fn(self.x, self.u, self.xdot)
        loss2 = self.loss_fn(self.x, self.u, self.xdot, xdot_var=torch.ones_like(self.xdot) * 0.5)
        assert loss1.item() != loss2.item()

    def test_large_variance_stable(self):
        var = torch.ones(self.N, self.nx) * 1e6
        loss = self.loss_fn(self.x, self.u, self.xdot, xdot_var=var)
        assert torch.isfinite(loss)

    def test_tiny_variance_stable(self):
        var = torch.ones(self.N, self.nx) * 1e-12
        loss = self.loss_fn(self.x, self.u, self.xdot, xdot_var=var)
        assert torch.isfinite(loss)
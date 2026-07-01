import torch
import pytest
from neuromancer.dynamics import ode, physics
from neuromancer.dynamics.ode import PHSODE
from neuromancer.dynamics.gp_phs import PHSMatrices
from neuromancer.modules.hamiltonian_approximator import HamiltonianApproximator
from hypothesis import given, settings, strategies as st
import random
from neuromancer.modules.blocks import MLP

ode_param_systems_auto = [v for v in ode.ode_param_systems_auto.values()]
ode_param_systems_nonauto = [v for v in ode.ode_param_systems_nonauto.values()]
ode_hybrid_systems_auto = [v for v in ode.ode_hybrid_systems_auto.values()]

ode_networked_systems = [v for v in ode.ode_networked_systems.values()]
agent_list = [v for v in physics.agents.values()]
coupling_list = [v for v in physics.couplings.values()]
bias = ['additive','compositional']

@given(st.integers(1, 500),
       st.sampled_from(ode_param_systems_auto))
@settings(max_examples=200, deadline=None)
def test_ode_auto_param_shape(batchsize, ode):
    model = ode()
    x = torch.randn([batchsize, model.in_features])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == model.out_features


@given(st.integers(1, 500),
       st.sampled_from(ode_param_systems_nonauto))
@settings(max_examples=200, deadline=None)
def test_ode_nonauto_param_shape(batchsize, ode):
    model = ode()
    nx = model.out_features
    nu = model.in_features - model.out_features
    x = torch.randn([batchsize, nx])
    u = torch.randn([batchsize, nu])
    y = model(x, u)
    assert y.shape[0] == batchsize and y.shape[1] == model.out_features


@given(st.integers(1, 500),
       st.sampled_from(ode_hybrid_systems_auto))
@settings(max_examples=200, deadline=None)
def test_ode_auto_hybrid_shape(batchsize, ode):
    # this test is intented only for hybrid ode's whose black box parts map R^2 to R
    block = MLP(2, 1, bias=True, hsizes=[20, 20])
    model = ode(block)
    x = torch.randn([batchsize, model.in_features])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == model.out_features


@given(st.integers(1, 100),
       st.integers(0, 1000),
       st.integers(0, 100),
       st.integers(1, 500),
       st.sampled_from(bias),
       st.sampled_from(ode_networked_systems))
@settings(max_examples=200, deadline=None)
def test_random_network(nAgents, nCouplings, nu, batchsize, bias, system):

    # Number of agents in total:
    insize = nAgents + nu
   
    # Instantiate the agents:
    agents = [random.choice(agent_list)(state_names=["T"]) for _ in range(insize)]
    map = physics.map_from_agents(agents)

    # Define the graph and interactions:
    adjacency = list(torch.randint(insize, (nCouplings, 2)))
    couplings = [random.choice(coupling_list)(feature_name="T", pins=[pair]) for pair in adjacency]

    ode = system(
        map=map,
        agents=agents,
        couplings=couplings,
        insize=insize,
        outsize=nAgents,
        inductive_bias=bias)

    x = torch.randn([batchsize, insize])
    y = ode(x)

    assert y.shape[0] == batchsize and y.shape[1] == nAgents


# ── PHSODE ────────────────────────────────────────────────────────────────────

def _make_ham(nx):
    x_fit = torch.randn(10, nx)
    H_fit = (x_fit ** 2).sum(-1)
    ham = HamiltonianApproximator(
        method='gp',
        lengthscale=torch.ones(nx),
        signal_var=torch.tensor(1.0),
    )
    ham.fit(x_fit, H_fit)
    return ham


def _make_phsode(nx=2, nu=1, method='rk4'):
    J_upper = {(0, 1): lambda x: torch.ones(x.shape[0])}
    R_diag  = {0: lambda x: torch.ones(x.shape[0]),
               1: lambda x: torch.ones(x.shape[0])}
    G_full  = {(0, 0): lambda x: torch.ones(x.shape[0])}
    phs = PHSMatrices(nx, nu, J_upper, R_diag, G_full)
    ham = _make_ham(nx)
    return PHSODE(phs, ham, nx, nu, method=method), phs, ham


class TestPHSODE:
    def setup_method(self):
        self.nx, self.nu = 2, 1
        self.phsode, self.phs, self.ham = _make_phsode(self.nx, self.nu)
        self.batch = 3

    def test_ode_equations_shape(self):
        x = torch.randn(self.batch, self.nx)
        u = torch.randn(self.batch, self.nu)
        xdot = self.phsode.ode_equations(x, u, self.ham)
        assert xdot.shape == (self.batch, self.nx)

    def test_ode_equations_no_nan(self):
        x = torch.randn(self.batch, self.nx)
        u = torch.randn(self.batch, self.nu)
        xdot = self.phsode.ode_equations(x, u, self.ham)
        assert not xdot.isnan().any()

    def test_resolve_u_constant(self):
        x = torch.randn(self.batch, self.nx)
        u = torch.randn(self.batch, self.nu)
        t_eval = torch.linspace(0.0, 1.0, 10)
        u_out = self.phsode._resolve_u(u, x, t_eval[0], t_eval)
        assert torch.allclose(u_out, u)

    def test_resolve_u_time_varying(self):
        T = 8
        x = torch.randn(self.batch, self.nx)
        u_seq = torch.randn(T, self.batch, self.nu)
        t_eval = torch.linspace(0.0, 1.0, T)
        u_out = self.phsode._resolve_u(u_seq, x, t_eval[3], t_eval)
        assert torch.allclose(u_out, u_seq[3])

    def test_resolve_u_callable(self):
        x = torch.randn(self.batch, self.nx)
        t_eval = torch.linspace(0.0, 1.0, 10)
        call_log = []
        def feedback(x_, t_):
            call_log.append(True)
            return torch.zeros(x_.shape[0], self.nu)
        self.phsode._resolve_u(feedback, x, t_eval[0], t_eval)
        assert call_log

    def test_simulate_single_shape(self):
        x0 = torch.randn(self.batch, self.nx)
        t_eval = torch.linspace(0.0, 0.1, 5)
        u = torch.zeros(self.batch, self.nu)
        traj = self.phsode._simulate_single(x0, t_eval, u, self.ham)
        assert traj.shape == (5, self.batch, self.nx)

    def test_simulate_returns_required_keys(self):
        x0 = torch.randn(self.batch, self.nx)
        u = torch.zeros(self.batch, self.nu)
        result = self.phsode.simulate(x0, (0.0, 0.04), u, dt=0.02)
        for key in ('mean', 'std', 'samples', 't_eval'):
            assert key in result

    def test_simulate_output_shapes(self):
        x0 = torch.randn(self.batch, self.nx)
        u = torch.zeros(self.batch, self.nu)
        t_eval = torch.linspace(0.0, 0.06, 4)
        result = self.phsode.simulate(x0, (0.0, 0.06), u, t_eval=t_eval)
        T = len(t_eval)
        assert result['mean'].shape    == (T, self.batch, self.nx)
        assert result['std'].shape     == (T, self.batch, self.nx)
        assert result['samples'].shape == (1, T, self.batch, self.nx)
        assert result['t_eval'].shape  == (T,)

    def test_ensemble_size_matches_hamiltonians(self):
        hams = [_make_ham(self.nx) for _ in range(3)]
        ode_ens = PHSODE(self.phs, hams, self.nx, self.nu, method='rk4')
        x0 = torch.randn(self.batch, self.nx)
        u = torch.zeros(self.batch, self.nu)
        t_eval = torch.linspace(0.0, 0.04, 3)
        result = ode_ens.simulate(x0, (0.0, 0.04), u, t_eval=t_eval)
        assert result['samples'].shape[0] == 3

    def test_forward_shape(self):
        x = torch.randn(self.batch, self.nx)
        u = torch.randn(self.batch, self.nu)

        out = self.phsode(x, u)

        assert out.shape == (self.batch, self.nx)
    
    def test_forward_matches_ode_equations(self):
        x = torch.randn(self.batch, self.nx)
        u = torch.randn(self.batch, self.nu)

        out1 = self.phsode(x, u)
        out2 = self.phsode.ode_equations(
            x, u, self.ham
        )

        assert torch.allclose(out1, out2)

    def test_forward_rejects_non_batched_input(self):
        x = torch.randn(self.nx)
        u = torch.randn(1, self.nu)

        with pytest.raises(AssertionError):
            self.phsode(x, u)

    def test_resolve_u_invalid_shape(self):
        x = torch.randn(self.batch, self.nx)

        bad_u = torch.randn(2, 3, 4, 5)

        with pytest.raises(ValueError):
            self.phsode._resolve_u(
                bad_u,
                x,
                torch.tensor(0.0),
                torch.linspace(0, 1, 5),
            )

    def test_resolve_u_uses_nearest_time(self):
        T = 5

        x = torch.randn(self.batch, self.nx)

        u_seq = torch.arange(
            T * self.batch * self.nu,
            dtype=torch.float32,
        ).reshape(T, self.batch, self.nu)

        t_eval = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0]
        )

        t = torch.tensor(2.2)

        u_out = self.phsode._resolve_u(
            u_seq,
            x,
            t,
            t_eval,
        )

        assert torch.allclose(
            u_out,
            u_seq[2]
        )
    
    def test_simulate_generates_correct_t_eval(self):
        x0 = torch.randn(
            self.batch,
            self.nx,
        )

        u = torch.zeros(
            self.batch,
            self.nu,
        )

        result = self.phsode.simulate(
            x0,
            (0.0, 0.1),
            u,
            dt=0.05,
        )

        expected = torch.tensor(
            [0.0, 0.05, 0.10]
        )

        assert torch.allclose(
            result["t_eval"],
            expected,
        )

    def test_simulate_preserves_t_eval(self):
        x0 = torch.randn(
            self.batch,
            self.nx,
        )

        u = torch.zeros(
            self.batch,
            self.nu,
        )

        t_eval = torch.tensor(
            [0.0, 0.03, 0.07]
        )

        result = self.phsode.simulate(
            x0,
            (0.0, 0.07),
            u,
            t_eval=t_eval,
        )

        assert torch.allclose(
            result["t_eval"],
            t_eval,
        )

    def test_ensemble_mean_std_match_samples(self):
        hams = [
            _make_ham(self.nx)
            for _ in range(3)
        ]

        ode = PHSODE(
            self.phs,
            hams,
            self.nx,
            self.nu,
            method="rk4",
        )

        x0 = torch.randn(
            self.batch,
            self.nx,
        )

        u = torch.zeros(
            self.batch,
            self.nu,
        )

        result = ode.simulate(
            x0,
            (0.0, 0.04),
            u,
            dt=0.02,
        )

        assert torch.allclose(
            result["mean"],
            result["samples"].mean(0),
        )

        assert torch.allclose(
            result["std"],
            result["samples"].std(0),
        )

    def test_single_hamiltonian_std_is_finite(self):
        x0 = torch.randn(
            self.batch,
            self.nx,
        )

        u = torch.zeros(
            self.batch,
            self.nu,
        )

        result = self.phsode.simulate(
            x0,
            (0.0, 0.05),
            u,
            dt=0.01,
        )

        assert torch.isfinite(
            result["std"]
        ).all()

    def test_feedback_control_changes_trajectory(self):
        x0 = torch.randn(
            self.batch,
            self.nx,
        )

        t_eval = torch.linspace(
            0,
            0.1,
            5,
        )

        traj_zero = self.phsode.simulate(
            x0,
            (0.0, 0.1),
            lambda x, t:
                torch.zeros(
                    x.shape[0],
                    self.nu,
                ),
            t_eval=t_eval,
        )

        traj_feedback = self.phsode.simulate(
            x0,
            (0.0, 0.1),
            lambda x, t:
                torch.ones(
                    x.shape[0],
                    self.nu,
                ),
            t_eval=t_eval,
        )

        assert not torch.allclose(
            traj_zero["mean"],
            traj_feedback["mean"],
        )
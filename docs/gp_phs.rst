Gaussian Process Port-Hamiltonian Systems
=========================================

Gaussian Process Port-Hamiltonian Systems (GP-PHS) combine Gaussian
Processes with the structure of Port-Hamiltonian systems to learn
continuous-time dynamics while preserving physical properties such as
energy balance and passivity. GP-PHS places a Gaussian Process prior on the Hamiltonian,
yielding probabilistic dynamics with uncertainty quantification.

For further details, see the original paper:
`Gaussian Process Port-Hamiltonian Systems: Bayesian Learning with Physics Prior <https://arxiv.org/abs/2305.09017>`_

Tutorials
---------

- `Double-well Oscillator <https://colab.research.google.com/drive/12w4pyQRXxjUhxb6bU5YVedYhlVAQrmWY?usp=sharing>`_
- `Nonlinear Controlled 2-DOF Port-Hamiltonian Oscillator <https://colab.research.google.com/drive/1loRhSxpP5xddiRRP1DSQFKsQIq9SUUKO?usp=sharing>`_
- `DC Motor with Elastic Load <https://colab.research.google.com/drive/11YDGDonR2qb0gganLuj29R1sgU3rQzKR?usp=sharing>`_
- `Driven Duffing Oscillator, Chaotic Regime Stress Test <https://colab.research.google.com/drive/1P3B6fqqZ1TR20O-o3Cz5lUgFay-ffPL6?usp=sharing>`_

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _data-preparation:

Data Preparation
----------------

**Modules:** ``neuromancer.psl.gp_smoother`` |
``neuromancer.dataset``

The data preparation pipeline transforms raw state observations into
the format required by ``GPPHSNode`` for training. It has two stages:

#. **GP Smoothing**: fit a GP to noisy state measurements to obtain
   smoothed states, derivative estimates, and derivative variances.
#. **DataLoader construction**: split the smoothed arrays into
   train/dev/test sets and wrap them in NeuroMANCER-compatible
   DataLoaders.

The two stages are intentionally separate so you can substitute your
own derivative estimates if you already have them, bypassing the GP
smoother.


gp_smooth
~~~~~~~~~

**Module:** ``neuromancer.psl.gp_smoother``

.. code:: python

   gp_smooth(
       t:         np.ndarray,
       x:         np.ndarray,
       n_iter:    int = 100,
       lr:        float = 0.1,
       noise_var: float = 0.01,
   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

Fits an independent scalar RBF GP to each state dimension using the
observed timestamps ``t`` and noisy state measurements ``x``. Returns
smoothed states, derivative means, and derivative variances.

For each state dimension, the GP is:

::

   xₐ(t) ~ GP(0, k_rbf(t, t'))

Parameters
""""""""""

+-----------------------+------------------------+-----------------------+
| Parameter             | Type                   | Description           |
+=======================+========================+=======================+
| ``t``                 | ``np.ndarray (T,)``    | Timestamps            |
+-----------------------+------------------------+-----------------------+
| ``x``                 | ``np.ndarray (T, nx)`` | Noisy state           |
|                       |                        | observations          |
+-----------------------+------------------------+-----------------------+
| ``n_iter``            | ``int``                | Number of Adam        |
|                       |                        | optimization steps    |
|                       |                        | for hyperparameter    |
|                       |                        | fitting. Default      |
|                       |                        | ``100``.              |
+-----------------------+------------------------+-----------------------+
| ``lr``                | ``float``              | Learning rate for     |
|                       |                        | hyperparameter        |
|                       |                        | optimization. Default |
|                       |                        | ``0.1``.              |
+-----------------------+------------------------+-----------------------+
| ``noise_var``         | ``float``              | Initial observation   |
|                       |                        | noise variance.       |
|                       |                        | Default ``0.01``.     |
+-----------------------+------------------------+-----------------------+

Returns
"""""""

+-----------------------+-----------------------+-----------------------+
| Output                | Shape                 | Description           |
+=======================+=======================+=======================+
| ``x_smooth``          | ``(T, nx)``           | GP posterior mean —   |
|                       |                       | smoothed state        |
|                       |                       | estimates             |
+-----------------------+-----------------------+-----------------------+
| ``x_dot_smooth``      | ``(T, nx)``           | Derivative posterior  |
|                       |                       | mean at each          |
|                       |                       | observation time      |
+-----------------------+-----------------------+-----------------------+
| ``x_dot_var``         | ``(T, nx)``           | Derivative posterior  |
|                       |                       | variance — used as    |
|                       |                       | ``Δ`` diagonal in the |
|                       |                       | GP-PHS NLML           |
+-----------------------+-----------------------+-----------------------+

All outputs are NumPy arrays.

Usage
"""""

.. code:: python

   from neuromancer.psl import gp_smoother

   x_smooth, xdot_smooth, xdot_var = gp_smoother.gp_smooth(t, x_noisy)


get_gpphs_dataloaders
~~~~~~~~~~~~~~~~~~~~~

**Module:** ``neuromancer.dataset``

.. code:: python

   get_gpphs_dataloaders(
       x:           np.ndarray,
       x_dot:       np.ndarray,
       u:           np.ndarray = None,
       x_dot_var:   np.ndarray = None,
       split_ratio: List[float] = None,
       batch_size:  int = 32,
       num_workers: int = 0,
   ) -> Tuple[DataLoader, DataLoader, DataLoader]

Builds NeuroMANCER-compatible DataLoaders from pre-computed smoothed
state and derivative arrays. Validates and reshapes inputs,
constructs the data dictionary, then delegates to
``get_static_dataloaders`` for splitting and wrapping.

Parameters
""""""""""

+-----------------------+------------------------+-------------------------+
| Parameter             | Type                   | Description             |
+=======================+========================+=========================+
| ``x``                 | ``np.ndarray (T, nx)`` | Smoothed state          |
|                       |                        | observations            |
+-----------------------+------------------------+-------------------------+
| ``x_dot``             | ``np.ndarray (T, nx)`` | Derivative estimates    |
+-----------------------+------------------------+-------------------------+
| ``u``                 | ``np.ndarray (T, nu)`` | Control inputs.         |
|                       |                        | Optional — pass         |
|                       |                        | ``None`` for            |
|                       |                        | uncontrolled systems.   |
+-----------------------+------------------------+-------------------------+
| ``x_dot_var``         | ``np.ndarray (T, nx)`` | Derivative variances    |
|                       |                        | from ``gp_smooth``.     |
|                       |                        | Optional but            |
|                       |                        | recommended.            |
+-----------------------+------------------------+-------------------------+
| ``split_ratio``       | ``List[float]``        | Train/dev/test split as |
|                       |                        | percentages summing to  |
|                       |                        | 100, e.g.               |
|                       |                        | ``[75.0, 15.0, 10.0]``. |
+-----------------------+------------------------+-------------------------+
| ``batch_size``        | ``int``                | Batch size for all      |
|                       |                        | three loaders. Default  |
|                       |                        | ``32``. For GP-PHS,     |
|                       |                        | setting this to the     |
|                       |                        | full dataset size is    |
|                       |                        | recommended since the   |
|                       |                        | NLML benefits from      |
|                       |                        | seeing all data per     |
|                       |                        | step.                   |
+-----------------------+------------------------+-------------------------+
| ``num_workers``       | ``int``                | Number of DataLoader    |
|                       |                        | worker processes.       |
|                       |                        | Default ``0``.          |
+-----------------------+------------------------+-------------------------+

Returns
"""""""

``(train_loader, dev_loader, test_loader)`` — three NeuroMANCER
``DataLoader`` instances. Each batch contains the keys ``'X'``,
``'Xdot'``, and optionally ``'U'`` and ``'Xdot_var'``, matching the
input signature of ``GPPHSNode.forward``.

Validation
""""""""""

============================================ =========================
Check                                        Behavior
============================================ =========================
``x`` and ``x_dot`` shapes do not match      Raises ``AssertionError``
``u`` row count does not match ``x``         Raises ``AssertionError``
``x_dot_var`` shape does not match ``x_dot`` Raises ``AssertionError``
============================================ =========================

Usage
"""""

.. code:: python

   from neuromancer import dataset

   train_loader, dev_loader, test_loader = dataset.get_gpphs_dataloaders(
       x=x_smooth,
       x_dot=xdot_smooth,
       u=u_i,
       x_dot_var=xdot_var_np,
       split_ratio=[75.0, 15.0, 10.0],
       batch_size=N_POINTS,
   )

**1D inputs are reshaped automatically.** If ``x``, ``x_dot``, ``u``,
or ``x_dot_var`` are passed as ``(T,)`` arrays, they are reshaped to
``(T, 1)`` automatically.

.. _phsmatrices:

PHSMatrices
-----------

**Module:** ``neuromancer.dynamics.gp_phs`` | **Base class:**
``torch.nn.Module``

``PHSMatrices`` encodes the structural matrices of a Port-Hamiltonian
System. **J** (skew-symmetric interconnection), **R** (diagonal
dissipation), and **G** (input coupling).

Constructor
~~~~~~~~~~~

.. code:: python

   PHSMatrices(
       nx: int,
       nu: int,
       J_upper: Dict[Tuple[int, int], Callable],
       R_diag:  Dict[int, Callable],
       G_full:  Dict[Tuple[int, int], Callable],
   )

Parameters
""""""""""

+-----------------------+------------------------------------+-----------------------+
| Parameter             | Type                               | Description           |
+=======================+====================================+=======================+
| ``nx``                | ``int``                            | State dimension.      |
|                       |                                    | Determines the size   |
|                       |                                    | of J and R as         |
|                       |                                    | ``(nx, nx)`` and G as |
|                       |                                    | ``(nx, nu)``.         |
+-----------------------+------------------------------------+-----------------------+
| ``nu``                | ``int``                            | Control input         |
|                       |                                    | dimension. Determines |
|                       |                                    | the number of columns |
|                       |                                    | in G.                 |
+-----------------------+------------------------------------+-----------------------+
| ``J_upper``           | ``Dict[Tuple[int,int], Callable]`` | Upper-triangle        |
|                       |                                    | entries of J. Keys    |
|                       |                                    | are index pairs       |
|                       |                                    | ``(i, j)`` with       |
|                       |                                    | ``i < j``. Each value |
|                       |                                    | is a callable         |
|                       |                                    | ``f(x) -> Tensor`` of |
|                       |                                    | shape ``(batch,)``.   |
+-----------------------+------------------------------------+-----------------------+
| ``R_diag``            | ``Dict[int, Callable]``            | Diagonal entries of   |
|                       |                                    | R. Keys are row       |
|                       |                                    | indices ``i``. Each   |
|                       |                                    | value is a callable   |
|                       |                                    | ``f(x) -> Tensor`` of |
|                       |                                    | shape ``(batch,)``.   |
+-----------------------+------------------------------------+-----------------------+
| ``G_full``            | ``Dict[Tuple[int,int], Callable]`` | Arbitrary entries of  |
|                       |                                    | G. Keys are index     |
|                       |                                    | pairs ``(i, j)``      |
|                       |                                    | where ``i`` indexes a |
|                       |                                    | state row and ``j``   |
|                       |                                    | indexes a control     |
|                       |                                    | column. Each value is |
|                       |                                    | a callable            |
|                       |                                    | ``f(x) -> Tensor`` of |
|                       |                                    | shape ``(batch,)``.   |
+-----------------------+------------------------------------+-----------------------+

Structural Constraints
~~~~~~~~~~~~~~~~~~~~~~

**Skew-symmetry of J.** You only define the strictly upper triangle
(``i < j``). The constructor rejects any key that violates this. The
lower triangle is filled at evaluation time as ``J[j, i] = -J[i, j]``.
The diagonal is always zero.

**Diagonal dissipation matrix R.** Each diagonal entry is supplied as a
callable ``f(x) -> Tensor``. During construction, GP-PHS automatically
detects whether each callable is constant or state dependent using
automatic differentiation.

- **Constant entries** are automatically assigned a differentiable
  positivity transform (implemented using the softplus function), guaranteeing that the corresponding diagonal
  entry remains nonnegative throughout optimization.

- **State-dependent entries** are evaluated exactly as provided. Since
  positive semi-definiteness cannot, in general, be guaranteed for an
  arbitrary state-dependent function, these entries are validated after
  training.

**G is unconstrained.** No structural constraint is imposed.

**Note:** Any unspecified entry is assumed to be zero.

Parameter Auto-Detection
~~~~~~~~~~~~~~~~~~~~~~~~

GP-PHS automatically detects whether each diagonal entry of ``R`` is
constant or state dependent by checking whether the supplied callable
depends on the state variable ``x`` using PyTorch automatic
differentiation.

For constant entries, positivity is enforced automatically through the
associated learnable parameter.

Detection supports:

- Parameters captured through Python closures
- Parameters stored inside ``nn.Module`` objects
- Parameters referenced from notebook scope
- Parameters referenced from module-level globals

For example:

.. code:: python

   b = variable(
       torch.tensor([1.0], requires_grad=True),
       display_name='b'
   )

   R_diag = {
       1: lambda x: b.value * torch.ones(x.shape[0])
   }

The callable above is detected as constant, since it does not depend on
the values of ``x``.

Methods
~~~~~~~

forward(x)
^^^^^^^^^^

.. code:: python

   forward(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]

Evaluates all three matrices at the current state batch. Calls
``get_J``, ``get_R``, and ``get_G`` in sequence.

======== =============== ===========
Argument Shape           Description
======== =============== ===========
``x``    ``(batch, nx)`` State batch
======== =============== ===========

**Returns:** ``(J, R, G)`` where each is a ``Tensor``:

====== =================== ==============
Output Shape               Constraint
====== =================== ==============
``J``  ``(batch, nx, nx)`` Skew-symmetric
``R``  ``(batch, nx, nx)`` Diagonal (PSD guaranteed for constant entries)
``G``  ``(batch, nx, nu)`` Unconstrained
====== =================== ==============

Validation and Warnings
~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------------------+
| Check                             | Behavior                                      |
+===================================+===============================================+
| ``J_upper`` key ``(i,j)`` has     | Raises ``ValueError``                         |
| ``i >= j`` or is out of range     |                                               |
+-----------------------------------+-----------------------------------------------+
| ``R_diag`` key ``i`` is out of    | Raises ``ValueError``                         |
| ``[0, nx)``                       |                                               |
+-----------------------------------+-----------------------------------------------+
| ``G_full`` key ``(i,j)`` is out   | Raises ``ValueError``                         |
| of range                          |                                               |
+-----------------------------------+-----------------------------------------------+
| ``R_diag`` is missing one or more | Issues ``UserWarning`` (R will be             |
| diagonal indices                  | degenerate)                                   |
+-----------------------------------+-----------------------------------------------+
| State-dependent diagonal entry of | Raises ``ValueError`` after training if any   |
| ``R`` becomes negative            | sampled state produces a negative diagonal    |
|                                   | entry.                                        |
+-----------------------------------+-----------------------------------------------+

Usage Example
~~~~~~~~~~~~~

This example models a simple mass-spring-damper system with two
states (``NX=2``, position and momentum) and one control input
(``NU=1``). The damping coefficient ``b`` is a learnable parameter
declared using ``variable``.

.. code:: python

   import torch
   from neuromancer.constraint import variable
   import neuromancer.dynamics.gp_phs as gp_phs

   b = variable(
       torch.tensor([1.0], requires_grad=True),
       display_name='b'
   )

   phs = gp_phs.PHSMatrices(
       nx=NX, nu=NU,
       J_upper={
           (0, 1): lambda x: torch.ones(x.shape[0])
       },
       R_diag={
           0: lambda x: torch.zeros(x.shape[0]),
           1: lambda x: b.value * torch.ones(x.shape[0])
       },
       G_full={
           (0, 0): lambda x: torch.zeros(x.shape[0]),
           (1, 0): lambda x: torch.ones(x.shape[0])
       },
   )

The assembled matrices at any state batch ``x`` will have the
following structure:

.. code:: text

   J = [[ 0,  1],   R = [[0,        0  ],   G = [[0],
        [-1,  0]]        [0, b.value  ]]          [1]]

Parameter Utilities
~~~~~~~~~~~~~~~~~~~

param_table
^^^^^^^^^^^

.. code:: python

   param_table(
       param_map: Dict[str, Tuple[Any, Optional[float]]]
   ) -> pd.DataFrame

Prints a comparison table of learned versus reference parameter
values.

For each parameter:

.. code:: python

   learned = param.value.detach().item()

Relative error is computed as:

.. code:: python

   100 * abs(learned - true_val) / (abs(true_val) + 1e-6)

Rows with ``true_val=None`` are excluded from the mean and median
error statistics.

Usage
"""""

.. code:: python

   df = gp_phs.param_table({
       'b': (b, B_TRUE),
   })

Example output:

.. code:: text

   Parameter  True  Learned  Rel. Error %
   b          0.50    0.503          0.60

   Mean relative error:   0.60%
   Median relative error: 0.60%

.. _phskernel:

PHSKernel
---------

**Module:** ``neuromancer.dynamics.gp_phs`` | **Base class:**
``gpytorch.kernels.Kernel``

``PHSKernel`` is a structured, non-stationary GP kernel that encodes
Port-Hamiltonian dynamics directly into the covariance function. It
enforces the ``(J-R)∇H`` part of the PHS equation at the kernel
level by sandwiching the Hessian of a base RBF kernel between the
``(J-R)`` matrices evaluated at the input points:

::

   k_phs(x, x') = σ²f · (J(x) - R(x)) · H_rbf(x, x') · (J(x') - R(x'))ᵀ

where ``H_rbf`` is the matrix of mixed second-order partial
derivatives of a squared exponential kernel with diagonal lengthscale
matrix Λ. This construction guarantees that samples drawn from the GP
are consistent with Hamiltonian structure.

``PHSKernel`` is not instantiated directly by the user. It is
constructed internally by ``GPPHSNode``, which wires it together with
the likelihood and the GP model.

Constructor
~~~~~~~~~~~

.. code:: python

   PHSKernel(
       phs_matrices: PHSMatrices,
       nx: int,
       **kwargs,
   )

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Parameter             | Type                  | Description           |
+=======================+=======================+=======================+
| ``phs_matrices``      | ``PHSMatrices``       | The structural matrix |
|                       |                       | definition from Step  |
|                       |                       | 2. Provides           |
|                       |                       | ``get_J(x)`` and      |
|                       |                       | ``get_R(x)`` at       |
|                       |                       | forward time. Note    |
|                       |                       | that ``G`` does not   |
|                       |                       | appear in the kernel  |
|                       |                       | — input coupling is   |
|                       |                       | applied externally.   |
+-----------------------+-----------------------+-----------------------+
| ``nx``                | ``int``               | State dimension.      |
|                       |                       | Determines the size   |
|                       |                       | of the output kernel  |
|                       |                       | block and the length  |
|                       |                       | of the learnable      |
|                       |                       | lengthscale vector.   |
+-----------------------+-----------------------+-----------------------+
| ``**kwargs``          |                       | Passed to the         |
|                       |                       | GPyTorch ``Kernel``   |
|                       |                       | base class (e.g.      |
|                       |                       | ``active_dims``).     |
+-----------------------+-----------------------+-----------------------+

Learnable Parameters
~~~~~~~~~~~~~~~~~~~~

These are registered automatically and included in the optimizer
alongside the physical parameters declared in ``PHSMatrices``.

+---------------------+-----------------+-----------------+-----------------+
| Parameter           | Shape           | Constraint      | Description     |
+=====================+=================+=================+=================+
| ``raw_lengthscale`` | ``(nx,)``       | Softplus,       | Diagonal of the |
|                     |                 | strictly        | lengthscale     |
|                     |                 | positive        | matrix Λ. One   |
|                     |                 |                 | lengthscale per |
|                     |                 |                 | state           |
|                     |                 |                 | dimension.      |
+---------------------+-----------------+-----------------+-----------------+
| ``raw_signal_var``  | ``()``          | Softplus,       | Overall signal  |
|                     |                 | strictly        | variance σ²f.   |
|                     |                 | positive        | Scales the      |
|                     |                 |                 | kernel          |
|                     |                 |                 | amplitude.      |
+---------------------+-----------------+-----------------+-----------------+

Methods
~~~~~~~

forward(x1, x2)
^^^^^^^^^^^^^^^

.. code:: python

   forward(x1: Tensor, x2: Tensor, **params) -> DenseLinearOperator

Computes the full PHS kernel matrix between two state batches. The
output is a ``DenseLinearOperator`` of shape ``(N·nx, M·nx)``, where
each ``(nx, nx)`` block corresponds to a pair of input points and
encodes the covariance between all state derivative dimensions at
those points.

Parameters
""""""""""

======== =========== ======================
Argument Shape       Description
======== =========== ======================
``x1``   ``(N, nx)`` First batch of states
``x2``   ``(M, nx)`` Second batch of states
======== =========== ======================

_rbf_and_hessian(x1, x2)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   _rbf_and_hessian(
       x1: Tensor,
       x2: Tensor
   ) -> Tuple[Tensor, Tensor]

**The Hessian is computed analytically.** There is no
finite-difference or autograd approximation. The mixed partial
``∂²k/∂xᵢ∂x'ⱼ`` is derived in closed form from the RBF kernel and
computed fully vectorized over the batch, which keeps training
efficient.

Returns
"""""""

+-----------------------+-----------------------+-----------------------+
| Output                | Shape                 | Description           |
+=======================+=======================+=======================+
| ``k_rbf``             | ``(N, M)``            | Scalar RBF kernel     |
|                       |                       | values                |
+-----------------------+-----------------------+-----------------------+
| ``H``                 | ``(N, M, nx, nx)``    | Mixed second-order    |
|                       |                       | partial derivatives   |
|                       |                       | ``∂²k/∂xᵢ∂x'ⱼ``       |
+-----------------------+-----------------------+-----------------------+

Numerical Stability
~~~~~~~~~~~~~~~~~~~

If ``R`` has zero diagonal entries—as is common when some state
dimensions are conservative—the matrix ``(J-R)`` becomes
rank-deficient and the kernel matrix ``K`` can become singular. To
prevent Cholesky failures during GP training, ``forward`` adds a
scaled jitter term to the diagonal:

.. code:: python

   jitter = K_out.diagonal().mean() * 1e-4
   K_out = K_out + torch.eye(...) * jitter

This keeps ``K`` positive-definite without meaningfully distorting the
kernel signal.


.. _phsmeanfunction:

PHSMeanFunction
---------------

**Module:** ``neuromancer.dynamics.gp_phs`` | **Base class:**
``gpytorch.means.Mean``

``PHSMeanFunction`` defines the prior mean of the GP-PHS model. It
encodes the contribution of control inputs to the state derivatives
through the input coupling matrix ``G``:

::

   m(x, u) = G(x) · u

``PHSMeanFunction`` is constructed and managed internally by
``GPPHSModel``.

Constructor
~~~~~~~~~~~

.. code:: python

   PHSMeanFunction(
       phs_matrices: PHSMatrices,
       nx: int,
       nu: int,
   )

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Parameter             | Type                  | Description           |
+=======================+=======================+=======================+
| ``phs_matrices``      | ``PHSMatrices``       | Provides ``get_G(x)`` |
|                       |                       | at forward time.      |
+-----------------------+-----------------------+-----------------------+
| ``nx``                | ``int``               | State dimension.      |
+-----------------------+-----------------------+-----------------------+
| ``nu``                | ``int``               | Control input         |
|                       |                       | dimension.            |
+-----------------------+-----------------------+-----------------------+

Methods
~~~~~~~

forward(x, u)
^^^^^^^^^^^^^

.. code:: python

   forward(x: Tensor, u: Tensor) -> Tensor

Computes the prior mean vector for a batch of state-input pairs.

Parameters
""""""""""

======== =========== ===================
Argument Shape       Description
======== =========== ===================
``x``    ``(N, nx)`` State batch
``u``    ``(N, nu)`` Control input batch
======== =========== ===================

Returns
"""""""

Returns a ``Tensor`` of shape ``(N·nx,)``—the flattened mean vector,
matching the layout of the ``(N·nx, M·nx)`` kernel matrix produced by
``PHSKernel``.

The computation is a batched matrix-vector product. For each data
point ``n``:

::

   mean[n] = G(xₙ) · uₙ    # (nx,)

The results are stacked to ``(N, nx)`` and then flattened to
``(N·nx,)``.

.. _gpphsmodel:

GPPHSModel
----------

**Module:** ``neuromancer.dynamics.gp_phs`` | **Base class:**
``torch.nn.Module``

``GPPHSModel`` assembles the prior mean and kernel into a single GPyTorch-compatible
model that returns a ``MultivariateNormal`` distribution over state
derivatives:

::

   ẋ ~ GP(G(x)u, k_phs(x, x'))

The mean function encodes the known control contribution ``G(x)·u``,
and the kernel encodes the PHS structure of the unknown Hamiltonian
gradient ``(J-R)∇H``. Together they define a structured prior that
constrains the GP to produce dynamics consistent with PHS theory.

Why Not ``ExactGP``
~~~~~~~~~~~~~~~~~~~

GPyTorch's ``ExactGP`` assumes ``N`` scalar inputs map to ``N``
scalar outputs. The PHS model has ``N`` state inputs mapping to
``N·nx`` vector outputs—one derivative per state dimension per data
point. This shape conflicts with ``ExactGP``'s internal noise
handling. ``GPPHSModel`` therefore inherits from ``nn.Module``
directly, and the NLML is computed externally in ``GPPHSLoss``.

Constructor
~~~~~~~~~~~

.. code:: python

   GPPHSModel(
       phs_matrices: PHSMatrices,
       nx: int,
       nu: int,
   )

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Parameter             | Type                  | Description           |
+=======================+=======================+=======================+
| ``phs_matrices``      | ``PHSMatrices``       | Passed to both        |
|                       |                       | ``PHSMeanFunction``   |
|                       |                       | and ``PHSKernel``.    |
+-----------------------+-----------------------+-----------------------+
| ``nx``                | ``int``               | State dimension.      |
+-----------------------+-----------------------+-----------------------+
| ``nu``                | ``int``               | Control input         |
|                       |                       | dimension.            |
+-----------------------+-----------------------+-----------------------+

Methods
~~~~~~~

forward(x, u)
^^^^^^^^^^^^^

.. code:: python

   forward(x: Tensor, u: Tensor) -> MultivariateNormal

Evaluates the GP prior at the given state-input batch.

Parameters
""""""""""

======== =========== ===================
Argument Shape       Description
======== =========== ===================
``x``    ``(N, nx)`` State batch
``u``    ``(N, nu)`` Control input batch
======== =========== ===================

Returns
"""""""

Returns a ``gpytorch.distributions.MultivariateNormal`` with:

+-----------------------+-----------------------+-----------------------+
| Field                 | Shape                 | Description           |
+=======================+=======================+=======================+
| ``mean``              | ``(N·nx,)``           | Prior mean            |
|                       |                       | ``G(x)·u``, flattened |
+-----------------------+-----------------------+-----------------------+
| ``covariance_matrix`` | ``(N·nx, N·nx)``      | PHS kernel matrix     |
|                       |                       | ``k_phs(x, x')``      |
+-----------------------+-----------------------+-----------------------+

Usage
~~~~~

``GPPHSModel`` is constructed and managed internally by
``GPPHSNode``. In normal usage you do not instantiate it directly.

If you need direct access to the GP model after training—for
example to inspect the learned kernel or call the posterior—it is
available through:

.. code:: python

   gpphs_node.callable.gp_model              # GPPHSModel instance
   gpphs_node.callable.gp_model.covar_module # PHSKernel
   gpphs_node.callable.gp_model.mean_module  # PHSMeanFunction

.. _gpposterior:

GPPosterior
-----------

**Module:** ``neuromancer.dynamics.gp_phs`` | **Base class:**
``torch.nn.Module``

``GPPosterior`` computes the Bayesian posterior over the Hamiltonian
``H(x*)`` at test points, conditioned on the observed state
derivatives from training. It is the bridge between training and
inference: once ``GPPHSNode`` has been trained, ``GPPosterior`` takes
the learned parameters and produces a distribution over ``H`` that
can be sampled, visualized, and used to build a callable ``H*``.

The posterior is derived from the joint distribution over state
derivatives ``ẋ`` and Hamiltonian values ``H(x*)``:

::

   [ẋ      ]       [K_phs           k_ẋH(X, x*) ]
   [H(x*)] ~ N(0,  [k_ẋH(X,x*)ᵀ   k_HH(x*, x*) ])

Conditioning on the observed ``ẋ`` (after subtracting the known
``G·u`` contribution) gives the posterior mean and covariance over
``H``:

::

   μ_H = k_ẋH(X, x*)ᵀ · (K_phs + Δ)⁻¹ · (ẋ - G·u)
   Σ_H = k_HH(x*, x*) - k_ẋH(X, x*)ᵀ · (K_phs + Δ)⁻¹ · k_ẋH(X, x*)

where ``Δ`` is the derivative noise matrix supplied by the GP
smoother.

``GPPosterior`` is obtained by calling ``posterior()`` on a trained
``GPPHSNode``. You do not instantiate it directly.

Constructor
~~~~~~~~~~~

.. code:: python

   GPPosterior(
       model:        GPPHSModel,
       likelihood:   GaussianLikelihood,
       phs_matrices: PHSMatrices,
       lengthscale:  Tensor,
       signal_var:   Tensor,
       noise_var:    Tensor,
   )

Parameters
""""""""""

+-----------------------+------------------------+-----------------------+
| Parameter             | Type                   | Description           |
+=======================+========================+=======================+
| ``model``             | ``GPPHSModel``         | The trained GP model. |
|                       |                        | Set to eval mode on   |
|                       |                        | construction.         |
+-----------------------+------------------------+-----------------------+
| ``likelihood``        | ``GaussianLikelihood`` | The trained           |
|                       |                        | likelihood. Set to    |
|                       |                        | eval mode on          |
|                       |                        | construction.         |
+-----------------------+------------------------+-----------------------+
| ``phs_matrices``      | ``PHSMatrices``        | The trained           |
|                       |                        | structural matrices.  |
|                       |                        | Provides ``get_J``,   |
|                       |                        | ``get_R``, ``get_G``  |
|                       |                        | at inference time.    |
+-----------------------+------------------------+-----------------------+
| ``lengthscale``       | ``Tensor (nx,)``       | Learned lengthscale   |
|                       |                        | vector, detached from |
|                       |                        | the computation       |
|                       |                        | graph.                |
+-----------------------+------------------------+-----------------------+
| ``signal_var``        | ``Tensor``             | Learned signal        |
|                       |                        | variance σ²f,         |
|                       |                        | detached.             |
+-----------------------+------------------------+-----------------------+
| ``noise_var``         | ``Tensor``             | Learned noise         |
|                       |                        | variance, detached.   |
+-----------------------+------------------------+-----------------------+

All three tensor arguments are registered as buffers, not parameters.
No gradients are computed inside ``GPPosterior``.

Methods
~~~~~~~

predict(smoothed, us, xdots, xdot_vars, test_x, n_samples)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   predict(
       smoothed:   List[np.ndarray],
       us:         List[np.ndarray],
       xdots:      List[np.ndarray],
       xdot_vars:  List[np.ndarray],
       test_x:     Tensor = None,
       n_samples:  int = 20,
   ) -> Tuple[Tensor, Tensor, Tensor]

The primary interface for computing the Hamiltonian posterior.
Accepts lists of NumPy arrays—one per trajectory—concatenates
them, converts to tensors, and calls ``forward``. This matches the
natural output format of the GP smoother, where each trajectory is
stored separately.

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Argument              | Type                  | Description           |
+=======================+=======================+=======================+
| ``smoothed``          | ``List[np.ndarray]``  | Smoothed states, one  |
|                       |                       | array of shape        |
|                       |                       | ``(N_i, nx)`` per     |
|                       |                       | trajectory            |
+-----------------------+-----------------------+-----------------------+
| ``us``                | ``List[np.ndarray]``  | Control inputs, one   |
|                       |                       | array of shape        |
|                       |                       | ``(N_i, nu)`` per     |
|                       |                       | trajectory            |
+-----------------------+-----------------------+-----------------------+
| ``xdots``             | ``List[np.ndarray]``  | Smoothed derivatives, |
|                       |                       | one array of shape    |
|                       |                       | ``(N_i, nx)`` per     |
|                       |                       | trajectory            |
+-----------------------+-----------------------+-----------------------+
| ``xdot_vars``         | ``List[np.ndarray]``  | Derivative variances  |
|                       |                       | from the GP smoother, |
|                       |                       | one array of shape    |
|                       |                       | ``(N_i, nx)`` per     |
|                       |                       | trajectory            |
+-----------------------+-----------------------+-----------------------+
| ``test_x``            | ``Tensor``            | States at which to    |
|                       |                       | evaluate ``H``.       |
|                       |                       | Defaults to the       |
|                       |                       | concatenated training |
|                       |                       | states if ``None``.   |
+-----------------------+-----------------------+-----------------------+
| ``n_samples``         | ``int``               | Number of posterior   |
|                       |                       | samples to draw.      |
|                       |                       | Default ``50``.       |
+-----------------------+-----------------------+-----------------------+

Returns
"""""""

Returns ``(H_mean, H_var, H_samples)``:

+-----------------------+-----------------------+-----------------------+
| Output                | Shape                 | Description           |
+=======================+=======================+=======================+
| ``H_mean``            | ``(M,)``              | Posterior mean of     |
|                       |                       | ``H`` at test points. |
+-----------------------+-----------------------+-----------------------+
| ``H_var``             | ``(M,)``              | Posterior variance of |
|                       |                       | ``H`` at test points, |
|                       |                       | clamped to            |
|                       |                       | non-negative.         |
+-----------------------+-----------------------+-----------------------+
| ``H_samples``         | ``(n_samples, M)``    | Samples drawn from    |
|                       |                       | the posterior over    |
|                       |                       | ``H``.                |
+-----------------------+-----------------------+-----------------------+

forward(train_x, train_u, train_xdot, test_x, n_samples, xdot_var)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   forward(
       train_x:    Tensor,
       train_u:    Tensor,
       train_xdot: Tensor,
       test_x:     Tensor,
       n_samples:  int = 20,
       xdot_var:   Tensor = None,
   ) -> Tuple[Tensor, Tensor, Tensor]

Core posterior computation. Accepts tensors directly. In most cases
you should use ``predict`` instead, which handles list-of-trajectory
inputs and NumPy conversion.

Returns
"""""""

The computation proceeds as follows:

#. Build ``k_ẋH(X, x*)`` — cross-kernel between training derivatives
   and test Hamiltonian values.
#. Build ``k_HH(x*, x*)`` — scalar RBF kernel at test points.
#. Build ``K_phs + Δ`` — training kernel matrix plus derivative
   noise.
#. Subtract the prior mean ``G(x)·u`` from ``train_xdot`` to isolate
   the Hamiltonian residual.
#. Solve for

   ::

      α = (K_phs + Δ)⁻¹ · (ẋ - G·u)

   via Cholesky decomposition.

#. Compute posterior mean ``μ_H`` and covariance ``Σ_H``.
#. Draw samples using eigendecomposition of ``Σ_H`` to handle
   near-indefinite covariances numerically.

The return signature is the same as ``predict``:
``(H_mean, H_var, H_samples)``.

Usage
~~~~~

.. code:: python

   # After training
   posterior = gpphs_node.callable.posterior()

   # Compute Hamiltonian posterior
   H_mean, H_var, H_samples = posterior.predict(
       smoothed=[x_smooth],
       us=[u_i],
       xdots=[xdot_smooth],
       xdot_vars=[xdot_var_np],
       n_samples=50,
   )

Each argument is a list with one array per trajectory. For a single
trajectory, wrap the array in a list as shown above.

For multiple trajectories, pass one array per trajectory and they will
be concatenated automatically:

.. code:: python

   H_mean, H_var, H_samples = posterior.predict(
       smoothed=[x_smooth_1, x_smooth_2],
       us=[u_1, u_2],
       xdots=[xdot_1, xdot_2],
       xdot_vars=[xdot_var_1, xdot_var_2],
       n_samples=50,
   )

Notes
~~~~~

**Hamiltonians are identifiable only up to an additive constant.**
The GP models ``∇H``, not ``H`` directly. Two Hamiltonians that differ
by a constant produce identical dynamics. The posterior mean is
therefore aligned to the training data, but its absolute value has no
physical meaning on its own. Comparisons against a true Hamiltonian
should account for this offset.

**No gradients are computed.** The entire ``forward`` method runs
inside ``torch.no_grad()``. ``GPPosterior`` is an inference-only
object.

.. _gpphsnode:

GPPHSNode
---------

**Module:** ``neuromancer.dynamics.gp_phs`` | **Base class:**
``torch.nn.Module``

``GPPHSNode`` is the NeuroMANCER-compatible wrapper that connects the
GP-PHS model to the ``Problem`` / ``Trainer`` training loop. It owns
the likelihood, the GP model, and the PHS matrices, and exposes a
single ``forward`` method that computes the Negative Log Marginal
Likelihood (NLML) used as the training objective.

All learnable components—kernel hyperparameters, likelihood noise,
and physical parameters from ``PHSMatrices``—are registered as
``nn.Module`` children and are therefore captured automatically by
``problem.parameters()`` and the optimizer. You do not need to
collect or pass them manually.

Constructor
~~~~~~~~~~~

.. code:: python

   GPPHSNode(
       phs_matrices: PHSMatrices,
       nx: int,
       nu: int,
   )

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Parameter             | Type                  | Description           |
+=======================+=======================+=======================+
| ``phs_matrices``      | ``PHSMatrices``       | The structural matrix |
|                       |                       | definition from Step  |
|                       |                       | 2. Owned by this      |
|                       |                       | module for its        |
|                       |                       | lifetime.             |
+-----------------------+-----------------------+-----------------------+
| ``nx``                | ``int``               | State dimension.      |
+-----------------------+-----------------------+-----------------------+
| ``nu``                | ``int``               | Control input         |
|                       |                       | dimension.            |
+-----------------------+-----------------------+-----------------------+

Owned Parameters
~~~~~~~~~~~~~~~~

All of the following are registered as child modules and included in
``problem.parameters()`` automatically.

+---------------------+-----------------+------------------------+--------------------+
| Parameter           | Shape           | Registered in          | Description        |
+=====================+=================+========================+====================+
| Physical params     | scalar per      | ``PHSMatrices``        | Declared via       |
| (e.g. ``b``)        | parameter       |                        | ``positive_param`` |
|                     |                 |                        | or                 |
|                     |                 |                        | ``nonneg_param``   |
|                     |                 |                        | in Step 2.         |
+---------------------+-----------------+------------------------+--------------------+
| ``raw_lengthscale`` | ``(nx,)``       | ``PHSKernel``          | One lengthscale    |
|                     |                 |                        | per state          |
|                     |                 |                        | dimension.         |
+---------------------+-----------------+------------------------+--------------------+
| ``raw_signal_var``  | ``(1,)``        | ``PHSKernel``          | Overall kernel     |
|                     |                 |                        | signal variance    |
|                     |                 |                        | σ²f.               |
+---------------------+-----------------+------------------------+--------------------+
| ``raw_noise``       | ``(1,)``        | ``GaussianLikelihood`` | Observation noise  |
|                     |                 |                        | variance,          |
|                     |                 |                        | initialized to     |
|                     |                 |                        | ``1e-3`` with a    |
|                     |                 |                        | lower bound of     |
|                     |                 |                        | ``1e-4``.          |
+---------------------+-----------------+------------------------+--------------------+

For a model with ``P`` physical parameters and state dimension
``nx``, the total number of parameter tensors seen by the optimizer
is ``P + nx + 2``.

Methods
~~~~~~~

forward(X, U, Xdot, Xdot_var)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   forward(
       X:        Tensor,
       U:        Tensor,
       Xdot:     Tensor,
       Xdot_var: Tensor = None,
   ) -> Tensor

Computes and returns the scalar NLML for the current batch using
``GPPHSLoss``. This is stored under the key ``'nlml'`` in the
NeuroMANCER problem dictionary and is what the optimizer minimizes.

On the first forward pass, ``raw_signal_var`` is initialized from the
empirical variance of ``Xdot``. This prevents the optimizer from
starting in a near-zero basin where the GP ignores dynamics structure
entirely. This initialization happens once and does not repeat on
subsequent forward calls.

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Argument              | Shape                 | Description           |
+=======================+=======================+=======================+
| ``X``                 | ``(N, nx)``           | State batch.          |
+-----------------------+-----------------------+-----------------------+
| ``U``                 | ``(N, nu)``           | Control input batch.  |
+-----------------------+-----------------------+-----------------------+
| ``Xdot``              | ``(N, nx)``           | Observed state        |
|                       |                       | derivatives,          |
|                       |                       | typically the         |
|                       |                       | smoothed output from  |
|                       |                       | the GP smoother.      |
+-----------------------+-----------------------+-----------------------+
| ``Xdot_var``          | ``(N, nx)``           | Per-observation       |
|                       |                       | derivative variances  |
|                       |                       | from the GP smoother. |
|                       |                       | Optional but          |
|                       |                       | recommended.          |
+-----------------------+-----------------------+-----------------------+

Returns
"""""""

Returns a scalar ``Tensor`` containing the NLML value for the current
batch.

posterior()
^^^^^^^^^^^

.. code:: python

   posterior() -> GPPosterior

Constructs a ``GPPosterior`` object using the currently learned model
parameters. Call this after training to move into the Hamiltonian
posterior computation (Step 5). The returned object captures a
snapshot of the learned ``lengthscale``, ``signal_var``, and
``noise_var`` at the time of the call.

Returns
"""""""

Returns a ``GPPosterior`` instance.

Usage
~~~~~

.. code:: python

   import neuromancer.dynamics.gp_phs as gp_phs
   from neuromancer.system import Node
   from neuromancer.constraint import variable
   from neuromancer.loss import PenaltyLoss
   from neuromancer.problem import Problem

   # build node, loss, and problem
   gpphs_module = gp_phs.GPPHSNode(phs, NX, NU)
   gpphs_node = Node(
       gpphs_module,
       ['X', 'U', 'Xdot', 'Xdot_var'],
       ['nlml'],
       name='gp_phs',
   )

   nlml = variable('nlml')
   nlml_loss = nlml.minimize()
   nlml_loss.name = 'nlml_loss'

   loss = PenaltyLoss([nlml_loss], [])
   problem = Problem([gpphs_node], loss)

   print(f'{len(list(problem.parameters()))} total parameter tensors')

After training, move to the Hamiltonian posterior:

.. code:: python

   posterior = gpphs_node.callable.posterior()

Note the ``.callable`` accessor—NeuroMANCER's ``Node`` wraps the
module, so the ``GPPHSNode`` instance is accessed through
``gpphs_node.callable``, not ``gpphs_node`` directly.

Notes
~~~~~

**Do not call ``posterior()`` before training.** The method will
succeed technically since it only reads the current parameter values,
but those values will be at their initialization point and the
resulting posterior will not reflect learned dynamics.

.. _gpphsloss:

GPPHSLoss
---------

**Module:** ``neuromancer.loss`` | **Base class:**
``torch.nn.Module``

``GPPHSLoss`` computes the Negative Log Marginal Likelihood (NLML)
for the GP-PHS model via direct Cholesky decomposition:

::

   NLML = 0.5 · rᵀ(K + Δ)⁻¹r + log|L| + 0.5 · N·nx · log(2π)

where ``r = ẋ - G(x)·u`` is the residual after subtracting the prior
mean, ``K`` is the PHS kernel matrix, ``Δ`` is the derivative noise
matrix from the GP smoother, and ``L`` is the Cholesky factor of
``K + Δ``.

GPyTorch's ``ExactMarginalLogLikelihood`` cannot be used here because
``ExactGP`` assumes ``N`` inputs mapping to ``N`` scalar outputs. The
PHS model has ``N`` inputs mapping to ``N·nx`` outputs, so the noise
shape conflicts. ``GPPHSLoss`` handles this directly.

``GPPHSLoss`` is constructed and managed internally by
``GPPHSNode``. You do not instantiate it directly.

Constructor
~~~~~~~~~~~

.. code:: python

   GPPHSLoss(
       model: GPPHSModel,
       likelihood: GaussianLikelihood,
   )

Methods
~~~~~~~

forward(x, u, xdot, xdot_var)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   forward(
       x:        Tensor,
       u:        Tensor,
       xdot:     Tensor,
       xdot_var: Tensor = None,
   ) -> Tensor

Parameters
""""""""""

+-----------------------+-----------------------+---------------------------+
| Argument              | Shape                 | Description               |
+=======================+=======================+===========================+
| ``x``                 | ``(N, nx)``           | State batch.              |
+-----------------------+-----------------------+---------------------------+
| ``u``                 | ``(N, nu)``           | Control input batch.      |
+-----------------------+-----------------------+---------------------------+
| ``xdot``              | ``(N, nx)``           | Observed state            |
|                       |                       | derivatives.              |
+-----------------------+-----------------------+---------------------------+
| ``xdot_var``          | ``(N, nx)``           | Per-point derivative      |
|                       |                       | variances from            |
|                       |                       | ``gp_smooth``. If         |
|                       |                       | ``None``, falls back to   |
|                       |                       | ``likelihood.noise · I``. |
+-----------------------+-----------------------+---------------------------+

Returns
"""""""

Returns a scalar NLML. Minimizing this maximizes the marginal
likelihood over the kernel hyperparameters and physical parameters
simultaneously.

Notes
~~~~~

**A small scaled jitter is added to the diagonal** before
the Cholesky decomposition for floating-point stability. The PHS
kernel is PSD by construction, so this does not affect the solution
meaningfully.

.. _hamiltonianapproximator:

HamiltonianApproximator
-----------------------

**Module:** ``neuromancer.modules.hamiltonian_approximator`` |
**Base class:** ``torch.nn.Module``

``HamiltonianApproximator`` fits a callable ``H*(x)`` to a set of
sampled Hamiltonian values from ``GPPosterior`` and exposes both
evaluation and gradient computation at arbitrary query points. Once
fitted, it can be used to compute the full PHS dynamics:

::

   ẋ = (J(x) - R(x)) · ∇H*(x) + G(x) · u

In practice you build one approximator per posterior sample, forming
an ensemble that carries uncertainty from the Hamiltonian posterior
through to derivative and trajectory predictions.

Two methods are available:

+-----------------------+-----------------------+-----------------------+
| Method                | Description           | Gradient              |
+=======================+=======================+=======================+
| ``'gp'``              | GP with SE kernel.    | Analytic              |
|                       | Uses the learned      |                       |
|                       | lengthscales and      |                       |
|                       | signal variance from  |                       |
|                       | the trained GP-PHS    |                       |
|                       | model.                |                       |
+-----------------------+-----------------------+-----------------------+
| ``'spline'``          | Thin-plate spline via | Autograd              |
|                       | ``scipy``. No         |                       |
|                       | hyperparameters       |                       |
|                       | required.             |                       |
+-----------------------+-----------------------+-----------------------+

``'gp'`` is recommended when lengthscale and signal variance are
available from the posterior, since the kernel structure is
consistent with the training model.

Constructor
~~~~~~~~~~~

.. code:: python

   HamiltonianApproximator(
       method: str = 'spline',
       lengthscale: Tensor = None,
       signal_var: Tensor = None,
       noise: float = 0.0,
   )

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Parameter             | Type                  | Description           |
+=======================+=======================+=======================+
| ``method``            | ``str``               | ``'gp'`` or           |
|                       |                       | ``'spline'``. Default |
|                       |                       | ``'spline'``.         |
+-----------------------+-----------------------+-----------------------+
| ``lengthscale``       | ``Tensor (nx,)``      | Learned lengthscale   |
|                       |                       | vector from the       |
|                       |                       | posterior. Required   |
|                       |                       | for ``method='gp'``.  |
+-----------------------+-----------------------+-----------------------+
| ``signal_var``        | ``Tensor``            | Learned signal        |
|                       |                       | variance from the     |
|                       |                       | posterior. Required   |
|                       |                       | for ``method='gp'``.  |
+-----------------------+-----------------------+-----------------------+
| ``noise``             | ``float``             | Nugget added to the   |
|                       |                       | GP diagonal for       |
|                       |                       | numerical stability.  |
|                       |                       | Default ``0.0``.      |
+-----------------------+-----------------------+-----------------------+

Methods
~~~~~~~

fit(x, H_vals)
^^^^^^^^^^^^^^

.. code:: python

   fit(x: Tensor, H_vals: Tensor) -> HamiltonianApproximator

Fits the approximator to a set of state-Hamiltonian pairs. Returns
``self`` so it can be chained directly with the constructor.

Parameters
""""""""""

+-----------------------+-----------------------+-----------------------+
| Argument              | Shape                 | Description           |
+=======================+=======================+=======================+
| ``x``                 | ``(M, nx)``           | State points.         |
+-----------------------+-----------------------+-----------------------+
| ``H_vals``            | ``(M,)``              | Sampled Hamiltonian   |
|                       |                       | values at those       |
|                       |                       | states, typically one |
|                       |                       | row from              |
|                       |                       | ``H_samples``.        |
+-----------------------+-----------------------+-----------------------+

Returns
"""""""

Returns ``self`` so the method can be chained directly with the
constructor.

Usage
~~~~~

After computing the Hamiltonian posterior in Step 5, build one
approximator per sample using the learned lengthscales and signal
variance from the posterior:

.. code:: python

   x_smooth_t = torch.tensor(x_smooth, dtype=torch.float32)

   ensemble = [
       hamiltonian_approximator.HamiltonianApproximator(
           method='gp',
           lengthscale=posterior.lengthscale,
           signal_var=posterior.signal_var,
       ).fit(x_smooth_t, H_samples[i])
       for i in range(H_samples.shape[0])
   ]

Each element of ``ensemble`` is a fitted
``HamiltonianApproximator`` corresponding to one posterior sample of
``H``. The ensemble as a whole represents uncertainty in the
Hamiltonian.

To compute the mean and uncertainty of ``∇H*`` across the ensemble:

.. code:: python

   grad_H_stack = torch.stack(
       [h.gradient(x_smooth_t) for h in ensemble]
   )

   grad_H_mean = grad_H_stack.mean(0)   # (T, nx)
   grad_H_std = grad_H_stack.std(0)     # (T, nx)

.. _phsode:

PHSODE
------

**Module:** ``neuromancer.dynamics.ode`` | **Base class:**
``ODESystem``

``PHSODE`` simulates Port-Hamiltonian trajectories using the learned
PHS matrices and a fitted ``HamiltonianApproximator`` ensemble. It
integrates

::

   ẋ = (J(x) - R(x)) · ∇H*(x) + G(x) · u

When an ensemble of approximators is passed, one trajectory is
simulated per sample, giving a distribution over trajectories that
reflects uncertainty in the learned Hamiltonian.

Constructor
~~~~~~~~~~~

.. code:: python

   PHSODE(
       phs_matrices: PHSMatrices,
       hamiltonian: HamiltonianApproximator | List[HamiltonianApproximator],
       nx: int,
       nu: int,
       method: str = 'dopri5',
   )

Parameters
""""""""""

+-----------------------+-----------------------------+-----------------------+
| Parameter             | Type                        | Description           |
+=======================+=============================+=======================+
| ``phs_matrices``      | ``PHSMatrices``             | Trained structural    |
|                       |                             | matrices. Provides    |
|                       |                             | ``J(x)``, ``R(x)``,   |
|                       |                             | and ``G(x)`` during   |
|                       |                             | integration.          |
+-----------------------+-----------------------------+-----------------------+
| ``hamiltonian``       | ``HamiltonianApproximator`` | Single approximator   |
|                       | or ``List``                 | or ensemble. If a     |
|                       |                             | list is supplied, one |
|                       |                             | trajectory is         |
|                       |                             | simulated per         |
|                       |                             | approximator.         |
+-----------------------+-----------------------------+-----------------------+
| ``nx``                | ``int``                     | State dimension.      |
+-----------------------+-----------------------------+-----------------------+
| ``nu``                | ``int``                     | Control input         |
|                       |                             | dimension.            |
+-----------------------+-----------------------------+-----------------------+
| ``method``            | ``str``                     | ODE solver.           |
|                       |                             | ``'dopri5'``          |
|                       |                             | (adaptive RK45,       |
|                       |                             | default), ``'rk4'``   |
|                       |                             | (fixed-step RK4), or  |
|                       |                             | ``'euler'``.          |
+-----------------------+-----------------------------+-----------------------+

Methods
~~~~~~~

simulate(x0, t_span, u, dt, t_eval)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   simulate(
       x0: Tensor,
       t_span: Tuple[float, float],
       u: Tensor | Callable,
       dt: float = 0.01,
       t_eval: Tensor = None,
   ) -> Dict

Simulates PHS trajectories over a time horizon. If an ensemble was
passed to the constructor, one trajectory is generated per
Hamiltonian approximator and the full trajectory distribution is
returned.

Parameters
""""""""""

+-----------------------+------------------------+-----------------------+
| Parameter             | Type                   | Description           |
+=======================+========================+=======================+
| ``x0``                | ``Tensor (batch, nx)`` | Initial state. The    |
|                       |                        | batch dimension       |
|                       |                        | allows multiple       |
|                       |                        | initial conditions.   |
+-----------------------+------------------------+-----------------------+
| ``t_span``            | ``(t0, tf)``           | Start and end time.   |
+-----------------------+------------------------+-----------------------+
| ``u``                 | see below              | Control input.        |
+-----------------------+------------------------+-----------------------+
| ``dt``                | ``float``              | Time step used to     |
|                       |                        | construct             |
|                       |                        | ``t_eval`` if not     |
|                       |                        | provided.             |
+-----------------------+------------------------+-----------------------+
| ``t_eval``            | ``Tensor (T,)``        | Specific evaluation   |
|                       |                        | times. If ``None``,   |
|                       |                        | generated from        |
|                       |                        | ``t_span`` and        |
|                       |                        | ``dt``.               |
+-----------------------+------------------------+-----------------------+

Control Input Modes
"""""""""""""""""""

``u`` supports three forms:

+-----------------------+-----------------------------------+-----------------------+
| Form                  | Shape                             | Description           |
+=======================+===================================+=======================+
| Constant              | ``(batch, nu)``                   | Same control input at |
|                       |                                   | every integration     |
|                       |                                   | step.                 |
+-----------------------+-----------------------------------+-----------------------+
| Time-varying          | ``(T, batch, nu)``                | One control value per |
|                       |                                   | time step, aligned    |
|                       |                                   | with ``t_eval``.      |
+-----------------------+-----------------------------------+-----------------------+
| Feedback              | ``Callable(x, t) -> (batch, nu)`` | Evaluated at every    |
|                       |                                   | integration step      |
|                       |                                   | using the current     |
|                       |                                   | state and time.       |
+-----------------------+-----------------------------------+-----------------------+

Returns
"""""""

Returns a dictionary with the following keys:

+-----------------------+-----------------------+-----------------------+
| Key                   | Shape                 | Description           |
+=======================+=======================+=======================+
| ``'mean'``            | ``(T, batch, nx)``    | Mean trajectory over  |
|                       |                       | the ensemble.         |
+-----------------------+-----------------------+-----------------------+
| ``'std'``             | ``(T, batch, nx)``    | Standard deviation    |
|                       |                       | over the ensemble.    |
|                       |                       | Zero if a single      |
|                       |                       | approximator is used. |
+-----------------------+-----------------------+-----------------------+
| ``'samples'``         | ``(S, T, batch, nx)`` | Individual simulated  |
|                       |                       | trajectories, where   |
|                       |                       | ``S`` is the ensemble |
|                       |                       | size.                 |
+-----------------------+-----------------------+-----------------------+
| ``'t_eval'``          | ``(T,)``              | Evaluation time       |
|                       |                       | points.               |
+-----------------------+-----------------------+-----------------------+

Usage
~~~~~

.. code:: python

   from neuromancer.dynamics import ode
   import torch
   import numpy as np

   dt = 0.01
   T_SIM = 25.0

   # Time-varying control input: (T, batch=1, nu=1)
   u_sim = torch.zeros(
       len(np.arange(0.0, T_SIM + dt, dt)),
       1,
       NU,
       dtype=torch.float32,
   )

   # Build PHSODE with the posterior ensemble
   phsode = ode.PHSODE(
       phs,
       ensemble,
       NX,
       NU,
       method='rk4',
   )

   with torch.no_grad():
       result = phsode.simulate(
           x0=torch.tensor([[5.0, 0.0]], dtype=torch.float32),
           t_span=(0.0, T_SIM),
           u=u_sim,
           dt=dt,
       )

   # Extract results — squeeze batch dimension
   t_gp = result['t_eval'].numpy()
   mean = result['mean'][:, 0, :].numpy()
   std = result['std'][:, 0, :].numpy()

The ``[:, 0, :]`` indexing removes the batch dimension when
simulating a single initial condition.

Notes
~~~~~

**The batch dimension allows multiple initial conditions.** Pass
``x0`` with shape ``(B, nx)`` to simulate ``B`` trajectories
simultaneously. The time-varying control input should then have shape
``(T, B, nu)``.

**All integration runs inside ``torch.no_grad()``.** Gradients are
not needed for trajectory simulation, and disabling them reduces
memory usage significantly for large ensembles.

.. _visualization:

Visualization
~~~~~~~~~~~~~

The ``neuromancer.psl.plot`` module provides convenience functions for
visualizing the GP-PHS workflow and simulation results.

**GP smoothing**

Plots noisy and smoothed state trajectories together with the estimated
state derivatives and their uncertainty.

.. code:: python

   plot.pltGPSmooth(
       t_all,
       trajs,
       smoothed,
       xdots,
       xdot_vars,
       x_trues=trajs_true,
   )

**Hamiltonian posterior**

Plots the learned Hamiltonian posterior mean with ±2σ uncertainty bands
and, optionally, the true Hamiltonian.

.. code:: python

   plot.pltHamiltonian(
       H_mean.numpy(),
       H_var.numpy(),
       H_true=H_true_all,
   )

**Derivative prediction**

Plots learned state derivatives against reference derivatives with
optional uncertainty bands.

.. code:: python

   plot.pltXdot(
       xdot_pred,
       xdot_true,
       xdot_std=xdot_pred_std,
   )

**Trajectory rollout**

Plots simulated GP-PHS trajectories with optional uncertainty bands and
reference trajectories.

.. code:: python

   plot.pltTrajectory(
       t_gp,
       mean,
       std=std,
       x_true=traj_true,
   )
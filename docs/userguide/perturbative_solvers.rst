How-to use Dyson and Magnus based solvers
=========================================

.. warning::

    This is an advanced topic --- utilizing perturbation-theory based solvers requires detailed
    knowledge of the structure of the differential equations involved, as well as manual tuning of
    the solver parameters. See the :class:`.DysonSolver` and :class:`.MagnusSolver` documentation
    for API details. Also, see :footcite:`puzzuoli_algorithms_2023` for a detailed explanation of
    the solvers, which varies and builds on the core idea introduced in
    :footcite:`shillito_fast_2020`.

.. note::

    The circumstances under which perturbative solvers outperform traditional solvers, and which
    parameter sets to use, is nuanced. Perturbative solvers executed with JAX are setup to use more
    parallelization within a single solver run than typical solvers, and thus it is
    circumstance-specific whether the trade-off between speed of a single run and resource
    consumption is advantageous. Due to the parallelized nature, the comparison of execution times
    demonstrated in this userguide are highly hardware-dependent.


In this tutorial we walk through how to use perturbation-theory based solvers. For information on
how these solvers work, see the :class:`.DysonSolver` and :class:`.MagnusSolver` class
documentation, as well as the perturbative expansion background information provided in
:ref:`Time-dependent perturbation theory and multi-variable series expansions review <perturbation
review>`.

We use a simple transmon model:

.. math:: H(t) = 2 \pi \nu N + \pi \alpha N(N-I) + s(t) \times 2 \pi r (a + a^\dagger)

where:

-  :math:`N`, :math:`a`, and :math:`a^\dagger` are, respectively, the number, annihilation, and
   creation operators.
-  :math:`\nu` is the qubit frequency and :math:`r` is the drive strength.
-  :math:`s(t)` is the drive signal, which we will take to be on resonance with envelope :math:`f(t)
   = A \frac{4t (T - t)}{T^2}` for a given amplitude :math:`A` and total time :math:`T`.

We will walk through the following steps:

1. Configure JAX.
2. Construct the model.
3. How-to construct and simulate using the Dyson-based perturbative solver.
4. Simulate using a traditional ODE solver, comparing speed.
5. How-to construct and simulate using the Magnus-based perturbative solver.

1. Configure JAX
----------------

First, configure JAX to run on CPU in 64 bit mode. See the :ref:`userguide on using JAX <how-to use
different array libraries>` for a more detailed explanation of how to work with JAX in Qiskit
Dynamics.

.. plot::
    :context: close-figs
    :include-source:
    
    # configure jax to use 64 bit mode
    import jax
    jax.config.update("jax_enable_x64", True)

    # tell JAX we are using CPU if using a system without a GPU
    jax.config.update('jax_platform_name', 'cpu')


2. Construct the model
----------------------

First, we construct the model described in the introduction. We use a relatively high dimension for
the oscillator system state space to accentuate the speed difference between the perturbative
solvers and the traditional ODE solver. The higher dimensionality introduces higher frequencies into
the model, which will slow down both the ODE solver and the initial construction of the perturbative
solver. However after the initial construction, the higher frequencies in the model have no impact
on the perturbative solver speed.

.. plot::
    :context: close-figs
    :include-source:

    import numpy as np

    dim = 10  # Oscillator dimension

    v = 5.  # Transmon frequency in GHz
    anharm = -0.33  # Transmon anharmonicity in GHz
    r = 0.02  # Transmon drive coupling in GHz

    # Construct cavity operators
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))

    # Static part of Hamiltonian
    static_hamiltonian = 2 * np.pi * v * N + np.pi * anharm * N * (N - np.eye(dim))
    # Drive term of Hamiltonian
    drive_hamiltonian = 2 * np.pi * r * (a + adag)

    # total simulation time
    T = 1. / r

    # Drive envelope function
    envelope_func = lambda t: t * (T - t) / (T**2 / 4)

3. How-to construct and simulate using the Dyson-based perturbative solver
--------------------------------------------------------------------------

Setting up a :class:`.DysonSolver` requires more setup than the standard :class:`.Solver`, as the
user must specify several configuration parameters, along with the structure of the differential
equation:

- The :class:`.DysonSolver` requires direct specification of the LMDE to the solver. If we are
  simulating the Schrodinger equation, we need to multiply the Hamiltonian terms by ``-1j`` when
  describing the LMDE operators.
- The :class:`.DysonSolver` is a fixed step solver, with the step size being fixed at instantiation.
  This step size must be chosen in conjunction with the ``expansion_order`` to ensure that a
  suitable accuracy is attained.
- Over each fixed time-step the :class:`.DysonSolver` solves by computing a truncated perturbative
  expansion.

  - To compute the truncated perturbative expansion, the signal envelopes are approximated as a
    linear combination of Chebyshev polynomials.
  - The order of the Chebyshev approximations, along with central carrier frequencies for defining
    the “envelope” of each :class:`.Signal`, must be provided at instantiation.

See the :class:`.DysonSolver` API docs for more details.

For our example Hamiltonian we configure the :class:`.DysonSolver` as follows:

.. plot::
    :context: close-figs
    :include-source:

    import time
    from qiskit_dynamics import DysonSolver

    start_time = time.time()

    dt = 0.1
    dyson_solver = DysonSolver(
        operators=[-1j * drive_hamiltonian],
        rotating_frame=-1j * static_hamiltonian,
        dt=dt,
        carrier_freqs=[v],
        chebyshev_orders=[1],
        expansion_order=7,
        integration_method='jax_odeint',
        atol=1e-12,
        rtol=1e-12
    )

    print(f"Run time: {time.time() - start_time}")

The above parameters are chosen so that the :class:`.DysonSolver` is fast and produces high accuracy
solutions (measured and confirmed after the fact). The relatively large step size ``dt = 0.1`` is
chosen for speed: the larger the step size, the fewer steps required. To ensure high accuracy given
the large step size, we choose a high expansion order, and utilize a linear envelope approximation
scheme by setting the ``chebyshev_order`` to ``1`` for the single drive signal.

Similar to the :class:`.Solver` interface, the :meth:`.DysonSolver.solve` method can be called to
simulate the system for a given list of signals, initial state, start time, and number of time steps
of length ``dt``.

To properly compare the speed of :class:`.DysonSolver` to a traditional ODE solver, we write
JAX-compilable functions wrapping each that, given an amplitude value, returns the final unitary
over the interval ``[0, (T // dt) * dt]`` for an on-resonance drive with envelope shape given by
``envelope_func`` above. Running compiled versions of these functions gives a sense of the speeds
attainable by these solvers.

.. plot::
    :context: close-figs
    :include-source:

    from qiskit_dynamics import Signal
    from jax import jit

    # Jit the function to improve performance for repeated calls
    @jit
    def dyson_sim(amp):
        """For a given envelope amplitude, simulate the final unitary using the
        Dyson solver.
        """
        drive_signal = Signal(lambda t: amp * envelope_func(t), carrier_freq=v)
        return dyson_solver.solve(
            signals=[drive_signal],
            y0=np.eye(dim, dtype=complex),
            t0=0.,
            n_steps=int(T // dt)
        ).y[-1]

First run includes compile time.

.. plot::
    :context: close-figs
    :include-source:

    import time

    start_time = time.time()
    
    yf_dyson = dyson_sim(1.).block_until_ready()

    print(f"Run time: {time.time() - start_time}")


Once JIT compilation has been performance we can benchmark the performance of the JIT-compiled
solver:

.. plot::
    :context: close-figs
    :include-source:

    start_time = time.time()
    
    yf_dyson = dyson_sim(1.).block_until_ready()

    print(f"Run time: {time.time() - start_time}")


4. Comparison to traditional ODE solver
---------------------------------------

We now construct the same simulation using a standard solver to compare accuracy and simulation
speed.

.. plot::
    :context: close-figs
    :include-source:

    from qiskit_dynamics import Solver

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=[drive_hamiltonian],
        rotating_frame=static_hamiltonian
    )

    # specify tolerance as an argument to run the simulation at different tolerances
    def ode_sim(amp, tol):
        drive_signal = Signal(lambda t: amp * envelope_func(t), carrier_freq=v)
        res = solver.solve(
            t_span=[0., int(T // dt) * dt],
            y0=np.eye(dim, dtype=complex),
            signals=[drive_signal],
            method='jax_odeint',
            atol=tol,
            rtol=tol
        )
        return res.y[-1]

Simulate with low tolerance for comparison to high accuracy solution.

.. plot::
    :context: close-figs
    :include-source:

    yf_low_tol = ode_sim(1., 1e-13)
    np.linalg.norm(yf_low_tol - yf_dyson)


For speed comparison, compile at a tolerance with similar accuracy.

.. plot::
    :context: close-figs
    :include-source:

    jit_ode_sim = jit(lambda amp: ode_sim(amp, 1e-8))

    start_time = time.time()

    yf_ode = jit_ode_sim(1.).block_until_ready()

    print(f"Run time: {time.time() - start_time}")

Measure compiled time.

.. plot::
    :context: close-figs
    :include-source:

    start_time = time.time()

    yf_ode = jit_ode_sim(1.).block_until_ready()

    print(f"Run time: {time.time() - start_time}")


Confirm similar accuracy solution.

.. plot::
    :context: close-figs
    :include-source:

    np.linalg.norm(yf_low_tol - yf_ode)

Here we see that, once compiled, the Dyson-based solver has a significant speed advantage over the
traditional solver, at the expense of the initial compilation time and the technical aspect of using
the solver.

5. How-to construct and simulate using the Magnus-based perturbation solver
---------------------------------------------------------------------------

Next, we repeat our example using the Magnus-based perturbative solver. Setup of the
:class:`.MagnusSolver` is similar to the :class:`.DysonSolver`, but it uses the Magnus expansion and
matrix exponentiation to simulate over each fixed time step.

.. plot::
    :context: close-figs
    :include-source:

    from qiskit_dynamics import MagnusSolver

    start_time = time.time()

    dt = 0.1
    magnus_solver = MagnusSolver(
        operators=[-1j * drive_hamiltonian],
        rotating_frame=-1j * static_hamiltonian,
        dt=dt,
        carrier_freqs=[v],
        chebyshev_orders=[1],
        expansion_order=3,
        integration_method='jax_odeint',
        atol=1e-12,
        rtol=1e-12
    )

    print(f"Run time: {time.time() - start_time}")


Setup simulation function.

.. plot::
    :context: close-figs
    :include-source:

    @jit
    def magnus_sim(amp):
        drive_signal = Signal(lambda t: amp * envelope_func(t), carrier_freq=v)
        return magnus_solver.solve(
            signals=[drive_signal],
            y0=np.eye(dim, dtype=complex),
            t0=0.,
            n_steps=int(T // dt)
        ).y[-1]


First run includes compile time.

.. plot::
    :context: close-figs
    :include-source:

    start_time = time.time()

    yf_magnus = magnus_sim(1.).block_until_ready()

    print(f"Run time: {time.time() - start_time}")

Second run demonstrates speed of the simulation.

.. plot::
    :context: close-figs
    :include-source:

    start_time = time.time()
    
    yf_magnus = magnus_sim(1.).block_until_ready()

    print(f"Run time: {time.time() - start_time}")


.. plot::
    :context: close-figs
    :include-source:

    np.linalg.norm(yf_magnus - yf_low_tol)


Observe comparable accuracy at a lower order in the expansion, albeit with a modest speed up as
compared to the Dyson-based solver.

.. footbibliography::

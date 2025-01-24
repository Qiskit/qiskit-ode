# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,no-member,attribute-defined-outside-init

r"""
Solver functions.
"""

from typing import Optional, Union, Callable, Tuple, TypeVar
from warnings import warn

from scipy.integrate import OdeSolver

from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError
from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics.arraylias import ArrayLike

from qiskit_dynamics.models import (
    BaseGeneratorModel,
    GeneratorModel,
    LindbladModel,
)
from qiskit_dynamics.models.hamiltonian_model import HamiltonianModel

from .solver_utils import is_lindblad_model_not_vectorized
from .fixed_step_solvers import (
    RK4_solver,
    jax_RK4_solver,
    scipy_expm_solver,
    lanczos_diag_solver,
    jax_lanczos_diag_solver,
    jax_expm_solver,
    jax_RK4_parallel_solver,
    jax_expm_parallel_solver,
)
from .scipy_solve_ivp import scipy_solve_ivp, SOLVE_IVP_METHODS
from .jax_odeint import jax_odeint
from .diffrax_solver import diffrax_solver

ODE_METHODS = (
    ["RK45", "RK23", "BDF", "DOP853", "Radau", "LSODA"]  # scipy solvers
    + ["RK4"]  # fixed step solvers
    + ["jax_odeint", "jax_RK4"]  # jax solvers
)
LMDE_METHODS = [
    "scipy_expm",
    "lanczos_diag",
    "jax_lanczos_diag",
    "jax_expm",
    "jax_expm_parallel",
    "jax_RK4_parallel",
]

# diffrax solver type placeholder
# pylint: disable=typevar-name-mismatch
DiffraxAbstractSolver = TypeVar("AbstractSolver")


def _is_jax_method(method: any) -> bool:
    """Check if method is a jax solver method."""
    if method in [
        "jax_odeint",
        "jax_RK4",
        "jax_expm",
        "jax_expm_parallel",
        "jax_RK4_parallel",
        "jax_lanczos_diag",
    ]:
        return True

    # only other jax methods are diffrax methods
    return _is_diffrax_method(method)


def _is_diffrax_method(method: any) -> bool:
    """Check if method is a diffrax method."""
    try:
        from diffrax import AbstractSolver

        return isinstance(method, AbstractSolver)
    except ImportError:
        return False


def _lanczos_validation(
    rhs: Union[Callable, BaseGeneratorModel],
    t_span: ArrayLike,
    y0: ArrayLike,
    k_dim: int,
):
    """Validation checks to run lanczos based solvers."""
    t_span = unp.asarray(t_span)
    y0 = unp.asarray(y0)
    if isinstance(rhs, BaseGeneratorModel):
        if not isinstance(rhs, HamiltonianModel):
            raise QiskitError(
                """Lanczos solver can only be used for HamiltonianModel or function-based
                    anti-Hermitian generators."""
            )
        if "sparse" not in rhs.array_library:
            warn(
                """lanczos_diag should be used with a generator in sparse mode
                for better performance.""",
                category=Warning,
                stacklevel=2,
            )

    dim = rhs(t_span[0]).shape[0]
    if k_dim > dim:
        raise QiskitError("k_dim can be no larger than the dimension of the generator.")

    if y0.ndim not in [1, 2]:
        raise QiskitError("y0 must be 1d or 2d.")


def solve_ode(
    rhs: Union[Callable, BaseGeneratorModel],
    t_span: ArrayLike,
    y0: ArrayLike,
    method: Optional[Union[str, OdeSolver, DiffraxAbstractSolver]] = "DOP853",
    t_eval: Optional[ArrayLike] = None,
    **kwargs,
) -> OdeResult:
    r"""General interface for solving Ordinary Differential Equations (ODEs).

    ODEs are differential equations of the form

    .. math::

        \dot{y}(t) = f(t, y(t)),

    where :math:`f` is a callable function and the state :math:`y(t)` is an
    arbitrarily-shaped complex array.

    The ``method`` argument exposes a variety of underlying ODE solvers. Optional
    arguments for any of the solver routines can be passed via ``kwargs``.
    Available methods are:

    - ``scipy.integrate.solve_ivp`` - supports methods
      ``['RK45', 'RK23', 'BDF', 'DOP853', 'Radau', 'LSODA']`` or by passing a valid
      ``scipy`` :class:`OdeSolver` instance.
    - ``'RK4'``: A fixed-step 4th order Runge-Kutta solver.
      Requires additional kwarg ``max_dt``, indicating the maximum step
      size to take. This solver will break integration periods into even
      sub-intervals no larger than ``max_dt``, and step over each sub-interval
      using the standard 4th order Runge-Kutta integration rule.
    - ``'jax_RK4'``: JAX backend implementation of ``'RK4'`` method.
    - ``'jax_odeint'``: Calls ``jax.experimental.ode.odeint`` variable step solver.
    - ``diffrax.diffeqsolve`` - a JAX solver function, called by passing ``method``
      as a valid ``diffrax.AbstractSolver`` instance. Requires the ``diffrax`` library.

    Results are returned as a :class:`OdeResult` object.

    Args:
        rhs: RHS function :math:`f(t, y)`.
        t_span: ``Tuple`` or ``list`` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
        t_eval: Times at which to return the solution. Must lie within ``t_span``. If unspecified,
                the solution will be returned at the points in ``t_span``.
        **kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If specified method does not exist.
    """

    if method not in ODE_METHODS and not (
        (isinstance(method, type) and (issubclass(method, OdeSolver))) or _is_diffrax_method(method)
    ):
        raise QiskitError("Method " + str(method) + " not supported by solve_ode.")

    y0 = unp.asarray(y0)

    if isinstance(rhs, BaseGeneratorModel):
        _, solver_rhs, y0, model_in_frame_basis = setup_generator_model_rhs_y0_in_frame_basis(
            rhs, y0
        )
    else:
        solver_rhs = rhs

    # solve the problem using specified method
    if method in SOLVE_IVP_METHODS or (isinstance(method, type) and issubclass(method, OdeSolver)):
        results = scipy_solve_ivp(solver_rhs, t_span, y0, method, t_eval=t_eval, **kwargs)
    elif isinstance(method, str) and method == "RK4":
        results = RK4_solver(solver_rhs, t_span, y0, t_eval=t_eval, **kwargs)
    elif isinstance(method, str) and method == "jax_RK4":
        results = jax_RK4_solver(solver_rhs, t_span, y0, t_eval=t_eval, **kwargs)
    elif isinstance(method, str) and method == "jax_odeint":
        results = jax_odeint(solver_rhs, t_span, y0, t_eval=t_eval, **kwargs)
    elif _is_diffrax_method(method):
        results = diffrax_solver(solver_rhs, t_span, y0, method=method, t_eval=t_eval, **kwargs)

    # convert results out of frame basis if necessary
    if isinstance(rhs, BaseGeneratorModel):
        if not model_in_frame_basis:
            results.y = results_y_out_of_frame_basis(rhs, results.y, y0.ndim)

        # convert model back to original basis
        rhs.in_frame_basis = model_in_frame_basis

    return results


def solve_lmde(
    generator: Union[Callable, BaseGeneratorModel],
    t_span: ArrayLike,
    y0: ArrayLike,
    method: Optional[Union[str, OdeSolver, DiffraxAbstractSolver]] = "DOP853",
    t_eval: Optional[ArrayLike] = None,
    **kwargs,
) -> OdeResult:
    r"""General interface for solving Linear Matrix Differential Equations (LMDEs)
    in standard form.

    LMDEs in standard form are differential equations of the form:

    .. math::

        \dot{y}(t) = G(t)y(t).

    where :math:`G(t)` is a square matrix valued-function called the *generator*, and :math:`y(t)`
    is an array of appropriate shape.

    Thus function accepts :math:`G(t)` as a ``qiskit_dynamics`` model class, or as an arbitrary
    callable.

    .. note::

        Not all model classes are by-default in standard form. E.g.
        :class:`~qiskit_dynamics.models.LindbladModel` represents an LMDE which is not typically
        written in standard form. As such, using LMDE-specific methods with this generator requires
        the equation to be vectorized.

    The ``method`` argument exposes solvers specialized to both LMDEs, as well as general ODE
    solvers. If the method is not specific to LMDEs, the problem will be passed to
    :meth:`~qiskit_dynamics.solve_ode` by automatically setting up the RHS function :math:`f(t, y) =
    G(t)y`.

    Optional arguments for any of the solver routines can be passed via ``kwargs``. Available
    LMDE-specific methods are:

    - ``'scipy_expm'``: A fixed-step matrix-exponential solver using ``scipy.linalg.expm``. Requires
      additional kwarg ``max_dt`` indicating the maximum step size to take. This solver will break
      integration periods into even sub-intervals no larger than ``max_dt`` and solve over each
      sub-interval. The optional kwarg ``magnus_order`` controls the integration rule: if
      ``magnus_order==1``, the generator is sampled at the interval midpoint and exponentiated, and
      if ``magnus_order==2`` or ``magnus_order==3``, higher-order exponentiation rules are adopted
      from :footcite:`blanes_magnus_2009`. The ``magnus_order`` parameter defaults to ``1``.
    - ``'lanczos_diag'``: A fixed-step matrix-exponential solver, similar to ``'scipy_expm'`` but
      restricted to anti-Hermitian generators. The matrix exponential is performed by diagonalizing
      an approximate projection of the generator to a small subspace (the Krylov Subspace), obtained
      via the Lanczos algorithm, and then exponentiating the eigenvalues. Requires additional kwargs
      ``max_dt`` and ``k_dim`` indicating the maximum step size to take and Krylov subspace
      dimension, respectively. ``k_dim`` acts as an adjustable accuracy parameter and can be no
      larger than the dimension of the generator. The method is recommended for sparse systems with
      large dimension.
    - ``'jax_lanczos_diag'``: JAX implementation of ``'lanczos_diag'``, with the same arguments and
      behaviour. Note that this method contains calls to ``jax.numpy.eigh``, which may have limited
      validity when automatically differentiated.
    - ``'jax_expm'``: JAX-implemented version of ``'scipy_expm'``, with the same arguments and
      behaviour. Note that this method cannot be used for a model using a sparse array library.
    - ``'jax_expm_parallel'``: Same as ``'jax_expm'``, however all loops are implemented using
      parallel operations. I.e. all matrix-exponentials for taking a single step are computed in
      parallel using ``jax.vmap``, and are subsequently multiplied together in parallel using
      ``jax.lax.associative_scan``. This method is only recommended for use with GPU execution. Note
      that this method cannot be used for a model using a sparse array library.
    - ``'jax_RK4_parallel'``: 4th order Runge-Kutta fixed step solver. Under the assumption of the
      structure of an LMDE, utilizes the same parallelization approach as ``'jax_expm_parallel'``,
      however the single step rule is the standard 4th order Runge-Kutta rule, rather than
      matrix-exponentiation. Requires and utilizes the ``max_dt`` kwarg in the same manner as
      ``method='scipy_expm'``. This method is only recommended for use with GPU execution.

    Results are returned as a :class:`OdeResult` object.

    Args:
        generator: Representation of generator function :math:`G(t)`.
        t_span: ``Tuple`` or `list` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
        t_eval: Times at which to return the solution. Must lie within ``t_span``. If unspecified,
            the solution will be returned at the points in ``t_span``.
        **kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If specified method does not exist,
                     if dimension of ``y0`` is incompatible with generator dimension, or if an
                     LMDE-specific method is passed with a LindbladModel.
    Additional Information:
        While all :class:`~qiskit_dynamics.models.BaseGeneratorModel` subclasses represent LMDEs,
        they are not all in standard form by defualt. Using an LMDE-specific models like
        :class:`~qiskit_dynamics.models.LindbladModel` requires first setting the model to be
        vectorized.
    """

    # delegate to solve_ode if necessary
    if (
        method in ODE_METHODS
        or (isinstance(method, type) and (issubclass(method, OdeSolver)))
        or _is_diffrax_method(method)
    ):
        if isinstance(generator, BaseGeneratorModel):
            rhs = generator
        else:
            # treat generator as a function
            def rhs(t, y):
                return generator(t) @ y

        return solve_ode(rhs, t_span, y0, method=method, t_eval=t_eval, **kwargs)

    # raise error if neither an ODE_METHOD or an LMDE_METHOD
    if method not in LMDE_METHODS:
        raise QiskitError(f"Method {method} not supported by solve_lmde.")

    # lmde-specific methods can't be used with LindbladModel unless vectorized
    if is_lindblad_model_not_vectorized(generator):
        raise QiskitError(
            "LMDE-specific methods with LindbladModel requires setting a vectorized=True."
        )

    y0 = unp.asarray(y0)

    # setup generator and rhs functions to pass to numerical methods
    if isinstance(generator, BaseGeneratorModel):
        solver_generator, _, y0, model_in_frame_basis = setup_generator_model_rhs_y0_in_frame_basis(
            generator, y0
        )
    else:
        solver_generator = generator

    if method == "scipy_expm":
        results = scipy_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    elif "lanczos_diag" in method:
        _lanczos_validation(generator, t_span, y0, kwargs["k_dim"])
        if method == "lanczos_diag":
            results = lanczos_diag_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
        elif method == "jax_lanczos_diag":
            results = jax_lanczos_diag_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    elif method == "jax_expm":
        if isinstance(generator, BaseGeneratorModel) and "sparse" in generator.array_library:
            raise QiskitError("jax_expm cannot be used with a generator in sparse mode.")
        results = jax_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    elif method == "jax_expm_parallel":
        results = jax_expm_parallel_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    elif method == "jax_RK4_parallel":
        results = jax_RK4_parallel_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)

    # convert results to correct basis if necessary
    if isinstance(generator, BaseGeneratorModel):
        if not model_in_frame_basis:
            # pylint: disable=possibly-used-before-assignment
            results.y = results_y_out_of_frame_basis(generator, results.y, y0.ndim)

        generator.in_frame_basis = model_in_frame_basis

    return results


def setup_generator_model_rhs_y0_in_frame_basis(
    generator_model: BaseGeneratorModel, y0: ArrayLike
) -> Tuple[Callable, Callable, ArrayLike]:
    """Helper function for setting up a subclass of
    :class:`~qiskit_dynamics.models.BaseGeneratorModel` to be solved in the frame basis.

    Note: this function modifies ``generator_model`` to function in the frame basis.

    Args:
        generator_model: Subclass of :class:`~qiskit_dynamics.models.BaseGeneratorModel`.
        y0: Initial state.

    Returns:
        Callable for generator in frame basis, callable for RHS in frame basis, y0
        in frame basis, and boolean indicating whether model was already specified in frame basis.
    """

    model_in_frame_basis = generator_model.in_frame_basis

    # if model not specified in frame basis, transform initial state into frame basis
    if not model_in_frame_basis:
        if isinstance(generator_model, LindbladModel) and generator_model.vectorized:
            if generator_model.rotating_frame.frame_basis is not None:
                y0 = generator_model.rotating_frame.vectorized_frame_basis_adjoint @ y0
        elif isinstance(generator_model, LindbladModel):
            y0 = generator_model.rotating_frame.operator_into_frame_basis(y0)
        elif isinstance(generator_model, GeneratorModel):
            y0 = generator_model.rotating_frame.state_into_frame_basis(y0)

    # set model to operator in frame basis
    generator_model.in_frame_basis = True

    # define rhs functions in frame basis
    def generator(t):
        return generator_model(t)

    def rhs(t, y):
        return generator_model(t, y)

    return generator, rhs, y0, model_in_frame_basis


def results_y_out_of_frame_basis(
    generator_model: BaseGeneratorModel, results_y: ArrayLike, y0_ndim: int
) -> ArrayLike:
    """Convert the results of a simulation for :class:`~qiskit_dynamics.models.BaseGeneratorModel`
    out of the frame basis.

    Args:
        generator_model: Subclass of :class:`~qiskit_dynamics.models.BaseGeneratorModel`.
        results_y: Array whose first index corresponds to the evaluation points of the state
                   for the results of ``solve_lmde`` or ``solve_ode``.
        y0_ndim: Number of dimensions of initial state.

    Returns:
        Callable for generator in frame basis, Callable for RHS in frame basis, and y0
        transformed to frame basis.
    """
    # for left multiplication cases, if number of input dimensions is 1
    # vectorized basis transformation requires transposing before and after
    if y0_ndim == 1:
        results_y = results_y.T

    if isinstance(generator_model, LindbladModel) and generator_model.vectorized:
        if generator_model.rotating_frame.frame_basis is not None:
            results_y = generator_model.rotating_frame.vectorized_frame_basis @ results_y
    elif isinstance(generator_model, LindbladModel):
        results_y = generator_model.rotating_frame.operator_out_of_frame_basis(results_y)
    else:
        results_y = generator_model.rotating_frame.state_out_of_frame_basis(results_y)

    if y0_ndim == 1:
        results_y = results_y.T

    return results_y

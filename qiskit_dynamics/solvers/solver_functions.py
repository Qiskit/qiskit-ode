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

from typing import Optional, Union, Callable, Tuple, List

from scipy.integrate import OdeSolver

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator

from qiskit import QiskitError
from qiskit_dynamics import dispatch
from qiskit_dynamics.dispatch import Array, requires_backend

from qiskit_dynamics.models import (
    GeneratorModel,
    RotatingFrame,
    rotating_wave_approximation,
    HamiltonianModel,
    LindbladModel,
)

from ..models.generator_models import BaseGeneratorModel

from .solver_utils import is_lindblad_model_not_vectorized
from .fixed_step_solvers import scipy_expm_solver, jax_expm_solver
from .scipy_solve_ivp import scipy_solve_ivp, SOLVE_IVP_METHODS
from .jax_odeint import jax_odeint

try:
    from jax.lax import scan
except ImportError:
    pass


ODE_METHODS = ["RK45", "RK23", "BDF", "DOP853", "Radau", "LSODA"] + ["jax_odeint"]
LMDE_METHODS = ["scipy_expm", "jax_expm"]


def solve_ode(
    rhs: Union[Callable, BaseGeneratorModel],
    t_span: Array,
    y0: Array,
    method: Optional[Union[str, OdeSolver]] = "DOP853",
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    **kwargs,
):
    r"""General interface for solving Ordinary Differential Equations (ODEs).

    ODEs are differential equations of the form

    .. math::

        \dot{y}(t) = f(t, y(t)),

    where :math:`f` is a callable function and the state :math:`y(t)` is an
    arbitrarily-shaped complex :class:`Array`.

    The ``method`` argument exposes a variety of underlying ODE solvers. Optional
    arguments for any of the solver routines can be passed via ``kwargs``.
    Available methods are:

    - ``scipy.integrate.solve_ivp`` - supports methods
      ``['RK45', 'RK23', 'BDF', 'DOP853', 'Radau', 'LSODA']`` or by passing a valid
      ``scipy`` :class:`OdeSolver` instance.
    - ``jax.experimental.ode.odeint`` - accessed via passing
      ``method='jax_odeint'``.

    Results are returned as a :class:`OdeResult` object.

    Args:
        rhs: RHS function :math:`f(t, y)`.
        t_span: ``Tuple`` or ``list`` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
        t_eval: Times at which to return the solution. Must lie within ``t_span``. If unspecified,
                the solution will be returned at the points in ``t_span``.
        kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If specified method does not exist.
    """

    if method not in ODE_METHODS and not (
        isinstance(method, type) and issubclass(method, OdeSolver)
    ):
        raise QiskitError("Method " + str(method) + " not supported by solve_ode.")

    y0 = Array(y0)

    if isinstance(rhs, BaseGeneratorModel):
        _, solver_rhs, y0 = setup_generator_model_rhs_y0_in_frame_basis(rhs, y0)
    else:
        solver_rhs = rhs

    # solve the problem using specified method
    if method in SOLVE_IVP_METHODS or (isinstance(method, type) and issubclass(method, OdeSolver)):
        results = scipy_solve_ivp(solver_rhs, t_span, y0, method, t_eval, **kwargs)
    elif isinstance(method, str) and method == "jax_odeint":
        results = jax_odeint(solver_rhs, t_span, y0, t_eval, **kwargs)

    # convert results to correct basis if necessary
    if isinstance(rhs, BaseGeneratorModel):
        results.y = results_y_out_of_frame_basis(rhs, Array(results.y), y0.ndim)

    return results


def solve_lmde(
    generator: Union[Callable, BaseGeneratorModel],
    t_span: Array,
    y0: Array,
    method: Optional[Union[str, OdeSolver]] = "DOP853",
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    **kwargs,
):
    r"""General interface for solving Linear Matrix Differential Equations (LMDEs)
    in standard form.

    LMDEs in standard form are differential equations of the form:

    .. math::

        \dot{y}(t) = G(t)y(t).

    where :math:`G(t)` is a square matrix valued-function called the *generator*,
    and :math:`y(t)` is an :class:`Array` of appropriate shape.

    Thus function accepts :math:`G(t)` as a ``qiskit_dynamics`` model class,
    or as an arbitrary callable.

    .. note::

        Not all model classes are by-default in standard form. E.g.
        :class:`~qiskit_dynamics.models.LindbladModel` represents an LMDE which is not
        typically written in standard form. As such, using LMDE-specific methods with this generator
        requires setting a vectorized evaluation mode.

    The ``method`` argument exposes solvers specialized to both LMDEs, as
    well as general ODE solvers. If the method is not specific to LMDEs,
    the problem will be passed to :meth:`~qiskit_dynamics.solve_ode` by automatically setting
    up the RHS function :math:`f(t, y) = G(t)y`.

    Optional arguments for any of the solver routines can be passed via ``kwargs``.
    Available LMDE-specific methods are:

    - ``'scipy_expm'``: A matrix-exponential solver using ``scipy.linalg.expm``.
      Requires additional kwarg ``max_dt``, indicating the maximum step
      size to take. This solver will break integration periods into even
      sub-intervals no larger than ``max_dt``, and solve over each sub-intervals via
      matrix exponentiation of the generator sampled at the midpoint.
    - ``'jax_expm'``: JAX-implemented version of ``'scipy_expm'``, with the same arguments and
      logic.


    Results are returned as a :class:`OdeResult` object.

    Args:
        generator: Representation of generator function :math:`G(t)`.
        t_span: ``Tuple`` or `list` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
        t_eval: Times at which to return the solution. Must lie within ``t_span``. If unspecified,
                the solution will be returned at the points in ``t_span``.
        kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If specified method does not exist,
                     if dimension of ``y0`` is incompatible with generator dimension,
                     or if an LMDE-specific method is passed with a LindbladModel.
    Additional Information:
        While all :class:`~qiskit_dynamics.models.BaseGeneratorModel` subclasses
        represent LMDEs, they are not all in standard form by defualt. Using an
        LMDE-specific models like :class:`~qiskit_dynamics.models.LindbladModel`
        requires first setting a vectorized evaluation mode.
    """

    # delegate to solve_ode if necessary
    if method in ODE_METHODS or (isinstance(method, type) and issubclass(method, OdeSolver)):
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
            """LMDE-specific methods with LindbladModel requires setting a
               vectorized evaluation mode."""
        )

    y0 = Array(y0)

    # setup generator and rhs functions to pass to numerical methods
    if isinstance(generator, BaseGeneratorModel):
        solver_generator, _, y0 = setup_generator_model_rhs_y0_in_frame_basis(generator, y0)
    else:
        solver_generator = generator

    if method == "scipy_expm":
        results = scipy_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    elif method == "jax_expm":
        results = jax_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)

    # convert results to correct basis if necessary
    if isinstance(generator, BaseGeneratorModel):
        results.y = results_y_out_of_frame_basis(generator, Array(results.y), y0.ndim)

    return results


def setup_generator_model_rhs_y0_in_frame_basis(
    generator_model: BaseGeneratorModel, y0: Array
) -> Tuple[Callable, Callable, Array]:
    """Helper function for setting up a subclass of
    :class:`~qiskit_dynamics.models.BaseGeneratorModel` to be solved in the frame basis.

    Args:
        generator_model: Subclass of :class:`~qiskit_dynamics.models.BaseGeneratorModel`.
        y0: Initial state.

    Returns:
        Callable for generator in frame basis, Callable for RHS in frame basis, and y0
        transformed to frame basis.
    """

    if (
        isinstance(generator_model, LindbladModel)
        and "vectorized" in generator_model.evaluation_mode
    ):
        if generator_model.rotating_frame.frame_basis is not None:
            y0 = generator_model.rotating_frame.vectorized_frame_basis_adjoint @ y0
    elif isinstance(generator_model, LindbladModel):
        y0 = generator_model.rotating_frame.operator_into_frame_basis(y0)
    elif isinstance(generator_model, GeneratorModel):
        y0 = generator_model.rotating_frame.state_into_frame_basis(y0)

    # define rhs functions in frame basis
    def generator(t):
        return generator_model(t, in_frame_basis=True)

    def rhs(t, y):
        return generator_model(t, y, in_frame_basis=True)

    return generator, rhs, y0


def results_y_out_of_frame_basis(
    generator_model: BaseGeneratorModel, results_y: Array, y0_ndim: int
) -> Array:
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

    if (
        isinstance(generator_model, LindbladModel)
        and "vectorized" in generator_model.evaluation_mode
    ):
        if generator_model.rotating_frame.frame_basis is not None:
            results_y = generator_model.rotating_frame.vectorized_frame_basis @ results_y
    elif isinstance(generator_model, LindbladModel):
        results_y = generator_model.rotating_frame.operator_out_of_frame_basis(results_y)
    else:
        results_y = generator_model.rotating_frame.state_out_of_frame_basis(results_y)

    if y0_ndim == 1:
        results_y = results_y.T

    return results_y

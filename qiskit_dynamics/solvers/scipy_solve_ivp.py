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
# pylint: disable=invalid-name

"""
Wrapper for calling scipy.integrate.solve_ivp.
"""

from typing import Callable, Union, Optional, Tuple, List

import numpy as np
from scipy.integrate import solve_ivp, OdeSolver
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError
from qiskit_dynamics.dispatch import Array
from ..type_utils import StateTypeConverter

# Supported scipy ODE methods
COMPLEX_METHODS = ["RK45", "RK23", "BDF", "DOP853"]
REAL_METHODS = ["LSODA", "Radau"]
SOLVE_IVP_METHODS = COMPLEX_METHODS + REAL_METHODS


def scipy_solve_ivp(
    rhs: Callable,
    t_span: Array,
    y0: Array,
    method: Union[str, OdeSolver],
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    **kwargs,
):
    """Routine for calling `scipy.integrate.solve_ivp`.

    Args:
        rhs: Callable of the form :math:`f(t, y)`.
        t_span: Interval to solve over.
        y0: Initial state.
        method: Solver method.
        t_eval: Points at which to evaluate the solution.
        kwargs: Optional arguments to be passed to ``solve_ivp``.

    Returns:
        OdeResult: results object

    Raises:
        QiskitError: If unsupported kwarg present.
    """

    if kwargs.get("dense_output", False) is True:
        raise QiskitError("dense_output not supported for solve_ivp.")

    # solve_ivp requires 1d arrays internally
    internal_state_spec = {"type": "array", "ndim": 1}
    type_converter = StateTypeConverter.from_outer_instance_inner_type_spec(y0, internal_state_spec)

    # modify the rhs to work with 1d arrays or real solvers
    rhs = type_converter.rhs_outer_to_inner(rhs)

    # convert y0 to the flattened version
    y0 = type_converter.outer_to_inner(y0)

    # Check if solver is real only
    # TODO: Also check if model or y0 are complex
    #       if they are both real we don't need to embed.
    embed_real = method in REAL_METHODS
    if embed_real:
        rhs = real_rhs(rhs)
        y0 = c2r(y0)

    results = solve_ivp(rhs, t_span=t_span, y0=y0.data, t_eval=t_eval, method=method, **kwargs)
    if embed_real:
        results.y = r2c(results.y)

    # convert to the standardized results format
    # solve_ivp returns the states as a 2d array with columns being the states
    results.y = results.y.transpose()
    results.y = Array([type_converter.inner_to_outer(y) for y in results.y])

    return OdeResult(**dict(results))


def real_rhs(rhs):
    """Convert complex RHS to real RHS function"""

    def _real_rhs(t, y):
        return c2r(rhs(t, r2c(y)))

    return _real_rhs


def c2r(arr):
    """Convert complex array to a real array"""
    return np.concatenate([np.real(arr), np.imag(arr)])


def r2c(arr):
    """Convert a real array to a complex array"""
    size = arr.shape[0] // 2
    return arr[:size] + 1j * arr[size:]

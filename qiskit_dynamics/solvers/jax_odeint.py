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
# pylint: disable=invalid-name

"""
Wrapper for jax.experimental.ode.odeint
"""

from typing import Callable, Optional, Union, Tuple, List
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit_dynamics.dispatch import requires_backend, Array, wrap

from .solver_utils import merge_t_args, trim_t_results

try:
    from jax.experimental.ode import odeint as _odeint

    odeint = wrap(_odeint)
except ImportError:
    pass


@requires_backend("jax")
def jax_odeint(
    rhs: Callable,
    t_span: Array,
    y0: Array,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    **kwargs,
):
    """Routine for calling `jax.experimental.ode.odeint`

    Args:
        rhs: Callable of the form :math:`f(t, y)`
        t_span: Interval to solve over.
        y0: Initial state.
        t_eval: Optional list of time points at which to return the solution.
        kwargs: Optional arguments to be passed to ``odeint``.

    Returns:
        OdeResult: Results object.
    """

    t_list = merge_t_args(t_span, t_eval)

    # determine direction of integration
    t_direction = np.sign(Array(t_list[-1] - t_list[0], backend="jax")).data

    results = odeint(
        lambda y, t: t_direction * Array(rhs(t_direction * t, y)).data,
        y0=Array(y0).data,
        t=t_direction * Array(t_list).data,
        **kwargs,
    )

    results = OdeResult(t=t_list, y=Array(results, backend="jax"))

    return trim_t_results(results, t_span, t_eval)

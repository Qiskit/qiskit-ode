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
Utility functions for solvers.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.models import LindbladModel


def is_lindblad_model_vectorized(obj: any) -> bool:
    """Return True if obj is a vectorized LindbladModel."""
    return isinstance(obj, LindbladModel) and ("vectorized" in obj.evaluation_mode)


def is_lindblad_model_not_vectorized(obj: any) -> bool:
    """Return True if obj is a non-vectorized LindbladModel."""
    return isinstance(obj, LindbladModel) and ("vectorized" not in obj.evaluation_mode)


def merge_t_args(
    t_span: Union[List, Tuple, Array], t_eval: Optional[Union[List, Tuple, Array]] = None
) -> Union[List, Tuple, Array]:
    """Merge ``t_span`` and ``t_eval`` into a single array without
    duplicates. Validity of the passed ``t_span`` and ``t_eval``
    follow scipy ``solve_ivp`` validation logic:
    ``t_eval`` must be contained in ``t_span``, and be strictly
    increasing if ``t_span[1] > t_span[0]`` or strictly
    decreasing if ``t_span[1] < t_span[0]``.

    Note: this is done explicitly with ``numpy``, and hence this is
    not differentiable or compilable using jax.

    If ``t_eval is None`` returns ``t_span`` with no modification.

    Args:
        t_span: Interval to solve over.
        t_eval: Time points to include in returned results.

    Returns:
        Union[List, Tuple, Array]: Combined list of times.

    Raises:
        ValueError: If one of several validation checks fail.
    """

    if t_eval is None:
        return t_span

    t_span = Array(t_span, backend="numpy")

    t_min = np.min(t_span)
    t_max = np.max(t_span)
    t_direction = np.sign(t_span[1] - t_span[0])

    t_eval = Array(t_eval, backend="numpy")

    if t_eval.ndim > 1:
        raise ValueError("t_eval must be 1 dimensional.")

    if np.min(t_eval) < t_min or np.max(t_eval) > t_max:
        raise ValueError("t_eval entries must lie in t_span.")

    diff = np.diff(t_eval)

    if np.any(t_direction * diff <= 0.0):
        raise ValueError("t_eval must be ordered according to the direction of integration.")

    # if endpoints are not included in t_span, add them
    if t_eval[0] != t_span[0]:
        t_eval = np.append(t_span[0], t_eval)

    if t_span[1] != t_eval[-1]:
        t_eval = np.append(t_eval, t_span[1])

    return Array(t_eval, backend="numpy")


def trim_t_results(
    results: OdeResult,
    t_span: Union[List, Tuple, Array],
    t_eval: Optional[Union[List, Tuple, Array]] = None,
) -> OdeResult:
    """Trim ``OdeResult`` object based on value of ``t_span`` and ``t_eval``.

    Args:
        results: Result object, assumed to contain solution at time points
                 from the output of ``validate_and_merge_t_span_t_eval(t_span, t_eval)``.
        t_span: Interval to solve over.
        t_eval: Time points to include in returned results.

    Returns:
        OdeResult: Results with only times/solutions in ``t_eval``. If ``t_eval``
                   is ``None``, does nothing, returning solver default output.
    """

    if t_eval is None:
        return results

    t_span = Array(t_span, backend="numpy")

    # remove endpoints if not included in t_eval
    if t_eval[0] != t_span[0]:
        results.t = results.t[1:]
        results.y = Array(results.y[1:])

    if t_eval[-1] != t_span[1]:
        results.t = results.t[:-1]
        results.y = Array(results.y[:-1])

    return results

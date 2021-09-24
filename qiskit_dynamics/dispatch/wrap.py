# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Functions for working with Array dispatch."""

import functools
from types import FunctionType
from typing import Callable

from .array import Array


def wrap(func: Callable, wrap_return: bool = True, decorator: bool = False) -> Callable:
    """Wrap an array backend function to work with Arrays.

    Args:
        func: a function to wrap.
        wrap_return: If ``True`` convert results that are registered array
                     backend types into Array objects (Default: True).
        decorator: If ``True`` the wrapped decorator function ``func`` will
                   also wrap the decorated functions (Default: False).

    Returns:
        Callable: The wrapped function.
    """
    # pylint: disable=protected-access

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):

        # Check if we are wrapping a decorator by checking that
        # the first argument is of FunctionType
        if decorator and args:
            is_decorator = isinstance(args[0], FunctionType)
        else:
            is_decorator = False

        args = tuple(_wrap_function(x) if isinstance(x, FunctionType) else x for x in args)
        kwargs = dict(
            (key, _wrap_function(val)) if isinstance(val, FunctionType) else (key, val)
            for key, val in kwargs.items()
        )

        # Return the wrapped function
        if not is_decorator:
            # Evaluate unwrapped function
            result = _wrap_function(func)(*args, **kwargs)

            # Optional wrap array return types back to Arrays
            if wrap_return:
                result = Array._wrap(result)
            return result

        # Wrap the decorated function returned by the decorator
        decorated = _wrap_function(func)(*args, **kwargs)

        @functools.wraps(args[0])
        def wrapped_decorated(*f_args, **f_kwargs):

            f_args = tuple(_wrap_function(x) if isinstance(x, FunctionType) else x for x in f_args)
            f_kwargs = dict(
                (key, _wrap_function(val)) if isinstance(val, FunctionType) else (key, val)
                for key, val in f_kwargs.items()
            )
            result = _wrap_function(decorated)(*f_args, **f_kwargs)

            if wrap_return:
                result = Array._wrap(result)
            return result

        return wrapped_decorated

    return wrapped_func


def _wrap_function(func: callable) -> callable:
    """Wrap a function to handle Array-like inputs and returns"""

    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):

        # Unwrap inputs
        args = tuple(
            x.__qiskit_array__().data if hasattr(x, "__qiskit_array__") else x for x in args
        )
        kwargs = dict(
            (key, val.__qiskit_array__().data) if hasattr(val, "__qiskit_array__") else (key, val)
            for key, val in kwargs.items()
        )

        # Evaluate function with unwrapped inputs
        result = func(*args, **kwargs)

        # Unwrap result
        if isinstance(result, tuple):
            result = tuple(
                x.__qiskit_array__().data if hasattr(x, "__qiskit_array__") else x for x in result
            )
        elif hasattr(result, "__qiskit_array__"):
            result = result.__qiskit_array__().data
        return result

    return wrapped_function

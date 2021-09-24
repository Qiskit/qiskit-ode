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

"""
=================================================
Dispatch Module (:mod:`qiskit_dynamics.dispatch`)
=================================================

.. currentmodule:: qiskit_dynamics.dispatch

This module contains an :class:`Array` class that wraps N-dimensional array
objects from different libraries for use with Numpy functions through a common
interface and functions for working with Array objects and other libraries.

Contents
========

.. autosummary::
    :toctree: ../stubs/

    Array
    wrap
    set_default_backend
    default_backend
    available_backends
    backend_types
    asarray
    requires_backend


Array Class
===========

The :class:`Array` class provides a Numpy compatible wrapper to supported Python
array libraries. When applying Numpy functions to the array class these will be
dispatched to the corresponding function for the current Array backend, if a
compatible function has been specified in the dispatch system.

The following array libraries have built in support for the dispatch module

* `Numpy <https://numpy.org/>`_
* `Jax <https://github.com/google/jax>`_

Basic Usage
-----------

When using the default ``numpy`` backend :class:`Array` objects can be used
interchangable ``numpy.ndarray``. When ``numpy`` functions are applied to
an :class:`Array` object the return type will be an :class:`Array`
instead of an ``ndarray``

.. jupyter-execute::

    from qiskit_dynamics.dispatch import Array
    import numpy as np

    # Initialize an Array
    a = Array(np.arange(10))

    # Apply numpy ufuncs
    np.cos(a) + 1j * np.sin(a)

For the JAX Array backend only Numpy functions that have a corresponding function
in the ``jax`` library that has been registed with the dispatch module can be
applied to the functions. Trying to apply an unsupported ``numpy`` function to
these arrays will raise an exception.


Attributes and Methods
----------------------

The :class:`Array` class exposes the same methods and attributes of the wrapped
array class and adds two additional attributes

* :attr`~Array.data` which returns the wrapped array object
* :attr`~Array.backend` which returns the backend string of the wrapped array.

All other attributes and methods of the wrapped array are accessable
through this class, but with any array return types wrapped into
:class:`Array` objects. For example

.. jupyter-execute::

    # Call numpy.ndarray method
    a.reshape((2, 5))

Array Initialization
--------------------

An `Array` object can be initialized as `Array(data)` from any ``data`` that is

1. An ``array`` object of one of the supported backends.
2. An object that can be used to initialize the backend array.
3. An object of a class that defines a `__qiskit_array__` method with the
   following signature

    .. code-block:: python

        def __qiskit_array__(self,
                             dtype: Optional[any] = None,
                             backend: Optional[str]=None) -> Array:
            # conversion from class to Array

The :meth:`Array.__init__` method has optional kwargs

* ``dtype`` (``any``): Equivalent to the ``numpy.array`` ``dtype`` kwarg.
  For other array backends the specified dtype must be supported on the
  array backend. If not specified the dytpe will be inferred from the
  input data.
* ``order`` (``str``): Equivalent to the ``numpy.array`` ``order`` kwarg.
  For other array backendsthe specified order value must be supported on
  the array backend.
* ``backend`` (``str``): The array backend to use. If not specified this
  will be inferred from the input data type if it is a backend array
  instance, otherwise it will be the :func:`default_backend`.


Using Arrays with other Libraries
=================================

The :class`Array` class is intended to be used with array functions in
Python libraries that work with any of the supported Array backends.

Wrapping Functions
------------------

Functions from other libraries can be wrapped to work with Array objects using
the :func:`wrap` function as

.. code-block:: python

    from qiskit_dynamics.dispatch import Array, wrap

    # Some 3rd-party library
    import library

    # Wrap library function f
    f = dispatch.wrap(library.f)

    # Apply f to an Array
    a = Array([0, 1, 2, 4])
    b = f(a)

The wrapped function will automatically unwrapping any Array args and kwargs
for calling the wrapped function. If the result of the function is a backend array
or an instance of a class that defines a `__qiskit_array__` method the wrapped
function will return this as an Array object. Similarly if the result is a tuple
this conversion will be applied to each element of the tuple where appropriate.

Wrapping Decorators
-------------------

The :func:`wrap` function can also be used to wrap function decorators by setting
the ``decorator=True`` kwarg.

For example to wrap autograd and jit decorators from the Kax library to work with
Array functions we can do the following

.. code-block:: python

    import jax

    jit = wrap(jax.jit, decorator=True)
    grad = wrap(jax.grad, decorator=True)
    value_and_grad = wrap(jax.value_and_grad, decorator=True)

    f = jit(obj)
    g = grad(obj)
    h = value_and_grad(obj)

.. note:

    Setting ``decorator=True`` when calling :func:`wrap` requires that the
    signature of the function being wrapped is
    ``func(f: Callable, ...) -> Callable``. Using it is equivalent to nested
    wrapping

    .. code-block:: python

        f_wrapped = wrap(func, decorator=True)(f)

    is equivalent to

    .. code-block:: python

        f_wrapped = wrap(wrap(func)(f))
"""

# Import Array
from .array import Array

# Import wrapper function
from .wrap import wrap

# Import dispatch utilities
from .dispatch import (
    set_default_backend,
    default_backend,
    available_backends,
    backend_types,
    asarray,
    requires_backend,
)

# Register backends
from .backends import *

# If only one backend is available, set it as the default
if len(available_backends()) == 1:
    set_default_backend(available_backends()[0])

# Monkey patch quantum info
# pylint: disable= wrong-import-position
from .patch_qi import *

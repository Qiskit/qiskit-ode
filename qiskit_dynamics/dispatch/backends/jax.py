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
"""Register jax backend for Dispatch"""
# pylint: disable=import-error


try:
    import jax
    from jax.interpreters.xla import DeviceArray
    from jax.core import Tracer
    from jax.interpreters.ad import JVPTracer
    from jax.interpreters.partial_eval import JaxprTracer

    JAX_TYPES = (DeviceArray, Tracer, JaxprTracer, JVPTracer)

    try:
        # This class is not in older versions of Jax
        from jax.interpreters.partial_eval import DynamicJaxprTracer

        JAX_TYPES += (DynamicJaxprTracer,)
    except ImportError:
        pass

    from ..dispatch import Dispatch
    import numpy as np
    from .numpy import _numpy_repr

    __all__ = []

    # Custom handling of functions not in jax.numpy
    HANDLED_FUNCTIONS = {}

    @Dispatch.register_asarray("jax", JAX_TYPES)
    def _jax_asarray(array, dtype=None, order=None):
        """Wrapper for jax.numpy.asarray"""
        if (
            isinstance(array, DeviceArray)
            and order is None
            and (dtype is None or dtype == array.dtype)
        ):
            return array
        return jax.numpy.asarray(array, dtype=dtype, order=order)

    @Dispatch.register_repr("jax")
    def _jax_repr(array, prefix="", suffix=""):
        """Wrapper for showing DeviceArray"""
        if hasattr(array, "_value"):
            return _numpy_repr(array._value, prefix=prefix, suffix=suffix)
        return prefix + repr(array) + suffix

    @Dispatch.register_array_ufunc("jax")
    def _jax_array_ufunc(ufunc, method):
        """Wrapper mapping a numpy.ufunc to jax.numpy.ufunc"""
        if method != "__call__":
            return NotImplemented
        name = ufunc.__name__
        if hasattr(jax.numpy, name):
            return getattr(jax.numpy, name)
        return NotImplemented

    @Dispatch.register_array_function("jax")
    def _jax_array_function(func):
        """Wrapper mapping a numpy function to jax.numpy function"""
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func]
        name = func.__name__
        if hasattr(jax.numpy, name):
            return getattr(jax.numpy, name)
        if hasattr(jax.numpy.linalg, name):
            return getattr(jax.numpy.linalg, name)
        return NotImplemented

    # Custom function handling

    @Dispatch.implements(np.copy, HANDLED_FUNCTIONS)
    def _copy(array, order="K"):
        return jax.numpy.array(array, copy=True, order=order)


except ModuleNotFoundError:
    pass

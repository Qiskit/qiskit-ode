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
"""Register numpy backend for Dispatch"""

import numpy
from ..exceptions import DispatchError
from ..dispatch import Dispatch

__all__ = []


@Dispatch.register_asarray("numpy", numpy.ndarray)
def _numpy_asarray(array, dtype=None, order=None):
    """Wrapper for numpy.asarray"""
    if (
        isinstance(array, numpy.ndarray)
        and order is None
        and (dtype is None or (dtype == array.dtype and array.dtype != "O"))
    ):
        return array
    ret = numpy.asarray(array, dtype=dtype, order=order)
    if ret.dtype == "O":
        raise DispatchError("Dispatch does not support numpy object arrays.")
    return ret


@Dispatch.register_repr("numpy")
def _numpy_repr(array, prefix="", suffix=""):
    """Wrapper for showing Numpy array in custom class"""
    max_line_width = numpy.get_printoptions()["linewidth"]
    array_str = numpy.array2string(
        array,
        separator=", ",
        prefix=prefix,
        suffix="," if suffix else None,
        max_line_width=max_line_width,
    )
    sep = ""
    if len(suffix) > 1:
        last_line_width = len(array_str) - array_str.rfind("\n") + 1
        if last_line_width + len(suffix) + 1 > max_line_width:
            sep = ",\n" + " " * len(prefix)
        else:
            sep = ", "
    return prefix + array_str + sep + suffix


@Dispatch.register_array_ufunc("numpy")
def _numpy_array_ufunc(ufunc, method):
    """Trival wrapper for numpy.ufunc"""
    return getattr(ufunc, method)


@Dispatch.register_array_function("numpy")
def _numpy_array_function(func):
    """Trival wrapper for numpy array function"""
    return func

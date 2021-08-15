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

"""Utilities for type handling/conversion, primarily dealing with
reshaping arrays, and handling qiskit types that wrap arrays.
"""

from typing import Union, List

import numpy as np

from qiskit.quantum_info.operators import Operator

from qiskit_dynamics.dispatch import Array


class StateTypeConverter:
    """Contains descriptions of two type specifications for DE solvers/methods,
    with functions for converting states and rhs functions between
    representations.

    A type specification is a `dict` describing a specific expected type,
    e.g. an array of a given
    shape. Currently only handled types `Array`s, specified via:
        - {'type': 'array', 'shape': tuple}

    While this class stores exact type specifications, it can be
    instantiated with a concrete type and a more general type.
    This facilitates the situation in which a solver requires a 1d `Array`,
    which is specified by the type:
        - {'type': 'array', 'ndim': 1}
    """

    def __init__(self, inner_type_spec, outer_type_spec=None, order="F"):
        """Instantiate with the inner and return types for the state.

        Args:
            inner_type_spec (dict): inner type
            outer_type_spec (dict): outer type
            order (str): order argument to be used in array reshaping.
                        Defaults to `'F'`, which corresponds to column
                        stacking convention when switching between 2d and 1d
                        arrays.
        """

        self.inner_type_spec = inner_type_spec
        self.outer_type_spec = self.inner_type_spec if outer_type_spec is None else outer_type_spec
        self.order = order

    @classmethod
    def from_instances(cls, inner_y, outer_y=None, order="F"):
        """Instantiate from concrete instances. Type of instances must
        be supported by `type_spec_from_instance`. If `outer_y is None` the
        outer type is set to the inner type.

        Args:
            inner_y (array): concrete representative of inner type
            outer_y (array): concrete representative of outer type
            order (str): order argument to be used in array reshaping.

        Returns:
            StateTypeConverter: type converter as specified by args
        """
        inner_type_spec = type_spec_from_instance(inner_y)

        outer_type_spec = None
        if outer_y is not None:
            outer_type_spec = type_spec_from_instance(outer_y)

        return cls(inner_type_spec, outer_type_spec, order)

    @classmethod
    def from_outer_instance_inner_type_spec(cls, outer_y, inner_type_spec=None, order="F"):
        """Instantiate from concrete instance of the outer type,
        and an inner type-spec. The inner type spec can be either
        be fully specified, or be more general (i.e. to
        facilitate the situation in which a solver needs a 1d array).

        Accepted general data types:
            - {'type': 'array'}
            - {'type': 'array', 'ndim': 1}

        Args:
            outer_y (array): concrete outer data type
            inner_type_spec (dict): inner, potentially general, type spec
            order (str): order argument to be used in array reshaping.

        Returns:
            StateTypeConverter: type converter as specified by args

        Raises:
            Exception: if inner_type_spec is not properly specified or is
            not a handled type
        """

        # if no inner_type_spec given just instantiate both inner
        # and outer to the outer_y
        if inner_type_spec is None:
            return cls.from_instances(outer_y, order=order)

        inner_type = inner_type_spec.get("type")
        if inner_type is None:
            raise Exception("inner_type_spec needs a 'type' key.")

        if inner_type == "array":
            outer_y_as_array = Array(outer_y)

            # if a specific shape is given attempt to instantiate from a
            # reshaped outer_y
            shape = inner_type_spec.get("shape")
            if shape is not None:
                return cls.from_instances(
                    outer_y_as_array.reshape(shape, order=order), outer_y, order=order
                )

            # handle the case that ndim == 1 is given
            ndim = inner_type_spec.get("ndim")
            if ndim == 1:
                return cls.from_instances(
                    outer_y_as_array.flatten(order=order), outer_y, order=order
                )

            # if neither shape nor ndim is given, assume it can be an array
            # of any shape
            return cls.from_instances(outer_y_as_array, outer_y, order=order)

        raise Exception("inner_type_spec not a handled type.")

    def inner_to_outer(self, y):
        """Convert a state of inner type to one of outer type."""
        return convert_state(y, self.outer_type_spec, self.order)

    def outer_to_inner(self, y):
        """Convert a state of outer type to one of inner type."""
        return convert_state(y, self.inner_type_spec, self.order)

    def rhs_outer_to_inner(self, rhs):
        """Convert an rhs function f(t, y) to work on inner type."""

        # if inner and outer type specs are the same do nothing
        if self.inner_type_spec == self.outer_type_spec:
            return rhs

        def new_rhs(t, y):
            outer_y = self.inner_to_outer(y)
            rhs_val = rhs(t, outer_y)
            return self.outer_to_inner(rhs_val)

        return new_rhs

    def generator_outer_to_inner(self, generator):
        """Convert generator from outer to inner type."""
        # if inner and outer type specs are the same do nothing
        if self.inner_type_spec == self.outer_type_spec:
            return generator

        # raise exceptions based on assumptions
        if (self.inner_type_spec["type"] != "array") or (self.outer_type_spec["type"] != "array"):
            raise Exception(
                """RHS generator transformation only
                               valid for state types Array."""
            )
        if len(self.inner_type_spec["shape"]) != 1:
            raise Exception(
                """RHS generator transformation only valid
                               if inner_type is 1d."""
            )

        if self.order == "C":
            # create identity of size the second dimension of the
            # outer type
            ident = Array(np.eye(self.outer_type_spec["shape"][1]))

            def new_generator(t):
                return Array(np.kron(generator(t), ident))

            return new_generator
        elif self.order == "F":
            # create identity of size the second dimension of the
            # outer type
            ident = Array(np.eye(self.outer_type_spec["shape"][1]))

            def new_generator(t):
                return Array(np.kron(ident, generator(t)))

            return new_generator
        else:
            raise Exception("""Unsupported order for generator conversion.""")


def convert_state(y: Array, type_spec: dict, order="F"):
    """Convert the de state y into the type specified by type_spec.
    Accepted values of type_spec are given at the beginning of the file.

    Args:
        y: the state to convert.
        type_spec (dict): the type description to convert to.
        order (str): order argument for any array reshaping function.

    Returns:
        Array: converted state.
    """

    new_y = None

    if type_spec["type"] == "array":
        # default array data type to complex
        new_y = Array(y, dtype=type_spec.get("dtype", "complex"))

        shape = type_spec.get("shape")
        if shape is not None:
            new_y = new_y.reshape(shape, order=order)

    return new_y


def type_spec_from_instance(y):
    """Determine type spec from an instance."""
    type_spec = {}
    if isinstance(y, (Array, np.ndarray)):
        type_spec["type"] = "array"
        type_spec["shape"] = y.shape

    return type_spec


def vec_commutator(A: Array):
    r"""Linear algebraic vectorization of the linear map X -> [A, X]
    in column-stacking convention. In column-stacking convention we have

    .. math::
        vec(ABC) = C^T \otimes A vec(B),

    so for the commutator we have

    .. math::
        [A, \cdot] = A \cdot - \cdot A \mapsto id \otimes A - A^T \otimes id

    Note: this function is also "vectorized" in the programming sense.

    Args:
        A: Either a 2d array representing the matrix A described above,
           or a 3d array representing a list of matrices.

    Returns:
        Array: vectorized version of the map.
    """
    iden = Array(np.eye(A.shape[-1]))
    axes = list(range(A.ndim))
    axes[-1] = axes[-2]
    axes[-2] += 1
    return np.kron(iden, A) - np.kron(A.transpose(axes), iden)


def vec_dissipator(L: Array):
    r"""Linear algebraic vectorization of the linear map
    X -> L X L^\dagger - 0.5 * (L^\dagger L X + X L^\dagger L)
    in column stacking convention.

    This gives

    .. math::
        \overline{L} \otimes L - 0.5(id \otimes L^\dagger L +
            (L^\dagger L)^T \otimes id)

    Note: this function is also "vectorized" in the programming sense.
    """
    iden = Array(np.eye(L.shape[-1]))
    axes = list(range(L.ndim))

    axes[-1] = axes[-2]
    axes[-2] += 1
    Lconj = L.conj()
    LdagL = Lconj.transpose(axes) @ L
    LdagLtrans = LdagL.transpose(axes)

    return np.kron(Lconj, iden) @ np.kron(iden, L) - 0.5 * (
        np.kron(iden, LdagL) + np.kron(LdagLtrans, iden)
    )


def to_array(op: Union[Operator, Array, List[Operator], List[Array]]):
    """Convert an operator or list of operators to an Array.

    Args:
        op: Either an Operator to be converted to an array, a list of Operators
            to be converted to a 3d array, or an array (which simply gets
            returned)
    Returns:
        Array: Array version of input
    """
    if op is None or isinstance(op, Array):
        return op

    if isinstance(op, list) and isinstance(op[0], Operator):
        shape = op[0].data.shape
        dtype = op[0].data.dtype
        arr = np.empty((len(op), *shape), dtype=dtype)
        for i, sub_op in enumerate(op):
            arr[i] = sub_op.data
        return Array(arr)

    if isinstance(op, Operator):
        return Array(op.data)

    return Array(op)


def is_scalar(val) -> bool:
    """Returns True if argument is native scalar type (complex, float, int), or False otherwise."""
    return val is complex or val is float or val is int

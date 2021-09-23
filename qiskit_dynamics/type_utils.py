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
from collections.abc import Iterable
import numpy as np
from scipy.sparse import issparse, spmatrix
from scipy.sparse import kron as sparse_kron
from scipy.sparse import identity as sparse_identity
from scipy.sparse.csr import csr_matrix

from qiskit.quantum_info.operators import Operator
from qiskit_dynamics import dispatch
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
           or a 3d array representing a list of matrices, or a 1d array
           of dtype object containing 2d sparse matrices.

    Returns:
        Array: vectorized version of the map.
    """

    if issparse(A):
        # single, sparse matrix
        sp_iden = sparse_identity(A.shape[-1], format="csr")
        return sparse_kron(sp_iden, A) - sparse_kron(A.T, sp_iden)
    if A.dtype == "object":
        # taken to be 1d array of 2d sparse matrices
        sp_iden = sparse_identity(A[0].shape[-1], format="csr")
        out = np.empty_like(A)
        for i, mat in enumerate(A):
            out[i] = sparse_kron(sp_iden, mat) - sparse_kron(mat.T, sp_iden)
        return out

    A = to_array(A)
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

    if L.dtype == "object":
        # taken to be 1d array of 2d sparse matrices
        sp_iden = sparse_identity(L[0].shape[-1], format="csr")
        out = np.empty_like(L)
        for i, mat in enumerate(L):
            out[i] = sparse_kron(mat.conj(), mat) - 0.5 * (
                sparse_kron(sp_iden, mat.conj().T * mat) + sparse_kron(mat.T * mat.conj(), sp_iden)
            )
        return out

    iden = Array(np.eye(L.shape[-1]))
    axes = list(range(L.ndim))

    L = to_array(L)
    axes[-1] = axes[-2]
    axes[-2] += 1
    Lconj = L.conj()
    LdagL = Lconj.transpose(axes) @ L
    LdagLtrans = LdagL.transpose(axes)

    return np.kron(Lconj, iden) @ np.kron(iden, L) - 0.5 * (
        np.kron(iden, LdagL) + np.kron(LdagLtrans, iden)
    )


def isinstance_qutip_qobj(obj):
    """Check if the object is a qutip Qobj.

    Bool: True if obj is qutip Qobj
    """
    if (
        type(obj).__name__ == "Qobj"
        and hasattr(obj, "_data")
        and type(obj._data).__name__ == "fast_csr_matrix"
    ):
        return True
    return False


def to_array(op: Union[Operator, Array, List[Operator], List[Array], spmatrix], no_iter=False):
    """Convert an operator or list of operators to an Array.
    Args:
        op: Either an Operator to be converted to an array, a list of Operators
            to be converted to a 3d array, or an array (which simply gets
            returned)
        no_iter (Bool): Boolean determining whether to look inside iterables to make multiple
            Arrays. If recurring, this should be True to avoid making each element of the
            input array into a separate Array.
    Returns:
        Array: Array version of input
    """
    if op is None:
        return op

    if isinstance(op, np.ndarray):
        if dispatch.default_backend() in [None, "numpy"]:
            return op
        else:
            return Array(op)

    if isinstance(op, Array):
        return op

    if issparse(op):
        return Array(op.toarray())

    if isinstance(op, Iterable) and not no_iter:
        op = Array([to_array(sub_op, no_iter=True) for sub_op in op])
    elif isinstance(op, Iterable) and no_iter:
        return op
    else:
        op = Array(op)

    if op.backend == "numpy":
        return op.data
    else:
        return op


def to_csr(
    op: Union[Operator, Array, List[Operator], List[Array], spmatrix], no_iter=False
) -> csr_matrix:
    """Convert an operator or list of operators to a sparse matrix.
    Args:
        op: Either an Operator to be converted to an sparse matrix, a list of Operators
            to be converted to a 3d sparse matrix, or a sparse matrix (which simply gets
            returned)
        no_iter (Bool): Boolean determining whether to look inside iterables to make multiple
            csr_matrices. If recurring, this should be True to avoid making each element of
            the input array a separate sparse matrix
    Returns:
        csr_matrix: Sparse matrix version of input
    """
    if op is None:
        return op

    if isinstance(op, csr_matrix):
        return op
    if isinstance_qutip_qobj(op):
        return op.data
    if isinstance(op, (Array, np.ndarray)) and op.ndim < 3:
        return csr_matrix(op)

    if isinstance(op, Iterable) and not no_iter:
        return [to_csr(item, no_iter=True) for item in op]
    else:
        return csr_matrix(op)


def to_numeric_matrix_type(
    op: Union[Operator, Array, spmatrix, List[Operator], List[Array], List[spmatrix]]
):
    """Given an operator, array, sparse matrix, or a list of operators, arrays, or sparse matrices,
    attempts to leave them in their original form, only converting the operator to an array,
    and converting lists as necessary. Summarized below:
    - operator is converted to array
    - spmatrix and Array are unchanged
    - lists of Arrays and sparse matrices are passed to their respective to_ functions
    - anything else is passed to to_array
    Args:
        op: An operator, array, sparse matrix, or list of operators, arrays or sparse matrices.
    Returns:
        Array: Array version of input
        csr_matrix: Sparse matrix version of input
    """

    if op is None:
        return op

    elif isinstance_qutip_qobj(op):
        return to_csr(op.data)

    elif isinstance(op, Array):
        return op
    elif isinstance(op, spmatrix):
        return op
    elif isinstance(op, Operator):
        return to_array(op)

    elif isinstance(op, Iterable) and isinstance(op[0], spmatrix):
        return to_csr(op)

    elif isinstance(op, Iterable) and isinstance_qutip_qobj(op[0]):
        return to_csr(op)

    else:
        return to_array(op)

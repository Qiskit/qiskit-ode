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
"""Monkey patch qiskit.quantum_info to work easier with Array."""

from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    Operator,
    SuperOp,
    Choi,
    PTM,
    Chi,
    Pauli,
    Clifford,
    SparsePauliOp,
)
from .array import Array

__all__ = []


def __qiskit_array__(self, dtype=None, backend=None):
    """Convert qi operator to an Array"""
    return Array(self.data, dtype=dtype, backend=backend)


def __to_matrix_array__(self, dtype=None, backend=None):
    """Convert object to array through to_matrix method"""
    return Array(self.to_matrix(), dtype=dtype, backend=backend)


# Monkey patch quantum info operators

Statevector.__qiskit_array__ = __qiskit_array__
DensityMatrix.__qiskit_array__ = __qiskit_array__
Operator.__qiskit_array__ = __qiskit_array__
SuperOp.__qiskit_array__ = __qiskit_array__
Choi.__qiskit_array__ = __qiskit_array__
Chi.__qiskit_array__ = __qiskit_array__
PTM.__qiskit_array__ = __qiskit_array__
Pauli.__qiskit_array__ = __to_matrix_array__
Clifford.__qiskit_array__ = __to_matrix_array__
SparsePauliOp.__qiskit_array__ = __to_matrix_array__

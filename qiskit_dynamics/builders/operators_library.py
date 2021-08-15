# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,no-member,attribute-defined-outside-init
from .dynamical_operators import DynamicalOperator, Id, Zero
from typing import Any
import numpy as np


"""
These are the predefined operators available for creating dynamical system simulations.
Currently implemented are spin operators and truncated harmonic oscillator operators,
in addition to some more general operators, all realizing numpy matrices by default.
The classes ``Id`` and ``Zero`` are implemented separately, in the file dynamical_operators.py
together with the base class ``DynamicalOperator``.
Currently supported:
Identity, Zero, and a standard basis Projector in any dimension.
Spin-1/2 (qubit) operators: x, y, z, sigma^(+/-) and six projectors on the state with +/- eigenvalues
	along the axes.
Oscillator canonical operators, ladder operators, number and number^2 operators, in any dimension.
"""

# TODO: Generalize the spin operators up to a low dimension (say 4 or 5).


class Projector(DynamicalOperator):
	"""A dynamical operator that builds a numpy projector matrix in the standard basis."""
	def __init__(self, system_id = '', row = 0, col = 0):
		self._row = row
		self._col = col
		super().__init__(system_id, 'proj' + str(row) + '_' + str(col))

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		result = np.zeros((dim, dim), complex)
		row = self._row; col = self._col
		if row < dim and col < dim:
			result[row, col] = 1
			return result
		raise Exception(f"A projector with row = {row} and column = {col } "
			f"is incompatible with matrix generation of dimension {dim}.")


class Sx(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli x matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'x')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'x':
			return np.asarray([[0, 1], [1, 0]], complex)
		super().get_operator_matrix(dim)


class Sy(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli y matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'y')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'y':
			return np.asarray([[0, -1j], [1j, 0]], complex)
		super().get_operator_matrix(dim)


class Sz(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli z matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'z')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'z':
			return np.asarray([[1, 0], [0, -1]], complex)
		super().get_operator_matrix(dim)


class Sp(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli ladder |1><0| matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'sp')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'sp':
			return np.asarray([[0, 1], [0, 0]], complex)
		super().get_operator_matrix(dim)


class Sm(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli ladder |0><1| matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'sm')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'sm':
			return np.asarray([[0, 0], [1, 0]], complex)
		super().get_operator_matrix(dim)


class PlusZ(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for Up (|0><0|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '0')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '0':
			return np.asarray([[1, 0], [0, 0]], complex)
		super().get_operator_matrix(dim)


class MinusZ(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for Down (|1><1|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '1')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '1':
			return np.asarray([[0, 0], [0, 1]], complex)
		super().get_operator_matrix(dim)


class PlusX(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for plus state (|+><+|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '+')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '+':
			return np.asarray([[0.5, 0.5], [0.5, 0.5]], complex)
		super().get_operator_matrix(dim)


class MinusX(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for minus state (|-><-|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '-')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '-':
			return np.asarray([[0.5, -0.5], [-0.5, 0.5]], complex)
		super().get_operator_matrix(dim)


class PlusY(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for right y (|i><i|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'r')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'r':
			return np.asarray([[0.5, -0.5j], [0.5j, 0.5]], complex)
		super().get_operator_matrix(dim)


class MinusY(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for left y (|-i><-i|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'l')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'l':
			return np.asarray([[0.5, 0.5j], [-0.5j, 0.5]], complex)
		super().get_operator_matrix(dim)


class On(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for an oscillator number operator."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'n')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'n':
			return np.diag(np.asarray(range(dim), complex))
		super().get_operator_matrix(dim)


class On2(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for an oscillator n^2 operator."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'n2')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'n2':
			return np.diag(np.asarray(range(dim), complex) ** 2)
		super().get_operator_matrix(dim)


class Oa(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for an oscillator number operator."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'a')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'a':
			return np.diag(np.asarray(range(1, dim), complex) ** .5, 1)
		super().get_operator_matrix(dim)


class Oa_(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for an oscillator number operator."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'a')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'a':
			return np.diag(np.asarray(range(1, dim), complex) ** .5, -1)
		super().get_operator_matrix(dim)


class Oq(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for an oscillator number operator."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'q')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'q':
			return (.5 ** .5) * (np.diag(np.asarray(range(1, dim), complex) ** .5, 1)
					+ np.diag(np.asarray(range(1, dim), complex) ** .5, -1))
		super().get_operator_matrix(dim)


class Op(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for an oscillator number operator."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'p')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'p':
			return -1j * (.5 ** .5) * (np.diag(np.asarray(range(1, dim), complex) ** .5, 1)
					- np.diag(np.asarray(range(1, dim), complex) ** .5, -1))
		super().get_operator_matrix(dim)


# class SmSp_SpSm(DynamicalOperator):
# 	def __init__(self, system_id = ''):
# 		super().__init__(system_id, 'smspspsm')
#
# 	def get_operator_matrix(self, dim: int) -> Any:
# 		"""Returns a matrix describing a realization of the operator specified in the parameters.
#
# 		Args:
# 			dim: The physical dimension of the matrix to generate.
# 		"""
# 		if dim == 2 and self.s_type == 'smspspsm':
# 			smsp = np.kron(np.asarray([[0, 0], [1, 0]], complex), np.asarray([[0, 1], [0, 0]], complex))
# 			return smsp + np.transpose(smsp)
# 		raise Exception(
# 			f"Operator type {self.s_type} unknown or unsupported for matrix generation with dimension {dim}.")

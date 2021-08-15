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
from typing import OrderedDict, Dict, List, Union, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from ..type_utils import is_scalar


class DynamicalOperator(ABC):
	"""A class for operators used in defining dynamical simulations."""

	def __init__(self, system_id: Any = '', s_type = '', matrix: Optional[Any] = None):
		""" Initialization of an operator using (optional) id of the subsystem, type, and a matrix.

		Args:
			system_id: A unique identifier of the subsystem (degree of freedom) of the operator,
				or an empty string to defer such an identification.
			s_type: A string name of the type of operator. If not given, and the matrix argument
				is not None, then a unique id is automatically generated from the matrix's
				``__hash__()`` or ``__str()__`` methods.
			matrix: An explicit matrix realization of the operator, for use when building
				matrices using ``OperatorBuilder``.

		Raises:
			An exception if the matrix argument is not None, s_type is an empty string, but
				the matrix does not implement either __hash__() or __str__().
		"""
		self._matrix = matrix
		self.system_id = system_id
		if matrix is not None and s_type == '':
			if matrix.__hash__ is not None:
				s_type = str(matrix.__hash__())
			elif matrix.__str__ is not None:
				s_type = str(matrix.__str__().__hash__())
				# TODO: Is using __str__() robust enough? It works with numpy. If this is not a good solution,
				# the user must supply a type string.
			else:
				raise Exception("A unique type string for the argument matrix could not be generated.")
		s_type = s_type.lower()
		self._s_type = s_type
		self.compound_type = ''
		self.compound_ops = None

	@property
	def s_type(self) -> str:
		"""A string defining the operator type. Must be unique to identify the type."""
		return self._s_type

	def __add__(self, other):
		"""Addition of two DynamicalOperators. Returns a new (compound) DynamicalOperator."""
		if not isinstance(other, DynamicalOperator):
			raise Exception("Both operands in an addition must be instances of a DynamicalOperator.")
		result = self.new_operator()
		result.compound_type = '+'
		result.compound_ops = [self, other]
		return result

	# def __pow__(self, power, modulo=None):
	# 	if type(power) is not int:
	# 		raise Exception("Only integer (positive?) powers are currently supported for operators.")
	# 	result = self.new_operator()
	# 	result.compound_type = '**'
	# 	result.compound_ops = [self, power]
	# 	return result

	def __sub__(self, other):
		"""Subtraction of two DynamicalOperators. Returns a new (compound) DynamicalOperator."""
		return self.__add__(-other)

	def __mul__(self, other):
		"""Multiplication by a DynamicalOperator or a scalar."""
		result = self.new_operator()
		if isinstance(other, DynamicalOperator):
			result.compound_type = '@'  # Indicates operator * operator for OperatorBuilder
			result.compound_ops = [self, other]
			# For a product of two operators, their order must be preserved
			return result
		else:
			other_type = type(other)
			if is_scalar(other_type):
				result.compound_type = '*'  # Indicates operator * scalar for OperatorBuilder
				result.compound_ops = [self, other]
				# For a product of an operator and a scalar, we can put the operator first always.
				# This is used to simplify OperatorBuilder code below, and must not be changed.
				return result
		raise Exception("The second operand of a multiplication must be a DynamicalOperator class or a scalar.")

	def __rmul__(self, other):
		"""Multiplication of a DynamicalOperator by a scalar."""
		result = self.__mul__(other)
		return result

	def __neg__(self):
		"""Unary negation of a DynamicalOperator."""
		result = self.__rmul__(-1.)
		return result

	def __pos__(self):
		"""Unary plus operator prepending a DynamicalOperator."""
		return self

	def new_operator(self):
		"""A method that must be implemented by subclasses, to return the correct instance subclass.

		Called from operators to create new compound operators in the tree of expressions.
		"""
		return DynamicalOperator()

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		This method must be overridden by subclasses in order to support building to a matrix.
		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		raise Exception(
			f"Operator type {self.s_type} unknown or unsupported for matrix generation with dimension {dim}.")

	def kron_two_matrices(self, left_matrix: Any, right_matrix: Any):
		"""Returns the matrix Kronecker product of the two arguments.

		This function is not declared as static in order to allow subclasses to override the
		default implementation. However, fields of the ``self`` object are not being used.
		Args:
			left_matrix: First matrix.
			right_matrix: Second matrix.

		Returns:
			The Kronecker product of the arguments.
		"""
		return np.kron(left_matrix, right_matrix)  # TODO verify whether ordering matters

	def build_one_dict(self, operators_repo: dict,
						prune_subsystems: Optional[dict] = None) -> dict:
		"""Recursively build a flat dictionary out of a (sub-)tree of DynamicalOperators.

		Args:
			operators_repo: A dictionary referencing the DynamicalOperators used in the building.
				It is being updated by this method.
			prune_subsystems: An optional dict specifying subsystem_dims to remove, as the keys of
				entries in the dict. The values are not used.
		Returns:
			The structure of the returned flat dict is as follows: Each key identifies uniquely an
			operator that is a product of operators, e.g. "X_0 * Z_0 * Y_2" is the unique operator
			that is the ordered product of 3 operators, X on subsystem 0, Z on subsystem 0, and Y on
			subsystem 2. The value is a multiplicative scalar coefficient for this operator.
			The different entries of the dictionary are to be summed over.
		Raises:
			An exception if an unidentified operation was found in the tree.
		"""
		if self.compound_type == '+':  # The sub-tree root is a sum of two operators
			result = {}
			for op in self.compound_ops:
				# Build a dict out of each summand.
				op_dict: dict = op.build_one_dict(operators_repo, prune_subsystems)
				# We now iterate over all members in the flattened dict, and add them to the result
				# dict - if the unique key already appears there, the scalars are added.
				for key, val in op_dict.items():
					val_sum = val + result.get(key, complex(0.))
					result[key] = val_sum
		elif self.compound_type == '@':  # The sub-tree root is a product of two operators
			new_key = []
			new_val = complex(1.)
			for op in self.compound_ops:
				op_dict = op.build_one_dict(operators_repo, prune_subsystems)
				# Note that operators_repo can be extended with operators that appear in products
				# that will be later pruned, and hence operators_repo does not represent only
				# operators that actually appear in the final built dictionary, in the case of pruning.
				if len(op_dict) == 0:
					# If one of the product terms is an empty dictionary (as result of pruning),
					# we remove completely the product as well.
					new_val = complex(0.)
					break
				for key, val in op_dict.items():
					if key is not None:
						for key_element in key:
							new_key.append(key_element)
							# The key of the product operator will be a concatenation of unique keys,
							# order preserved.
					new_val *= val  # The scalar factor will be a product of the scalars.
			if new_val != complex(0.):
				result = {tuple(new_key): new_val}
			else:  # Return an empty dictionary, when one of the product terms is pruned.
				result = dict()
		elif self.compound_type == '*':  # The sub-tree root is a product of operator * scalar
			# Since this product is commutative, the operator is always first in order,
			# as implemented in DynamicalOperator.__mul__ and  DynamicalOperator.__rmul__
			op = self.compound_ops[0]
			scalar = self.compound_ops[1]
			op_dict = op.build_one_dict(operators_repo, prune_subsystems)
			for key, val in op_dict.items():
				op_dict[key] = val * scalar
			result = op_dict
		elif self.compound_type == '':
			val = complex(1.)
			if prune_subsystems is not None and self.system_id in prune_subsystems:
				result = dict()
			else:
				operators_repo[DynamicalOperatorKey(self)] = self
				# TODO Note that there's no way to verify uniqueness of the implementing operator
				result = {tuple([DynamicalOperatorKey(self)]): val}
		else:
			raise Exception(f"Unknown/unsupported composite operator {self.compound_type}.")
		return result


class DynamicalOperatorKey:
	"""A container for a unique key identifying an operator and a subsystem."""

	def __init__(self, op: DynamicalOperator):
		self.system_id = op.system_id
		self.s_type = op.s_type

	def __hash__(self):
		return hash((self.system_id, self.s_type))

	def __eq__(self, other):
		return isinstance(other, DynamicalOperatorKey) and\
			self.system_id == other.system_id and self.s_type == other.s_type

	def __str__(self):
		return f'({self.system_id}, {self.s_type})'


class Id(DynamicalOperator):
	"""A dynamical operator that builds a numpy identity matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'i')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'i':
			return np.identity(dim, complex)
		super().get_operator_matrix(dim)


class Zero(DynamicalOperator):
	"""A dynamical operator that builds a numpy null (zero) matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'null')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if self.s_type == 'null':
			return np.zeros((dim, dim), complex)
		super().get_operator_matrix(dim)


def build_dictionaries(operators: Union[DynamicalOperator, List[DynamicalOperator]],
					   prune_subsystems: Optional[dict] = None)\
		-> (Union[dict, List[dict]], dict):
	"""Builds a list of flat descriptive dictionaries from a list of DynamicalOperator trees.

	Args:
		operators: A DynamicalOperator or a list of DynamicalOperators, for each one the return
			value will contain a flattened descriptive dict.
		prune_subsystems: An optional dict specifying subsystem_dims to remove, as the keys of
			entries in the dict. The values are not used.

	Returns:
		A tuple with the first entry being a dictionary or a list of dictionaries (matching the
			operators parameter), and the second entry being an ``operators_repo``: a dictionary
			referencing the DynamicalOperators used in the building. This operators_repo is
			necessary for building into matrices.
		The structure of each resulting flat dict is as follows: Each key identifies uniquely an
		operator that is a product of operators, e.g. "X_0 * Z_0 * Y_2" is the unique operator
		that is the ordered product of 3 operators, X on subsystem 0, Z on subsystem 0, and Y on
		subsystem 2. The value is a multiplicative scalar coefficient for this operator.
		The different entries of the dictionary are to be summed over.
	"""
	results = []
	b_flatten = False  # If operators is one instance return a dict, otherwise a list of dicts
	if type(operators) != list:
		b_flatten = True
		operators = [operators]
	operators_repo = dict()
	for op in operators:
		results.append(op.build_one_dict(operators_repo, prune_subsystems))
	if b_flatten:
		results = results[0]
	return results, operators_repo


def build_matrices(operators: Union[DynamicalOperator, Dict, List[DynamicalOperator], List[Dict]],
				   subsystem_dims: OrderedDict,
				   prune_subsystems: Optional[dict] = None,
				   operators_repo: Optional[Dict] = None,
				   null_matrix_op: Optional[DynamicalOperator] = None,
				   id_matrix_op: Optional[DynamicalOperator] = None) -> Any:
	"""Build a (possibly list) of matrices from DynamicalOperator or dictionaries thereof.

	Args:
		operators: A DynamicalOperator, a list of DynamicalOperators, a flattened dictionary
			previously built using ``build_dictionaries``, or a list of such dictionaries.
		subsystem_dims: An ordered dictionary for each subsystem (identified using the system_id
			field of the DynamicalOperator), indicating the matrix dimension to assign for
			it. Subsystems which are to be removed from the matrix building, must be specified
			in the ``prune_subsystems`` parameter, if relevant.
		prune_subsystems: An optional dict specifying subsystem_dims to remove, as the keys of
			entries in the dict. The values are not used. This parameter can only be used if
			the ``operators`` parameter corresponds to DynamicalOperators.
		operators_repo: A dictionary referencing the DynamicalOperators used in the building.
		null_matrix_op: An optional DynamicalOperator instance for building an null (zeros) matrix.
			The default building implementation returns numpy matrices. The corresponding instance
			is also being used to invoke ``kron_two_matrices()`` to implement a kronecker product
			of two subsystem matrices.
		id_matrix_op: An optional DynamicalOperator instance for building an Identity matrix.
			The default building implementation returns numpy matrices.
	Returns:
		A matrix or a list of matrices, of the type as returned by
		DynamicOperator.get_operator_matrix() or the subclass instances passed in the
		``operators`` parameter.
	Raises:
		An exception if an unidentified operation was found in the tree.
	"""
	if len(subsystem_dims) == 0:
		return None
	b_flatten = False
	if type(operators) != list:
		b_flatten = True
		operators = [operators]
	dims = subsystem_dims.values()
	if null_matrix_op is None:
		null_matrix_op = Zero()
	if id_matrix_op is None:
		id_matrix_op = Id()
	total_dim = 1
	for dim in dims:
		if dim > 0:
			total_dim *= dim
	if len(operators) == 0:
		return null_matrix_op.get_operator_matrix(total_dim)
	b_dictionaries = False
	for op in operators:
		# Verify operator types are known and identical
		op_type = type(op)
		if op_type is dict:
			b_dictionaries = True
		elif not isinstance(op, DynamicalOperator):
			raise Exception(f"Unsupported class type in parameter operators: {op_type}.")
		elif b_dictionaries:
			raise Exception("All operators must be of the same type (a dictionary or a DynamicalOperator).")

	sys_ids = []
	for sys_id in subsystem_dims.keys():
		sys_ids.append(sys_id)
	identity_matrices = []
	if b_dictionaries:
		operators_dict: List[Dict] = operators
	else:
		operators_dict, operators_repo = build_dictionaries(operators, prune_subsystems)
	results = []
	for dim in dims:
		identity_matrices.append(id_matrix_op.get_operator_matrix(dim))
	for op_dict in operators_dict:
		results.append(_build_one_matrix(op_dict, operators_repo, null_matrix_op,
										 total_dim, subsystem_dims, sys_ids, identity_matrices))
	if b_flatten:
		results = results[0]
	return results


def _build_one_matrix(operator_dict: Dict, operators_repo: Dict,
					  null_matrix_op, total_dim, subsystem_dims, sys_ids, identity_matrices) -> np.ndarray:
	matrix = null_matrix_op.get_operator_matrix(total_dim)
	for key, val in operator_dict.items():
		sub_matrices = {}
		for key_element in key:
			operator_key: DynamicalOperatorKey = key_element
			dim = subsystem_dims.get(operator_key.system_id, None)
			if dim is None:
				raise Exception(
					f"An operator was defined with id = {operator_key.system_id}, "
					"but this id does not appear in the subsystem_dims parameter.")
			dyn_op: DynamicalOperator = operators_repo[operator_key]
			new_sub_matrix = dyn_op.get_operator_matrix(dim)
			sub_matrix = sub_matrices.get(operator_key.system_id, None)
			if sub_matrix is not None:
				new_sub_matrix = sub_matrix @ new_sub_matrix  # note that order of matrix product matters
			sub_matrices[operator_key.system_id] = new_sub_matrix
		op_matrix = None
		n_subsystems = len(sys_ids)
		for i in range(n_subsystems):
			sub_matrix = sub_matrices.get(sys_ids[i], identity_matrices[i])
			if i == 0:
				op_matrix = sub_matrix
			else:
				op_matrix = null_matrix_op.kron_two_matrices(op_matrix, sub_matrix)
		matrix += val * op_matrix
	return matrix

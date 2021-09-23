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
# pylint: disable=arguments-differ,signature-differs

"""Operator collections as math/calculation objects for Model classes"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from copy import deepcopy
import numpy as np
from qiskit.quantum_info.operators.operator import Operator
from scipy.sparse.csr import csr_matrix
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.type_utils import to_array, to_csr, vec_commutator, vec_dissipator


class BaseOperatorCollection(ABC):
    r"""Abstract class representing a two-variable matrix function.

    This class represents a function :math:`c,y \mapsto \Lambda(c, y)`,
    which is assumed to be decomposed as
    :math:`\Lambda(c, y) = \Lambda_d(y) + \sum_jc_j\Lambda_j(y)`
    for linear maps :math:`\Lambda_d` and :math:`\Lambda_j`, with
    :math:`\Lambda_d` referred to as the "drift" component.

    Describes an interface for evaluating the map or its action on ``y``,
    given the 1d set of values :math:`c_j`.
    """

    @property
    def drift(self) -> Array:
        """Returns drift part of operator collection."""
        return self._drift

    @drift.setter
    def drift(self, new_drift: Optional[Array] = None):
        """Sets drift term."""
        self._drift = new_drift

    @property
    @abstractmethod
    def num_operators(self) -> int:
        """Returns number of operators the collection
        is storing."""
        pass

    @abstractmethod
    def evaluate(self, signal_values: Array) -> Array:
        r"""If possible, compute a matrix representation of the functino where the second
        argument is left unspecified.
        """
        pass

    @abstractmethod
    def evaluate_rhs(self, signal_values: Union[List[Array], Array], y: Array) -> Array:
        r"""Compute the function."""
        pass

    def __call__(
        self, signal_values: Union[List[Array], Array], y: Optional[Array] = None
    ) -> Array:
        """Call either ``self.evaluate`` or ``self.evaluate_rhs`` depending on number of
        arguments.
        """

        if y is None:
            return self.evaluate(signal_values)

        return self.evaluate_rhs(signal_values, y)

    def copy(self):
        """Return a copy of self."""
        return deepcopy(self)


class DenseOperatorCollection(BaseOperatorCollection):
    r"""Concrete operator collection representing a function computing left
    multiplication by an affine combination of matrices.

    This class represents a function of the form :math:`Lambda(c, y) = (G_d + \sum_j c_j G_j) y`,
    where the :math:`G_d` and :math:`G_j` are dense arrays.
    """

    def __init__(
        self,
        operators: Union[Array, List[Operator]],
        drift: Optional[Union[Array, Operator]] = None,
    ):
        """Initialize.
        Args:
            operators: (k,n,n) Array specifying the terms :math:`G_j`.
            drift: (n,n) Array specifying the extra drift :math:`G_d`.
        """
        self._operators = to_array(operators)
        self.drift = to_array(drift)

    @property
    def num_operators(self) -> int:
        return self._operators.shape[0]

    def evaluate(self, signal_values: Array) -> Array:
        r"""Evaluate the affine combination of matrices."""
        if self._drift is None:
            return np.tensordot(signal_values, self._operators, axes=1)
        else:
            return np.tensordot(signal_values, self._operators, axes=1) + self._drift

    def evaluate_rhs(self, signal_values: Array, y: Array) -> Array:
        """Evaluates the function."""
        return np.dot(self.evaluate(signal_values), y)


class SparseOperatorCollection(BaseOperatorCollection):
    r"""Sparse version of DenseOperatorCollection."""

    def __init__(
        self,
        operators: Union[Array, List[Operator]],
        drift: Optional[Union[Array, Operator]] = None,
        decimals: Optional[int] = 10,
    ):
        """
        Initialize.

        Args:
            operators: (k,n,n) Array specifying the terms :math:`G_j`.
            drift: (n,n) Array specifying the drift term :math:`G_d`.
            decimals: Values will be rounded at ``decimals`` places after decimal.
                Avoids storing excess sparse entries for entries close to zero."""
        if isinstance(drift, Operator):
            drift = to_csr(drift)
        self.drift = np.round(drift, decimals)
        self._operators = np.empty(shape=len(operators), dtype="O")
        # pylint: disable=consider-using-enumerate
        for i in range(len(operators)):
            if isinstance(operators[i], Operator):
                operators[i] = to_csr(operators[i])
            self._operators[i] = csr_matrix(np.round(operators[i], decimals))

    @property
    def num_operators(self) -> int:
        return self._operators.shape[0]

    @property
    def drift(self) -> Array:
        return super().drift

    @drift.setter
    def drift(self, new_drift):
        if isinstance(new_drift, csr_matrix):
            self._drift = new_drift
        else:
            self._drift = csr_matrix(new_drift)

    def evaluate(self, signal_values: Array) -> csr_matrix:
        r"""Sparse version of ``DenseOperatorCollection.evaluate``.

        Args:
            signal_values: Coefficients :math:`c_j`.

        Returns:
            Generator as sparse array."""
        signal_values = signal_values.reshape(1, signal_values.shape[-1])
        if self._drift is None:
            return np.tensordot(signal_values, self._operators, axes=1)[0]
        else:
            return np.tensordot(signal_values, self._operators, axes=1)[0] + self._drift

    def evaluate_rhs(self, signal_values: Array, y: Array) -> Array:
        if len(y.shape) == 2:
            # For y a matrix with y[:,i] storing the i^{th} state, it is faster to
            # first evaluate the generator in most cases
            gen = np.tensordot(signal_values, self._operators, axes=1) + self.drift
            out = gen.dot(y)
        elif len(y.shape) == 1:
            # for y a vector, it is typically faster to use the following, very
            # strange-looking implementation
            tmparr = np.empty(shape=(1), dtype="O")
            tmparr[0] = y
            out = np.dot(signal_values, self._operators * tmparr) + self.drift.dot(y)
        return out


class DenseLindbladCollection(BaseOperatorCollection):
    r"""Object for computing the right hand side of the Lindblad equation
    with dense arrays.

    In particular, this object represents the function:
        .. math::
            \Lambda(c_1, c_2, \rho) = -i[H_d + \sum_j c_{1,j}H_j,\rho]
                                      + \sum_jc_{2,j}(L_j\rho L_j^\dagger
                                        - (1/2) * {L_j^\daggerL_j,\rho})

    where :math:`\[,\]` and :math:`\{,\}` are the operator
    commutator and anticommutator, respectively.
    """

    def __init__(
        self,
        hamiltonian_operators: Union[Array, List[Operator]],
        drift: Union[Array, Operator],
        dissipator_operators: Optional[Union[Array, List[Operator]]] = None,
    ):
        r"""Initialization.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                                   as :math:`H(t) = H_d + \sum_j c_j H_j` by specifying
                                   :math:`H_j`. (k,n,n) array.
            drift: Treated as a constant term :math:`H_d` to be added to the
                   Hamiltonian of the system.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation.
                                  (m,n,n) Array.
        """

        self._hamiltonian_operators = to_array(hamiltonian_operators)
        self._dissipator_operators = to_array(dissipator_operators)
        if dissipator_operators is not None:
            self._dissipator_operators_conj = np.conjugate(
                np.transpose(to_array(dissipator_operators), [0, 2, 1])
            ).copy()
            self._dissipator_products = np.matmul(
                self._dissipator_operators_conj, self._dissipator_operators
            )
        self.drift = to_array(drift)

    @property
    def num_operators(self):
        return self._hamiltonian_operators.shape[-3], self._dissipator_operators.shape[-3]

    def evaluate(self, ham_sig_vals: Array, dis_sig_vals: Array) -> Array:
        raise ValueError("Non-vectorized Lindblad collections cannot be evaluated without a state.")

    def evaluate_hamiltonian(self, signal_values: Array) -> Array:
        r"""Compute the Hamiltonian.

        Args:
            signal_values: [Real] values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.
        Returns:
            Hamiltonian matrix.
        """
        return np.tensordot(signal_values, self._hamiltonian_operators, axes=1) + self.drift

    def evaluate_rhs(self, ham_sig_vals: Array, dis_sig_vals: Array, y: Array) -> Array:
        r"""Evaluates Lindblad equation RHS given a pair of signal values
        for the hamiltonian terms and the dissipator terms. Expresses
        the RHS of the Lindblad equation as :math:`(A+B)y + y(A-B) + C`, where
            .. math::
            A = (-1/2)*\sum_j\gamma(t) L_j^\dagger L_j,

            B = -iH,

            C = \sum_j \gamma_j(t) L_j y L_j^\dagger.

        Args:
            ham_sig_vals: hamiltonian coefficient values, :math:`s_j(t)`.
            dis_sig_vals: dissipator signal values, :math:`\gamma_j(t)`.
            y: density matrix as (n,n) Array representing the state at time :math:`t`.
        Returns:
            RHS of Lindblad equation
            .. math::
                -i[H,y] + \sum_j\gamma_j(t)(L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y}).
        """

        hamiltonian_matrix = -1j * self.evaluate_hamiltonian(ham_sig_vals)  # B matrix

        if self._dissipator_operators is not None:
            dissipators_matrix = (-1 / 2) * np.tensordot(  # A matrix
                dis_sig_vals, self._dissipator_products, axes=1
            )

            left_mult_contribution = np.matmul(hamiltonian_matrix + dissipators_matrix, y)
            right_mult_contribution = np.matmul(y, -hamiltonian_matrix + dissipators_matrix)

            if len(y.shape) == 3:
                # Must do array broadcasting and transposition to ensure vectorization works
                y = np.broadcast_to(y, (1, y.shape[0], y.shape[1], y.shape[2])).transpose(
                    [1, 0, 2, 3]
                )

            both_mult_contribution = np.tensordot(
                dis_sig_vals,
                np.matmul(
                    self._dissipator_operators, np.matmul(y, self._dissipator_operators_conj)
                ),
                axes=(-1, -3),
            )  # C

            return left_mult_contribution + right_mult_contribution + both_mult_contribution

        else:
            return np.dot(hamiltonian_matrix, y) - np.dot(y, hamiltonian_matrix)

    def __call__(self, ham_sig_vals: Array, dis_sig_vals: Array, y: Optional[Array]) -> Array:
        if y is None:
            return self.evaluate(ham_sig_vals, dis_sig_vals)

        return self.evaluate_rhs(ham_sig_vals, dis_sig_vals, y)


class DenseVectorizedLindbladCollection(DenseOperatorCollection):
    r"""Vectorized version of DenseLindbladCollection.

    This class evaluates the right hand side of the Lindblad equation in vectorized form, i.e.
    in which the state :math:`\rho`, an :math:`(n,n)` matrix, is embedded in a vector space of
    dimension :math:`n^2` using the column stacking convention."""

    def __init__(
        self,
        hamiltonian_operators: Union[Array, List[Operator]],
        drift: Union[Array, Operator],
        dissipator_operators: Optional[Union[Array, List[Operator]]] = None,
    ):
        r"""Initialize.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) Array.
            drift: Constant term to be added to the Hamiltonian of the system. (n,n) Array.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) Array.
        """

        # Convert Hamiltonian to commutator formalism
        vec_ham_ops = -1j * vec_commutator(to_array(hamiltonian_operators))
        vec_drift = -1j * vec_commutator(to_array(drift))
        total_ops = None
        if dissipator_operators is not None:
            vec_diss_ops = vec_dissipator(to_array(dissipator_operators))
            total_ops = np.append(vec_ham_ops, vec_diss_ops, axis=0)
            self.empty_dissipators = False
        else:
            total_ops = vec_ham_ops
            self.empty_dissipators = True

        super().__init__(total_ops, drift=vec_drift)

    def evaluate_rhs(self, ham_sig_vals: Array, dis_sig_vals: Array, y: Array) -> Array:
        r"""Evaluates the RHS of the Lindblad equation using
        vectorized maps.

        Args:
            ham_sig_vals: hamiltonian signal coefficients.
            dis_sig_vals: dissipator signal coefficients.
                If none involved, pass None.
            y: Density matrix represented as a vector using column-stacking
                convention.
        Returns:
            Vectorized RHS of Lindblad equation :math:`\dot{\rho}` in column-stacking
                convention."""
        return np.dot(self.evaluate(ham_sig_vals, dis_sig_vals), y)

    def evaluate(self, ham_sig_vals: Array, dis_sig_vals: Array) -> Array:
        r"""Evaluates the RHS of the Lindblad equation using
        vectorized maps.

        Args:
            ham_sig_vals: stores the Hamiltonian signal coefficients.
            dis_sig_vals: stores the dissipator signal coefficients.
        Returns:
            Vectorized generator of Lindblad equation :math:`\dot{\rho}` in column-stacking
                convention."""
        if self.empty_dissipators:
            signal_values = ham_sig_vals
        else:
            signal_values = np.append(ham_sig_vals, dis_sig_vals, axis=-1)
        return super().evaluate(signal_values)

    def evaluate_hamiltonian(self, ham_sig_vals: Array) -> Array:
        r"""Computes the Hamiltonian commutator part of the Lindblad equation,
        vectorized in column-stacking convention.

        Args:
            ham_sig_vals: [Real] values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.
        Returns:
            Vectorized commutator :math:`[H,\cdot]`"""
        if self.empty_dissipators:
            signal_values = ham_sig_vals
        else:
            zero_padding = np.zeros(self.num_operators - len(ham_sig_vals))
            signal_values = np.append(ham_sig_vals, zero_padding, axis=0)
        return 1j * super().evaluate(signal_values)


class SparseVectorizedLindbladCollection(SparseOperatorCollection):
    r"""Vectorized version of SparseLindbladCollection."""

    def __init__(
        self,
        hamiltonian_operators: Union[Array, List[Operator]],
        drift: Union[Array, Operator],
        dissipator_operators: Optional[Union[Array, List[Operator]]] = None,
    ):
        r"""Initialize.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) Array.
            drift: Constant term to be added to the Hamiltonian of the system. (n,n) Array.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) Array.
        """

        # Convert Hamiltonian to commutator formalism
        vec_ham_ops = -1j * vec_commutator(hamiltonian_operators)
        vec_drift = -1j * vec_commutator(drift)
        total_ops = None
        if dissipator_operators is not None:
            vec_diss_ops = vec_dissipator(to_array(dissipator_operators))
            total_ops = np.append(vec_ham_ops, vec_diss_ops, axis=0)
            self.empty_dissipators = False
        else:
            total_ops = vec_ham_ops
            self.empty_dissipators = True

        super().__init__(total_ops, drift=vec_drift)

    def evaluate_rhs(self, ham_sig_vals: Array, dis_sig_vals: Array, y: Array) -> Array:
        r"""Evaluates the RHS of the Lindblad equation using
        vectorized maps.

        Args:
            ham_sig_vals: hamiltonian signal coefficients.
            dis_sig_vals: dissipator signal coefficients.
                If none involved, pass None.
            y: Density matrix represented as a vector using column-stacking
                convention.
        Returns:
            Vectorized RHS of Lindblad equation :math:`\dot{\rho}` in column-stacking
                convention."""
        return self.evaluate(ham_sig_vals, dis_sig_vals) @ y

    def evaluate(self, ham_sig_vals: Array, dis_sig_vals: Array) -> Array:
        r"""Evaluates the RHS of the Lindblad equation using
        vectorized maps.

        Args:
            ham_sig_vals: stores the Hamiltonian signal coefficients.
            dis_sig_vals: stores the dissipator signal coefficients.
        Returns:
            Vectorized generator of Lindblad equation :math:`\dot{\rho}` in column-stacking
                convention."""
        if self.empty_dissipators:
            signal_values = ham_sig_vals
        else:
            signal_values = np.append(ham_sig_vals, dis_sig_vals, axis=-1)
        return super().evaluate(signal_values)

    def evaluate_hamiltonian(self, ham_sig_vals: Array) -> Array:
        r"""Computes the Hamiltonian commutator part of the Lindblad equation,
        vectorized in column-stacking convention.

        Args:
            ham_sig_vals: [Real] values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.
        Returns:
            Vectorized commutator :math:`[H,\cdot]`"""
        if self.empty_dissipators:
            signal_values = ham_sig_vals
        else:
            zero_padding = np.zeros(self.num_operators - len(ham_sig_vals))
            signal_values = np.append(ham_sig_vals, zero_padding, axis=0)
        return 1j * super().evaluate(signal_values)


class SparseLindbladCollection(DenseLindbladCollection):
    """Sparse version of DenseLindbladCollection."""

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        hamiltonian_operators: Union[Array, List[Operator]],
        drift: Union[Array, Operator],
        dissipator_operators: Optional[Union[Array, List[Operator]]] = None,
        decimals: Optional[int] = 10,
    ):
        r"""Initializes sparse version of DenseLindbladCollection.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) array.
            drift: Constant term :math:`H_d` to be added to the Hamiltonian of the system.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) array.
            decimals: operator values will be rounded to ``decimals`` places after the
                decimal place to avoid excess storage of near-zero values
                in sparse format.
        """

        self._hamiltonian_operators = np.empty(shape=len(hamiltonian_operators), dtype="O")
        # pylint: disable=consider-using-enumerate
        for i in range(len(hamiltonian_operators)):
            if isinstance(hamiltonian_operators[i], Operator):
                hamiltonian_operators[i] = to_array(hamiltonian_operators[i])
            self._hamiltonian_operators[i] = csr_matrix(
                np.round(hamiltonian_operators[i], decimals)
            )
        if isinstance(drift, Operator):
            drift = to_array(drift)
        self.drift = csr_matrix(np.round(drift, decimals))
        if dissipator_operators is not None:
            self._dissipator_operators = np.empty(shape=len(dissipator_operators), dtype="O")
            self._dissipator_operators_conj = np.empty_like(self._dissipator_operators)
            # pylint: disable=consider-using-enumerate
            for i in range(len(dissipator_operators)):
                if isinstance(dissipator_operators[i], Operator):
                    dissipator_operators[i] = to_array(dissipator_operators[i])
                self._dissipator_operators[i] = csr_matrix(
                    np.round(dissipator_operators[i], decimals)
                )
                self._dissipator_operators_conj[i] = (
                    self._dissipator_operators[i].conjugate().transpose()
                )
            self._dissipator_products = self._dissipator_operators_conj * self._dissipator_operators
        else:
            self._dissipator_operators = None

    def evaluate_hamiltonian(self, signal_values: Array) -> csr_matrix:
        return np.sum(signal_values * self._hamiltonian_operators, axis=-1) + self.drift

    def evaluate_rhs(self, ham_sig_vals: Array, dis_sig_vals: Array, y: Array) -> Array:
        r"""Evaluates the RHS of the LindbladModel for a given list of signal values.

        Args:
            ham_sig_vals: stores Hamiltonian signal values :math:`s_j(t)`.
            dis_sig_vals: stores dissipator signal values :math:`\gamma_j(t)`.
                Pass None if no dissipator operators involved.
            y: density matrix of system. (k,n,n) Array.
        Returns:
            RHS of Lindbladian.

        Calculation details:
            * for csr_matrices is equivalent to matrix multiplicaiton.
            We use numpy array broadcasting rules, combined with the above
            fact, to achieve speeds that are substantially faster than a for loop.
            First, in the case of a single (n,n) density matrix, we package the entire
            array as a single-element array whose entry is the array. In the case of
            multiple density matrices a (k,n,n) Array, we package everything as a
            (k,1) Array whose [j,0] entry is the [j,:,:] density matrix.

            In calculating the left- and right-mult contributions, we package
            H+L and H-L as (1) object arrays whose single entry stores the relevant
            sparse matrix. We can then multiply our packaged density matrix and
            [H\pm L]. Using numpy broadcasting rules, [H\pm L] will be broadcast
            to a (k,1) Array for elementwise multiplication with our packaged density
            matrices. After this, elementwise multiplication is applied. This in turn
            references each object's __mul__ function, which–for our csr_matrix components
            means matrix multiplication.

            In calculating the left-right-multiplication part, we use our (m)-shape
            object arrays holding the dissipator operators to perform multiplication.
            We can take an elementwise product with our packaged density matrix, at which
            point our dissipator operators are broadcast as (m) -> (1,m) -> (k,m) shaped,
            and our packaged density matrix as (k,1) -> (k,m). Elementwise multiplication
            is then applied, which is interpreted as matrix multiplication. This yields
            an array where entry [i,j] is an object storing the results of s_jL_j\rho_i L_j^\dagger.
            We can then sum over j and unpackage our object array to get our desired result.
        """
        hamiltonian_matrix = -1j * self.evaluate_hamiltonian(ham_sig_vals)  # B matrix

        # For fast matrix multiplicaiton we need to package (n,n) Arrays as (1)
        # Arrays of dtype object, or (k,n,n) Arrays as (k,1) Arrays of dtype object
        y = package_density_matrices(y)

        if self._dissipator_operators is not None:
            dissipators_matrix = (-1 / 2) * np.sum(
                dis_sig_vals * self._dissipator_products, axis=-1
            )

            left_mult_contribution = np.squeeze([hamiltonian_matrix + dissipators_matrix] * y)
            right_mult_contribution = np.squeeze(y * [-hamiltonian_matrix + dissipators_matrix])

            # both_mult_contribution[i] = \gamma_i L_i\rho L_i^\dagger performed in array language
            both_mult_contribution = (
                (dis_sig_vals * self._dissipator_operators) * y * self._dissipator_operators_conj
            )
            # sum on i
            both_mult_contribution = np.sum(both_mult_contribution, axis=-1)

            out = left_mult_contribution + right_mult_contribution + both_mult_contribution

        else:
            out = (([hamiltonian_matrix] * y) - (y * [hamiltonian_matrix]))[0]
        if len(y.shape) == 2:
            # Very slow; avoid if not necessary (or if better implementation found). Needs to
            # map a (k) Array of dtype object with j^{th} entry a (n,n) Array -> (k,n,n) Array.
            out = unpackage_density_matrices(out.reshape(y.shape[0], 1))

        return out


def package_density_matrices(y: Array) -> Array:
    """Sends a (k,n,n) Array y of density matrices to a
    (k,1) Array of dtype object, where entry [j,0] is
    y[j]. Formally avoids For loops through vectorization.
    Args:
        y: (k,n,n) Array.
    Returns:
        Array with dtype object."""
    # As written here, only works for (n,n) Arrays
    obj_arr = np.empty(shape=(1), dtype="O")
    obj_arr[0] = y
    return obj_arr


# Using vectorization with signature, works on (k,n,n) Arrays -> (k,1) Array
package_density_matrices = np.vectorize(package_density_matrices, signature="(n,n)->(1)")


def unpackage_density_matrices(y: Array) -> Array:
    """Inverse function of package_density_matrices,
    Much slower than packaging. Avoid using unless
    absolutely needed (as in case of passing multiple
    density matrices to SparseLindbladCollection.evaluate_rhs)."""
    return y[0]


unpackage_density_matrices = np.vectorize(unpackage_density_matrices, signature="(1)->(n,n)")

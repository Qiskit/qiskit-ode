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

"""Tests for operator_collections.py."""

import numpy as np
import numpy.random as rand

from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.operator_collections import (
    DenseOperatorCollection,
    DenseLindbladCollection,
    DenseVectorizedLindbladCollection,
    SparseLindbladCollection,
    SparseOperatorCollection,
)
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestDenseOperatorCollection(QiskitDynamicsTestCase):
    """Tests for DenseOperatorCollection."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        self.test_operator_list = Array([self.X, self.Y, self.Z])

        self.signals = SignalList([Signal(1, j / 3) for j in range(3)])
        self.sigvals = np.real(self.signals.complex_value(2))

        self.simple_collection = DenseOperatorCollection(
            self.test_operator_list, static_operator=None
        )

    def test_known_values_basic_functionality(self):
        """Test DenseOperatorCollection evaluation against
        analytically known values."""
        res = self.simple_collection(self.sigvals)
        self.assertAllClose(res, Array([[-0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 0.5 + 0j]]))

        res = (DenseOperatorCollection(self.test_operator_list, static_operator=np.eye(2)))(
            self.sigvals
        )
        self.assertAllClose(res, Array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]))

    def test_basic_functionality_pseudorandom(self):
        """Test DenseOperatorCollection evaluation
        using pseudorandom arrays."""
        rand.seed(0)
        vals = rand.uniform(-1, 1, 32) + 1j * rand.uniform(-1, 1, (10, 32))
        arr = rand.uniform(-1, 1, (32, 128, 128))
        res = (DenseOperatorCollection(arr))(vals)
        for i in range(10):
            total = 0
            for j in range(32):
                total = total + vals[i, j] * arr[j]
            self.assertAllClose(res[i], total)


class TestDenseOperatorCollectionJax(TestDenseOperatorCollection, TestJaxBase):
    """Jax version of TestDenseOperatorCollection tests.

    Note: This class has more tests due to inheritance.
    """

    def test_functions_jitable(self):
        """Tests that all class functions are jittable."""
        doc = DenseOperatorCollection(
            Array(self.test_operator_list), static_operator=Array(self.test_operator_list[0])
        )
        self.jit_wrap(doc.evaluate)(Array(self.sigvals))
        self.jit_wrap(doc.evaluate_rhs)(Array(self.sigvals), self.X)

    def test_functions_gradable(self):
        """Tests that all class functions are gradable."""
        doc = DenseOperatorCollection(
            Array(self.test_operator_list), static_operator=Array(self.test_operator_list[0])
        )
        self.jit_grad_wrap(doc.evaluate)(Array(self.sigvals))
        self.jit_grad_wrap(doc.evaluate_rhs)(Array(self.sigvals), self.X)


class TestSparseOperatorCollection(QiskitDynamicsTestCase):
    """Tests for SparseOperatorCopllection."""

    def setUp(self):
        pass

    def test_consistency_with_dense_pseudorandom(self):
        """Tests if SparseOperatorCollection agrees with
        the DenseOperatorCollection."""
        r = lambda *args: np.random.uniform(-1, 1, [*args]) + 1j * np.random.uniform(-1, 1, [*args])
        state = r(16)
        mat = r(4, 16, 16)
        sigVals = r(4)
        static_operator = r(16, 16)
        dense_collection = DenseOperatorCollection(mat, static_operator)
        sparse_collection = SparseOperatorCollection(mat, static_operator)
        dense_val = dense_collection(sigVals)
        sparse_val = sparse_collection(sigVals)
        self.assertAllClose(dense_val, sparse_val.toarray())
        sparse_val = sparse_collection(sigVals, state)
        self.assertAllClose(dense_val @ state, sparse_val)

    def test_constructor_takes_operators(self):
        """Checks that the SparseOperatorcollection constructor
        is able to convert Operator types to csr_matrix."""
        ham_ops = []
        ham_ops_alt = []
        r = lambda *args: np.random.uniform(-1, 1, [*args]) + 1j * np.random.uniform(-1, 1, [*args])
        # pylint: disable=unused-variable
        for i in range(4):
            op = r(3, 3)
            ham_ops.append(Operator(op))
            ham_ops_alt.append(Array(op))
        sigVals = r(4)
        static_operator_numpy_array = r(3, 3)
        sparse_collection_operator_list = SparseOperatorCollection(
            ham_ops, static_operator=Operator(static_operator_numpy_array)
        )
        sparse_collection_array_list = SparseOperatorCollection(
            ham_ops_alt, static_operator=Array(static_operator_numpy_array)
        )
        sparse_collection_pure_array = SparseOperatorCollection(
            Array(ham_ops), static_operator=Array(static_operator_numpy_array)
        )
        a = sparse_collection_operator_list(sigVals)
        b = sparse_collection_array_list(sigVals)
        c = sparse_collection_pure_array(sigVals)
        self.assertAllClose(c.toarray(), a.toarray())
        self.assertAllClose(c.toarray(), b.toarray())


class TestDenseLindbladCollection(QiskitDynamicsTestCase):
    """Tests for DenseLindbladCollection."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)
        rand.seed(2134024)
        n = 16
        k = 8
        m = 4
        l = 2
        self.hamiltonian_operators = rand.uniform(-1, 1, (k, n, n))
        self.dissipator_operators = rand.uniform(-1, 1, (m, n, n))
        self.static_hamiltonian = rand.uniform(-1, 1, (n, n))
        self.rho = rand.uniform(-1, 1, (n, n))
        self.multiple_rho = rand.uniform(-1, 1, (l, n, n))
        self.ham_sig_vals = rand.uniform(-1, 1, (k))
        self.dis_sig_vals = rand.uniform(-1, 1, (m))

    def test_no_static_hamiltonian_no_dissipator(self):
        """Tests whether collections function correctly with no static_hamiltonian
        and no dissipators."""

        ham_only_collection = DenseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=0 * self.static_hamiltonian,
            dissipator_operators=None,
        )
        hamiltonian = np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
        res = ham_only_collection(self.ham_sig_vals, None, self.rho)

        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

    def test_static_hamiltonian_no_dissipator(self):
        """Tests if providing static_hamiltonian but no dissipator is OK."""
        # Now, test adding a static_hamiltonian
        ham_static_hamiltonian_collection = DenseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=None,
        )
        hamiltonian = (
            np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
            + self.static_hamiltonian
        )
        res = ham_static_hamiltonian_collection(self.ham_sig_vals, None, self.rho)
        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

    def test_static_hamiltonian_dissipator(self):
        """Tests if providing both static_hamiltonian and dissipator is OK."""
        full_lindblad_collection = DenseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
        )
        res = full_lindblad_collection(self.ham_sig_vals, self.dis_sig_vals, self.rho)
        hamiltonian = (
            np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
            + self.static_hamiltonian
        )
        ham_terms = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        dis_anticommutator = (-1 / 2) * np.tensordot(
            self.dis_sig_vals,
            np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1]))
            @ self.dissipator_operators,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(self.rho) + self.rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            self.dis_sig_vals,
            self.dissipator_operators
            @ self.rho
            @ np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(ham_terms + dis_anticommutator + dis_extra, res)

    def test_multiple_density_matrix_evaluation(self):
        """Test to ensure that passing multiple density matrices as a (k,n,n) Array functions."""

        # Now, test if vectorization works as intended
        full_lindblad_collection = DenseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
        )
        res = full_lindblad_collection(self.ham_sig_vals, self.dis_sig_vals, self.multiple_rho)
        for i in range(len(self.multiple_rho)):
            self.assertAllClose(
                res[i],
                full_lindblad_collection(
                    self.ham_sig_vals, self.dis_sig_vals, self.multiple_rho[i]
                ),
            )


class TestDenseLindbladCollectionJax(TestDenseLindbladCollection, TestJaxBase):
    """Jax version of TestDenseLindbladCollection tests.

    Note: This class has more tests due to inheritance.
    """

    def test_functions_jitable(self):
        """Tests that all class functions are jittable"""
        dlc = DenseLindbladCollection(
            Array(self.hamiltonian_operators),
            static_hamiltonian=Array(self.static_hamiltonian),
            dissipator_operators=Array(self.dissipator_operators),
        )

        self.jit_wrap(dlc.evaluate_rhs)(
            Array(self.ham_sig_vals), Array(self.dis_sig_vals), self.rho
        )
        self.jit_wrap(dlc.evaluate_hamiltonian)(Array(self.ham_sig_vals))

    def test_functions_gradable(self):
        """Tests if all class functions are gradable"""
        dlc = DenseLindbladCollection(
            Array(self.hamiltonian_operators),
            static_hamiltonian=Array(self.static_hamiltonian),
            dissipator_operators=Array(self.dissipator_operators),
        )
        self.jit_grad_wrap(dlc.evaluate_rhs)(
            Array(self.ham_sig_vals), Array(self.dis_sig_vals), self.rho
        )
        self.jit_grad_wrap(dlc.evaluate_hamiltonian)(Array(self.ham_sig_vals))


class TestDenseVectorizedLindbladCollection(QiskitDynamicsTestCase):
    """Tests for DenseVectorizedLindbladCollection.
    Mostly checks consistency with DenseLindbladCollection."""

    def setUp(self) -> None:
        rand.seed(123098341)
        n = 16
        k = 4
        m = 2
        r = lambda *args: rand.uniform(-1, 1, [*args]) + 1j * rand.uniform(-1, 1, [*args])

        self.r = r
        self.rand_ham = r(k, n, n)
        self.rand_dis = r(m, n, n)
        self.rand_dft = r(n, n)
        self.rho = r(n, n)
        self.t = r()
        self.rand_ham_sigs = SignalList([Signal(r(), r(), r()) for j in range(k)])
        self.rand_dis_sigs = SignalList([Signal(r(), r(), r()) for j in range(m)])

    def test_consistency_static_hamiltonian_dissipator(self):
        """Check consistency with DenseLindbladCollection when hamiltonian,
        static_hamiltonian, and dissipator terms defined."""
        stdLindblad = DenseLindbladCollection(
            self.rand_ham, static_hamiltonian=self.rand_dft, dissipator_operators=self.rand_dis
        )
        vecLindblad = DenseVectorizedLindbladCollection(
            self.rand_ham, static_hamiltonian=self.rand_dft, dissipator_operators=self.rand_dis
        )

        a = stdLindblad.evaluate_rhs(
            self.rand_ham_sigs(self.t), self.rand_dis_sigs(self.t), self.rho
        ).flatten(order="F")
        b = vecLindblad.evaluate_rhs(
            self.rand_ham_sigs(self.t), self.rand_dis_sigs(self.t), self.rho.flatten(order="F")
        )
        self.assertAllClose(a, b)

    def test_consistency_static_hamiltonian_no_dissipator(self):
        """Check consistency when only hamiltonian and static_hamiltonian terms defined."""
        stdLindblad = DenseLindbladCollection(
            self.rand_ham, static_hamiltonian=self.rand_dft, dissipator_operators=None
        )
        vecLindblad = DenseVectorizedLindbladCollection(
            self.rand_ham, static_hamiltonian=self.rand_dft, dissipator_operators=None
        )

        a = stdLindblad.evaluate_rhs(self.rand_ham_sigs(self.t), None, self.rho).flatten(order="F")
        b = vecLindblad.evaluate_rhs(self.rand_ham_sigs(self.t), None, self.rho.flatten(order="F"))
        self.assertAllClose(a, b)


class TestDenseVectorizedLindbladCollectionJax(TestDenseVectorizedLindbladCollection, TestJaxBase):
    """Jax version of TestDenseVectorizedLindbladCollection tests.

    Note: This class has more tests due to inheritance.

    Note: The evaluation processes for DenseVectorizedLindbladCollection
    are not directly jitable or compilable. The compilation of these steps
    is taken care of by the tests for LindbladModel.
    """

    # DenseVectorizedLindbladCollection has methods that cannot be jited
    # or graded. Despite this, the more important methods of evaluate
    # and evaluate_rhs of LindbladModel with evaluation_mode = "dense_vectorized"
    # are jitable and gradable.


class TestSparseLindbladCollection(QiskitDynamicsTestCase):
    """Tests for SparseLindbladCollection."""

    def setUp(self):
        rand.seed(9098321)
        n = 16
        k = 8
        m = 4
        l = 2
        self.hamiltonian_operators = rand.uniform(-1, 1, (k, n, n))
        self.dissipator_operators = rand.uniform(-1, 1, (m, n, n))
        self.static_hamiltonian = rand.uniform(-1, 1, (n, n))
        self.rho = rand.uniform(-1, 1, (n, n))
        self.multiple_rho = rand.uniform(-1, 1, (l, n, n))
        self.ham_sig_vals = rand.uniform(-1, 1, (k))
        self.dis_sig_vals = rand.uniform(-1, 1, (m))
        self.r = lambda *args: rand.uniform(-1, 1, args) + 1j * rand.uniform(-1, 1, args)

    def test_no_static_hamiltonian_no_dissipator(self):
        """Tests whether collections function correctly with no static_hamiltonian
        and no dissipators."""

        ham_only_collection = SparseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=0 * self.static_hamiltonian,
            dissipator_operators=None,
        )
        hamiltonian = np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
        res = ham_only_collection(self.ham_sig_vals, None, self.rho)

        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

    def test_static_hamiltonian_no_dissipator(self):
        """Tests if providing static_hamiltonian but no dissipator is OK."""
        # Now, test adding a static_hamiltonian
        ham_static_hamiltonian_collection = SparseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=None,
        )
        hamiltonian = (
            np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
            + self.static_hamiltonian
        )
        res = ham_static_hamiltonian_collection(self.ham_sig_vals, None, self.rho)
        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

    def test_static_hamiltonian_dissipator(self):
        """Tests if providing both static_hamiltonian and dissipator is OK."""
        full_lindblad_collection = SparseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
        )
        res = full_lindblad_collection(self.ham_sig_vals, self.dis_sig_vals, self.rho)
        hamiltonian = (
            np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
            + self.static_hamiltonian
        )
        ham_terms = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        dis_anticommutator = (-1 / 2) * np.tensordot(
            self.dis_sig_vals,
            np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1]))
            @ self.dissipator_operators,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(self.rho) + self.rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            self.dis_sig_vals,
            self.dissipator_operators
            @ self.rho
            @ np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(ham_terms + dis_anticommutator + dis_extra, res)

    def test_multiple_density_matrix_evaluation(self):
        """Test to ensure that passing multiple density matrices as a (k,n,n) Array functions."""

        # Now, test if vectorization works as intended
        full_lindblad_collection = SparseLindbladCollection(
            self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
        )
        res = full_lindblad_collection(self.ham_sig_vals, self.dis_sig_vals, self.multiple_rho)
        for i in range(len(self.multiple_rho)):
            self.assertAllClose(
                res[i],
                full_lindblad_collection(
                    self.ham_sig_vals, self.dis_sig_vals, self.multiple_rho[i]
                ),
            )

    def test_operator_type_construction(self):
        """Tests if SparseLindbladCollection can take Operator specification of components."""
        ham_op_terms = []
        ham_ar_terms = []
        dis_op_terms = []
        dis_ar_terms = []
        # pylint: disable=unused-variable
        for i in range(4):
            H_i = self.r(3, 3)
            L_i = self.r(3, 3)
            ham_op_terms.append(Operator(H_i))
            ham_ar_terms.append(Array(H_i))
            dis_op_terms.append(Operator(L_i))
            dis_ar_terms.append(Array(L_i))
        H_d = self.r(3, 3)
        op_static_hamiltonian = Operator(H_d)
        ar_static_hamiltonian = Array(H_d)
        op_collection = SparseLindbladCollection(
            hamiltonian_operators=ham_op_terms,
            static_hamiltonian=op_static_hamiltonian,
            dissipator_operators=dis_op_terms,
        )
        ar_collection = SparseLindbladCollection(
            hamiltonian_operators=ham_ar_terms,
            static_hamiltonian=ar_static_hamiltonian,
            dissipator_operators=dis_ar_terms,
        )
        sigVals = self.r(4)
        rho = self.r(3, 3)
        many_rho = self.r(16, 3, 3)
        self.assertAllClose(
            op_collection(sigVals, sigVals, rho), ar_collection(sigVals, sigVals, rho)
        )
        self.assertAllClose(
            op_collection(sigVals, sigVals, many_rho), ar_collection(sigVals, sigVals, many_rho)
        )

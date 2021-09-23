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
# pylint: disable=invalid-name,broad-except

"""Tests for solve_lmde and related functions."""

import numpy as np
from scipy.linalg import expm

from qiskit import QiskitError
from qiskit.quantum_info import Operator

from qiskit_dynamics.models import GeneratorModel, HamiltonianModel, LindbladModel
from qiskit_dynamics.signals import Signal, DiscreteSignal
from qiskit_dynamics import solve_lmde
from qiskit_dynamics.solvers.solver_functions import (
    setup_generator_model_rhs_y0_in_frame_basis,
    results_y_out_of_frame_basis,
)
from qiskit_dynamics.dispatch import Array

from ..common import QiskitDynamicsTestCase, TestJaxBase


class Testsolve_lmde_exceptions(QiskitDynamicsTestCase):
    """Test for solve_lmde error raising."""

    def setUp(self):
        self.lindblad_model = LindbladModel(
            hamiltonian_operators=[np.diag([1.0, -1.0])], hamiltonian_signals=[1.0]
        )

    def test_lmde_method_non_vectorized_lindblad(self):
        """Test error raising if LMDE method is specified for non-vectorized Lindblad."""

        with self.assertRaises(QiskitError) as qe:
            solve_lmde(
                self.lindblad_model, t_span=[0.0, 1.0], y0=np.diag([1.0, 0.0]), method="scipy_expm"
            )
        self.assertTrue("vectorized evaluation" in str(qe.exception))

    def test_method_does_not_exist(self):
        """Test method does not exist exception."""

        with self.assertRaises(QiskitError) as qe:
            solve_lmde(lambda t: t, t_span=[0.0, 1.0], y0=np.array([1.0]), method="notamethod")

        self.assertTrue("not supported" in str(qe.exception))


class TestLMDEGeneratorModelSetup(QiskitDynamicsTestCase):
    """Tests for LMDE GeneratorModel setup methods."""

    def setUp(self):
        """Define some models with different properties."""

        self.ham_model = HamiltonianModel(
            operators=[Operator.from_label("X")],
            signals=[Signal(1.0, 5.0)],
            static_operator=Operator.from_label("Z"),
        )

        self.lindblad_model = LindbladModel(
            hamiltonian_operators=[Operator.from_label("X")],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_hamiltonian=Operator.from_label("Z"),
            dissipator_operators=[Operator.from_label("Y")],
        )

        self.vec_lindblad_model = self.lindblad_model.copy()
        self.vec_lindblad_model.evaluation_mode = "dense_vectorized"

        self.frame_op = 1.2 * Operator.from_label("X") - 3.132 * Operator.from_label("Y")
        _, U = np.linalg.eigh(self.frame_op)
        self.U = U
        self.Uadj = self.U.conj().transpose()

    def test_hamiltonian_setup_no_frame(self):
        """Test functions for Hamiltonian with no frame."""

        y0 = np.array([3.43, 1.31])
        gen, rhs, new_y0 = setup_generator_model_rhs_y0_in_frame_basis(self.ham_model, y0)

        # expect nothing to happen
        self.assertAllClose(y0, new_y0)
        t = 231.232
        self.assertAllClose(gen(t), self.ham_model(t))
        self.assertAllClose(rhs(t, y0), self.ham_model(t, y0))

    def test_hamiltonian_setup(self):
        """Test functions for Hamiltonian with frame."""

        ham_model = self.ham_model.copy()
        ham_model.rotating_frame = self.frame_op

        y0 = np.array([3.43, 1.31])
        gen, rhs, new_y0 = setup_generator_model_rhs_y0_in_frame_basis(ham_model, y0)

        # expect nothing to happen
        self.assertAllClose(self.Uadj @ y0, new_y0)
        t = 231.232
        self.assertAllClose(gen(t), ham_model(t, in_frame_basis=True))
        self.assertAllClose(rhs(t, y0), ham_model(t, y0, in_frame_basis=True))

    def test_lindblad_setup_no_frame(self):
        """Test functions for LindbladModel with no frame."""

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]])
        _, rhs, new_y0 = setup_generator_model_rhs_y0_in_frame_basis(self.lindblad_model, y0)

        # expect nothing to happen
        self.assertAllClose(y0, new_y0)
        t = 231.232
        self.assertAllClose(rhs(t, y0), self.lindblad_model(t, y0))

    def test_lindblad_setup(self):
        """Test functions for LindbladModel with frame."""

        lindblad_model = self.lindblad_model.copy()
        lindblad_model.rotating_frame = self.frame_op

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]])
        _, rhs, new_y0 = setup_generator_model_rhs_y0_in_frame_basis(lindblad_model, y0)

        # expect nothing to happen
        self.assertAllClose(self.Uadj @ y0 @ self.U, new_y0)
        t = 231.232
        self.assertAllClose(rhs(t, y0), lindblad_model(t, y0, in_frame_basis=True))

    def test_vectorized_lindblad_setup_no_frame(self):
        """Test functions for vectorized LindbladModel with no frame."""

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]]).flatten()
        gen, rhs, new_y0 = setup_generator_model_rhs_y0_in_frame_basis(self.vec_lindblad_model, y0)

        # expect nothing to happen
        self.assertAllClose(y0, new_y0)
        t = 231.232
        self.assertAllClose(gen(t), self.vec_lindblad_model(t))
        self.assertAllClose(rhs(t, y0), self.vec_lindblad_model(t, y0))

    def test_vectorized_lindblad_setup(self):
        """Test functions for vectorized LindbladModel with frame."""

        vec_lindblad_model = self.vec_lindblad_model.copy()
        vec_lindblad_model.rotating_frame = self.frame_op

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]]).flatten()
        gen, rhs, new_y0 = setup_generator_model_rhs_y0_in_frame_basis(vec_lindblad_model, y0)

        # expect nothing to happen
        self.assertAllClose(np.kron(self.Uadj.conj(), self.Uadj) @ y0, new_y0)
        t = 231.232
        self.assertAllClose(gen(t), vec_lindblad_model(t, in_frame_basis=True))
        self.assertAllClose(rhs(t, y0), vec_lindblad_model(t, y0, in_frame_basis=True))

    def test_hamiltonian_results_conversion_no_frame(self):
        """Test hamiltonian results conversion with no frame."""

        # test 1d input
        results_y = np.array([[1.0, 23.3], [2.32, 1.232]])
        output = results_y_out_of_frame_basis(self.ham_model, results_y, y0_ndim=1)
        self.assertAllClose(results_y, output)

        # test 2d input
        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(self.ham_model, results_y, y0_ndim=2)
        self.assertAllClose(results_y, output)

    def test_hamiltonian_results_conversion(self):
        """Test hamiltonian results conversion."""

        ham_model = self.ham_model.copy()
        ham_model.rotating_frame = self.frame_op

        # test 1d input
        results_y = np.array([[1.0, 23.3], [2.32, 1.232]])
        output = results_y_out_of_frame_basis(ham_model, results_y, y0_ndim=1)
        expected = [self.U @ y for y in results_y]
        self.assertAllClose(expected, output)

        # test 2d input
        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(ham_model, results_y, y0_ndim=2)
        expected = [self.U @ y for y in results_y]
        self.assertAllClose(expected, output)

    def test_lindblad_results_conversion_no_frame(self):
        """Test lindblad results conversion with no frame."""

        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(self.lindblad_model, results_y, y0_ndim=2)
        self.assertAllClose(results_y, output)

    def test_lindblad_results_conversion(self):
        """Test lindblad results conversion."""

        lindblad_model = self.lindblad_model.copy()
        lindblad_model.rotating_frame = self.frame_op

        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(lindblad_model, results_y, y0_ndim=2)
        expected = [self.U @ y @ self.Uadj for y in results_y]
        self.assertAllClose(expected, output)

    def test_vectorized_lindblad_results_conversion_no_frame(self):
        """Test vectorized lindblad results conversion with no frame."""

        # test 1d input
        results_y = np.array([[1.0, 23.3, 1.23, 0.123], [2.32, 1.232, 1j, 21.2]])
        output = results_y_out_of_frame_basis(self.vec_lindblad_model, results_y, y0_ndim=1)
        self.assertAllClose(results_y, output)

        # test 2d input
        results_y = np.array(
            [
                [[1.0, 23.3], [23, 231j], [2.231, 32.32], [2.321, 2.231]],
                [[2.32, 1.232], [1j, 2.0 + 3j], [2.32, 2.12314], [334.0, 34.3]],
            ]
        )
        output = results_y_out_of_frame_basis(self.vec_lindblad_model, results_y, y0_ndim=2)
        self.assertAllClose(results_y, output)

    def test_vectorized_lindblad_results_conversion(self):
        """Test vectorized lindblad results conversion."""

        vec_lindblad_model = self.vec_lindblad_model.copy()
        vec_lindblad_model.rotating_frame = self.frame_op
        P = np.kron(self.U.conj(), self.U)

        # test 1d input
        results_y = np.array([[1.0, 23.3, 1.23, 0.123], [2.32, 1.232, 1j, 21.2]])
        output = results_y_out_of_frame_basis(vec_lindblad_model, results_y, y0_ndim=1)
        expected = [P @ y for y in results_y]
        self.assertAllClose(expected, output)

        # test 2d input
        results_y = np.array(
            [
                [[1.0, 23.3], [23, 231j], [2.231, 32.32], [2.321, 2.231]],
                [[2.32, 1.232], [1j, 2.0 + 3j], [2.32, 2.12314], [334.0, 34.3]],
            ]
        )
        output = results_y_out_of_frame_basis(vec_lindblad_model, results_y, y0_ndim=2)
        expected = [P @ y for y in results_y]
        self.assertAllClose(expected, output)


# pylint: disable=too-many-instance-attributes
class Testsolve_lmde_Base(QiskitDynamicsTestCase):
    """Some reusable routines for high level solve_lmde tests."""

    def setUp(self):
        self.t_span = [0.0, 1.0]
        self.y0 = Array(np.eye(2, dtype=complex))

        self.X = Array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self.Y = Array([[0.0, -1j], [1j, 0.0]], dtype=complex)
        self.Z = Array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        # simple generator and rhs
        # pylint: disable=unused-argument
        def generator(t):
            return -1j * 2 * np.pi * self.X / 2

        self.basic_generator = generator

    def _fixed_step_LMDE_method_tests(self, method):
        results = solve_lmde(
            self.basic_generator, t_span=self.t_span, y0=self.y0, method=method, max_dt=0.1
        )

        expected = expm(-1j * np.pi * self.X.data)

        self.assertAllClose(results.y[-1], expected)

        # randomized LMDE example
        dim = 7
        b = 0.5
        rng = np.random.default_rng(3093)
        static_operator = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        operators = rng.uniform(low=-b, high=b, size=(1, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(1, dim, dim)
        )
        frame_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = frame_op.conj().transpose() - frame_op
        y0 = rng.uniform(low=-b, high=b, size=(dim,)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim,)
        )

        sig = DiscreteSignal(
            samples=rng.uniform(low=-b, high=b, size=(5,)), dt=0.1, carrier_freq=1.0
        )
        model = GeneratorModel(
            operators=operators,
            signals=[sig],
            static_operator=static_operator,
            rotating_frame=frame_op,
        )

        results = solve_lmde(model, t_span=[0, 0.5], y0=y0, method=method, max_dt=0.01)
        yf = model.rotating_frame.state_out_of_frame(0.5, results.y[-1])

        # simulate directly out of frame
        def generator(t):
            return static_operator + sig(t) * operators[0]

        results2 = solve_lmde(generator, t_span=[0, 0.5], y0=y0, method=method, max_dt=0.01)

        # check consistency - this is relatively low tolerance due to the solver tolerance
        self.assertAllClose(yf, results2.y[-1], atol=1e-5, rtol=1e-5)


class Testsolve_lmde_scipy_expm(Testsolve_lmde_Base):
    """Basic tests for solve_lmde with method=='expm'."""

    def test_scipy_expm_solver(self):
        """Test scipy_expm_solver."""
        self._fixed_step_LMDE_method_tests("scipy_expm")


class Testsolve_lmde_jax_expm(Testsolve_lmde_Base, TestJaxBase):
    """Basic tests for solve_lmde with method=='jax_expm'."""

    def test_jax_expm_solver(self):
        """Test jax_expm_solver."""
        self._fixed_step_LMDE_method_tests("jax_expm")

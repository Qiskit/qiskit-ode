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
# pylint: disable=invalid-name

"""
Tests for solver classes module.
"""

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info import Operator, Statevector, SuperOp, DensityMatrix

from qiskit_dynamics import Solver
from qiskit_dynamics.signals import Signal

from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestSolverExceptions(QiskitDynamicsTestCase):
    """Tests for Solver exception raising based on input types."""

    def setUp(self):
        X = Operator.from_label("X")
        self.ham_solver = Solver(hamiltonian_operators=[X], hamiltonian_signals=[1.0])

        self.lindblad_solver = Solver(
            hamiltonian_operators=[X], hamiltonian_signals=[1.0], dissipator_operators=[X]
        )

        self.vec_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[1.0],
            dissipator_operators=[X],
            evaluation_mode="dense_vectorized",
        )

    def test_hamiltonian_shape_error(self):
        """Test error raising if invalid shape for Hamiltonian model."""

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve([0.0, 1.0], np.array([1.0, 0.0, 0.0]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve([0.0, 1.0], np.array([[[1.0, 0.0, 0.0]]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve([0.0, 1.0], Statevector(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_lindblad_shape_error(self):
        """Test error raising if invalid shape for Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0.0, 1.0], np.array([1.0, 0.0, 0.0]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0.0, 1.0], np.array([[[1.0, 0.0, 0.0]]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0.0, 1.0], Statevector(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_vectorized_lindblad_shape_error(self):
        """Test error raising if invalid shape for vectorized Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve([0.0, 1.0], np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve([0.0, 1.0], DensityMatrix(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve([0.0, 1.0], Statevector(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_non_vectorized_SuperOp_error(self):
        """Test SuperOp simulation attempt for non-vectorized Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0.0, 1.0], SuperOp(np.eye(4)))
        self.assertTrue("Simulating SuperOp" in str(qe.exception))


class TestSolver(QiskitDynamicsTestCase):
    """Tests for Solver class."""

    def setUp(self):
        """Set up some simple models."""
        X = 2 * np.pi * Operator.from_label("X") / 2
        Z = 2 * np.pi * Operator.from_label("Z") / 2
        self.ham_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.rwa_ham_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            rwa_cutoff_freq=2 * 5.0,
        )

        self.lindblad_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            dissipator_operators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.vec_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            dissipator_operators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            evaluation_mode="dense_vectorized",
        )

        # lindblad solver with no dissipation for testing
        self.vec_lindblad_solver_no_diss = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            dissipator_operators=[0.0 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            evaluation_mode="dense_vectorized",
        )
        self.method = "DOP853"

    def test_lindblad_solve_statevector(self):
        """Test correct conversion of Statevector to DensityMatrix."""

        results = self.lindblad_solver.solve(
            [0.0, 1.0], y0=Statevector([0.0, 1.0]), method=self.method
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(results.y[-1].data[0, 0] > 0.99 and results.y[-1].data[0, 0] < 0.999)

    def test_vec_lindblad_statevector(self):
        """Test correct conversion of Statevector to DensityMatrix and vectorized solving."""

        results = self.vec_lindblad_solver.solve(
            [0.0, 1.0], y0=Statevector([0.0, 1.0]), method=self.method
        )
        results2 = self.lindblad_solver.solve(
            [0.0, 1.0], y0=Statevector([0.0, 1.0]), method=self.method
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertAllClose(results.y[-1].data, results2.y[-1].data)

    def test_array_vectorized_lindblad(self):
        """Test Lindblad solver is array-vectorized."""
        results = self.lindblad_solver.solve(
            [0.0, 1.0],
            y0=np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]]),
            method=self.method,
        )
        self.assertTrue(results.y[-1][0, 0, 0] > 0.99 and results.y[-1][0, 0, 0] < 0.999)
        self.assertTrue(results.y[-1][1, 1, 1] > 0.99 and results.y[-1][1, 1, 1] < 0.999)

    def test_rwa_hamiltonian(self):
        """Test perfect inversion for pi pulse with RWA."""
        results = self.rwa_ham_solver.solve(
            [0.0, 1.0], y0=np.array([0.0, 1.0]), atol=1e-10, rtol=1e-10, method=self.method
        )
        self.assertTrue(np.abs(results.y[-1][0]) > (1 - 1e-8))

    def test_hamiltonian_DensityMatrix(self):
        """Test correct conjugation of Hamiltonian-based density matrix simulation."""
        results = self.ham_solver.solve(
            [0.0, 1.0],
            y0=DensityMatrix(np.array([0.0, 1.0])),
            atol=1e-10,
            rtol=1e-10,
            method=self.method,
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(np.abs(results.y[-1].data[0, 0]) > 0.999)

    def test_hamiltonian_SuperOp(self):
        """Test Hamiltonian-based SuperOp simulation."""
        results = self.rwa_ham_solver.solve(
            [0.0, 1.0], y0=SuperOp(np.eye(4)), atol=1e-10, rtol=1e-10, method=self.method
        )
        self.assertTrue(isinstance(results.y[-1], SuperOp))
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAllClose(results.y[-1].data, np.kron(X, X))

    def test_hamiltonian_lindblad_SuperOp_consistency(self):
        """Test Hamiltonian-based SuperOp simulation."""
        results = self.ham_solver.solve(
            [0.0, 0.432], y0=SuperOp(np.eye(4)), atol=1e-10, rtol=1e-10, method=self.method
        )
        results2 = self.vec_lindblad_solver_no_diss.solve(
            [0.0, 0.432], y0=SuperOp(np.eye(4)), atol=1e-10, rtol=1e-10
        )
        self.assertAllClose(results.y[-1].data, results2.y[-1].data)


class TestSolverJax(TestSolver, TestJaxBase):
    """JAX version of TestSolver."""

    def setUp(self):
        """Set method to 'jax_odeint' to speed up running of jax version of tests."""
        super().setUp()
        self.method = "jax_odeint"

    def test_jit_solve(self):
        """Test jitting setting signals and solving."""

        def func(a):
            ham_solver = self.ham_solver.copy()
            ham_solver.signals = [Signal(lambda t: a, 5.0)]
            yf = ham_solver.solve(
                np.array([0.0, 1.0]), y0=np.array([0.0, 1.0]), method=self.method
            ).y[-1]
            return yf

        jit_func = self.jit_wrap(func)
        self.assertAllClose(jit_func(2.0), func(2.0))

    def test_jit_grad_solve(self):
        """Test jitting setting signals and solving."""

        def func(a):
            lindblad_solver = self.lindblad_solver.copy()
            lindblad_solver.signals = [[Signal(lambda t: a, 5.0)], [1.0]]
            yf = lindblad_solver.solve(
                [0.0, 1.0], y0=np.array([[0.0, 1.0], [0.0, 1.0]]), method=self.method
            ).y[-1]
            return yf

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)

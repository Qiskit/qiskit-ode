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

"""tests for qiskit_dynamics.models.HamiltonianModel"""

import numpy as np

from scipy.linalg import expm

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import HamiltonianModel
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestHamiltonianModelValidation(QiskitDynamicsTestCase):
    """Test validation handling of HamiltonianModel."""

    def test_operators_not_hermitian(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]

        with self.assertRaises(QiskitError) as qe:
            HamiltonianModel(operators=operators)
        self.assertTrue("operators must be Hermitian." in str(qe.exception))

    def test_static_operator_not_hermitian(self):
        """Test raising error if static_operator is not Hermitian."""

        static_operator = np.array([[0.0, 1.0], [0.0, 0.0]])
        operators = [np.array([[0.0, 1.0], [1.0, 0.0]])]

        with self.assertRaises(QiskitError) as qe:
            HamiltonianModel(operators=operators, static_operator=static_operator)
        self.assertTrue("static_operator must be Hermitian." in str(qe.exception))

    def test_validate_false(self):
        """Verify setting validate=False avoids error raising."""

        ham_model = HamiltonianModel(
            operators=[np.array([[0.0, 1.0], [0.0, 0.0]])], signals=[1.0], validate=False
        )

        self.assertAllClose(ham_model(1.0), -1j * np.array([[0.0, 1.0], [0.0, 0.0]]))


class TestHamiltonianModel(QiskitDynamicsTestCase):
    """Tests for HamiltonianModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        # define a basic hamiltonian
        w = 2.0
        r = 0.5
        operators = [2 * np.pi * self.Z / 2, 2 * np.pi * r * self.X / 2]
        signals = [w, Signal(1.0, w)]

        self.w = w
        self.r = r
        self.basic_hamiltonian = HamiltonianModel(operators=operators, signals=signals)

    def _basic_frame_evaluate_test(self, frame_operator, t):
        """Routine for testing setting of valid frame operators using
        basic_hamiltonian.

        Adapted from the version of this test in
        test_operator_models.py, but relative to the way HamiltonianModel
        modifies frame handling.

        Args:
            frame_operator (Array): now assumed to be a Hermitian operator H, with the
                            frame being entered being F=-1j * H
            t (float): time of frame transformation
        """

        self.basic_hamiltonian.rotating_frame = frame_operator

        # convert to 2d array
        if isinstance(frame_operator, Operator):
            frame_operator = Array(frame_operator.data)
        if isinstance(frame_operator, Array) and frame_operator.ndim == 1:
            frame_operator = np.diag(frame_operator)

        value = self.basic_hamiltonian(t) / -1j

        twopi = 2 * np.pi

        # frame is F=-1j * H, and need to compute exp(-F * t)
        U = expm(1j * np.array(frame_operator) * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (
            twopi * self.w * U @ self.Z.data @ U.conj().transpose() / 2
            + d_coeff * twopi * U @ self.X.data @ U.conj().transpose() / 2
            - frame_operator
        )

        self.assertAllClose(value, expected)

    def test_diag_frame_operator_basic_hamiltonian(self):
        """Test setting a diagonal frame operator for the internally
        set up basic hamiltonian.
        """

        self._basic_frame_evaluate_test(Array([1.0, -1.0]), 1.123)
        self._basic_frame_evaluate_test(Array([1.0, -1.0]), np.pi)

    def test_non_diag_frame_operator_basic_hamiltonian(self):
        """Test setting a non-diagonal frame operator for the internally
        set up basic model.
        """
        self._basic_frame_evaluate_test(self.Y + self.Z, 1.123)
        self._basic_frame_evaluate_test(self.Y - self.Z, np.pi)

    def test_evaluate_no_frame_basic_hamiltonian(self):
        """Test generator evaluation without a frame in the basic model."""

        t = 3.21412
        value = self.basic_hamiltonian(t) / -1j
        twopi = 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = twopi * self.w * self.Z.data / 2 + twopi * d_coeff * self.X.data / 2

        self.assertAllClose(value, expected)

    def test_evaluate_in_frame_basis_basic_hamiltonian(self):
        """Test generator evaluation in frame basis in the basic_hamiltonian."""

        frame_op = (self.X + 0.2 * self.Y + 0.1 * self.Z).data

        # enter the frame given by the -1j * X
        self.basic_hamiltonian.rotating_frame = frame_op

        # get the frame basis used in model. Note that the Frame object
        # orders the basis according to the ordering of eigh
        _, U = np.linalg.eigh(frame_op)

        t = 3.21412
        value = self.basic_hamiltonian(t, in_frame_basis=True) / -1j

        # compose the frame basis transformation with the exponential
        # frame rotation (this will be multiplied on the right)
        U = expm(-1j * frame_op * t) @ U
        Uadj = U.conj().transpose()

        twopi = 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = (
            Uadj
            @ (twopi * self.w * self.Z.data / 2 + twopi * d_coeff * self.X.data / 2 - frame_op)
            @ U
        )

        self.assertAllClose(value, expected)

    def test_evaluate_pseudorandom(self):
        """Test evaluate with pseudorandom inputs."""

        rng = np.random.default_rng(30493)
        num_terms = 3
        dim = 5
        b = 1.0  # bound on size of random terms

        # random hermitian frame operator
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op + rand_op.conj().transpose())

        # random hermitian operators
        randoperators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms, dim, dim)
        )
        randoperators = Array(randoperators + randoperators.conj().transpose([0, 2, 1]))

        rand_coeffs = rng.uniform(low=-b, high=b, size=(num_terms)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms)
        )
        rand_carriers = Array(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = Array(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

        rng = np.random.default_rng(94818)
        num_terms = 5
        dim = 10
        b = 1.0  # bound on size of random terms

        # random hermitian frame operator
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op + rand_op.conj().transpose())

        # random hermitian operators
        randoperators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms, dim, dim)
        )
        randoperators = Array(randoperators + randoperators.conj().transpose([0, 2, 1]))

        rand_coeffs = rng.uniform(low=-b, high=b, size=(num_terms)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms)
        )
        rand_carriers = Array(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = Array(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

    def _test_evaluate(self, frame_op, operators, coefficients, carriers, phases):

        sig_list = []
        for coeff, freq, phase in zip(coefficients, carriers, phases):

            def get_env_func(coeff=coeff):
                # pylint: disable=unused-argument
                def env(t):
                    return coeff

                return env

            sig_list.append(Signal(get_env_func(), freq, phase))
        sig_list = SignalList(sig_list)
        model = HamiltonianModel(
            operators, static_operator=None, signals=sig_list, rotating_frame=frame_op
        )

        value = model(1.0, in_frame_basis=False) / -1j
        coeffs = np.real(coefficients * np.exp(1j * 2 * np.pi * carriers * 1.0 + 1j * phases))
        expected = (
            expm(1j * np.array(frame_op))
            @ np.tensordot(coeffs, operators, axes=1)
            @ expm(-1j * np.array(frame_op))
            - frame_op
        )
        self.assertAllClose(model._signals(1), coeffs)
        self.assertAllClose(model.get_operators(), operators)

        self.assertAllClose(value, expected)


class TestHamiltonianModelJax(TestHamiltonianModel, TestJaxBase):
    """Jax version of TestHamiltonianModel tests.

    Note: This class has contains more tests due to inheritance.
    """

    def test_jitable_funcs(self):
        """Tests whether all functions are jitable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        self.jit_wrap(self.basic_hamiltonian.evaluate)(1)
        self.jit_wrap(self.basic_hamiltonian.evaluate_rhs)(1, Array(np.array([0.2, 0.4])))

        self.basic_hamiltonian.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_wrap(self.basic_hamiltonian.evaluate)(1)
        self.jit_wrap(self.basic_hamiltonian.evaluate_rhs)(1, Array(np.array([0.2, 0.4])))

        self.basic_hamiltonian.rotating_frame = None

    def test_gradable_funcs(self):
        """Tests whether all functions are gradable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        self.jit_grad_wrap(self.basic_hamiltonian.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_hamiltonian.evaluate_rhs)(1.0, Array(np.array([0.2, 0.4])))

        self.basic_hamiltonian.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_grad_wrap(self.basic_hamiltonian.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_hamiltonian.evaluate_rhs)(1.0, Array(np.array([0.2, 0.4])))

        self.basic_hamiltonian.rotating_frame = None

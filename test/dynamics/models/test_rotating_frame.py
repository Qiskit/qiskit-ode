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

"""tests for rotating_frame.py"""

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.rotating_frame import RotatingFrame
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestRotatingFrame(QiskitDynamicsTestCase):
    """Tests for RotatingFrame."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

    def test_instantiation_errors(self):
        """Check different modes of error raising for frame setting."""

        with self.assertRaises(QiskitError):
            RotatingFrame(Array([1.0, 1j]))

        with self.assertRaises(QiskitError):
            RotatingFrame(Array([[1.0, 0.0], [0.0, 1j]]))

        with self.assertRaises(QiskitError):
            RotatingFrame(self.Z + 1j * self.X)

    def test_state_out_of_frame_basis(self):
        """Test state_out_of_frame_basis."""

        rng = np.random.default_rng(10933)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = Array(rand_op - rand_op.conj().transpose())
        rotating_frame = RotatingFrame(frame_op)

        _, U = np.linalg.eigh(1j * frame_op)

        y0 = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        val = rotating_frame.state_into_frame_basis(y0)
        expected = U.conj().transpose() @ y0
        self.assertAllClose(val, expected)

        val = rotating_frame.state_out_of_frame_basis(y0)
        expected = U @ y0
        self.assertAllClose(val, expected)

    def test_operator_into_frame_basis(self):
        """Test state_into_frame_basis."""

        rng = np.random.default_rng(98747)
        rand_op = rng.uniform(low=-10, high=10, size=(10, 10)) + 1j * rng.uniform(
            low=-10, high=10, size=(10, 10)
        )

        frame_op = Array(rand_op - rand_op.conj().transpose())
        rotating_frame = RotatingFrame(frame_op)

        _, U = np.linalg.eigh(1j * frame_op)
        Uadj = U.conj().transpose()

        y0 = Array(
            rng.uniform(low=-10, high=10, size=(10, 10))
            + 1j * rng.uniform(low=-10, high=10, size=(10, 10))
        )

        val = rotating_frame.operator_into_frame_basis(y0)
        expected = U.conj().transpose() @ y0 @ U
        self.assertAllClose(val, expected)

        val = rotating_frame.operator_out_of_frame_basis(y0)
        expected = U @ y0 @ Uadj
        self.assertAllClose(val, expected)

    def test_state_transformations_no_frame(self):
        """Test frame transformations with no frame."""

        rotating_frame = RotatingFrame(Array(np.zeros(2)))

        t = 0.123
        y = Array([1.0, 1j])
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)

        t = 100.12498
        y = Array(np.eye(2))
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)

    def test_state_into_frame_2_level(self):
        """Test state_into_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (self.X + 0.1 * self.Y + 12.0 * self.Z).data
        t = 1312.132
        y0 = Array([[1.0, 2.0], [3.0, 4.0]])

        self._test_state_into_frame(t, frame_op, y0)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    def test_state_into_frame_pseudo_random(self):
        """Test state_into_frame with pseudo-random matrices."""
        rng = np.random.default_rng(30493)
        rand_op = rng.uniform(low=-10, high=10, size=(5, 5)) + 1j * rng.uniform(
            low=-10, high=10, size=(5, 5)
        )

        frame_op = Array(rand_op - rand_op.conj().transpose())

        t = 1312.132
        y0 = Array(
            rng.uniform(low=-10, high=10, size=(5, 5))
            + 1j * rng.uniform(low=-10, high=10, size=(5, 5))
        )

        self._test_state_into_frame(t, frame_op, y0)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_state_into_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.state_into_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = U.conj().transpose() @ expected

        expected = np.diag(np.exp(-t * Array(evals))) @ expected

        if not return_in_frame_basis:
            expected = U @ expected

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_state_out_of_frame_2_level(self):
        """Test state_out_of_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (3.1 * self.X + 1.1 * self.Y + 12.0 * self.Z).data
        t = 122.132
        y0 = Array([[1.0, 2.0], [3.0, 4.0]])
        self._test_state_out_of_frame(t, frame_op, y0)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    def test_state_out_of_frame_pseudo_random(self):
        """Test state_out_of_frame with pseudo-random matrices."""
        rng = np.random.default_rng(1382)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = Array(rand_op - rand_op.conj().transpose())

        t = rng.uniform(low=-100, high=100)
        y0 = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_state_out_of_frame(t, frame_op, y0)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_state_out_of_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * Array(evals)

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.state_out_of_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = U.conj().transpose() @ expected

        expected = np.diag(np.exp(t * evals)) @ expected

        if not return_in_frame_basis:
            expected = U @ expected

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_operator_into_frame(self):
        """Test operator_into_frame."""
        rng = np.random.default_rng(94994)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = Array(rand_op - rand_op.conj().transpose())

        t = rng.uniform(low=-100, high=100)
        y0 = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_operator_into_frame(t, frame_op, y0)
        self._test_operator_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_operator_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_operator_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_operator_into_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * Array(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.operator_into_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(-t * evals)) @ expected @ np.diag(np.exp(t * evals))

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_operator_out_of_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(37164093)
        rand_op = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_operator_out_of_frame(t, frame_op, y0)
        self._test_operator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_operator_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_operator_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_operator_out_of_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * Array(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.operator_out_of_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(t * evals)) @ expected @ np.diag(np.exp(-t * evals))

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_generator_into_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(111)
        rand_op = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_generator_into_frame(t, frame_op, y0)
        self._test_generator_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_generator_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_generator_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_generator_into_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        """Helper function for testing generator_into_frame."""
        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * Array(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.generator_into_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(-t * evals)) @ expected @ np.diag(np.exp(t * evals))
        expected = expected - np.diag(evals)

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_generator_out_of_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(111)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = Array(rand_op - rand_op.conj().transpose())

        t = rng.uniform(low=-100, high=100)
        y0 = Array(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_generator_out_of_frame(t, frame_op, y0)
        self._test_generator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_generator_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_generator_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_generator_out_of_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        """Helper function for testing generator_into_frame."""
        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * Array(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.generator_out_of_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = np.diag(np.exp(t * evals)) @ expected @ np.diag(np.exp(-t * evals))
        expected = expected + np.diag(evals)

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_vectorized_conjugate_and_add_conventions(self):
        """Test whether passing a vectorized (dim**2, k) operator to _conjugate_and_add
        with vectorized_operators = True is the same as passing a (k,dim,dim) array of
        operators."""
        vectorized_rhos = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ).transpose()

        nonvectord_rhos = vectorized_rhos.reshape((2, 2, 3), order="F").transpose([2, 0, 1])
        rotating_frame = RotatingFrame(np.array([1j, 2j]))

        vectorized_result = rotating_frame._conjugate_and_add(
            0.1, vectorized_rhos, vectorized_operators=True
        )
        nonvectord_result = (
            rotating_frame._conjugate_and_add(0.1, nonvectord_rhos)
            .reshape(3, 4, order="F")
            .transpose()
        )
        self.assertAllClose(vectorized_result, nonvectord_result)

    def test_vectorized_frame_basis(self):
        """Test correct lazy evaluation of vectorized_frame_basis."""

        rng = np.random.default_rng(12983)
        F = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )
        F = F - F.conj().transpose()

        rotating_frame = RotatingFrame(F)

        op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        op1 = rotating_frame.operator_into_frame_basis(op)
        op2 = rotating_frame.vectorized_frame_basis_adjoint @ op.flatten(order="F")

        self.assertAllClose(op1, op2.reshape((6, 6), order="F"))

        op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        op1 = rotating_frame.operator_out_of_frame_basis(op)
        op2 = rotating_frame.vectorized_frame_basis @ op.flatten(order="F")

        self.assertAllClose(op1, op2.reshape((6, 6), order="F"))


class TestRotatingFrameJax(TestRotatingFrame, TestJaxBase):
    """Jax version of TestRotatingFrame tests.

    Note: This class has more tests due to inheritance.
    """

    def test_instantiation_errors(self):
        """Check different modes of error raising for frame setting.
        Needs to be overwrititen for jax due to different behaviour.
        """

        # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp

        rotating_frame = RotatingFrame(Array([1.0, 1j]))
        self.assertTrue(jnp.isnan(rotating_frame.frame_diag[0]))

        rotating_frame = RotatingFrame(Array([[1.0, 0.0], [0.0, 1j]]))
        self.assertTrue(jnp.isnan(rotating_frame.frame_diag[0]))

        rotating_frame = RotatingFrame(self.Z + 1j * self.X)
        self.assertTrue(jnp.isnan(rotating_frame.frame_diag[0]))

    def test_jitting(self):
        """Test jitting of state_into_frame and _conjugate_and_add."""

        rotating_frame = RotatingFrame(Array([1., -1.]))

        self.jit_wrap(rotating_frame.state_into_frame)(t=0.1, y=np.array([0., 1.]))
        self.jit_wrap(rotating_frame._conjugate_and_add)(t=0.1, y=np.array([[0., 1.], [1., 0.]]))

    def test_jit_and_grad(self):
        """Test jitting and gradding of state_into_frame and _conjugate_and_add."""

        rotating_frame = RotatingFrame(Array([1., -1.]))

        self.jit_grad_wrap(rotating_frame.state_into_frame)(t=0.1, y=np.array([0., 1.]))
        self.jit_grad_wrap(rotating_frame._conjugate_and_add)(t=0.1, y=np.array([[0., 1.], [1., 0.]]))

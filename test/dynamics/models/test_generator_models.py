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

"""Tests for generator_models.py. """

import unittest

from scipy.sparse import issparse, csr_matrix
from scipy.linalg import expm
import numpy as np
import numpy.random as rand
from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import GeneratorModel, RotatingFrame
from qiskit_dynamics.models.generator_models import setup_operators_in_frame
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase

"""
    Unskip this ******************************************************************************************
"""


@unittest.skip("will fix later and don't want this to impact general testing")
class Testsetup_operators_in_frame(QiskitDynamicsTestCase):
    """Tests for setup_operators_in_frame."""

    def test_all_None(self):
        """Test all arguments being None."""

        static_operator, operators = setup_operators_in_frame(None, None, None, None)

        self.assertTrue(static_operator is None)
        self.assertTrue(operators is None)

    def test_array_inputs(self):
        """Test correct handling when operators are arrays."""

        static_operator = -1j * np.array([[1.0, 0.0], [0.0, -1.0]])
        operators = -1j * np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1j], [1j, 0.0]]])
        old_frame = None
        new_frame = -1j * np.array([1.0, -1.0])

        out_static, out_operators = setup_operators_in_frame(
            static_operator, operators, new_frame=new_frame
        )

        self.assertTrue(isinstance(out_static, np.ndarray))
        self.assertTrue(isinstance(out_operators, np.ndarray))

        self.assertAllClose(out_operators, operators)
        self.assertAllClose(out_static, static_operator)

    def test_csr_inputs(self):
        """Test correct handling when operators are csr matrices."""

        static_operator = csr_matrix(-1j * np.array([[1.0, 0.0], [0.0, -1.0]]))
        operators = [
            -1j * csr_matrix([[0.0, 1.0], [1.0, 0.0]]),
            -1j * csr_matrix([[0.0, -1j], [1j, 0.0]]),
        ]
        old_frame = None
        new_frame = -1j * np.array([1.0, -1.0])

        out_static, out_operators = setup_operators_in_frame(
            static_operator, operators, new_frame=new_frame
        )

        self.assertTrue(isinstance(out_static, csr_matrix))
        self.assertTrue(isinstance(out_operators, csr_matrix))

        self.assertAllClose(out_operators, operators)
        self.assertAllClose(out_static, static_operator)


class TestGeneratorModel(QiskitDynamicsTestCase):
    """Tests for GeneratorModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        # define a basic model
        w = 2.0
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * r * self.X / 2]
        signals = [w, Signal(1.0, w)]

        self.w = 2
        self.r = r
        self.basic_model = GeneratorModel(operators=operators, signals=signals)

    def test_diag_frame_operator_basic_model(self):
        """Test setting a diagonal frame operator for the internally
        set up basic model.
        """

        self._basic_frame_evaluate_test(Array([1j, -1j]), 1.123)
        self._basic_frame_evaluate_test(Array([1j, -1j]), np.pi)

    def test_non_diag_frame_operator_basic_model(self):
        """Test setting a non-diagonal frame operator for the internally
        set up basic model.
        """
        self._basic_frame_evaluate_test(-1j * (self.Y + self.Z), 1.123)
        self._basic_frame_evaluate_test(-1j * (self.Y - self.Z), np.pi)

    def _basic_frame_evaluate_test(self, frame_operator, t):
        """Routine for testing setting of valid frame operators using the
        basic_model.
        """
        self.basic_model.rotating_frame = frame_operator

        # convert to 2d array
        if isinstance(frame_operator, Operator):
            frame_operator = Array(frame_operator.data)
        if isinstance(frame_operator, Array) and frame_operator.ndim == 1:
            frame_operator = np.diag(frame_operator)

        value = self.basic_model(t)

        i2pi = -1j * 2 * np.pi

        U = expm(-np.array(frame_operator) * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (
            i2pi * self.w * U @ self.Z.data @ U.conj().transpose() / 2
            + d_coeff * i2pi * U @ self.X.data @ U.conj().transpose() / 2
            - frame_operator
        )

        self.assertAllClose(value, expected)

    def test_evaluate_no_frame_basic_model(self):
        """Test evaluation without a frame in the basic model."""

        t = 3.21412
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = i2pi * self.w * self.Z.data / 2 + i2pi * d_coeff * self.X.data / 2

        self.assertAllClose(value, expected)

    def test_evaluate_generator_in_frame_basis_basic_model(self):
        """Test generator evaluation in frame basis in the basic_model."""

        frame_op = -1j * (self.X + 0.2 * self.Y + 0.1 * self.Z).data

        # enter the frame given by the -1j * X
        self.basic_model.rotating_frame = frame_op

        # get the frame basis that is used in model
        _, U = np.linalg.eigh(1j * frame_op)

        t = 3.21412
        value = self.basic_model(t, in_frame_basis=True)

        # compose the frame basis transformation with the exponential
        # frame rotation (this will be multiplied on the right)
        U = expm(np.array(frame_op) * t) @ U
        Uadj = U.conj().transpose()

        i2pi = -1j * 2 * np.pi
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)
        expected = (
            Uadj
            @ (i2pi * self.w * self.Z.data / 2 + i2pi * d_coeff * self.X.data / 2 - frame_op)
            @ U
        )

        self.assertAllClose(value, expected)

    def test_evaluate_pseudorandom(self):
        """Test evaluate with pseudorandom inputs."""

        rng = np.random.default_rng(30493)
        num_terms = 3
        dim = 5
        b = 1.0  # bound on size of random terms
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op - rand_op.conj().transpose())
        randoperators = rng.uniform(low=-b, high=b, size=(num_terms, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_terms, dim, dim)
        )

        rand_coeffs = Array(
            rng.uniform(low=-b, high=b, size=(num_terms))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms))
        )
        rand_carriers = Array(rng.uniform(low=-b, high=b, size=(num_terms)))
        rand_phases = Array(rng.uniform(low=-b, high=b, size=(num_terms)))

        self._test_evaluate(frame_op, randoperators, rand_coeffs, rand_carriers, rand_phases)

        rng = np.random.default_rng(94818)
        num_terms = 5
        dim = 10
        b = 1.0  # bound on size of random terms
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op - rand_op.conj().transpose())
        randoperators = Array(
            rng.uniform(low=-b, high=b, size=(num_terms, dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms, dim, dim))
        )

        rand_coeffs = Array(
            rng.uniform(low=-b, high=b, size=(num_terms))
            + 1j * rng.uniform(low=-b, high=b, size=(num_terms))
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
        model = GeneratorModel(operators, signals=sig_list)
        model.rotating_frame = frame_op

        value = model(1.0)
        coeffs = np.real(coefficients * np.exp(1j * 2 * np.pi * carriers * 1.0 + 1j * phases))

        self.assertAllClose(
            model.rotating_frame.operator_out_of_frame_basis(
                model._operator_collection.evaluate(coeffs)
            ),
            np.tensordot(coeffs, operators, axes=1) - frame_op,
        )

        expected = (
            expm(-np.array(frame_op))
            @ np.tensordot(coeffs, operators, axes=1)
            @ expm(np.array(frame_op))
            - frame_op
        )

        self.assertAllClose(value, expected)
        if value.backend != "jax":
            model.evaluation_mode = "sparse"
            state = np.arange(operators.shape[-1] * 4).reshape(operators.shape[-1], 4)
            value = model(1.0)
            if issparse(value):
                value = value.toarray()
            self.assertAllClose(value, expected)

            val_with_state = model(1.0, state)
            self.assertAllClose(val_with_state, value @ state)

    def test_signal_setting(self):
        """Test updating the signals."""

        signals = [Signal(lambda t: 2 * t, 1.0), Signal(lambda t: t ** 2, 2.0)]
        self.basic_model.signals = signals

        t = 0.1
        value = self.basic_model(t)
        i2pi = -1j * 2 * np.pi
        Z_coeff = (2 * t) * np.cos(2 * np.pi * 1 * t)
        X_coeff = self.r * (t ** 2) * np.cos(2 * np.pi * 2 * t)
        expected = i2pi * Z_coeff * self.Z.data / 2 + i2pi * X_coeff * self.X.data / 2
        self.assertAllClose(value, expected)

    def test_signal_setting_None(self):
        """Test setting signals to None"""

        self.basic_model.signals = None
        self.assertTrue(self.basic_model.signals is None)

    def test_signal_setting_incorrect_length(self):
        """Test error being raised if signals is the wrong length."""

        with self.assertRaises(QiskitError):
            self.basic_model.signals = [1.0]

    def test_evaluate_analytic(self):
        """Test for checking that with known operators that
        the Model returns the analyticlly known values."""

        test_operator_list = Array([self.X, self.Y, self.Z])
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        simple_model = GeneratorModel(
            test_operator_list, static_operator=None, signals=signals, rotating_frame=None
        )

        res = simple_model.evaluate(2)
        self.assertAllClose(res, Array([[-0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 0.5 + 0j]]))

        simple_model.set_static_operator(np.eye(2), operator_in_frame_basis=False)
        res = simple_model.evaluate(2)
        self.assertAllClose(res, Array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]))
        simple_model.set_static_operator(None, operator_in_frame_basis=False)

    def test_evaluate_in_frame_basis_analytic(self):
        """Tests evaluation in rotating frame against analytic values."""
        test_operator_list = Array([self.X, self.Y, self.Z])
        signals = SignalList([Signal(1, j / 3) for j in range(3)])
        simple_model = GeneratorModel(
            test_operator_list, static_operator=None, signals=signals, rotating_frame=None
        )
        simple_model.set_static_operator(np.eye(2), operator_in_frame_basis=False)
        fop = Array([[0, 1j], [1j, 0]])
        simple_model.rotating_frame = fop
        res = simple_model(2, in_frame_basis=False)
        expected = (
            expm(np.array(-2 * fop))
            @ (Array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]) - fop)
            @ expm(np.array(2 * fop))
        )
        self.assertAllClose(res, expected)

        res = simple_model(2, in_frame_basis=True)
        expected = (
            np.array([[1, 1], [1, -1]])
            / np.sqrt(2)
            @ expected
            @ np.array([[1, 1], [1, -1]])
            / np.sqrt(2)
        )

        self.assertAllClose(res, expected)

    def test_order_of_assigning_properties(self):
        """Tests whether setting the frame, static_operator, and signals
        in the constructor is the same as constructing without
        and then assigning them in any order.
        """

        paulis = Array([self.X, self.Y, self.Z])
        extra = Array(np.eye(2))
        state = Array([0.2, 0.5])

        t = 2

        sarr = [Signal(1, j / 3) for j in range(3)]
        sigvals = SignalList(sarr)(t)

        farr = Array(np.array([[3j, 2j], [2j, 0]]))
        farr2 = Array(np.array([[1j, 2], [-2, 3j]]))
        evals, evect = np.linalg.eig(farr)
        diafarr = np.diag(evals)

        paulis_in_frame_basis = np.conjugate(np.transpose(evect)) @ paulis @ evect

        ## Run checks without rotating frame for now
        gm1 = GeneratorModel(paulis, sarr, extra)
        gm2 = GeneratorModel(paulis, static_operator=extra)
        gm2.signals = sarr
        gm3 = GeneratorModel(paulis, static_operator=extra)
        gm3.signals = SignalList(sarr)

        # All should be the same, because there are no frames involved
        t11 = gm1.evaluate(t, False)
        t12 = gm1.evaluate(t, True)
        t21 = gm2.evaluate(t, False)
        t22 = gm2.evaluate(t, True)
        t31 = gm3.evaluate(t, False)
        t32 = gm3.evaluate(t, True)
        t_analytical = Array([[0.5, 1.0 + 0.5j], [1.0 - 0.5j, 1.5]])

        self.assertAllClose(t11, t12)
        self.assertAllClose(t11, t21)
        self.assertAllClose(t11, t22)
        self.assertAllClose(t11, t31)
        self.assertAllClose(t11, t32)
        self.assertAllClose(t11, t_analytical)

        # now work with a specific statevector
        ts1 = gm1.evaluate_rhs(t, state, in_frame_basis=False)
        ts2 = gm1.evaluate_rhs(t, state, in_frame_basis=True)
        ts_analytical = Array([0.6 + 0.25j, 0.95 - 0.1j])

        self.assertAllClose(ts1, ts2)
        self.assertAllClose(ts1, ts_analytical)
        self.assertAllClose(gm1(t, in_frame_basis=False) @ state, ts1)
        self.assertAllClose(gm1(t, in_frame_basis=True) @ state, ts1)

        ## Now, run checks with frame
        # If passing a frame in the first place, operators must be in frame basis.abs
        # Testing at the same time whether having static_operatorrift = None is an issue.
        gm1 = GeneratorModel(paulis, signals=sarr, rotating_frame=RotatingFrame(farr))
        gm2 = GeneratorModel(paulis, rotating_frame=farr)
        gm2.signals = SignalList(sarr)
        gm3 = GeneratorModel(paulis, rotating_frame=farr)
        gm3.signals = sarr
        # Does adding a frame after make a difference?
        # If so, does it make a difference if we add signals or the frame first?
        gm4 = GeneratorModel(paulis)
        gm4.signals = sarr
        gm4.rotating_frame = farr
        gm5 = GeneratorModel(paulis)
        gm5.rotating_frame = farr
        gm5.signals = sarr
        gm6 = GeneratorModel(paulis, signals=sarr)
        gm6.rotating_frame = farr
        # If we go to one frame, then transform back, does this make a difference?
        gm7 = GeneratorModel(paulis, signals=sarr)
        gm7.rotating_frame = farr2
        gm7.rotating_frame = farr

        t_in_frame_actual = Array(
            np.diag(np.exp(-t * evals))
            @ (np.tensordot(sigvals, paulis_in_frame_basis, axes=1) - diafarr)
            @ np.diag(np.exp(t * evals))
        )
        tf1 = gm1.evaluate(t, in_frame_basis=True)
        tf2 = gm2.evaluate(t, in_frame_basis=True)
        tf3 = gm3.evaluate(t, in_frame_basis=True)
        tf4 = gm4.evaluate(t, in_frame_basis=True)
        tf5 = gm5.evaluate(t, in_frame_basis=True)
        tf6 = gm6.evaluate(t, in_frame_basis=True)
        tf7 = gm7.evaluate(t, in_frame_basis=True)

        self.assertAllClose(t_in_frame_actual, tf1)
        self.assertAllClose(tf1, tf2)
        self.assertAllClose(tf1, tf3)
        self.assertAllClose(tf1, tf4)
        self.assertAllClose(tf1, tf5)
        self.assertAllClose(tf1, tf6)
        self.assertAllClose(tf1, tf7)

    def test_evaluate_rhs_lmult_equivalent_analytic(self):
        """Tests whether evaluate(t) @ state == evaluate_rhs(t,state)
        for analytically known values."""

        paulis = Array([self.X, self.Y, self.Z])
        extra = Array(np.eye(2))

        t = 2

        farr = Array(np.array([[3j, 2j], [2j, 0]]))
        evals, evect = np.linalg.eig(farr)
        diafarr = np.diag(evals)

        paulis_in_frame_basis = np.conjugate(np.transpose(evect)) @ paulis @ evect
        extra_in_basis = evect.T.conj() @ extra @ evect

        sarr = [Signal(1, j / 3) for j in range(3)]
        sigvals = SignalList(sarr)(t)

        t_in_frame_actual = Array(
            np.diag(np.exp(-t * evals))
            @ (np.tensordot(sigvals, paulis_in_frame_basis, axes=1) + extra_in_basis - diafarr)
            @ np.diag(np.exp(t * evals))
        )

        state = Array([0.3, 0.1])
        state_in_frame_basis = np.conjugate(np.transpose(evect)) @ state

        gm1 = GeneratorModel(paulis, signals=sarr, rotating_frame=farr, static_operator=extra)
        self.assertAllClose(gm1(t, in_frame_basis=True), t_in_frame_actual)
        self.assertAllClose(
            gm1(t, state_in_frame_basis, in_frame_basis=True),
            t_in_frame_actual @ state_in_frame_basis,
        )

        t_not_in_frame_actual = Array(
            expm(np.array(-t * farr))
            @ (np.tensordot(sigvals, paulis, axes=1) + extra - farr)
            @ expm(np.array(t * farr))
        )

        gm2 = GeneratorModel(paulis, signals=sarr, rotating_frame=farr, static_operator=extra)
        self.assertAllClose(gm2(t, in_frame_basis=False), t_not_in_frame_actual)
        self.assertAllClose(gm1(t, state, in_frame_basis=False), t_not_in_frame_actual @ state)

        # now, remove the frame
        gm1.rotating_frame = None
        gm2.rotating_frame = None

        t_expected = np.tensordot(sigvals, paulis, axes=1) + extra

        state_in_frame_basis = state

        self.assertAllClose(gm1.get_operators(True), gm1.get_operators(True))
        self.assertAllClose(gm1.get_static_operator(True), gm1.get_static_operator(False))

        self.assertAllClose(gm1(t, in_frame_basis=True), t_expected)
        self.assertAllClose(gm1(t, state, in_frame_basis=True), t_expected @ state_in_frame_basis)
        self.assertAllClose(gm2(t, in_frame_basis=False), t_expected)
        self.assertAllClose(gm2(t, state, in_frame_basis=False), t_expected @ state_in_frame_basis)

    def test_evaluate_rhs_vectorized_pseudorandom(self):
        """Test for whether evaluating a model at m different
        states, each with an n-length statevector, by passing
        an (m,n) Array provides the same value as passing each
        (n) Array individually."""
        rand.seed(9231)
        n = 8
        k = 4
        m = 2
        t = rand.rand()
        sig_list = SignalList([Signal(rand.rand(), rand.rand()) for j in range(k)])
        normal_states = rand.uniform(-1, 1, (n))
        vectorized_states = rand.uniform(-1, 1, (n, m))

        operators = rand.uniform(-1, 1, (k, n, n))

        gm = GeneratorModel(operators, static_operator=None, signals=sig_list)
        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                gm.evaluate_rhs(t, vectorized_states)[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        farr = rand.uniform(-1, 1, (n, n))
        farr = farr - np.conjugate(np.transpose(farr))

        gm.rotating_frame = farr

        self.assertTrue(gm.evaluate_rhs(t, normal_states).shape == (n,))
        self.assertTrue(gm.evaluate_rhs(t, vectorized_states).shape == (n, m))
        vectorized_result = gm.evaluate_rhs(t, vectorized_states)
        for i in range(m):
            self.assertAllClose(
                vectorized_result[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i]),
            )

        vectorized_result = gm.evaluate_rhs(t, vectorized_states, in_frame_basis=True)

        self.assertTrue(gm.evaluate_rhs(t, normal_states, in_frame_basis=True).shape == (n,))
        self.assertTrue(vectorized_result.shape == (n, m))
        for i in range(m):
            self.assertAllClose(
                vectorized_result[:, i],
                gm.evaluate_rhs(t, vectorized_states[:, i], in_frame_basis=True),
            )


class TestGeneratorModelJax(TestGeneratorModel, TestJaxBase):
    """Jax version of TestGeneratorModel tests.

    Note: This class contains more tests due to inheritance.
    """

    def test_jitable_funcs(self):
        """Tests whether all functions are jitable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        self.jit_wrap(self.basic_model.evaluate)(1)
        self.jit_wrap(self.basic_model.evaluate_rhs)(1, Array(np.array([0.2, 0.4])))

        self.basic_model.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_wrap(self.basic_model.evaluate)(1)
        self.jit_wrap(self.basic_model.evaluate_rhs)(1, Array(np.array([0.2, 0.4])))

        self.basic_model.rotating_frame = None

    def test_gradable_funcs(self):
        """Tests whether all functions are gradable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        self.jit_grad_wrap(self.basic_model.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_model.evaluate_rhs)(1.0, Array(np.array([0.2, 0.4])))

        self.basic_model.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_grad_wrap(self.basic_model.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_model.evaluate_rhs)(1.0, Array(np.array([0.2, 0.4])))

        self.basic_model.rotating_frame = None

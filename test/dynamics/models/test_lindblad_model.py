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
# pylint: disable=invalid-name,redundant-keyword-arg

"""Tests for qiskit_dynamics.models.lindblad_models.py. Most
of the actual calculation checking is handled at the level of a
models.operator_collection.DenseLindbladOperatorCollection test."""

import numpy as np

from scipy.linalg import expm

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.dispatch.dispatch import Dispatch
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestLindbladModelValidation(QiskitDynamicsTestCase):
    """Test validation handling of LindbladModel."""

    def test_operators_not_hermitian(self):
        """Test raising error if hamiltonian_operators are not Hermitian."""

        hamiltonian_operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]

        with self.assertRaises(QiskitError) as qe:
            LindbladModel(hamiltonian_operators=hamiltonian_operators)
        self.assertTrue("hamiltonian_operators must be Hermitian." in str(qe.exception))

    def test_static_operator_not_hermitian(self):
        """Test raising error if static_hamiltonian is not Hermitian."""

        static_hamiltonian = np.array([[0.0, 1.0], [0.0, 0.0]])
        hamiltonian_operators = [np.array([[0.0, 1.0], [1.0, 0.0]])]

        with self.assertRaises(QiskitError) as qe:
            LindbladModel(
                hamiltonian_operators=hamiltonian_operators, static_hamiltonian=static_hamiltonian
            )
        self.assertTrue("static_hamiltonian must be Hermitian." in str(qe.exception))

    def test_validate_false(self):
        """Verify setting validate=False avoids error raising."""

        lindblad_model = LindbladModel(
            hamiltonian_operators=[np.array([[0.0, 1.0], [0.0, 0.0]])],
            hamiltonian_signals=[1.0],
            validate=False,
        )

        self.assertAllClose(lindblad_model(1.0, np.eye(2)), np.zeros(2))


class TestLindbladModel(QiskitDynamicsTestCase):
    """Tests for LindbladModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        # define a basic hamiltonian
        w = 2.0
        r = 0.5
        ham_operators = [2 * np.pi * self.Z / 2, 2 * np.pi * r * self.X / 2]
        ham_signals = [w, Signal(1.0, w)]

        self.w = w
        self.r = r

        dissipator_operators = Array([[[0.0, 0.0], [1.0, 0.0]]])

        self.basic_lindblad = LindbladModel(
            hamiltonian_operators=ham_operators,
            hamiltonian_signals=ham_signals,
            dissipator_operators=dissipator_operators,
        )

    def test_basic_lindblad_lmult(self):
        """Test lmult method of Lindblad generator OperatorModel."""
        A = Array([[1.0, 2.0], [3.0, 4.0]])

        t = 1.123
        ham = (
            2 * np.pi * self.w * self.Z.data / 2
            + 2 * np.pi * self.r * np.cos(2 * np.pi * self.w * t) * self.X.data / 2
        )
        sm = Array([[0.0, 0.0], [1.0, 0.0]])

        expected = self._evaluate_lindblad_rhs(A, ham, [sm])
        value = self.basic_lindblad(t, A)
        self.assertAllClose(expected, value)

    # pylint: disable=no-self-use,too-many-arguments
    def _evaluate_lindblad_rhs(
        self, A, ham, dissipators=None, dissipator_coeffs=None, frame_op=None, t=0.0
    ):
        """Evaluate the Lindblad equation

        frame_op assumed anti-Hermitian

        Note: here we force everything into numpy arrays as these parts of
        the test are just for confirmation
        """
        # if a frame operator is given, transform the model pieces into
        # the frame
        if frame_op is not None:
            frame_op = np.array(frame_op)
            U = expm(-t * frame_op)
            Uadj = U.conj().transpose()

            ham = U @ ham @ Uadj - 1j * frame_op

            if dissipators is not None:
                dissipators = np.array(dissipators)
                dissipators = [U @ D @ Uadj for D in dissipators]

        ham = np.array(ham)
        A = np.array(A)
        ham_part = -1j * (ham @ A - A @ ham)

        if dissipators is None:
            return ham_part

        dissipators = np.array(dissipators)

        if dissipator_coeffs is None:
            dissipator_coeffs = np.ones(len(dissipators))
        else:
            dissipator_coeffs = np.array(dissipator_coeffs)

        diss_part = np.zeros_like(A)
        for c, D in zip(dissipator_coeffs, dissipators):
            Dadj = D.conj().transpose()
            DadjD = Dadj @ D
            diss_part += c * (D @ A @ Dadj - 0.5 * (DadjD @ A + A @ DadjD))

        return ham_part + diss_part

        # pylint: disable=too-many-locals

    def test_lindblad_pseudorandom(self):
        """Test LindbladModel with structureless
        pseudorandom model parameters.
        """
        rng = np.random.default_rng(9848)
        dim = 10
        num_ham = 4
        num_diss = 3

        b = 1.0  # bound on size of random terms

        # generate random hamiltonian
        randoperators = rng.uniform(low=-b, high=b, size=(num_ham, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_ham, dim, dim)
        )
        rand_ham_ops = Array(randoperators + randoperators.conj().transpose([0, 2, 1]))

        # generate random hamiltonian coefficients
        rand_ham_coeffs = rng.uniform(low=-b, high=b, size=(num_ham)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_ham)
        )
        rand_ham_carriers = Array(rng.uniform(low=-b, high=b, size=(num_ham)))
        rand_ham_phases = Array(rng.uniform(low=-b, high=b, size=(num_ham)))

        ham_sigs = []
        for coeff, freq, phase in zip(rand_ham_coeffs, rand_ham_carriers, rand_ham_phases):
            ham_sigs.append(Signal(coeff, freq, phase))

        ham_sigs = SignalList(ham_sigs)

        # generate random dissipators
        rand_diss = Array(
            rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(num_diss, dim, dim))
        )

        # random dissipator coefficients
        rand_diss_coeffs = rng.uniform(low=-b, high=b, size=(num_diss)) + 1j * rng.uniform(
            low=-b, high=b, size=(num_diss)
        )
        rand_diss_carriers = Array(rng.uniform(low=-b, high=b, size=(num_diss)))
        rand_diss_phases = Array(rng.uniform(low=-b, high=b, size=(num_diss)))

        diss_sigs = []
        for coeff, freq, phase in zip(rand_diss_coeffs, rand_diss_carriers, rand_diss_phases):
            diss_sigs.append(Signal(coeff, freq, phase))

        diss_sigs = SignalList(diss_sigs)

        # random anti-hermitian frame operator
        rand_op = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )
        frame_op = Array(rand_op - rand_op.conj().transpose())
        evect = -1j * np.linalg.eigh(1j * frame_op)[1]
        f = lambda x: evect.T.conj() @ x @ evect

        lindblad_frame_op = frame_op

        # construct model
        hamiltonian = HamiltonianModel(operators=rand_ham_ops, signals=ham_sigs)
        lindblad_model = LindbladModel.from_hamiltonian(
            hamiltonian=hamiltonian, dissipator_operators=rand_diss, dissipator_signals=diss_sigs
        )
        lindblad_model.rotating_frame = lindblad_frame_op

        A = Array(
            rng.uniform(low=-b, high=b, size=(dim, dim))
            + 1j * rng.uniform(low=-b, high=b, size=(dim, dim))
        )

        t = rng.uniform(low=-b, high=b)
        value = lindblad_model(t, A, in_frame_basis=False)
        value_in_frame_basis = lindblad_model(
            t, lindblad_model.rotating_frame.operator_into_frame_basis(A), in_frame_basis=True
        )

        ham_coeffs = np.real(
            rand_ham_coeffs * np.exp(1j * 2 * np.pi * rand_ham_carriers * t + 1j * rand_ham_phases)
        )
        ham = np.tensordot(ham_coeffs, rand_ham_ops, axes=1)

        diss_coeffs = np.real(
            rand_diss_coeffs
            * np.exp(1j * 2 * np.pi * rand_diss_carriers * t + 1j * rand_diss_phases)
        )

        expected = self._evaluate_lindblad_rhs(
            A, ham, dissipators=rand_diss, dissipator_coeffs=diss_coeffs, frame_op=frame_op, t=t
        )

        self.assertAllClose(ham_coeffs, ham_sigs(t))
        self.assertAllClose(diss_coeffs, diss_sigs(t))
        self.assertAllClose(f(rand_diss), lindblad_model._dissipator_operators)
        self.assertAllClose(f(rand_ham_ops), lindblad_model._hamiltonian_operators)
        self.assertAllClose(
            f(-1j * frame_op), lindblad_model.get_static_hamiltonian(in_frame_basis=True)
        )
        self.assertAllClose(
            -1j * frame_op, lindblad_model.get_static_hamiltonian(in_frame_basis=False)
        )
        self.assertAllClose(
            f(-1j * frame_op), lindblad_model._operator_collection.static_hamiltonian
        )
        self.assertAllClose(expected, value)

        lindblad_model.evaluation_mode = "dense_vectorized"
        vectorized_value = lindblad_model.evaluate_rhs(
            t, A.flatten(order="F"), in_frame_basis=False
        ).reshape((dim, dim), order="F")
        self.assertAllClose(value, vectorized_value)

        vec_gen = lindblad_model.evaluate(t, in_frame_basis=False)
        vectorized_value_lmult = (vec_gen @ A.flatten(order="F")).reshape((dim, dim), order="F")
        self.assertAllClose(value, vectorized_value_lmult)

        rho_in_frame_basis = lindblad_model.rotating_frame.operator_into_frame_basis(A)
        vectorized_value_lmult_fb = (
            lindblad_model.evaluate(t, in_frame_basis=True) @ rho_in_frame_basis.flatten(order="F")
        ).reshape((dim, dim), order="F")
        self.assertAllClose(vectorized_value_lmult_fb, value_in_frame_basis)

        if Dispatch.DEFAULT_BACKEND != "jax":
            lindblad_model.evaluation_mode = "sparse"
            sparse_value = lindblad_model.evaluate_rhs(t, A, in_frame_basis=False)
            self.assertAllClose(value, sparse_value)

            lindblad_model.evaluation_mode = "sparse_vectorized"
            sparse_vectorized_value = lindblad_model.evaluate_rhs(
                t, A.flatten(order="F"), in_frame_basis=False
            ).reshape((dim, dim), order="F")
            self.assertAllClose(value, sparse_vectorized_value)

            sparse_vec_gen = lindblad_model.evaluate(t, in_frame_basis=False)
            sparse_vectorized_value_lmult = (sparse_vec_gen @ A.flatten(order="F")).reshape(
                (dim, dim), order="F"
            )
            self.assertAllClose(sparse_vectorized_value_lmult, value)


class TestLindbladModelJax(TestLindbladModel, TestJaxBase):
    """Jax version of TestLindbladModel tests.

    Note: This class has contains more tests due to inheritance.
    """

    def test_jitable_funcs(self):
        """Tests whether all functions are jitable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        self.jit_wrap(self.basic_lindblad.evaluate_rhs)(
            1.0, Array(np.array([[0.2, 0.4], [0.6, 0.8]]))
        )

        self.basic_lindblad.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_wrap(self.basic_lindblad.evaluate_rhs)(
            1.0, Array(np.array([[0.2, 0.4], [0.6, 0.8]]))
        )

        self.basic_lindblad.rotating_frame = None

        self.basic_lindblad.evaluation_mode = "dense_vectorized"

        self.jit_wrap(self.basic_lindblad.evaluate)(1.0)
        self.jit_wrap(self.basic_lindblad.evaluate_rhs)(1.0, Array(np.array([0.2, 0.4, 0.6, 0.8])))

        self.basic_lindblad.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_wrap(self.basic_lindblad.evaluate)(1.0)
        self.jit_wrap(self.basic_lindblad.evaluate_rhs)(1.0, Array(np.array([0.2, 0.4, 0.6, 0.8])))

        self.basic_lindblad.rotating_frame = None

    def test_gradable_funcs(self):
        """Tests whether all functions are gradable.
        Checks if having a frame makes a difference, as well as
        all jax-compatible evaluation_modes."""
        self.jit_grad_wrap(self.basic_lindblad.evaluate_rhs)(
            1.0, Array(np.array([[0.2, 0.4], [0.6, 0.8]]))
        )

        self.basic_lindblad.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_grad_wrap(self.basic_lindblad.evaluate_rhs)(
            1.0, Array(np.array([[0.2, 0.4], [0.6, 0.8]]))
        )

        self.basic_lindblad.rotating_frame = None

        self.basic_lindblad.evaluation_mode = "dense_vectorized"

        self.jit_grad_wrap(self.basic_lindblad.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_lindblad.evaluate_rhs)(
            1.0, Array(np.array([0.2, 0.4, 0.6, 0.8]))
        )

        self.basic_lindblad.rotating_frame = Array(np.array([[3j, 2j], [2j, 0]]))

        self.jit_grad_wrap(self.basic_lindblad.evaluate)(1.0)
        self.jit_grad_wrap(self.basic_lindblad.evaluate_rhs)(
            1.0, Array(np.array([0.2, 0.4, 0.6, 0.8]))
        )

        self.basic_lindblad.rotating_frame = None


def get_const_func(const):
    """Helper function for defining a constant function."""
    # pylint: disable=unused-argument
    def env(t):
        return const

    return env

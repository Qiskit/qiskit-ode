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

"""tests for qiskit_dynamics.models.rotating_wave_approximation"""

import numpy as np
from qiskit_dynamics.dispatch.array import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.models import (
    GeneratorModel,
    rotating_wave_approximation,
    LindbladModel,
    RotatingFrame,
)
from qiskit_dynamics.models.rotating_wave_approximation import get_rwa_operators
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestRotatingWave(QiskitDynamicsTestCase):
    """Tests the rotating_wave_approximation function."""

    def setUp(self):
        pass

    def test_generator_model_rwa_with_frame(self):
        """Tests whether RWA functionality mapping GeneratorModels -> GeneratorModels
        is correct by comparing it to an analytically calculable case."""
        frame_op = np.array([[4j, 1j], [1j, 2j]])
        sigs = SignalList([Signal(1 + 1j * k, k / 3, np.pi / 2 * k) for k in range(1, 4)])
        self.assertAllClose(sigs(0), np.array([-1, -1, 3]))
        ops = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

        GM = GeneratorModel(ops, static_operator=None, signals=sigs, rotating_frame=frame_op)
        self.assertAllClose(GM(0), np.array([[3 - 4j, -1], [-1 - 2j, -3 - 2j]]))
        self.assertAllClose(
            GM(1),
            [[-0.552558 - 4j, 3.18653 - 1.43887j], [3.18653 - 0.561126j, 0.552558 - 2j]],
            rtol=1e-5,
        )
        GM2 = rotating_wave_approximation(GM, 1 / 2 / np.pi)
        self.assertAllClose(GM2.signals(0), [-1, -1, 3, 1, -2, -1])
        self.assertAllClose(
            GM2(0), [[0.25 - 4.0j, -0.25 - 0.646447j], [-0.25 - 1.35355j, -0.25 - 2.0j]], rtol=1e-5
        )
        self.assertAllClose(
            GM2(1),
            [
                [0.0181527 - 4.0j, -0.0181527 - 0.500659j],
                [-0.0181527 - 1.49934j, -0.0181527 - 2.0j],
            ],
            rtol=1e-5,
        )

    def test_no_rotating_frame(self):
        """Tests whether RWA works in the absence of a rotating frame"""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
        dft = Array(np.ones((2, 2)))
        GM = GeneratorModel(ops, signals=sigs, static_operator=dft, rotating_frame=None)
        GMP = rotating_wave_approximation(GM, 2)
        self.assertAllClose(GMP.get_static_operator(True), Array(np.ones((2, 2))))
        post_rwa_ops = Array(np.array([1, 0, 1, 0, 0, 0, 0, 0]).reshape((8, 1, 1))) * Array(
            np.ones((8, 2, 2))
        )
        self.assertAllClose(GMP.get_operators(True), post_rwa_ops)

    def test_signal_translator_generator_model(self):
        """Tests signal translation from pre-RWA to post-RWA through
        rotating_wave_approximation.get_rwa_signals when passed a
        GeneratorModel."""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
        dft = Array(np.ones((2, 2)))
        GM = GeneratorModel(ops, signals=sigs, static_operator=dft, rotating_frame=None)
        f = rotating_wave_approximation(GM, 100, return_signal_map=True)[1]
        vals = f(sigs).complex_value(3)
        self.assertAllClose(vals[:4], GM.signals.complex_value(3))
        s_prime = [
            Signal(1, 0, -np.pi / 2),
            Signal(1, -3, -np.pi / 2),
            Signal(1, 1, -np.pi / 2),
            Signal(1, 3, -np.pi / 2),
        ]
        self.assertAllClose(vals[4:], SignalList(s_prime).complex_value(3))
        self.assertTrue(f(None) is None)

    def test_signal_translator_lindblad_model(self):
        """Like test_signal_translator_generator_model, but for LindbladModels."""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
        s_prime = [
            Signal(1, 0, -np.pi / 2),
            Signal(1, -3, -np.pi / 2),
            Signal(1, 1, -np.pi / 2),
            Signal(1, 3, -np.pi / 2),
        ]
        dft = Array(np.ones((2, 2)))
        LM = LindbladModel(ops, sigs, ops, sigs, dft)
        f = rotating_wave_approximation(LM, 100, return_signal_map=True)[1]
        rwa_ham_sig, rwa_dis_sig = f(sigs, sigs)
        self.assertAllClose(rwa_ham_sig.complex_value(2)[:4], SignalList(sigs).complex_value(2))
        self.assertAllClose(rwa_dis_sig.complex_value(2)[:4], SignalList(sigs).complex_value(2))
        self.assertAllClose(rwa_ham_sig.complex_value(2)[4:], SignalList(s_prime).complex_value(2))
        self.assertAllClose(rwa_dis_sig.complex_value(2)[4:], SignalList(s_prime).complex_value(2))

        self.assertTrue(f(None, None) == (None, None))

    def test_rwa_operators(self):
        """Tests get_rwa_operators using pseudorandom numbers."""
        np.random.seed(123098123)
        r = lambda *args: Array(np.random.uniform(-1, 1, args))
        rj = lambda *args: r(*args) + 1j * r(*args)
        ops = rj(4, 3, 3)
        carrier_freqs = r(4)
        cutoff_freq = 0.3
        sigs = SignalList([Signal(r(), freq, r()) for freq in carrier_freqs])

        frame_op = r(3, 3)
        frame_op = frame_op - frame_op.conj().T
        rotating_frame = RotatingFrame(frame_op)

        ops_in_fb = rotating_frame.operator_into_frame_basis(ops)
        diag = rotating_frame.frame_diag
        diff_matrix = np.broadcast_to(diag, (3, 3)) - np.broadcast_to(diag, (3, 3)).T
        frame_freqs = diff_matrix.imag / (2 * np.pi)

        rwa_ops = get_rwa_operators(ops_in_fb, sigs, rotating_frame, frame_freqs, cutoff_freq)

        carrier_freqs = carrier_freqs.reshape(4, 1, 1)
        frame_freqs = frame_freqs.reshape(1, 3, 3)
        G_p = ops_in_fb * (np.abs(carrier_freqs + frame_freqs) < cutoff_freq).astype(int)
        G_m = ops_in_fb * (np.abs(-carrier_freqs + frame_freqs) < cutoff_freq).astype(int)
        self.assertAllClose(
            rwa_ops[:4], rotating_frame.operator_out_of_frame_basis((G_p + G_m) / 2)
        )
        self.assertAllClose(
            rwa_ops[4:], rotating_frame.operator_out_of_frame_basis(1j * (G_p - G_m) / 2)
        )


class TestRotatingWaveJax(TestRotatingWave, TestJaxBase):
    """Jax version of TestRotatingWave tests.

    Note: This class has no body but contains tests due to inheritance.
    """

    def test_jitable_gradable_rwa(self):
        """Tests whether a function involving the RWA is jitable and gradable."""

        def _simple_function_using_rwa(w):
            """Simple function that involves taking the rotating wave approximation."""
            sigs = [Signal(1, 0), Signal(lambda t: w * t, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
            ops = Array(np.ones((4, 2, 2)))
            dft = Array(np.ones((2, 2)))
            GM = GeneratorModel(ops, signals=sigs, static_operator=dft, rotating_frame=None)
            return rotating_wave_approximation(GM, 2)(2)

        self.jit_wrap(_simple_function_using_rwa)(1.0)
        self.jit_grad_wrap(_simple_function_using_rwa)(1.0)

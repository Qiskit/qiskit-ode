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
# pylint: disable=invalid-name,broad-except

"""Tests for solve_ode."""

import numpy as np
from scipy.linalg import expm

from qiskit import QiskitError
from qiskit_dynamics import solve_ode
from qiskit_dynamics.models import GeneratorModel
from qiskit_dynamics.signals import Signal, DiscreteSignal
from qiskit_dynamics.dispatch import Array

from ..common import QiskitDynamicsTestCase, TestJaxBase


class Testsolve_ode_exceptions(QiskitDynamicsTestCase):
    """Test exceptions of solve_ode."""

    def test_method_does_not_exist(self):
        """Test method does not exist exception."""

        with self.assertRaises(QiskitError) as qe:
            solve_ode(lambda t, y: y, t_span=[0.0, 1.0], y0=np.array([1.0]), method="notamethod")

        self.assertTrue("not supported" in str(qe.exception))


# pylint: disable=too-many-instance-attributes
class Testsolve_ode_Base(QiskitDynamicsTestCase):
    """Some reusable routines for testing basic solving functionality."""

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

        def rhs(t, y):
            return generator(t) @ y

        self.basic_generator = generator
        self.basic_rhs = rhs

        # define simple model
        self.w = 2.0
        self.r = 0.1
        signals = [self.w, Signal(lambda t: 1.0, self.w)]
        operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * self.r * self.X / 2]
        self.basic_model = GeneratorModel(operators=operators, signals=signals)

    def test_solve_ode_w_GeneratorModel(self):
        """Test solve on a GeneratorModel."""

        results = solve_ode(
            self.basic_model,
            y0=Array([0.0, 1.0], dtype=complex),
            t_span=[0, 1 / self.r],
            rtol=1e-9,
            atol=1e-9,
        )
        yf = results.y[-1]

        self.assertTrue(np.abs(yf[0]) ** 2 > 0.999)

    def _variable_step_method_standard_tests(self, method):
        """tests to run on a variable step solver."""

        # simple case with basic_rhs
        results = solve_ode(
            self.basic_rhs, t_span=self.t_span, y0=self.y0, method=method, atol=1e-10, rtol=1e-10
        )

        expected = expm(-1j * np.pi * self.X.data)

        self.assertAllClose(results.y[-1], expected)

        # non-LMDE example
        # pylint: disable=unused-argument
        def quad_rhs(t, y):
            return Array([t ** 2], dtype=float)

        results = solve_ode(
            quad_rhs, t_span=[0.0, 1.0], y0=Array([0.0]), method=method, atol=1e-10, rtol=1e-10
        )
        expected = Array([1.0 / 3])
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

        results = solve_ode(model, t_span=[0, 0.5], y0=y0, method=method, atol=1e-8, rtol=1e-8)
        yf = model.rotating_frame.state_out_of_frame(0.5, results.y[-1])

        # simulate directly out of frame
        def rhs(t, y):
            return (static_operator + sig(t) * operators[0]) @ y

        results2 = solve_ode(rhs, t_span=[0, 0.5], y0=y0, method=method, atol=1e-8, rtol=1e-8)
        # check consistency - this is relatively low tolerance due to the solver tolerance
        self.assertAllClose(yf, results2.y[-1], atol=1e-5, rtol=1e-5)


class Testsolve_ode_numpy(Testsolve_ode_Base):
    """Basic tests for `numpy`-based ODE solver methods."""

    def test_standard_problems_solve_ivp(self):
        """Run standard tests for variable step methods in `solve_ivp`."""

        self._variable_step_method_standard_tests("RK45")
        self._variable_step_method_standard_tests("RK23")
        self._variable_step_method_standard_tests("BDF")
        self._variable_step_method_standard_tests("DOP853")


class Testsolve_ode_jax(Testsolve_ode_Base, TestJaxBase):
    """Basic tests for jax ODE solvers."""

    def test_standard_problems_jax(self):
        """Run standard tests for variable step `jax` methods."""
        self._variable_step_method_standard_tests("jax_odeint")

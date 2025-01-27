# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
Integration tests that ensure this module interacts properly with qiskit-experiments.
"""

import numpy as np

from qiskit_dynamics import pulse
from qiskit.circuit import Parameter
from qiskit.circuit.library import XGate, SXGate
from qiskit.transpiler import Target
from qiskit.providers.backend import QubitProperties
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.library.calibration.rough_frequency import RoughFrequencyCal

from qiskit_dynamics import Solver, DynamicsBackend
from ..common import JAXTestBase


class TestExperimentsIntegration(JAXTestBase):
    """Test class for verifying correct integration with qiskit-experiments. Set to use JAX for
    speed.
    """

    def setUp(self):
        """Build simple simulator for multiple tests."""

        # build the solver
        solver = Solver(
            static_hamiltonian=2 * np.pi * 5.0 * np.array([[-1.0, 0.0], [0.0, 1.0]]) / 2,
            hamiltonian_operators=[2 * np.pi * np.array([[0.0, 1.0], [1.0, 0.0]]) / 2],
            rotating_frame=2 * np.pi * 5.0 * np.array([[-1.0, 0.0], [0.0, 1.0]]) / 2,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": 5.0},
            dt=1.0 / 4.5,
            array_library="jax",
        )

        # build target gate definitions
        target = Target()
        target.qubit_properties = [QubitProperties(frequency=5.0)]

        target.add_instruction(XGate())
        target.add_instruction(SXGate())

        self.simple_backend = DynamicsBackend(
            solver=solver,
            target=target,
            solver_options={"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8},
        )

    def test_RoughFrequencyCal_calibration(self):
        """Test RoughFrequencyCal outputs a sensible answer."""

        cals = Calibrations()

        dur = Parameter("dur")
        sigma = Parameter("sigma")
        drive = pulse.DriveChannel(Parameter("ch0"))

        # Define and add template schedules.
        with pulse.build(name="x") as x:
            pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("beta")), drive)

        with pulse.build(name="sx") as sx:
            pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("beta")), drive)

        cals.add_schedule(x, num_qubits=1)
        cals.add_schedule(sx, num_qubits=1)

        for sched in ["x", "sx"]:
            cals.add_parameter_value(80, "sigma", schedule=sched)
            cals.add_parameter_value(0.5, "beta", schedule=sched)
            cals.add_parameter_value(320, "dur", schedule=sched)
            cals.add_parameter_value(0.5, "amp", schedule=sched)

        freq_estimate = 5.005
        frequencies = np.linspace(freq_estimate - 1e-2, freq_estimate + 1e-2, 51)

        self.simple_backend.set_options(seed_simulator=5243234)
        spec = RoughFrequencyCal([0], cals, frequencies, backend=self.simple_backend)
        spec.set_experiment_options(amp=0.05, sigma=80, duration=320)

        spec_data = spec.run().block_for_results()
        freq_fit = spec_data.analysis_results("f01").value.nominal_value

        self.assertTrue(np.abs(freq_fit - 5.0) < 1e-1)

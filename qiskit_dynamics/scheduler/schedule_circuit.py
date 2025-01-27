# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QuantumCircuit to Pulse scheduler."""
import logging
from typing import Optional, Union, List
from time import time
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.utils import parallel_map

from qiskit_dynamics.pulse import InstructionScheduleMap
from qiskit_dynamics.pulse.schedule import Schedule
from qiskit_dynamics.scheduler.config import ScheduleConfig
from qiskit_dynamics.scheduler.methods import as_soon_as_possible, as_late_as_possible
from qiskit.providers import BackendV1, BackendV2, Backend

logger = logging.getLogger(__name__)


def _log_schedule_time(start_time, end_time):
    log_msg = f"Total Scheduling Time - {((end_time - start_time) * 1000):.5f} (ms)"
    logger.info(log_msg)


def schedule(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Optional[Backend] = None,
    inst_map: Optional[InstructionScheduleMap] = None,
    meas_map: Optional[List[List[int]]] = None,
    dt: Optional[float] = None,
    method: Optional[Union[str, List[str]]] = None,
) -> Union[Schedule, List[Schedule]]:
    """
    Schedule a circuit to a pulse ``Schedule``, using the backend, according to any specified
    methods. Supported methods are documented in :py:mod:`qiskit.scheduler.schedule_circuit`.

    Args:
        circuits: The quantum circuit or circuits to translate
        backend: A backend instance, which contains hardware-specific data required for scheduling
        inst_map: Mapping of circuit operations to pulse schedules. If ``None``, defaults to the
                  ``backend``\'s ``instruction_schedule_map``
        meas_map: List of sets of qubits that must be measured together. If ``None``, defaults to
                  the ``backend``\'s ``meas_map``
        dt: The output sample rate of backend control electronics. For scheduled circuits
            which contain time information, dt is required. If not provided, it will be
            obtained from the backend configuration
        method: Optionally specify a particular scheduling method

    Returns:
        A pulse ``Schedule`` that implements the input circuit

    Raises:
        QiskitError: If ``inst_map`` and ``meas_map`` are not passed and ``backend`` is not passed
    """
    arg_circuits_list = isinstance(circuits, list)
    start_time = time()
    if backend and getattr(backend, "version", 0) > 1:
        if inst_map is None:
            inst_map = backend.instruction_schedule_map
        if meas_map is None:
            meas_map = backend.meas_map
        if dt is None:
            dt = backend.dt
    else:
        if inst_map is None:
            if backend is None:
                raise QiskitError(
                    "Must supply either a backend or InstructionScheduleMap for scheduling passes."
                )
            defaults = backend.defaults()
            if defaults is None:
                raise QiskitError(
                    "The backend defaults are unavailable. The backend may not support pulse."
                )
            inst_map = defaults.instruction_schedule_map
        if meas_map is None:
            if backend is None:
                raise QiskitError(
                    "Must supply either a backend or a meas_map for scheduling passes."
                )
            meas_map = backend.configuration().meas_map
        if dt is None:
            if backend is not None:
                dt = backend.configuration().dt

    schedule_config = ScheduleConfig(inst_map=inst_map, meas_map=meas_map, dt=dt)
    circuits = circuits if isinstance(circuits, list) else [circuits]
    schedules = parallel_map(schedule_circuit, circuits, (schedule_config, method, backend))
    end_time = time()
    _log_schedule_time(start_time, end_time)
    if arg_circuits_list:
        return schedules
    else:
        return schedules[0]


def schedule_circuit(
    circuit: QuantumCircuit,
    schedule_config: ScheduleConfig,
    method: Optional[str] = None,
    backend: Optional[Union[BackendV1, BackendV2]] = None,
) -> Schedule:
    """
    Basic scheduling pass from a circuit to a pulse Schedule, using the backend. If no method is
    specified, then a basic, as late as possible scheduling pass is performed, i.e. pulses are
    scheduled to occur as late as possible.

    Supported methods:

        * ``'as_soon_as_possible'``: Schedule pulses greedily, as early as possible on a
          qubit resource. (alias: ``'asap'``)
        * ``'as_late_as_possible'``: Schedule pulses late-- keep qubits in the ground state when
          possible. (alias: ``'alap'``)

    Args:
        circuit: The quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.
        method: The scheduling pass method to use.
        backend: A backend used to build the Schedule, the backend could be BackendV1
                 or BackendV2.

    Returns:
        Schedule corresponding to the input circuit.

    Raises:
        QiskitError: If method isn't recognized.
    """
    methods = {
        "as_soon_as_possible": as_soon_as_possible,
        "asap": as_soon_as_possible,
        "as_late_as_possible": as_late_as_possible,
        "alap": as_late_as_possible,
    }
    if method is None:
        method = "as_late_as_possible"
    try:
        return methods[method](circuit, schedule_config, backend)
    except KeyError as ex:
        raise QiskitError(f"Scheduling method {method} isn't recognized.") from ex

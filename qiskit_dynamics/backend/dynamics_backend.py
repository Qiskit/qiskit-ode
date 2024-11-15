# -*- coding: utf-8 -*-

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
# pylint: disable=invalid-name, arguments-differ

"""
Pulse-enabled simulator backend.
"""

import datetime
import inspect
import uuid
import warnings

from typing import List, Optional, Union, Dict, Tuple
import copy
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import pulse
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.qobj.common import QobjHeader
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import Measure
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.providers.options import Options
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.models.backendconfiguration import PulseBackendConfiguration
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qiskit import QiskitError, QuantumCircuit
from qiskit import schedule as build_schedule
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.states.quantum_state import QuantumState


from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.arraylias.alias import ArrayLike
from qiskit_dynamics.solvers.solver_classes import Solver

from .dynamics_job import DynamicsJob
from .backend_utils import (
    _get_dressed_state_decomposition,
    _get_lab_frame_static_hamiltonian,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
    _get_iq_data,
)
from .backend_string_parser import parse_backend_hamiltonian_dict


class DynamicsBackend(BackendV2):
    r"""Pulse-level simulator backend.

    This class provides a :class:`~qiskit.providers.backend.BackendV2` interface wrapper around a
    :class:`.Solver` instance setup to simulate pulse schedules. The backend can be configured to
    take advantage of standard transpilation infrastructure to describe pulse-level simulations in
    terms of :class:`~qiskit.circuit.QuantumCircuit`\s. Results are returned as
    :class:`~qiskit.result.Result` instances.

    A minimal :class:`.DynamicsBackend` requires specifying only a :class:`.Solver` instance and a
    list of subsystem dimensions, indicating the subsystem decomposition of the model in
    :class:`.Solver`. For example, the following code builds a :class:`.DynamicsBackend` around a
    :class:`.Solver` and indicates that the system specified by the :class:`.Solver` decomposes as
    two ``3`` dimensional subsystems.

    .. code-block:: python

        backend = DynamicsBackend(
            solver=solver, subsystem_dims=[3, 3]
        )

    Without further configuration, the above ``backend`` can be used to simulate either
    :class:`~qiskit.pulse.Schedule` or :class:`~qiskit.pulse.ScheduleBlock` instances.

    Pulse-level simulations defined in terms of :class:`~qiskit.circuit.QuantumCircuit` instances
    can also be performed if each gate in the circuit has a corresponding pulse-level definition,
    either as an attached calibration, or as an instruction contained in ``backend.target``.

    Additionally, a :class:`.DynamicsBackend` can be instantiated from an existing backend using the
    :meth:`.DynamicsBackend.from_backend` method, utilizing the additional ``subsystem_list``
    argument to specify which qubits to include in the model:

    .. code-block:: python

        backend = DynamicsBackend.from_backend(backend, subsystem_list=[0, 1])


    **Supported options**

    The behaviour of the backend can be configured via the following options. These can either be
    passed as optional keyword arguments at construction, set with the
    :meth:`.DynamicsBackend.set_options` method after construction, or passed as runtime arguments
    to :meth:`.DynamicsBackend.run`.

    * ``shots``: Number of shots per experiment. Defaults to ``1024``.
    * ``solver``: The Qiskit Dynamics :class:`.Solver` instance used for simulation.
    * ``solver_options``: Dictionary containing optional kwargs for passing to :meth:`Solver.solve`,
      indicating solver methods and options. Defaults to the empty dictionary ``{}``.
    * ``subsystem_dims``: Dimensions of subsystems making up the system in ``solver``. Defaults to
      ``[solver.model.dim]``.
    * ``meas_map``: Measurement map. Defaults to ``[[idx] for idx in range(len(subsystem_dims))]``.
    * ``control_channel_map``: A dictionary mapping control channel labels to indices, to be used
      for control channel index lookup in the :meth:`DynamicsBackend.control_channel` method.
    * ``initial_state``: Initial state for simulation, either the string ``"ground_state"``,
      indicating that the ground state for the system Hamiltonian should be used, or an arbitrary
      ``Statevector`` or ``DensityMatrix``. Defaults to ``"ground_state"``.
    * ``normalize_states``: Boolean indicating whether to normalize states before computing outcome
      probabilities, and normalize probablities before sampling. Defaults to ``True``. Setting to
      ``False`` can result in errors if the solution tolerance results in probabilities with
      significant numerical deviation from a proper probability distribution.
    * ``meas_level``: Form of measurement output. Supported values are ``1`` and ``2``. ``1``
      returns IQ points and ``2`` returns counts. Defaults to ``meas_level == 2``.
    * ``meas_return``: Level of measurement data to return. For ``meas_level = 1`` ``"single"``
      returns output from every shot. ``"avg"`` returns average over shots of measurement output.
      Defaults to ``"avg"``.
    * ``iq_centers``: Centers for IQ distribution when using ``meas_level==1`` results. Must have
      type ``List[List[List[float, float]]]`` formatted as ``iq_centers[subsystem][level] = [I,
      Q]``. If ``None``, the ``iq_centers`` are dynamically generated to be equally spaced points on
      a unit circle with ground-state at ``(1, 0)``. The default is ``None``.
    * ``iq_width``: Standard deviation of IQ distribution around the centers for ``meas_level==1``.
      Must be a positive float. Defaults to ``0.2``.
    * ``max_outcome_level``: For ``meas_level == 2``, the maximum outcome for each subsystem. Values
      will be rounded down to be no larger than ``max_outcome_level``. Must be a positive integer or
      ``None``. If ``None``, no rounding occurs. Defaults to ``1``.
    * ``memory``: Boolean indicating whether to return a list of explicit measurement outcomes for
      every experimental shot. Defaults to ``True``.
    * ``seed_simulator``: Seed to use in random sampling. Defaults to ``None``.
    * ``experiment_result_function``: Function for computing the ``ExperimentResult`` for each
      simulated experiment. This option defaults to :func:`default_experiment_result_function`, and
      any other function set to this option must have the same signature. Note that the default
      utilizes various other options that control results computation, and hence changing it will
      impact the meaning of other options.
    * ``configuration``: A :class:`PulseBackendConfiguration` instance or ``None``. This option
      defaults to ``None``, and is not required for the functioning of this class, but is provided
      for compatibility. A set configuration will be returned by
      :meth:`DynamicsBackend.configuration()`.
    * ``defaults``: A :class:`PulseDefaults` instance or ``None``. This option defaults to ``None``,
      and is not required for the functioning of this class, but is provided for compatibility. A
      set defaults will be returned by :meth:`DynamicsBackend.defaults()`.
    """

    def __init__(
        self,
        solver: Solver,
        target: Optional[Target] = None,
        **options,
    ):
        """Instantiate with a :class:`.Solver` instance and additional options.

        Args:
            solver: Solver instance configured for pulse simulation.
            target: Target object.
            options: Additional configuration options for the simulator.

        Raises:
            QiskitError: If any instantiation arguments fail validation checks.
        """

        super().__init__(
            name="DynamicsBackend",
            description="Pulse enabled simulator backend.",
            backend_version="0.1",
        )

        # Dressed states of solver, will be calculated when solver option is set
        self._dressed_evals = None
        self._dressed_states = None
        self._dressed_states_adjoint = None

        # add subsystem_dims to options so set_options validation works
        if "subsystem_dims" not in options:
            options["subsystem_dims"] = [solver.model.dim]

        # Set simulator options
        self.set_options(solver=solver, **options)

        if self.options.meas_map is None:
            meas_map = [[idx] for idx in range(len(self.options.subsystem_dims))]
            self.set_options(meas_map=meas_map)

        # self._target = target or Target() doesn't work as bool(target) can be False
        if target is None:
            target = Target()
        else:
            target = copy.copy(target)

        # add default simulator measure instructions
        measure_properties = {}
        instruction_schedule_map = target.instruction_schedule_map()
        for qubit in range(len(self.options.subsystem_dims)):
            if not instruction_schedule_map.has(instruction="measure", qubits=qubit):
                with pulse.build() as meas_sched:
                    pulse.acquire(
                        duration=1, qubit_or_channel=qubit, register=pulse.MemorySlot(qubit)
                    )

                measure_properties[(qubit,)] = InstructionProperties(calibration=meas_sched)

        if bool(measure_properties):
            target.add_instruction(Measure(), measure_properties)

        target.dt = solver._dt
        target.num_qubits = len(self.options.subsystem_dims)

        self._target = target

    def _default_options(self):
        return Options(
            shots=1024,
            solver=None,
            solver_options={},
            subsystem_dims=None,
            meas_map=None,
            control_channel_map=None,
            normalize_states=True,
            initial_state="ground_state",
            meas_level=MeasLevel.CLASSIFIED,
            meas_return=MeasReturnType.AVERAGE,
            iq_centers=None,
            iq_width=0.2,
            max_outcome_level=1,
            memory=True,
            seed_simulator=None,
            experiment_result_function=default_experiment_result_function,
            configuration=None,
            defaults=None,
        )

    def set_options(self, **fields):
        """Set options for DynamicsBackend."""

        validate_subsystem_dims = False
        validate_iq_centers = False

        for key, value in fields.items():
            if not hasattr(self._options, key):
                raise AttributeError(f"Invalid option {key}")

            # validation checks
            if key == "initial_state":
                if value != "ground_state" and not isinstance(value, (Statevector, DensityMatrix)):
                    raise QiskitError(
                        'initial_state must be either "ground_state", or a Statevector or '
                        "DensityMatrix instance."
                    )
            elif key == "meas_level" and value not in [1, 2]:
                raise QiskitError("Only meas_level 1 and 2 are supported by DynamicsBackend.")
            elif key == "meas_return" and value not in ["single", "avg"]:
                raise QiskitError("meas_return must be either 'single' or 'avg'.")
            elif key == "max_outcome_level":
                if (value is not None) and (not isinstance(value, int) or (value <= 0)):
                    raise QiskitError("max_outcome_level must be a positive integer or None.")
            elif key == "experiment_result_function" and not callable(value):
                raise QiskitError("experiment_result_function must be callable.")
            elif key == "configuration" and not isinstance(value, PulseBackendConfiguration):
                raise QiskitError(
                    "configuration option must be an instance of PulseBackendConfiguration."
                )
            elif key == "defaults" and not isinstance(value, PulseDefaults):
                raise QiskitError("defaults option must be an instance of PulseDefaults.")
            elif key == "iq_width" and (not isinstance(value, float) or (value <= 0)):
                raise QiskitError("iq_width must be a positive float.")
            elif key == "iq_centers":
                if (value is not None) and not all(
                    (isinstance(level, List) and len(level) == 2)
                    for sub_system in value
                    for level in sub_system
                ):
                    raise QiskitError(
                        "The iq_centers option must be either None or of type "
                        "List[List[List[int, int]]], where the innermost list is the (I, Q) pair."
                    )
                validate_iq_centers = True
            elif key == "subsystem_dims":
                validate_subsystem_dims = True
                validate_iq_centers = True
            elif key == "solver":
                validate_subsystem_dims = True
            elif key == "control_channel_map":
                if value is not None:
                    if not isinstance(value, dict):
                        raise QiskitError(
                            "The control_channel_map option must either be None or a dictionary."
                        )
                    if not all(isinstance(x, int) for x in value.values()):
                        raise QiskitError("The control_channel_map values must be of type int.")

            # special setting routines
            if key == "solver":
                self._set_solver(value)
            else:
                self._options.update_options(**{key: value})

        # perform additional consistency validations if certain options were modified
        if (
            validate_subsystem_dims
            and np.prod(self._options.subsystem_dims) != self._options.solver.model.dim
        ):
            raise QiskitError(
                "DynamicsBackend options subsystem_dims and solver.model.dim are inconsistent."
            )

        if validate_iq_centers and (self._options.iq_centers is not None):
            if [
                len(sub_system) for sub_system in self._options.iq_centers
            ] != self._options.subsystem_dims:
                raise QiskitError(
                    """iq_centers option is not consistent with subsystem_dims. Must be None
                or of type List[List[List[int, int]]], where the outermost list is of length equal
                to the number of subsystems, and each inner list of length equal to the
                corresponding subsystem dimension."""
                )

    def _set_solver(self, solver):
        """Configure simulator based on provided solver."""
        if solver._dt is None:
            raise QiskitError(
                "Solver passed to DynamicsBackend is not configured for Pulse simulation."
            )

        self._options.update_options(solver=solver)
        # Get dressed states
        static_hamiltonian = _get_lab_frame_static_hamiltonian(solver.model)
        dressed_evals, dressed_states = _get_dressed_state_decomposition(static_hamiltonian)
        self._dressed_evals = dressed_evals
        self._dressed_states = dressed_states
        self._dressed_states_adjoint = self._dressed_states.conj().transpose()

    def solve(
        self,
        solve_input: List[Union[QuantumCircuit, Schedule, ScheduleBlock]],
        t_span: Optional[ArrayLike] = None,
        y0: Optional[Union[ArrayLike, QuantumState, BaseOperator]] = None,
        convert_results: Optional[bool] = True,
        validate: Optional[bool] = True,
    ) -> Union[OdeResult, List[OdeResult]]:
        """Simulate a list of :class:`~qiskit.circuit.QuantumCircuit`,
        :class:`~qiskit.pulse.Schedule`, or :class:`~qiskit.pulse.ScheduleBlock` instances and
        return the ``OdeResult``.

        This method is analogous to :meth:`.Solver.solve`, however it additionally utilizes
        transpilation and the backend configuration to convert
        :class:`~qiskit.circuit.QuantumCircuit` instances into pulse-level schedules for simulation.
        The options for the solver will be drawn from ``self.options.solver_options``, and if
        ``y0`` is not specified, it will be set from ``self.options.initial_state``.

        Args:
            t_span: Time interval to integrate over. Defaults to ``None``, in which case the
                    interval is set to ``[[0, input.duration] for input in solve_input]]``.
            y0: Initial state.
            solve_input: Time evolution of the system in terms of quantum circuits or qiskit
                         pulse schedules.
            convert_results: If ``True``, convert returned solver state results to the same class
                             as y0. If ``False``, states will be returned in the native array type
                             used by the specified solver method.
            validate: Whether or not to run validation checks on the input.

        Returns:
            OdeResult: object with formatted output types.
        """
        if validate:
            _validate_run_input(solve_input)
        schedules, _ = _to_schedule_list(solve_input, backend=self)

        # use default y0 if not given as parameter
        if y0 is None:
            y0 = self.options.initial_state
        if isinstance(y0, str) and y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])
        if t_span is None:
            t_span = [[0, sched.duration * self.dt] for sched in schedules]
        solver_results = self.options.solver.solve(
            t_span=t_span,
            y0=y0,
            signals=schedules,
            convert_results=convert_results,
            **self.options.solver_options,
        )
        return solver_results

    # pylint: disable=arguments-differ
    def run(
        self,
        run_input: List[Union[QuantumCircuit, Schedule, ScheduleBlock]],
        validate: Optional[bool] = True,
        **options,
    ) -> DynamicsJob:
        """Run a list of simulations.

        Args:
            run_input: A list of simulations, specified by ``QuantumCircuit``, ``Schedule``, or
                ``ScheduleBlock`` instances.
            validate: Whether or not to run validation checks on the input.
            **options: Additional run options to temporarily override current backend options.

        Returns:
            DynamicsJob object containing results and status.

        Raises:
            QiskitError: If invalid options are set.
        """

        if validate:
            _validate_run_input(run_input)

        # Configure run options for simulation
        if options:
            backend = copy.deepcopy(self)
            backend.set_options(**options)
        else:
            backend = self

        schedules, num_memory_slots_list = _to_schedule_list(run_input, backend=backend)

        # get the acquires sample times and subsystem measurement information
        (
            t_span,
            measurement_subsystems_list,
            memory_slot_indices_list,
        ) = _get_acquire_instruction_timings(
            schedules, backend.options.subsystem_dims, backend.options.solver._dt
        )

        # Build and submit job
        job_id = str(uuid.uuid4())
        dynamics_job = DynamicsJob(
            backend=backend,
            job_id=job_id,
            fn=backend._run,
            fn_kwargs={
                "t_span": t_span,
                "schedules": schedules,
                "measurement_subsystems_list": measurement_subsystems_list,
                "memory_slot_indices_list": memory_slot_indices_list,
                "num_memory_slots_list": num_memory_slots_list,
            },
        )
        dynamics_job.submit()

        return dynamics_job

    def _run(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        """Simulate a list of schedules."""

        # simulate all schedules
        y0 = self.options.initial_state
        if y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **self.options.solver_options
        )

        # compute results for each experiment
        experiment_names = [schedule.name for schedule in schedules]
        experiment_metadatas = [schedule.metadata for schedule in schedules]
        rng = np.random.default_rng(self.options.seed_simulator)
        experiment_results = []
        for (
            experiment_name,
            solver_result,
            measurement_subsystems,
            memory_slot_indices,
            num_memory_slots,
            experiment_metadata,
        ) in zip(
            experiment_names,
            solver_results,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
            experiment_metadatas,
        ):
            experiment_results.append(
                self.options.experiment_result_function(
                    experiment_name,
                    solver_result,
                    measurement_subsystems,
                    memory_slot_indices,
                    num_memory_slots,
                    self,
                    seed=rng.integers(low=0, high=9223372036854775807),
                    metadata=experiment_metadata,
                )
            )

        # Construct full result object
        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )

    @property
    def max_circuits(self):
        return None

    @property
    def target(self) -> Target:
        return self._target

    @property
    def meas_map(self) -> List[List[int]]:
        return self.options.meas_map

    def _get_qubit_channel(
        self, qubit: int, ChannelClass: pulse.channels.Channel, method_name: str
    ):
        """Construct a channel instance for a given qubit."""
        if qubit < len(self.options.subsystem_dims):
            return ChannelClass(qubit)

        raise QiskitError(f"{method_name} requested for qubit {qubit}, which is out of bounds.")

    def drive_channel(self, qubit: int) -> pulse.DriveChannel:
        """Return the drive channel for a given qubit."""
        return self._get_qubit_channel(qubit, pulse.DriveChannel, "drive_channel")

    def measure_channel(self, qubit: int) -> pulse.MeasureChannel:
        """Return the measure channel for a given qubit."""
        return self._get_qubit_channel(qubit, pulse.MeasureChannel, "measure_channel")

    def acquire_channel(self, qubit: int) -> pulse.AcquireChannel:
        """Return the measure channel for a given qubit."""
        return self._get_qubit_channel(qubit, pulse.AcquireChannel, "acquire_channel")

    def control_channel(
        self, qubits: Union[Tuple[int, int], List[Tuple[int, int]]]
    ) -> List[pulse.ControlChannel]:
        """Return the control channel with a given label specified by qubits.

        This method requires the ``control_channel_map`` option is set, and otherwise will raise
        a ``NotImplementedError``.

        Args:
            qubits: The label for the control channel, or a list of labels.
        Returns:
            A list containing the control channels specified by qubits.
        Raises:
            NotImplementedError: If the control_channel_map option is not set for this backend.
            QiskitError: If a requested channel is not in the control_channel_map.
        """
        if self.options.control_channel_map is None:
            raise NotImplementedError

        if not isinstance(qubits, list):
            qubits = [qubits]

        control_channels = []
        for x in qubits:
            if x not in self.options.control_channel_map:
                raise QiskitError(f"Key {x} not in control_channel_map.")
            control_channels.append(pulse.ControlChannel(self.options.control_channel_map[x]))

        return control_channels

    def configuration(self) -> PulseBackendConfiguration:
        """Get the backend configuration."""
        return self.options.configuration

    def defaults(self) -> PulseDefaults:
        """Get the backend defaults."""
        return self.options.defaults

    @classmethod
    def from_backend(
        cls,
        backend: BackendV1,
        subsystem_list: Optional[List[int]] = None,
        rotating_frame: Optional[Union[ArrayLike, RotatingFrame, str]] = "auto",
        array_library: Optional[str] = None,
        vectorized: Optional[bool] = False,
        rwa_cutoff_freq: Optional[float] = None,
        **options,
    ) -> "DynamicsBackend":
        """Construct a DynamicsBackend instance from an existing Backend instance.

        .. warning::

            Due to inevitable model inaccuracies, gates calibrated on a real backend will not have
            the same performance on the :class:`.DynamicsBackend` instance returned by this method.
            As such, gates and calibrations are not be copied into the constructed
            :class:`.DynamicsBackend`.

        The ``backend`` must contain sufficient information in the ``target``, ``configuration``,
        and/or ``defaults`` attributes to be able to run simulations. The following table indicates
        which parameters are required, along with their primary and secondary sources:

        .. list-table:: Backend parameter locations
            :widths: 10 25 25
            :header-rows: 1

            * - Parameter
              - Primary source
              - Secondary source
            * - ``hamiltonian`` dictionary.
              - ``configuration.hamiltonian``
              - N/A
            * - Control channel frequency specification.
              - ``configuration.u_channel_lo``
              - N/A
            * - Number of qubits in the backend model.
              - ``target.num_qubits``
              - ``configuration.n_qubits``
            * - Pulse schedule sample size ``dt``.
              - ``target.dt``
              - ``configuration.dt``
            * - Drive channel frequencies.
              - ``target.qubit_properties``
              - ``defaults.qubit_freq_est``
            * - Measurement channel frequencies, if measurement channels explicitly appear in the
                model.
              - ``defaults.meas_freq_est``
              - N/A

        .. note::

            The ``target``, ``configuration``, and ``defaults`` attributes of the original backend
            are not copied into the constructed :class:`DynamicsBackend` instance, only the required
            data stored within these attributes will be extracted. If necessary, these attributes
            can be set and configured by the user.

        The optional argument ``subsystem_list`` specifies which subset of qubits to model in the
        constructed :class:`DynamicsBackend`. All other qubits are dropped from the model.

        Configuration of the underlying :class:`.Solver` is controlled via the ``rotating_frame``,
        ``array_library``, ``vectorized``, and ``rwa_cutoff_freq`` options. In contrast
        to :class:`.Solver` initialization, ``rotating_frame`` defaults to the string ``"auto"``,
        which allows this method to choose the rotating frame based on ``array_library``:

        * If a dense ``array_library`` is chosen, the rotating frame will be set to the
          ``static_hamiltonian`` indicated by the Hamiltonian in ``backend.configuration()``.
        * If a sparse ``array_library`` is chosen, the rotating frame will be set to the diagonal of
          ``static_hamiltonian``.

        Otherwise the ``rotating_frame``, ``array_library``, ``vectorized``, and
        ``rwa_cutoff_freq`` are passed directly to the :class:`.Solver` initialization.

        Args:
            backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
                Note that while the type hint indicates that `backend` should be a
                :class:`~qiskit.providers.backend.BackendV1` instance, this method also works for
                :class:`~qiskit.providers.backend.BackendV2` instances that have been set up with
                sufficiently populated ``configuration`` and ``defaults`` for backwards
                compatibility.
            subsystem_list: The list of qubits in the backend to include in the model.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            array_library: Array library with which to store the operators in the :class:`.Solver`.
                See the
                :ref:`model evaluation section of the Models API documentation <model evaluation>`
                for a more detailed description of this argument.
            vectorized: If a Lindblad terms are present, whether or not to build the
                :class:`.Solver` in a vectorized mode.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            **options: Additional options to be applied in construction of the
                :class:`.DynamicsBackend`.

        Returns:
            DynamicsBackend

        Raises:
            QiskitError: If any required parameters are missing from the passed backend.
        """

        # get available target, config, and defaults objects
        backend_target = getattr(backend, "target", None)

        if not hasattr(backend, "configuration"):
            raise QiskitError(
                "DynamicsBackend.from_backend requires that the backend argument has a "
                "configuration method."
            )
        backend_config = backend.configuration()

        backend_defaults = None
        if hasattr(backend, "defaults"):
            backend_defaults = backend.defaults()

        # get and parse Hamiltonian string dictionary
        if backend_target is not None:
            backend_num_qubits = backend_target.num_qubits
        else:
            backend_num_qubits = backend_config.n_qubits

        if subsystem_list is not None:
            subsystem_list = sorted(subsystem_list)
            if subsystem_list[-1] >= backend_num_qubits:
                raise QiskitError(
                    f"subsystem_list contained {subsystem_list[-1]}, which is out of bounds for "
                    f"backend with {backend_num_qubits} qubits."
                )
        else:
            subsystem_list = list(range(backend_num_qubits))

        if backend_config.hamiltonian is None:
            raise QiskitError(
                "DynamicsBackend.from_backend requires that backend.configuration() has a "
                "hamiltonian."
            )

        (
            static_hamiltonian,
            hamiltonian_operators,
            hamiltonian_channels,
            subsystem_dims_dict,
        ) = parse_backend_hamiltonian_dict(backend_config.hamiltonian, subsystem_list)
        subsystem_dims = [subsystem_dims_dict.get(idx, 1) for idx in range(backend_num_qubits)]

        # construct model frequencies dictionary from backend
        channel_freqs = _get_backend_channel_freqs(
            backend_target=backend_target,
            backend_config=backend_config,
            backend_defaults=backend_defaults,
            channels=hamiltonian_channels,
        )

        # Add control_channel_map from backend (only if not specified before by user)
        if "control_channel_map" not in options:
            if hasattr(backend, "control_channels"):
                control_channel_map_backend = {
                    qubits: backend.control_channels[qubits][0].index
                    for qubits in backend.control_channels
                }

            elif hasattr(backend.configuration(), "control_channels"):
                control_channel_map_backend = {
                    qubits: backend.configuration().control_channels[qubits][0].index
                    for qubits in backend.configuration().control_channels
                }

            else:
                control_channel_map_backend = {}

            # Reduce control_channel_map based on which channels are in the model
            if bool(control_channel_map_backend):
                control_channel_map = {}
                for label, idx in control_channel_map_backend.items():
                    if f"u{idx}" in hamiltonian_channels:
                        control_channel_map[label] = idx
                options["control_channel_map"] = control_channel_map

        # build the solver
        if rotating_frame == "auto":
            if array_library is not None and "sparse" in array_library:
                rotating_frame = np.diag(static_hamiltonian)
            else:
                rotating_frame = static_hamiltonian

        # get time step size
        if backend_target is not None and backend_target.dt is not None:
            dt = backend_target.dt
        else:
            # config is guaranteed to have a dt
            dt = backend_config.dt

        solver = Solver(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            dt=dt,
            rotating_frame=rotating_frame,
            array_library=array_library,
            vectorized=vectorized,
            rwa_cutoff_freq=rwa_cutoff_freq,
        )

        return cls(
            solver=solver,
            target=Target(dt=dt),
            subsystem_dims=subsystem_dims,
            **options,
        )


def default_experiment_result_function(
    experiment_name: str,
    solver_result: OdeResult,
    measurement_subsystems: List[int],
    memory_slot_indices: List[int],
    num_memory_slots: Union[None, int],
    backend: DynamicsBackend,
    seed: Optional[int] = None,
    metadata: Optional[Dict] = None,
) -> ExperimentResult:
    """Default routine for generating ExperimentResult object.

    To generate the results for a given experiment, this method takes the following steps:

    * The final state is transformed out of the rotating frame and into the lab frame using
      ``backend.options.solver``.
    * If ``backend.options.normalize_states==True``, the final state is normalized.
    * Measurement results are computed, in the dressed basis, based on both the measurement-related
      options in ``backend.options`` and the measurement specification extracted from the specific
      experiment.

    Args:
        experiment_name: Name of experiment.
        solver_result: Result object from :class:`Solver.solve`.
        measurement_subsystems: Labels of subsystems in the model being measured.
        memory_slot_indices: Indices of memory slots to store the results in for each subsystem.
        num_memory_slots: Total number of memory slots in the returned output. If ``None``,
            ``max(memory_slot_indices)`` will be used.
        backend: The backend instance that ran the simulation. Various options and properties
            are utilized.
        seed: Seed for any random number generation involved (e.g. when computing outcome samples).
        metadata: Metadata to add to the header of the
            :class:`~qiskit.result.models.ExperimentResult` object.

    Returns:
        :class:`~qiskit.result.models.ExperimentResult` object containing results.

    Raises:
        QiskitError: If a specified option is unsupported.
    """

    yf = solver_result.y[-1]
    tf = solver_result.t[-1]

    # Take state out of frame, put in dressed basis, and normalize
    if isinstance(yf, Statevector):
        yf = np.array(backend.options.solver.model.rotating_frame.state_out_of_frame(t=tf, y=yf))
        yf = backend._dressed_states_adjoint @ yf
        yf = Statevector(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.linalg.norm(yf.data)
    elif isinstance(yf, DensityMatrix):
        yf = np.array(
            backend.options.solver.model.rotating_frame.operator_out_of_frame(t=tf, operator=yf)
        )
        yf = backend._dressed_states_adjoint @ yf @ backend._dressed_states
        yf = DensityMatrix(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.diag(yf.data).sum()

    if backend.options.meas_level == MeasLevel.CLASSIFIED:
        memory_slot_probabilities = _get_memory_slot_probabilities(
            probability_dict=yf.probabilities_dict(qargs=measurement_subsystems),
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            max_outcome_value=backend.options.max_outcome_level,
        )

        # sample
        memory_samples = _sample_probability_dict(
            memory_slot_probabilities,
            shots=backend.options.shots,
            normalize_probabilities=backend.options.normalize_states,
            seed=seed,
        )
        counts = _get_counts_from_samples(memory_samples)

        # construct results object
        exp_data = ExperimentResultData(
            counts=counts, memory=memory_samples if backend.options.memory else None
        )
        return ExperimentResult(
            shots=backend.options.shots,
            success=True,
            data=exp_data,
            meas_level=MeasLevel.CLASSIFIED,
            seed=seed,
            header=QobjHeader(name=experiment_name, metadata=metadata),
        )
    elif backend.options.meas_level == MeasLevel.KERNELED:
        iq_centers = backend.options.iq_centers
        if iq_centers is None:
            # Default iq_centers
            iq_centers = []
            for sub_dim in backend.options.subsystem_dims:
                theta = 2 * np.pi / sub_dim
                iq_centers.append(
                    [(np.cos(idx * theta), np.sin(idx * theta)) for idx in range(sub_dim)]
                )

        # generate IQ
        measurement_data = _get_iq_data(
            yf,
            measurement_subsystems=measurement_subsystems,
            iq_centers=iq_centers,
            iq_width=backend.options.iq_width,
            shots=backend.options.shots,
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            seed=seed,
        )

        if backend.options.meas_return == MeasReturnType.AVERAGE:
            measurement_data = np.average(measurement_data, axis=0)

        # construct results object
        exp_data = ExperimentResultData(memory=measurement_data.tolist())
        return ExperimentResult(
            shots=backend.options.shots,
            success=True,
            data=exp_data,
            meas_level=MeasLevel.KERNELED,
            seed=seed,
            header=QobjHeader(name=experiment_name, metadata=metadata),
        )

    else:
        raise QiskitError(f"meas_level=={backend.options.meas_level} not implemented.")


def _validate_run_input(run_input, accept_list=True, caller_func_name: str = None):
    """Raise errors if the run_input is not one of QuantumCircuit, Schedule, ScheduleBlock, or
    a list of these.
    """
    if caller_func_name is None:
        caller_func_name = inspect.stack()[1].function
    if isinstance(run_input, list) and accept_list:
        # if list apply recursively, but no longer accept lists
        for x in run_input:
            _validate_run_input(x, accept_list=False, caller_func_name=caller_func_name)
    elif not isinstance(run_input, (QuantumCircuit, Schedule, ScheduleBlock)):
        raise QiskitError(
            f"Input type {type(run_input)} not supported by DynamicsBackend.{caller_func_name}."
        )


def _get_acquire_instruction_timings(
    schedules: List[Schedule], subsystem_dims: List[int], dt: float
) -> Tuple[List[List[float]], List[List[int]], List[List[int]]]:
    """Get the required data from the acquire commands in each schedule.

    Additionally validates that each schedule has Acquire instructions occurring at one time, at
    least one memory slot is being listed, and all measured subsystem indices are less than
    ``len(subsystem_dims)``. Additionally, a warning is raised if a 'trivial' subsystem is measured,
    i.e. one with dimension 1.

    Args:
        schedules: A list of schedules.
        subsystem_dims: List of subsystem dimensions.
        dt: The sample size.
    Returns:
        A tuple of lists containing, for each schedule: the list of integration intervals required
        for each schedule (in absolute time, from 0.0 to the beginning of the acquire instructions),
        a list of the subsystems being measured, and a list of the memory slots indices in which to
        store the results of each subsystem measurement.
    Raises:
        QiskitError: If a schedule contains no measurement, if a schedule contains measurements at
            different times, or if a measurement has an invalid subsystem label.
    """
    t_span_list = []
    measurement_subsystems_list = []
    memory_slot_indices_list = []
    for schedule in schedules:
        schedule_acquires = []
        schedule_acquire_times = []
        for start_time, inst in schedule.instructions:
            # only track acquires saving in a memory slot
            if isinstance(inst, pulse.Acquire) and inst.mem_slot is not None:
                schedule_acquires.append(inst)
                schedule_acquire_times.append(start_time)

        # validate
        if len(schedule_acquire_times) == 0:
            raise QiskitError(
                "At least one measurement saving a a result in a MemorySlot "
                "must be present in each schedule."
            )

        for acquire_time in schedule_acquire_times[1:]:
            if acquire_time != schedule_acquire_times[0]:
                raise QiskitError("DynamicsBackend.run only supports measurements at one time.")

        # use dt to convert acquire start time from sample index to the integration interval
        t_span_list.append([0.0, dt * schedule_acquire_times[0]])
        measurement_subsystems = []
        memory_slot_indices = []
        for inst in schedule_acquires:
            if not inst.channel.index < len(subsystem_dims):
                raise QiskitError(
                    f"Attempted to measure out of bounds subsystem {inst.channel.index}."
                )

            if subsystem_dims[inst.channel.index] == 1:
                warnings.warn(f"Measuring trivial subsystem {inst.channel.index} with dimension 1.")

            measurement_subsystems.append(inst.channel.index)

            memory_slot_indices.append(inst.mem_slot.index)

        measurement_subsystems_list.append(measurement_subsystems)
        memory_slot_indices_list.append(memory_slot_indices)

    return t_span_list, measurement_subsystems_list, memory_slot_indices_list


def _to_schedule_list(
    run_input: List[Union[QuantumCircuit, Schedule, ScheduleBlock]], backend: BackendV2
):
    """Convert all inputs to schedules, and store the number of classical registers present
    in any circuits.
    """
    if not isinstance(run_input, list):
        run_input = [run_input]

    schedules = []
    num_memslots = []
    for sched in run_input:
        num_memslots.append(None)
        if isinstance(sched, ScheduleBlock):
            schedules.append(block_to_schedule(sched))
        elif isinstance(sched, Schedule):
            schedules.append(sched)
        elif isinstance(sched, QuantumCircuit):
            num_memslots[-1] = sum(creg.size for creg in sched.cregs)
            schedules.append(build_schedule(sched, backend, dt=backend.options.solver._dt))
        else:
            raise QiskitError(f"Type {type(sched)} cannot be converted to Schedule.")
    return schedules, num_memslots


def _get_backend_channel_freqs(
    backend_target: Optional[Target],
    backend_config: PulseBackendConfiguration,
    backend_defaults: Optional[PulseDefaults],
    channels: List[str],
) -> Dict[str, float]:
    """Extract frequencies of channels from a backend configuration and defaults.

    Args:
        backend_target: A backend target object or ``None``.
        backend_config: A backend configuration object.
        backend_defaults: A backend defaults object or ``None``.
        channels: Channel labels given as strings, assumed to be unique.

    Returns:
        Dict: Mapping of channel labels to frequencies.

    Raises:
        QiskitError: If the frequency for one of the channels cannot be found.
    """

    # partition types of channels
    drive_channels = []
    meas_channels = []
    u_channels = []

    for channel in channels:
        if channel[0] == "d":
            drive_channels.append(channel)
        elif channel[0] == "m":
            meas_channels.append(channel)
        elif channel[0] == "u":
            u_channels.append(channel)
        else:
            raise QiskitError("Unrecognized channel type requested.")

    # extract and validate channel frequency parameters
    if drive_channels:
        # get drive channel frequencies
        drive_frequencies = []
        if (backend_target is not None) and (backend_target.qubit_properties is not None):
            drive_frequencies = [q.frequency for q in backend_target.qubit_properties]
        elif backend_defaults is not None:
            drive_frequencies = backend_defaults.qubit_freq_est
        else:
            raise QiskitError(
                "DriveChannels in model but frequencies not available in target or defaults."
            )

    if meas_channels:
        if backend_defaults is not None:
            meas_frequencies = backend_defaults.meas_freq_est
        else:
            raise QiskitError("MeasureChannels in model but defaults does not have meas_freq_est.")

    # backend_config.u_channel_lo is guaranteed to be a list
    u_channel_lo = backend_config.u_channel_lo

    # populate frequencies
    channel_freqs = {}

    for channel in drive_channels:
        idx = int(channel[1:])
        if idx >= len(drive_frequencies):
            raise QiskitError(f"DriveChannel index {idx} is out of bounds.")
        channel_freqs[channel] = drive_frequencies[idx]

    for channel in meas_channels:
        idx = int(channel[1:])
        if idx >= len(meas_frequencies):
            raise QiskitError(f"MeasureChannel index {idx} is out of bounds.")
        channel_freqs[channel] = meas_frequencies[idx]

    for channel in u_channels:
        idx = int(channel[1:])
        if idx >= len(u_channel_lo):
            raise QiskitError(f"ControlChannel index {idx} is out of bounds.")
        freq = 0.0
        for channel_lo in u_channel_lo[idx]:
            freq += drive_frequencies[channel_lo.q] * channel_lo.scale

        channel_freqs[channel] = freq

    # validate that all channels have frequencies
    for channel in channels:
        if channel not in channel_freqs:
            raise QiskitError(f"No carrier frequency found for channel {channel}.")

    return channel_freqs

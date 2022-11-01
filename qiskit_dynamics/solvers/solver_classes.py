# -*- coding: utf-8 -*-

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

r"""
Solver classes.
"""

from typing import Optional, Union, Tuple, Any, Type, List, Callable

from contextlib import contextmanager
from functools import partial
import numpy as np

from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import

from qiskit import QiskitError
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator, DensityMatrix

from qiskit_dynamics.models import (
    HamiltonianModel,
    LindbladModel,
    RotatingFrame,
    rotating_wave_approximation,
)
from qiskit_dynamics.signals import Signal, DiscreteSignal, SignalList
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.array import Array
from qiskit_dynamics.dispatch.dispatch import Dispatch

from .solver_functions import solve_lmde, _is_diffrax_method
from .solver_utils import (
    is_lindblad_model_vectorized,
    is_lindblad_model_not_vectorized,
    jit_with_static_mutables,
    setup_args_lists,
)


class Solver:
    r"""Solver class for simulating both Hamiltonian and Lindblad dynamics, with high
    level type-handling of input states.

    If only Hamiltonian information is provided, this class will internally construct
    a :class:`.HamiltonianModel` instance, and simulate the model
    using the Schrodinger equation :math:`\dot{y}(t) = -iH(t)y(t)`
    (see the :meth:`.Solver.solve` method documentation for details
    on how different initial state types are handled).
    :class:`.HamiltonianModel` represents a decomposition of the Hamiltonian of the form:

    .. math::

        H(t) = H_0 + \sum_i s_i(t) H_i,

    where :math:`H_0` is the static component, the :math:`H_i` are the
    time-dependent components of the Hamiltonian, and the :math:`s_i(t)` are the
    time-dependent signals, specifiable as either :class:`.Signal`
    objects, or constructed from Qiskit Pulse schedules if :class:`.Solver`
    is configured for Pulse simulation (see below).

    If dissipators are specified as part of the model, then a
    :class:`.LindbladModel` is constructed, and simulations are performed
    by solving the Lindblad equation:

    .. math::

        \dot{y}(t) = -i[H(t), y(t)] + \mathcal{D}_0(y(t)) + \mathcal{D}(t)(y(t)),

    where :math:`H(t)` is the Hamiltonian part, specified as above, and :math:`\mathcal{D}_0`
    and :math:`\mathcal{D}(t)` are the static and time-dependent portions of the dissipator,
    given by:

    .. math::

        \mathcal{D}_0(y(t)) = \sum_j N_j y(t) N_j^\dagger
                                      - \frac{1}{2} \{N_j^\dagger N_j, y(t)\},

    and

    .. math::

        \mathcal{D}(t)(y(t)) = \sum_j \gamma_j(t) L_j y(t) L_j^\dagger
                                  - \frac{1}{2} \{L_j^\dagger L_j, y(t)\},

    with :math:`N_j` the static dissipators, :math:`L_j` the time-dependent dissipator
    operators, and :math:`\gamma_j(t)` the time-dependent signals
    specifiable as either :class:`.Signal`
    objects, or constructed from Qiskit Pulse schedules if :class:`.Solver`
    is configured for Pulse simulation (see below).

    Transformations on the model can be specified via the optional arguments:

    * ``rotating_frame``: Transforms the model into a rotating frame. Note that the
      operator specifying the frame will be substracted from the ``static_hamiltonian``.
      If supplied as a 1d array, ``rotating_frame`` is interpreted as the diagonal
      elements of a diagonal matrix. Given a frame operator :math:`F = -i H_0`,
      for the Schrodinger equation entering the rotating frame of :math:`F`, corresponds
      to transforming the solution as :math:`y(t) \mapsto exp(-tF)y(t)`, and for the
      Lindblad equation it corresponds to transforming the solution as
      :math:`y(t) \mapsto exp(-tF)y(t)exp(tF)`.
      See :class:`.RotatingFrame` for more details.
    * ``in_frame_basis``: Whether to represent the model in the basis in which the frame
      operator is diagonal, henceforth called the "frame basis".
      If ``rotating_frame`` is ``None`` or was supplied as a 1d array,
      this kwarg has no effect. If ``rotating_frame`` was specified as a 2d array,
      the frame basis is the diagonalizing basis supplied by ``np.linalg.eigh``.
      If ``in_frame_basis==True``, this objects behaves as if all
      operators were supplied in the frame basis: calls to ``solve`` will assume the initial
      state is supplied in the frame basis, and the results will be returned in the frame basis.
      If ``in_frame_basis==False``, the system will still be solved in the frame basis for
      efficiency, however the initial state (and final output states) will automatically be
      transformed into (and, respectively, out of) the frame basis.
    * ``rwa_cutoff_freq`` and ``rwa_carrier_freqs``: Performs a rotating wave approximation (RWA)
      on the model with cutoff frequency ``rwa_cutoff_freq``, assuming the time-dependent
      coefficients of the model have carrier frequencies specified by ``rwa_carrier_freqs``.
      If ``dissipator_operators is None``, ``rwa_carrier_freqs`` must be a list of floats
      of length equal to ``hamiltonian_operators``, and if ``dissipator_operators is not None``,
      ``rwa_carrier_freqs`` must be a ``tuple`` of lists of floats, with the first entry
      the list of carrier frequencies for ``hamiltonian_operators``, and the second
      entry the list of carrier frequencies for ``dissipator_operators``.
      See :func:`.rotating_wave_approximation` for details on
      the mathematical approximation.

      .. note::
            When using the ``rwa_cutoff_freq`` optional argument,
            :class:`.Solver` cannot be instantiated within
            a JAX-transformable function. However, after construction, instances can
            still be used within JAX-transformable functions regardless of whether an
            ``rwa_cutoff_freq`` is set.

    :class:`.Solver` can be configured to simulate Qiskit Pulse schedules
    by setting all of the following parameters, which determine how Pulse schedules are
    interpreted:

    * ``hamiltonian_channels``: List of channels in string format corresponding to the
      time-dependent coefficients of ``hamiltonian_operators``.
    * ``dissipator_channels``: List of channels in string format corresponding to time-dependent
      coefficients of ``dissipator_operators``.
    * ``channel_carrier_freqs``: Dictionary mapping channel names to frequencies. A frequency
      must be specified for every channel appearing in ``hamiltonian_channels`` and
      ``dissipator_channels``. When simulating ``schedule``\s, these frequencies are
      interpreted as the analog carrier frequencies associated with the channel; deviations from
      these frequencies due to ``SetFrequency`` or ``ShiftFrequency`` instructions are
      implemented by digitally modulating the samples for the channel envelope.
      If an ``rwa_cutoff_freq`` is specified, and no ``rwa_carrier_freqs`` is specified, these
      frequencies will be used for the RWA.
    * ``dt``: The envelope sample width.

    If configured to simulate Pulse schedules while ``Array.default_backend() == 'jax'``,
    calling :meth:`.Solver.solve` will automatically compile
    simulation runs when calling with a JAX-based solver method.

    The evolution given by the model can be simulated by calling :meth:`.Solver.solve`, which
    calls :func:`.solve_lmde`, and does various automatic
    type handling operations for :mod:`qiskit.quantum_info` state and super operator types.
    """
    _JIT_MIN_PADDING_FRAC = 0.2
    """Smallest allowed maximum duration of a schedule list to allow ``_jit_cache`` fetching. This
    is used to limit excessive zero-padding. For example, if the longest schedule in a schedule list
    is 100 and the ``_jit_cache`` contains entries at 50 and 10000, then neither would be fetched
    because the former is smaller than 100, and the latter is greater than 
    ``_JIT_MIN_PADDING_FRAC * 10000``."""

    _JIT_DURATION_PADDING = 1.5
    """By how much to pad new entries in the ``_jit_cache``."""

    def __init__(
        self,
        static_hamiltonian: Optional[Array] = None,
        hamiltonian_operators: Optional[Array] = None,
        static_dissipators: Optional[Array] = None,
        dissipator_operators: Optional[Array] = None,
        hamiltonian_channels: Optional[List[str]] = None,
        dissipator_channels: Optional[List[str]] = None,
        channel_carrier_freqs: Optional[dict] = None,
        dt: Optional[float] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        rwa_carrier_freqs: Optional[Union[Array, Tuple[Array, Array]]] = None,
        validate: bool = True,
    ):
        """Initialize solver with model information.

        Args:
            static_hamiltonian: Constant Hamiltonian term. If a ``rotating_frame``
                                is specified, the ``frame_operator`` will be subtracted from
                                the static_hamiltonian.
            hamiltonian_operators: Hamiltonian operators.
            static_dissipators: Constant dissipation operators.
            dissipator_operators: Dissipation operators with time-dependent coefficients.
            hamiltonian_channels: List of channel names in pulse schedules corresponding to
                                  Hamiltonian operators.
            dissipator_channels: List of channel names in pulse schedules corresponding to
                                 dissipator operators.
            channel_carrier_freqs: Dictionary mapping channel names to floats which represent
                                   the carrier frequency of the pulse channel with the
                                   corresponding name.
            dt: Sample rate for simulating pulse schedules.
            rotating_frame: Rotating frame to transform the model into. Rotating frames which
                            are diagonal can be supplied as a 1d array of the diagonal elements,
                            to explicitly indicate that they are diagonal.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                            frame operator is diagonalized. See class documentation for a more
                            detailed explanation on how this argument affects object behaviour.
            evaluation_mode: Method for model evaluation. See documentation for
                             ``HamiltonianModel.evaluation_mode`` or
                             ``LindbladModel.evaluation_mode``.
                             (if dissipators in model) for valid modes.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency. If ``None``, no
                             approximation is made.
            rwa_carrier_freqs: Carrier frequencies to use for rotating wave approximation.
                               If no time dependent coefficients in model leave as ``None``,
                               if no time-dependent dissipators specify as a list of frequencies
                               for each Hamiltonian operator, and if time-dependent dissipators
                               present specify as a tuple of lists of frequencies, one for
                               Hamiltonian operators and one for dissipators.
            validate: Whether or not to validate Hamiltonian operators as being Hermitian.

        Raises:
            QiskitError: If arguments concerning pulse-schedule interpretation are insufficiently
            specified.
        """

        # set pulse specific information if specified
        self._hamiltonian_channels = None
        self._dissipator_channels = None
        self._all_channels = None
        self._channel_carrier_freqs = None
        self._schedule_converter = None

        # cache of which jit depths have been used so far
        self._jit_cache = set()

        if any([dt, channel_carrier_freqs, hamiltonian_channels, dissipator_channels]):
            all_channels = []

            if hamiltonian_channels is not None:
                hamiltonian_channels = [chan.lower() for chan in hamiltonian_channels]
                if hamiltonian_operators is None or len(hamiltonian_operators) != len(
                    hamiltonian_channels
                ):
                    raise QiskitError(
                        """hamiltonian_channels must have same length as hamiltonian_operators"""
                    )
                for chan in hamiltonian_channels:
                    if chan not in all_channels:
                        all_channels.append(chan)

            self._hamiltonian_channels = hamiltonian_channels

            if dissipator_channels is not None:
                dissipator_channels = [chan.lower() for chan in dissipator_channels]
                for chan in dissipator_channels:
                    if chan not in all_channels:
                        all_channels.append(chan)
                if dissipator_operators is None or len(dissipator_operators) != len(
                    dissipator_channels
                ):
                    raise QiskitError(
                        """dissipator_channels must have same length as dissipator_operators"""
                    )

            self._dissipator_channels = dissipator_channels
            self._all_channels = all_channels

            if channel_carrier_freqs is None:
                channel_carrier_freqs = {}
            else:
                channel_carrier_freqs = {
                    key.lower(): val for key, val in channel_carrier_freqs.items()
                }

            for chan in all_channels:
                if chan not in channel_carrier_freqs:
                    raise QiskitError(
                        f"""Channel '{chan}' does not have carrier frequency specified in
                        channel_carrier_freqs."""
                    )

            if len(channel_carrier_freqs) == 0:
                channel_carrier_freqs = None
            self._channel_carrier_freqs = channel_carrier_freqs

            if dt is not None:
                self._schedule_converter = InstructionToSignals(
                    dt=dt, carriers=self._channel_carrier_freqs, channels=self._all_channels
                )
            else:
                raise QiskitError("dt must be specified if channel information is provided.")

        # setup model
        model = None
        if static_dissipators is None and dissipator_operators is None:
            model = HamiltonianModel(
                static_operator=static_hamiltonian,
                operators=hamiltonian_operators,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )
        else:
            model = LindbladModel(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                static_dissipators=static_dissipators,
                dissipator_operators=dissipator_operators,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )

        self._rwa_signal_map = None
        self._model = model
        if rwa_cutoff_freq:

            # if rwa_carrier_freqs is None, take from channel_carrier_freqs or set all to 0.
            if rwa_carrier_freqs is None:
                if self._channel_carrier_freqs is not None:
                    if self._hamiltonian_channels is not None:
                        rwa_carrier_freqs = [
                            self._channel_carrier_freqs[c] for c in self._hamiltonian_channels
                        ]

                    if self._dissipator_channels is not None:
                        rwa_carrier_freqs = (
                            rwa_carrier_freqs,
                            [self._channel_carrier_freqs[c] for c in self._dissipator_channels],
                        )
                else:
                    rwa_carrier_freqs = []
                    if hamiltonian_operators is not None:
                        rwa_carrier_freqs = [0.0] * len(hamiltonian_operators)

                    if dissipator_operators is not None:
                        rwa_carrier_freqs = (rwa_carrier_freqs, [0.0] * len(dissipator_operators))

            if isinstance(rwa_carrier_freqs, tuple):
                rwa_ham_sigs = None
                rwa_lindblad_sigs = None
                if rwa_carrier_freqs[0]:
                    rwa_ham_sigs = [Signal(1.0, carrier_freq=freq) for freq in rwa_carrier_freqs[0]]
                if rwa_carrier_freqs[1]:
                    rwa_lindblad_sigs = [
                        Signal(1.0, carrier_freq=freq) for freq in rwa_carrier_freqs[1]
                    ]

                self._model.signals = (rwa_ham_sigs, rwa_lindblad_sigs)
            else:
                rwa_sigs = [Signal(1.0, carrier_freq=freq) for freq in rwa_carrier_freqs]

                if isinstance(model, LindbladModel):
                    rwa_sigs = (rwa_sigs, None)

                self._model.signals = rwa_sigs

            self._model, rwa_signal_map = rotating_wave_approximation(
                self._model, rwa_cutoff_freq, return_signal_map=True
            )
            self._rwa_signal_map = rwa_signal_map

    @property
    def model(self) -> Union[HamiltonianModel, LindbladModel]:
        """The model of the system, either a Hamiltonian or Lindblad model."""
        return self._model

    def solve(
        self,
        t_span: Array,
        y0: Union[Array, QuantumState, BaseOperator],
        signals: Optional[
            Union[
                List[Union[Schedule, ScheduleBlock]],
                List[Signal],
                Tuple[List[Signal], List[Signal]],
            ]
        ] = None,
        convert_results: bool = True,
        **kwargs,
    ) -> Union[OdeResult, List[OdeResult]]:
        r"""Solve a dynamical problem, or a set of dynamical problems.

        Calls :func:`.solve_lmde`, and returns an ``OdeResult``
        object in the style of ``scipy.integrate.solve_ivp``, with results
        formatted to be the same types as the input. See Additional Information
        for special handling of various input types, and for specifying multiple
        simulations at once.

        Args:
            t_span: Time interval to integrate over.
            y0: Initial state.
            signals: Specification of time-dependent coefficients to simulate, either in
                     Signal format or as Qiskit Pulse Pulse schedules.
                     If specifying in Signal format, if ``dissipator_operators is None``,
                     specify as a list of signals for the Hamiltonian component, otherwise
                     specify as a tuple of two lists, one for Hamiltonian components, and
                     one for the ``dissipator_operators`` coefficients.
            convert_results: If ``True``, convert returned solver state results to the same class
                             as y0. If ``False``, states will be returned in the native array type
                             used by the specified solver method.
            **kwargs: Keyword args passed to :func:`.solve_lmde`.

        Returns:
            OdeResult: object with formatted output types.

        Raises:
            QiskitError: Initial state ``y0`` is of invalid shape. If ``signals`` specifies
                         ``Schedule`` simulation but ``Solver`` hasn't been configured to
                         simulate pulse schedules.

        Additional Information:

        The behaviour of this method is impacted by the input type of ``y0``
        and the internal model, summarized in the following table:

        .. list-table:: Type-based behaviour
           :widths: 10 10 10 70
           :header-rows: 1

           * - ``y0`` type
             - Model type
             - ``yf`` type
             - Description
           * - ``Array``, ``np.ndarray``, ``Operator``
             - Any
             - Same as ``y0``
             - Solves either the Schrodinger equation or Lindblad equation
               with initial state ``y0`` as specified.
           * - ``Statevector``
             - ``HamiltonianModel``
             - ``Statevector``
             - Solves the Schrodinger equation with initial state ``y0``.
           * - ``DensityMatrix``
             - ``HamiltonianModel``
             - ``DensityMatrix``
             - Solves the Schrodinger equation with initial state the identity matrix to compute
               the unitary, then conjugates ``y0`` with the result to solve for the density matrix.
           * - ``Statevector``, ``DensityMatrix``
             - ``LindbladModel``
             - ``DensityMatrix``
             - Solve the Lindblad equation with initial state ``y0``, converting to a
               ``DensityMatrix`` first if ``y0`` is a ``Statevector``.
           * - ``QuantumChannel``
             - ``HamiltonianModel``
             - ``SuperOp``
             - Converts ``y0`` to a ``SuperOp`` representation, then solves the Schrodinger
               equation with initial state the identity matrix to compute the unitary and
               composes with ``y0``.
           * - ``QuantumChannel``
             - ``LindbladModel``
             - ``SuperOp``
             - Solves the vectorized Lindblad equation with initial state ``y0``.
               ``evaluation_mode`` must be set to a vectorized option.

        In some cases (e.g. if using JAX), wrapping the returned states in the type
        given in the ``yf`` type column above may be undesirable. Setting
        ``convert_results=False`` prevents this wrapping, while still allowing
        usage of the automatic type-handling for the input state.

        In addition to the above, this method can be used to specify multiple simulations
        simultaneously. This can be done by specifying one or more of the arguments
        ``t_span``, ``y0``, or ``signals`` as a list of valid inputs.
        For this mode of operation, all of these arguments must be either lists of the same
        length, or a single valid input, which will be used repeatedly.

        For example the following code runs three simulations, returning results in a list:

        .. code-block:: python

            t_span = [span1, span2, span3]
            y0 = [state1, state2, state3]
            signals = [signals1, signals2, signals3]

            results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        The following code block runs three simulations, for different sets of signals,
        repeatedly using the same ``t_span`` and ``y0``:

        .. code-block:: python

            t_span = [t0, tf]
            y0 = state1
            signals = [signals1, signals2, signal3]

            results = solver.solve(t_span=t_span, y0=y0, signals=signals)
        """
        # validate and setup list of simulations
        [t_span_list, y0_list, signals_list], multiple_sims = setup_args_lists(
            args_list=[t_span, y0, signals],
            args_names=["t_span", "y0", "signals"],
            args_to_list=[t_span_to_list, _y0_to_list, _signals_to_list],
        )

        # turn any present schedules into signals and find longest DiscreteSignal
        max_samples = 0
        for idx, sigs in enumerate(signals_list):
            if isinstance(sigs, ScheduleBlock):
                signals = block_to_schedule(sigs)
            if isinstance(sigs, Schedule):
                signals_list[idx] = sigs = self._schedule_converter.get_signals(sigs)
            if all(isinstance(signal, DiscreteSignal) for signal in sigs):
                max_samples = max(max_samples, max(signal.duration for signal in sigs))

        # decide on whether jax compilation is a possibility
        method = kwargs.get("method", "")
        can_jit_compile = Array.default_backend() == "jax"
        can_jit_compile &= method == "jax_odeint" or _is_diffrax_method(method)

        all_results = []
        for t_span_elem, y0_elem, sigs in zip(t_span_list, y0_list, signals_list):
            # setup initial state
            y0, y0_input, y0_cls, state_type_wrapper = validate_and_format_initial_state(
                y0, self.model
            )

            results = None
            if can_jit_compile and all(isinstance(signal, DiscreteSignal) for signal in sigs):
                samples = np.zeros(self._jit_shape(max_samples), dtype=complex)
                dts = []
                start_times = []
                for idx, signal in enumerate(sigs):
                    samples[idx, : signal.duration] = signal.samples
                    dts.append(signal.dt)
                    start_times.append(signal.start_time)

                results_t, results_y = self._jit_sim(
                    Array(t_span_elem).data,
                    Array(y0_elem).data,
                    samples,
                    Array(dts).data,
                    Array(start_times).data,
                    Array(y0_input).data,
                    y0_cls,
                    kwargs,
                )
                results = OdeResult(t=results_t, y=Array(results_y, dtype=complex))
            else:
                with self.signal_assignment(sigs):
                    results = solve_lmde(
                        generator=self.model, t_span=t_span_elem, y0=y0_elem, **kwargs
                    )
                    results.y = format_final_states(results.y, self.model, y0_input, y0_cls)

            if y0_cls is not None and convert_results:
                results.y = [state_type_wrapper(yi) for yi in results.y]

            all_results.append(results)

        return all_results if multiple_sims else all_results[0]

    @contextmanager
    def signal_assignment(self, signals: Union[Tuple[List[Signal], List[Signal]], List[Signal]]):
        """Context manager which temporarily changes the model's signals to those provided
        and changes them back when the context is exited."""
        old_signals = self.model.signals

        # if Lindblad model and signals are given as a list set as Hamiltonian part of signals
        if isinstance(self.model, LindbladModel) and isinstance(signals, (list, SignalList)):
            signals = (signals, None)

        if self._rwa_signal_map:
            signals = self._rwa_signal_map(signals)
        self.model.signals = signals

        yield

        self.model.signals = old_signals

    def _jit_shape(self, required_duration: int):
        """If possible, return a nearby samples shape so that we don't have to recompile. Otherwise,
        add a new one to the cache."""
        for duration in sorted(self._jit_cache):
            if duration * self._JIT_MIN_PADDING_FRAC < required_duration <= duration:
                # something appropriate was found in the cache
                break
        else:
            duration = round(self._JIT_DURATION_PADDING * required_duration)
            self._jit_cache.add(duration)

        return len(self._all_channels), duration

    @partial(
        jit_with_static_mutables,
        static_argnames=["self", "y_cls"],
        static_mutable_argnames=["kwargs"],
    )
    def _jit_sim(self, t_span, y0, all_samples, dts, start_times, y_input, y_cls, kwargs):
        # re-construct signals from the samples
        signals = []
        for idx, (samples, dt, start_time) in enumerate(zip(all_samples, dts, start_times)):
            carrier_freq = self._channel_carrier_freqs[self._all_channels[idx]]
            signals.append(
                DiscreteSignal(
                    samples=samples, dt=dt, start_time=start_time, carrier_freq=carrier_freq
                )
            )

        # map signals to correct structure for model
        signals = organize_signals_to_channels(
            signals,
            self._all_channels,
            self.model.__class__,
            self._hamiltonian_channels,
            self._dissipator_channels,
        )

        with self.signal_assignment(signals):
            results = solve_lmde(generator=self.model, t_span=t_span, y0=y0, **kwargs)
            results.y = format_final_states(results.y, self.model, y_input, y_cls)

        return Array(results.t).data, Array(results.y).data

    def _schedule_to_signals(self, schedule: Union[List[Signal], Schedule]):
        """Convert a schedule into the signal format required by the model."""
        if self._schedule_converter is None:
            raise QiskitError("Solver instance not configured for pulse Schedule simulation.")

        return organize_signals_to_channels(
            self._schedule_converter.get_signals(schedule),
            self._all_channels,
            self.model.__class__,
            self._hamiltonian_channels,
            self._dissipator_channels,
        )


def initial_state_converter(obj: Any) -> Tuple[Array, Type, Callable]:
    """Convert initial state object to an Array, the type of the initial input, and return
    function for constructing a state of the same type.

    Args:
        obj: An initial state.

    Returns:
        tuple: (Array, Type, Callable)
    """
    # pylint: disable=invalid-name
    y0_cls = None
    if isinstance(obj, Array):
        y0, y0_cls, wrapper = obj, None, lambda x: x
    if isinstance(obj, QuantumState):
        y0, y0_cls = Array(obj.data), obj.__class__
        wrapper = lambda x: y0_cls(np.array(x), dims=obj.dims())
    elif isinstance(obj, QuantumChannel):
        y0, y0_cls = Array(SuperOp(obj).data), SuperOp
        wrapper = lambda x: SuperOp(
            np.array(x), input_dims=obj.input_dims(), output_dims=obj.output_dims()
        )
    elif isinstance(obj, (BaseOperator, Gate, QuantumCircuit)):
        y0, y0_cls = Array(Operator(obj.data)), Operator
        wrapper = lambda x: Operator(
            np.array(x), input_dims=obj.input_dims(), output_dims=obj.output_dims()
        )
    else:
        y0, y0_cls, wrapper = Array(obj), None, lambda x: x

    return y0, y0_cls, wrapper


def validate_and_format_initial_state(y0: any, model: Union[HamiltonianModel, LindbladModel]):
    """Format initial state for simulation. This function encodes the logic of how
    simulations are run based on initial state type.

    Args:
        y0: The user-specified input state.
        model: The model contained in the solver.

    Returns:
        Tuple containing the input state to pass to the solver, the user-specified input
        as an array, the class of the user specified input, and a function for converting
        the output states to the right class.

    Raises:
        QiskitError: Initial state ``y0`` is of invalid shape relative to the model.
    """

    if isinstance(y0, QuantumState) and isinstance(model, LindbladModel):
        y0 = DensityMatrix(y0)

    y0, y0_cls, wrapper = initial_state_converter(y0)

    y0_input = y0

    # validate types
    if (y0_cls is SuperOp) and is_lindblad_model_not_vectorized(model):
        raise QiskitError(
            """Simulating SuperOp for a LindbladModel requires setting
            vectorized evaluation. Set LindbladModel.evaluation_mode to a vectorized option.
            """
        )

    # if Simulating density matrix or SuperOp with a HamiltonianModel, simulate the unitary
    if y0_cls in [DensityMatrix, SuperOp] and isinstance(model, HamiltonianModel):
        y0 = np.eye(model.dim, dtype=complex)
    # if LindbladModel is vectorized and simulating a density matrix, flatten
    elif (
        (y0_cls is DensityMatrix)
        and isinstance(model, LindbladModel)
        and "vectorized" in model.evaluation_mode
    ):
        y0 = y0.flatten(order="F")

    # validate y0 shape before passing to solve_lmde
    if isinstance(model, HamiltonianModel) and (y0.shape[0] != model.dim or y0.ndim > 2):
        raise QiskitError("""Shape mismatch for initial state y0 and HamiltonianModel.""")
    if is_lindblad_model_vectorized(model) and (y0.shape[0] != model.dim**2 or y0.ndim > 2):
        raise QiskitError(
            """Shape mismatch for initial state y0 and LindbladModel
                             in vectorized evaluation mode."""
        )
    if is_lindblad_model_not_vectorized(model) and y0.shape[-2:] != (
        model.dim,
        model.dim,
    ):
        raise QiskitError("""Shape mismatch for initial state y0 and LindbladModel.""")

    return y0, y0_input, y0_cls, wrapper


def format_final_states(y, model, y0_input, y0_cls):
    """Format final states for a single simulation."""

    y = Array(y)

    if y0_cls is DensityMatrix and isinstance(model, HamiltonianModel):
        # conjugate by unitary
        return y @ y0_input @ y.conj().transpose((0, 2, 1))
    elif y0_cls is SuperOp and isinstance(model, HamiltonianModel):
        # convert to SuperOp and compose
        return (
            np.einsum("nka,nlb->nklab", y.conj(), y).reshape(
                y.shape[0], y.shape[1] ** 2, y.shape[1] ** 2
            )
            @ y0_input
        )
    elif (y0_cls is DensityMatrix) and is_lindblad_model_vectorized(model):
        return y.reshape((len(y),) + y0_input.shape, order="F")

    return y


def t_span_to_list(t_span):
    """Check if t_span is validly specified as a single interval or a list of intervals,
    and return as a list in either case."""
    was_list = False
    t_span_ndim = _nested_ndim(t_span)
    if t_span_ndim > 2:
        raise QiskitError("t_span must be either 1d or 2d.")
    if t_span_ndim == 1:
        t_span = [t_span]
    else:
        was_list = True

    return t_span, was_list


def _y0_to_list(y0):
    """Check if y0 is validly specified as a single initial state or a list of initial states,
    and return as a list in either case."""
    was_list = False
    if not isinstance(y0, list):
        y0 = [y0]
    else:
        was_list = True

    return y0, was_list


def _signals_to_list(signals):
    """Check if signals is validly specified as a single signal specification or a list of
    such specifications, and return as a list in either case."""
    was_list = False
    if signals is None:
        signals = [signals]
    elif isinstance(signals, tuple):
        # single Lindblad
        signals = [signals]
    elif isinstance(signals, list) and isinstance(signals[0], tuple):
        # multiple lindblad
        was_list = True
    elif isinstance(signals, (Schedule, ScheduleBlock)):
        # pulse simulation
        signals = [signals]
    elif isinstance(signals, list) and isinstance(signals[0], (Schedule, ScheduleBlock)):
        # multiple pulse simulation
        was_list = True
    elif isinstance(signals, list) and isinstance(signals[0], (list, SignalList)):
        # multiple Hamiltonian signals lists
        was_list = True
    elif isinstance(signals, SignalList) or (
        isinstance(signals, list) and not isinstance(signals[0], (list, SignalList))
    ):
        # single Hamiltonian signals list
        signals = [signals]
    else:
        raise QiskitError("Signals specified in invalid format.")

    return signals, was_list


def organize_signals_to_channels(
    all_signals, all_channels, model_class, hamiltonian_channels, dissipator_channels
):
    """Restructures a list of signals with order corresponding to all_channels, into the correctly
    formatted data structure to pass into model.signals, according to the ordering specified
    by hamiltonian_channels and dissipator_channels.
    """

    if model_class == HamiltonianModel:
        if hamiltonian_channels is not None:
            return [all_signals[all_channels.index(chan)] for chan in hamiltonian_channels]
        else:
            return None
    else:
        hamiltonian_signals = None
        dissipator_signals = None
        if hamiltonian_channels is not None:
            hamiltonian_signals = [
                all_signals[all_channels.index(chan)] for chan in hamiltonian_channels
            ]
        if dissipator_channels is not None:
            dissipator_signals = [
                all_signals[all_channels.index(chan)] for chan in dissipator_channels
            ]
        return (hamiltonian_signals, dissipator_signals)


def _nested_ndim(x):
    """Determine the 'ndim' of x, which could be composed of nested lists and array types."""
    if isinstance(x, (list, tuple)):
        return 1 + _nested_ndim(x[0])
    elif issubclass(type(x), Dispatch.REGISTERED_TYPES) or isinstance(x, Array):
        return x.ndim

    # assume scalar
    return 0

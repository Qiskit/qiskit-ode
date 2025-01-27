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

"""
Pulse schedule to Signals converter.
"""

from typing import Callable, Dict, List, Optional
import functools
from warnings import warn

import numpy as np
import sympy as sym

from qiskit_dynamics.pulse import (
    Schedule,
    Play,
    ShiftPhase,
    SetPhase,
    ShiftFrequency,
    SetFrequency,
    Waveform,
    MeasureChannel,
    DriveChannel,
    ControlChannel,
    AcquireChannel,
)
from qiskit_dynamics.pulse.exceptions import PulseError
from qiskit_dynamics.pulse.library import SymbolicPulse
from qiskit import QiskitError

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import ArrayLike

from qiskit_dynamics.signals import DiscreteSignal

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class InstructionToSignals:
    """Converts pulse instructions to signals to be used in models.

    The :class:`InstructionsToSignals` class converts a pulse schedule to a list of signals that can
    be given to a model. This conversion is done by calling the :meth:`get_signals` method on a
    schedule. The converter applies to instances of :class:`~qiskit_dynamics.pulse.Schedule`. Instances of
    :class:`~qiskit_dynamics.pulse.ScheduleBlock` must first be converted to :class:`~qiskit_dynamics.pulse.Schedule`
    using the :func:`~qiskit_dynamics.pulse.transforms.block_to_schedule` function in Qiskit Pulse.

    The converter can be initialized with the optional arguments ``carriers`` and ``channels``. When
    ``channels`` is given, only the signals specified by name in ``channels`` are returned. The
    ``carriers`` dictionary specifies the analog carrier frequency of each channel. Here, the keys
    are the channel name, e.g. ``d12`` for drive channel number ``12``, and the values are the
    corresponding frequency. If a channel is not present in ``carriers`` it is assumed that the
    analog carrier frequency is zero.

    See the :meth:`get_signals` method documentation for a detailed description of how pulse
    schedules are interpreted and translated into :class:`.DiscreteSignal` objects.
    """

    def __init__(
        self,
        dt: float,
        carriers: Optional[Dict[str, float]] = None,
        channels: Optional[List[str]] = None,
    ):
        """Initialize pulse schedule to signals converter.

        Args:
            dt: Length of the samples. This is required by the converter as pulse schedule are
                specified in units of dt and typically do not carry the value of dt with them.
            carriers: A dict of analog carrier frequencies. The keys are the names of the channels
                and the values are the corresponding carrier frequency.
            channels: A list of channels that the :meth:`get_signals` method should return. This
                argument will cause :meth:`get_signals` to return the signals in the same order as
                the channels. Channels present in the schedule but absent from channels will not be
                included in the returned object. If None is given (the default) then all channels
                present in the pulse schedule are returned.
        """

        self._dt = dt
        self._channels = channels
        self._carriers = carriers or {}

    def get_signals(self, schedule: Schedule) -> List[DiscreteSignal]:
        r"""Convert a schedule to a corresponding list of DiscreteSignal instances.

        Which channels are converted, and the order they are returned, is controlled by the
        ``channels`` argument at instantiation. The ``carriers`` instantiation argument sets the
        analog carrier frequency for each channel, which is fixed for the full duration. For a given
        channel, the :math:`k^{th}` envelope sample for the corresponding :class:`.DiscreteSignal`
        is determined according to the following formula:

        .. math::
            f(k) \exp(i(2\pi \Delta\nu(k) k dt + \phi(k) + 2 \pi \phi_a(k))),

        where:

        * :math:`f(k)` is the waveform value at the :math:`k^{th}` time step as specified by
          ``Play`` instructions.
        * :math:`\Delta\nu(k)` is the frequency deviation at time step :math:`k` from the analog
          carrier as the result of ``SetFrequency`` and ``ShiftFrequency`` instructions. As evident
          by the formula, carrier frequency deviations as a result of these instructions are handled
          digitally, with the analog carrier frequency being fixed for the entirety of the schedule.
        * :math:`dt` is the sample rate as specified by the ``dt`` instantiation argument.
        * :math:`\phi(k)` is the channel phase at time step :math:`k`, as determined by
          ``ShiftPhase`` and ``SetPhase`` instructions.
        * :math:`\phi_a(k)` is the phase correction term at time step :math:`k`, impacted by
          ``SetFrequency`` and ``ShiftFrequency`` instructions, described below.

        In detail, the sample array for the output signal for each channel is generated by iterating
        over each instruction in the schedule in temporal order. New samples are appended with every
        ``Play`` instruction on the given channel, using the waveform values and the current value
        of the tracked parameters :math:`\Delta\nu`, :math:`\phi`, and :math:`\phi_a`, which are
        initialized to :math:`0`. Explicitly, each instruction is interpreted as follows:

        * ``Play`` instructions add new samples to the sample array, according to the above formula,
          using the waveform specified in the instruction and the current values of
          :math:`\Delta\nu`, :math:`\phi`, and :math:`\phi_a`.
        * ``ShiftPhase``, with a phase value :math:`\psi`, updates :math:`\phi \mapsto \phi + \psi`.
        * ``SetPhase``, with a phase value :math:`\psi`, updates :math:`\phi \mapsto \psi`.
        * ``ShiftFrequency``, with a frequency value :math:`\mu` at time-step :math:`k`, updates
          :math:`\phi_a \mapsto \phi_a - \mu k dt` and :math:`\Delta\nu \mapsto \Delta\nu + \mu`.
          The simultaneous shifting of both :math:`\Delta\nu` and :math:`\phi_a` ensures that the
          carrier wave, as a combination of the analog and digital components, is continuous across
          ``ShiftFrequency`` instructions (up to the sampling rate :math:`dt`).
        * ``SetFrequency``, with a frequency value :math:`\mu` at time-step :math:`k`, updates
          :math:`\phi_a \mapsto \phi_a - (\mu - (\Delta\nu + \nu)) k dt` and
          :math:`\Delta\nu \mapsto \mu - \nu`, where :math:`\nu` is the analog carrier frequency.
          Similarly to ``ShiftFrequency``, the shift rule for :math:`\phi_a` is defined to maintain
          carrier wave continuity.

        If, at any sample point :math:`k`, :math:`\Delta\nu(k)` is larger than the Nyquist sampling
        rate given by ``dt``, a warning will be raised.

        Args:
            schedule: The schedule to represent in terms of signals. Instances of
                :class:`~qiskit_dynamics.pulse.ScheduleBlock` must first be converted to
                :class:`~qiskit_dynamics.pulse.Schedule` using the
                :func:`~qiskit_dynamics.pulse.transforms.block_to_schedule` function in Qiskit Pulse.

        Returns:
            A list of :class:`.DiscreteSignal` instances.
        """

        signals, phases, frequency_shifts, phase_accumulations = {}, {}, {}, {}

        if self._channels is not None:
            schedule = schedule.filter(channels=[self._get_channel(ch) for ch in self._channels])

        for chan in schedule.channels:
            phases[chan.name] = 0.0
            frequency_shifts[chan.name] = 0.0
            phase_accumulations[chan.name] = 0.0

            carrier_freq = self._carriers.get(chan.name, 0.0)

            signals[chan.name] = DiscreteSignal(
                samples=[],
                dt=self._dt,
                name=chan.name,
                carrier_freq=carrier_freq,
            )

        for start_sample, inst in schedule.instructions:
            # get channel name if instruction has it
            chan = inst.channel.name if hasattr(inst, "channel") else None

            if isinstance(inst, Play):
                # get the instruction samples
                inst_samples = None
                if isinstance(inst.pulse, Waveform):
                    inst_samples = inst.pulse.samples
                else:
                    inst_samples = get_samples(inst.pulse)

                # build sample array to append to signal
                times = self._dt * (start_sample + np.arange(len(inst_samples)))
                samples = inst_samples * unp.exp(
                    2.0j * np.pi * frequency_shifts[chan] * times
                    + 1.0j * phases[chan]
                    + 2.0j * np.pi * phase_accumulations[chan]
                )
                signals[chan].add_samples(start_sample, samples)

            if isinstance(inst, ShiftPhase):
                phases[chan] += inst.phase

            if isinstance(inst, SetPhase):
                phases[chan] = inst.phase

            if isinstance(inst, ShiftFrequency):
                frequency_shifts[chan] = frequency_shifts[chan] + inst.frequency
                phase_accumulations[chan] = (
                    phase_accumulations[chan] - inst.frequency * start_sample * self._dt
                )
                _nyquist_warn(frequency_shifts[chan], self._dt, chan)

            if isinstance(inst, SetFrequency):
                phase_accumulations[chan] = phase_accumulations[chan] - (
                    (inst.frequency - (frequency_shifts[chan] + signals[chan].carrier_freq))
                    * start_sample
                    * self._dt
                )
                frequency_shifts[chan] = inst.frequency - signals[chan].carrier_freq
                _nyquist_warn(frequency_shifts[chan], self._dt, chan)

        # ensure all signals have the same number of samples
        max_duration = 0
        for sig in signals.values():
            max_duration = max(max_duration, sig.duration)

        for sig in signals.values():
            if sig.duration < max_duration:
                sig.add_samples(
                    start_sample=sig.duration,
                    samples=np.zeros(max_duration - sig.duration, dtype=complex),
                )

        # filter the channels
        if self._channels is None:
            return list(signals.values())

        return_signals = []
        for chan_name in self._channels:
            signal = signals.get(
                chan_name, DiscreteSignal(samples=[], dt=self._dt, name=chan_name, carrier_freq=0.0)
            )

            return_signals.append(signal)
        return return_signals

    @staticmethod
    def get_awg_signals(
        signals: List[DiscreteSignal], if_modulation: float
    ) -> List[DiscreteSignal]:
        r"""
        Create signals that correspond to the output ports of an Arbitrary Waveform Generator to be
        used with IQ mixers. For each signal in the list the number of signals is double to create
        the I and Q components. The I and Q signals represent the real and imaginary parts,
        respectively, of

        .. math::
            \Omega(t) e^{i \omega_{if} t}

        where :math:`\Omega` is the complex-valued pulse envelope and :math:`\omega_{if}` is the
        intermediate frequency.

        Args:
            signals: A list of signals for which to create I and Q.
            if_modulation: The intermediate frequency with which the AWG modulates the pulse
                envelopes.

        Returns:
            iq signals: A list of signals which is twice as long as the input list of signals. For
                each input signal get_awg_signals returns two
        """
        new_signals = []

        for sig in signals:
            new_freq = sig.carrier_freq + if_modulation

            samples_i = sig.samples
            samples_q = unp.imag(samples_i) - 1.0j * unp.real(samples_i)

            sig_i = DiscreteSignal(
                sig.dt,
                samples_i,
                sig.start_time,
                new_freq,
                sig.phase,
                sig.name + "_i",
            )
            sig_q = DiscreteSignal(
                sig.dt,
                samples_q,
                sig.start_time,
                new_freq,
                sig.phase,
                sig.name + "_q",
            )

            new_signals += [sig_i, sig_q]

        return new_signals

    def _get_channel(self, channel_name: str):
        """Return the channel corresponding to the given name."""

        try:
            prefix = channel_name[0]
            index = int(channel_name[1:])

            if prefix == "d":
                return DriveChannel(index)

            if prefix == "m":
                return MeasureChannel(index)

            if prefix == "u":
                return ControlChannel(index)

            if prefix == "a":
                return AcquireChannel(index)

            raise QiskitError(
                f"Unsupported channel name {channel_name} in {self.__class__.__name__}"
            )

        except (KeyError, IndexError, ValueError) as error:
            raise QiskitError(
                f"Invalid channel name {channel_name} given to {self.__class__.__name__}."
            ) from error


def get_samples(pulse: SymbolicPulse) -> ArrayLike:
    """Return samples filled according to the formula that the pulse represents and the parameter
    values it contains.

    Args:
        pulse: SymbolicPulse class.
    Returns:
        Samples of the pulse.
    Raises:
        PulseError: When parameters are not assigned.
        PulseError: When expression for pulse envelope is not assigned.
        PulseError: When a free symbol value is not defined in the pulse instance parameters.
    """
    envelope = pulse.envelope
    pulse_params = pulse.parameters
    if pulse.is_parameterized():
        raise PulseError("Unassigned parameter exists. All parameters must be assigned.")

    if envelope is None:
        raise PulseError("Pulse envelope expression is not assigned.")

    args = []
    try:
        backend = (
            "jax" if any(isinstance(v, jax.core.Tracer) for v in pulse_params.values()) else "numpy"
        )
    except (ImportError, NameError):
        backend = "numpy"
    for symbol in sorted(envelope.free_symbols, key=lambda s: s.name):
        if symbol.name == "t":
            times = unp.arange(0, pulse_params["duration"]) + 1 / 2
            args.insert(0, times)
            continue
        try:
            args.append(pulse_params[symbol.name])
        except KeyError as ex:
            raise PulseError(
                f"Pulse parameter '{symbol.name}' is not defined for this instance. "
                "Please check your waveform expression is correct."
            ) from ex
    return _lru_cache_expr(
        envelope,
        backend,
    )(*args)


@functools.lru_cache(maxsize=None)
def _lru_cache_expr(expr: sym.Expr, backend) -> Callable:
    """A helper function to get lambdified expression.

    Args:
        expr: Symbolic expression to evaluate.
        backend: Array backend.
    Returns:
        lambdified expression.
    """
    params = []
    for param in sorted(expr.free_symbols, key=lambda s: s.name):
        if param.name == "t":
            params.insert(0, param)
            continue
        params.append(param)
    return sym.lambdify(params, expr, modules=backend)


def _nyquist_warn(frequency_shift: ArrayLike, dt: float, channel: str):
    """Raise a warning if the frequency shift is above the Nyquist frequency given by ``dt``."""
    if (
        isinstance(frequency_shift, (int, float, list, np.ndarray))
        or not isinstance(jnp.array(0), jax.core.Tracer)
    ) and np.abs(frequency_shift) > 0.5 / dt:
        warn(
            "Due to SetFrequency and ShiftFrequency instructions, the digital carrier frequency "
            f"of channel {channel} is larger than the Nyquist frequency of the envelope sample "
            "size dt. As shifts of the frequency from the analog frequency are handled digitally, "
            "this will result in aliasing effects."
        )

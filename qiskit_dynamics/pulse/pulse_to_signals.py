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

"""
Pulse schedule to Signals converter.
"""

from typing import List
import numpy as np

from qiskit.pulse import (
    Schedule,
    Play,
    ShiftPhase,
    SetPhase,
    ShiftFrequency,
    SetFrequency,
    Waveform,
)
from qiskit import QiskitError
from qiskit_dynamics.signals import DiscreteSignal


class InstructionToSignals:
    """Converts pulse instructions to Signals to be used in models.

    The :class:`InstructionsToSignals` class converts a pulse schedule to a list
    of signals that can be given to a model. This conversion is done by calling
    the :meth:`get_signals` method on a schedule.
    """

    def __init__(self, dt: float, carriers: List[float] = None):
        """Initialize pulse schedule to signals converter.

        Args:
            dt: length of the samples. This is required by the converter as pulse
                schedule are specified in units of dt and typically do not carry the
                value of dt with them.
            carriers: a list of carrier frequencies. If it is not None there
                must be at least as many carrier frequencies as there are
                channels in the schedules that will be converted.
        """

        self._dt = dt
        self._carriers = carriers

    def get_signals(self, schedule: Schedule) -> List[DiscreteSignal]:
        """
        Args:
            schedule: The schedule to represent in terms of signals.

        Returns:
            a list of piecewise constant signals.

        Raises:
            qiskit.QiskitError: if not enough frequencies supplied
        """

        if self._carriers and len(self._carriers) < len(schedule.channels):
            raise QiskitError("Not enough carrier frequencies supplied.")

        signals, phases, frequency_shifts = {}, {}, {}

        for idx, chan in enumerate(schedule.channels):
            if self._carriers:
                carrier_freq = self._carriers[idx]
            else:
                carrier_freq = 0.0

            phases[chan.name] = 0.0
            frequency_shifts[chan.name] = 0.0
            signals[chan.name] = DiscreteSignal(
                samples=[], dt=self._dt, name=chan.name, carrier_freq=carrier_freq
            )

        for start_sample, inst in schedule.instructions:
            chan = inst.channel.name
            phi = phases[chan]
            freq = frequency_shifts[chan]

            if isinstance(inst, Play):
                start_idx = len(signals[chan].samples)

                # get the instruction samples
                inst_samples = None
                if isinstance(inst.pulse, Waveform):
                    inst_samples = inst.pulse.samples
                else:
                    inst_samples = inst.pulse.get_waveform().samples

                # build sample array to append to signal
                samples = []
                for idx, sample in enumerate(inst_samples):
                    time = self._dt * (idx + start_idx)
                    samples.append(sample * np.exp(2.0j * np.pi * freq * time + 1.0j * phi))

                signals[chan].add_samples(start_sample, samples)

            if isinstance(inst, ShiftPhase):
                phases[chan] += inst.phase

            if isinstance(inst, ShiftFrequency):
                frequency_shifts[chan] += inst.frequency

            if isinstance(inst, SetPhase):
                phases[chan] = inst.phase

            if isinstance(inst, SetFrequency):
                frequency_shifts[chan] = inst.frequency - signals[chan].carrier_freq

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

        return list(signals.values())

    @staticmethod
    def get_awg_signals(
        signals: List[DiscreteSignal], if_modulation: float
    ) -> List[DiscreteSignal]:
        r"""
        Create signals that correspond to the output ports of an Arbitrary Waveform Generator
        to be used with IQ mixers. For each signal in the list the number of signals is double
        to create the I and Q components. The I and Q signals represent the real and imaginary
        parts, respectively, of

        .. math::
            \Omega(t) e^{i \omega_{if} t}

        where :math:`\Omega` is the complex-valued pulse envelope and :math:`\omega_{if}` is the
        intermediate frequency.

        Args:
            signals: A list of signals for which to create I and Q.
            if_modulation: The intermediate frequency with which the AWG modulates the pulse
                envelopes.

        Returns:
            iq signals: A list of signals which is twice as long as the input list of signals.
                For each input signal get_awg_signals returns two
        """
        new_signals = []

        for sig in signals:
            new_freq = sig.carrier_freq + if_modulation

            samples_i = sig.samples
            samples_q = np.imag(samples_i) - 1.0j * np.real(samples_i)

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

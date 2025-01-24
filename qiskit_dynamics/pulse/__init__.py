# -*- coding: utf-8 -*-

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
====================================
Pulse (:mod:`qiskit_dynamics.pulse`)
====================================

.. currentmodule:: qiskit_dynamics.pulse

This module contains tools to interface :mod:`qiskit.pulse` with Qiskit Dynamics. Qiskit Dynamics
simulates time evolution using the :class:`Signal` class, however :mod:`qiskit.pulse` specifies
pulse instructions using a :class:`~qiskit.pulse.Schedule` or :class:`~qiskit.pulse.ScheduleBlock`.
This module contains the required converters to convert from a :mod:`qiskit.pulse` control
specification into :class:`Signal` instances for simulation.

Converters
==========

The conversion from a :class:`~qiskit.pulse.Schedule` to a list of :class:`Signal` instances is done
with the :class:`InstructionToSignals` converter. The following codeblock shows a simple example
instantiation, and how to use it to convert a :class:`~qiskit.pulse.Schedule` to a list of
:class:`Signal` instances.

.. code-block:: python

    converter = InstructionToSignals(dt=1, carriers=None)
    signals = converter.get_signals(sched)

An example schedule, and the corresponding converted signals, is shown below.

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    import qiskit.pulse as pulse

    from qiskit_dynamics.pulse import InstructionToSignals


    with pulse.build(name="schedule") as sched:
        pulse.play(pulse.Drag(20, 0.5, 4, 0.5), pulse.DriveChannel(0))
        pulse.shift_phase(1.0, pulse.DriveChannel(0))
        pulse.play(pulse.Drag(20, 0.5, 4, 0.5), pulse.DriveChannel(0))
        pulse.shift_frequency(0.5, pulse.DriveChannel(0))
        pulse.play(pulse.GaussianSquare(200, 0.3, 4, 150), pulse.DriveChannel(0))
        pulse.play(pulse.GaussianSquare(200, 0.3, 4, 150), pulse.DriveChannel(1))

    fig = plt.figure(constrained_layout=True, figsize=(10, 7))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, :])
    ax2 = fig.add_subplot(spec[1, 0])
    ax3 = fig.add_subplot(spec[1, 1])

    converter = InstructionToSignals(dt=1, carriers=None)

    signals = converter.get_signals(sched)

    signals[0].draw(0, 239, 400, axis=ax2, title="Signal from DriveChannel(0)")
    signals[1].draw(0, 239, 400, axis=ax3, title="Signal from DriveChannel(1)")
    sched.draw(axis=ax1)

Converter class
===============

.. autosummary::
   :toctree: ../stubs/

   InstructionToSignals
"""
from .pulse_to_signals import InstructionToSignals

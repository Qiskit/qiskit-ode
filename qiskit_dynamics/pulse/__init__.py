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
==============================================
Pulse (:mod:`qiskit_dynamics.pulse`)
==============================================

.. currentmodule:: qiskit_dynamics.pulse

This module contains tools to interface qiskit pulse with qiskit dynamics. Indeed,
Qiskit-dynamics simulates time evolution using the :class:`Signal` class. However,
Qiskit pulse specifies pulse instructions using a schedule. This module contains
the required converters to convert Qiskit pulse schedules into signals such that
they can be simulated.

Converters
==========

The conversion from a pulse schedule to a list of signals is done with the
:class:`InstructionToSignals` converter. The example below shows a schedule and
the resulting signals converted using

.. code-block:: python

    converter = InstructionToSignals(dt=1, carriers=None)
    signals = converter.get_signals(sched)

.. jupyter-execute::
    :hide-code:

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

    converter.get_signals(sched)[0].draw(0, 239, 400, axis=ax2, title="Signal from DriveChannel(0)")
    converter.get_signals(sched)[1].draw(0, 239, 400, axis=ax3, title="Signal from DriveChannel(1)")
    sched.draw(axis=ax1)

Qiskit Pulse
============

.. autosummary::
   :toctree: ../stubs/

   InstructionToSignals
"""

from .pulse_to_signals import InstructionToSignals

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

r"""
========================================
Signals (:mod:`qiskit_dynamics.signals`)
========================================

.. currentmodule:: qiskit_dynamics.signals

This module contains classes for representing the time-dependent coefficients in
matrix differential equations.

These classes, referred to as *signals*, represent classes of real-valued functions, either
of the form, or built from functions of the following form:

.. math::
    s(t) = \textnormal{Re}[f(t)e^{i(2 \pi \nu t + \phi)}],

where

    * :math:`f` is a complex-valued function called the *envelope*,
    * :math:`\nu \in \mathbb{R}` is the *carrier frequency*, and
    * :math:`\phi \in \mathbb{R}` is the *phase*.

Furthermore, this module contains *transfer functions* which transform one or more signal
into other signals.

Signal API summary
==================

All signal classes share a common API for evaluation and visualization:

    * The signal value at a given time ``t`` is evaluated by treating the ``signal`` as a
      callable: ``signal(t)``.
    * The envelope :math:`f(t)` is evaluated via: ``signal.envelope(t)``.
    * The complex value :math:`f(t)e^{i(2 \pi \nu t + \phi)}` via: ``signal.complex_value(t)``.
    * The ``signal.draw`` method provides a common visualization interface.

In addition to the above, all signal types allow for algebraic operations, which should be
understood in terms of algebraic operations on functions. E.g. two signals can be added
together via

.. code-block:: python

    signal_sum = signal1 + signal2

and satisfy

.. code-block:: python

    signal_sum(t) == signal1(t) + signal2(t)

Signal multiplication is defined similarly, and signals can be added or multiplied with constants
as well.

The remainder of this document gives further detail about some special functionality of
these classes, but the following table provides a list of the different signal classes,
along with a high level description of their role.

.. list-table:: Types of signal objects
   :widths: 10 50
   :header-rows: 1

   * - Class name
     - Description
   * - :class:`~qiskit_dynamics.signals.Signal`
     - Envelope specified as a python ``Callable``, allowing for complete generality.
   * - :class:`~qiskit_dynamics.signals.DiscreteSignal`
     - Piecewise constant envelope, implemented with array-based operations, geared towards
       performance.
   * - :class:`~qiskit_dynamics.signals.SignalSum`
     - A sum of :class:`~qiskit_dynamics.signals.Signal` or
       :class:`~qiskit_dynamics.signals.DiscreteSignal` objects.
       Evaluation of envelopes returns an array of envelopes in the sum.
   * - :class:`~qiskit_dynamics.signals.DiscreteSignalSum`
     - A sum of :class:`~qiskit_dynamics.signals.DiscreteSignal` objects with the same
       start time, number of samples, and sample duration. Implemented with array-based operations.


Constant Signal
===============

:class:`~qiskit_dynamics.signals.Signal` supports specification of a *constant signal*:

.. code-block:: python

    const = Signal(2.)

This initializes the object to always return the constant ``2.``, and allows constants to be
treated on the same footing as arbitrary :class:`~qiskit_dynamics.signals.Signal` instances.
A :class:`~qiskit_dynamics.signals.Signal` operating in constant-mode can be checked via the
boolean attribute ``const.is_constant``.


Algebraic operations
====================

Algebraic operations are supported by the :class:`~qiskit_dynamics.signals.SignalSum`
object. Any two signal classes can be added together, producing a
:class:`~qiskit_dynamics.signals.SignalSum`. Multiplication is also supported
via :class:`~qiskit_dynamics.signals.SignalSum` using the identity:

.. math::

    Re[f(t)e^{i(2 \pi \nu t + \phi)}] \times &Re[g(t)e^{i(2 \pi \omega t + \psi)}]
         \\&= Re[\frac{1}{2} f(t)g(t)e^{i(2\pi (\omega + \nu)t + (\phi + \psi))} ]
          + Re[\frac{1}{2} f(t)\overline{g(t)}e^{i(2\pi (\omega - \nu)t + (\phi - \psi))} ].

I.e. multiplication of two base signals produces a :class:`~qiskit_dynamics.signals.SignalSum`
with two elements, whose envelopes, frequencies, and phases are as given by the above formula.
Multiplication of sums is handled via distribution of this formula over the sum.

In the special case that
:class:`~qiskit_dynamics.signals.DiscreteSignal`\s with compatible sample structure
(same number of samples, ``dt``, and start time) are added together,
a :class:`~qiskit_dynamics.signals.DiscreteSignalSum` is produced.
:class:`~qiskit_dynamics.signals.DiscreteSignalSum` stores a sum of compatible
:class:`~qiskit_dynamics.signals.DiscreteSignal`\s by joining the underlying arrays,
so that the sum can be evaluated using purely array-based operations. Multiplication
of :class:`~qiskit_dynamics.signals.DiscreteSignal`\s with compatible sample structure
is handled similarly.

Sampling
========

Both :class:`~qiskit_dynamics.signals.DiscreteSignal` and
:class:`~qiskit_dynamics.signals.DiscreteSignalSum` feature constructors
(:meth:`~qiskit_dynamics.signals.DiscreteSignal.from_Signal` and
:meth:`~qiskit_dynamics.signals.DiscreteSignalSum.from_SignalSum` respectively)
which build an instance by sampling a :class:`~qiskit_dynamics.signals.Signal` or
:class:`~qiskit_dynamics.signals.SignalSum`. These constructors have the
option to just sample the envelope (and keep the carrier analog), or to also
sample the carrier. Below is a visualization of a signal superimposed with
sampled versions, both in the case of sampling the carrier, and in the case of
sampling just the envelope (and keeping the carrier analog).

.. jupyter-execute::
    :hide-code:

    from qiskit_dynamics.signals import Signal, DiscreteSignal
    from matplotlib import pyplot as plt

    # discretize a signal with and without samplying the carrier
    signal = Signal(lambda t: t, carrier_freq=2.)
    discrete_signal = DiscreteSignal.from_Signal(signal, dt=0.1, start_time=0.,
                                                 n_samples=10, sample_carrier=True)
    discrete_signal2 = DiscreteSignal.from_Signal(signal, dt=0.1, start_time=0., n_samples=10)

    # plot the signal against each discretization
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    signal.draw(t0=0., tf=1., n=100, axis=axs[0])
    discrete_signal.draw(t0=0., tf=1., n=100, axis=axs[0],
                         title='Signal v.s. Sampled envelope and carrier')
    signal.draw(t0=0., tf=1., n=100, axis=axs[1])
    discrete_signal2.draw(t0=0., tf=1., n=100, axis=axs[1],
                          title='Signal v.s. Sampled envelope')


Transfer Functions
==================

A transfer function is a mapping from one or more :class:`Signal` to one or more :class:`Signal`.
Transfer functions can, for example, be used to model the effect of the electronics finite
response. The code below shows the example of an :class:`IQMixer`. Here, two signals modulated at
100 MHz and with a relative :math:`\pi/2` phase shift are passed through an IQ-mixer with a
carrier frequency of 400 MHz to create a signal at 500 MHz. Note that the code below does not make
any assumptions about the time and frequency units which we interpret as ns and GHz, respectively.

.. jupyter-execute::

    import numpy as np
    from qiskit_dynamics.signals import DiscreteSignal, Sampler, IQMixer

    dt = 0.25
    in_phase = DiscreteSignal(dt, [1.0]*200, carrier_freq=0.1, phase=0)
    quadrature = DiscreteSignal(dt, [1.0]*200, carrier_freq=0.1, phase=np.pi/2)

    sampler = Sampler(dt/25, 5000)
    in_phase = sampler(in_phase)
    quadrature = sampler(quadrature)

    mixer = IQMixer(0.4)
    rf = mixer(in_phase, quadrature)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    in_phase.draw(0, 25, 100, axis=axs[0])
    quadrature.draw(0, 25, 100, axis=axs[0], title='In-phase and quadrature signals')
    rf.draw(0, 24, 2000, axis=axs[1], title='Mixer output')

Signal Classes
==============

.. autosummary::
   :toctree: ../stubs/

   Signal
   DiscreteSignal
   SignalSum
   DiscreteSignalSum
   SignalList

Transfer Function Classes
=========================

.. autosummary::
   :toctree: ../stubs/

   Convolution
   IQMixer
"""

from .signals import Signal, DiscreteSignal, SignalSum, DiscreteSignalSum, SignalList
from .transfer_functions import Convolution, Sampler, IQMixer

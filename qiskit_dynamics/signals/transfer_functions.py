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
# pylint: disable=invalid-name

"""
Module for transformations between signals.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union, List
import numpy as np

from qiskit import QiskitError
from qiskit_dynamics.dispatch import Array

from .signals import Signal, DiscreteSignal


class BaseTransferFunction(ABC):
    """Base class for transforming signals."""

    @property
    @abstractmethod
    def n_inputs(self):
        """Number of input signals to the transfer function."""
        pass

    def __call__(self, *args, **kwargs) -> Union[Signal, List[Signal]]:
        """
        Apply the transfer function to the input signals.

        Args:
            *args: The signals to which the transfer function will be applied.
            **kwargs: Key word arguments to control the transfer functions.

        Returns:
            Signal: The transformed signal.

        Raises:
            QiskitError: if the number of args is not correct.
        """

        if len(args) != self.n_inputs:
            raise QiskitError(
                self.__class__.__name__ + " expected %i input signals but %i "
                "were given" % (len(args), self.n_inputs)
            )

        return self._apply(*args, **kwargs)

    @abstractmethod
    def _apply(self, *args, **kwargs) -> Union[Signal, List[Signal]]:
        """
        Applies a transformation on a signal, such as a convolution,
        low pass filter, etc.

        Args:
            *args: The signals to which the transfer function will be applied.
            **kwargs: Key word arguments to control the transfer functions.

        Returns:
            Signal: The transformed signal.
        """
        pass


class Convolution(BaseTransferFunction):
    """Applies a convolution as a sum

        (f*g)(n) = sum_k f(k)g(n-k)

    The implementation is quadratic in the number of samples in the signal.
    """

    def __init__(self, func: Callable):
        """
        Args:
            func: The convolution function specified in time.
                  This function will be normalized to one before doing
                  the convolution. To scale signals multiply them by a float.
        """
        self._func = func

    @property
    def n_inputs(self):
        return 1

    # pylint: disable=arguments-differ
    def _apply(self, signal: Signal) -> Signal:
        """
        Applies a transformation on a signal, such as a convolution,
        low pass filter, etc. Once a convolution is applied the signal
        can longer have a carrier as the carrier is part of the signal
        value and gets convolved.

        Args:
            signal: A signal or list of signals to which the
                    transfer function will be applied.

        Returns:
            signal: The transformed signal or list of signals.

        Raises:
            QiskitError: if the signal is not pwc.
        """
        if isinstance(signal, DiscreteSignal):
            # Perform a discrete time convolution.
            dt = signal.dt
            func_samples = Array([self._func(dt * i) for i in range(signal.duration)])
            func_samples = func_samples / sum(func_samples)
            sig_samples = signal(dt * np.arange(signal.duration))

            convoluted_samples = list(np.convolve(func_samples, sig_samples))

            return DiscreteSignal(dt, convoluted_samples, carrier_freq=0.0, phase=0.0)
        else:
            raise QiskitError("Transfer function not defined on input.")


class FFTConvolution(BaseTransferFunction):
    """
    Applies a convolution by moving into the fourier domain.
    """

    def __init__(self, func: Callable):
        self._func = func

    @property
    def n_inputs(self):
        return 1

    # pylint: disable=arguments-differ
    def _apply(self, signal: Signal) -> Signal:
        raise NotImplementedError


class Sampler(BaseTransferFunction):
    """
    Re sample a signal by wrapping DiscreteSignal.from_Signal.
    """

    def __init__(self, dt: float, n_samples: int, start_time: float = 0):
        """
        Args:
            dt: The new sample period.
            n_samples: number of samples to resample with.
            start_time: start time from which to resample.
        """
        self._dt = dt
        self._n_samples = n_samples
        self._start_time = start_time

    @property
    def n_inputs(self):
        """Number of input signals to the transfer function."""
        return 1

    # pylint: disable=arguments-differ
    def _apply(self, signal: Signal) -> Signal:
        """Apply the transfer function to the signal."""
        return DiscreteSignal.from_Signal(
            signal, dt=self._dt, n_samples=self._n_samples, start_time=self._start_time
        )


class IQMixer(BaseTransferFunction):
    """
    Implements an IQ Mixer. The IQ mixer takes as input three signals:
    - in-phase signal: I cos(w_if t + phi_I)
    - quadrature: Q cos(w_if t + phi_Q)
    - local oscillator: K cos(w_lo t)

    In this implementation the local oscillator is specified by its frequency
    w_lo and, without loss of generality we assume K = 1. Furthermore, we
    require that the carrier frequency of the I and Q be identical.

    The output RF signal is defined by

    s_rf = I [cos(wp t + phi_I) + cos(wm t + phi_I)]/2
         + Q [cos(wp t + phi_Q - pi/2) + cos(wm t + phi_Q + pi/2)]/2

    where wp = w_lo + w_if and wp = w_lo - w_if.

    The output of this transfer function will produce a piece-wise constant
    that does not have a carrier frequency or phase. All information is in the
    samples. Mixer imperfections are not included.
    """

    def __init__(self, lo: float):
        """
        Args:
            lo: The carrier of the IQ mixer.
        """
        self._lo = lo

    @property
    def n_inputs(self):
        return 2

    # pylint: disable=arguments-differ
    def _apply(self, si: Signal, sq: Signal) -> Signal:
        """
        Args:
            si: In phase signal
            sq: Quadrature signal.

        Returns:
            The up-converted signal.

        Raises:
            QiskitError: if the carriers frequencies of I and Q differ.
        """

        # Check compatibility of the input signals
        if si.carrier_freq != sq.carrier_freq:
            raise QiskitError("IQ mixer requires the same sideband frequencies for I and Q.")

        phi_i, phi_q = si.phase, sq.phase
        wp, wm = self._lo + si.carrier_freq, self._lo - si.carrier_freq
        wp *= 2 * np.pi
        wm *= 2 * np.pi

        def mixer_func(t):
            """Function of the IQ mixer."""
            osc_i = np.cos(wp * t + phi_i) + np.cos(wm * t + phi_i)
            osc_q = np.cos(wp * t + phi_q - np.pi / 2) + np.cos(wm * t + phi_q + np.pi / 2)
            return si.envelope(t) * osc_i / 2 + sq.envelope(t) * osc_q / 2

        return Signal(mixer_func, carrier_freq=0, phase=0)

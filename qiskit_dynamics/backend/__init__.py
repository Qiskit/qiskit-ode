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

r"""
========================================
Backend (:mod:`qiskit_dynamics.backend`)
========================================

.. currentmodule:: qiskit_dynamics.backend

This module contains the :class:`.DynamicsBackend` class, which provides a
:class:`~qiskit.providers.backend.BackendV2` interface for running pulse-level simulations with
Qiskit Dynamics. The :class:`.DynamicsBackend` can directly simulate :class:`~qiskit_dynamics.pulse.Schedule`
and :class:`~qiskit_dynamics.pulse.ScheduleBlock` instances, and can also be configured to simulate
:class:`~qiskit.circuit.QuantumCircuit`\s at the pulse-level via circuit to pulse transpilation.

This module also exposes some functions utilized by :class:`.DynamicsBackend` that may be of use to
experienced users. The function :func:`.default_experiment_result_function` is the default method by
which results are computed and returned to the user after the underlying differential equation is
solved. This function can be overridden with a custom user-defined function by setting the
``experiment_result_function`` option of :class:`.DynamicsBackend`. The
:func:`.parse_backend_hamiltonian_dict` function is used by :meth:`.DynamicsBackend.from_backend` to
construct model matrices from the backend Hamiltonian description. The function documentation gives
a detailed explanation on the expected input formatting.


Classes and functions
=====================

.. autosummary::
   :toctree: ../stubs/

   DynamicsBackend
   default_experiment_result_function
   parse_backend_hamiltonian_dict
"""

from .dynamics_backend import DynamicsBackend, default_experiment_result_function
from .backend_string_parser import parse_backend_hamiltonian_dict

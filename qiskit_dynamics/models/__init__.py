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
======================================
Models (:mod:`qiskit_dynamics.models`)
======================================

.. currentmodule:: qiskit_dynamics.models

This module contains classes for constructing the right-hand side of an ordinary differential
equations. In this package, a "model of a quantum system" means a description of a differential
equation used to model a physical quantum system, in this case either the
*Schrodinger equation*:

.. math::
    \dot{y}(t) = -i H(t)y(t),

where :math:`H(t)` is the Hamiltonian, or *Lindblad equation*:

.. math::
    \dot{\rho}(t) = -i[H(t), \rho(t)] +
                \sum_j g_j(t) \left(L_j\rho(t)L_j^\dagger -
                \frac{1}{2}\{L_j^\dagger L_j, \rho(t)\}\right),

where the second term is called the *dissipator* term. Each :math:`L_j` is a dissipation operator
*dissipator*, and :math:`[\cdot, \cdot]` and :math:`\{\cdot, \cdot\}` are, respectively, the
matrix commutator and anti-commutator.

The classes for representing the Schrodinger and Lindblad equations are,
respectively, :class:`~qiskit_dynamics.models.HamiltonianModel` and
:class:`~qiskit_dynamics.models.LindbladModel`. Model classes primarily
serve a *computational* purpose, and expose functions for evaluating
model expressions, such as :math:`t \mapsto H(t)` or :math:`t,y \mapsto -iH(t)y` in the
case of a Hamiltonian, and with similar functionality for
:class:`~qiskit_dynamics.models.LindbladModel`.


Rotating frames
===============

Frame transformations are a common technique for solving time-dependent quantum differential
equations. For example, for a Hamiltonian, this corresponds to the transformation

.. math::
    H(t) \mapsto e^{iH_0t}(H(t) - H_0)e^{-iH_0t},

for a Hermitian operator :math:`H_0` called the *frame operator*.

.. note::
    The *frame operator* is commonly equivalently expressed as the corresponding anti-Hermitian
    operator under the association :math:`F = -iH_0`. This package refers to either :math:`F`
    or :math:`H_0` as the *frame operator*, with this association being understood.

Any model class can be transformed into a rotating frame by setting the
``rotating_frame`` property:

.. code-block:: python

    model.rotating_frame = frame_operator

where ``frame_operator`` is a specification of either :math:`H_0` or :math:`F = -iH_0`
(see the documentation for :class:`~qiskit_dynamics.models.RotatingFrame` for valid types
and behaviours).
Setting this property modifies the behaviour of the evaluation functions, e.g. a
:class:`~qiskit_dynamics.models.HamiltonianModel` will
compute :math:`e^{-tF}(-iH(t) - F)e^{tF}` in place of :math:`H(t)`.
:class:`~qiskit_dynamics.models.LindbladModel` has similar behaviour.

Internally, the model classes make use of the :class:`~qiskit_dynamics.models.RotatingFrame`
class, which is instantiated when the ``rotating_frame`` property is set. This class contains
helper functions for transforming various objects into and out of the rotating frame. This class
works directly with the anti-Hermitian form :math:`F = -iH_0`, however can be instantiated
with a Hermitian operator :math:`H_0` from which :math:`F` is automatically constructed.

Rotating wave approximation
===========================

The rotating wave approximation (RWA) is a transformation in which rapidly oscillating
time-dependent components, above a given cutoff frequency, are removed from a model.
This transformation is implemented in
:meth:`~qiskit_dynamics.models.rotating_wave_approximation`, see its documentation for
details.


Numerical methods and evaluation modes
======================================

All model classes offer different underlying numerical implementations that a user can choose
using the ``set_evaluation_mode`` method. For example,
:class:`~qiskit_dynamics.models.HamiltonianModel` can internally use
either sparse or dense arrays to compute :math:`H(t)` or a product :math:`-iH(t)y`.
The default is dense arrays, and a model can be set to use sparse arrays via:

.. code-block:: python

    model.set_evaluation_mode('sparse')

See ``set_evaluation_mode`` for each model class for available modes.

.. note::

    When setting a rotating frame, models internally store their operators
    in the basis in which the frame operator is diagonal. In general, sparsity of an operator
    is not perserved by basis transformations. Hence, preserving internal sparsity with
    rotating frames requires more restrictive choice of frames. For example, diagonal frame
    operators exactly preserve sparsity.


Model classes
=============

.. autosummary::
   :toctree: ../stubs/

   HamiltonianModel
   LindbladModel
   GeneratorModel

Model transformations
=====================

.. autosummary::
   :toctree: ../stubs/

   RotatingFrame
   rotating_wave_approximation
"""

from .rotating_frame import RotatingFrame
from .generator_models import BaseGeneratorModel, GeneratorModel
from .hamiltonian_models import HamiltonianModel
from .lindblad_models import LindbladModel
from .rotating_wave_approximation import rotating_wave_approximation

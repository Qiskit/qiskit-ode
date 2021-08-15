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
# pylint: disable=invalid-name,no-member,attribute-defined-outside-init
from collections import OrderedDict

from qiskit_dynamics.builders.dynamical_operators import *
from qiskit_dynamics.builders.operators_library import *
from qiskit_dynamics.signals import Signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os

N = 4
Jz = 1.

r_qubits = range(N)
subsystem_dims = OrderedDict()
H = .5 * Sz(0)

for i_qubit in r_qubits:
	subsystem_dims[i_qubit] = 2
	if i_qubit > 0:
		H += .5 * (Sz(i_qubit) + Sx(i_qubit))
		H += .5 * Jz * Sz(i_qubit - 1) * Sz(i_qubit)

H1_dict = build_dictionaries(H)
H1_matrix = build_matrices(H, subsystem_dims)

prune_subsystems = {2: 0, 3: 0}
subsystem_dims2 = subsystem_dims.copy()
for sys_id in prune_subsystems:
	subsystem_dims2.pop(sys_id)

H2_dict = build_dictionaries(H, prune_subsystems)
H2_matrix = build_matrices(H, subsystem_dims2, prune_subsystems)

tmp = 2


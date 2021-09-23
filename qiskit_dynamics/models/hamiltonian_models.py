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
Hamiltonian models module.
"""

from typing import Union, List, Optional
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.type_utils import to_array
from .generator_models import GeneratorModel, construct_operator_collection, setup_operators_in_frame
from .rotating_frame import RotatingFrame


class HamiltonianModel(GeneratorModel):
    r"""A model of a Hamiltonian for the Schrodinger equation.

    This class represents a Hamiltonian as a time-dependent decomposition the form:

    .. math::
        H(t) = H_d + \sum_j s_j(t) H_j,

    where :math:`H_j` are Hermitian operators, :math:`H_d` is the static component,
    and the :math:`s_j(t)` are either :class:`~qiskit_dynamics.signals.Signal` objects or
    are numerical constants. Constructing a :class:`~qiskit_dynamics.models.HamiltonianModel`
    requires specifying the above decomposition, e.g.:

    .. code-block:: python

        hamiltonian = HamiltonianModel(operators, signals, static_operator)

    This class inherits most functionality from :class:`GeneratorModel`,
    with the following modifications:

        * The operators :math:`H_d` and :math:`H_j` are assumed and verified to be Hermitian.
        * Rotating frames are dealt with assuming the structure of the Schrodinger
          equation. I.e. Evaluating the Hamiltonian :math:`H(t)` in a
          frame :math:`F = -iH_0`, evaluates the expression
          :math:`e^{-tF}H(t)e^{tF} - H_0`.
    """

    def __init__(
        self,
        operators: List[Operator],
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        static_operator: Optional[Array] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        evaluation_mode: str = "dense",
        validate: bool = True,
    ):
        """Initialize, ensuring that the operators are Hermitian.

        Args:
            operators: list of Operator objects.
            signals: List of coefficients :math:`s_i(t)`. Not required at instantiation, but
                     necessary for evaluation of the model.
            static_operator: Optional, time-independent term in the Hamiltonian.
            rotating_frame: Rotating frame operator.
                            If specified with a 1d array, it is interpreted as the
                            diagonal of a diagonal matrix. Assumed to store
                            the antihermitian matrix F = -iH.
            evaluation_mode: Evaluation mode to use. Supported options are:
                                - 'dense' (DenseOperatorCollection)
                                - 'sparse' (SparseOperatorCollection)
                                See ``GeneratorModel.evaluation_mode`` for more details.
            validate: If True check input operators are Hermitian.

        Raises:
            QiskitError: if operators are not Hermitian
        """

        # verify operators are Hermitian, and if so instantiate
        operators = to_array(operators)
        static_operator = to_array(static_operator)

        if validate:
            if (operators is not None) and (not is_hermitian(operators)):
                raise QiskitError("""HamiltonianModel operators must be Hermitian.""")
            if (static_operator is not None) and (not is_hermitian(static_operator)):
                raise QiskitError("""HamiltonianModel static_operator must be Hermitian.""")

        super().__init__(
            operators=operators,
            signals=signals,
            static_operator=static_operator,
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
        )

    @property
    def rotating_frame(self) -> RotatingFrame:
        return super().rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]) -> Array:
        """Sets frame. RotatingFrame objects will always store antihermitian F = -iH.
        The static_operator needs to be adjusted by -H in the new frame."""
        new_frame = RotatingFrame(rotating_frame)

        static_op = self.get_static_operator(in_frame_basis=True)
        if static_op is not None:
            static_op = -1j * static_op

        ops = self.get_operators(in_frame_basis=True)
        if ops is not None:
            ops = -1j * ops

        new_static_operator, new_operators = setup_operators_in_frame(
            static_op,
            ops,
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        self._rotating_frame = new_frame

        if new_static_operator is not None:
            new_static_operator = 1j * new_static_operator

        if new_operators is not None:
            new_operators = 1j * new_operators

        self._operator_collection = construct_operator_collection(
            self.evaluation_mode, new_static_operator, new_operators
        )

    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        return -1j * super().evaluate(time, in_frame_basis=in_frame_basis)

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        return -1j * super().evaluate_rhs(time, y, in_frame_basis=in_frame_basis)


def is_hermitian(operators: Array, tol: Optional[float] = 1e-10) -> bool:
    """Validate that operators are Hermitian.

    Args:
        operators: Either a 2d array representing a single operator, or a 3d array
                   representing a list of operators.

    Returns:
        bool: Whether or not the operators are Hermitian to within tolerance.
    """

    adj = None
    if operators.ndim == 2:
        adj = np.transpose(np.conjugate(operators))
    elif operators.ndim == 3:
        adj = np.transpose(np.conjugate(operators), (0, 2, 1))

    return np.linalg.norm(adj - operators) < 1e-10

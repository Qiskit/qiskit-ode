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
Generator models module.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union, List, Optional
from copy import copy
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.operator_collections import (
    DenseOperatorCollection,
    SparseOperatorCollection,
)
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.type_utils import to_array, to_csr
from .rotating_frame import RotatingFrame


class BaseGeneratorModel(ABC):
    r"""Defines an interface for a time-dependent
    linear differential equation of the form :math:`\dot{y}(t) = \Lambda(t, y)`,
    where :math:`\Lambda` is linear in :math:`y`. The core functionality is
    evaluation of :math:`\Lambda(t, y)`, as well as,
    if possible, evaluation of the map :math:`\Lambda(t, \cdot)`.

    Additionally, the class defines interfaces for:

        - Setting a "drift" term, representing the time-independent part of :math:`\Lambda`.
        - Setting a "rotating frame", specified either directly as a :class:`RotatingFrame`
        instance, or an operator from which a :class:`RotatingFrame` instance can be constructed.
        The exact meaning of this transformation is determined by the structure of
        :math:`\Lambda(t, y)`, and is therefore by handled by concrete subclasses.
        - Setting an internal "evaluation mode", to set the specific numerical methods to use
        when evaluating :math:`\Lambda(t, y)`.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Gets matrix dimension."""
        pass

    @property
    @abstractmethod
    def rotating_frame(self) -> RotatingFrame:
        """Get the rotating frame."""
        pass

    @rotating_frame.setter
    @abstractmethod
    def rotating_frame(self, rotating_frame: RotatingFrame):
        """Set the rotating frame; either an already instantiated :class:`RotatingFrame` object,
        or a valid argument for the constructor of :class:`RotatingFrame`.
        """
        pass

    @property
    def evaluation_mode(self) -> str:
        """Numerical evaluation mode of the model."""
        # pylint: disable=no-member
        return self._evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, new_mode: str):
        """Sets evaluation mode of model."""

        # default error message to be used by all subclasses.
        raise NotImplementedError(
            "Evaluation mode '"
            + str(new_mode)
            + "' is not supported. "
            + "Call help("
            + str(self.__class__.__name__)
            + ".evaluation_mode) "
            + "for available options."
        )

    @abstractmethod
    def get_operators(
        self, in_frame_basis: Optional[bool] = False
    ) -> Union[Array, Tuple[Array], Callable]:
        """Get the operators used in the model construction.

        Args:
            in_frame_basis: Flag indicating whether to return the operators
            in the basis in which the rotating frame operator is diagonal.
        Returns:
            The operators in the basis specified by `in_frame_basis`.
        """
        pass

    def get_drift(self, in_frame_basis: Optional[bool] = False) -> Array:
        """Get the drift term.

        Args:
            in_frame_basis: Flag for whether the returned drift should be
            in the basis in which the frame is diagonal.
        Returns:
            The drift term.
        """
        if not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(self._drift)
        else:
            return self._drift

    def set_drift(
        self,
        new_drift: Array,
        operator_in_frame_basis: Optional[bool] = False,
    ):
        """Sets drift term. Note that if the model has a rotating frame this will override
        any contributions to the drift due to the frame transformation.

        Args:
            new_drift: The drift operator.
            operator_in_frame_basis: Whether `new_drift` is already in the rotating
            frame basis.
        """
        if new_drift is None:
            new_drift = np.zeros((self.dim, self.dim))

        new_drift = to_array(new_drift)

        if not operator_in_frame_basis and self.rotating_frame is not None:
            new_drift = self.rotating_frame.operator_into_frame_basis(new_drift)
        # pylint: disable = attribute-defined-outside-init
        self._drift = new_drift
        # pylint: disable=no-member
        if self._operator_collection is not None:
            # pylint: disable=no-member
            self._operator_collection.drift = new_drift

    @abstractmethod
    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        r"""If possible, evaluate the map :math:`\Lambda(t, \cdot)`.

        Args:
            time: Time.
            in_frame_basis: Whether to return the result in the rotating frame basis.
        """
        pass

    @abstractmethod
    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        r"""Evaluate the right hand side :math:`\dot{y}(t) = \Lambda(t, y)`.

        Args:
            time: Time.
            y: State of the differential equation.
            in_frame_basis: Whether `y` is in the rotating frame basis and the results should be
                returned in the rotatign frame basis.
        """
        pass

    def copy(self):
        """Return a copy of self."""
        return copy(self)

    def __call__(
        self, time: float, y: Optional[Array] = None, in_frame_basis: Optional[bool] = False
    ) -> Array:
        r"""Evaluate generator RHS functions. If ``y is None``,
        attemps to evaluate :math:`\Lambda(t, \cdot)`, otherwise, calculates
        :math:`\Lambda(t, y)`

        Args:
            time: Time.
            y: Optional state.
            in_frame_basis: Whether or not to evaluate in the rotating frame basis.

        Returns:
            Array: Either the evaluated model or the RHS for the given y
        """

        if y is None:
            return self.evaluate(time, in_frame_basis=in_frame_basis)

        return self.evaluate_rhs(time, y, in_frame_basis=in_frame_basis)


class CallableGenerator(BaseGeneratorModel):
    r"""Specifies a linear matrix differential equation of the form
    :math:`\dot{y}=\Lambda(t, y)=G(t)y` with :math:`G(t)` passed as a callable function.
    """

    def __init__(
        self,
        generator: Callable,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        drift: Optional[Union[Operator, Array]] = None,
        dim: Optional[int] = None,
    ):
        self._generator = generator
        self.rotating_frame = rotating_frame
        self._drift = drift
        self._evaluation_mode = "callable_generator"
        self._operator_collection = None
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        else:
            raise ValueError(
                "Dimension of CallableGenerator object should be specified at initialization."
            )

    def get_drift(self, in_frame_basis: Optional[bool] = False) -> Array:
        if in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_into_frame_basis(self._drift)
        else:
            return self._drift

    def get_operators(self, in_frame_basis: Optional[bool] = False) -> Callable:
        """`CallableGenerator` does not have decomposition in terms of operators."""
        raise QiskitError("CallableGenerator does not have decomposition in terms of operators.")

    def set_drift(
        self,
        new_drift: Array,
        operator_in_frame_basis: Optional[bool] = False,
    ):
        # subtracting the frame operator from the generator is handled at evaluation time.
        if operator_in_frame_basis and self.rotating_frame is not None:
            self._drift = self.rotating_frame.operator_out_of_frame_basis(new_drift)
        else:
            self._drift = new_drift

        if self.rotating_frame.frame_diag is not None:
            self._drift = self._drift + self.rotating_frame.operator_out_of_frame_basis(
                self.rotating_frame.frame_diag
            )

    @property
    def evaluation_mode(self) -> str:
        """CallableGenerator does not support evaluation_mode."""
        pass

    @evaluation_mode.setter
    def evaluation_mode(self, new_mode: str):
        """Setting the evaluation mode for CallableGenerator
        is not supported."""
        raise NotImplementedError(
            "Setting implementation mode for CallableGenerator is not supported."
        )

    @property
    def rotating_frame(self) -> RotatingFrame:
        """Return the frame."""
        return self._rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]):
        """Set the frame; either an already instantiated :class:`RotatingFrame` object
        a valid argument for the constructor of :class:`RotatingFrame`, or `None`.
        """
        self._rotating_frame = RotatingFrame(rotating_frame)

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        return self.evaluate(time, in_frame_basis=in_frame_basis) @ y

    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluate the model in array format.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            Array: The evaluated model.
        """

        # evaluate generator and map it into the rotating frame
        gen = self._generator(time)
        if self._drift is not None:
            gen = gen + self._drift
        return self.rotating_frame.generator_into_frame(
            time, gen, operator_in_frame_basis=False, return_in_frame_basis=in_frame_basis
        )


class GeneratorModel(BaseGeneratorModel):
    r"""A model for a a linear matrix differential equation in standard form.

    :class:`GeneratorModel` is a concrete instance of :class:`BaseGeneratorModel`, where the
    map :math:`\Lambda(t, y)` is explicitly constructed as:

    .. math::
        \Lambda(t, y) = G(t)y,

        G(t) = \sum_i s_i(t) G_i + G_d

    where the :math:`G_i` are matrices (represented by :class:`Operator`
    or :class:`Array` objects), the :math:`s_i(t)` are signals represented by
    a list of :class:`Signal` objects, and
    :math:`G_d` is the constant-in-time drift term of the generator.
    """

    def __init__(
        self,
        operators: Array,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        drift: Optional[Array] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        evaluation_mode: str = "dense",
    ):
        """Initialize.

        Args:
            operators: A list of operators :math:`G_i`.
            signals: Stores the terms :math:`s_i(t)`. While required for evaluation,
                     :class:`GeneratorModel` signals are not required at instantiation.
            rotating_frame: Rotating frame operator.
            evaluation_mode: Evaluation mode to use. See ``GeneratorModel.evaluation_mode``
                             for more details. Supported options are:

                                - 'dense' (DenseOperatorCollection)
                                - 'sparse' (SparseOperatorCollection)

        """

        # initialize internal operator representation
        self._operator_collection = None
        self._operators = to_array(operators)
        self._drift = None
        self._evaluation_mode = None
        self.set_drift(drift, operator_in_frame_basis=True)

        # set frame and transform operators into frame basis.
        self._rotating_frame = None
        self.rotating_frame = RotatingFrame(rotating_frame)

        # initialize signal-related attributes
        self._signals = None
        self.signals = signals

        self.evaluation_mode = evaluation_mode

    def get_operators(self, in_frame_basis: Optional[bool] = False) -> Array:
        if not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(self._operators)
        else:
            return self._operators

    @property
    def dim(self) -> int:
        return self._operators.shape[-1]

    @property
    def evaluation_mode(self) -> str:
        """Numerical evaluation mode of the model.

        Available options:

            - 'dense': Stores/evaluates operators using dense Arrays.
            - 'sparse': stores/evaluates operators using scipy
            :class:`csr_matrix` types. Not compatible with JAX.
        """
        return super().evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, new_mode: str):
        """Set evaluation mode.

        Args:
            new_mode: String specifying new mode. Available options
                      are 'dense' and 'sparse'. See property doc string for details.

        Raises:
            NotImplementedError: if new_mode is not one of the above
            supported evaluation modes.
        """

        if new_mode == "dense":
            self._operators = to_array(self._operators)
            self._operator_collection = DenseOperatorCollection(
                self.get_operators(in_frame_basis=True), drift=self.get_drift(in_frame_basis=True)
            )
        elif new_mode == "sparse":
            self._operators = to_csr(self._operators)
            self._operator_collection = SparseOperatorCollection(
                self.get_operators(in_frame_basis=True), drift=self.get_drift(in_frame_basis=True)
            )
        else:
            # raise error with standard message
            # this is equivalent to calling the setter of the base class
            super(__class__, self.__class__).evaluation_mode.fset(self, new_mode)

        self._evaluation_mode = new_mode

    @property
    def signals(self) -> SignalList:
        """Return the signals in the model."""
        return self._signals

    @signals.setter
    def signals(self, signals: Union[SignalList, List[Signal]]):
        """Set the signals."""

        if signals is None:
            self._signals = None
        else:
            # if signals is a list, instantiate a SignalList
            if isinstance(signals, list):
                signals = SignalList(signals)

            # if it isn't a SignalList by now, raise an error
            if not isinstance(signals, SignalList):
                raise QiskitError("Signals specified in unaccepted format.")

            # verify signal length is same as operators
            if len(signals) != self.get_operators().shape[0]:
                raise QiskitError(
                    """Signals needs to have the same length as
                                    operators."""
                )

            self._signals = signals

    @property
    def rotating_frame(self) -> RotatingFrame:
        """Return the rotating frame."""
        return self._rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]):
        if self._rotating_frame is not None and self._rotating_frame.frame_diag is not None:
            self._drift = self._drift + Array(np.diag(self._rotating_frame.frame_diag))
            self._operators = self.rotating_frame.operator_out_of_frame_basis(self._operators)
            self._drift = self.rotating_frame.operator_out_of_frame_basis(self._drift)

        self._rotating_frame = RotatingFrame(rotating_frame)

        if self._rotating_frame.frame_diag is not None:
            self._operators = self.rotating_frame.operator_into_frame_basis(self._operators)
            self._drift = self.rotating_frame.operator_into_frame_basis(self._drift)
            self._drift = self._drift - Array(np.diag(self._rotating_frame.frame_diag))

        # Reset internal operator collection
        if self.evaluation_mode is not None:
            self.evaluation_mode = self.evaluation_mode

    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.

        Args:
            time: Time to evaluate the model.
            in_frame_basis: Whether to evaluate in the basis in which the rotating frame
            operator is diagonal.

        Returns:
            Array: the evaluated model as a (n,n) matrix

        Raises:
            QiskitError: If model cannot be evaluated.

        """

        if self._signals is None:
            raise QiskitError("GeneratorModel cannot be evaluated without signals.")

        # pylint: disable=not-callable
        sig_vals = self._signals(time)

        # Evaluated in frame basis, but without rotations
        op_combo = self._operator_collection(sig_vals)

        # Apply rotations e^{-Ft}Ae^{Ft} in frame basis where F = D
        return self.rotating_frame.operator_into_frame(
            time, op_combo, operator_in_frame_basis=True, return_in_frame_basis=in_frame_basis
        )

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        r"""Evaluate `G(t) @ y`.

        Args:
            time: Time to evaluate the model.
            y: Array specifying system state, in basis specified by
            in_frame_basis.
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal.

        Returns:
            Array defined by :math:`G(t) \times y`.

        Raises:
            QiskitError: If model cannot be evaluated.

        """

        if self._signals is None:
            raise QiskitError("""GeneratorModel cannot be evaluated without signals.""")

        sig_vals = self._signals.__call__(time)

        # Evaluated in frame basis, but without rotations e^{\pm Ft}
        op_combo = self._operator_collection(sig_vals)

        if self.rotating_frame is not None:
            # First, compute e^{tF}y as a pre-rotation in the frame basis
            out = self.rotating_frame.state_out_of_frame(
                time, y, y_in_frame_basis=in_frame_basis, return_in_frame_basis=True
            )
            # Then, compute the product Ae^{tF}y
            out = op_combo @ out
            # Finally, we have the full operator e^{-tF}Ae^{tF}y
            out = self.rotating_frame.state_into_frame(
                time, out, y_in_frame_basis=True, return_in_frame_basis=in_frame_basis
            )
        else:
            return op_combo @ y

        return out

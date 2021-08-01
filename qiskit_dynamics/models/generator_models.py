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
from typing import Callable, Union, List, Optional
from copy import deepcopy
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.operator_collections import DenseOperatorCollection
from qiskit_dynamics import dispatch
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from .frame import BaseFrame, Frame


class BaseGeneratorModel(ABC):
    r"""BaseGeneratorModel is an abstract interface for a time-dependent operator
    :math:`G(t)`, with functionality of relevance for differential
    equations of the form :math:`\dot{y}(t) = G(t)y(t)`.

    The core functionality is evaluation of :math:`G(t)` and the products
    :math:`AG(t)` and :math:`G(t)A`, for operators :math:`A` of suitable
    shape.

    Additionally, this abstract class requires implementation of 3 properties
    to facilitate the use of this object in solving differential equations:
        - A "drift", which is meant to return the "time-independent" part of
          :math:`G(t)`
        - A "frame", here specified as a :class:`BaseFrame` object, which
          represents an anti-Hermitian operator :math:`F`, specifying
          the transformation :math:`G(t) \mapsto G'(t) = e^{-tF}G(t)e^{tF} - F`.

          If a frame is set, the evaluation functions are modified to work
          with G'(t). Furthermore, all evaluation functions have the option
          to return the results in a basis in which :math:`F` is diagonalized,
          to save on the cost of computing :math:`e^{\pm tF}`.
    """

    @property
    @abstractmethod
    def hilbert_space_dimension(self) -> int:
        """Gets Hilbert space dimension."""
        pass

    @property
    @abstractmethod
    def operators(self) -> Union[Array,list[Array]]:
        """Get the originally passed operators by the user"""
        pass

    @property
    def drift(self) -> Array:
        """Gets the originally passed drift term"""
        return self._drift

    @drift.setter
    def drift(self,new_drift: Array):
        """Sets drift term."""
        if new_drift is None:
            new_drift = np.zeros((self.hilbert_space_dimension,self.hilbert_space_dimension))

        new_drift = Array(np.array(new_drift))
        self._drift = new_drift
        # pylint: disable=no-member
        if self._operator_collection is not None:
            # pylint: disable=no-member
            self._operator_collection.drift = new_drift

    @property
    @abstractmethod
    def evaluation_mode(self) -> str:
        """Returns the current implementation mode,
        e.g. sparse/dense, vectorized/not"""
        return self._evaluation_mode

    @evaluation_mode.setter
    @abstractmethod
    def evaluation_mode(self,new_mode: str):
        """Sets evaluation mode of model. 
        Will replace _operator_collection with the
        correct type of operator collection"""
        self._evaluation_mode = new_mode
        pass

    @property
    @abstractmethod
    def frame(self) -> BaseFrame:
        """Get the frame."""
        pass

    @frame.setter
    @abstractmethod
    def frame(self, frame: BaseFrame):
        """Set the frame; either an already instantiated :class:`Frame` object
        a valid argument for the constructor of :class:`Frame`, or `None`.
        Takes care of putting all operators into the basis in which the frame
        matrix F is diagonal.
        """
        pass

    @abstractmethod
    def evaluate_rhs(
        self, time: float, y: Array, in_frame_basis: Optional[bool] = False
    ) -> Array:
        r"""Given some representation y of the system's state,
        evaluate the RHS of the model y'(t) = \Lambda(y,t)
        at the time t.
        Args:
            time: Time
            y: State in the same basis as the model is
            being evaluated.
            in_frame_basis: boolean flag; True if the
                result should be in the frame basis
                or in the lab basis."""
        pass

    @abstractmethod
    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False):
        """If possible, expresses the model at time t
        without reference to the state of the system.
        Args:
            time: Time
            in_frame_basis: boolean flag; True if the
                result should be in the frame basis
                or in the lab basis."""
        pass

    def copy(self):
        """Return a copy of self."""
        return deepcopy(self)

    def __call__(
        self, time: float, y: Optional[Array] = None, in_frame_basis: Optional[bool] = False
    ):
        """Evaluate generator RHS functions. If ``y is None``,
        evaluates the model, and otherwise evaluates ``G(t) @ y``.

        Args:
            time: Time.
            y: Optional state.
            in_frame_basis: Whether or not to evaluate in the frame basis.

        Returns:
            Array: Either the evaluated model or the RHS for the given y
        """

        if y is None:
            return self.evaluate_generator(time, in_frame_basis=in_frame_basis)

        return self.evaluate_rhs(time, y, in_frame_basis=in_frame_basis)


class CallableGenerator(BaseGeneratorModel):
    """Generator specified as a callable"""

    def __init__(
        self,
        generator: Callable,
        frame: Optional[Union[Operator, Array, BaseFrame]] = None,
        drift: Optional[Union[Operator, Array]] = None,
    ):

        self._generator = dispatch.wrap(generator)
        self.frame = frame
        self._drift = drift
        self._evaluation_mode = "callable_generator"
        self._operator_collection = None

    @property
    def hilbert_space_dimension(self) -> int:
        return self._generator(0).shape[-1]

    @property
    def operators(self) -> Callable:
        return self._generator

    @property
    def evaluation_mode(self) -> str:
        return self._evaluation_mode
    
    @evaluation_mode.setter
    def evaluation_mode(self,new_mode: str):
        raise NotImplementedError("Setting implementation mode for CallableGenerator is not supported.")

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, Array, Frame]):
        """Set the frame; either an already instantiated :class:`Frame` object
        a valid argument for the constructor of :class:`Frame`, or `None`.
        """
        self._frame = Frame(frame)

    def evaluate_rhs(
        self, time: float, y: Array, in_frame_basis: Optional[bool] = False
    ) -> Array:
        return self.evaluate_generator(time, in_frame_basis=in_frame_basis) @ y

    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluate the model in array format.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            Array: the evaluated model

        Raises:
            QiskitError: If model cannot be evaluated.
        """

        # evaluate generator and map it into the frame
        gen = self._generator(time)
        return self.frame.generator_into_frame(
            time, gen, operator_in_frame=False, return_in_frame_basis=in_frame_basis
        )


class GeneratorModel(BaseGeneratorModel):
    r"""GeneratorModel is a concrete instance of BaseGeneratorModel, where the
    operator :math:`G(t)` is explicitly constructed as:

    .. math::

        G(t) = \sum_{i=0}^{k-1} s_i(t) G_i,

    where the :math:`G_i` are matrices (represented by :class:`Operator`
    objects), and the :math:`s_i(t)` given by signals represented by a
    :class:`SignalList` object, or a list of :class:`Signal` objects.

    The signals in the model can be specified at instantiation, or afterwards
    by setting the ``signals`` attribute, by giving a
    list of :class:`Signal` objects or a :class:`SignalList`.

    For specifying a frame, this object works with the concrete
    :class:`Frame`, a subclass of :class:`BaseFrame`.

    To do:
        insert mathematical description of frame/cutoff_freq handling
    """

    def __init__(
        self,
        operators: Array,
        drift: Optional[Array] = None,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        frame: Optional[Union[Operator, Array, BaseFrame]] = None,
        evaluation_mode: str = "dense_operator_collection"
    ):
        """Initialize.

        Args:
            operators: A rank-3 Array of operator components. If
                a frame object is provided, each operator is assumed
                to be in the basis in which the frame operator is
                diagonal.
            drift: Optional, constant terms to add to G. Useful for
                frame transformations. If a frame, but not a drift,
                is provided, will be set to -F. If both are provided,
                the drift will be set to drift - F.
            signals: Specifiable as either a SignalList, a list of
                Signal objects, or as the inputs to signal_mapping.
                GeneratorModel can be instantiated without specifying
                signals, but it can not perform any actions without them.
            frame: Rotating frame operator. If specified with a 1d
                array, it is interpreted as the diagonal of a
                diagonal matrix. If provided, it is assumed that all
                operators are in frame basis.
        """

        # initialize internal operator representation
        self._operator_collection = None
        self._operators = Array(np.array(operators))
        self._drift = None
        self.drift = drift
        self.evaluation_mode = evaluation_mode

        # set frame and transform operators into frame basis.
        self._frame = None
        self.frame = Frame(frame)

        # initialize signal-related attributes
        self._signals = None
        self.signals = signals

    @property
    def operators(self) -> Array:
        return self._operators

    @property
    def hilbert_space_dimension(self) -> int:
        return self._operators.shape[-1]
    
    @property
    def evaluation_mode(self) -> str:
        return super().evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self,new_mode: str):
        if new_mode == "dense_operator_collection":
            self._operator_collection = DenseOperatorCollection(self._operators,drift=self.drift)
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
            if len(signals) != self._operator_collection.num_operators:
                raise QiskitError(
                    """Signals needs to have the same length as
                                    operators."""
                )

            self._signals = signals

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, Array, Frame]):
        if self._frame is not None and self._frame.frame_diag is not None:
            self.drift = self.drift + Array(np.diag(self._frame.frame_diag))
            self._operators = self.frame.operator_out_of_frame_basis(self._operators)
            self.drift = self.frame.operator_out_of_frame_basis(self.drift)

        self._frame = Frame(frame)

        if self._frame.frame_diag is not None:
            self._operators = self.frame.operator_into_frame_basis(self._operators)
            self.drift = self.frame.operator_into_frame_basis(self.drift)
            self.drift = self.drift - Array(np.diag(self._frame.frame_diag))

        # Reset internal operation collection
        self.evaluation_mode = self.evaluation_mode

    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.
        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal
        Returns:
            Array: the evaluated model as a (n,n) matrix
        Raises:
            QiskitError: If model cannot be evaluated."""

        if self._signals is None:
            raise QiskitError("""GeneratorModel cannot be evaluated without signals.""")

        sig_vals = self._signals.__call__(time)

        # Evaluated in frame basis, but without rotations
        op_combo = self._operator_collection(sig_vals)

        # Apply rotations e^{-Ft}Ae^{Ft} in frame basis where F = D
        return self.frame.operator_into_frame(time,op_combo,operator_in_frame_basis=True,return_in_frame_basis=in_frame_basis)

    def evaluate_rhs(
        self, time: Union[float, int], y: Array, in_frame_basis: Optional[bool] = False
    ) -> Array:
        """Evaluate the model in array format as a vector, given the current state.
        Args:
            time: Time to evaluate the model
            y: (n) Array specifying system state, in basis choice specified by
                in_frame_basis. If not in frame basis, assumed to not include
                the rotating term e^{-Ft}. If in the frame basis, assumed to
                include the rotating term e^{-Ft}.
            in_frame_basis: Whether to evaluate in the basis in which the frame
                operator is diagonal
        Returns:
            Array: the evaluated model as (n) vector
        Raises:
            QiskitError: If model cannot be evaluated.
        """

        if self._signals is None:
            raise QiskitError("""GeneratorModel cannot be evaluated without signals.""")

        sig_vals = self._signals.__call__(time)

        # Evaluated in frame basis, but without rotations e^{\pm Ft}
        op_combo = self._operator_collection(sig_vals)

        if self.frame is not None:
            #First, compute e^{tF}y as a pre-rotation in the frame basis
            out = self.frame.state_out_of_frame(time,y,y_in_frame_basis = in_frame_basis,return_in_frame_basis=True)
            #Then, compute the product Ae^{tF}y
            out = np.dot(op_combo,out)
            #Finally, we have the full operator e^{-tF}Ae^{tF}y
            out = self.frame.state_into_frame(time,out,y_in_frame_basis=True,return_in_frame_basis=in_frame_basis)
        else:
            return np.dot(op_combo, y)

        return out

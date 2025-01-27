"""
Module for updating the QuantumCircuit class to include calibration information.
"""

import typing

from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    Clbit,
    IfElseOp,
    WhileLoopOp,
    SwitchCaseOp,
    _classical_resource_map,
)
from qiskit.circuit.parameter import Parameter, ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import (
    _ParameterBindsDict,
    _ParameterBindsSequence,
    _OuterCircuitScopeInterface,
)
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.quantumcircuit import QubitSpecifier, ClbitSpecifier
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.circuit.store import Store
from qiskit.circuit.classical import expr
import copy as _copy


import collections
from collections import defaultdict
from typing import Union, Mapping, Iterable, Optional, Sequence, Literal
import numpy as np


def op_start_times(self) -> list[int]:
    """Return a list of operation start times.

    This attribute is enabled once one of scheduling analysis passes
    runs on the quantum circuit.

    Returns:
        List of integers representing instruction start times.
        The index corresponds to the index of instruction in :attr:`QuantumCircuit.data`.

    Raises:
        AttributeError: When circuit is not scheduled.
    """
    if self._op_start_times is None:
        raise AttributeError(
            "This circuit is not scheduled. "
            "To schedule it run the circuit through one of the transpiler scheduling passes."
        )
    return self._op_start_times


def calibrations_getter(self) -> dict:
    """Return calibration dictionary.

    The custom pulse definition of a given gate is of the form
    ``{'gate_name': {(qubits, params): schedule}}``
    """
    return dict(self._calibrations)


def calibrations_setter(self, calibrations: dict):
    """Set the circuit calibration data from a dictionary of calibration definition.

    Args:
        calibrations (dict): A dictionary of input in the format
           ``{'gate_name': {(qubits, gate_params): schedule}}``
    """
    self._calibrations = defaultdict(dict, calibrations)


def has_calibration_for(self, instruction: Union[CircuitInstruction, tuple]):
    """Return True if the circuit has a calibration defined for the instruction context. In this
    case, the operation does not need to be translated to the device basis.
    """
    if isinstance(instruction, CircuitInstruction):
        operation = instruction.operation
        qubits = instruction.qubits
    else:
        operation, qubits, _ = instruction
    if not self.calibrations or operation.name not in self.calibrations:
        return False
    qubits = tuple(self.qubits.index(qubit) for qubit in qubits)
    params = []
    for p in operation.params:
        if isinstance(p, ParameterExpression) and not p.parameters:
            params.append(float(p))
        else:
            params.append(p)
    params = tuple(params)
    return (qubits, params) in self.calibrations[operation.name]


def assign_parameters(  # pylint: disable=missing-raises-doc
    self,
    parameters: Union[Mapping[Parameter, ParameterValueType], Iterable[ParameterValueType]],
    inplace: bool = False,
    *,
    flat_input: bool = False,
    strict: bool = True,
) -> Optional["QuantumCircuit"]:
    """Assign parameters to new parameters or values.

    If ``parameters`` is passed as a dictionary, the keys should be :class:`.Parameter`
    instances in the current circuit. The values of the dictionary can either be numeric values
    or new parameter objects.

    If ``parameters`` is passed as a list or array, the elements are assigned to the
    current parameters in the order of :attr:`parameters` which is sorted
    alphabetically (while respecting the ordering in :class:`.ParameterVector` objects).

    The values can be assigned to the current circuit object or to a copy of it.

    .. note::
        When ``parameters`` is given as a mapping, it is permissible to have keys that are
        strings of the parameter names; these will be looked up using :meth:`get_parameter`.
        You can also have keys that are :class:`.ParameterVector` instances, and in this case,
        the dictionary value should be a sequence of values of the same length as the vector.

        If you use either of these cases, you must leave the setting ``flat_input=False``;
        changing this to ``True`` enables the fast path, where all keys must be
        :class:`.Parameter` instances.

    Args:
        parameters: Either a dictionary or iterable specifying the new parameter values.
        inplace: If False, a copy of the circuit with the bound parameters is returned.
            If True the circuit instance itself is modified.
        flat_input: If ``True`` and ``parameters`` is a mapping type, it is assumed to be
            exactly a mapping of ``{parameter: value}``.  By default (``False``), the mapping
            may also contain :class:`.ParameterVector` keys that point to a corresponding
            sequence of values, and these will be unrolled during the mapping, or string keys,
            which will be converted to :class:`.Parameter` instances using
            :meth:`get_parameter`.
        strict: If ``False``, any parameters given in the mapping that are not used in the
            circuit will be ignored.  If ``True`` (the default), an error will be raised
            indicating a logic error.

    Raises:
        CircuitError: If parameters is a dict and contains parameters not present in the
            circuit.
        ValueError: If parameters is a list/array and the length mismatches the number of free
            parameters in the circuit.

    Returns:
        A copy of the circuit with bound parameters if ``inplace`` is False, otherwise None.

    Examples:

        Create a parameterized circuit and assign the parameters in-place.

        .. plot::
           :include-source:

           from qiskit.circuit import QuantumCircuit, Parameter

           circuit = QuantumCircuit(2)
           params = [Parameter('A'), Parameter('B'), Parameter('C')]
           circuit.ry(params[0], 0)
           circuit.crx(params[1], 0, 1)
           circuit.draw('mpl')
           circuit.assign_parameters({params[0]: params[2]}, inplace=True)
           circuit.draw('mpl')

        Bind the values out-of-place by list and get a copy of the original circuit.

        .. plot::
           :include-source:

           from qiskit.circuit import QuantumCircuit, ParameterVector

           circuit = QuantumCircuit(2)
           params = ParameterVector('P', 2)
           circuit.ry(params[0], 0)
           circuit.crx(params[1], 0, 1)

           bound_circuit = circuit.assign_parameters([1, 2])
           bound_circuit.draw('mpl')

           circuit.draw('mpl')

    """
    if inplace:
        target = self
    else:
        if not isinstance(parameters, dict):
            # We're going to need to access the sorted order wihin the inner Rust method on
            # `target`, so warm up our own cache first so that subsequent calls to
            # `assign_parameters` on `self` benefit as well.
            _ = self._data.parameters
        target = self.copy()
        target._increment_instances()
        target._name_update()

    if isinstance(parameters, collections.abc.Mapping):
        raw_mapping = parameters if flat_input else self._unroll_param_dict(parameters)
        our_parameters = self._data.unsorted_parameters()
        if strict and (extras := raw_mapping.keys() - our_parameters):
            raise CircuitError(
                f"Cannot bind parameters ({', '.join(str(x) for x in extras)}) not present in"
                " the circuit."
            )
        parameter_binds = _ParameterBindsDict(raw_mapping, our_parameters)
        target._data.assign_parameters_mapping(parameter_binds)
    else:
        parameter_binds = _ParameterBindsSequence(target._data.parameters, parameters)
        target._data.assign_parameters_iterable(parameters)

    # Finally, assign the parameters inside any of the calibrations.  We don't track these in
    # the `ParameterTable`, so we manually reconstruct things.
    def map_calibration(qubits, parameters, schedule):
        modified = False
        new_parameters = list(parameters)
        for i, parameter in enumerate(new_parameters):
            if not isinstance(parameter, ParameterExpression):
                continue
            if not (contained := parameter.parameters & parameter_binds.mapping.keys()):
                continue
            for to_bind in contained:
                parameter = parameter.assign(to_bind, parameter_binds.mapping[to_bind])
            if not parameter.parameters:
                parameter = parameter.numeric()
                if isinstance(parameter, complex):
                    raise TypeError(f"Calibration cannot use complex number: '{parameter}'")
            new_parameters[i] = parameter
            modified = True
        if modified:
            schedule.assign_parameters(parameter_binds.mapping)
        return (qubits, tuple(new_parameters)), schedule

    target._calibrations = defaultdict(
        dict,
        (
            (
                gate,
                dict(
                    map_calibration(qubits, parameters, schedule)
                    for (qubits, parameters), schedule in calibrations.items()
                ),
            )
            for gate, calibrations in target._calibrations.items()
        ),
    )
    return None if inplace else target


def add_calibration(
    self,
    gate: Union[Gate, str],
    qubits: Sequence[int],
    # Schedule has the type `qiskit.pulse.Schedule`, but `qiskit.pulse` cannot be imported
    # while this module is, and so Sphinx will not accept a forward reference to it.  Sphinx
    # needs the types available at runtime, whereas mypy will accept it, because it handles the
    # type checking by static analysis.
    schedule,
    params: Optional[Sequence[ParameterValueType]] = None,
) -> None:
    """Register a low-level, custom pulse definition for the given gate.

    Args:
        gate (Union[Gate, str]): Gate information.
        qubits (Union[int, Tuple[int]]): List of qubits to be measured.
        schedule (Schedule): Schedule information.
        params (Optional[List[Union[float, Parameter]]]): A list of parameters.

    Raises:
        Exception: if the gate is of type string and params is None.
    """

    def _format(operand):
        try:
            # Using float/complex value as a dict key is not good idea.
            # This makes the mapping quite sensitive to the rounding error.
            # However, the mechanism is already tied to the execution model (i.e. pulse gate)
            # and we cannot easily update this rule.
            # The same logic exists in DAGCircuit.add_calibration.
            evaluated = complex(operand)
            if np.isreal(evaluated):
                evaluated = float(evaluated.real)
                if evaluated.is_integer():
                    evaluated = int(evaluated)
            return evaluated
        except TypeError:
            # Unassigned parameter
            return operand

    if isinstance(gate, Gate):
        params = gate.params
        gate = gate.name
    if params is not None:
        params = tuple(map(_format, params))
    else:
        params = ()

    self._calibrations[gate][(tuple(qubits), params)] = schedule


def compose(
    self,
    other: Union["QuantumCircuit", Instruction],
    qubits: Optional[Union[QubitSpecifier, Sequence[QubitSpecifier]]] = None,
    clbits: Optional[Union[ClbitSpecifier, Sequence[ClbitSpecifier]]] = None,
    front: bool = False,
    inplace: bool = False,
    wrap: bool = False,
    *,
    copy: bool = True,
    var_remap: Optional[Mapping[Union[str, expr.Var], Union[str, expr.Var]]] = None,
    inline_captures: bool = False,
) -> Optional["QuantumCircuit"]:
    """Apply the instructions from one circuit onto specified qubits and/or clbits on another.

    .. note::

        By default, this creates a new circuit object, leaving ``self`` untouched.  For most
        uses of this function, it is far more efficient to set ``inplace=True`` and modify the
        base circuit in-place.

    When dealing with realtime variables (:class:`.expr.Var` instances), there are two principal
    strategies for using :meth:`compose`:

    1. The ``other`` circuit is treated as entirely additive, including its variables.  The
       variables in ``other`` must be entirely distinct from those in ``self`` (use
       ``var_remap`` to help with this), and all variables in ``other`` will be declared anew in
       the output with matching input/capture/local scoping to how they are in ``other``.  This
       is generally what you want if you're joining two unrelated circuits.

    2. The ``other`` circuit was created as an exact extension to ``self`` to be inlined onto
       it, including acting on the existing variables in their states at the end of ``self``.
       In this case, ``other`` should be created with all these variables to be inlined declared
       as "captures", and then you can use ``inline_captures=True`` in this method to link them.
       This is generally what you want if you're building up a circuit by defining layers
       on-the-fly, or rebuilding a circuit using layers taken from itself.  You might find the
       ``vars_mode="captures"`` argument to :meth:`copy_empty_like` useful to create each
       layer's base, in this case.

    Args:
        other (qiskit.circuit.Instruction or QuantumCircuit):
            (sub)circuit or instruction to compose onto self.  If not a :obj:`.QuantumCircuit`,
            this can be anything that :obj:`.append` will accept.
        qubits (list[Qubit|int]): qubits of self to compose onto.
        clbits (list[Clbit|int]): clbits of self to compose onto.
        front (bool): If ``True``, front composition will be performed.  This is not possible within
            control-flow builder context managers.
        inplace (bool): If ``True``, modify the object. Otherwise, return composed circuit.
        copy (bool): If ``True`` (the default), then the input is treated as shared, and any
            contained instructions will be copied, if they might need to be mutated in the
            future.  You can set this to ``False`` if the input should be considered owned by
            the base circuit, in order to avoid unnecessary copies; in this case, it is not
            valid to use ``other`` afterward, and some instructions may have been mutated in
            place.
        var_remap (Mapping): mapping to use to rewrite :class:`.expr.Var` nodes in ``other`` as
            they are inlined into ``self``.  This can be used to avoid naming conflicts.

            Both keys and values can be given as strings or direct :class:`.expr.Var` instances.
            If a key is a string, it matches any :class:`~.expr.Var` with the same name.  If a
            value is a string, whenever a new key matches a it, a new :class:`~.expr.Var` is
            created with the correct type.  If a value is a :class:`~.expr.Var`, its
            :class:`~.expr.Expr.type` must exactly match that of the variable it is replacing.
        inline_captures (bool): if ``True``, then all "captured" :class:`~.expr.Var` nodes in
            the ``other`` :class:`.QuantumCircuit` are assumed to refer to variables already
            declared in ``self`` (as any input/capture/local type), and the uses in ``other``
            will apply to the existing variables.  If you want to build up a layer for an
            existing circuit to use with :meth:`compose`, you might find the
            ``vars_mode="captures"`` argument to :meth:`copy_empty_like` useful.  Any remapping
            in ``vars_remap`` occurs before evaluating this variable inlining.

            If this is ``False`` (the default), then all variables in ``other`` will be required
            to be distinct from those in ``self``, and new declarations will be made for them.
        wrap (bool): If True, wraps the other circuit into a gate (or instruction, depending on
            whether it contains only unitary instructions) before composing it onto self.
            Rather than using this option, it is almost always better to manually control this
            yourself by using :meth:`to_instruction` or :meth:`to_gate`, and then call
            :meth:`append`.

    Returns:
        QuantumCircuit: the composed circuit (returns None if inplace==True).

    Raises:
        CircuitError: if no correct wire mapping can be made between the two circuits, such as
            if ``other`` is wider than ``self``.
        CircuitError: if trying to emit a new circuit while ``self`` has a partially built
            control-flow context active, such as the context-manager forms of :meth:`if_test`,
            :meth:`for_loop` and :meth:`while_loop`.
        CircuitError: if trying to compose to the front of a circuit when a control-flow builder
            block is active; there is no clear meaning to this action.

    Examples:
        .. code-block:: python

            >>> lhs.compose(rhs, qubits=[3, 2], inplace=True)

        .. parsed-literal::

                        ┌───┐                   ┌─────┐                ┌───┐
            lqr_1_0: ───┤ H ├───    rqr_0: ──■──┤ Tdg ├    lqr_1_0: ───┤ H ├───────────────
                        ├───┤              ┌─┴─┐└─────┘                ├───┤
            lqr_1_1: ───┤ X ├───    rqr_1: ┤ X ├───────    lqr_1_1: ───┤ X ├───────────────
                     ┌──┴───┴──┐           └───┘                    ┌──┴───┴──┐┌───┐
            lqr_1_2: ┤ U1(0.1) ├  +                     =  lqr_1_2: ┤ U1(0.1) ├┤ X ├───────
                     └─────────┘                                    └─────────┘└─┬─┘┌─────┐
            lqr_2_0: ─────■─────                           lqr_2_0: ─────■───────■──┤ Tdg ├
                        ┌─┴─┐                                          ┌─┴─┐        └─────┘
            lqr_2_1: ───┤ X ├───                           lqr_2_1: ───┤ X ├───────────────
                        └───┘                                          └───┘
            lcr_0: 0 ═══════════                           lcr_0: 0 ═══════════════════════

            lcr_1: 0 ═══════════                           lcr_1: 0 ═══════════════════════

    """

    if inplace and front and self._control_flow_scopes:
        # If we're composing onto ourselves while in a stateful control-flow builder context,
        # there's no clear meaning to composition to the "front" of the circuit.
        raise CircuitError(
            "Cannot compose to the front of a circuit while a control-flow context is active."
        )
    if not inplace and self._control_flow_scopes:
        # If we're inside a stateful control-flow builder scope, even if we successfully cloned
        # the partial builder scope (not simple), the scope wouldn't be controlled by an active
        # `with` statement, so the output circuit would be permanently broken.
        raise CircuitError(
            "Cannot emit a new composed circuit while a control-flow context is active."
        )

    # Avoid mutating `dest` until as much of the error checking as possible is complete, to
    # avoid an in-place composition getting `self` in a partially mutated state for a simple
    # error that the user might want to correct in an interactive session.
    dest = self if inplace else self.copy()

    var_remap = {} if var_remap is None else var_remap

    # This doesn't use `functools.cache` so we can access it during the variable remapping of
    # instructions.  We cache all replacement lookups for a) speed and b) to ensure that
    # the same variable _always_ maps to the same replacement even if it's used in different
    # places in the recursion tree (such as being a captured variable).
    def replace_var(var: expr.Var, cache: Mapping[expr.Var, expr.Var]) -> expr.Var:
        # This is closing over an argument to `compose`.
        nonlocal var_remap

        if out := cache.get(var):
            return out
        if (replacement := var_remap.get(var)) or (replacement := var_remap.get(var.name)):
            if isinstance(replacement, str):
                replacement = expr.Var.new(replacement, var.type)
            if replacement.type != var.type:
                raise CircuitError(
                    f"mismatched types in replacement for '{var.name}':"
                    f" '{var.type}' cannot become '{replacement.type}'"
                )
        else:
            replacement = var
        cache[var] = replacement
        return replacement

    # As a special case, allow composing some clbits onto no clbits - normally the destination
    # has to be strictly larger. This allows composing final measurements onto unitary circuits.
    if isinstance(other, QuantumCircuit):
        if not self.clbits and other.clbits:
            if dest._control_flow_scopes:
                raise CircuitError("cannot implicitly add clbits while within a control-flow scope")
            dest.add_bits(other.clbits)
            for reg in other.cregs:
                dest.add_register(reg)

    if wrap and isinstance(other, QuantumCircuit):
        other = (
            other.to_gate()
            if all(isinstance(ins.operation, Gate) for ins in other.data)
            else other.to_instruction()
        )

    if not isinstance(other, QuantumCircuit):
        if qubits is None:
            qubits = self.qubits[: other.num_qubits]
        if clbits is None:
            clbits = self.clbits[: other.num_clbits]
        if front:
            # Need to keep a reference to the data for use after we've emptied it.
            old_data = dest._data.copy(copy_instructions=copy)
            dest.clear()
            dest.append(other, qubits, clbits, copy=copy)
            for instruction in old_data:
                dest._append(instruction)
        else:
            dest.append(other, qargs=qubits, cargs=clbits, copy=copy)
        return None if inplace else dest

    if other.num_qubits > dest.num_qubits or other.num_clbits > dest.num_clbits:
        raise CircuitError(
            "Trying to compose with another QuantumCircuit which has more 'in' edges."
        )

    # Maps bits in 'other' to bits in 'dest'.
    mapped_qubits: list[Qubit]
    mapped_clbits: list[Clbit]
    edge_map: dict[Union[Qubit, Clbit], Union[Qubit, Clbit]] = {}
    if qubits is None:
        mapped_qubits = dest.qubits
        edge_map.update(zip(other.qubits, dest.qubits))
    else:
        mapped_qubits = dest._qbit_argument_conversion(qubits)
        if len(mapped_qubits) != other.num_qubits:
            raise CircuitError(
                f"Number of items in qubits parameter ({len(mapped_qubits)}) does not"
                f" match number of qubits in the circuit ({other.num_qubits})."
            )
        if len(set(mapped_qubits)) != len(mapped_qubits):
            raise CircuitError(
                f"Duplicate qubits referenced in 'qubits' parameter: '{mapped_qubits}'"
            )
        edge_map.update(zip(other.qubits, mapped_qubits))

    if clbits is None:
        mapped_clbits = dest.clbits
        edge_map.update(zip(other.clbits, dest.clbits))
    else:
        mapped_clbits = dest._cbit_argument_conversion(clbits)
        if len(mapped_clbits) != other.num_clbits:
            raise CircuitError(
                f"Number of items in clbits parameter ({len(mapped_clbits)}) does not"
                f" match number of clbits in the circuit ({other.num_clbits})."
            )
        if len(set(mapped_clbits)) != len(mapped_clbits):
            raise CircuitError(
                f"Duplicate clbits referenced in 'clbits' parameter: '{mapped_clbits}'"
            )
        edge_map.update(zip(other.clbits, dest._cbit_argument_conversion(clbits)))

    for gate, cals in other.calibrations.items():
        dest._calibrations[gate].update(cals)

    dest.duration = None
    dest.unit = "dt"
    dest.global_phase += other.global_phase

    # This is required to trigger data builds if the `other` is an unbuilt `BlueprintCircuit`,
    # so we can the access the complete `CircuitData` object at `_data`.
    _ = other.data

    def copy_with_remapping(
        source, dest, bit_map, var_map, inline_captures, new_qubits=None, new_clbits=None
    ):
        # Copy the instructions from `source` into `dest`, remapping variables in instructions
        # according to `var_map`.  If `new_qubits` or `new_clbits` are given, the qubits and
        # clbits of the source instruction are remapped to those as well.
        for var in source.iter_input_vars():
            dest.add_input(replace_var(var, var_map))
        if inline_captures:
            for var in source.iter_captured_vars():
                replacement = replace_var(var, var_map)
                if not dest.has_var(replace_var(var, var_map)):
                    if var is replacement:
                        raise CircuitError(
                            f"Variable '{var}' to be inlined is not in the base circuit."
                            " If you wanted it to be automatically added, use"
                            " `inline_captures=False`."
                        )
                    raise CircuitError(
                        f"Replacement '{replacement}' for variable '{var}' is not in the"
                        " base circuit.  Is the replacement correct?"
                    )
        else:
            for var in source.iter_captured_vars():
                dest.add_capture(replace_var(var, var_map))
        for var in source.iter_declared_vars():
            dest.add_uninitialized_var(replace_var(var, var_map))

        def recurse_block(block):
            # Recurse the remapping into a control-flow block.  Note that this doesn't remap the
            # clbits within; the story around nested classical-register-based control-flow
            # doesn't really work in the current data model, and we hope to replace it with
            # `Expr`-based control-flow everywhere.
            new_block = block.copy_empty_like()
            new_block._vars_input = {}
            new_block._vars_capture = {}
            new_block._vars_local = {}
            # For the recursion, we never want to inline captured variables because we're not
            # copying onto a base that has variables.
            copy_with_remapping(block, new_block, bit_map, var_map, inline_captures=False)
            return new_block

        variable_mapper = _classical_resource_map.VariableMapper(
            dest.cregs, bit_map, var_map, add_register=dest.add_register
        )

        def map_vars(op):
            n_op = op
            is_control_flow = isinstance(n_op, ControlFlowOp)
            if not is_control_flow and (condition := getattr(n_op, "condition", None)) is not None:
                n_op = n_op.copy() if n_op is op and copy else n_op
                n_op.condition = variable_mapper.map_condition(condition)
            elif is_control_flow:
                n_op = n_op.replace_blocks(recurse_block(block) for block in n_op.blocks)
                if isinstance(n_op, (IfElseOp, WhileLoopOp)):
                    n_op.condition = variable_mapper.map_condition(n_op.condition)
                elif isinstance(n_op, SwitchCaseOp):
                    n_op.target = variable_mapper.map_target(n_op.target)
            elif isinstance(n_op, Store):
                n_op = Store(
                    variable_mapper.map_expr(n_op.lvalue), variable_mapper.map_expr(n_op.rvalue)
                )
            return n_op.copy() if n_op is op and copy else n_op

        instructions = source._data.copy(copy_instructions=copy)
        instructions.replace_bits(qubits=new_qubits, clbits=new_clbits)
        instructions.map_nonstandard_ops(map_vars)
        dest._current_scope().extend(instructions)

    append_existing = None
    if front:
        append_existing = dest._data.copy(copy_instructions=copy)
        dest.clear()
    copy_with_remapping(
        other,
        dest,
        bit_map=edge_map,
        # The actual `Var: Var` map gets built up from the more freeform user input as we
        # encounter the variables, since the user might be using string keys to refer to more
        # than one variable in separated scopes of control-flow operations.
        var_map={},
        inline_captures=inline_captures,
        new_qubits=mapped_qubits,
        new_clbits=mapped_clbits,
    )
    if append_existing:
        dest._current_scope().extend(append_existing)

    return None if inplace else dest


def copy_empty_like(
    self,
    name: Optional[str] = None,
    *,
    vars_mode: Literal["alike", "captures", "drop"] = "alike",
) -> typing.Self:
    """Return a copy of self with the same structure but empty.

    That structure includes:

    * name, calibrations and other metadata
    * global phase
    * all the qubits and clbits, including the registers
    * the realtime variables defined in the circuit, handled according to the ``vars`` keyword
      argument.

    .. warning::

        If the circuit contains any local variable declarations (those added by the
        ``declarations`` argument to the circuit constructor, or using :meth:`add_var`), they
        may be **uninitialized** in the output circuit.  You will need to manually add store
        instructions for them (see :class:`.Store` and :meth:`.QuantumCircuit.store`) to
        initialize them.

    Args:
        name: Name for the copied circuit. If None, then the name stays the same.
        vars_mode: The mode to handle realtime variables in.

            alike
                The variables in the output circuit will have the same declaration semantics as
                in the original circuit.  For example, ``input`` variables in the source will be
                ``input`` variables in the output circuit.

            captures
                All variables will be converted to captured variables.  This is useful when you
                are building a new layer for an existing circuit that you will want to
                :meth:`compose` onto the base, since :meth:`compose` can inline captures onto
                the base circuit (but not other variables).

            drop
                The output circuit will have no variables defined.

    Returns:
        QuantumCircuit: An empty copy of self.
    """
    if not (name is None or isinstance(name, str)):
        raise TypeError(
            f"invalid name for a circuit: '{name}'. The name must be a string or 'None'."
        )
    cpy = _copy.copy(self)
    # copy registers correctly, in copy.copy they are only copied via reference
    cpy.qregs = self.qregs.copy()
    cpy.cregs = self.cregs.copy()
    cpy._builder_api = _OuterCircuitScopeInterface(cpy)
    cpy._ancillas = self._ancillas.copy()
    cpy._qubit_indices = self._qubit_indices.copy()
    cpy._clbit_indices = self._clbit_indices.copy()

    if vars_mode == "alike":
        # Note that this causes the local variables to be uninitialised, because the stores are
        # not copied.  This can leave the circuit in a potentially dangerous state for users if
        # they don't re-add initializer stores.
        cpy._vars_local = self._vars_local.copy()
        cpy._vars_input = self._vars_input.copy()
        cpy._vars_capture = self._vars_capture.copy()
    elif vars_mode == "captures":
        cpy._vars_local = {}
        cpy._vars_input = {}
        cpy._vars_capture = {var.name: var for var in self.iter_vars()}
    elif vars_mode == "drop":
        cpy._vars_local = {}
        cpy._vars_input = {}
        cpy._vars_capture = {}
    else:  # pragma: no cover
        raise ValueError(f"unknown vars_mode: '{vars_mode}'")

    cpy._data = CircuitData(
        self._data.qubits, self._data.clbits, global_phase=self._data.global_phase
    )

    cpy._calibrations = _copy.deepcopy(self._calibrations)
    cpy._metadata = _copy.deepcopy(self._metadata)

    if name:
        cpy.name = name
    return cpy


if not hasattr(QuantumCircuit, "calibrations"):
    setattr(QuantumCircuit, "_calibrations", defaultdict(dict))
    setattr(QuantumCircuit, "calibrations", property(calibrations_getter, calibrations_setter))
    setattr(QuantumCircuit, "add_calibration", add_calibration)
    setattr(QuantumCircuit, "has_calibration_for", has_calibration_for)
    setattr(QuantumCircuit, "_op_start_times", None)
    setattr(QuantumCircuit, "assign_parameters", assign_parameters)
    setattr(QuantumCircuit, "compose", compose)
    setattr(QuantumCircuit, "copy_empty_like", copy_empty_like)

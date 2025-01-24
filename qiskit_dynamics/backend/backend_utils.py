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
# pylint: disable=invalid-name

"""
Utility functions for Dynamics Backend.
"""


from typing import Optional, Union, List, Dict

import numpy as np
from qiskit import QiskitError
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix


from qiskit_dynamics.arraylias.alias import ArrayLike, _to_dense
from qiskit_dynamics.models import HamiltonianModel, LindbladModel


def _get_dressed_state_decomposition(
    operator: ArrayLike, rtol=1e-8, atol=1e-5
) -> Union[Dict[str, np.ndarray], List[float], Dict[str, float]]:
    """Get the eigenvalues and eigenvectors of a nearly-diagonal hermitian operator, sorted
    according to overlap with the elementary basis.

    This function is essentially a wrapper around ``numpy.linalg.eigh``, but
    sorts the eigenvectors according to the value of ``np.argmax(np.abs(evec))``. It also
    validates that this is unique for each eigenvector.

    Args:
        operator: Hermitian operator.
        subsystem_dims: Dimensions of the subsystems composing the system.
        rtol: Relative tolerance for Hermiticity check.
        atol: Absolute tolerance for Hermiticity check.

    Returns:
        Tuple: a pair of arrays, one containing eigenvalues and one containing corresponding
        eigenvectors.

    Raises:
        QiskitError: If ``np.argmax(np.abs(evec))`` is non-unique across eigenvectors, or if
        operator is not Hermitian.
    """

    if not is_hermitian_matrix(operator, rtol=rtol, atol=atol):
        raise QiskitError("_get_dressed_state_decomposition received non-Hermitian operator.")

    evals, evecs = np.linalg.eigh(np.array(operator))

    dressed_evals = np.zeros_like(evals)
    dressed_states = np.zeros_like(evecs)

    found_positions = []
    for eigval, evec in zip(evals, evecs.transpose()):
        position = np.argmax(np.abs(evec))
        if position in found_positions:
            raise QiskitError(
                """Dressed-state sorting failed due to non-unique np.argmax(np.abs(evec))
                for eigenvectors."""
            )

        found_positions.append(position)

        dressed_states[:, position] = evec
        dressed_evals[position] = eigval

    return dressed_evals, dressed_states


def _get_lab_frame_static_hamiltonian(model: Union[HamiltonianModel, LindbladModel]) -> np.ndarray:
    """Get the static Hamiltonian in the lab frame and standard basis.

    This function assumes that the model was constructed with operators specified in the lab frame
    (regardless of the rotating frame) and in the standard basis.

    Args:
        model: The model.

    Returns:
        np.ndarray
    """
    static_hamiltonian = None
    if isinstance(model, HamiltonianModel):
        static_hamiltonian = _to_dense(model.static_operator)
    else:
        static_hamiltonian = _to_dense(model.static_hamiltonian)

    static_hamiltonian = 1j * model.rotating_frame.generator_out_of_frame(
        t=0.0, operator=-1j * static_hamiltonian
    )

    return np.array(static_hamiltonian)


def _get_memory_slot_probabilities(
    probability_dict: Dict,
    memory_slot_indices: List[int],
    num_memory_slots: Optional[int] = None,
    max_outcome_value: Optional[int] = None,
) -> Dict:
    """Construct probability dictionary for memory slot outcomes from a probability dictionary for
    state level measurement outcomes.

    Args:
        probability_dict: A list of probabilities for the outcomes of state measurement. Keys
            are assumed to all be strings of integers of the same length.
        memory_slot_indices: Indices of which memory slots store the digits of the keys of
            probability_dict.
        num_memory_slots: Total number of memory slots for results. If None,
            defaults to the maximum index in memory_slot_indices. The default value
            of unused memory slots is 0.
        max_outcome_value: Maximum value that can be stored in a memory slot. All outcomes higher
            than this will be rounded down.

    Returns:
        Dict: Keys are memory slot outcomes, values are the probabilities of those outcomes.
    """
    num_memory_slots = num_memory_slots or (max(memory_slot_indices) + 1)
    memory_slot_probs = {}
    for level_str, prob in probability_dict.items():
        memory_slot_result = ["0"] * num_memory_slots

        for idx, level in zip(memory_slot_indices, reversed(level_str)):
            if max_outcome_value and int(level) > max_outcome_value:
                level = str(max_outcome_value)
            memory_slot_result[-(idx + 1)] = level

        memory_slot_result = hex(int("".join(memory_slot_result), 2))
        if memory_slot_result in memory_slot_probs:
            memory_slot_probs[memory_slot_result] += prob
        else:
            memory_slot_probs[memory_slot_result] = prob

    return memory_slot_probs


def _sample_probability_dict(
    probability_dict: Dict,
    shots: int,
    normalize_probabilities: bool = True,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample outcomes based on probability dictionary.

    Args:
        probability_dict: Dictionary representing probability distribution, with keys being
            outcomes, values being probabilities.
        shots: Number of shots.
        normalize_probabilities: Whether or not to normalize the probabilities to sum to 1 before
            sampling.
        seed: Seed to use in rng construction.

    Return:
        List: of entries of probability_dict, sampled according to the probabilities.
    """
    rng = np.random.default_rng(seed=seed)
    alphabet, probs = zip(*probability_dict.items())

    if normalize_probabilities:
        probs = np.array(probs)
        probs = probs / probs.sum()

    return rng.choice(alphabet, size=shots, replace=True, p=probs)


def _get_counts_from_samples(samples: list) -> Dict:
    """Count items in list."""
    return dict(zip(*np.unique(samples, return_counts=True)))


def _get_subsystem_probabilities(probability_tensor: np.ndarray, sub_idx: int) -> np.ndarray:
    """Marginalize a probability array to a single subsystem. Adapted from
    ``qiskit.quantum_info.QuantumState._subsystem_probabilities``.

    Args:
        probability_tensor: K-dimensional probability array, where the probability of outcome
            ``(idx1, ..., idxk)`` is ``probability_tensor[idx1, ..., idxk]``.
        sub_idx: Subsystem index to return marginalized probabilities.
            ``sub_idx`` is indexed in reverse order to be consistent with qiskit.

    Returns:
        The marginalized probability for the specified subsystem.
    """

    # Convert qargs to tensor axes
    ndim = probability_tensor.ndim
    sub_axis = ndim - 1 - sub_idx

    # Get sum axis for marginalized subsystems
    sum_axis = tuple(i for i in range(ndim) if i != sub_axis)
    if sum_axis:
        probability_tensor = probability_tensor.sum(axis=sum_axis)

    return probability_tensor


def _get_iq_data(
    state: Union[Statevector, DensityMatrix],
    measurement_subsystems: List[int],
    iq_centers: List[List[List[float]]],
    iq_width: float,
    shots: int,
    memory_slot_indices: List[int],
    num_memory_slots: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generates IQ data for each physical level.

    Args:
        state: Quantum state. measurement_subsystems: Labels of subsystems in the system being
        measured. memory_slot_indices: Indices of which memory slots store the data of subsystems.
        num_memory_slots: Total number of memory slots for results. If None,
            defaults to the maximum index in memory_slot_indices.
        iq_centers: centers for IQ distribution. provided in the format
            ``iq_centers[subsystem][level] = [I,Q]``.
        iq_width: Standard deviation of IQ distribution around the centers. shots: Number of Shots
        seed: Seed for sample generation.

    Returns:
        (I,Q) data as ndarray[shot index, qubit index] = [I,Q]

    Raises:
        QiskitError: If number of centers and levels don't match.
    """
    rng = np.random.default_rng(seed)
    subsystem_dims = [dim for dim in state.dims() if dim != 1]
    probabilities = state.probabilities()
    probabilities_tensor = probabilities.reshape(list(reversed(subsystem_dims)))

    full_i, full_q = [], []
    for sub_idx in measurement_subsystems:
        # Get probabilities for each subsystem
        sub_probability = _get_subsystem_probabilities(probabilities_tensor, sub_idx=sub_idx)
        # No. of shots for each level
        counts_n = rng.multinomial(shots, sub_probability / sum(sub_probability), size=1).T

        if len(counts_n) != len(iq_centers[sub_idx]):
            raise QiskitError(
                f"""Number of centers {len(iq_centers[sub_idx])} not equal
                to number of levels {len(counts_n)}"""
            )

        sub_i, sub_q = [], []
        for idx, count_i in enumerate(counts_n):
            sub_i.append(rng.normal(loc=iq_centers[sub_idx][idx][0], scale=iq_width, size=count_i))
            sub_q.append(rng.normal(loc=iq_centers[sub_idx][idx][1], scale=iq_width, size=count_i))

        full_i.append(np.concatenate(sub_i))
        full_q.append(np.concatenate(sub_q))
    full_iq = np.array([full_i, full_q]).T

    num_memory_slots = num_memory_slots or (max(memory_slot_indices) + 1)
    mem_slot_iq = np.zeros((shots, num_memory_slots, 2))

    for idx, mem_idx in enumerate(memory_slot_indices):
        mem_slot_iq[:, mem_idx, :] = full_iq[:, idx, :]

    return mem_slot_iq

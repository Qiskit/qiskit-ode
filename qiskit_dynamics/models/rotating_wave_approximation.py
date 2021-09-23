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
# pylint: disable=invalid-name, inconsistent-return-statements

"""Functions for performing the Rotating Wave Approximation
on Model classes."""


from typing import List, Optional, Union
import numpy as np
from qiskit_dynamics.models import (
    BaseGeneratorModel,
    GeneratorModel,
    HamiltonianModel,
    LindbladModel,
    RotatingFrame,
)
from qiskit_dynamics.signals import SignalSum, Signal, SignalList
from qiskit_dynamics.dispatch import Array


def rotating_wave_approximation(
    model: BaseGeneratorModel,
    cutoff_freq: float,
    return_signal_map: Optional[bool] = False,
) -> BaseGeneratorModel:
    r"""Construct a new model by performing the rotating wave approximation with a
    given cutoff frequency.

    Performs elementwise rotating wave approximation (RWA) with
    cutoff frequency ``cutoff_freq`` on each operator in a model, returning a new model.
    The new model contains a modified list of signal coefficients,
    and setting the optional argument ``return_signal_map=True`` results in
    the additional return of the function ``f`` which maps the signals
    of the input model to those of the output RWA model, such that the
    code blocks:

    .. code-block:: python

        model.signals = new_signals
        rwa_model = rotating_wave_approximation(model, cutoff_freq)

    and

    .. code-block:: python

        rwa_model, f = rotating_wave_approximation(model, cutoff_freq, return_signal_map=True)
        rwa_model.signals = f(new_signals)

    result in an ``rwa_model`` with the same updated signals.

    Formalism: When considering :math:`s_i(t) e^{-tF}G_ie^{tF}`, in the basis in which
    :math:`F` is diagonal, the :math:`(j, k)` element of :math:`G_i` has effective frequency
    :math:`\tilde\nu_{ijk}^\pm = \pm\nu_i + Im[-d_j+d_k]/2\pi`, where the :math:`\pm\nu_i`
    comes from expressing
    :math:`s_i(t) = Re[a_i(t)e^{2\pi i\nu_i t}] = a_i(t)e^{i(2\pi\nu_i t+\phi_i)}/2 + c.c.`
    and the other term comes from the rotating frame. Define :math:`G_i^\pm` the matrix whose
    entries :math:`(G_i^\pm)_{jk}` are the entries of :math:`G_i` s.t.
    :math:`|\nu_{ijk}^\pm|<\nu_*` for some
    cutoff frequency :math:`\nu_*`. Then, after the RWA, we may write

    .. math::
        s_i(t)G_i \to G_i^+ a_ie^{i(2\pi \nu_i t+\phi_i)}/2
                        + G_i^- \overline{a_i}e^{-i(2\pi \nu_i t+\phi_i)}/2.

    When we regroup these to use only the real components of Signal objects, we find that

    .. math::
        s_i(t)G_i \to  s_i(t)(G_i^+ + G_i^-)/2 + s_i'(t)(iG_i^+-iG_i^-)

    where :math:`s_i'(t)` is a Signal with the same frequency and amplitude as :math:`s_i`,
    but with a phase shift of :math:`\phi_i - \pi/2`.

    Args:
        model: The GeneratorModel to which you
        wish to apply the RWA.
        cutoff_freq: The maximum (magnitude) of
        frequency you wish to allow.
        return_signal_map: Whether to also return a function f that
            converts pre-RWA SignalLists to post-RWA SignalLists.
    Returns:
        GeneratorModel with twice as many terms, and, if return_signal_map,
        also the function f.
    Raises:
        ValueError: If the model has no signals.
    """

    if model.signals is None:
        raise ValueError("Model must have nontrivial signals to perform the RWA.")

    n = model.dim

    frame_freqs = None
    if model.rotating_frame is None or model.rotating_frame.frame_diag is None:
        frame_freqs = np.zeros((n, n))
    else:
        diag = model.rotating_frame.frame_diag
        diff_matrix = np.broadcast_to(diag, (n, n)) - np.broadcast_to(diag, (n, n)).T
        frame_freqs = diff_matrix.imag / (2 * np.pi)

    if model.rotating_frame.frame_diag is not None:
        frame_shift = np.diag(model.rotating_frame.frame_diag)
        if isinstance(model, (HamiltonianModel, LindbladModel)):
            frame_shift = 1j * frame_shift
    else:
        frame_shift = 0

    if isinstance(model, GeneratorModel):
        cur_drift = model.get_static_operator(True) + frame_shift  # undo frame shifting for RWA
        rwa_drift = cur_drift * (abs(frame_freqs) < cutoff_freq).astype(int)
        rwa_drift = model.rotating_frame.operator_out_of_frame_basis(rwa_drift)

        rwa_operators = get_rwa_operators(
            model.get_operators(True), model.signals, model.rotating_frame, frame_freqs, cutoff_freq
        )
        rwa_signals = get_rwa_signals(model.signals)

        # works for both GeneratorModel and HamiltonianModel
        rwa_model = model.__class__(
            rwa_operators,
            rwa_signals,
            static_operator=rwa_drift,
            rotating_frame=model.rotating_frame,
            evaluation_mode=model.evaluation_mode,
        )
        if return_signal_map:
            return rwa_model, get_rwa_signals
        return rwa_model

    elif isinstance(model, LindbladModel):
        cur_drift = model.get_static_hamiltonian(True) + frame_shift  # undo frame shifting for RWA
        rwa_drift = cur_drift * (abs(frame_freqs) < cutoff_freq).astype(int)
        rwa_drift = model.rotating_frame.operator_out_of_frame_basis(rwa_drift)

        cur_ham_ops, cur_dis_ops = model.get_operators(in_frame_basis=True)
        cur_ham_sig, cur_dis_sig = model.signals

        rwa_ham_ops = get_rwa_operators(
            cur_ham_ops, cur_ham_sig, model.rotating_frame, frame_freqs, cutoff_freq
        )
        rwa_ham_sig = get_rwa_signals(cur_ham_sig)

        if cur_dis_ops is not None and cur_dis_sig is not None:
            rwa_dis_ops = get_rwa_operators(
                cur_dis_ops, cur_dis_sig, model.rotating_frame, frame_freqs, cutoff_freq
            )
            rwa_dis_sig = get_rwa_signals(cur_dis_sig)

        rwa_model = LindbladModel(
            rwa_ham_ops,
            hamiltonian_signals=rwa_ham_sig,
            dissipator_operators=rwa_dis_ops,
            dissipator_signals=rwa_dis_sig,
            static_hamiltonian=rwa_drift,
            rotating_frame=model.rotating_frame,
            evaluation_mode=model.evaluation_mode,
        )

        if return_signal_map:
            signal_translator = lambda a, b: (get_rwa_signals(a), get_rwa_signals(b))
            return rwa_model, signal_translator
        return rwa_model


def get_rwa_operators(
    current_ops: Array,
    current_sigs: SignalList,
    rotating_frame: RotatingFrame,
    frame_freqs: Array,
    cutoff_freq: float,
):
    r"""Given a set of operators as a (k,n,n) Array, a set of
    frequencies (frame_freqs)_{jk} = Im[-d_j+d_k] where d_i
    the i^{th} eigenlvalue of the frame operator F, the current
    signals of a model, and a cutoff frequency, returns
    the new operators and signals that should be passed to
    create a new Model class after the RWA.

    Args:
        current_ops: the current operator list, (k,n,n) Array
        current_sigs: (k) length SignalList
        rotating_frame: current RotatingFrame object of the pre-RWA model
        frame_freqs: the effective frequencies of different
            matrix elements due to the conjugation by e^{\pm Ft}
            in the rotating frame.
        cutoff_freq: maximum frequency allowed under the RWA.
    Returns:
        SignaLList: (2k,n,n) Array of new operators post RWA.
    """
    current_sigs = current_sigs.flatten()
    carrier_freqs = np.zeros(current_ops.shape[0])

    for i, sig_sum in enumerate(current_sigs.components):
        sig = sig_sum.components[0]
        carrier_freqs[i] = sig.carrier_freq

    num_components = len(carrier_freqs)
    n = current_ops.shape[-1]

    frame_freqs = np.broadcast_to(frame_freqs, (num_components, n, n))
    carrier_freqs = carrier_freqs.reshape((num_components, 1, 1))

    pos_pass = np.abs(carrier_freqs + frame_freqs) < cutoff_freq
    pos_terms = current_ops * pos_pass.astype(int)  # G_i^+

    neg_pass = np.abs(-carrier_freqs + frame_freqs) < cutoff_freq
    neg_terms = current_ops * neg_pass.astype(int)  # G_i^-

    real_component = pos_terms / 2 + neg_terms / 2
    imag_component = 1j * pos_terms / 2 - 1j * neg_terms / 2

    rwa_operators = rotating_frame.operator_out_of_frame_basis(
        np.append(real_component, imag_component, axis=0)
    )

    return rwa_operators


def get_rwa_signals(curr_signal_list: Union[List[Signal], SignalList]):
    """Helper function that converts pre-RWA
    signals to post-RWA signals"""
    if curr_signal_list is None:
        return curr_signal_list

    real_signal_components = []
    imag_signal_components = []

    if not isinstance(curr_signal_list, SignalList):
        curr_signal_list = SignalList(curr_signal_list)

    curr_signal_list = curr_signal_list.flatten()

    for sig_sum in curr_signal_list.components:
        sig = sig_sum.components[0]
        real_signal_components.append(sig)
        imag_signal_components.append(
            SignalSum(Signal(sig.envelope, sig.carrier_freq, sig.phase - np.pi / 2))
        )

    rwa_signals = SignalList(real_signal_components + imag_signal_components)

    return rwa_signals

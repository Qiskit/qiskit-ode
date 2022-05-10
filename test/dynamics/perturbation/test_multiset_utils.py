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

"""Test multiset_utils functions."""

from multiset import Multiset

from qiskit import QiskitError

from qiskit_dynamics.perturbation.multiset_utils import validate_non_negative_ints, multiset_to_sorted_list, clean_multisets, submultiset_filter

from ..common import QiskitDynamicsTestCase


class TestValidation(QiskitDynamicsTestCase):
    """Test validation of multisets."""

    def test_non_int(self):
        """Test that a non-integer entry raises an error."""

        with self.assertRaises(QiskitError) as qe:
            validate_non_negative_ints(Multiset('abc'))
        self.assertTrue('non-negative integers' in str(qe.exception))

    def test_negative_int(self):
        """Test that a negative integer raises an error."""

        with self.assertRaises(QiskitError) as qe:
            validate_non_negative_ints(Multiset([1, 2, -1]))
        self.assertTrue('non-negative integers' in str(qe.exception))


class TestToSortedList(QiskitDynamicsTestCase):
    """Test conversion to sorted list."""

    def test_to_sorted_list(self):
        """Test correct conversion to sorted list."""

        ms = Multiset([0, 2, 4, 1, 2, 5, 0, 2])
        self.assertTrue([0, 0, 1, 2, 2, 2, 4, 5] == multiset_to_sorted_list(ms))

        ms = Multiset({1: 3, 4: 2, 0: 1})
        self.assertTrue([0, 1, 1, 1, 4, 4] == multiset_to_sorted_list(ms))


class TestCleanMultisets(QiskitDynamicsTestCase):
    """Test clean_multisets function for removing duplicates and sorting."""

    def test_clean_multisets(self):
        """Test clean_multisets."""
        output = clean_multisets(
            [
                Multiset({0: 1}),
                Multiset({0: 1, 1: 1}),
                Multiset({0: 2, 1: 1}),
                Multiset({0: 2, 1: 1}),
            ]
        )
        expected = [Multiset({0: 1}), Multiset({0: 1, 1: 1}), Multiset({0: 2, 1: 1})]
        self.assertTrue(output == expected)

        output = clean_multisets(
            [
                Multiset({1: 1, 2: 1, 3: 1}),
                Multiset({0: 1, 1: 1}),
                Multiset({1: 1, 2: 1, 3: 1}),
                Multiset({0: 1}),
            ]
        )
        expected = [Multiset({0: 1}), Multiset({0: 1, 1: 1}), Multiset({1: 1, 2: 1, 3: 1})]
        self.assertTrue(output == expected)


class TestSubmultisetFilter(QiskitDynamicsTestCase):
    """Test submultiset_filter function."""

    def test_submultiset_filter(self):
        """Test submultiset_filter utility function."""

        multiset_list = [Multiset({0: 2, 1: 1}), Multiset({0: 2, 2: 1})]

        multiset_candidates = [Multiset({0: 1}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(
            submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates
        )

        multiset_candidates = [Multiset({0: 2}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(
            submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates
        )

        multiset_candidates = [Multiset({0: 3}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(
            submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates[1:]
        )

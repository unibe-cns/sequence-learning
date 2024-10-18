#!/usr/bin/env python3

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elise.data import Pattern

DT = 0.1


###############################
# test the Base Pattern class #
###############################


@pytest.fixture()
def base_sequence():
    pattern = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return pattern


@pytest.fixture()
def base_pattern(base_sequence):
    pattern = Pattern(base_sequence, dt=DT)
    return pattern


def test_pattern_init(base_pattern, base_sequence):
    assert base_pattern.t_max == len(base_sequence) * DT
    assert base_pattern.shape == base_sequence.shape
    assert_allclose(base_sequence, base_pattern.pattern)


def test_pattern_len(base_pattern, base_sequence):
    assert len(base_pattern) == base_sequence.shape[0]


@pytest.mark.parametrize("idx", [0, (2, 1), np.s_[4, 2], np.s_[:, 1], slice(0, 4, 2)])
def test_pattern_get_item(base_pattern, base_sequence, idx):
    assert_allclose(base_pattern[idx], base_sequence[idx])


###########################################
# End of tests for the Base Pattern class #
###########################################

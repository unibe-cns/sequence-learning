#!/usr/bin/env python3

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elise.data import Dataloader, Pattern

DT = 0.1

# TODO use the factory approach

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


class TestPattern:
    def test_init(self, base_pattern, base_sequence):
        assert base_pattern.dur == len(base_sequence) * DT
        assert base_pattern.shape == base_sequence.shape
        assert_allclose(base_sequence, base_pattern.pattern)

    def test_len(self, base_pattern, base_sequence):
        assert len(base_pattern) == base_sequence.shape[0]

    @pytest.mark.parametrize(
        "idx", [0, (2, 1), np.s_[4, 2], np.s_[:, 1], slice(0, 4, 2)]
    )
    def test_getitem(self, base_pattern, base_sequence, idx):
        assert_allclose(base_pattern[idx], base_sequence[idx])


###################
# Test Dataloader #
###################


# TODO: turn this into a factory!
@pytest.fixture()
def sample_dataloader(base_pattern):
    return Dataloader(base_pattern)


@pytest.fixture()
def sample_dataloader_2(base_pattern):
    return Dataloader(base_pattern, pre_transforms=[])


class TestDataloader:
    def test_init(self, sample_dataloader, base_sequence):
        assert_allclose(sample_dataloader.pat[:, :], base_sequence)
        assert sample_dataloader.dt == pytest.approx(DT)

    @pytest.mark.parametrize(
        ("t", "idx"),
        [
            (0.1, 1),
            (0.5, 5),
            (0.6, 0),
            (0.7, 1),
            (1.1, -1),
            (1.2, 0),
            (0.05, 0),
            (0.15, 1),
            (0.55, 5),
            (0.61, 0),
        ],
    )
    def test_call(self, sample_dataloader, base_sequence, t, idx):
        assert_allclose(sample_dataloader(t), base_sequence[idx])

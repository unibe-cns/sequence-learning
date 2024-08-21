#!/usr/bin/env python3

import unittest

import numpy as np
import pytest

from seqlearn.main import SomaticWeights, WeightConfig


@pytest.fixture
def default_weight_config():
    return WeightConfig(
        {
            "num_lat": 50,
            "num_vis": 13,
            "p": 0.5,
            "q": 0.3,
            "p0": 0.1,
        }
    )


def test_create_weight_matrix_basic(default_weight_config):
    sw = SomaticWeights(default_weight_config)
    weight_matrix, num_in, num_out = sw.create_weight_matrix()

    assert isinstance(weight_matrix, np.ndarray)
    assert weight_matrix.shape == (63, 63)  # 50 + 13 = 63
    assert np.all((weight_matrix == 0) | (weight_matrix == 1))
    assert np.all(np.diag(weight_matrix) == 0)


def test_connectivity_constraints(default_weight_config):
    sw = SomaticWeights(default_weight_config)
    weight_matrix, _, _ = sw.create_weight_matrix()

    assert np.all(weight_matrix[:13, :13] == 0)  # No connections between output neurons
    assert np.all(
        weight_matrix[:13, :] == 0
    )  # No connections from latent to output neurons


def test_statistical_properties(default_weight_config):
    sw = SomaticWeights(default_weight_config)
    weight_matrix, num_in, num_out = sw.create_weight_matrix()

    total_connections = np.sum(weight_matrix)
    expected_connections = (
        np.sum(num_in) - 13
    )  # Subtract initial connections to output neurons
    assert total_connections == expected_connections


def test_consistency(default_weight_config):
    np.random.seed(42)
    sw1 = SomaticWeights(default_weight_config)
    matrix1, _, _ = sw1.create_weight_matrix()

    np.random.seed(42)
    sw2 = SomaticWeights(default_weight_config)
    matrix2, _, _ = sw2.create_weight_matrix()

    assert np.all(matrix1 == matrix2)


def test_invalid_inputs():
    invalid_config = WeightConfig(
        {
            "num_lat": -5,
            "num_vis": 2,
            "p": 1.5,
            "q": 0.3,
            "p0": 0.1,
        }
    )
    with pytest.raises(ValueError):
        SomaticWeights(invalid_config)


if __name__ == "__main__":
    unittest.main()

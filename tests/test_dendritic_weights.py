#!/usr/bin/env python3

import numpy as np

from elise.weights import DendriticWeights


def test_create_weight_matrix_basic(default_weight_config, default_network_config):
    dw = DendriticWeights(default_weight_config)
    weight_matrix, _ = dw(
        default_network_config.num_vis, default_network_config.num_lat
    )
    assert isinstance(weight_matrix, np.ndarray)
    assert weight_matrix.shape == (63, 63)  # 50 + 13 = 63


def test_connectivity_constraints(default_weight_config, default_network_config):
    dw = DendriticWeights(default_weight_config)
    weight_matrix, _ = dw(
        default_network_config.num_vis, default_network_config.num_lat
    )
    assert isinstance(weight_matrix, np.ndarray)
    assert np.all(np.diag(weight_matrix) == 0)


# Test consistency in dend
def test_consistency(default_weight_config, default_network_config):
    dw1 = DendriticWeights(default_weight_config)
    dw2 = DendriticWeights(default_weight_config)
    matrix1, _ = dw1(default_network_config.num_vis, default_network_config.num_lat)
    matrix2, _ = dw2(default_network_config.num_vis, default_network_config.num_lat)
    np.testing.assert_allclose(matrix1, matrix2)

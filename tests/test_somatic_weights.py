#!/usr/bin/env python3
import numpy as np

from elise.model import SomaticWeights


def test_create_weight_matrix_basic(default_weight_config, default_network_config):
    sw = SomaticWeights(default_weight_config)
    weight_matrix, _ = sw(
        default_network_config.num_vis, default_network_config.num_lat
    )

    assert isinstance(weight_matrix, np.ndarray)
    assert weight_matrix.shape == (63, 63)  # 50 + 13 = 63
    assert np.all((weight_matrix == 0) | (weight_matrix == 1))
    assert np.all(np.diag(weight_matrix) == 0)


def test_connectivity_constraints(default_weight_config, default_network_config):
    sw = SomaticWeights(default_weight_config)
    weight_matrix, _ = sw(
        default_network_config.num_vis, default_network_config.num_lat
    )

    assert np.all(weight_matrix[:13, :13] == 0)  # No connections between output neurons
    assert np.all(
        weight_matrix[:13, :] == 0
    )  # No connections from latent to output neurons


def test_consistency(default_weight_config, default_network_config):
    sw1 = SomaticWeights(default_weight_config)
    sw2 = SomaticWeights(default_weight_config)
    matrix1, _ = sw1(default_network_config.num_vis, default_network_config.num_lat)
    matrix2, _ = sw2(default_network_config.num_vis, default_network_config.num_lat)
    np.testing.assert_allclose(matrix1, matrix2)

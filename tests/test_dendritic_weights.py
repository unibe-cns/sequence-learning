#!/usr/bin/env python3


import numpy as np
import pytest

from elise.config import NetworkConfig, WeightConfig
from elise.model import DendriticWeights


@pytest.fixture
def default_weight_config():
    return WeightConfig(
        den_seed=42,
        W_vis_vis=[0.0, 0.5],
        W_vis_lat=[0.0, 0.5],
        W_lat_lat=[0.0, 0.5],
        W_lat_vis=[0.0, 0.5],
    )


@pytest.fixture
def default_network_config():
    return NetworkConfig(
        num_lat=50,
        num_vis=13,
    )


def test_create_weight_matrix_basic(default_weight_config, default_network_config):
    dw = DendriticWeights(default_weight_config)
    weight_matrix = dw(default_network_config.num_vis, default_network_config.num_lat)
    assert isinstance(weight_matrix, np.ndarray)
    assert weight_matrix.shape == (63, 63)  # 50 + 13 = 63


def test_connectivity_constraints(default_weight_config, default_network_config):
    dw = DendriticWeights(default_weight_config)
    weight_matrix = dw(default_network_config.num_vis, default_network_config.num_lat)
    assert isinstance(weight_matrix, np.ndarray)
    assert np.all(np.diag(weight_matrix) == 0)


# Test consistency in dend
def test_consistency(default_weight_config, default_network_config):
    dw1 = DendriticWeights(default_weight_config)
    dw2 = DendriticWeights(default_weight_config)
    matrix1 = dw1(default_network_config.num_vis, default_network_config.num_lat)
    matrix2 = dw2(default_network_config.num_vis, default_network_config.num_lat)
    np.testing.assert_allclose(matrix1, matrix2)

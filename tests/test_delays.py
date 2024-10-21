#!/usr/bin/env python3

import numpy as np
import pytest

from elise.config import NetworkConfig, WeightConfig
from elise.model import DendriticWeights, SomaticWeights


@pytest.fixture
def default_weight_config():
    return WeightConfig(
        w_den_seed=42,
        d_den=[5, 15],
    )


@pytest.fixture
def default_network_config():
    return NetworkConfig(
        num_lat=50,
        num_vis=13,
    )


# Test Dendritic delays
def test_den_delay_basic(default_weight_config, default_network_config):
    dw = DendriticWeights(default_weight_config)
    _, delays = dw(default_network_config.num_vis, default_network_config.num_lat)
    assert len(delays) == 63  # 50 + 13 = 63
    assert np.all((delays >= 5) & (delays <= 15))
    assert np.all(np.equal(np.mod(delays, 1), 0))


def test_den_delay_consistency(default_weight_config, default_network_config):
    dw1 = DendriticWeights(default_weight_config)
    dw2 = DendriticWeights(default_weight_config)
    _, delays1 = dw1(default_network_config.num_vis, default_network_config.num_lat)
    _, delays2 = dw2(default_network_config.num_vis, default_network_config.num_lat)
    np.testing.assert_allclose(delays1, delays2)


# Test Somatic delays
def test_som_delaybasic(default_weight_config, default_network_config):
    sw = SomaticWeights(default_weight_config)
    _, delays = sw(default_network_config.num_vis, default_network_config.num_lat)
    assert isinstance(delays, np.ndarray)
    assert len(delays) == 63  # 50 + 13 = 63
    assert np.all((delays >= 5) & (delays <= 15))
    assert np.all(np.equal(np.mod(delays, 1), 0))


def test_som_matrix_consistency(default_weight_config, default_network_config):
    sw1 = SomaticWeights(default_weight_config)
    sw2 = SomaticWeights(default_weight_config)
    _, delays1 = sw1(default_network_config.num_vis, default_network_config.num_lat)
    _, delays2 = sw2(default_network_config.num_vis, default_network_config.num_lat)
    np.testing.assert_allclose(delays1, delays2)

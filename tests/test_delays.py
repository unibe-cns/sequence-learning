#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
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


@pytest.mark.parametrize("weight_class", [DendriticWeights, SomaticWeights])
def test_delay_basic(weight_class, default_weight_config, default_network_config):
    w = weight_class(default_weight_config)
    _, delays = w(default_network_config.num_vis, default_network_config.num_lat)
    assert len(delays) == 63  # 50 + 13 = 63
    npt.assert_array_less(4, delays)
    npt.assert_array_less(delays, 16)
    assert np.all(np.equal(np.mod(delays, 1), 0))


@pytest.mark.parametrize("weight_class", [DendriticWeights, SomaticWeights])
def test_delay_consistency(weight_class, default_weight_config, default_network_config):
    w1 = weight_class(default_weight_config)
    w2 = weight_class(default_weight_config)
    _, delays1 = w1(default_network_config.num_vis, default_network_config.num_lat)
    _, delays2 = w2(default_network_config.num_vis, default_network_config.num_lat)
    npt.assert_allclose(delays1, delays2)

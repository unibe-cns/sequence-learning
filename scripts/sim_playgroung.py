#!/usr/bin/env python3

import numpy as np

from elise.config import NetworkConfig, NeuronConfig, WeightConfig
from elise.model import DendriticWeights, Network, Neurons, SomaticWeights
from elise.rate_buffer import Buffer

if __name__ == "__main__":
    # Plot theoretical distribution
    # Plot for p = 0.1, 0.3, 0.5
    #
    # Test dendritic weight matrix creation

    dendritic_weights = DendriticWeights(WeightConfig)
    somatic_weights = SomaticWeights(WeightConfig)
    neurons = Neurons
    rate_buffer = Buffer

    # Create network
    network = Network(
        NetworkConfig,
        WeightConfig,
        NeuronConfig,
        dendritic_weights,
        somatic_weights,
        neurons,
        rate_buffer,
    )

    u_inp = np.random.rand(13)
    network.simulation_step(0.1, u_inp)

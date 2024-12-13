#!/usr/bin/env python3

import numpy as np

from elise.config import FullConfig
from elise.data import Dataloader, MultiHotPattern
from elise.logger import Logger
from elise.model import Network
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
from elise.simulator import Simulation
from elise.weights import DendriticWeights, SomaticWeights

# Config
full_config = FullConfig("config.toml")
neuron_params = full_config.neuron_params
network_params = full_config.network_params
simulation_params = full_config.simulation_params
weight_params = full_config.weight_params

# Network
rate_buffer = Buffer
dendritic_weights = DendriticWeights(weight_params)
somatic_weights = SomaticWeights(weight_params)
network = Network(
    network_params, neuron_params, dendritic_weights, somatic_weights, rate_buffer
)

# Logger
logger = Logger("log.txt")

# Pattern & Dataloader
pat = np.loadtxt("fuer_elise_short.txt", delimiter=",", skiprows=1).astype(int)
pat_duration = 1.0
pattern_width = network_params.num_vis
fuer_elise = MultiHotPattern(pat, pat_duration, pattern_width)
data_loader = Dataloader(fuer_elise)

# Simulator
optimizer = SimpleUpdater
sim = Simulation(simulation_params, optimizer, network, logger, data_loader)

sim.run()

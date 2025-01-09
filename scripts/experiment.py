#!/usr/bin/env python3

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_activity, plot_weights
from tqdm import tqdm

from elise.config import FullConfig
from elise.data import Dataloader, MultiHotPattern
from elise.logger import Logger
from elise.model import Network
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
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

# Simulator
optimizer = SimpleUpdater

dt = simulation_params.dt
training_cycles = simulation_params.training_cycles
replay_cycles = simulation_params.replay_cycles
validation_cycles = simulation_params.validation_cycles
validation_interval = simulation_params.validation_interval

# Optimizer
eta_lat = simulation_params.eta_lat
eta_vis = simulation_params.eta_out
optimizer_lat = optimizer(eta_lat)
optimizer_vis = optimizer(eta_vis)

network.prepare_for_simulation(dt, optimizer_vis, optimizer_lat)


def to_biounits(x):
    return (
        neuron_params.E_l + x * 20.0
    )  # TODO: include the 20. as a parameter in the config.


# Pattern & Dataloader
pat = np.loadtxt("fuer_elise_short.txt", delimiter=",", skiprows=1).astype(int)
pat_duration = 250.0
pattern_width = network_params.num_vis
fuer_elise = MultiHotPattern(pat, pat_duration, pattern_width)
dataloader = Dataloader(fuer_elise, pre_transforms=[to_biounits])

full_pattern = dataloader.get_full_pattern(dt)

training_duration = 10 * pat_duration
validation_duration = 0 * pat_duration
replay_duration = 6 * pat_duration
num_epochs = 10

pre_transform = []

# Training
simulation_output = []
target = []
r_bar = []
weights = defaultdict(list)

weight_types = ["vis_vis", "vis_lat", "lat_lat", "lat_vis"]

for epoch in tqdm(range(num_epochs)):
    for _, u_tgt in dataloader.iter(0, training_duration, dt):
        network(u_inp=u_tgt)

        output = network.get_output()

        simulation_output.append(output)

        target.append(u_tgt)

    for _, u_tgt in dataloader.iter(0, validation_duration, dt):
        network(u_inp=None)

        output = network.get_output()

        simulation_output.append(output)

        target.append(u_tgt)


for _, u_tgt in dataloader.iter(0, replay_duration, dt):
    network(u_inp=None)

    output = network.get_output()

    simulation_output.append(output)

    target.append(u_tgt)

simulation_output = np.array(simulation_output)
target = np.array(target)

plot_activity(simulation_output, target, full_pattern, replay_duration, dt)
plot_weights(network)

plt.show()

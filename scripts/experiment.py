#!/usr/bin/env python3

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_activity, plot_weights
from tqdm import tqdm

from elise.config import FullConfig
from elise.data import Dataloader, MultiHotPattern
from elise.logger import Logger
from elise.model import Network, eq_phi
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
from elise.stats import mse, pearson_coef, window_slider
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


def eq_phi_(x):
    return eq_phi(
        x,
        neuron_params.a,
        neuron_params.b,
    )


def calc_val_mse(data, target):
    return np.min(window_slider(data, target, mse))


def calc_val_pearson(data, target):
    return np.max(window_slider(data, target, pearson_coef))


# Pattern & Dataloader
pat = np.loadtxt("fuer_elise_short.txt", delimiter=",", skiprows=1).astype(int)
pat_duration = 250.0
pattern_width = network_params.num_vis
fuer_elise = MultiHotPattern(pat, pat_duration, pattern_width)
dataloader = Dataloader(fuer_elise, pre_transforms=[to_biounits])

full_pattern_u = dataloader.get_full_pattern(dt)
full_pattern_r = eq_phi_(full_pattern_u)

training_duration = 10 * pat_duration
validation_duration = 1.5 * pat_duration
replay_duration = 6 * pat_duration
num_epochs = 10

pre_transform = []

# Training
simulation_output = []
target = []
r_bar = []
weights = defaultdict(list)

all_mses = defaultdict(list)
all_corr_coefs = defaultdict(list)

weight_types = ["vis_vis", "vis_lat", "lat_lat", "lat_vis"]

for epoch in tqdm(range(num_epochs)):
    # TRAINING ITERATION
    for _, u_tgt in dataloader.iter(0, training_duration, dt):
        network(u_inp=u_tgt)

        output = network.get_output()

        simulation_output.append(output)

        target.append(u_tgt)

    # VALIDATION ITERATION
    val_output = []
    for _, u_tgt in dataloader.iter(0, validation_duration, dt):
        network(u_inp=None)

        output = network.get_output()

        simulation_output.append(output)
        val_output.append(output)

        target.append(u_tgt)

    val_output_u = np.array(val_output)
    val_output_r = eq_phi_(val_output_u)

    all_mses["u"].append(calc_val_mse(val_output_u, full_pattern_u))
    all_mses["r"].append(calc_val_mse(val_output_r, full_pattern_r))

    all_corr_coefs["u"].append(calc_val_pearson(val_output_u, full_pattern_u))
    all_corr_coefs["r"].append(calc_val_pearson(val_output_r, full_pattern_r))

# FINAL REPLAY
for _, u_tgt in dataloader.iter(0, replay_duration, dt):
    network(u_inp=None)

    output = network.get_output()

    simulation_output.append(output)

    target.append(u_tgt)

simulation_output = np.array(simulation_output)
target = np.array(target)

plot_activity(simulation_output, target, full_pattern_u, replay_duration, dt)
plot_weights(network)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].set_title("validation.")
ax[0].plot(all_mses["u"])
ax[1].plot(all_mses["r"])
ax[0].set_ylabel("MSE on u")
ax[1].set_ylabel("MSE on r")
ax[2].plot(all_corr_coefs["u"])
ax[3].plot(all_corr_coefs["r"])
ax[2].set_ylabel("pearson on u")
ax[3].set_ylabel("pearson on r")
ax[3].set_xlabel("epochs")

plt.show()

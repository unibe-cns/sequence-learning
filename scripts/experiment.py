#!/usr/bin/env python3

import ast
import pickle as pkl
from collections import defaultdict

import numpy as np
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


def get_pattern_from_txt(filename):
    with open(filename, "r") as file:
        content = file.read().strip()

    result = ast.literal_eval(f"[{content}]")

    return result


# Pattern & Dataloader
pat = get_pattern_from_txt("patterns/small_complex.txt")
pat_duration = 250.0
pattern_width = network_params.num_vis
fuer_elise = MultiHotPattern(pat, pat_duration, pattern_width)
dataloader = Dataloader(fuer_elise, pre_transforms=[to_biounits])

full_pattern_u = dataloader.get_full_pattern(dt)
full_pattern_r = eq_phi_(full_pattern_u)

training_duration = 25 * pat_duration
validation_duration = 2 * pat_duration
replay_duration = 4 * pat_duration
num_epochs = 50

pre_transform = []

# Training
simulation_output = []
target = []
weights = defaultdict(list)

all_mses = defaultdict(list)
all_corr_coefs = defaultdict(list)

weight_types = ["vis_vis", "vis_lat", "lat_lat", "lat_vis"]

output_rates = []
target_rates = []

for epoch in tqdm(range(num_epochs)):
    # TRAINING ITERATION
    for _, u_tgt in dataloader.iter(0, training_duration, dt):
        network(u_inp=u_tgt)

        output = network.get_output()

        simulation_output.append(output)

        target.append(u_tgt)

    # VALIDATION ITERATION
    val_output = []
    tgt_output = []
    for _, u_tgt in dataloader.iter(0, validation_duration, dt):
        network(u_inp=None)

        output = network.get_output()

        simulation_output.append(output)
        val_output.append(output)
        tgt_output.append(u_tgt)

    val_output_u = np.array(val_output)
    tgt_output_u = np.array(tgt_output)

    # Convert to rate
    val_output_r = eq_phi_(val_output_u)
    val_target_r = eq_phi_(tgt_output_u)

    output_rates.append(val_output_r.T)
    target_rates.append(val_target_r.T)

    all_mses["u"].append(calc_val_mse(val_output_u, full_pattern_u))
    all_mses["r"].append(calc_val_mse(val_output_r, full_pattern_r))

    all_corr_coefs["u"].append(calc_val_pearson(val_output_u, full_pattern_u))
    all_corr_coefs["r"].append(calc_val_pearson(val_output_r, full_pattern_r))


# FINAL REPLAY
pat = get_pattern_from_txt("patterns/disruption_excitation.txt")
pattern_width = network_params.num_vis
disruption_duration = 250.0
pat = MultiHotPattern(pat, pat_duration, pattern_width)
disruption_excitation = Dataloader(pat, pre_transforms=[to_biounits])

for _, u_tgt in disruption_excitation.iter(0, disruption_duration, dt):
    network.u[: network.num_vis] = u_tgt

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

results = {
    "simulation_output": simulation_output,
    "target": target,
    "full_pattern_u": full_pattern_u,
    "replay_duration": replay_duration,
    "validation_duration": validation_duration,
    "disruption_duration": disruption_duration,
    "dt": dt,
    "network": network,
    "all_mses": all_mses,
    "all_corr_coefs": all_corr_coefs,
    "output_rates": output_rates,
    "target_rates": target_rates,
    "num_epochs": num_epochs,
}

with open("results.pkl", "wb") as f:
    pkl.dump(results, f)

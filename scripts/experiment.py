#!/usr/bin/env python3

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_weights
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
replay_duration = 2 * pat_duration
num_epochs = 250

pat = np.loadtxt("fuer_elise_short.txt", delimiter=",", skiprows=1).astype(int)
pat_duration = 25.0
pattern_width = network_params.num_vis
fuer_elise = MultiHotPattern(pat, pat_duration, pattern_width)
data_loader = Dataloader(fuer_elise, pre_transforms=[to_biounits])

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

# r_bar = np.array(r_bar)

# # Plot r bar
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# ax.plot(r_bar)
# ax.set_title("R bar")
# ax.set_xlabel("Time step")
# ax.set_ylabel("R bar")
# ax.grid(True, linestyle='--', alpha=0.3)


# fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
# fig.suptitle('Weight Progression Throughout Simulation', fontsize=16)

# weight_types = ['vis_vis', 'vis_lat', 'lat_lat', 'lat_vis']
# titles = ['Visible-to-Visible',
# 'Visible-to-Lateral',
# 'Lateral-to-Lateral',
# 'Lateral-to-Visible']

# for i, (weight_type, title) in enumerate(zip(weight_types, titles)):
#     weight_data = weights[weight_type]

#     # Plot each weight progression
#     for j in range(weight_data.shape[1]):
#         axs[i].plot(weight_data[:, j], alpha=0.5, linewidth=0.5)

#     axs[i].set_title(f'{title} Weights')
#     axs[i].set_ylabel('Weight Value')
#     axs[i].grid(True, linestyle='--', alpha=0.3)

# axs[-1].set_xlabel('Time Step')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title


# def plot_weights_over_time(weights):

#     fig, axs = plt.subplots(4, 1, figsize=(20, 20))
#     axs = axs.flatten()
#     ax_labels = ['vis_vis', 'vis_lat', 'lat_lat', 'lat_vis']
#     ax_titles = ['Vis-Vis Weights',
#     'Vis-Lat Weights',
#     'Lat-Lat Weights',
#     'Lat-Vis Weights']
#     ax[0].set_title('Weights Over Time')

#     for i, ax in enumerate(axs):

#         ax.imshow(weights[ax_labels[i]], aspect="auto", interpolation="none")
#         ax.set_title(ax_titles[i])
#         ax.set_ylabel("Neuron index")
#         ax.set_xlabel("Neuron index")


#     plt.tight_layout()

# plt.show()

# Create a figure with two subplots, sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot simulation output
last = (2 * len(full_pattern)) + int(replay_duration / dt)

im1 = ax1.imshow(simulation_output[-last:].T, aspect="auto", interpolation="none")
colorbar1 = fig.colorbar(im1, ax=ax1)
colorbar1.set_label("Output activity")
ax1.set_ylabel("Neuron index")
ax1.set_title("Simulation output")

# Plot target output
im2 = ax2.imshow(target[-last:].T, aspect="auto", interpolation="none")
colorbar2 = fig.colorbar(im2, ax=ax2)
colorbar2.set_label("Output activity")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Neuron index")
ax2.set_title("Target output")

# Adjust layout and show the plot
plt.tight_layout()

plt.show()

plot_weights(network)
breakpoint()

import copy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from elise.config import FullConfig
from elise.model import DendriticWeights, Network, SomaticWeights
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer

if __name__ == "__main__":
    # Plot theoretical distribution
    # Plot for p = 0.1, 0.3, 0.5
    #

    # Test dendritic weight matrix creation
    config_location = "config.toml"
    config = FullConfig(config_location)

    dendritic_weights = DendriticWeights(config.weight_params)
    somatic_weights = SomaticWeights(config.weight_params)
    rate_buffer = Buffer

    # Create network
    network = Network(
        config.network_params,
        config.neuron_params,
        dendritic_weights,
        somatic_weights,
        rate_buffer,
    )

    eta = 0.001
    optimizer_vis = SimpleUpdater(eta)
    optimizer_lat = SimpleUpdater(eta)

    def create_simple_seq(size):
        start = np.eye(size) * 0.9
        middle = start[::-1]
        end = middle[::2]

        return np.concatenate([start, middle, end])

    reps = 5000
    size = config.network_params.num_vis
    sequence = create_simple_seq(size)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(sequence, aspect="auto")
    ax.set_title("Sequence")
    plt.show()

    network.prepare_for_simulation(dt=0.1)

    latent_activity = []
    visual_activity = []
    pattern = []
    weight_av = []

    # Learning
    for rep in tqdm(range(reps)):
        for u_inp in sequence:
            for i in range(25):
                network.simulation_step(u_inp, optimizer_vis, optimizer_lat)

                if rep > reps - 4:
                    lat_act = copy.deepcopy(network.u[size:])
                    vis_act = copy.deepcopy(network.u[:size])

                    latent_activity.append(lat_act)
                    visual_activity.append(vis_act)
                    weight_av.append(np.mean(network.dendritic_weights))
                    pattern.append(u_inp)

    # Replay without input
    for i in range(500):
        u_inp = np.zeros(size)
        network.simulation_step(u_inp, optimizer_vis, optimizer_lat)

        lat_act = copy.deepcopy(network.u[size:])
        vis_act = copy.deepcopy(network.u[:size])

        latent_activity.append(lat_act)
        visual_activity.append(vis_act)
        weight_av.append(np.mean(network.dendritic_weights))
        pattern.append(u_inp)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(latent_activity, aspect="auto")
    ax.set_title("Latent activity")

    fig, ax = plt.subplots(1, 1)
    ax.imshow(visual_activity, aspect="auto")
    ax.set_title("Visual activity")

    fig, ax = plt.subplots(1, 1)
    ax.plot(weight_av)
    ax.set_title("Average weight")

    fig, ax = plt.subplots(1, 1)
    ax.imshow(pattern, aspect="auto")
    ax.set_title("Input")

    # Plot weight matrix
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].imshow(network.dendritic_weights, aspect="auto")
    ax[1].imshow(network.somatic_weights, aspect="auto")
    ax[0].set_title("Dendritic Weight matrix")

    plt.show()

#!/usr/bin/env python3

import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np


def plot_weights(network):
    # Weights
    somatic_weights = network.somatic_weights
    dendritic_weights = network.dendritic_weights

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    # Plot somatic weights
    axs[0].imshow(somatic_weights, aspect="auto", interpolation="none")
    axs[0].set_title("Somatic weights")

    # Plot dendritic weights
    im2 = axs[1].imshow(dendritic_weights, aspect="auto", interpolation="none")
    axs[1].set_title("Dendritic weights")
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label("Weight strength")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig


def plot_weights_over_time(weights):
    fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    fig.suptitle("Weight Progression Throughout Simulation", fontsize=16)

    weight_types = ["vis_vis", "vis_lat", "lat_lat", "lat_vis"]
    titles = [
        "Visible-to-Visible",
        "Visible-to-Lateral",
        "Lateral-to-Lateral",
        "Lateral-to-Visible",
    ]

    for i, (weight_type, title) in enumerate(zip(weight_types, titles)):
        weight_data = weights[weight_type]

        # Plot each weight progression
        for j in range(weight_data.shape[1]):
            axs[i].plot(weight_data[:, j], alpha=0.5, linewidth=0.5)

        axs[i].set_title(f"{title} Weights")
        axs[i].set_ylabel("Weight Value")
        axs[i].grid(True, linestyle="--", alpha=0.3)

    axs[-1].set_xlabel("Time Step")

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust layout to make room for the title
    return fig


def plot_activity(simulation_output, target, full_pattern, replay_duration, dt):
    # Create a figure with two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    # Plot simulation output
    last = (4 * len(full_pattern)) + int(replay_duration / dt)

    colormap = plt.get_cmap("Blues")
    im1 = ax1.imshow(
        simulation_output[-last:].T, aspect="auto", interpolation="none", cmap=colormap
    )
    colorbar1 = fig.colorbar(im1, ax=ax1)
    colorbar1.set_label("Membrane potential")
    ax1.set_ylabel("Neuron index")
    ax1.set_title("Output Membrane Potential")
    # Add a vertical line after 2 pattern durations
    ax1.axvline(2 * len(full_pattern), color="red", linestyle="--")

    # Plot target output
    im2 = ax2.imshow(
        target[-last:].T, aspect="auto", interpolation="none", cmap=colormap
    )
    colorbar2 = fig.colorbar(im2, ax=ax2)
    colorbar2.set_label("Membrane potential")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Neuron index")
    ax2.set_title("Target Membrane Potential")
    plt.tight_layout()
    ax2.axvline(2 * len(full_pattern), color="red", linestyle="--")

    # Adjust layout and show the plot

    # Saving the figure as pdf
    return fig


def plot_arnos_activity(output_rates, target_rates, num_epochs, nr_plots):
    fig, ax = plt.subplots(nr_plots, 1, figsize=(nr_plots + 1, 6), sharex=True)

    every = int(num_epochs / nr_plots)
    for i in range(nr_plots):
        current_output = output_rates[i * every]
        current_target = target_rates[i * every]

        thresh = 0.35

        # Output rates > threshold = 1, else 0 not using where
        current_output = (current_output > thresh).astype(int)
        current_target = (current_target > thresh).astype(int)

        # Create a new RGB image (white background)
        height, width = current_target.shape
        result = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Create masks for different conditions
        only_target = np.logical_and(current_target == 1, current_output == 0)
        only_output = np.logical_and(current_output == 1, current_target == 0)
        both = np.logical_and(current_output == 1, current_target == 1)

        # Set colors
        result[only_target] = [0, 255, 255]  # Cyan for only target
        result[only_output] = [255, 0, 0]  # Red for only output
        result[both] = [0, 0, 0]  # Black for both

        ax[i].imshow(result, aspect="auto", interpolation="none")
        ax[i].set_title(f"Epoch {i*every}")
        ax[i].set_ylabel("Neuron index")
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])

    ax[-1].set_xlabel("Time (ms)")
    plt.tight_layout()

    return fig


def plot_correlation(all_mses, all_corr_coefs):
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


if __name__ == "__main__":
    # Load results
    with open("results.pkl", "rb") as f:
        results = pkl.load(f)

    output_rates = results["output_rates"]
    target_rates = results["target_rates"]
    num_epochs = results["num_epochs"]
    full_pattern_u = results["full_pattern_u"]
    replay_duration = results["replay_duration"]
    dt = results["dt"]
    network = results["network"]
    simulation_output = results["simulation_output"]
    target = results["target"]

    dpi = 300
    fig = plot_arnos_activity(output_rates, target_rates, num_epochs, nr_plots=4)
    plt.savefig("figs/arnos_activity.png", dpi=dpi)
    fig = plot_activity(simulation_output, target, full_pattern_u, replay_duration, dt)
    plt.savefig("figs/activity.png", dpi=dpi)
    fig = plot_weights(network)
    plt.savefig("figs/weights.png", dpi=dpi)
    fig = plot_correlation(results["all_mses"], results["all_corr_coefs"])
    plt.savefig("figs/mse.png", dpi=dpi)

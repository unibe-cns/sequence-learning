#!/usr/bin/env python3

import matplotlib.pyplot as plt


def plot_weights(network):
    # Weights
    somatic_weights = network.somatic_weights
    dendritic_weights = network.dendritic_weights

    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    # Plot somatic weights
    im1 = axs[0].imshow(somatic_weights, aspect="auto", interpolation="none")
    axs[0].set_title("Somatic weights")
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar1.set_label("Weight value")

    # Plot dendritic weights
    im2 = axs[1].imshow(dendritic_weights, aspect="auto", interpolation="none")
    axs[1].set_title("Dendritic weights")
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar2.set_label("Weight value")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


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


def plot_activity(simulation_output, target, full_pattern, replay_duration, dt):
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
    plt.tight_layout()

    # Adjust layout and show the plot

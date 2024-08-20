#!/usr/bin/env python3
"""Main script for creating weight matrix."""

# TODO add Dendritic weights
# TODO add delays to Dentritic weights class
# TODO add delays to Somatic weights class
# TODO add network class
# - Add one that follows dependency inversion principle
# - Add one that does not use dependency inversion principle

from abc import abstractmethod
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import toml


class Weights:
    """
    Base class for weight matrices in neural networks.

    This class initializes basic parameters and structures for weight matrices,
    including the number of latent and output neurons, and arrays for weights
    and connection counts.

    :param config: A dictionary containing configuration parameters.
    :type config: dict
    """

    def __init__(self, config):
        self.num_latent = config["weight_params"]["latent_neurons"]
        self.num_output = config["weight_params"]["output_neurons"]
        self.num_total = self.num_latent + self.num_output
        self.weights = None
        self.num_in = None
        self.num_out = None

    @abstractmethod
    def create_weight_matrix(self):
        """
        Abstract method to create the weight matrix.

        This method should be implemented by subclasses to define
        how the weight matrix is created.
        """
        pass


class DendriticWeights(Weights):
    """
    Class for creating dendritic weight matrices.

    This class extends the Weights class to create weight matrices
    specific to dendritic connections in neural networks.

    :param config: A dictionary containing configuration parameters.
    :type config: dict
    """

    def __init__(self, config):
        super().__init__(config)
        self.W_out_out = config["weight_params"]["W_out_out"]
        self.W_out_lat = config["weight_params"]["W_out_lat"]
        self.W_lat_out = config["weight_params"]["W_lat_out"]
        self.W_lat_lat = config["weight_params"]["W_lat_lat"]
        self.weights = self.create_weight_matrix(
            self.W_out_out,
            self.W_out_lat,
            self.W_lat_out,
            self.W_lat_lat,
            self.num_latent,
            self.num_output,
        )

    def create_weight_matrix(self):
        """
        Create the dendritic weight matrix.

        This method should be implemented to define how the dendritic
        weight matrix is created based on the specified parameters.
        """
        pass


class SomaticWeights(Weights):
    """
    Class for creating somatic weight matrices.

    This class extends the Weights class to create weight matrices
    specific to somatic connections in neural networks.

    :param config: A dictionary containing configuration parameters.
    :type config: dict
    """

    def __init__(self, config):
        super().__init__(config)
        self.p = config["weight_params"]["p"]
        self.q = config["weight_params"]["q"]
        self.p0 = config["weight_params"]["p0"]
        self.p_first = 1 - self.p0
        self.weights, self.num_in, self.num_out = self.create_weight_matrix(
            self.p, self.q, self.p0, self.num_latent, self.num_output, self.num_total
        )

    def create_weight_matrix(
        self,
        p: float,
        q: float,
        p0: float,
        num_latent: int,
        num_output: int,
        num_total: int,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        neurons_outgoing = np.arange(self.num_total)
        neurons_incoming = np.arange(self.num_output, self.num_total)
        weight_matrix = np.zeros((self.num_total, self.num_total))

        # Incoming connections
        num_in = np.zeros(self.num_total)
        num_in[: self.num_output] = 1

        # Outgoing connections
        num_out = np.zeros(self.num_total)

        # Neurons that can make a connection
        neurons_unspent = np.arange(num_total)
        neurons_looking = neurons_outgoing[num_in > 0]
        while len(neurons_looking) > 0:
            for idx_pre in neurons_looking:
                while np.isin(idx_pre, neurons_looking):
                    # Probability for forming connection
                    if num_out[idx_pre] == 0:
                        prob_out = 1 - p0
                    else:
                        prob_out = np.power(self.p, num_out[idx_pre])

                    # Test for formation
                    formation = np.random.binomial(1, prob_out)
                    if not formation:
                        # Remove neuron pre from list of unspent neurons
                        neurons_unspent = neurons_unspent[neurons_unspent != idx_pre]
                        neurons_looking = np.delete(
                            neurons_looking, 0
                        )  # TODO Test doing it like unspent
                    else:
                        # Possible post partners exluding self connection
                        possible_post = neurons_incoming[neurons_incoming != idx_pre]
                        formed = 0
                        while not formed:
                            post_idx = np.random.choice(possible_post)
                            prob_in = np.power(self.q, num_in[post_idx])
                            accept = np.random.binomial(1, prob_in)

                            if accept:
                                # Add connection to matrix
                                weight_matrix[post_idx, idx_pre] = 1
                                num_out[idx_pre] += 1
                                num_in[post_idx] += 1  # TODO PROBLEM with indexing??
                                if np.isin(post_idx, neurons_unspent):
                                    neurons_unspent = neurons_unspent[
                                        neurons_unspent != post_idx
                                    ]
                                    neurons_looking = np.append(
                                        neurons_looking, post_idx
                                    )
                                formed = 1

                                if np.sum(weight_matrix) != np.sum(num_in) - num_output:
                                    print("Problem with total connections")
                                    breakpoint()

                            else:
                                continue

        return weight_matrix, num_in, num_out


# Main
if __name__ == "__main__":
    # Load config file
    config = toml.load("config.toml")

    # Create weight matrix
    weights = SomaticWeights(config)

    # Out
    weight_matrix = weights.weights
    num_in = weights.num_in
    num_out = weights.num_out

    # Set seed
    np.random.seed(config["seed"])

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(num_in)
    ax[0].set_title("Number of incoming connections")
    ax[1].hist(num_out)
    ax[1].set_title("Number of outgoing connections")
    plt.show()

    # Scatter plot of num in vs num out
    fig, ax = plt.subplots()
    ax.scatter(num_in, num_out)
    ax.set_xlabel("Number of incoming connections")
    ax.set_ylabel("Number of outgoing connections")
    plt.show()

    # Plot weight matrix
    fig, ax = plt.subplots()
    ax.imshow(weight_matrix)
    plt.show()

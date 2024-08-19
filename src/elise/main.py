#!/usr/bin/env python3
"""Main script for creating weight matrix."""

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
        self.n_latent = config["weight_params"]["latent_neurons"]
        self.n_output = config["weight_params"]["output_neurons"]
        self.weights = np.zeros((self.n_latent, self.n_latent))
        self.num_in = np.zeros(self.n_latent)
        self.num_out = np.zeros(self.n_latent)

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
            self.n_latent,
            self.n_output,
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
            self.p, self.q, self.p0, self.n_latent, self.n_output
        )

    def create_weight_matrix(
        self, p: float, q: float, p0: float, n: int, n_inputs: int
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Create the weight matrix."""
        """
        Create a weight matrix for a neural network with probabilistic connections.

        This function generates a weight matrix for a neural network where connections
        are formed based on probabilistic rules. It also tracks the number of incoming
        and outgoing connections for each neuron.

        Parameters
        ----------
        p : float
            Probability factor for outgoing connections.
        q : float
            Probability factor for incoming connections.
        p0 : float
            Initial probability for forming the first outgoing connection.
        n : int
            Total number of neurons in the network.
        n_inputs : int
            Number of input neurons.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
            A tuple containing:
            - weight_matrix : npt.NDArray
                An n x n binary matrix representing the connections between neurons.
            - num_in : npt.NDArray
                An array of length n representing the number of incoming connections
        for each neuron.
            - num_out : npt.NDArray
                An array of length n representing the number of outgoing connections
        for each neuron.

        Notes
        -----
        The connection formation process is as follows:
        1. Start with input neurons as potential connection sources.
        2. For each source neuron, attempt to form connections based on probabilities.
        3. If a connection is formed, select a target neuron and test for acceptance.
        4. Update the weight matrix and connection counts accordingly.
        5. Continue until all possible connections have been attempted.

        The probability of forming an outgoing connection decreases with each successful
        connection, while the probability of accepting an incoming connection decreases
        with the number of existing incoming connections.
        """
        neurons_out = np.arange(self.n_latent)
        neurons_in = np.arange(self.n_output, self.n_latent)
        weight_matrix = np.zeros((self.n_latent, self.n_latent))

        # Incoming connections
        num_in = np.zeros(self.n_latent)
        num_in[: self.n_output] = 1

        # Outgoing connections
        num_out = np.zeros(self.n_latent)

        # Neurons that can make a connection
        neurons_unspent = np.arange(n)
        neurons_looking = neurons_out[num_in > 0]
        while len(neurons_looking) > 0:
            for neuron_pre in neurons_looking:
                while np.isin(neuron_pre, neurons_looking):
                    # Probability for forming connection
                    nr_out = num_out[neuron_pre]
                    if nr_out == 0:
                        prob_out = 1 - p0
                    else:
                        prob_out = np.power(self.p, nr_out)

                    # Test for formation
                    formation = np.random.binomial(1, prob_out)
                    if not formation:
                        # Remove from list of neurons unspent
                        neurons_unspent = neurons_unspent[neurons_unspent != neuron_pre]
                        neurons_looking = np.delete(neurons_looking, 0)
                    else:
                        # Possible post partners exluding self connection
                        possible_post = neurons_in[neurons_in != neuron_pre]
                        formed = 0
                        while not formed:
                            neuron_post = np.random.choice(possible_post)
                            prob_in = np.power(self.q, num_in[neuron_post])
                            accept = np.random.binomial(1, prob_in)

                            if accept:
                                # Add connection to matrix
                                weight_matrix[neuron_post, neuron_pre] = 1
                                num_out[neuron_pre] += 1
                                num_in[neuron_post] += 1
                                if np.isin(neuron_post, neurons_unspent):
                                    neurons_unspent = neurons_unspent[
                                        neurons_unspent != neuron_post
                                    ]
                                    neurons_looking = np.append(
                                        neurons_looking, neuron_post
                                    )
                                formed = 1

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

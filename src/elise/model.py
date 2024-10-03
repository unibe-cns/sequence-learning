#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .config import NetworkConfig, NeuronConfig, WeightConfig
from .rate_buffer import Buffer


class Weights(ABC):
    """
    Base class for weight matrices in neural networks.
    """

    def __init__(self, weight_config: WeightConfig):
        pass

    @abstractmethod
    def _create_weight_matrix(self, num_vis: int, num_lat: int):
        pass

    @abstractmethod
    def __call__(self, num_vis: int, num_lat: int):
        pass


class DendriticWeights(Weights):
    """
    Class for creating dendritic weight matrices.

    This class extends the base Weights class and provides functionality to create
    dendritic weight matrices based on the given configuration.

    :param weight_config: Configuration object containing weight parameters.
    :type weight_config: WeightConfig
    """

    def __init__(self, weight_config: WeightConfig):
        """
        Initialize the DendriticWeights object.

        :param weight_config: Configuration object containing weight parameters.
        :type weight_config: WeightConfig
        """
        super().__init__(weight_config)
        self.W_vis_vis = weight_config.W_vis_vis
        self.W_vis_lat = weight_config.W_vis_lat
        self.W_lat_vis = weight_config.W_lat_vis
        self.W_lat_lat = weight_config.W_lat_lat
        self.rng = np.random.default_rng(seed=weight_config.den_seed)

    def __call__(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create a dendritic weight matrix.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        weight_matrix = self._create_weight_matrix(num_vis, num_lat)
        return weight_matrix

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create the dendritic weight matrix.

        This method initializes a weight matrix and fills it with random values
        based on the configuration parameters.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        # Initialize weight matrix
        weights = np.zeros((num_vis + num_lat, num_vis + num_lat))
        weights[num_vis:, num_vis:] = self.rng.uniform(
            self.W_lat_lat[0], self.W_lat_lat[1], (num_lat, num_lat)
        )  # Lat to Lat
        weights[num_vis:, :num_vis] = self.rng.uniform(
            self.W_lat_vis[0], self.W_lat_vis[1], (num_lat, num_vis)
        )  # Lat to Vis
        weights[:num_vis, num_vis:] = self.rng.uniform(
            self.W_vis_lat[0], self.W_vis_lat[1], (num_vis, num_lat)
        )  # Vis to Lat
        weights[:num_vis, :num_vis:] = self.rng.uniform(
            self.W_vis_vis[0], self.W_vis_vis[1], (num_vis, num_vis)
        )  # Vis to Vis

        # remove self connections
        np.fill_diagonal(weights, 0)

        return weights


class SomaticWeights(Weights):
    """
    Class for creating somatic weight matrices.

    This class extends the base Weights class and provides functionality to create
    somatic weight matrices based on probabilistic connection rules.

    :param weight_config: Configuration object containing weight parameters.
    :type weight_config: WeightConfig
    """

    def __init__(self, weight_config: WeightConfig):
        """
        Initialize the SomaticWeights object.

        :param weight_config: Configuration object containing weight parameters.
        :type weight_config: WeightConfig
        """
        super().__init__(weight_config)
        self.p = weight_config.p
        self.q = weight_config.q
        self.p0 = weight_config.p0
        self.p_first = 1 - self.p0
        self.rng = np.random.default_rng(seed=weight_config.som_seed)

    def __call__(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create a somatic weight matrix.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        weight_matrix = self._create_weight_matrix(num_vis, num_lat)
        return weight_matrix

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create the somatic weight matrix based on probabilistic connection rules.

        This method implements a complex algorithm to create connections between
        neurons based on various probabilities and rules.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        num_total = num_vis + num_lat
        neurons_outgoing = np.arange(num_total)
        neurons_incoming = np.arange(num_vis, num_total)
        weight_matrix = np.zeros((num_total, num_total))

        # Incoming connections
        connections_in = np.zeros(num_total)
        connections_in[:num_vis] = 1

        # Outgoing connections
        connections_out = np.zeros(num_total)

        # Neurons that can make a connection
        neurons_unspent = np.arange(num_total)
        neurons_looking = neurons_outgoing[connections_in > 0]

        while len(neurons_looking) > 0:
            for idx_pre in neurons_looking:
                while np.isin(idx_pre, neurons_looking):
                    # Probability for forming connection
                    if connections_out[idx_pre] == 0:
                        prob_out = 1 - self.p0
                    else:
                        prob_out = np.power(self.p, connections_out[idx_pre])

                    # Test for formation
                    formation = self.rng.binomial(1, prob_out)
                    if not formation:
                        # Remove neuron pre from list of unspent neurons
                        neurons_unspent = neurons_unspent[neurons_unspent != idx_pre]
                        neurons_looking = np.delete(neurons_looking, 0)
                    else:
                        # Exclude connections to self and to neurons already connected
                        possible_post = neurons_incoming[neurons_incoming != idx_pre]
                        neurons_connected = np.where(weight_matrix[:, idx_pre] == 1)[0]
                        possible_post = np.setdiff1d(possible_post, neurons_connected)

                        formed = 0
                        while not formed:
                            post_idx = self.rng.choice(possible_post)
                            prob_in = np.power(self.q, connections_in[post_idx])
                            accept = self.rng.binomial(1, prob_in)
                            if accept:
                                # Add connection to matrix
                                weight_matrix[post_idx, idx_pre] = 1
                                connections_out[idx_pre] += 1
                                connections_in[post_idx] += 1
                                if np.isin(post_idx, neurons_unspent):
                                    neurons_unspent = neurons_unspent[
                                        neurons_unspent != post_idx
                                    ]
                                    neurons_looking = np.append(
                                        neurons_looking, post_idx
                                    )
                                formed = 1
                            else:
                                continue

        if np.sum(weight_matrix) != np.sum(connections_in) - num_vis:
            print("Problem with total connections")

        return weight_matrix


class Neurons:
    def __init__(
        self, neuron_params: NeuronConfig, num_neurons: int, rate_buffer: Buffer
    ):
        # Parameters
        self.C_v = neuron_params.C_v
        self.C_u = neuron_params.C_u
        self.E_l = neuron_params.E_l
        self.E_exc = neuron_params.E_exc
        self.E_inh = neuron_params.E_inh
        self.g_l = neuron_params.g_l
        self.g_den = neuron_params.g_den
        self.g_exc = neuron_params.g_exc
        self.g_inh = neuron_params.g_inh
        self.a = neuron_params.a
        self.b = neuron_params.b
        self.d_den = neuron_params.d_den
        self.d_som = neuron_params.d_som
        self.d_int = neuron_params.d_int
        self.lam = neuron_params.lam

        # Dynamical variables
        self.v = np.ones(num_neurons) * self.E_l
        self.u = np.ones(num_neurons) * self.E_l
        self.r_bar = np.ones(num_neurons) * self.phi(self.E_l)
        self.r = np.ones(num_neurons) * self.phi(self.E_l)
        self.I_den = np.zeros(num_neurons)
        self.I_som = np.zeros(num_neurons)
        self.rate_buffer = rate_buffer(num_neurons)

    def phi(self, v):
        return 0.5 * (1 + np.tanh(v / 2))


class Network:
    def __init__(
        self,
        network_params: NetworkConfig,
        weight_params: WeightConfig,
        neuron_params: NeuronConfig,
        dendritic_weights: DendriticWeights,
        somatic_weights: SomaticWeights,
        neurons: Neurons,
        rate_buffer: Buffer,
    ):
        self.num_lat = network_params.num_lat
        self.num_vis = network_params.num_vis
        self.num_all = self.num_lat + self.num_vis
        self.dendritic_weights = dendritic_weights(
            weight_params, self.num_lat, self.num_vis
        )
        self.somatic_weights = somatic_weights(
            weight_params, self.num_vis, self.num_lat
        )
        self.neurons = neurons(neuron_params, self.num_all, rate_buffer)

    def simulation_step(self, dt):
        # Implement simulation step logic here
        pass

    def update(self):
        # Implement update logic here
        pass

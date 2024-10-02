#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .config import NetworkConfig, NeuronConfig, WeightConfig
from .rate_buffer import Buffer


class Weights(ABC):
    """
    Base class for weight matrices for elise networks.

    :param weight_config: Configuration object containing weight parameters.
    :type weight_config: WeightConfig
    """

    """
    Base class for weight matrices in neural networks.
    """

    def __init__(self, weight_config: WeightConfig):
        pass

    @abstractmethod
    def __call__(self, num_vis: int, num_lat: int):
        pass

    @abstractmethod
    def _create_weight_matrix(self, num_vis: int, num_lat: int):
        pass

    @abstractmethod
    def _create_delay_matrix(
        self, num_vis: int, num_lat: int, weight_matrix: npt.NDArray
    ):
        pass


class DendriticWeights(Weights):
    """
    Class for creating dendritic weight matrices.

    This class implements the Weights interface to generate weight and delay
    matrices for dendritic connections in a neural network.

    :param weight_config: Configuration object containing weight parameters.
    :type weight_config: WeightConfig

    :ivar W_vis_vis: Weight range for visible-to-visible connections.
    :ivar W_vis_lat: Weight range for visible-to-lateral connections.
    :ivar W_lat_vis: Weight range for lateral-to-visible connections.
    :ivar W_lat_lat: Weight range for lateral-to-lateral connections.
    :ivar d_den: Range for dendritic delay values.
    :ivar rng: Random number generator.
    """

    def __init__(self, weight_config: WeightConfig):
        super().__init__(weight_config)
        self.W_vis_vis = weight_config.W_vis_vis
        self.W_vis_lat = weight_config.W_vis_lat
        self.W_lat_vis = weight_config.W_lat_vis
        self.W_lat_lat = weight_config.W_lat_lat
        self.d_den = weight_config.d_den
        self.rng = np.random.default_rng(seed=weight_config.den_seed)

    def __call__(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Generate weight and delay matrices for dendritic connections.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the weight matrix and delay matrix.
        :rtype: Tuple[npt.NDArray]
        """
        weight_matrix = self._create_weight_matrix(num_vis, num_lat)
        delay_matrix = self._create_delay_matrix(num_vis, num_lat, weight_matrix)
        return weight_matrix, delay_matrix

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create a dendritic weight matrix.

        This method generates a weight matrix for dendritic
        connections using the configured weight ranges.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The generated weight matrix.
        :rtype: npt.NDArray
        """
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

        return weights

    def _create_delay_matrix(
        self, num_vis: int, num_lat: int, weight_matrix: npt.NDArray
    ) -> Tuple[npt.NDArray]:
        """
        Create a delay matrix based on the weight matrix.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :param weight_matrix: The weight matrix.
        :type weight_matrix: npt.NDArray
        :return: The delay matrix.
        :rtype: npt.NDArray
        """

        mask = weight_matrix == 0
        delay_matrix = np.random.randint(
            self.d_den[0], self.d_den[1], weight_matrix.shape
        )
        delay_matrix[mask] = 0

        return delay_matrix


class SomaticWeights(Weights):
    """
    Class for creating somatic weight matrices.

    This class extends the Weights class and provides methods to generate
    weight and delay matrices for somatic connections in a neural network.

    :ivar p: Probability parameter for outgoing connections.
    :ivar q: Probability parameter for incoming connections.
    :ivar p0: Initial probability for the first connection.
    :ivar p_first: Probability for the first connection (1 - p0).
    :ivar rng: Random number generator.
    :ivar d_som: Range for somatic delay values.
    """

    def __init__(self, weight_config: WeightConfig):
        super().__init__(weight_config)
        self.p = weight_config.p
        self.q = weight_config.q
        self.p0 = weight_config.p0
        self.p_first = 1 - self.p0
        self.rng = np.random.default_rng(seed=weight_config.som_seed)
        self.d_som = weight_config.d_som

    def __call__(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Generate weight and delay matrices for somatic connections.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the weight matrix and delay matrix.
        :rtype: Tuple[npt.NDArray]
        """
        weight_matrix = self._create_weight_matrix(num_vis, num_lat)
        delay_matrix = self._create_delay_matrix(num_vis, num_lat, weight_matrix)
        return weight_matrix, delay_matrix

    def _create_delay_matrix(
        self, num_vis: int, num_lat: int, weight_matrix: npt.NDArray
    ) -> Tuple[npt.NDArray]:
        """
        Create a delay matrix based on the weight matrix.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :param weight_matrix: The weight matrix.
        :type weight_matrix: npt.NDArray
        :return: The delay matrix.
        :rtype: npt.NDArray
        """
        mask = weight_matrix == 0
        delay_matrix = np.random.randint(
            self.d_som[0], self.d_som[1], weight_matrix.shape
        )
        delay_matrix[mask] = 0
        return delay_matrix

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create a somatic weight matrix based on probabilistic connection rules.

        This method implements a complex algorithm to generate connections
        between neurons based on probabilistic rules.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The generated weight matrix.
        :rtype: npt.NDArray

        :note: The method uses instance attributes p, q, p0, and rng
        for its calculations.
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
        self.dendritic_weights, self.dendritic_delays = dendritic_weights(
            weight_params, self.num_lat, self.num_vis
        )
        self.somatic_weights, self.somatic_delays = somatic_weights(
            weight_params, self.num_vis, self.num_lat
        )
        self.neurons = neurons(neuron_params, self.num_all, rate_buffer)

    def simulation_step(self, dt):
        # Implement simulation step logic here
        pass

    def update(self):
        # Implement update logic here
        pass

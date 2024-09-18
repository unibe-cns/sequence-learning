#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt
from buffer import Buffer
from config import NetworkConfig, NeuronConfig, WeightConfig


class Weights(ABC):
    """
    Base class for weight matrices in neural networks.
    """

    def __init__(self, weight_config: WeightConfig, num_vis, num_lat):
        self.connections_in = None
        self.connections_out = None
        self.weights = None
        self.num_vis = num_vis
        self.num_lat = num_lat

    @abstractmethod
    def create_weight_matrix(self, num_vis, num_lat):
        pass


class DendriticWeights(Weights):
    """
    Class for creating dendritic weight matrices.
    """

    def __init__(self, weight_config: WeightConfig):
        super().__init__(weight_config)
        self.W_out_out = weight_config.W_out_out
        self.W_out_lat = weight_config.W_out_lat
        self.W_lat_out = weight_config.W_lat_out
        self.W_lat_lat = weight_config.W_lat_lat
        self.weights = self.create_weight_matrix(self.num_vis, self.num_lat)

    def create_weight_matrix(self, num_vis, num_lat):
        # Implement the dendritic weight matrix creation logic here
        # Using self.W_out_out, self.W_out_lat, self.W_lat_out, self.W_lat_lat
        pass


class SomaticWeights(Weights):
    """
    Class for creating somatic weight matrices.
    """

    def __init__(self, weight_config: WeightConfig):
        super().__init__(weight_config)
        self.p = weight_config.p
        self.q = weight_config.q
        self.p0 = weight_config.p0
        self.p_first = 1 - self.p0
        self.weights = self.create_weight_matrix(self.num_vis, self.num_lat)

    def create_weight_matrix(
        self, num_vis, num_lat
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Create a somatic weight matrix based on probabilistic connection rules.
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
                    formation = np.random.binomial(1, prob_out)
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
                            post_idx = np.random.choice(possible_post)
                            prob_in = np.power(self.q, connections_in[post_idx])
                            accept = np.random.binomial(1, prob_in)
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

        self.weights = weight_matrix
        self.connections_in = connections_in
        self.connections_out = connections_out


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
        self.g_lat = neuron_params.g_lat
        self.g_den = neuron_params.g_den
        self.g_exc = neuron_params.g_exc
        self.g_inh = neuron_params.g_inh
        self.a = neuron_params.a
        self.b = neuron_params.b
        self.d_den = neuron_params.d_den
        self.d_som = neuron_params.d_som
        self.d_int = neuron_params.d_int

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
        dendritic_weights: DendriticWeights,
        somatic_weights: SomaticWeights,
        neurons: Neurons,
        rate_buffer: Buffer,
    ):
        self.num_lat = network_params.num_lat
        self.num_vis = network_params.num_vis
        self.num_all = self.num_lat + self.num_vis

        self.dendritic_weights = dendritic_weights(self.num_lat, self.num_vis)
        self.somatic_weights = somatic_weights(self.num_lat, self.num_vis)
        self.neurons = neurons(self.num_all, rate_buffer)

    def simulation_step(self, dt):
        # Implement simulation step logic here
        pass

    def update(self):
        # Implement update logic here
        pass

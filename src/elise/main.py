#!/usr/bin/env python3
"""Main script for creating weight matrix."""

# TODO add Dendritic weights
# TODO add delays to Dentritic weights class
# TODO add delays to Somatic weights class
# TODO add network class
# - Add one that follows dependency inversion principle
# - Add one that does not use dependency inversion principle

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import toml


class Config:
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self._config = toml.load(f)

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})


class WeightConfig:
    def __init__(self, config: Dict[str, Any]):
        self.num_lat: int = config.get("num_lat", 50)
        self.num_vis: int = config.get("num_vis", 13)
        self.p: float = config.get("p", 0.5)
        self.q: float = config.get("q", 0.3)
        self.p0: float = config.get("p0", 0.1)
        self.sparsity: float = config.get("sparsity", 0.3)
        self.W_out_out: list = config.get("W_out_out", [0.0, 0.5])
        self.W_out_lat: list = config.get("W_out_lat", [0.0, 0.5])
        self.W_lat_lat: list = config.get("W_lat_lat", [0.0, 0.5])
        self.W_lat_out: list = config.get("W_lat_out", [0.0, 0.5])
        self.d_som_min: int = config.get("d_som_min", 5)
        self.d_som_max: int = config.get("d_som_max", 15)
        self.d_den_min: int = config.get("d_den_min", 5)
        self.d_den_max: int = config.get("d_den_max", 15)


class SimulationConfig:
    def __init__(self, config: Dict[str, Any]):
        self.input: str = config.get("input", "test")
        self.dt: float = config.get("dt", 0.01)
        self.training_cycles: float = config.get("training_cycles", 10e4)
        self.replay_cycles: int = config.get("replay_cycles", 100)
        self.validation_interval: int = config.get("validation_interval", 20)
        self.eta_out: float = config.get("eta_out", 10e-4)
        self.eta_lat: float = config.get("eta_lat", 10e-3)


class NeuronConfig:
    def __init__(self, config: Dict[str, Any]):
        self.C_v: float = config.get("C_v", 1.0)
        self.C_u: float = config.get("C_u", 1.0)
        self.E_l: float = config.get("E_l", -70.0)
        self.E_exc: float = config.get("E_e", 0.0)
        self.E_inh: float = config.get("E_i", -75.0)
        self.g_lat: float = config.get("g_l", 0.1)
        self.g_den: float = config.get("g_d", 2.0)
        self.g_exc: float = config.get("g_e", 0.3)
        self.g_inh: float = config.get("g_i", 6.0)
        self.a: float = config.get("a", 0.3)
        self.b: float = config.get("b", -58.0)
        self.d_den: list = config.get("d_d", [5, 15])
        self.d_som: list = config.get("d_s", [5, 15])
        self.d_int: int = config.get("d_t", 25)


class FullConfig:
    _instance = None

    def __new__(cls, config_file: str = "config.toml"):
        if cls._instance is None:
            cls._instance = super(FullConfig, cls).__new__(cls)
            cls._instance._initialize(config_file)
        return cls._instance

    def _initialize(self, config_file: str):
        config = Config(config_file)
        self.seed: int = config.get_section("").get("seed", 69)
        self.weight_params = WeightConfig(config.get_section("weight_params"))
        self.simulation_params = SimulationConfig(
            config.get_section("simulation_params")
        )
        self.neuron_params = NeuronConfig(config.get_section("neuron_params"))


class Weights(ABC):
    """
    Base class for weight matrices in neural networks.
    """

    def __init__(self, weight_config: WeightConfig):
        self.num_latent = weight_config.num_lat
        self.num_output = weight_config.num_vis
        self.num_total = self.num_latent + self.num_output
        self.weights = None
        self.num_in = None
        self.num_out = None

    @abstractmethod
    def create_weight_matrix(self):
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
        self.weights = self.create_weight_matrix()

    def create_weight_matrix(self):
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
        self.weights, self.num_in, self.num_out = self.create_weight_matrix()

    def create_weight_matrix(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Create a somatic weight matrix based on probabilistic connection rules.

        This method generates a weight matrix for somatic connections.
        It uses a probabilistic approach to form connections, considering
        both outgoing and incoming connection probabilities.

        The algorithm iterates through neurons, forming connections based on the
        following rules:
        - The probability of forming an outgoing connection decreases with each
        connection made (controlled by self.p and self.p0).
        - The probability of accepting an incoming connection decreases with each
        connection received (controlled by self.q).
        - Connections are only formed from output to latent neurons and between
        latent neurons.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
            A tuple containing three NumPy arrays:
            - weight_matrix: 2D array representing the connectivity matrix
            - num_in: 1D array of incoming connection counts for each neuron
            - num_out: 1D array of outgoing connection counts for each neuron

        Notes
        -----
        - The method uses class attributes self.num_total, self.num_output, self.p,
        self.q, and self.p0 in its calculations.
        - If the total number of connections doesn't match the expected value,
        the method will print an error message and invoke the debugger.
        """
        neurons_outgoing = np.arange(self.num_total)
        neurons_incoming = np.arange(self.num_output, self.num_total)
        weight_matrix = np.zeros((self.num_total, self.num_total))

        # Incoming connections
        num_in = np.zeros(self.num_total)
        num_in[: self.num_output] = 1

        # Outgoing connections
        num_out = np.zeros(self.num_total)

        # Neurons that can make a connection
        neurons_unspent = np.arange(self.num_total)
        neurons_looking = neurons_outgoing[num_in > 0]

        while len(neurons_looking) > 0:
            for idx_pre in neurons_looking:
                while np.isin(idx_pre, neurons_looking):
                    # Probability for forming connection
                    if num_out[idx_pre] == 0:
                        prob_out = 1 - self.p0
                    else:
                        prob_out = np.power(self.p, num_out[idx_pre])

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
                            prob_in = np.power(self.q, num_in[post_idx])
                            accept = np.random.binomial(1, prob_in)
                            if accept:
                                # Add connection to matrix
                                weight_matrix[post_idx, idx_pre] = 1
                                num_out[idx_pre] += 1
                                num_in[post_idx] += 1
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

        if np.sum(weight_matrix) != np.sum(num_in) - self.num_output:
            print("Problem with total connections")
            breakpoint()

        return weight_matrix, num_in, num_out


# Main
if __name__ == "__main__":
    # Load config file
    # Usage
    config1 = FullConfig("config.toml")
    config2 = FullConfig("config.toml")  # This will return the same instance as config1

    # Usage
    full_config = FullConfig(
        "config.toml"
    )  # This creates or gets the singleton instance
    dendritic_weights = DendriticWeights(full_config.weight_params)
    somatic_weights = SomaticWeights(full_config.weight_params)

    print(config1 is config2)  # This will print True

    print(config1.seed)
    print(config1.weight_params.num_lat)
    print(config1.simulation_params.dt)
    print(config1.neuron_params.E_l)

    # Out
    weights = somatic_weights.weights
    num_in = somatic_weights.num_in
    num_out = somatic_weights.num_out

    # Set seed
    np.random.seed(full_config.seed)

    breakpoint()
    # Plot weight matrix
    fig, ax = plt.subplots()
    ax.imshow(weights)
    plt.show()

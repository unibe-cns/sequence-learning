#!/usr/bin/env python3
"""Main script for creating weight matrix."""
from config import FullConfig
from dataloader import Dataloader
from logger import Logger
from model import DendriticWeights, Network, Neurons, SomaticWeights
from rate_buffer import RollBuffer


# IMplement simulation class with dependency inversion principle
class Simulation:
    def __init__(self, simulation_params, network, dataloader, logger):
        self.network = network
        self.logger = logger
        self.dataloader = dataloader

    def run(self):
        pass


if __name__ == "__main__":
    # Load config file

    # Usage
    full_config = FullConfig(
        "config.toml"
    )  # This creates or gets the singleton instance

    weight_params = full_config.weight_params
    network_params = full_config.network_params
    neuron_params = full_config.neuron_params

    # Create weight matrices
    dendritic_weights = DendriticWeights
    somatic_weights = SomaticWeights
    rate_buffer = RollBuffer
    neurons = Neurons

    # Create network
    network = Network(
        network_params,
        weight_params,
        neuron_params,
        dendritic_weights,
        somatic_weights,
        neurons,
        rate_buffer,
    )

    # Create dataloader
    dataloader = Dataloader

    # Create logger
    logger = Logger

    # Simulation
    simulation = Simulation(full_config.simulation_params, network, dataloader, logger)

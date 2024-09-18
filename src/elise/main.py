#!/usr/bin/env python3
"""Main script for creating weight matrix."""
from config import FullConfig
from dataloader import Dataloader
from logger import Logger
from model import DendriticWeights, Network, Neurons, SomaticWeights
from rate_buffer import Buffer


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

    # Create weight matrices
    dendritic_weights = DendriticWeights
    somatic_weights = SomaticWeights

    # Neurons
    neurons = Neurons

    # Buffer
    rate_buffer = Buffer

    # Create network
    network = Network(
        full_config.network_params,
        dendritic_weights,
        somatic_weights,
        neurons,
        rate_buffer,
    )

    # Create dataloader
    dataloader = Dataloader(full_config.data_params)

    # Create logger
    logger = Logger()

    # Simulation
    simulation = Simulation(full_config.simulation_params, network, dataloader, logger)

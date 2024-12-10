#!/usr/bin/env python3

from elise.config import SimulationConfig
from elise.logger import Logger
from elise.model import Network
from elise.optimizer import Optimizer


class Simulation:
    def __init__(
        self,
        simulation_params: SimulationConfig,
        Optimizer: Optimizer,
        network: Network,
        logger: Logger,
    ):
        # Params
        self.dt = simulation_params.dt
        self.training_cycles = simulation_params.training_cycles
        self.replay_cycles = simulation_params.replay_cycles
        self.validation_interval = simulation_params.validation_interval

        self.network = network
        self.logger = logger

        # Optimizer
        self.eta_lat = simulation_params.eta_lat / self.dt
        self.eta_vis = simulation_params.eta_out / self.dt
        self.optimizer_lat = Optimizer(self.eta_lat)
        self.optimizer_vis = Optimizer(self.eta_vis)

        self.network.prepare_for_simulation(
            self.dt, optimizer_vis=self.optimizer_vis, optimizer_lat=self.optimizer_lat
        )

#!/usr/bin/env python3

from elise.config import SimulationConfig
from elise.data import DataLoader
from elise.logger import Logger
from elise.model import Network
from elise.optimizer import Optimizer


class Simulation:
    def __init__(
        self,
        simulation_params: SimulationConfig,
        optimizer: Optimizer,
        network: Network,
        logger: Logger,
        dataloader: DataLoader,
    ):
        # Params
        self.dt = simulation_params.dt
        self.training_cycles = simulation_params.training_cycles
        self.replay_cycles = simulation_params.replay_cycles
        self.validation_interval = simulation_params.validation_interval

        self.network = network
        self.logger = logger
        self.dataloader = dataloader

        # Optimizer
        self.eta_lat = simulation_params.eta_lat / self.dt
        self.eta_vis = simulation_params.eta_out / self.dt
        self.optimizer_lat = optimizer(self.eta_lat)
        self.optimizer_out = optimizer(self.eta_vis)

        self.network.prepare_for_simulation(
            self.dt, self.optimizer_vis, self.optimizer_lat
        )

    def epoch(self, teacher=None):
        for u_tgt in self.dataloader:
            if teacher:
                self.network(u_inp=u_tgt)
            elif not teacher:
                self.network(u_inp=None)
            else:
                raise ValueError("Teacher must be a boolean value")

            # Intra-epoch logging

    def training(self):
        for i in range(self.training_cycles):
            self.epoch(teacher=True)

            if i % self.validation_interval == 0:
                self.epoch(teacher=False)

                # Validation logging

    def replay(self):
        for i in range(self.replay_cycles):
            self.epoch(teacher=False)

            # Replay logging

    def run(self):
        self.training()
        self.replay()

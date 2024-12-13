#!/usr/bin/env python3

from tqdm import tqdm

from elise.config import SimulationConfig
from elise.data import Dataloader
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
        dataloader: Dataloader,
    ):
        # Params
        self.dt = simulation_params.dt
        self.training_cycles = simulation_params.training_cycles
        self.replay_cycles = simulation_params.replay_cycles
        self.validation_cycles = simulation_params.validation_cycles
        self.validation_interval = simulation_params.validation_interval

        self.network = network
        self.logger = logger
        self.dataloader = dataloader

        # Optimizer
        self.eta_lat = simulation_params.eta_lat / self.dt
        self.eta_vis = simulation_params.eta_out / self.dt
        self.optimizer_lat = optimizer(self.eta_lat)
        self.optimizer_vis = optimizer(self.eta_vis)

        self.network.prepare_for_simulation(
            self.dt, self.optimizer_vis, self.optimizer_lat
        )

    def epoch(self, teacher=None):
        start = 0
        stop = self.dataloader.duration
        for _, u_tgt in self.dataloader.iter(start, stop, self.dt):
            if teacher:
                self.network(u_inp=u_tgt)
            elif not teacher:
                self.network(u_inp=None)
            else:
                raise ValueError("Teacher must be a boolean value")

            # Intra-epoch logging

    def training(self):
        print("Training")
        for tc in tqdm(range(self.training_cycles)):
            self.epoch(teacher=True)

            if tc % self.validation_interval == 0:
                for v in range(self.validation_cycles):
                    self.epoch(teacher=False)

                    # Validation logging

    def replay(self):
        print("Replay")
        for rc in tqdm(range(self.replay_cycles)):
            self.epoch(teacher=False)

            # Replay logging

    def run(self):
        self.training()
        self.replay()

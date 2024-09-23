#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, Dict

import toml


class Config:
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self._config = toml.load(f)

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})


@dataclass
class NetworkConfig:
    num_lat: int = 50
    num_vis: int = 13


@dataclass
class WeightConfig:
    p: float = 0.5
    q: float = 0.3
    p0: float = 0.1
    sparsity: float = 0.3
    W_out_out: list = [0.0, 0.5]
    W_out_lat: list = [0.0, 0.5]
    W_lat_lat: list = [0.0, 0.5]
    W_lat_out: list = [0.0, 0.5]
    d_som_min: int = 5
    d_som_max: int = 15
    d_den_min: int = 5
    d_den_max: int = 15


@dataclass
class SimulationConfig:
    input: str = "test"
    dt: float = 0.01
    training_cycles: float = 10e4
    replay_cycles: int = 100
    validation_interval: int = 20
    eta_out: float = 10e-4
    eta_lat: float = 10e-3


@dataclass
class NeuronConfig:
    C_v: float = 1.0
    C_u: float = 1.0
    E_l: float = -70.0
    E_exc: float = 0.0
    E_inh: float = -75.0
    g_lat: float = 0.1
    g_den: float = 2.0
    g_exc: float = 0.3
    g_inh: float = 6.0
    a: float = 0.3
    b: float = -58.0
    d_den: list = [5, 15]
    d_som: list = [5, 15]
    d_int: int = 25


class FullConfig:
    def __init__(self, config_file: str):
        config = Config(config_file)
        self.seed: int = config.get_section("").get("seed", 69)
        self.network_params = NetworkConfig(config.get_section("network_params"))
        self.weight_params = WeightConfig(config.get_section("weight_params"))
        self.simulation_params = SimulationConfig(
            config.get_section("simulation_params")
        )
        self.neuron_params = NeuronConfig(config.get_section("neuron_params"))

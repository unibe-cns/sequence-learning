#!/usr/bin/env python3
from typing import Any, Dict

import toml


class Config:
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self._config = toml.load(f)

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})


class NetworkConfig:
    def __init__(self, config: Dict[str, Any]):
        self.num_lat: int = config.get("num_lat", 50)
        self.num_vis: int = config.get("num_vis", 13)


class WeightConfig:
    def __init__(self, config: Dict[str, Any]):
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
    def __init__(self, config_file: str):
        config = Config(config_file)
        self.seed: int = config.get_section("").get("seed", 69)
        self.network_params = NetworkConfig(config.get_section("network_params"))
        self.weight_params = WeightConfig(config.get_section("weight_params"))
        self.simulation_params = SimulationConfig(
            config.get_section("simulation_params")
        )
        self.neuron_params = NeuronConfig(config.get_section("neuron_params"))

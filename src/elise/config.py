#!
# /usr/bin/env python3
import tomllib as toml
from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple


@dataclass
class NetworkConfig:
    num_lat: int = 50
    num_vis: int = 13


@dataclass
class WeightConfig:
    som_seed: int = 42
    den_seed: int = 42
    p: float = 0.5
    q: float = 0.3
    p0: float = 0.1
    den_spars: float = 0.3
    W_vis_vis: tuple = (0.0, 0.5)
    W_vis_lat: tuple = (0.0, 0.5)
    W_lat_lat: tuple = (0.0, 0.5)
    W_lat_vis: tuple = (0.0, 0.5)
    d_den: Tuple[int, int] = (5, 15)
    d_som: Tuple[int, int] = (5, 15)


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
    g_l: float = 0.1
    g_den: float = 2.0
    g_exc: float = 0.3
    g_inh: float = 6.0
    a: float = 0.3
    b: float = -58.0
    d_int: int = 25
    lam: float = 0.6


class Config:
    """
    A class for loading and accessing configuration data from a TOML file.

    Attributes:
        _config (dict): The loaded configuration data.

    Methods:
        get_section(section: str) -> Dict[str, Any]:
    Retrieves a specific section from the configuration.
    """

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            self._config = toml.load(f)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Retrieves a specific section from the configuration.

        Args: section (str): The name of the section to retrieve.

        Returns:
            Dict[str, Any]:
            A dictionary containing the configuration data for the specified section.
        """
        return self._config.get(section, {})


class FullConfig:
    def __init__(self, config_file: str):
        config = Config(config_file)
        self.seed: int = config.get_section("").get("seed", 69)
        self.network_params = self._create_config(
            NetworkConfig, config.get_section("network_params")
        )
        self.weight_params = self._create_config(
            WeightConfig, config.get_section("weight_params")
        )
        self.simulation_params = self._create_config(
            SimulationConfig, config.get_section("simulation_params")
        )
        self.neuron_params = self._create_config(
            NeuronConfig, config.get_section("neuron_params")
        )

    def _create_config(self, config_class: Config, config_dict: Dict[str, Any]):
        kwargs = {}
        for field in fields(config_class):
            if field.name in config_dict:
                kwargs[field.name] = config_dict[field.name]
        return config_class(**kwargs)

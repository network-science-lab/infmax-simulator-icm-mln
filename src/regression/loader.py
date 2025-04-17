"""Loader for regression methods."""

from dataclasses import dataclass
from itertools import product
from typing import Any

from tqdm import tqdm

from src.regression.regr_methods import CachedCentralityRegressor, NeptuneRegressor
from src.sim_utils import Network, parse_network_config


@dataclass
class RegrResult:
    net_type: str
    net_name: str
    protocol: str
    p: float
    regr_name: str
    rmse_avg: float
    rmse_std: float
    r2_avg: float
    r2_std: float


def get_parameter_space(protocols: list[str], p_values: list[float], networks: list[str]) -> product:
    nets = []
    for net_type in networks:
        net_type, net_names = parse_network_config(net_type)
        print(f"Loading {net_type} networks")
        for net_name in tqdm(net_names):
            nets.append(Network(n_type=net_type, n_name=net_name, n_graph_pt=None, n_graph_nx=None))
    return product(protocols, p_values, nets)


def load_regr_models(config_regr: dict[str, Any], nb_repetitions: int, rng_seed: int) -> dict[str, Any]:
    return {
        m_config["name"]: load_regr_model(
            config_regr=m_config,
            rng_seed=rng_seed,
            nb_repetitions=nb_repetitions,
        )
        for m_config in config_regr
    }


def load_regr_model(config_regr: dict[str, Any], nb_repetitions: int, rng_seed: int) -> Any:
    if config_regr["class"] == "CachedCentralityRegressor":
        if config_regr["parameters"]["rng_seed"] == "auto":
            config_regr["parameters"]["rng_seed"] = rng_seed
        if config_regr["parameters"]["nb_repetitions"] == "auto":
            config_regr["parameters"]["nb_repetitions"] = nb_repetitions
        return CachedCentralityRegressor(**config_regr["parameters"])
    elif config_regr["class"] == "NeptuneRegressor":
        return NeptuneRegressor(**config_regr["parameters"])
    raise ValueError(f"Unknown regression model class: {config_regr['class']}!")

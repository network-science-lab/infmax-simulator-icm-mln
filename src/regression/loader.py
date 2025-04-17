"""Loader for regression methods."""

from dataclasses import dataclass
from typing import Any

from src.regression.regr_methods import CachedCentralityRegressor


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
    raise ValueError(f"Unknown regression model class: {config_regr['class']}!")

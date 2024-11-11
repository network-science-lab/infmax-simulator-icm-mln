"""Loader for influence maximisations methods."""

from dataclasses import dataclass
from typing import Any, Callable

import network_diffusion as nd


@dataclass
class SeedSet:
    """Aux. class to store a seedset and its metadata."""
    method_name: str
    repetition_nb: int
    seeds: set[str]


def load_infmax_models(config: dict[str, Any], random_seed: int, nb_seeds: int) -> dict[str, Callable]:
    return {m_config["name"]: load_infmax_model(m_config, random_seed, nb_seeds) for m_config in config}


def load_infmax_model(config: dict[str, Any], random_seed: int, nb_seeds: int) -> Any:
    if config["name"] in {"MultiNode2VecKMeans", "MultiNode2VecKMeansAuto"}:
        from multi_node2vec_kmeans.loader import load_model
        if config["parameters"]["rng_seed"] == "auto":
            config["parameters"]["rng_seed"] = random_seed
        if config["parameters"]["k_means"]["nb_seeds"] == "auto":
            config["parameters"]["k_means"]["nb_seeds"] = nb_seeds
        return load_model({"model": config})
    else:
        return lambda x: "dupa"  # TODO: add here GT loader


def if_stochastic(infmax_model: Callable) -> bool:
    match infmax_model.__class__.__name__:
        case "MultiNode2VecKMeans":
            return True
        case "MultiNode2VecKMeansAuto":
            return True
    return False


def get_seed_sets(
    infmax_models: dict[str, Callable],
    net: nd.MultilayerNetworkTorch,
    repetitions_diffusion: int,
) -> list[SeedSet]:
    """Obtain seed sets for a given infmax model on a given network and if needed repeat it."""
    seed_sets = []
    for ifm_name, ifm_obj in infmax_models.items():
        repetitions_infmax=repetitions_diffusion if if_stochastic(ifm_obj) else 1
        partial_seed_sets = [
            SeedSet(method_name=ifm_name, repetition_nb=i, seeds=ifm_obj(net))
            for i in range(repetitions_infmax)
        ]
        # with this bypass we don't distinguish between repetitions, but it works and IMO it's OK
        seed_sets.extend(partial_seed_sets)
    return seed_sets

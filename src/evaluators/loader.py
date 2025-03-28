"""Loader for influence maximisations methods."""

from dataclasses import dataclass
from typing import Any

from src.evaluators.infmax_methods import (
    BaseChoice,
    CachedCentralityChoice,
    GroundTruth,
    RandomChoice,
    NeptuneDownloader,
    NeighbourhoodSizeDiscount,
    DegreeCentralityDiscount,
)
from src.sim_utils import Network


@dataclass
class SeedSet:
    """Aux. class to store a seedset and its metadata."""
    method_name: str
    repetition_nb: int
    seeds: set[str]


def load_infmax_models(
    config_infmax: dict[str, Any],
    config_sp: dict[str, int],
    rng_seed: int,
    device: str,
) -> dict[str, BaseChoice]:
    return {
        m_config["name"]: load_infmax_model(
            config_infmax=m_config,
            config_sp=config_sp,
            rng_seed=rng_seed,
            device=device,
        )
        for m_config in config_infmax
    }


def load_infmax_model(
    config_infmax: dict[str, Any],
    config_sp: dict[str, int],
    rng_seed: int,
    device: str,
) -> Any:
    if config_infmax["class"] in {"MultiNode2VecKMeans", "MultiNode2VecKMeansAuto"}:
        from multi_node2vec_kmeans.loader import load_model
        if config_infmax["parameters"]["rng_seed"] == "auto":
            config_infmax["parameters"]["rng_seed"] = rng_seed
        config_infmax["name"] = config_infmax["class"]
        model = load_model({"model": config_infmax})
        model.is_stochastic = False
        return model
    elif config_infmax["class"] == "GBIM":
        from gbim_nsl_adaptation.loader import load_model
        if config_infmax["parameters"]["rng_seed"] == "auto":
            config_infmax["parameters"]["rng_seed"] = rng_seed
        if config_infmax["parameters"]["device"] == "auto":
            config_infmax["parameters"]["device"] = device
        config_infmax["name"] = config_infmax["class"]
        model = load_model({"model": config_infmax})
        model.is_stochastic = False
        return model
    elif config_infmax["class"] == "DeepIM":
        from deepim_nsl_adaptation.loader import load_model
        if config_infmax["parameters"]["rng_seed"] == "auto":
            config_infmax["parameters"]["rng_seed"] = rng_seed
        if config_infmax["parameters"]["device"] == "auto":
            config_infmax["parameters"]["device"] = device
        config_infmax["name"] = config_infmax["class"]
        model = load_model({"model": config_infmax})
        model.is_stochastic = False
        return model
    elif config_infmax["class"] == "CachedCentralityChoice":
        return CachedCentralityChoice(centrality_name=config_infmax["parameters"]["centrality"])
    elif config_infmax["class"] == "GroundTruth":
        return GroundTruth(**config_sp)
    elif config_infmax["class"] == "RandomChoice":
        return RandomChoice()
    elif config_infmax["class"] == "NeptuneDownloader":
        return NeptuneDownloader(**config_infmax["parameters"], **config_sp)
    elif config_infmax["class"] == "NeighbourhoodSizeDiscount":
        return NeighbourhoodSizeDiscount()
    elif config_infmax["class"] == "DegreeCentralityDiscount":
        return DegreeCentralityDiscount()
    raise ValueError(f"Unknown infmax model class: {config_infmax['class']}!")


def get_seed_sets(
    infmax_models: dict[str, BaseChoice],
    net: Network,
    repetitions_diffusion: int,
    protocol: str,
    p: float,
    nb_seeds: int,
) -> list[SeedSet]:
    """Obtain seed sets for a given infmax model on a given network and if needed repeat it."""
    seed_sets = []
    for ifm_name, ifm_obj in infmax_models.items():
        repetitions_infmax = repetitions_diffusion if ifm_obj.is_stochastic else 1
        partial_seed_sets = [
            SeedSet(
                method_name=ifm_name,
                repetition_nb=i,
                seeds=ifm_obj(
                    network_pt=net.n_graph_pt,
                    network_nx=net.n_graph_nx,
                    net_name=net.n_name,
                    net_type=net.n_type,
                    protocol=protocol,
                    p=p,
                    nb_seeds=nb_seeds,
                ),
            )
            for i in range(repetitions_infmax)
        ]
        # with this bypass we don't distinguish between repetitions, but it works and IMO it's OK
        seed_sets.extend(partial_seed_sets)
    return seed_sets

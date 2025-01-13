"""Loader for influence maximisations methods."""

from dataclasses import dataclass
from typing import Any, Callable

from src.evaluators.infmax_methods import CentralityChoice, GroundTruth, RandomChoice, DFChoice
from src.sim_utils import Network


@dataclass
class SeedSet:
    """Aux. class to store a seedset and its metadata."""
    method_name: str
    repetition_nb: int
    seeds: set[str]


def load_infmax_models(
    config_infmax: dict[str, Any],
    config_icm: dict[str, Any],
    random_seed: int,
    nb_seeds: int,
    device: str,
) -> dict[str, Callable]:
    return {
        m_config["name"]: load_infmax_model(
            config_infmax=m_config,
            config_icm=config_icm,
            random_seed=random_seed,
            nb_seeds=nb_seeds,
            device=device,
        )
        for m_config in config_infmax
    }


def load_infmax_model(
    config_infmax: dict[str, Any],
    config_icm: dict[str, Any],
    random_seed: int,
    nb_seeds: int,
    device: str,
) -> Any:
    if config_infmax["class"] in {"MultiNode2VecKMeans", "MultiNode2VecKMeansAuto"}:
        from multi_node2vec_kmeans.loader import load_model
        if config_infmax["parameters"]["rng_seed"] == "auto":
            config_infmax["parameters"]["rng_seed"] = random_seed
        if config_infmax["parameters"]["k_means"]["nb_seeds"] == "auto":
            config_infmax["parameters"]["k_means"]["nb_seeds"] = nb_seeds
        config_infmax["name"] = config_infmax["class"]
        return load_model({"model": config_infmax})
    elif config_infmax["class"] == "GBIM":
        from gbim_nsl_adaptation.loader import load_model
        if config_infmax["parameters"]["rng_seed"] == "auto":
            config_infmax["parameters"]["rng_seed"] = random_seed
        if config_infmax["parameters"]["device"] == "auto":
            config_infmax["parameters"]["device"] = device
        if config_infmax["parameters"]["common"]["nb_seeds"] == "auto":
            config_infmax["parameters"]["common"]["nb_seeds"] = nb_seeds
        config_infmax["name"] = config_infmax["class"]
        return load_model({"model": config_infmax})
    elif config_infmax["class"] == "DeepIM":
        from deepim_nsl_adaptation.loader import load_model
        if config_infmax["parameters"]["rng_seed"] == "auto":
            config_infmax["parameters"]["rng_seed"] = random_seed
        if config_infmax["parameters"]["device"] == "auto":
            config_infmax["parameters"]["device"] = device
        if config_infmax["parameters"]["common"]["nb_seeds"] == "auto":
            config_infmax["parameters"]["common"]["nb_seeds"] = nb_seeds
        config_infmax["name"] = config_infmax["class"]
        return load_model({"model": config_infmax})
    elif config_infmax["class"] == "DFChoice":
        return DFChoice(result_dir=config_infmax["parameters"]["result_dir"])
    elif config_infmax["class"] == "CentralityChoice":
        return CentralityChoice(
            nb_seeds=nb_seeds,
            centrality_name=config_infmax["parameters"]["centrality"],
        )
    elif config_infmax["class"] == "GroundTruth":
        return GroundTruth(
            nb_seeds=nb_seeds,
            average_protocol=config_infmax["parameters"]["average_protocol"],
            average_p_value=config_infmax["parameters"]["average_p_value"],
        )
    elif config_infmax["class"] == "RandomChoice":
        if config_infmax["parameters"]["nb_seeds"] == "auto":
            config_infmax["parameters"]["nb_seeds"] = nb_seeds
        return RandomChoice(nb_seeds=nb_seeds)
    raise ValueError(f"Unknown infmax model class: {config_infmax['class']}!")


def if_stochastic(infmax_model: Callable) -> bool:
    stochastic_models = {"MultiNode2VecKMeans", "MultiNode2VecKMeansAuto", "RandomChoice"}
    if infmax_model.__class__.__name__ in stochastic_models:
            return True
    return False


def get_seed_sets(
    infmax_models: dict[str, Callable],
    net: Network,
    repetitions_diffusion: int,
    protocol: str,
    p: float,
) -> list[SeedSet]:
    """Obtain seed sets for a given infmax model on a given network and if needed repeat it."""
    seed_sets = []
    for ifm_name, ifm_obj in infmax_models.items():
        repetitions_infmax = repetitions_diffusion if if_stochastic(ifm_obj) else 1
        partial_seed_sets = [
            SeedSet(
                method_name=ifm_name,
                repetition_nb=i,
                seeds=ifm_obj(
                    network=net.n_graph,
                    net_name=net.n_name,
                    net_type=net.n_type,
                    protocol=protocol,
                    p=p,
                ),
            )
            for i in range(repetitions_infmax)
        ]
        # with this bypass we don't distinguish between repetitions, but it works and IMO it's OK
        seed_sets.extend(partial_seed_sets)
    return seed_sets

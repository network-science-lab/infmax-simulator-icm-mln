"""Loader for influence maximisations methods."""

from typing import Any, Callable


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

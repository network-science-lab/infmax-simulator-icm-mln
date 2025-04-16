"""Main script with regresison on centralities."""

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src import os_utils, sim_utils
from src.regression.regr_methods import CachedCentralityRegressor



def run_experiments(config: dict[str, Any]) -> None:

    # get parameter space and experiment's hyperparams
    p_space = sim_utils.get_parameter_space(
        protocols=config["spreading_model"]["parameters"]["protocols"],
        p_values=config["spreading_model"]["parameters"]["p_values"],
        networks=config["networks"],
    )
    p_space = list(p_space)
    print(p_space)


    # initialise influence maximisation models
    # infmax_models = loader.load_infmax_models(
    #     config_infmax=[
    #         *config["infmax_models"],
    #         {"name": "ground_truth", "class": GroundTruth.__name__},
    #     ],
    #     config_sp=config["spreading_potential_score"],
    #     rng_seed=config["run"]["rng_seed"],
    #     device=config["run"]["device"],
    # )

    # # get the starting time
    # start_time = os_utils.get_current_time()
    # print(f"Experiments started at {start_time}")

    # # prepare output directory and deterimne how to store results
    # out_dir = Path(config["logging"]["out_dir"]) / start_time
    # out_dir.mkdir(exist_ok=True, parents=True)

    # # save the config
    # config["git_sha"] = os_utils.get_recent_git_sha()
    # with open(out_dir / f"config.yaml", "w", encoding="utf-8") as f:
    #     yaml.dump(config, f)
    
    # for regr_method in config["features"]:
    #     print(regr_method)
    result = CachedCentralityRegressor(
        centrality_names=["degree"],
        rng_seed=config["run"]["rng_seed"],
        nb_repetitions= config["run"]["nb_repetitions"],
    )(net_type="aucs", net_name="aucs", protocol="AND", p=-1)
    print(result)

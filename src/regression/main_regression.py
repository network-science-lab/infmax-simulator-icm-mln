"""Main script with regresison on centralities."""

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
import yaml

from src import os_utils, sim_utils
from src.regression.loader import RegrResult, load_regr_models, get_parameter_space


def run_experiments(config: dict[str, Any]) -> None:

    # get parameter space and experiment's hyperparams
    p_space = get_parameter_space(
        protocols=config["spreading_model"]["parameters"]["protocols"],
        p_values=config["spreading_model"]["parameters"]["p_values"],
        networks=config["networks"],
    )
    p_space = list(p_space)

    # initialise regression models
    regr_models = load_regr_models(
        config_regr=config["regr_models"],
        rng_seed=config["run"]["rng_seed"],
        nb_repetitions=config["run"]["nb_repetitions"],
    )

    # get the starting time
    start_time = os_utils.get_current_time()
    print(f"Experiments started at {start_time}")

    # prepare output directory and deterimne how to store results
    out_dir = Path(config["logging"]["out_dir"]) / start_time
    out_dir.mkdir(exist_ok=True, parents=True)

    # save the config
    config["git_sha"] = os_utils.get_recent_git_sha()
    with open(out_dir / f"config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # main loop
    results = []
    p_bar = tqdm(list(p_space), desc="", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):
        case_descr = sim_utils.get_case_name_base(
                investigated_case[0],
                investigated_case[1],
                f"{investigated_case[2].n_type}-{investigated_case[2].n_name}",
            )
        p_bar.set_description_str(f"{idx}/{len(p_bar)}-{case_descr}")
        for regr_name, regr_model in regr_models.items():
            result = regr_model(
                net_type=investigated_case[2].n_type,
                net_name=investigated_case[2].n_name,
                protocol=investigated_case[0],
                p=investigated_case[1],
            )
            results.append(
                RegrResult(
                    net_type=investigated_case[2].n_type,
                    net_name=investigated_case[2].n_name,
                    protocol=investigated_case[0],
                    p=investigated_case[1],
                    regr_name=regr_name,
                    rmse_avg=result["rmse_avg"],
                    rmse_std=result["rmse_std"],
                    r2_avg=result["r2_avg"],
                    r2_std=result["r2_std"],
                )
            )

    # save global logs and config
    pd.DataFrame(results).to_csv(out_dir / "results.csv")

    finish_time = os_utils.get_current_time()
    print(f"Evaluations finished at {finish_time}")
    print(f"Evaluations lasted {os_utils.get_diff_of_times(start_time, finish_time)} minutes")

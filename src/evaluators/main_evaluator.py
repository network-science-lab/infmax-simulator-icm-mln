"""Main runner of the evaluation pipeline."""

from pathlib import Path
from typing import Any, Callable

import yaml
from tqdm import tqdm

from src.generators import commons
from src.generators.utils import (
    get_current_time,
    get_diff_of_times,
    zip_detailed_logs,
)
from src.evaluators.evaluate_seed_set import evaluate_seed_set


def run_evaluations(config: dict[str, Any]) -> None:
    print(config)
    p_space = commons.get_parameter_space(
        protocols=config["spreading_model"]["parameters"]["protocols"],
        p_values=config["spreading_model"]["parameters"]["p_values"],
        networks=config["networks"],
        as_tensor=True if config["spreading_model"]["name"] == "toorch" else False,
    )
    repetitions = config["run"]["repetitions"]
    # step_func = get_step_func(config["run"]["experiment_step"])

    # # prepare output directory and deterimne how to store results
    # out_dir = Path(config["logging"]["out_dir"])
    # out_dir.mkdir(exist_ok=True, parents=True)
    # compress_to_zip = config["logging"]["compress_to_zip"]
    # average_results = config["run"]["average_results"]

    # # save config
    # with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
    #     yaml.dump(config, f)

    # get a start time
    start_time = get_current_time()
    print(f"Experiments started at {start_time}")

    # main loop
    p_bar = tqdm(list(p_space), desc="", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):
        print(investigated_case)
        # try:
        #     step_func.experiment_step(
        #         protocol=investigated_case[0],
        #         p=investigated_case[1],
        #         net_name=investigated_case[2].name,
        #         net=investigated_case[2].graph,
        #         repetitions_nb=repetitions,
        #         average_results=average_results,
        #         case_idx=idx,
        #         p_bar=p_bar,
        #         out_dir=out_dir,
        #     )
        # except BaseException as e:
        #     case_descr = commons.get_case_name_base(
        #         investigated_case[0], investigated_case[1], investigated_case[2].name
        #     )
        #     print(f"\nExperiment failed for case: {case_descr}")
        #     raise e

    # # save global logs and config
    # if compress_to_zip:
    #     zip_detailed_logs([out_dir], rm_logged_dirs=True)

    finish_time = get_current_time()
    print(f"Experiments finished at {finish_time}")
    print(f"Experiments lasted {get_diff_of_times(start_time, finish_time)} minutes")


    

    # from _data_set.nsl_data_utils.loaders.constants import LAZEGA, OR
    # from _data_set.nsl_data_utils.loaders.net_loader import load_network
    # from _data_set.nsl_data_utils.loaders.sp_loader import get_gt_data, load_sp

    # net_name = LAZEGA
    # proto = OR
    # p = 0.25
    # n_steps = 10000
    # budget = 5
    # n_repetitions = 30

    # net = load_network(net_name, as_tensor=True)
    # sp = load_sp(net_name)
    # seed_set = get_gt_data(sp[net_name], proto, p, budget)
    # raw_results = evaluate_seed_set(net[net_name], seed_set, proto, p, n_steps, n_repetitions)
    # print("Performance of given seed set:\n", raw_results.mean())

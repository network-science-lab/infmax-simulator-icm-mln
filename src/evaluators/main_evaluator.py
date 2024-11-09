"""Main runner of the evaluation pipeline."""

from pathlib import Path
from typing import Any, Callable

import yaml
from tqdm import tqdm

from src.generators import commons
from src.generators.utils import (
    get_current_time,
    get_diff_of_times,
    save_magrinal_efficiences,
    zip_detailed_logs,
)
from src.evaluators import evaluate_seed_set, loader
from src.icm import nd_model, torch_model
from src.utils import get_recent_git_sha


def get_step_func(spreading_model_name: str) -> Callable:
    if spreading_model_name == nd_model.FixedBudgetMICModel.__name__:
        raise NotImplementedError(f"Pipeline for {spreading_model_name} is not yet ready!")
    elif spreading_model_name == torch_model.TorchMICModel.__name__:
        step_func = evaluate_seed_set
    else:
        raise ValueError(f"Incorrect name of them model {spreading_model_name}")
    print(f"Inferred step function as: {step_func.__name__}")
    return step_func


def run_experiments(config: dict[str, Any]) -> None:

    # get parameter space and experiment's hyperparams
    step_func = get_step_func(config["spreading_model"]["name"])
    p_space = commons.get_parameter_space(
        protocols=config["spreading_model"]["parameters"]["protocols"],
        p_values=config["spreading_model"]["parameters"]["p_values"],
        networks=config["networks"],
        as_tensor=True if step_func == evaluate_seed_set else False,
    )
    repetitions = config["run"]["repetitions"]

    # initialise influence maximisation models
    infmax_models = loader.load_infmax_models(
        config=config["infmax_models"],
        random_seed=config["run"]["random_seed"],
        nb_seeds=config["run"]["nb_seeds"],
    )

    # prepare output directory and deterimne how to store results
    out_dir = Path(config["logging"]["out_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    compress_to_zip = config["logging"]["compress_to_zip"]
    average_results = config["run"]["average_results"]

    # save the config
    config["git_sha"] = get_recent_git_sha()
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # get a start time
    start_time = get_current_time()
    print(f"Experiments started at {start_time}")

    # main loop
    p_bar = tqdm(list(p_space), desc="", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):
        try:
            seed_sets = {
                ifm_name: ifm_obj(investigated_case[2].graph)
                for ifm_name, ifm_obj in infmax_models.items()
            }
            step_func.evaluation_step(
                protocol=investigated_case[0],
                p=investigated_case[1],
                net=investigated_case[2],
                seed_sets=seed_sets,
                repetitions_nb=repetitions,
                average_results=average_results,
                case_idx=idx,
                p_bar=None,
                out_dir=out_dir,
            )
        except BaseException as e:
            case_descr = commons.get_case_name_base(
                investigated_case[0],
                investigated_case[1],
                investigated_case[2].name,
            )
            print(f"\nExperiment failed for case: {case_descr}")
            raise e

    # save global logs and config
    if compress_to_zip:
        zip_detailed_logs([out_dir], rm_logged_dirs=True)

    finish_time = get_current_time()
    print(f"Experiments finished at {finish_time}")
    print(f"Experiments lasted {get_diff_of_times(start_time, finish_time)} minutes")



# TODO: add repetetitive selecting seed set
# TODO: add GT seed selector
# TODO: add add voterank

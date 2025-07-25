"""Main runner of the evaluation pipeline."""

from pathlib import Path
from typing import Any

import json
import yaml
from tqdm import tqdm

from src import os_utils, sim_utils
from src.evaluators import loader
from src.evaluators.infmax_methods import BaseChoice, GroundTruth
from src.evaluators.utils import log_function_details


@log_function_details
def process_case(
    repetitions_stochastic: Any,
    ifm_obj: BaseChoice,
    inv_net: Any | sim_utils.Network,
    inv_proto: Any,
    inv_p: Any,
    case_descr: str,
) -> dict[str, Any]:
    inv_seeds = []
    try:
        for _ in range(repetitions_stochastic if ifm_obj.is_stochastic else 1):
            repetition_seeds = ifm_obj(
                network_pt=inv_net.n_graph_pt,
                network_nx=inv_net.n_graph_nx,
                net_type=inv_net.n_type,
                net_name=inv_net.n_name,
                protocol=inv_proto,
                p=inv_p,
                nb_seeds=len(inv_net.n_graph_pt.actors_map),
            )
            inv_seeds.append(repetition_seeds)
    except BaseException as e:
        print(f"\nEvaluation failed for case: {case_descr}")
        raise e

    return {
        "net_type": inv_net.n_type,
        "net_name": inv_net.n_name,
        "protocol": inv_proto,
        "p": inv_p,
        "seed_sets": inv_seeds,
    }


def run_experiments(config: dict[str, Any]) -> None:

    # get parameter space and experiment's hyperparams
    p_space = sim_utils.get_parameter_space(
        protocols=config["spreading_model"]["parameters"]["protocols"],
        p_values=config["spreading_model"]["parameters"]["p_values"],
        networks=config["networks"],
    )
    p_space = list(p_space)
    repetitions_stochastic = config["run"]["nb_repetitions"]["stochastic_infmax"]

    # initialise influence maximisation models
    infmax_models = loader.load_infmax_models(
        config_infmax=[
            *config["infmax_models"],
            {"name": "ground_truth", "class": GroundTruth.__name__},
        ],
        config_sp=config["spreading_potential_score"],
        rng_seed=config["run"]["rng_seed"],
        device=config["run"]["device"],
    )

    # get a starting time
    start_time = os_utils.get_current_time()
    print(f"Evaluations started at {start_time}")

    # prepare output directory and deterimne how to store results
    out_dir = Path(config["logging"]["out_dir"]) / start_time
    out_dir.mkdir(exist_ok=True, parents=True)

    # save the config
    config["git_sha"] = os_utils.get_recent_git_sha()
    with open(out_dir / f"config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # main loop
    for ifm_name, ifm_obj in infmax_models.items():
        ifm_results = []
        p_bar = tqdm(p_space.copy(), desc="", leave=False, colour="green")
        for idx, investigated_case in enumerate(p_bar):

            inv_proto = investigated_case[0]
            inv_p = investigated_case[1]
            inv_net: sim_utils.Network = investigated_case[2]

            case_descr = sim_utils.get_case_name_base(inv_proto, inv_p, f"{inv_net.n_type}-{inv_net.n_name}")
            p_bar.set_description_str(f"[{ifm_name}]-{idx}/{len(p_bar)}-{case_descr}")

            partial_result = process_case(
                repetitions_stochastic=repetitions_stochastic,
                ifm_obj=ifm_obj,
                inv_net=inv_net,
                inv_proto=inv_proto,
                inv_p=inv_p,
                case_descr=case_descr,
                log_dir=out_dir,
            )
            ifm_results.append(partial_result)

        with open(out_dir / f"{ifm_name}.json", "w") as file:
            json.dump(ifm_results, file)

    finish_time = os_utils.get_current_time()
    print(f"Evaluations finished at {finish_time}")
    print(f"Evaluations lasted {os_utils.get_diff_of_times(start_time, finish_time)} minutes")

import itertools
import yaml

from pathlib import Path
from typing import List

import network_diffusion as nd
import pandas as pd

from misc.net_loader import load_network
from misc.utils import *
from tqdm import tqdm


def parameter_space(protocols, mi_values, networks):
    nets = [(n, load_network(n)) for n in networks]  # network name, network
    return itertools.product(protocols, mi_values, nets)


def run_experiments(config):

    # get parameter space and experiment's hyperparams
    p_space = parameter_space(
        protocols=config["model"]["parameters"]["protocols"],
        mi_values=config["model"]["parameters"]["mi_values"],
        networks=config["networks"],
    )
    seed_max_budget = config["model"]["parameters"]["max_seed_budget"]

    max_epochs_num = config["run"]["max_epochs_num"]
    patience = config["run"]["patience"]
    out_dir = Path(config["logging"]["out_dir"]) / config["logging"]["name"]
    out_dir.mkdir(exist_ok=True, parents=True)

    # save config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # init containers for results
    global_stats_handler = pd.DataFrame(data={})
    detailed_stats_dirs: list[Path] = []
    print(f"Experiments started at {get_current_time()}")
    
    p_bar = tqdm(list(p_space), desc="main loop", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):

        # obtain parameters of the propagation scenario
        protocol = investigated_case[0]
        mi_value = investigated_case[1]
        net_name, net = investigated_case[2]

        greedy_ranking: List[nd.MLNetworkActor] = []
        actors_num = net.get_actors_num()

        # repeat until spent seeding budget exceeds maximum value
        while (100 * len(greedy_ranking) / actors_num) <= seed_max_budget:

            # containers for the best actor in the run and its performance
            best_actor = None
            best_diffusion_len = max_epochs_num
            best_coverage = 0
            best_logs = None

            # obtain pool of actors and limit of budget in the run
            eval_seed_budget = 100 * (len(greedy_ranking) + 1) / actors_num
            available_actors = set(net.get_actors()).difference(
                set(greedy_ranking)
            )

            # update progress_bar
            case_name = (
                f"proto_{protocol}--a_seeds_{round(eval_seed_budget, 2)}"
                f"--mi_{round(mi_value, 3)}--net_{net_name}"
            )
            p_bar.set_description_str(str(case_name))

            # iterate greedly through all available actors to combination that
            # gves the best coverage
            for actor in available_actors:

                # initialise model with "ranking" that prioritises current actor
                apriori_ranking = [*greedy_ranking, actor, *available_actors.difference({actor})]
                mltm = nd.models.MLTModel(
                    protocol=protocol,
                    seed_selector=nd.seeding.MockyActorSelector(apriori_ranking),
                    seeding_budget=(100 - eval_seed_budget, eval_seed_budget),
                    mi_value=mi_value,
                )

                # run experiment on a deep copy of the network!
                experiment = nd.Simulator(model=mltm, network=net.copy())
                logs = experiment.perform_propagation(max_epochs_num, patience)

                # compute boost that current actor provides
                diffusion_len, active_actors, _ = extract_basic_stats(
                    detailed_logs=logs._local_stats
                )
                coverage = active_actors / actors_num * 100

                # if gain is relevant update the best currently actor
                if (
                    coverage > best_coverage or 
                    (
                        coverage == best_coverage and
                        diffusion_len < best_diffusion_len
                    )
                ):
                    best_actor = actor
                    best_diffusion_len = diffusion_len
                    best_coverage = coverage
                    best_logs = logs
                    print(
                        f"\n\tcurrently best actor '{best_actor.actor_id}' for "
                        f"greedy list: {[i.actor_id for i in greedy_ranking]}, "
                        f"coverage: {round(best_coverage, 2)}"
                    )

            # when the best combination is found update greedy ranking
            greedy_ranking.append(best_actor)

            # save logs for further analysis
            case_dir = out_dir.joinpath(f"{idx}-{case_name}")
            case_dir.mkdir(exist_ok=True, parents=True)
            detailed_stats_dirs.append(case_dir)
            best_logs.report(path=str(case_dir))

            # update global logs
            case = {
                "network": net_name,
                "protocol": protocol,
                "seeding_budget": eval_seed_budget,
                "mi_value": mi_value,
                "repetition_run": 1,
                "diffusion_len": best_diffusion_len,
                "active_actors_prct": best_coverage,
                "seed_actors_prct": eval_seed_budget,
                "gain": compute_gain(eval_seed_budget, best_coverage),
            }
            global_stats_handler = pd.concat(
                [global_stats_handler, pd.DataFrame.from_records([case])],
                ignore_index=True,
                axis=0,
            )

    # save global logs and config
    global_stats_handler.to_csv(out_dir.joinpath("results.csv"))
    zip_detailed_logs(detailed_stats_dirs, rm_logged_dirs=True)

    print(f"Experiments finished at {get_current_time()}")

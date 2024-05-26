import itertools
import json
import yaml

import network_diffusion as nd
import pandas as pd
from misc.net_loader import load_network
from misc.utils import *
from pathlib import Path
from tqdm import tqdm


def get_parameter_space(protocols, seed_budgets, mi_values, networks):
    seed_budgets_full = [(100 - i, i) for i in seed_budgets]
    return itertools.product(protocols, seed_budgets_full, mi_values, networks)


def load_nets_compute_rankings(seed_selector, networks, out_dir, ranking_path):
    ss = get_seed_selector(seed_selector["name"])(**seed_selector["parameters"])
    nets_and_ranks = {}  # network name:  (network graph, ranking)

    for idx, net_name in enumerate(networks):
        print(f"Computing ranking for: {net_name} ({idx+1}/{len(networks)})")

        # load network
        net_graph = load_network(net_name)
        ss_ranking_name = Path(f"{net_name}_{ss.__class__.__name__}.json")
        print("\tnetwork loaded")

        # compute ranking or load saved one
        if ranking_path:
            ranking_file = Path(ranking_path) /ss_ranking_name
            with open(ranking_file, "r") as f:
                ranking_dict = json.load(f)
            ranking = [nd.MLNetworkActor.from_dict(rd) for rd in ranking_dict]
        else:
            ranking = ss(net_graph, actorwise=True)
        print("\tranking computed/loaded")
        nets_and_ranks[net_name] = (net_graph, ranking)

        # save computed ranking
        with open(out_dir / ss_ranking_name, "w") as f:
            json.dump(ranking, f, cls=JSONEncoder)
            print(f"\tranking saved in the storage")

    return nets_and_ranks 


def run_experiments(config):
    print(f"Experiments started at {get_current_time()}")

    # get parameter space and experiment's hyperparams
    p_space = get_parameter_space(
        protocols=config["model"]["parameters"]["protocols"],
        seed_budgets=config["model"]["parameters"]["seed_budgets"],
        mi_values=config["model"]["parameters"]["mi_values"],
        networks=config["networks"],
    )
    max_epochs_num = config["run"]["max_epochs_num"]
    patience = config["run"]["patience"]
    logging_freq = config["logging"]["full_output_frequency"]
    ranking_path = config.get("ranking_path")
    out_dir = Path(config["logging"]["out_dir"]) / config["logging"]["name"]
    out_dir.mkdir(exist_ok=True, parents=True)

    # load networks, compute rankings and save them with config
    nets_and_ranks = load_nets_compute_rankings(
        seed_selector=config["model"]["parameters"]["ss_method"],
        networks=config["networks"],
        out_dir=out_dir,
        ranking_path=ranking_path
    )
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # init containers for results
    global_stats_handler = pd.DataFrame(data={})
    detailed_stats_dirs: list[Path] = []
    print(f"Simulations started at {get_current_time()}")

    p_bar = tqdm(list(p_space), desc="main loop", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):

        # obtain parameters of the propagation scenario
        protocol = investigated_case[0]
        seeding_budget = investigated_case[1]
        mi_value = investigated_case[2]
        net_name = investigated_case[3]
        net, ranking = nets_and_ranks[net_name]

        # initialise model - in order to speed up computations we use mocky 
        # selector and feed it with apriori computed ranking
        mltm = nd.models.MLTModel(
            protocol=protocol,
            seed_selector=nd.seeding.MockyActorSelector(ranking),
            seeding_budget = seeding_budget,
            mi_value=mi_value,
        )

        # update progress_bar
        case_name = (
            f"proto_{protocol}--a_seeds_{seeding_budget[1]}"
            f"--mi_{round(mi_value, 3)}--net_{net_name}--run_no_repetitions"
        )
        p_bar.set_description_str(str(case_name))

        try:
            # run experiment on a deep copy of the network!
            experiment = nd.Simulator(model=mltm, network=net.copy())
            logs = experiment.perform_propagation(
                n_epochs=max_epochs_num, patience=patience
            )

            # obtain global data and if case is even local one as well
            diffusion_len, active_actors, seed_actors = extract_basic_stats(
                detailed_logs=logs._local_stats
            )
            active_actors_prct = active_actors / net.get_actors_num() * 100
            seed_actors_prct = seed_actors / net.get_actors_num() * 100
            gain = compute_gain(seed_actors_prct, active_actors_prct)
            if idx % logging_freq == 0:
                case_dir = out_dir.joinpath(f"{idx}-{case_name}")
                case_dir.mkdir(exist_ok=True)
                detailed_stats_dirs.append(case_dir)
                logs.report(path=str(case_dir))

        except KeyboardInterrupt as e:
            raise e

        except BaseException as e:
            diffusion_len = None
            active_actors_prct = None
            seed_actors_prct = None
            gain = None
            print(f"Ooops something went wrong for case: {case_name}: {e}")

        # update global logs
        case = {
            "network": net_name,
            "protocol": protocol,
            "seeding_budget": seeding_budget[1],
            "mi_value": mi_value,
            "repetition_run": 1,
            "diffusion_len": diffusion_len,
            "active_actors_prct": active_actors_prct,
            "seed_actors_prct": seed_actors_prct,
            "gain": gain,
        }
        global_stats_handler = pd.concat(
            [global_stats_handler, pd.DataFrame.from_records([case])],
            ignore_index=True,
            axis=0,
        )

    # save global logs
    global_stats_handler.to_csv(out_dir.joinpath("results.csv"))
    zip_detailed_logs(detailed_stats_dirs, rm_logged_dirs=True)
    print(f"Experiments finished at {get_current_time()}")

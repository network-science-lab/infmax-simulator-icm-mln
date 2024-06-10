"""Main runner of the simulator."""

from dataclasses import dataclass
from itertools import product
from math import log10
from pathlib import Path
from typing import Any

import warnings
import yaml

from misc.net_loader import load_network
from misc.utils import (
    extract_simulation_result,
    get_current_time,
    get_diff_of_times,
    mean_repeated_results,
    save_magrinal_efficiences,
    SimulationResult,
    zip_detailed_logs,
)
from tqdm import tqdm

import network_diffusion as nd

warnings.filterwarnings(action="ignore", category=FutureWarning)


@dataclass(frozen=True)
class Network:
    name: str
    graph: nd.MultilayerNetwork


def get_parameter_space(protocols: list[str], p_values: list[float], networks: list[str]) -> product:
    nets = []
    for n in networks:
        print(f"Loading {n} network")
        nets.append(Network(n, load_network(n)))
    return product(protocols, p_values, nets)


def get_ranking(actor: nd.MLNetworkActor, actors: list[nd.MLNetworkActor]) -> nd.seeding.MockingActorSelector:
    ranking_list = [actor, *set(actors).difference({actor})]
    return nd.seeding.MockingActorSelector(ranking_list)


def get_case_name_base(protocol: str, p: float, net_name: str) -> str:
    return f"proto-{protocol}--p-{round(p, 3)}--net-{net_name}"


def get_case_name_rich(
    protocol: str,
    p: float,
    net_name: str,
    case_idx: int,
    cases_nb: int,
    actor_idx: int,
    actors_nb: int,
    rep_idx: int,
    reps_nb: int
) -> str:
    return (
        f"case-{str(case_idx).zfill(int(log10(cases_nb)+1))}/{cases_nb}--" +
        f"actor-{str(actor_idx).zfill(int(log10(actors_nb)+1))}/{actors_nb}--" +
        f"repet-{str(rep_idx).zfill(int(log10(reps_nb)+1))}/{reps_nb}--" +
        get_case_name_base(protocol, p, net_name)
    )


def experiment_step(
    protocol: str,
    p: float,
    net_name: str,
    net: nd.MultilayerNetwork,
    repetitions_nb: int,
    average_results: bool,
    case_idx: int,
    p_bar: tqdm,
    out_dir: Path,
) -> None:
    # obtain pool of actors and a budget in the run
    actors = net.get_actors()
    actors_nb = len(actors)
    active_fraction = 100 * (1 / actors_nb)
    seeding_budget = (100 - active_fraction, active_fraction, 0)

    # init a container for computed marginal efficiency of each actor
    marginal_efficiencies = []

    # iterate through all available actors and check their influencial power
    for actor_idx, actor in enumerate(actors):

        # initialise model with "ranking" that prioritises current actor
        apriori_ranking = get_ranking(actor, actors)
        micm = nd.models.MICModel(
            seeding_budget=seeding_budget,
            seed_selector=apriori_ranking,
            protocol=protocol,
            probability=p,
        )

        # repeat the simulation to get mean results
        repeated_results: list[SimulationResult] = []
        for rep in range(repetitions_nb):

            # update progress_bar
            p_bar.set_description_str(
                get_case_name_rich(
                    protocol=protocol,
                    p=p,
                    net_name=net_name,
                    case_idx=case_idx,
                    cases_nb=len(p_bar),
                    actor_idx=actor_idx,
                    actors_nb=actors_nb,
                    rep_idx=rep,
                    reps_nb=repetitions_nb
                )
            )

            # run experiment on a deep copy of the network!
            experiment = nd.Simulator(model=micm, network=net.copy())
            logs = experiment.perform_propagation(
                n_epochs=actors_nb * 2,  # this value is an "overkill"
                patience=1,
            )

            # compute boost that current actor provides
            simulation_result = extract_simulation_result(logs._local_stats, net, actor)
            repeated_results.append(simulation_result)
        
        # get mean value for each result
        if average_results:
            marginal_efficiencies.append(mean_repeated_results(repeated_results))
        else:
            marginal_efficiencies.extend(repeated_results)

    # save efficiences obtained for this case
    investigated_case_file_path = out_dir / f"{get_case_name_base(protocol, p, net_name)}.csv"
    save_magrinal_efficiences(marginal_efficiencies, investigated_case_file_path)


def run_experiments(config: dict[str, Any]) -> None:

    # get parameter space and experiment's hyperparams
    p_space = get_parameter_space(
        protocols=config["model"]["parameters"]["protocols"],
        p_values=config["model"]["parameters"]["p_values"],
        networks=config["networks"],
    )
    repetitions = config["run"]["repetitions"]

    # prepare output directory and deterimne how to store results
    out_dir = Path(config["logging"]["out_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    compress_to_zip = config["logging"]["compress_to_zip"]
    average_results = config["run"]["average_results"]

    # save config
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # get a start time
    start_time = get_current_time()
    print(f"Experiments started at {start_time}")

    # main loop
    p_bar = tqdm(list(p_space), desc="", leave=False, colour="green")
    for idx, investigated_case in enumerate(p_bar):
        try:
            experiment_step(
                protocol=investigated_case[0],
                p=investigated_case[1],
                net_name=investigated_case[2].name,
                net=investigated_case[2].graph,
                repetitions_nb=repetitions,
                average_results=average_results,
                case_idx=idx,
                p_bar=p_bar,
                out_dir=out_dir,
            )
        except BaseException as e:
            case_descr = get_case_name_base(
                investigated_case[0], investigated_case[1], investigated_case[2].name
            )
            print(f"\nExperiment failed for case: {case_descr}")
            raise e

    # save global logs and config
    if compress_to_zip:
        zip_detailed_logs([out_dir], rm_logged_dirs=True)

    finish_time = get_current_time()
    print(f"Experiments finished at {finish_time}")
    print(f"Experiments lasted {get_diff_of_times(start_time, finish_time)} minutes")

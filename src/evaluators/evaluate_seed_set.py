"""Functions to help in evaluating performance of a given seed set."""
import sys

from tqdm import tqdm

from src.utils import compute_gain
sys.path.append("/Users/michal/Development/infmax-simulator-icm-mln")

from typing import Any, Literal

import network_diffusion as nd
import pandas as pd
from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import get_gt_data, load_sp

from src.icm.torch_model import TorchMICModel, TorchMICSimulator
from src.generators import commons
from src.generators.utils import (
    mean_evaluation_results,
    save_magrinal_efficiences,
    SimulationResult,
    EvaluationResult,
    
)
from pathlib import Path


def evaluation_step(
    protocol: Literal["OR", "AND"],
    p: float,
    net: commons.Network,
    seed_sets: dict[str, set[str]],
    repetitions_nb: int,
    average_results: bool,
    case_idx: int,
    p_bar: tqdm,
    out_dir: Path,
) -> list[EvaluationResult]:
    """Run multilayer ICM on given seed set and model's parameters."""
    evaluation_results = []

    # iterate through provided seed sets and check their influencial power
    for infmax_name, seed_set in seed_sets.items():

        # initialise ICM
        micm = TorchMICModel(protocol=protocol, probability=p)

        # repeat the simulation to get mean results
        repeated_results: list[EvaluationResult] = []
        for rep in range(repetitions_nb):

            # run experiment on a deep copy of the network!
            simulator = TorchMICSimulator(
                model=micm,
                net=net.graph,
                n_steps=len(net.graph.actors_map) * 2,
                seed_set=seed_set,
            )
            logs = simulator.perform_propagation()

            # compute boost that current seed set provides
            simulation_result = EvaluationResult(
                infmax_model=infmax_name,
                seed_set=";".join(seed_set),
                gain=compute_gain(len(seed_set), logs["exposed"], logs["exposed"] + logs["not_exposed"]),
                simulation_length=logs["simulation_length"],
                exposed=logs["exposed"],
                not_exposed=logs["not_exposed"],
                peak_infected=logs["peak_infected"],
                peak_iteration=logs["peak_iteration"],
                expositions_rec=";".join([str(_) for _ in logs["expositions_rec"]]),

            )
            repeated_results.append(simulation_result)
    
        # get mean value for each result
        if average_results:
            evaluation_results.append(mean_evaluation_results(repeated_results))
        else:
            evaluation_results.extend(repeated_results)
    
    # save efficiences obtained for this case
    investigated_case_file_path = out_dir / f"{commons.get_case_name_base(protocol, p, f"{net.type}_{net.name}")}.csv"
    save_magrinal_efficiences(evaluation_results, investigated_case_file_path)



if __name__ == "__main__":

    from _data_set.nsl_data_utils.loaders.constants import *

    net_name = LAZEGA
    proto = OR
    p = 0.25
    n_steps = 10000
    budget = 5
    n_repetitions = 30

    net = load_network(net_name, as_tensor=True)
    sp = load_sp(net_name)
    seed_set = get_gt_data(sp[net_name], proto, p, budget)
    raw_results = evaluation_step(net[net_name], seed_set, proto, p, n_steps, n_repetitions)
    print("Performance of given seed set:\n", raw_results.mean())

"""Functions to help in evaluating performance of a given seed set."""

from pathlib import Path
from typing import Literal

from src import os_utils, sim_utils
from src.evaluators.utils import EvaluationResult, mean_evaluation_results
from src.icm.torch_model import TorchMICModel, TorchMICSimulator


def evaluation_step(
    protocol: Literal["OR", "AND"],
    p: float,
    net: sim_utils.Network,
    seed_sets: dict[str, set[str]],
    repetitions_nb: int,
    average_results: bool,
    case_name: int,
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
            gain = sim_utils.compute_gain(
                seed_nb=len(seed_set),
                exposed_nb=logs["exposed"],
                total_actors=logs["exposed"] + logs["not_exposed"],
            )

            # compute boost that current seed set provides
            simulation_result = EvaluationResult(
                infmax_model=infmax_name,
                seed_set=";".join(seed_set),
                gain=gain,
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
    out_path = out_dir / f"{case_name}.csv"
    os_utils.export_dataclasses(evaluation_results, out_path)
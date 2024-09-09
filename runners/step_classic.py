"""Main runner of the simulator."""

from pathlib import Path

from _data_set.nsl_data_utils.models.nd_model import FixedBudgetMICModel
from runners.utils import (
    extract_simulation_result,
    mean_repeated_results,
    save_magrinal_efficiences,
    SimulationResult,
)
from tqdm import tqdm
from runners import commons

import network_diffusion as nd


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
    actors = net.get_actors()
    marginal_efficiencies = []

    # iterate through all available actors and check their influencial power
    for actor_idx, actor in enumerate(actors):

        # initialise model with "ranking" that prioritises current actor
        apriori_ranking = commons.get_ranking(actor, actors)
        micm = FixedBudgetMICModel(
            seed_selector=apriori_ranking,
            protocol=protocol,
            probability=p,
        )

        # repeat the simulation to get mean results
        repeated_results: list[SimulationResult] = []
        for rep in range(repetitions_nb):

            # update progress_bar
            p_bar.set_description_str(
                commons.get_case_name_rich(
                    protocol=protocol,
                    p=p,
                    net_name=net_name,
                    case_idx=case_idx,
                    cases_nb=len(p_bar),
                    actor_idx=actor_idx,
                    actors_nb=len(actors),
                    rep_idx=rep,
                    reps_nb=repetitions_nb
                )
            )

            # run experiment on a deep copy of the network!
            experiment = nd.Simulator(model=micm, network=net.copy())
            logs = experiment.perform_propagation(
                n_epochs=len(actors) * 2,  # this value is an "overkill"
                patience=1,
            )

            # compute boost that current actor provides
            simulation_result = extract_simulation_result(logs.get_detailed_logs(), net, actor)
            repeated_results.append(simulation_result)
        
        # get mean value for each result
        if average_results:
            marginal_efficiencies.append(mean_repeated_results(repeated_results))
        else:
            marginal_efficiencies.extend(repeated_results)

    # save efficiences obtained for this case
    investigated_case_file_path = out_dir / f"{commons.get_case_name_base(protocol, p, net_name)}.csv"
    save_magrinal_efficiences(marginal_efficiencies, investigated_case_file_path)

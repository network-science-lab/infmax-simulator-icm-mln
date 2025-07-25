"""Single simulation step implemented with `networkx`."""

from pathlib import Path

import network_diffusion as nd
from tqdm import tqdm

from src import os_utils, sim_utils
from src.icm.nd_model import FixedBudgetMICModel
from src.generators.utils import (
    SimulationResult,
    extract_simulation_result,
    get_ranking,
    mean_simulation_results,
)


def experiment_step(
    protocol: str,
    p: float,
    net: sim_utils.Network,
    repetitions_nb: int,
    average_results: bool,
    case_idx: int,
    p_bar: tqdm,
    out_dir: Path,
    **kwargs,
) -> None:
    actors = net.n_graph_nx.get_actors()
    marginal_efficiencies = []

    # iterate through all available actors and check their influencial power
    for actor_idx, actor in enumerate(actors):

        # initialise model with "ranking" that prioritises current actor
        apriori_ranking = get_ranking(actor, actors)
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
                sim_utils.get_case_name_rich(
                    protocol=protocol,
                    p=p,
                    net_name=f"{net.n_type}-{net.n_name}" if net.n_type != net.n_name else net.n_name,
                    case_idx=case_idx,
                    cases_nb=len(p_bar),
                    actor_idx=actor_idx,
                    actors_nb=len(actors),
                    rep_idx=rep,
                    reps_nb=repetitions_nb
                )
            )

            # run experiment on a deep copy of the network!
            experiment = nd.Simulator(model=micm, network=net.n_graph_nx.copy())
            logs = experiment.perform_propagation(
                n_epochs=len(actors) * 2,  # this value is an "overkill"
                patience=1,
            )

            # compute boost that current actor provides
            simulation_result = extract_simulation_result(
                logs.get_detailed_logs(), net.n_graph_nx, actor
            )
            repeated_results.append(simulation_result)
        
        # get mean value for each result
        if average_results:
            marginal_efficiencies.append(mean_simulation_results(repeated_results))
        else:
            marginal_efficiencies.extend(repeated_results)

    # save efficiences obtained for this case
    if net.n_name != net.n_type:
        out_dir = out_dir / net.n_type
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"{sim_utils.get_case_name_base(protocol, p, net.n_name)}.csv"
    os_utils.export_dataclasses(marginal_efficiencies, out_path)

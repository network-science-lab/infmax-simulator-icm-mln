"""Main runner of the simulator."""

from pathlib import Path

from _data_set.nsl_data_utils.models.torch_model import TorchMICModel, TorchMICSimulator
from runners.utils import (
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
    net: nd.MultilayerNetworkTorch,
    repetitions_nb: int,
    average_results: bool,
    case_idx: int,
    p_bar: tqdm,
    out_dir: Path,
) -> None:
    marginal_efficiencies = []

    # iterate through all available actors and check their influencial power
    for actor_name, actor_idx in net.actors_map.items():

        # initialise model with "ranking" that prioritises current actor
        micm = TorchMICModel(protocol=protocol, probability=p)

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
                    actors_nb=len(net.actors_map),
                    rep_idx=rep,
                    reps_nb=repetitions_nb
                )
            )

            # run experiment on a deep copy of the network!
            experiment = TorchMICSimulator(model=micm, net=net, n_steps=len(net.actors_map) * 2, seed_set={actor_name})
            logs = experiment.perform_propagation()

            # compute boost that current actor provides
            simulation_result = SimulationResult(actor=actor_name, **logs)
            repeated_results.append(simulation_result)
        
        # get mean value for each result
        if average_results:
            marginal_efficiencies.append(mean_repeated_results(repeated_results))
        else:
            marginal_efficiencies.extend(repeated_results)

    # save efficiences obtained for this case
    investigated_case_file_path = out_dir / f"{commons.get_case_name_base(protocol, p, net_name)}.csv"
    save_magrinal_efficiences(marginal_efficiencies, investigated_case_file_path)

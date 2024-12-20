"""Single simulation step implemented with `torch`."""

from pathlib import Path

from tqdm import tqdm

from src import os_utils, sim_utils
from src.icm.torch_model import TorchMICModel, TorchMICSimulator
from src.generators.utils import(
        SimulationResult,
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
    device: str,
) -> None:
    marginal_efficiencies = []

    # iterate through all available actors and check their influencial power
    for actor_name, actor_idx in net.n_graph.actors_map.items():

        # initialise model with "ranking" that prioritises current actor
        micm = TorchMICModel(protocol=protocol, probability=p)

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
                    actors_nb=len(net.n_graph.actors_map),
                    rep_idx=rep,
                    reps_nb=repetitions_nb
                )
            )

            # run experiment on a deep copy of the network!
            experiment = TorchMICSimulator(
                model=micm,
                net=net.n_graph,
                n_steps=len(net.n_graph.actors_map) * 2,
                seed_set={actor_name},
                device=device,
            )
            logs = experiment.perform_propagation()

            # compute boost that current actor provides
            simulation_result = SimulationResult(
                actor=actor_name,
                simulation_length=logs["simulation_length"],
                exposed=logs["exposed"],
                not_exposed=logs["not_exposed"],
                peak_infected=logs["peak_infected"],
                peak_iteration=logs["peak_iteration"],
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

from dataclasses import dataclass, asdict
import datetime
import random
import shutil

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class SimulationResult:
    actor: int  # actor's id
    simulation_length: int  # nb. of simulation steps
    exposed: int  # nb. of infected actors
    not_exposed: int  # nb. of not infected actors
    peak_infected: int  # maximal nb. of infected actors in a single sim. step
    peak_iteration: int  # a sim. step when the peak occured


def extract_simulation_result(detailed_logs, net, actor):
    """Get length of diffusion, real number of seeds and final coverage."""
    actors_infected_total = 0
    peak_infections_nb = 0
    peak_iteration_nb = 0
    actors_nb = net.get_actors_num()

    # sort epochs indices
    epochs_sorted = sorted([int(e) for e in detailed_logs.keys()])

    # calculate metrics for each epoch
    for epoch_num in epochs_sorted:

        # obtain a number of actors in each state in the current epoch
        actorwise_log = nodewise_to_actorwise_epochlog(
            nodewise_epochlog=detailed_logs[epoch_num], actors_nb=actors_nb
        )
        actors_infected_epoch = actorwise_log["active_actors"] + actorwise_log["activated_actors"]

        # sanity checks
        if epoch_num == 0:
            assert actorwise_log["active_actors"] == 1, \
                f"Number of seeds must be 1 (got: {actorwise_log['active_actors']})"
            
        else:
            assert actors_infected_epoch >= actors_infected_total, \
                f"Results contradict themselves! \
                Number of active actors in {epoch_num} epoch: {actors_infected_epoch} \
                number of all actors active so far: {actors_infected_total}"
    
        # update peaks
        if actorwise_log["active_actors"] > peak_infections_nb:
            peak_infections_nb = actorwise_log["active_actors"]
            peak_iteration_nb = epoch_num
        
        # update nb of infected actors
        actors_infected_total = actors_infected_epoch

    return SimulationResult(
        actor=actor.actor_id,
        simulation_length=len(detailed_logs) - 1,
        exposed=actors_infected_total,
        not_exposed=actors_nb - actors_infected_total,
        peak_infected=peak_infections_nb,
        peak_iteration=peak_iteration_nb
    )


def mean_repeated_results(repeated_results: list[SimulationResult]) -> SimulationResult:
    rr_dict_all = [asdict(rr) for rr in repeated_results]
    rr_df_all = pd.DataFrame(rr_dict_all)
    rr_dict_mean = rr_df_all.groupby("actor").mean().reset_index().iloc[0].round(3).to_dict()
    return SimulationResult(**rr_dict_mean)


def save_magrinal_efficiences(marginal_efficiencies: list[SimulationResult], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in marginal_efficiencies]
    pd.DataFrame(me_dict_all).to_csv(out_path, index=False)


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2):
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return t_2 - t_1


def zip_detailed_logs(logged_dirs: list[Path], rm_logged_dirs: bool = True) -> None:
    if len(logged_dirs) == 0:
        print("No directories provided to create archive from.")
        return
    for dir_path in logged_dirs:
        shutil.make_archive(logged_dirs[0].parent / "_output", "zip", root_dir=str(dir_path))
    if rm_logged_dirs:
        for dir_path in logged_dirs:
            shutil.rmtree(dir_path)
    print(f"Compressed detailed logs.")


def nodewise_to_actorwise_epochlog(nodewise_epochlog, actors_nb):
    inactive_nodes, active_nodes, activated_nodes = [], [], []
    for node_log in nodewise_epochlog:
        new_state = node_log["new_state"]
        node_name = node_log["node_name"]
        if new_state == "0":
            inactive_nodes.append(node_name)
        elif new_state == "1":
            active_nodes.append(node_name)
        elif new_state == "-1":
            activated_nodes.append(node_name)
        else:
            raise ValueError
    actorwise_log = {
        "inactive_actors": len(set(inactive_nodes)),
        "active_actors": len(set(active_nodes)),
        "activated_actors": len(set(activated_nodes)),
    }
    assert actors_nb == sum(actorwise_log.values())
    return actorwise_log

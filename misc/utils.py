from dataclasses import dataclass, asdict
import datetime
import os
import random
import shutil
import subprocess

from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass(frozen=True)
class SimulationResult:
    actor_id: int
    simulation_length: int
    actors_infected: int
    actors_not_infected: int
    peak_infections_nb: int
    peak_iteration_nb: int


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
            assert (
                actorwise_log["active_actors"] == 1, 
                f"Number of seeds must be 1 (got: {actorwise_log['active_actors']} + )"
            )
        else:
            assert (
                actors_infected_epoch >= actors_infected_total,
                f"Results contradict themselves! \
                Number of active actors in {epoch_num} epoch: {actors_infected_epoch} \
                number of all actors active so far: {actors_infected_total}"
            )
    
        # update peaks
        if actorwise_log["active_actors"] > peak_infections_nb:
            peak_infections_nb = actorwise_log["active_actors"]
            peak_iteration_nb = epoch_num + 1  # we don't start counting from 0 :)
        
        # update nb of infected actors
        actors_infected_total = actors_infected_epoch

    return SimulationResult(
        actor_id=actor.actor_id,
        simulation_length=len(detailed_logs) - 1,
        actors_infected=actors_infected_total,  # do we consider within this number the seed as well?
        actors_not_infected=actors_nb - actors_infected_total,
        peak_infections_nb=peak_infections_nb,
        peak_iteration_nb=peak_iteration_nb
    )


def mean_repeated_results(repeated_results: list[SimulationResult]) -> SimulationResult:
    rr_dict_all = [asdict(rr) for rr in repeated_results]
    rr_df_all = pd.DataFrame(rr_dict_all)
    rr_dict_mean = rr_df_all.groupby("actor_id").mean().reset_index().iloc[0].round(3).to_dict()
    return SimulationResult(**rr_dict_mean)


def save_magrinal_efficiences(marginal_efficiencies: list[SimulationResult], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in marginal_efficiencies]
    pd.DataFrame(me_dict_all).to_csv(out_path)


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2):
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return round((t_2 - t_1).seconds / 60, 2)  # TODO: consider using timedelta


def zip_detailed_logs(logged_dirs: list[Path], rm_logged_dirs: bool = True) -> None:
    # Ensure at least one directory is provided
    if len(logged_dirs) == 0:
        print("No directories provided to create archive from.")
        return
    
    # Get the parent directory of the first directory in logged_dirs
    parent_dir = logged_dirs[0].parent
    
    # Create the name for the zip file based on the parent directory
    zip_filename = parent_dir / "_output.zip"

    # Create the archive
    try:
        logged_dirs_as_str = " ".join([str(ld) for ld in logged_dirs])
        command = f"zip -r {zip_filename} {logged_dirs_as_str}"
        subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

    # Optionally remove the logged directories
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

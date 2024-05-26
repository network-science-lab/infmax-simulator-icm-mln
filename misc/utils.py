from dataclasses import dataclass, asdict
import datetime
import json
import os
import random
import shutil
import subprocess
import sys

from pathlib import Path

import numpy as np
import network_diffusion as nd
import pandas as pd


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class SimulationResult:
    actor_id: int
    simulation_length: int
    actors_infected: int
    actors_not_infected: int
    peak_infections_nb: int
    peak_iteration_nb: int


def extract_simulation_result(detailed_logs, net, actor):
    """Get length of diffusion, real number of seeds and final coverage."""
    # simulation_length = 0
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
            if actorwise_log["active_actors"] != 1: raise ArithmeticError(
                "Number of seeds must be 1 (got: " + actorwise_log["active_actors"] + ")"
            )
        else:
            if actors_infected_epoch < actors_infected_total:
                raise ArithmeticError(
                    f"Results contradict themselves! \
                    Number of active actors in {epoch_num} epoch: {actors_infected_epoch} \
                    number of all actors active so far: {actors_infected_total}"
                )
    
        # update peaks
        if actorwise_log["active_actors"] > peak_infections_nb:
            peak_infections_nb = actorwise_log["active_actors"]
            peak_iteration_nb = epoch_num + 1  # we don't start counting from 0 :)
        
        # # update real length of diffusion
        # if actors_infected_epoch != actors_infected_total:
        #     simulation_length = epoch_num + 1  # we don't start counting from 0 :)

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
    rr_dict_mean = pd.DataFrame(rr_dict_all).mean().round(2).to_dict()
    return SimulationResult(**rr_dict_mean)


def save_magrinal_efficiences(marginal_efficiencies: list[SimulationResult], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in marginal_efficiencies]
    pd.DataFrame(me_dict_all).to_csv(out_path)


def compute_gain(seeds_prct, coverage_prct):
    max_available_gain = 100 - seeds_prct
    obtained_gain = coverage_prct - seeds_prct
    return 100 * obtained_gain / max_available_gain


def block_prints():
    sys.stdout = open(os.devnull, 'w')


def enable_prints():
    sys.stdout = sys.__stdout__


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2):
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return round((t_2 - t_1).seconds / 60, 2)


def get_seed_selector(selector_name):
    if selector_name == "cbim":
        return nd.seeding.CBIMSeedselector
    elif selector_name == "cim":
        return nd.seeding.CIMSeedSelector
    elif selector_name == "degree_centrality":
        return nd.seeding.DegreeCentralitySelector
    elif selector_name == "degree_centrality_discount":
        return nd.seeding.DegreeCentralityDiscountSelector
    elif selector_name == "k_shell":
        return nd.seeding.KShellSeedSelector
    elif selector_name == "k_shell_mln":
        return nd.seeding.KShellMLNSeedSelector
    elif selector_name == "kpp_shell":
        return nd.seeding.KPPShellSeedSelector
    elif selector_name == "neighbourhood_size":
        return nd.seeding.NeighbourhoodSizeSelector
    elif selector_name == "neighbourhood_size_discount":
        return nd.seeding.NeighbourhoodSizeDiscountSelector
    elif selector_name == "page_rank":
        return nd.seeding.PageRankSeedSelector
    elif selector_name == "page_rank_mln":
        return nd.seeding.PageRankMLNSeedSelector
    elif selector_name == "random":
        return nd.seeding.RandomSeedSelector
    elif selector_name == "vote_rank":
        return nd.seeding.VoteRankSeedSelector
    elif selector_name == "vote_rank_mln":
        return nd.seeding.VoteRankMLNSeedSelector
    raise AttributeError(f"{selector_name} is not a valid seed selector name!")


class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, nd.MLNetworkActor):
                return obj.__dict__
            return super().default(obj)


def zip_detailed_logs(logged_dirs: list[Path], rm_logged_dirs: bool = True) -> None:
    # Ensure at least one directory is provided
    if len(logged_dirs) == 0:
        print("No directories provided to create archive from.")
        return
    
    # Get the parent directory of the first directory in logged_dirs
    parent_dir = logged_dirs[0].parent
    
    # Create the name for the zip file based on the parent directory
    zip_filename = parent_dir / "detailed_logs.zip"

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
        if node_log["new_state"] == "0":
            inactive_nodes.append(node_log["node_name"])
        if node_log["new_state"] == "1":
            active_nodes.append(node_log["node_name"])
        if node_log["new_state"] == "-1":
            activated_nodes.append(node_log["node_name"])
    actorwise_log = {
        "inactive_actors": len(set(inactive_nodes)),
        "active_actors": len(set(active_nodes)),
        "activated_actors": len(set(activated_nodes)),
    }
    assert actors_nb == sum([v for v in actorwise_log.values()])
    return actorwise_log

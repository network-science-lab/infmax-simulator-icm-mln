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


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_basic_stats(detailed_logs):
    """Get length of diffusion, real number of seeds and final coverage."""
    length_of_diffusion = 0
    active_actors_num = 0
    seed_actors_num = 0
    active_nodes_list = []

    # sort epochs indices
    epochs_sorted = sorted([int(e) for e in detailed_logs.keys()])

    # calculate metrics from each epoch
    for epoch_num in epochs_sorted:

        # obtain a list and number of active nodes in current epoch
        active_nodes_epoch = []
        for node in detailed_logs[epoch_num]:
            if node["new_state"] == "1":
                active_nodes_epoch.append(node["node_name"])
        active_actors_epoch_num = len(set(active_nodes_epoch))
        
        # update real length of diffusion
        if active_actors_epoch_num != len(set(active_nodes_list)):
            length_of_diffusion = epoch_num

        # update a list of nodes that were activated during entire experiment
        active_nodes_list.extend(active_nodes_epoch)

        if epoch_num == 0:
            # obtain a pcerise number of actors that were seeds
            seed_actors_num = active_actors_epoch_num
        else:
            # sanity check to detect leaks i.e. nodes cannot be deactivated
            if active_actors_epoch_num < len(set(active_nodes_list)):
                raise AttributeError(
                    f"Results contradict themselves! \
                    Number of active actors in {epoch_num} epoch: {active_actors_epoch_num} \
                    number of all actors active so far: {len(set(active_nodes_list))}"
                )

    # get number of actors that were active at the steaady state of diffusion
    active_actors_num = len(set(active_nodes_list))

    return length_of_diffusion, active_actors_num, seed_actors_num


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

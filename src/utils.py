import datetime
import random
import shutil

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import git
import network_diffusion as nd
import numpy as np
import pandas as pd
import torch

from _data_set.nsl_data_utils.loaders.net_loader import load_network


def get_recent_git_sha() -> str:
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    return git_sha


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2):
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return t_2 - t_1


def set_rng_seed(seed):
    """Fix seed of the random numbers generator for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def export_dataclasses(marginal_efficiencies: list[dataclass], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in marginal_efficiencies]
    pd.DataFrame(me_dict_all).to_csv(out_path, index=False)


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


# TODO: extract below
def compute_gain(seed_nb: int, exposed_nb: int, total_actors: int) -> float:
    max_available_gain = total_actors - seed_nb
    obtained_gain = exposed_nb - seed_nb
    return 100 * obtained_gain / max_available_gain


@dataclass(frozen=True)
class Network:
    name: str
    type: str # this field has been added after generation of the dataset with spreading potentials!
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch


def get_parameter_space(
    protocols: list[str], p_values: list[float], networks: list[str], as_tensor: bool
) -> product:
    nets = []
    for net_type in networks:
        print(f"Loading {net_type} network")
        for net_name, net_graph in load_network(net_name=net_type, as_tensor=as_tensor).items():
            nets.append(Network(name=net_name, type=net_type, graph=net_graph))
    return product(protocols, p_values, nets)

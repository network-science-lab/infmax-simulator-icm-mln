"""Helpers related with simulations."""

import re
import warnings
from dataclasses import dataclass
from itertools import product
from math import log10

import network_diffusion as nd
from tqdm import tqdm

from _data_set.nsl_data_utils.loaders.net_loader import load_network, load_net_names

warnings.filterwarnings(action="ignore", category=FutureWarning)


@dataclass(frozen=True)
class Network:
    n_type: str # this field has been added after generation of the dataset with spreading potentials!
    n_name: str
    n_graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch

    @classmethod
    def from_str(cls, n_type: str, n_name: str, as_tensor: bool) -> "Network":
        return cls(n_type, n_name, load_network(n_type, n_name, as_tensor=as_tensor))


def compute_gain(seed_nb: int, exposed_nb: int, total_actors: int) -> float:
    max_available_gain = total_actors - seed_nb
    obtained_gain = exposed_nb - seed_nb
    return 100 * obtained_gain / max_available_gain


def parse_network_config(type_name: str) -> tuple[str, list[str]]:
    """Obtain network name for given string from the configuraiton file."""
    pattern = r"^([^-]+)(?:-(.*))?$"
    match = re.match(pattern, type_name)
    if match.group(2) == "*":  # a wildcard - return all possible names
        return match.group(1), load_net_names(match.group(1))
    elif match.group(2) is None:  # no name provided, hence network's type is network's name 
        return match.group(1), [match.group(1)]
    return match.group(1), [match.group(2)]  # by default name is anything after dash


def get_parameter_space(
    protocols: list[str], p_values: list[float], networks: list[str], as_tensor: bool
) -> product:
    nets = []
    for net_type in networks:
        net_type, net_names = parse_network_config(net_type)
        print(f"Loading {net_type} networks")
        for net_name in tqdm(net_names):
            nets.append(Network.from_str(n_type=net_type, n_name=net_name, as_tensor=as_tensor))
    return product(protocols, p_values, nets)


def get_case_name_base(protocol: str, p: float, net_name: str) -> str:
    return f"proto-{protocol}--p-{round(p, 3)}--net-{net_name}"


def get_case_name_rich(
    protocol: str,
    p: float,
    net_name: str,
    case_idx: int,
    cases_nb: int,
    actor_idx: int,
    actors_nb: int,
    rep_idx: int,
    reps_nb: int
) -> str:
    return (
        f"case-{str(case_idx).zfill(int(log10(cases_nb)+1))}/{cases_nb}--" +
        f"actor-{str(actor_idx).zfill(int(log10(actors_nb)+1))}/{actors_nb}--" +
        f"repet-{str(rep_idx).zfill(int(log10(reps_nb)+1))}/{reps_nb}--" +
        get_case_name_base(protocol, p, net_name)
    )

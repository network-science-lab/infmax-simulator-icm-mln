"""Helpers related with simulations."""

import warnings

from dataclasses import dataclass
from itertools import product
from math import log10

import network_diffusion as nd
from _data_set.nsl_data_utils.loaders.net_loader import load_network

warnings.filterwarnings(action="ignore", category=FutureWarning)


@dataclass(frozen=True)
class Network:
    name: str
    type: str # this field has been added after generation of the dataset with spreading potentials!
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch


def compute_gain(seed_nb: int, exposed_nb: int, total_actors: int) -> float:
    max_available_gain = total_actors - seed_nb
    obtained_gain = exposed_nb - seed_nb
    return 100 * obtained_gain / max_available_gain


def get_parameter_space(
    protocols: list[str], p_values: list[float], networks: list[str], as_tensor: bool
) -> product:
    nets = []
    for net_type in networks:
        print(f"Loading {net_type} network")
        for net_name, net_graph in load_network(net_name=net_type, as_tensor=as_tensor).items():
            nets.append(Network(name=net_name, type=net_type, graph=net_graph))
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

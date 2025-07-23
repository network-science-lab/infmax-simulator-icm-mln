"""Helpers related with simulations."""

import re
import warnings
from dataclasses import dataclass
from itertools import product
from math import log10

import networkx as nx
import network_diffusion as nd
from bidict import bidict
from tqdm import tqdm

from tsds_utils.loaders.net_loader import load_network, load_net_names

warnings.filterwarnings(action="ignore", category=FutureWarning)


@dataclass(frozen=True)
class Network:
    n_type: str # this field has been added after generation of the dataset with spread. potentials!
    n_name: str
    n_graph_pt: nd.MultilayerNetworkTorch
    n_graph_nx: nd.MultilayerNetwork

    @classmethod
    def from_str(cls, n_type: str, n_name: str) -> "Network":
        graph_nx: nd.MultilayerNetwork = load_network(n_type, n_name, as_tensor=False)
        # uncomment in case of problems with dtypes
        # l_dict = {}
        # for l_name, l_graph in graph_nx.layers.items():
        #     l_dict[l_name] = nx.relabel_nodes(l_graph, {n: str(n) for n in l_graph.nodes})
        # graph_nx = nd.MultilayerNetwork(layers=l_dict)
        graph_pt = nd.MultilayerNetworkTorch.from_mln(graph_nx)
        graph_pt.actors_map = bidict(
            {str(a_id): a_idx for a_id, a_idx in graph_pt.actors_map.items()}
        )
        return cls(n_type, n_name, graph_pt, graph_nx)

    def __repr__(self) -> str:
        return "{0} at {1}; {2}, {3}, {4} at {5}, {6} at {7}".format(
            self.__class__.__name__,
            id(self),
            self.n_type,
            self.n_name,
            self.n_graph_pt.__class__.__name__,
            id(self.n_graph_pt),
            self.n_graph_nx.__class__.__name__,
            id(self.n_graph_nx),
        )


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


def get_parameter_space(protocols: list[str], p_values: list[float], networks: list[str]) -> product:
    nets = []
    for net_type in networks:
        net_type, net_names = parse_network_config(net_type)
        print(f"Loading {net_type} networks")
        for net_name in tqdm(net_names):
            nets.append(Network.from_str(n_type=net_type, n_name=net_name))
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

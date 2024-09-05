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
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch


def get_parameter_space(
    protocols: list[str], p_values: list[float], networks: list[str], as_tensor: bool
) -> product:
    nets = []
    for n in networks:
        print(f"Loading {n} network")
        nets.append(Network(n, load_network(net_name=n, as_tensor=as_tensor)))
    return product(protocols, p_values, nets)


def get_ranking(actor: nd.MLNetworkActor, actors: list[nd.MLNetworkActor]) -> nd.seeding.MockingActorSelector:
    ranking_list = [actor, *set(actors).difference({actor})]
    return nd.seeding.MockingActorSelector(ranking_list)


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

"""Ground Truth seed selector."""

from random import shuffle
from typing import Literal

import network_diffusion as nd
import pandas as pd
from pathlib import Path

from _data_set.nsl_data_utils.loaders.constants import (
    ACTOR, AND, EXPOSED, NETWORK, OR, P, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, SIMULATION_LENGTH
)
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp_paths, load_sp
from _data_set.nsl_data_utils.loaders.centrality_loader import load_centralities_path, load_centralities

class DFChoice:
    def __init__(self, result_dir: str) -> None:
        self._result_dir = result_dir

    def __call__(self, net_type: str, net_name: str, **kwargs) -> list[str]:
        csv_path = Path(f'{self._result_dir}/{net_type}_{net_name}.csv')
        if not csv_path.exists():
            raise ValueError(f'There is no {net_type}_{net_name}.csv in {self._result_dir}')
        result = pd.read_csv(csv_path, index_col=0)
        return [str(i) for i in result.index.tolist()]


class CentralityChoice:

    centralities = [
        "degree",
        "betweenness",
        "closeness",
        "core_number",
        "neighbourhood_size",
        "voterank",
    ]

    def __init__(self, nb_seeds: int, centrality_name: str):
        self.nb_seeds = nb_seeds
        self.centrality_name = centrality_name
        if self.centrality_name not in self.centralities:
            raise ValueError("Unknown centrality name {self.centrality_name}!")
    
    def __call__(self, net_type: str, net_name: str, **kwargs) -> list[str]:
        centr_dir = load_centralities_path(network_type=net_type, network_name=net_name)
        centrs_df = load_centralities(csv_path=centr_dir)
        centr_df = centrs_df[self.centrality_name].sort_values(ascending=False)
        return list(centr_df[:self.nb_seeds].index)


class GroundTruth:

    _VALID_ICM_PARAMS = {AND: {0.80, 0.85, 0.90, 0.95}, OR: {0.05, 0.10, 0.15, 0.20}}

    def __init__(self, nb_seeds: int, average_protocol: bool, average_p_value: bool) -> None:
        self.nb_seeds = nb_seeds
        self.average_protocol = average_protocol
        self.average_p_value = average_p_value
    
    @staticmethod
    def get_top_k(
        sp_raw: pd.DataFrame,
        protocol: list[str],
        p: list[float],
        budget: int
    ) -> list[str]:
        """
        Get actors that performed the best in given spreading contitions.

        :param sp_raw: DataFrame loaded with `load_sp` with spreading potentials for a given network
        :param protocol: a list of protocols to consider 
        :param p: a list of probabilities to consider 
        :param budget: top-k actors to return, if not provided it will be discarded in the process
            of selecting top-k actors
        :return: IDs of actors that performed the best in given contidions
        """
        sp_filtered = sp_raw[sp_raw[PROTOCOL].isin(protocol) & sp_raw[P].isin(p)]
        sp_filtered = sp_filtered.drop([P, PROTOCOL, NETWORK], axis=1)
        sp_mean = sp_filtered.groupby(by=[ACTOR]).mean()
        sp_mean = sp_mean.sort_values(
            [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
            ascending=[False, True, True, False]
        )
        return sp_mean.iloc[:budget].index.tolist()

    def __call__(
            self,
            net_type: str,
            net_name: str,
            protocol: Literal["OR", "AND"],
            p: float,
            **kwargs,
    ) -> list[str]:
        assert p in self._VALID_ICM_PARAMS[protocol], "Eval. feasible only on narrow range of p!"
        sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
        raw_sp = load_sp(csv_paths=sp_paths)
        protocol = [OR, AND] if self.average_protocol else [protocol]
        p = [p_val for proto in protocol for p_val in self._VALID_ICM_PARAMS[proto]] \
            if self.average_p_value else [p]
        return self.get_top_k(sp_raw=raw_sp, budget=self.nb_seeds, protocol=protocol, p=p)


class RandomChoice:

    def __init__(self, nb_seeds: int) -> None:
        self.nb_seeds = nb_seeds
    
    def __call__(self, network: nd.MultilayerNetworkTorch, **kwargs) -> list[str]:
        actors = list(network.actors_map.keys())
        shuffle(actors)
        return actors[:self.nb_seeds]

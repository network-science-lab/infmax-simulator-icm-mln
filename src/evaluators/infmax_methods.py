"""Ground Truth seed selector."""

from abc import ABC
from dataclasses import dataclass
from random import shuffle
from typing import Literal

import network_diffusion as nd
import pandas as pd

from _data_set.nsl_data_utils.loaders.constants import (
    ACTOR, AND, EXPOSED, NETWORK, OR, P, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, SIMULATION_LENGTH
)
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp_paths, load_sp
from _data_set.nsl_data_utils.loaders.centrality_loader import load_centralities_path, load_centralities


class BaseChoice(ABC):

    is_stochastic: bool = None


# class DFChoice(BaseChoice):

#     def __init__(self, result_dir: str) -> None:
#         self._result_dir = result_dir

#     def __call__(self, net_type: str, net_name: str, **kwargs) -> list[str]:
#         csv_path = Path(f'{self._result_dir}/{net_type}_{net_name}.csv')
#         if not csv_path.exists():
#             raise ValueError(f'There is no {net_type}_{net_name}.csv in {self._result_dir}')
#         result = pd.read_csv(csv_path, index_col=0)
#         return [str(i) for i in result.index.tolist()]


class CentralityChoice(BaseChoice):

    centralities = [
        "degree",
        "betweenness",
        "closeness",
        "core_number",
        "neighbourhood_size",
        "voterank",
    ]
    is_stochastic = False

    def __init__(self, centrality_name: str):
        self.centrality_name = centrality_name
        if self.centrality_name not in self.centralities:
            raise ValueError("Unknown centrality name {self.centrality_name}!")
    
    def __call__(self, net_type: str, net_name: str, nb_seeds: int, **kwargs) -> list[str]:
        centr_dir = load_centralities_path(network_type=net_type, network_name=net_name)
        centrs_df = load_centralities(csv_path=centr_dir)
        centr_df = centrs_df[self.centrality_name].sort_values(ascending=False)
        return list(centr_df[:nb_seeds].index)


@dataclass
class SPScore:
    """A simple class to compute Spreading Potential Score."""

    exposed_weight: int
    simulation_length_weight: int
    peak_infected_weight: int
    peak_iteration_weight: int

    def __call__(self, sp: pd.DataFrame) -> pd.Series:
        for col in  [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION]:
            sp[col] /= sp[col].max()
        sp["score"] = (
            sp[EXPOSED] * self.exposed_weight +  # maximise
            (1 - sp[SIMULATION_LENGTH]) * self.simulation_length_weight +  # minimise
            sp[PEAK_INFECTED] * self.peak_infected_weight  +  # maximise
            (1 - sp[PEAK_ITERATION]) * self.peak_iteration_weight  # minimise
        )
        return sp.sort_values(by="score", ascending=False)["score"]


class GroundTruth(BaseChoice):

    is_stochastic = False
    valid_icm_params = {
        AND: {0.80, 0.85, 0.90, 0.95},
        OR: {0.05, 0.10, 0.15, 0.20},
        "WILDCARDS": {-1.},
    }

    @staticmethod
    def get_top_k(
        sp_raw: pd.DataFrame,
        protocol: list[str],
        p: list[float],
        nb_seeds: int,
        weights: dict[str, int],
    ) -> list[str]:
        """
        Get actors that performed the best in given spreading contitions.

        :param sp_raw: DataFrame loaded with `load_sp` with spreading potentials for a given network
        :param protocol: a list of protocols to consider 
        :param p: a list of probabilities to consider 
        :param nb_seeds: top-k actors to return
        :return: IDs of actors that performed the best in given contidions
        """
        sp_filtered = sp_raw[sp_raw[PROTOCOL].isin([protocol]) & sp_raw[P].isin(p)]
        sp_filtered = sp_filtered.drop([P, PROTOCOL, NETWORK], axis=1)
        sp_mean = sp_filtered.groupby(by=[ACTOR]).mean()
        sp_score = SPScore(
            exposed_weight=weights[EXPOSED],
            simulation_length_weight=weights[SIMULATION_LENGTH],
            peak_infected_weight=weights[PEAK_INFECTED],
            peak_iteration_weight=weights[PEAK_ITERATION],
        )(sp_mean)
        return sp_score.iloc[:nb_seeds].index.tolist()

    def __call__(
            self,
            net_type: str,
            net_name: str,
            protocol: Literal["OR", "AND"],
            p: float,  # -1 is a wildcard to get avg SP for all feasible p under a given protocol
            nb_seeds: int,
            weights: dict[str, int],
            **kwargs,
    ) -> list[str]:
        assert p in self.valid_icm_params[protocol] or p in self.valid_icm_params["WILDCARDS"], \
            "Evaluation is feasible only for a narrow range of p!"
        sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
        raw_sp = load_sp(csv_paths=sp_paths)
        p = [p_val for p_val in self.valid_icm_params[protocol]] if p == -1. else [p]
        return self.get_top_k(sp_raw=raw_sp, nb_seeds=nb_seeds, protocol=protocol, p=p)


class RandomChoice(BaseChoice):

    is_stochastic = True
    
    def __call__(self, network: nd.MultilayerNetworkTorch, nb_seeds: int, **kwargs) -> list[str]:
        actors = list(network.actors_map.keys())
        shuffle(actors)
        return actors[:nb_seeds]

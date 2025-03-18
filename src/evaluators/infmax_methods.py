"""Classess aimed to select seeds for further evaluation."""

import os
import tempfile
from abc import ABC
from random import shuffle
from typing import Literal

import neptune
import network_diffusion as nd
import pandas as pd

from _data_set.nsl_data_utils.loaders.constants import (
    ACTOR, AND, EXPOSED, NETWORK, OR, P, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, SIMULATION_LENGTH
)
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp_paths, load_sp
from _data_set.nsl_data_utils.loaders.centrality_loader import load_centralities_path, load_centralities
from src.evaluators.utils import SPScore


class BaseChoice(ABC):

    is_stochastic: bool = None


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


class GroundTruth(BaseChoice):

    is_stochastic = False
    valid_icm_params = {
        AND: {0.80, 0.85, 0.90, 0.95},
        OR: {0.05, 0.10, 0.15, 0.20},
        "WILDCARDS": {-1.},
    }

    def __init__(
        self, 
        exposed_weight: int,
        simulation_length_weight: int,
        peak_infected_weight: int,
        peak_iteration_weight: int,
    ) -> None:
        self.weights = {
            EXPOSED: exposed_weight,
            SIMULATION_LENGTH: simulation_length_weight,
            PEAK_INFECTED: peak_infected_weight,
            PEAK_ITERATION: peak_iteration_weight,
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
            **kwargs,
    ) -> list[str]:
        assert p in self.valid_icm_params[protocol] or p in self.valid_icm_params["WILDCARDS"], \
            "Evaluation is feasible only for a narrow range of p!"
        sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
        raw_sp = load_sp(csv_paths=sp_paths)
        p = [p_val for p_val in self.valid_icm_params[protocol]] if p == -1. else [p]
        return self.get_top_k(
            sp_raw=raw_sp,
            nb_seeds=nb_seeds,
            protocol=protocol,
            p=p,
            weights=self.weights,
        )


class RandomChoice(BaseChoice):

    is_stochastic = True
    
    def __call__(self, network: nd.MultilayerNetworkTorch, nb_seeds: int, **kwargs) -> list[str]:
        actors = list(network.actors_map.keys())
        shuffle(actors)
        return actors[:nb_seeds]


class NeptuneDownloader(BaseChoice):
    """A class to fetch rankings from neptune.ai."""

    is_stochastic = False

    def __init__(
        self,
        project: str,
        run: str,
        exposed_weight: int,
        simulation_length_weight: int,
        peak_infected_weight: int,
        peak_iteration_weight: int,
    ) -> None:
        self.session = neptune.init_run(
            api_token=os.getenv(key="NEPTUNE_API_KEY", default=neptune.ANONYMOUS_API_TOKEN),
            project=project,
            with_id=run,
        )
        self.weights = {
            EXPOSED: exposed_weight,
            SIMULATION_LENGTH: simulation_length_weight,
            PEAK_INFECTED: peak_infected_weight,
            PEAK_ITERATION: peak_iteration_weight,
        }

    @staticmethod
    def _load_csv(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, index_col=0)
        df.index.name = ACTOR
        return df

    def __call__(self, net_type: str, net_name: str, nb_seeds: int, **kwargs) -> list[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = f"{temp_dir}/predicted_sp.csv"
            self.session[f"evaluation/{net_type}/{net_name}"].download(destination=temp_path)
            df = self._load_csv(csv_path=temp_path)
        sp_score = SPScore(
            exposed_weight=self.weights[EXPOSED],
            simulation_length_weight=self.weights[SIMULATION_LENGTH],
            peak_infected_weight=self.weights[PEAK_INFECTED],
            peak_iteration_weight=self.weights[PEAK_ITERATION],
        )(df)
        return sp_score.iloc[:nb_seeds].index.tolist()

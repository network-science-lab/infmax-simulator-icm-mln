"""Classess aimed to get centrailties for further regression."""

from functools import lru_cache
import os
import tempfile
from abc import ABC, abstractmethod
from random import shuffle
from typing import Literal

import neptune
import network_diffusion as nd
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nsl_data_utils.loaders.constants import (
    ACTOR, AND, EXPOSED, NETWORK, OR, P, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, SIMULATION_LENGTH
)
from nsl_data_utils.loaders.sp_loader import load_sp_paths, load_sp
from nsl_data_utils.loaders.centrality_loader import load_centralities_path, load_centralities
from src.evaluators.utils import SPScore


class CachedCentralityRegressor:

    _centralities = {
        "degree",
        "betweenness",
        "closeness",
        "core_number",
        "neighbourhood_size",
        "voterank",
    }

    _valid_icm_params = {
        AND: {0.80, 0.85, 0.90, 0.95},
        OR: {0.05, 0.10, 0.15, 0.20},
        "WILDCARDS": {-1.},
    }

    def __init__(self, centrality_names: list[str], rng_seed: int, nb_repetitions: int) -> None:
        self.centrality_names = centrality_names
        self.rng_seed = rng_seed
        self.nb_repetitions = nb_repetitions
        if len(self._centralities.intersection(set(self.centrality_names))) == 0:
            raise ValueError("Unknown centrality name!")
    
    # @lru_cache
    @staticmethod
    def get_features(net_type: str, net_name: str, features: list[str]) -> pd.DataFrame:
        centr_paths = load_centralities_path(network_type=net_type, network_name=net_name)
        centr_raw = load_centralities(csv_path=centr_paths)
        centr_final = centr_raw[features]
        return centr_final
    
    # @lru_cache
    @staticmethod
    def get_gt(net_type: str, net_name: str, protocol: str, p: list[float]) -> pd.DataFrame:
        sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
        sp_raw = load_sp(csv_paths=sp_paths)
        sp_filtered = sp_raw[sp_raw[PROTOCOL].isin([protocol]) & sp_raw[P].isin(p)]
        sp_final = sp_filtered.drop(
            [P, PROTOCOL, NETWORK, "not_exposed"], axis=1
        ).groupby(by=[ACTOR]).mean()
        return sp_final
    
    @staticmethod
    def regress(x: np.ndarray, y: np.ndarray, rng: int) -> tuple[float, float]:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rng)
        model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return rmse, r2
    
    def __call__(
        self,
        net_type: str,
        net_name: str,
        protocol: Literal["OR", "AND"],
        p: float,  # -1 is a wildcard to get avg SP for all feasible p under a given protocol
    ) -> list[str]:
        assert p in self._valid_icm_params[protocol] or p in self._valid_icm_params["WILDCARDS"], \
            "Evaluation is feasible only for a narrow range of p!"

        # get input/output data
        x_raw = self.get_features(net_type, net_name, self.centrality_names)
        y_raw = self.get_gt(
            net_type,
            net_name,
            protocol,
            [_p for _p in self._valid_icm_params[protocol]] if p == -1. else [p],
        )
        xy_arr = pd.concat([x_raw, y_raw], axis=1).to_numpy()

        # now test the model
        rmses, r2s = [], []
        for _ in range(self.nb_repetitions):

            # shuffle input data
            _xy_arr = xy_arr[np.random.permutation(xy_arr.shape[0])]
            x_arr = _xy_arr[:, :-4]
            y_arr = _xy_arr[:, -4:]

            # perform regression and save results
            rmse, r2 = self.regress(x_arr, y_arr, self.rng_seed)
            rmses.append(rmse)
            r2s.append(r2)
        
        # return mean and std of these two metrics
        rmses = np.array(rmses)
        r2s = np.array(r2s)
        return {
            "rmse_avg": rmses.mean(),
            "rmse_std": rmses.std(),
            "r2_avg": r2s.mean(),
            "r2_std": r2s.std(),
        }


# class GroundTruth(BaseChoice):

#     is_stochastic = False
#     valid_icm_params = {
#         AND: {0.80, 0.85, 0.90, 0.95},
#         OR: {0.05, 0.10, 0.15, 0.20},
#         "WILDCARDS": {-1.},
#     }

#     def __init__(
#         self, 
#         exposed_weight: int,
#         simulation_length_weight: int,
#         peak_infected_weight: int,
#         peak_iteration_weight: int,
#     ) -> None:
#         self.weights = {
#             EXPOSED: exposed_weight,
#             SIMULATION_LENGTH: simulation_length_weight,
#             PEAK_INFECTED: peak_infected_weight,
#             PEAK_ITERATION: peak_iteration_weight,
#         }

#     @staticmethod
#     def get_sp_score(
#         sp_raw: pd.DataFrame,
#         protocol: list[str],
#         p: list[float],
#         weights: dict[str, int],
#     ) -> pd.Series:
#         """
#         Get actors that performed the best in given spreading contitions.

#         :param sp_raw: DataFrame loaded with `load_sp` with spreading potentials for a given network
#         :param protocol: a list of protocols to consider 
#         :param p: a list of probabilities to consider 
#         :param nb_seeds: top-k actors to return
#         :return: IDs of actors that performed the best in given contidions
#         """
#         sp_filtered = sp_raw[sp_raw[PROTOCOL].isin([protocol]) & sp_raw[P].isin(p)]
#         sp_filtered = sp_filtered.drop([P, PROTOCOL, NETWORK], axis=1)
#         sp_mean = sp_filtered.groupby(by=[ACTOR]).mean()
#         return SPScore(
#             exposed_weight=weights[EXPOSED],
#             simulation_length_weight=weights[SIMULATION_LENGTH],
#             peak_infected_weight=weights[PEAK_INFECTED],
#             peak_iteration_weight=weights[PEAK_ITERATION],
#         )(sp_mean)

#     def __call__(
#             self,
#             net_type: str,
#             net_name: str,
#             protocol: Literal["OR", "AND"],
#             p: float,  # -1 is a wildcard to get avg SP for all feasible p under a given protocol
#             nb_seeds: int | None,
#             **kwargs,
#     ) -> list[str] | pd.Series:
#         assert p in self.valid_icm_params[protocol] or p in self.valid_icm_params["WILDCARDS"], \
#             "Evaluation is feasible only for a narrow range of p!"
#         sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
#         raw_sp = load_sp(csv_paths=sp_paths)
#         p = [p_val for p_val in self.valid_icm_params[protocol]] if p == -1. else [p]
#         sp_score = self.get_sp_score(
#             sp_raw=raw_sp,
#             protocol=protocol,
#             p=p,
#             weights=self.weights,
#         )
#         if not nb_seeds:
#             return sp_score
#         return sp_score.iloc[:nb_seeds].index.tolist()


class NeptuneRegressor:
    """A class to fetch rankings from neptune.ai."""

    is_stochastic = False

    # def __init__(
    #     self,
    #     project: str,
    #     run: str,
    #     exposed_weight: int,
    #     simulation_length_weight: int,
    #     peak_infected_weight: int,
    #     peak_iteration_weight: int,
    # ) -> None:
    #     self.session = neptune.init_run(
    #         api_token=os.getenv(key="NEPTUNE_API_KEY", default=neptune.ANONYMOUS_API_TOKEN),
    #         project=project,
    #         with_id=run,
    #     )
    #     self.weights = {
    #         EXPOSED: exposed_weight,
    #         SIMULATION_LENGTH: simulation_length_weight,
    #         PEAK_INFECTED: peak_infected_weight,
    #         PEAK_ITERATION: peak_iteration_weight,
    #     }

    # @staticmethod
    # def _load_csv(csv_path: str) -> pd.DataFrame:
    #     df = pd.read_csv(csv_path, index_col=0)
    #     df.index.name = ACTOR
    #     df.index = df.index.astype(str)
    #     return df

    # def __call__(self, net_type: str, net_name: str, nb_seeds: int, **kwargs) -> list[str]:
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         temp_path = f"{temp_dir}/predicted_sp.csv"
    #         self.session[f"evaluation/{net_type}/{net_name}"].download(destination=temp_path)
    #         df = self._load_csv(csv_path=temp_path)
    #     sp_score = SPScore(
    #         exposed_weight=self.weights[EXPOSED],
    #         simulation_length_weight=self.weights[SIMULATION_LENGTH],
    #         peak_infected_weight=self.weights[PEAK_INFECTED],
    #         peak_iteration_weight=self.weights[PEAK_ITERATION],
    #     )(df)
    #     return sp_score.iloc[:nb_seeds].index.tolist()

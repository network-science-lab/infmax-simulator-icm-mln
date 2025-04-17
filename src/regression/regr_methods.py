"""Classess aimed to get centrailties for further regression."""

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Literal

import neptune
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nsl_data_utils.loaders.constants import (
    ACTOR, AND, EXPOSED, NETWORK, OR, P, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, SIMULATION_LENGTH
)
from nsl_data_utils.loaders.sp_loader import load_sp_paths, load_sp
from nsl_data_utils.loaders.centrality_loader import load_centralities_path, load_centralities


class BaseRegressor(ABC):

    _valid_icm_params = {
        AND: {0.80, 0.85, 0.90, 0.95},
        OR: {0.05, 0.10, 0.15, 0.20},
        "WILDCARDS": {-1.},
    }

    _sp_order = [EXPOSED, SIMULATION_LENGTH, PEAK_ITERATION, PEAK_INFECTED]

    def _validate_p(self, protocol: str, p: float) -> list[float]:
        assert p in self._valid_icm_params[protocol] or p in self._valid_icm_params["WILDCARDS"], \
        "Evaluation is feasible only for a narrow range of p!"
        return [_p for _p in self._valid_icm_params[protocol]] if p == -1. else [p]

    def get_gt(self, net_type: str, net_name: str, protocol: str, p: list[float]) -> pd.DataFrame:
        sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
        sp_raw = load_sp(csv_paths=sp_paths)
        sp_filtered = sp_raw[sp_raw[PROTOCOL].isin([protocol]) & sp_raw[P].isin(p)]
        sp_final = sp_filtered.drop(
            [P, PROTOCOL, NETWORK, "not_exposed"], axis=1
        ).groupby(by=[ACTOR]).mean()[self._sp_order]
        return sp_final / sp_final.max()

    @abstractmethod
    def __call__(
        self,
        net_type: str,
        net_name: str,
        protocol: Literal["OR", "AND"],
        p: float,
    ) -> list[str]:
        ...


class CachedCentralityRegressor(BaseRegressor):

    _centralities = {
        "degree",
        "betweenness",
        "closeness",
        "core_number",
        "neighbourhood_size",
        "voterank",
    }

    def __init__(self, centrality_names: list[str], rng_seed: int, nb_repetitions: int) -> None:
        self.centrality_names = centrality_names
        self.rng_seed = rng_seed
        self.shuffle_rng = np.random.RandomState(seed=rng_seed)
        self.nb_repetitions = nb_repetitions
        if len(self._centralities.intersection(set(self.centrality_names))) == 0:
            raise ValueError("Unknown centrality name!")
    
    @staticmethod
    def get_features(net_type: str, net_name: str, features: list[str]) -> pd.DataFrame:
        centr_paths = load_centralities_path(network_type=net_type, network_name=net_name)
        centr_raw = load_centralities(csv_path=centr_paths)
        centr_final = centr_raw[features]
        return centr_final / centr_final.max()
    
    @staticmethod
    def regress(x: np.ndarray, y: np.ndarray, rng: int) -> tuple[float, float]:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rng)
        model = LinearRegression()
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
        # get input/output data
        x_raw = self.get_features(net_type, net_name, self.centrality_names)
        y_raw = self.get_gt(net_type, net_name, protocol, self._validate_p(protocol, p))
        xy_arr = pd.concat([x_raw, y_raw], axis=1).to_numpy()

        # now test the model
        rmses, r2s = [], []
        for _ in range(self.nb_repetitions):

            # shuffle input data
            _xy_arr = xy_arr[self.shuffle_rng.permutation(xy_arr.shape[0])]
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


class NeptuneRegressor(BaseRegressor):

    def __init__(self, project: str, run: str, rng_seed: int, nb_repetitions: int) -> None:
        self.rng_seed = rng_seed
        self.shuffle_rng = np.random.RandomState(seed=rng_seed)
        self.nb_repetitions = nb_repetitions
        self.session = neptune.init_run(
            api_token=os.getenv(key="NEPTUNE_API_KEY", default=neptune.ANONYMOUS_API_TOKEN),
            project=project,
            with_id=run,
        )

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, index_col=0)
        df.index.name = ACTOR
        df.index = df.index.astype(str)
        df = df[self._sp_order]
        return df / df.max()

    def __call__(
        self,
        net_type: str,
        net_name: str,
        protocol: Literal["OR", "AND"],
        p: float,
    ) -> list[str]:
        # get input/output data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = f"{temp_dir}/predicted_sp.csv"
            self.session[f"evaluation/{net_type}/{net_name}"].download(destination=temp_path)
            y_pred = self._load_csv(csv_path=temp_path)
        y_gt = self.get_gt(net_type, net_name, protocol, self._validate_p(protocol, p))

        # align two dataframes and convert to numpy
        ygt_ypred = pd.concat([y_gt, y_pred], axis=1).to_numpy()

        # now test the model
        rmses, r2s = [], []
        for _ in range(self.nb_repetitions):

            # shuffle input data and split to reflect conditions as from CachedCentralityRegressor
            _ygt_ypred = ygt_ypred[self.shuffle_rng.permutation(ygt_ypred.shape[0])]
            _, ygt_ypred_test = train_test_split(_ygt_ypred, test_size=0.2, random_state=self.rng_seed)

            # check the quality of the model
            y_gt_test = ygt_ypred_test[:, :4]
            y_pred_test = ygt_ypred_test[:, 4:]
            rmse = root_mean_squared_error(y_gt_test, y_pred_test)
            r2 = r2_score(y_gt_test, y_pred_test)
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

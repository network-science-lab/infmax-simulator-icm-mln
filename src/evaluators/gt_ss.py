"""Ground Truth seed selector."""

from typing import Literal

import pandas as pd

from _data_set.nsl_data_utils.loaders.constants import (
    ACTOR, EXPOSED, NETWORK, P, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, SIMULATION_LENGTH
)
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp, _get_sp, _sort_csv_paths


class GroundTruth:

    def __init__(self, nb_seeds: int, average_protocol: bool, average_p_value: bool) -> None:
        self.nb_seeds = nb_seeds
        self.average_protocol = average_protocol
        self.average_p_value = average_p_value
    
    @staticmethod
    def get_top_k(
        sp_raw: pd.DataFrame,
        protocol: str | None,
        p: float | None,
        budget: int
    ) -> list[str]:
        """
        Get actors that performed the best in given spreading contitions.

        :param sp_raw: DataFrame loaded with `load_sp` with spreading potentials for a given network
        :param protocol: protocol of the multilayer ICM
        :param p: probability of the multilayer ICM, if not provided it will be discarded in the
            process of selecting top-k actors
        :param budget: top-k actors to return, if not provided it will be discarded in the process
            of selecting top-k actors
        :return: IDs of actors that performed the best in given contidions
        """
        if not protocol:
            protocol = 0
            sp_raw[PROTOCOL] = protocol
        if not p:
            p = 0
            sp_raw[P] = p
        sp_mean  = sp_raw.groupby(by=[NETWORK, PROTOCOL, ACTOR, P]).mean().reset_index()
        sp_mean = sp_mean[(sp_mean[PROTOCOL] == protocol) & (sp_mean[P] == p)]
        sp_mean = sp_mean.sort_values(
            [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
            ascending=[False, True, True, False]
        )
        return sp_mean.iloc[:budget][ACTOR].tolist()

    def __call__(
            self,
            net_type: str,
            net_name: str,
            protocol: Literal["OR", "AND"],
            p: float,
            **kwargs,
    ) -> list[str]:
        if net_type == net_name:
            raw_sp = load_sp(net_name=net_type)[net_name]
        else:
            csv_paths = _sort_csv_paths(f"{net_type}/*.csv")
            raw_sp = _get_sp(csv_paths[net_name])
        protocol = None if self.average_protocol else protocol
        p = None if self.average_p_value else p
        return self.get_top_k(sp_raw=raw_sp, budget=self.nb_seeds, protocol=protocol, p=p)
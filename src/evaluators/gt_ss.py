"""Ground Truth seed selector."""

from typing import Any, Literal

import network_diffusion as nd

from _data_set.nsl_data_utils.loaders.sp_loader import load_sp, get_gt_data


class GroundTruth:

    def __init__(
        self,
        config_icm: dict[str, Any],
        nb_seeds: int,
        average_protocol: bool,
        average_p_value: bool,
    ) -> None:
        self.config_icm = config_icm
        self.nb_seeds = nb_seeds
        self.average_protocol = average_protocol
        self.average_p_value = average_p_value

    def __call__(self, name: str, protocol: Literal["OR", "AND"], p: float, **kwargs):
        raw_sp = load_sp(net_name=name)
        print("aaa")

"""Functions to help in evaluating performance of a given seed set."""
import sys
sys.path.append("/Users/michal/Development/infmax-simulator-icm-mln")

from typing import Any, Literal

import network_diffusion as nd
import pandas as pd
from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import get_gt_data, load_sp

from src.icm.torch_model import TorchMICModel, TorchMICSimulator
from src.generators import commons
from src.generators.utils import (
    mean_repeated_results,
    save_magrinal_efficiences,
    SimulationResult,
)


def evaluation_step(
    protocol: Literal["OR", "AND"],
    p: float,
    net: nd.MultilayerNetworkTorch,
    seed_set: list[Any],
    repetitions_nb: int,
    average_results: bool,
) -> pd.DataFrame:
    """Run multilayer ICM on given seed set and model's parameters."""
    micm = TorchMICModel(protocol=protocol, probability=p)
    repeated_results: list[SimulationResult] = []

    for _ in range(repetitions_nb):
        simulator = TorchMICSimulator(
            model=micm,
            net=net,
            n_steps=len(net.actors_map) * 2,
            seed_set=seed_set,
            debug=True,
        )
        logs = simulator.perform_propagation()
        simulation_result = SimulationResult(actor=",".join(seed_set), **logs)
        repeated_results.append(simulation_result)
    
    if average_results:
        repeated_results = [mean_repeated_results(repeated_results)]
    return pd.DataFrame(repeated_results)


if __name__ == "__main__":

    from _data_set.nsl_data_utils.loaders.constants import *

    net_name = LAZEGA
    proto = OR
    p = 0.25
    n_steps = 10000
    budget = 5
    n_repetitions = 30

    net = load_network(net_name, as_tensor=True)
    sp = load_sp(net_name)
    seed_set = get_gt_data(sp[net_name], proto, p, budget)
    raw_results = evaluation_step(net[net_name], seed_set, proto, p, n_steps, n_repetitions)
    print("Performance of given seed set:\n", raw_results.mean())

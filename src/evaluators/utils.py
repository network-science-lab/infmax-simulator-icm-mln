"""Evaluator's utilities."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import time
import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from nsl_data_utils.loaders.constants import (
    EXPOSED, PEAK_INFECTED, PEAK_ITERATION, SIMULATION_LENGTH
)


@dataclass(frozen=True)
class EvaluationResult:
    infmax_model: str # name of the model used in the evaluation
    seed_set: str  # IDs of actors that were seeds aggr. into string (sep. by ;)
    gain: float # gain obtained using this seed set
    simulation_length: int  # nb. of simulation steps
    exposed: int  # nb. of active actors at the end of the simulation
    not_exposed: int  # nb. of actors that remained inactive
    peak_infected: int  # maximal nb. of infected actors in a single sim. step
    peak_iteration: int  # a sim. step when the peak occured
    expositions_rec: str  # record of new activations aggr. into string (sep. by ;)


def mean_evaluation_results(repeated_results: list[EvaluationResult]) -> EvaluationResult:
    rr_dict_all = [asdict(rr) for rr in repeated_results]
    rr_df_all = pd.DataFrame(rr_dict_all)
    rr_dict_mean = rr_df_all.drop("expositions_rec", axis=1).groupby(
        ["infmax_model", "seed_set"]
    ).mean().reset_index().iloc[0].round(3).to_dict()
    exp_recs_list = rr_df_all["expositions_rec"].map(lambda x: [int(xx) for xx in x.split(";")])
    exp_recs_padded = np.zeros([len(exp_recs_list), max([len(er) for er in exp_recs_list])])
    for run_idx, step_idx in enumerate(exp_recs_list):
        exp_recs_padded[run_idx][0:len(step_idx)] = step_idx
    exp_recs_mean = np.mean(exp_recs_padded, axis=0).round(3)
    rr_dict_mean["expositions_rec"] = ";".join(exp_recs_mean.astype(str).tolist())
    return EvaluationResult(**rr_dict_mean)


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
        sp = sp.fillna(value=0)
        sp["score"] = (
            sp[EXPOSED] * self.exposed_weight +  # maximise
            (1 - sp[SIMULATION_LENGTH]) * self.simulation_length_weight +  # minimise
            sp[PEAK_INFECTED] * self.peak_infected_weight  +  # maximise
            (1 - sp[PEAK_ITERATION]) * self.peak_iteration_weight  # minimise
        )
        return sp.sort_index().sort_values(by="score", ascending=False)["score"]


def safe_serialize(value: Any) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def log_function_details(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        log_folder = Path('logs')
        log_folder.mkdir(exist_ok=True)

        current_time = datetime.now().strftime('%Y-%m-%d')
        log_filename = log_folder / f'{current_time}.log'

        args_serialized = [safe_serialize(arg) for arg in args]
        kwargs_serialized = {key: safe_serialize(value) for key, value in kwargs.items()}
        kwargs_json = json.dumps(kwargs_serialized, indent=4)

        with log_filename.open('a', encoding='utf-8') as log_file:
            log_file.write(f"Function: {func.__name__}\n")
            log_file.write(f"Arguments: args={args_serialized}\n")
            if kwargs:
                log_file.write(f"Keyword Arguments: {kwargs_json}\n")
            log_file.write(f"Execution time: {execution_time:.4f} seconds\n")
            log_file.write("-" * 50 + "\n")

        return result
    return wrapper

"""blabla."""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


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
        ["infmax_model", "seed_ids"]
    ).mean().reset_index().iloc[0].round(3).to_dict()
    exp_recs_list = rr_df_all["expositions_rec"].map(lambda x: [int(xx) for xx in x.split(";")])
    exp_recs_padded = np.zeros([len(exp_recs_list), max([len(er) for er in exp_recs_list])])
    for run_idx, step_idx in enumerate(exp_recs_list):
        exp_recs_padded[run_idx][0:len(step_idx)] = step_idx
    exp_recs_mean = np.mean(exp_recs_padded, axis=0).round(3)
    rr_dict_mean["expositions_rec"] = ";".join(exp_recs_mean.astype(str).tolist())
    return EvaluationResult(**rr_dict_mean)

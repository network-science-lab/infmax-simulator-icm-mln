"""A script to plot IoU of seed sets constructed from full (i.e. comprising of all actors) ranks."""

from functools import lru_cache
from pathlib import Path
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from scripts.analysis.iou_ranking_plots import load_json_data, plot_accs, GTResults
from src.evaluators.infmax_methods import GroundTruth


class GTResultsScore(GTResults):
    """A class to store graound truth rankings."""

    def __init__(self, results_raw: dict[str, Any], sps_weights: dict[str, int]) -> None:
        super().__init__(results_raw)
        self.gt_choice = GroundTruth(**sps_weights)

    @lru_cache
    def get_sp(self, net_type: str, net_name: str, protocol: int, p: float) -> pd.Series:
        return self.gt_choice(
            net_type=net_type,
            net_name=net_name,
            protocol=protocol,
            p=p,
            nb_seeds=None,
        )


def acc(arr_y: list[Any], arr_yhat: list[list[Any]], cutoff: int, df_sps: pd.DataFrame) -> float:
    """Compute IoU for given cutoff."""
    y = set(arr_y[:cutoff])
    yhs = [set(ayh[:cutoff]) for ayh in arr_yhat]
    y_sps = df_sps.loc[df_sps.index.isin(y)].sum()
    yh_sps = np.mean([df_sps.loc[df_sps.index.isin(yh)].sum() for yh in yhs])
    res_acc = round(yh_sps / y_sps, 2)
    if res_acc > 1:
        raise ValueError(f"{cutoff}, {yh_sps}, {y_sps}")
    return res_acc


def cummulated_acc(arr_y: list[Any], arr_yhat: list[list[Any]], df_sps: pd.Series) -> np.array:
    """Compute IoU for cufoofs from 0% to 100% of actors."""
    assert all([len(arr_y) == len(ayh) for ayh in arr_yhat])
    assert np.all((df_sps.to_numpy()[1:] - df_sps.to_numpy()[:-1]) <= 0), "GT is not monotonic!"
    accs = []
    cutoffs = np.linspace(1, len(arr_y), len(arr_y) if len(arr_y) <= 1000 else 1000 , dtype=int)
    for cutoff in cutoffs:
        cutoff_acc = acc(arr_y=arr_y, arr_yhat=arr_yhat, cutoff=cutoff, df_sps=df_sps)
        accs.append(cutoff_acc)
    return np.array(accs)


def main(results_path: Path, out_path: Path) -> None:
    """A main function to produce visualisations."""

    print("loading config")
    with open(results_path / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    print("loading jsons")
    results_jsons = list(results_path.glob("*.json"))
    results_raw = {rj.stem: load_json_data(rj) for rj in results_jsons}
    gt_results = GTResultsScore(results_raw["ground_truth"], config["spreading_potential_score"])
    del results_raw["ground_truth"]

    all_accs = {}
    for im_name, im_results in results_raw.items():
        im_accs = {}
        for im_result in im_results:
            if im_result["net_type"] == im_result["net_name"]:
                case_name = im_result["net_type"]
            else:
                case_name = f"{im_result["net_type"]}_{im_result["net_name"]}"
            print(f"computing curve for {im_name}, {im_result['protocol']}, {im_result['p']}, {case_name}")

            # obtain cumulated accuracy of seed sets
            gt_result = gt_results.get_ranking(
                net_type=im_result["net_type"],
                net_name=im_result["net_name"],
                protocol=im_result["protocol"],
                p=im_result["p"],
            )
            df_sps = gt_results.get_sp(
                net_type=im_result["net_type"],
                net_name=im_result["net_name"],
                protocol=im_result["protocol"],
                p=im_result["p"],
            )
            ca = cummulated_acc(arr_y=gt_result, arr_yhat=im_result["seed_sets"], df_sps=df_sps)

            # save obtained curve in the dictionary and if needed create new fields there
            if not im_accs.get(im_result["protocol"]):
                im_accs[im_result["protocol"]] = {}    
            if not im_accs[im_result["protocol"]].get(im_result["p"]):
                im_accs[im_result["protocol"]][im_result["p"]] = {}
            if im_accs[im_result["protocol"]][im_result["p"]].get(case_name):
                raise ValueError(f"{case_name} already in parsed results!")
            im_accs[im_result["protocol"]][im_result["p"]][case_name] = ca

        # when iterating through the method is over - save all results
        all_accs[im_name] = im_accs

    # now draw the results
    pdf = PdfPages(out_path)
    avg_accs = {}
    for im_name, im_dict in all_accs.items():
        for protocol, p_dict in im_dict.items():
            for p, pp_dict in p_dict.items():
                print(f"plotitng curves for {im_name}, {protocol}, {p}")

                # plot for all networks for given params
                fig, acc_avg, auc_avg = plot_accs(
                    accs=pp_dict,
                    xlbl="size of cutoff (c)",
                    ylbl="SPS(y^hat_c) / SPS(y_c)",
                    plot_avg=True,
                )
                fig.suptitle(f"im_name: {im_name}, protocol: {protocol}, p: {p}, auc: {auc_avg}")
                fig.tight_layout()
                fig.savefig(pdf, format="pdf")
                plt.close(fig)

                # save average acc in to plot it again against all methods
                if not avg_accs.get(protocol):
                    avg_accs[protocol] = {}
                if not avg_accs[protocol].get(p):
                    avg_accs[protocol][p] = {}
                if avg_accs[protocol][p].get(im_name):
                    raise ValueError(f"{im_name} already in parsed results!")
                avg_accs[protocol][p][im_name] = {"acc": acc_avg, "auc": auc_avg}

    # draw average curves for each infmax method on a single canvas
    print("plotting average curves")
    for protocol, p_dict in avg_accs.items():
        for p, pp_dict in p_dict.items():
            # convert dict so that curves are inserted according to AuC (to sort legend on plots)
            sorted_dict = {
                k: v["acc"] for
                k, v in sorted(pp_dict.items(), key=lambda item: item[1]["auc"], reverse=True)
            }
            fig, _, _ = plot_accs(
                accs=sorted_dict,
                xlbl="size of cutoff (c)",
                ylbl="SPS(y^hat_c) / SPS(y_c)",
                plot_avg=False,
            )
            fig.suptitle(f"averaged AuC, protocol: {protocol}, p: {p}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")
            plt.close(fig)

    pdf.close()


if __name__ == "__main__":
    run_id = "20250324144648"
    results_path = Path(f"data/iou_curves/{run_id}")
    out_path = Path(f"data/iou_curves/{run_id}/comparison_score.pdf")
    main(results_path=results_path, out_path=out_path)

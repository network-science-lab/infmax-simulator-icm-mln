"""A script to plot IoU of seed sets constructed from full (i.e. comprising of all actors) ranks."""

from functools import lru_cache
from pathlib import Path
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from scripts.analysis.iou_ranking_plots import (
    AuxResults,
    GTResults,
    average_curves,
    load_cutoffs,
    load_json_data,
    parse_args,
    plot_accs,
    read_scores_auc,
    save_accs,
)
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


def cummulated_acc(arr_y: list[Any], arr_yhat: list[list[Any]], df_sps: pd.Series) -> np.ndarray:
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

    print("loading cutoffs")
    cutoffs = load_cutoffs(results_path / "cutoffs.csv")

    all_accs: list[AuxResults] = []
    im_names = set()
    protocols = set()
    ps = set()
    for im_name, im_results in results_raw.items():
        for im_result in im_results:
            # if im_result["net_type"] in {"timik1q2009", "arxiv_netscience_coauthorship"}:
            #     continue
            print(
                f"computing curve for {im_name}, {im_result['protocol']}, {im_result['p']}, "
                f"{im_result['net_type']}, {im_result['net_name']}"
            )

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
            ss_cutoff =  cutoffs.loc[
                (cutoffs["protocol"] == im_result["protocol"]) &
                (cutoffs["p"] == im_result["p"]) &
                (cutoffs["net_type"] == im_result["net_type"]) &
                (cutoffs["net_name"] == im_result["net_name"])
            ]["centile_size"].item()
            scores_auc = read_scores_auc(cumulated_accs=ca, ss_cutoff=ss_cutoff)

            # save obtained curve in a dataclass
            all_accs.append(
                AuxResults(
                    im_name=im_name,
                    protocol=im_result["protocol"],
                    p=im_result["p"],
                    net_type=im_result["net_type"],
                    net_name=im_result["net_name"],
                    cumulated_acc=ca,
                    auc_single=scores_auc["auc"]["single"],
                    auc_cutoff=scores_auc["auc"]["ss_cutoff"],
                    auc_full=scores_auc["auc"]["full"],
                    val_single=scores_auc["val"]["single"],
                    val_cutoff=scores_auc["val"]["ss_cutoff"],
                    val_full=scores_auc["val"]["full"],
                )
            )
            im_names.add(im_name)
            protocols.add(im_result["protocol"])
            ps.add(im_result["p"])

    # now draw the results
    pdf = PdfPages(out_path / "comparison_score.pdf")
    avg_accs = []
    for im_name in sorted(list(im_names)):
        for protocol in sorted(list(protocols)):
            for p in sorted(list(ps)):
                print(f"plotitng curves for {im_name}, {protocol}, {p}")

                # select results matching this case
                sub_results = [
                    aux_result for aux_result in all_accs if (
                        aux_result.im_name == im_name and
                        aux_result.protocol == protocol and
                        aux_result.p == p
                    )
                ]

                # compute average curve
                avg_acc = average_curves([sr.cumulated_acc for sr in sub_results])
                avg_ss_cutoff = int(
                    len(avg_acc) * cutoffs.loc[
                        (cutoffs["protocol"] == protocol) & (cutoffs["p"] == p)
                    ]["centile_nb"].mean()
                )
                avg_scores_auc = read_scores_auc(cumulated_accs=avg_acc, ss_cutoff=avg_ss_cutoff)
                avg_ar = AuxResults(
                    im_name=im_name,
                    protocol=protocol,
                    p=p,
                    net_type=im_name,  # a workaround
                    net_name=im_name,  # a workaround
                    cumulated_acc=avg_acc,
                    auc_single=avg_scores_auc["auc"]["single"],
                    auc_cutoff=avg_scores_auc["auc"]["ss_cutoff"],
                    auc_full=avg_scores_auc["auc"]["full"],
                    val_single=avg_scores_auc["val"]["single"],
                    val_cutoff=avg_scores_auc["val"]["ss_cutoff"],
                    val_full=avg_scores_auc["val"]["full"],
                )
                avg_accs.append(avg_ar)
                sub_results.append(avg_ar)

                # plot for all networks for given params
                fig = plot_accs(
                    accs=sub_results,
                    xlbl="size of cutoff",
                    ylbl="SPS(y^hat) / SPS(y)",
                    avg_idx=len(sub_results)-1,
                    curve_label="full",
                )
                fig.suptitle(
                    f"im: {im_name}, protocol: {protocol}, p: {p}, auc: {round(avg_ar.auc_full, 3)}"
                )
                # fig.tight_layout()
                fig.savefig(pdf, format="pdf")
                plt.close(fig)

    # draw average curves for each infmax method on a single canvas
    print("plotting average curves")
    for protocol in sorted(list(protocols)):
        for p in sorted(list(ps)):
            print(f"plotitng curves for {protocol}, {p}")
            sub_results = [
                aux_result for aux_result in avg_accs if (
                    aux_result.protocol == protocol and aux_result.p == p
                )
            ]
            fig = plot_accs(
                accs=sorted(sub_results, key=lambda item: item.val_cutoff, reverse=True),
                xlbl="size of cutoff",
                ylbl="SPS(y^hat) / SPS(y)",
                avg_idx=None,
                curve_label="full",
            )
            fig.suptitle(f"averaged SPS curves, protocol: {protocol}, p: {p}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")
            plt.close(fig)

    pdf.close()

    # save all particular accs as svc file
    save_accs(all_accs, out_path / "comparison_score_partial.csv")
    save_accs(avg_accs, out_path / "comparison_score_avg.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(results_path=args.run_dir, out_path=args.run_dir)

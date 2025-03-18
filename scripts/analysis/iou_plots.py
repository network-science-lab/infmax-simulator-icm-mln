"""A script to plot IoU of seed sets constructed from full (i.e. comprising of all actors) ranks."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def load_json_data(json_path: Path) -> dict[str, Any]:
    """Load json file with raw results."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


class GTResults:
    """A class to store graound truth rankings."""

    def __init__(self, results_raw: dict[str, Any]):
        self.results_raw = results_raw
    
    @lru_cache
    def get_ranking(self, net_type: str, net_name: str, protocol: int, p: float) -> list[Any]:
        for result_raw in self.results_raw:
            if (
                result_raw["net_type"] == net_type and
                result_raw["net_name"] == net_name and
                result_raw["protocol"] == protocol and
                result_raw["p"] == p
            ):
                return result_raw["seed_sets"][0]


def acc(arr_y: list[Any], arr_yhat: list[list[Any]], cutoff: int) -> float:
    """Compute IoU for given cutoff."""
    y = set(arr_y[:cutoff])
    yhs = [set(ayh[:cutoff]) for ayh in arr_yhat]
    accs = [len(y.intersection(yh)) / cutoff for yh in yhs]
    return np.mean(accs).item()


def cummulated_acc(arr_y: list[Any], arr_yhat: list[list[Any]]) -> np.array:
    """Compute IoU for cufoofs from 0% to 100% of actors."""
    assert all([len(arr_y) == len(ayh) for ayh in arr_yhat])
    accs = []
    cutoffs = np.linspace(1, len(arr_y), len(arr_y) if len(arr_y) <= 1000 else 1000 , dtype=int)
    for cutoff in cutoffs:
        cutoff_acc = acc(arr_y=arr_y, arr_yhat=arr_yhat, cutoff=cutoff)
        accs.append(cutoff_acc)
    return np.array(accs)


def get_cutoffs_fract(y_df_len: int) -> float:
    """Get cutoffs (for use as xs) for given curve to plot it."""
    return np.array([cutoff for cutoff in range(1, y_df_len + 1)]) / y_df_len


def average_curves(matrices: list[np.ndarray], kind: str = "linear") -> np.array:
    """Align lenghts of the curves by upsampling these which are shorter than the longest one."""
    target_length = max(len(m) for m in matrices)

    common_x = np.linspace(0, 1, target_length)
    resampled_curves = []

    for m in matrices:
        x_old = np.linspace(0, 1, len(m))
        interpolator = scipy.interpolate.interp1d(x_old, m, kind=kind, fill_value="extrapolate")
        resampled_curves.append(interpolator(common_x))

    avg_curve = np.mean(resampled_curves, axis=0)
    return avg_curve


def plot_accs(accs: dict[str, list[float]],  plot_avg: bool = True) -> Figure:
    """Plot curves for a given spreading conditions and networks."""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if plot_avg:
        alpha = 0.2
    else:
        alpha = 0.6

    acc_yhats = []
    for name, acc_yhat in accs.items():
        auc_yhat = np.trapezoid(acc_yhat, get_cutoffs_fract(len(acc_yhat)))
        ax.plot(
            get_cutoffs_fract(len(acc_yhat)),
            acc_yhat,
            label=f"{name}, {round(auc_yhat, 3)}",
            alpha=alpha,
        )
        acc_yhats.append(acc_yhat)

    if plot_avg:
        acc_avg = average_curves(acc_yhats)
        cutoffs_avg = get_cutoffs_fract(len(acc_avg))
        auc_avg = np.trapezoid(acc_avg, cutoffs_avg)
        ax.plot(cutoffs_avg, acc_avg, label=f"acc avg, {round(auc_avg, 3)}", color="green")

    cutoffs_rand = get_cutoffs_fract(100)
    auc_rand = np.trapezoid(cutoffs_rand, cutoffs_rand)
    ax.plot(cutoffs_rand, cutoffs_rand, "--", label=f"acc rand, {round(auc_rand, 3)}", color="red")

    ax.set_xlabel("size of cutoff")
    ax.set_xlim(0, 1)
    ax.set_ylabel("intersection(y_hat, y) / cutoff")
    ax.set_ylim(0, 1)
    ax.legend(
        loc="lower right",
        # ncol=2,
        bbox_to_anchor=(1.5, 0),
        # fancybox=True,
        handletextpad=0.05,
        borderaxespad=0.05,
        fontsize=7,
    )
    ax.set_aspect("equal", anchor="SW")

    if plot_avg:
        return fig, acc_avg, round(auc_avg, 3)
    else:
        return fig, None, None


def main(results_path: Path, out_path: Path) -> None:
    """A main function to produce visualisations."""
    print("loading jsons")
    results_jsons = list(results_path.glob("*.json"))
    results_raw = {rj.stem: load_json_data(rj) for rj in results_jsons}
    gt_results = GTResults(results_raw["ground_truth"])
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
            ca = cummulated_acc(arr_y=gt_result, arr_yhat=im_result["seed_sets"])

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
                fig, acc_avg, auc_avg = plot_accs(accs=pp_dict)
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
            fig, _, _ = plot_accs(accs=sorted_dict, plot_avg=False)
            fig.suptitle(f"averaged AuC, protocol: {protocol}, p: {p}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")
            plt.close(fig)

    pdf.close()


if __name__ == "__main__":
    run_id = "20250318113642"
    results_path = Path(f"data/iou_curves/{run_id}")
    out_path = Path(f"data/iou_curves/{run_id}/comparison.pdf")
    main(results_path=results_path, out_path=out_path)

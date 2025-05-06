"""A script to plot IoU of seed sets constructed from full (i.e. comprising of all actors) ranks."""

import argparse
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes


def parse_args(*args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        help="name of the configuration directory",
        nargs="?",
        type=Path,
        default=Path("data/iou_curves/final_real"),
    )
    parser.add_argument(
        "metric",
        help="name of the metric to use",
        nargs="?",
        type=str,
        choices=["jaccard", "prec", "avg_prec", "pos_acc"],
        default="pos_acc",
    )
    return parser.parse_args(*args)


def load_json_data(json_path: Path) -> dict[str, Any]:
    """Load json file with raw results."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def load_cutoffs(cutoffs_path: Path) -> pd.DataFrame:
    """Load csv with ss cutoffs."""
    return pd.read_csv(cutoffs_path, index_col=0)


class GTResults:
    """A class to store ground truth rankings."""

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


@dataclass
class AuxResults:
    im_name: str
    protocol: str
    p: float
    net_type: str
    net_name: str
    cumulated_acc: np.ndarray
    auc_single: float
    auc_cutoff: float
    auc_full: float
    val_single: float
    val_cutoff: float
    val_full: float
    avg_single: float = None
    avg_cutoff: float = None
    avg_full: float = None

    def to_dict_partial(self) -> dict[str, float | str]:
        self_dict = asdict(self)
        del self_dict["cumulated_acc"]
        return self_dict


def jaccard_at_k(arr_y: list[Any], arr_yhat: list[list[Any]], cutoff: int) -> float:
    """Compute IoU for given cutoff."""
    y = set(arr_y[:cutoff])
    yhs = [set(ayh[:cutoff]) for ayh in arr_yhat]
    accs = [len(y.intersection(yh)) / len(y.union(yh)) for yh in yhs]
    return np.mean(accs).item()


def precision_at_k(arr_y: list[Any], arr_yhat: list[list[Any]], cutoff: int) -> float:
    """Compute precision for given cutoff; in our case precision==recall."""
    y = set(arr_y[:cutoff])
    yhs = [set(ayh[:cutoff]) for ayh in arr_yhat]
    accs = [len(y.intersection(yh)) / cutoff for yh in yhs]
    return np.mean(accs).item()


# def average_precision_at_k(arr_y: list[Any], arr_yhat: list[list[Any]], cutoff: int) -> float:
#     """Compute average precision up to the given cutoff."""
#     precisions = []
#     for _cutoff in range(cutoff):
#         precisions.append(precision_at_k(arr_y, arr_yhat, _cutoff))
#     return np.mean(precisions).item()


def positional_accuracy(arr_y: list[Any], arr_yhat: list[list[Any]], cutoff: int) -> float:
    y = arr_y[:cutoff]
    yhs = [ayh[:cutoff] for ayh in arr_yhat]
    matches = [[gt == pred for gt, pred in zip(y, yh)] for yh in yhs]
    return np.mean(matches)


def cummulated_acc(arr_y: list[Any], arr_yhat: list[list[Any]], metric: str) -> np.array:
    """Compute IoU for cufoofs from 0% to 100% of actors."""
    assert all([len(arr_y) == len(ayh) for ayh in arr_yhat])
    if metric == "jaccard":
        acc = jaccard_at_k
    elif metric== "prec":
        acc = precision_at_k
    # elif metric == "avg_prec":
    #     acc = average_precision_at_k
    elif metric == "pos_acc":
        acc = positional_accuracy
    else:
        raise AttributeError("Incorrect name of metric!")
    accs = []
    cutoffs = np.linspace(1, len(arr_y), len(arr_y) if len(arr_y) <= 1000 else 1000 , dtype=int)
    for cutoff in cutoffs:
        cutoff_acc = acc(arr_y=arr_y, arr_yhat=arr_yhat, cutoff=cutoff)
        accs.append(cutoff_acc)
    return np.array(accs)


def get_cutoffs_fract(y_df_len: int) -> float:
    """Get cutoffs (for use as xs) for given curve to plot it."""
    return np.array([cutoff for cutoff in range(1, y_df_len + 1)]) / y_df_len


def read_scores_auc(cumulated_accs: np.ndarray, ss_cutoff: float) -> dict[str, float]:
    """Get scores and AuC for given cutoffs."""
    aucs = {"single": cumulated_accs[0]}  # this is due to approx. error in trapezoid func.
    vals = {"single": cumulated_accs[0]}
    avgs = {"single": cumulated_accs[0]}
    ss_cutoff_int = round(ss_cutoff * len(cumulated_accs))
    for cutoff_s, cutoff_n in zip([ss_cutoff_int, len(cumulated_accs)], ["ss_cutoff", "full"]):
        cutoff_ca = cumulated_accs[:cutoff_s]
        auc = np.trapezoid(cutoff_ca, get_cutoffs_fract(len(cutoff_ca)))
        aucs[cutoff_n] = auc
        vals[cutoff_n] = cutoff_ca[-1]
        avgs[cutoff_n] = np.mean(cutoff_ca).item()
    return {"val": vals, "auc": aucs, "avg": avgs}


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


def plot_accs(
    accs: list[AuxResults], 
    xlbl: str,
    ylbl: str,
    avg_idx: int | None,
    curve_label: str = Literal["reduced", "full"],
) -> Figure:
    """Plot curves for a given spreading conditions and networks."""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if avg_idx is not None:
        alpha = 0.3
    else:
        alpha = 0.6

    for idx, ar in enumerate(accs):
        if idx == avg_idx:
            continue
        label = ar.net_type if ar.net_type == ar.net_name else f"{ar.net_type}-{ar.net_name}"
        if label == "random_choice":
            label = "random"
        if curve_label == "full":
            label = f"{label}, {round(ar.val_single, 3)}, {round(ar.val_cutoff, 3)}, {round(ar.val_full, 3)}"
        ax.plot(get_cutoffs_fract(len(ar.cumulated_acc)), ar.cumulated_acc, label=label, alpha=alpha)

    if avg_idx is not None:
        ar = accs[avg_idx]
        label = "avg"
        if curve_label == "full":
            label = f"{label}, {round(ar.val_single, 3)}, {round(ar.val_cutoff, 3)}, {round(ar.val_full, 3)}"
        ax.plot(get_cutoffs_fract(len(ar.cumulated_acc)), ar.cumulated_acc, label=label, color="green")

    cutoffs_rand = get_cutoffs_fract(100)
    ax.plot(cutoffs_rand, cutoffs_rand, "--", label="x=y", color="red")

    ax.set_xlabel(xlbl)
    ax.set_xlim(0, 1)
    ax.set_ylabel(ylbl)
    ax.set_ylim(0, 1)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.05, 0),
        handletextpad=0.05,
        borderaxespad=0.05,
        fontsize=6,
    )
    ax.set_aspect("equal", anchor="SW")

    # draw the zoomed area
    axins = zoomed_inset_axes(ax, 2, loc="lower right", borderpad=1.8)
    for l_idx, line in enumerate(ax.lines):
        if avg_idx is not None:
            if l_idx == avg_idx:
                _alpha = 1
            else:
                _alpha = alpha
        else:
            _alpha = alpha
        axins.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), alpha=_alpha)
    axins.set_xlim(0.0, 0.2)
    axins.set_ylim(0.8, 1.0)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    fig.set_size_inches(6, 4)
    return fig


def save_accs(accs: list[AuxResults], out_path: Path) -> None:
    accs_dicts = [ar.to_dict_partial() for ar in accs]
    pd.DataFrame(accs_dicts).to_csv(out_path)


def main(results_path: Path, out_path: Path, metric: str) -> None:
    """A main function to produce visualisations."""
    print("loading jsons")
    results_jsons = list(results_path.glob("*.json"))
    results_raw = {rj.stem: load_json_data(rj) for rj in results_jsons}
    gt_results = GTResults(results_raw["ground_truth"])
    del results_raw["ground_truth"]

    print("loading cutoffs")
    cutoffs = load_cutoffs(results_path / "cutoffs.csv")

    all_accs: list[AuxResults] = []
    im_names = set()
    protocols = set()
    ps = set()
    for im_name, im_results in results_raw.items():
        for im_result in im_results:
            if im_result["net_type"] in {"timik1q2009", "arxiv_netscience_coauthorship"}:
                continue
            print(
                f"computing curve for {im_name}, {im_result['protocol']}, {im_result['p']}, "
                f"{im_result['net_type']}, {im_result['net_name']}"
            )

            # obtain cumulated accuracy of seed sets and read values we track
            gt_result = gt_results.get_ranking(
                net_type=im_result["net_type"],
                net_name=im_result["net_name"],
                protocol=im_result["protocol"],
                p=im_result["p"],
            )
            ca = cummulated_acc(arr_y=gt_result, arr_yhat=im_result["seed_sets"], metric=metric)
            ss_cutoff =  cutoffs.loc[
                (cutoffs["protocol"] == im_result["protocol"]) &
                (cutoffs["p"] == im_result["p"]) &
                (cutoffs["net_type"] == im_result["net_type"]) &
                (cutoffs["net_name"] == im_result["net_name"])
            ]["centile_nb"].item()
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
                    avg_single=scores_auc["avg"]["single"],
                    avg_cutoff=scores_auc["avg"]["ss_cutoff"],
                    avg_full=scores_auc["avg"]["full"],
                )
            )
            im_names.add(im_name)
            protocols.add(im_result["protocol"])
            ps.add(im_result["p"])

    # now draw the results
    pdf = PdfPages(out_path / f"comparison_ranking_{metric}.pdf")
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
                avg_ar = AuxResults(
                    im_name=im_name,
                    protocol=protocol,
                    p=p,
                    net_type=im_name,  # a workaround
                    net_name=im_name,  # a workaround
                    cumulated_acc=average_curves([sr.cumulated_acc for sr in sub_results]),
                    auc_single=np.mean([sr.auc_single for sr in sub_results]),
                    auc_cutoff=np.mean([sr.auc_cutoff for sr in sub_results]),
                    auc_full=np.mean([sr.auc_full for sr in sub_results]),
                    val_single=np.mean([sr.val_single for sr in sub_results]),
                    val_cutoff=np.mean([sr.val_cutoff for sr in sub_results]),
                    val_full=np.mean([sr.val_full for sr in sub_results]),
                    avg_single=np.mean([sr.avg_single for sr in sub_results]),
                    avg_cutoff=np.mean([sr.avg_cutoff for sr in sub_results]),
                    avg_full=np.mean([sr.avg_full for sr in sub_results]),
                )
                avg_accs.append(avg_ar)
                sub_results.append(avg_ar)

                # plot for all networks for given params
                fig = plot_accs(
                    accs=sub_results,
                    xlbl="size of cutoff",
                    ylbl="IoU(y_hat, y)",
                    avg_idx=len(sub_results)-1,
                    curve_label="full",
                )
                fig.suptitle(
                    f"im: {im_name}, protocol: {protocol}, p: {p}, auc: {round(avg_ar.auc_full, 3)}"
                )
                fig.tight_layout()
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
                ylbl="IoU(y_hat, y)",
                avg_idx=None,
                curve_label="full",
            )
            fig.suptitle(f"averaged IoU curves, protocol: {protocol}, p: {p}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")
            plt.close(fig)

    pdf.close()

    # save all particular accs as svc file
    save_accs(all_accs, out_path / f"comparison_ranking_{metric}_partial.csv")
    save_accs(avg_accs, out_path / f"comparison_ranking_{metric}_avg.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    out_path=Path("./dump")
    out_path.mkdir(exist_ok=True, parents=True)
    main(results_path=args.run_dir, out_path=out_path, metric=args.metric)

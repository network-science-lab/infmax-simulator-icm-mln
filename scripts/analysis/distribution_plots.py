"""A script to plot properties of actors from networks used ine evaluations."""

import argparse
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.evaluators.infmax_methods import (
    ACTOR,
    EXPOSED,
    NETWORK,
    P,
    PEAK_INFECTED,
    PEAK_ITERATION,
    PROTOCOL,
    SIMULATION_LENGTH,
    load_sp,
    load_sp_paths,
)
from src.evaluators.utils import SPScore
from src.sim_utils import parse_network_config


valid_icm_params = {
    "AND": {0.80, 0.85, 0.90, 0.95},
    "OR": {0.05, 0.10, 0.15, 0.20},
    "WILDCARDS": {-1.},
}


def parse_args(*args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_id",
        help="path to read configuration from",
        nargs="?",
        type=str,
        default="20250324173316",
    )
    return parser.parse_args(*args)


def parse_network_regexes(network_regexes: list[str]) -> list[tuple[str, str]]:
    networks = []
    for network_regex in network_regexes:
        print(network_regex)
        net_type, net_names = parse_network_config(network_regex)
        for net_name in net_names:
            networks.append((net_type, net_name))
    return networks


def read_config(config_path: Path) -> tuple[list[tuple[str, str]], tuple[str], tuple[float]]:
    """Read configuration file to obtain networks."""
    with open(config_path, "r") as file:
        config =  yaml.safe_load(file)
    parsed_networks = parse_network_regexes(config["networks"])
    return (
        parsed_networks,
        config["spreading_model"]["parameters"]["protocols"],
        config["spreading_model"]["parameters"]["p_values"],
        config["spreading_potential_score"],
    )


def read_sp(net_type: str, net_name: str, protocol: str, p: float) -> pd.DataFrame:
    """Read spreading potentials for given conditions."""
    sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
    sp_raw = load_sp(csv_paths=sp_paths)
    p = [p_val for p_val in valid_icm_params[protocol]] if p == -1. else [p]
    sp_filtered = sp_raw[sp_raw[PROTOCOL].isin([protocol]) & sp_raw[P].isin(p)]
    sp_filtered = sp_filtered.drop([P, PROTOCOL, NETWORK], axis=1)
    sp_mean = sp_filtered.groupby(by=[ACTOR]).mean()
    return sp_mean


def estimate_cutoff(scores: pd.DataFrame) -> dict[str, Any]:
    eighth_centile = np.percentile(scores, 80)
    border = np.argmin(np.abs(scores - eighth_centile))
    return {
        "border_actor": scores.index[border],
        "border_score": eighth_centile.item(),
        "centile_size": (border + 1).item(),
    }


def plot_sp_distributions(
    sp: pd.DataFrame, scores: pd.DataFrame, cutoff: dict[str, Any], max_score: float,
) -> Figure:
    fig, axs = plt.subplots(nrows=1, ncols=5)
    for ax, col, sort_ascending in zip(
        axs,
        [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
        [False, True, False, True]
    ):
        distribution = sp[col].sort_values(ascending=sort_ascending).values
        ax.plot(distribution)
        ax.set_xlabel("actor")
        ax.set_xlim(0, len(distribution))
        xticks = (np.array([0.0, 0.5, 1.0]) * len(distribution)).astype(int)
        ax.set_xticks(xticks)
        ax.set_ylabel("value")
        ax.set_ylim(0, max(distribution))
        ax.set_title(col)
    axs[-1].plot(scores)
    axs[-1].hlines(
        y=cutoff["border_score"],
        xmin=0,
        xmax=len(scores),
        linestyles="dashed",
        color="green",
        alpha=0.75,
    )
    axs[-1].vlines(
        x=cutoff["centile_size"],
        ymin=0,
        ymax=max_score,
        linestyles="dashed",
        color="red",
        alpha=0.75,
    )
    axs[-1].set_xlabel("actor")
    axs[-1].set_xlim(0, len(scores))
    axs[-1].set_xticks([cutoff["centile_size"]], labels=[cutoff["centile_size"]])
    axs[-1].set_ylabel("value")
    axs[-1].set_ylim(0, max_score)
    yticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) * max_score
    yticks[np.argmin(np.abs(yticks - cutoff["border_score"]))] = cutoff["border_score"]
    axs[-1].set_yticks(yticks.round(2))
    axs[-1].set_title("score")
    return fig


def main(results_path: Path, out_path: Path) -> None:
    """A main function to produce visualisations."""
    networks, protocols, ps, score_weights = read_config(results_path / "config.yaml")
    pdf = PdfPages(out_path / "distributions.pdf")
    cutoffs = []
    for network in networks:
        if network[0] == network[1]:
            case_name = network[0]
        else:
            case_name = f"{network[0]}-{network[1]}"
        print(network)
        for (protocol, p) in product(protocols, ps):
            print(protocol, p)
            sp = read_sp(net_type=network[0], net_name=network[1], protocol=protocol, p=p)
            sp_scores= SPScore(**score_weights)(sp)
            cutoff = estimate_cutoff(sp_scores)
            cutoff[NETWORK] = f"{network[0]}-{network[1]}"
            cutoff[PROTOCOL] = protocol
            cutoff[P] = p
            cutoffs.append(cutoff)
            max_score = sum([w for w in score_weights.values()])
            fig = plot_sp_distributions(sp, sp_scores, cutoffs[-1], max_score)
            fig.set_size_inches(10, 3)
            fig.suptitle(f"network: {case_name}, protocol: {protocol}, p: {p}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")
            plt.close(fig)
    pdf.close()
    pd.DataFrame(cutoffs).to_csv(out_path / "cutoffs.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    results_path = Path(f"data/iou_curves/{args.run_id}")
    out_path = Path(f"data/iou_curves/{args.run_id}/")
    main(results_path=results_path, out_path=out_path)

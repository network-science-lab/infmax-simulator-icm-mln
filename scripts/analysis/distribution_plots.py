"""A script to plot properties of actors from networks used ine evaluations."""

from itertools import product
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from nsl_data_utils.loaders.sp_loader import load_sp, load_sp_paths


valid_icm_params = {
    "AND": {0.80, 0.85, 0.90, 0.95},
    "OR": {0.05, 0.10, 0.15, 0.20},
    "WILDCARDS": {-1.},
}


def read_config(config_path: Path) -> tuple[list[tuple[str, str]], tuple[str], tuple[float]]:
    """Read configuration file to obtain networks."""
    with open(config_path, "r") as file:
        config =  yaml.safe_load(file)
    parsed_networks = []
    for cc in config["networks"]:
        ccs = cc.split("-")
        if len(ccs) == 1:
            parsed_networks.append((ccs[0], ccs[0]))
        elif len(ccs) == 2:
            parsed_networks.append(ccs)
        else:
            raise ValueError("Malfunction in network parsing!")
    return (
        parsed_networks,
        config["spreading_model"]["parameters"]["protocols"],
        config["spreading_model"]["parameters"]["p_values"],
    )


def read_sp(net_type: str, net_name: str, protocol: str, p: float) -> pd.DataFrame:
    """Read spreading potentials for given conditions."""
    sp_paths = load_sp_paths(net_type=net_type, net_name=net_name)
    sp_raw = load_sp(csv_paths=sp_paths)
    p = [p_val for p_val in valid_icm_params[protocol]] if p == -1. else [p]
    sp_filtered = sp_raw[sp_raw["protocol"].isin([protocol]) & sp_raw["p"].isin(p)]
    sp_filtered = sp_filtered.drop(["p", "protocol", "network"], axis=1)
    sp_mean = sp_filtered.groupby(by=["actor"]).mean()
    return sp_mean


def plot_sp_distributions(sp: pd.DataFrame) -> Figure:
    fig, axs = plt.subplots(nrows=1, ncols=4)
    for ax, col, sort_ascending in zip(
        axs,
        ["exposed", "simulation_length", "peak_infected", "peak_iteration"],
        [False, True, True, False]
    ):
        distribution = sp[col].sort_values(ascending=sort_ascending).values
        ax.plot(distribution) # , label=col)
        ax.set_xlabel("actor's label")
        ax.set_ylabel("value")
        ax.set_title(col)
    return fig


def main(results_path: Path, out_path: Path) -> None:
    """A main function to produce visualisations."""
    networks, protocols, ps = read_config(results_path / "config.yaml")
    pdf = PdfPages(out_path)
    for network in networks:
        if network[0] == network[1]:
            case_name = network[0]
        else:
            case_name = f"{network[0]}_{network[1]}"
        print(network)
        for (protocol, p) in product(protocols, ps):
            print(protocol, p)
            sp = read_sp(net_type=network[0], net_name=network[1], protocol=protocol, p=p)
            fig = plot_sp_distributions(sp)
            fig.suptitle(f"network: {case_name}, protocol: {protocol}, p: {p}")
            fig.tight_layout()
            fig.savefig(pdf, format="pdf")
            plt.close(fig)
    pdf.close()


if __name__ == "__main__":
    run_id = "20250317153249"
    # run_id = "20250317194630"
    results_path = Path(f"data/iou_curves/{run_id}")
    out_path = Path(f"data/iou_curves/{run_id}/distributions.pdf")
    main(results_path=results_path, out_path=out_path)

"""A script to plot properties of actors from networks used ine evaluations."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
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
    SPScore,
)


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
    sp_filtered = sp_raw[sp_raw[PROTOCOL].isin([protocol]) & sp_raw[P].isin(p)]
    sp_filtered = sp_filtered.drop([P, PROTOCOL, NETWORK], axis=1)
    sp_mean = sp_filtered.groupby(by=[ACTOR]).mean()
    return sp_mean


def plot_sp_distributions(sp: pd.DataFrame, score_weights: dict[str, int]) -> Figure:
    fig, axs = plt.subplots(nrows=1, ncols=5)
    for ax, col, sort_ascending in zip(
        axs,
        [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
        [False, True, False, True]
    ):
        distribution = sp[col].sort_values(ascending=sort_ascending).values
        ax.plot(distribution) # , label=col)
        ax.set_xlabel("actor's label")
        ax.set_ylabel("value")
        ax.set_title(col)
    scores= SPScore(
        exposed_weight=score_weights[EXPOSED],
        simulation_length_weight=score_weights[SIMULATION_LENGTH],
        peak_infected_weight=score_weights[PEAK_INFECTED],
        peak_iteration_weight=score_weights[PEAK_ITERATION],
    )(sp)
    axs[-1].plot(scores.values)
    axs[-1].set_xlabel("actor's label")
    axs[-1].set_ylabel("value")
    axs[-1].set_title("score")
    return fig


def main(results_path: Path, out_path: Path, score_weights: dict[str, int]) -> None:
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
            fig = plot_sp_distributions(sp, score_weights)
            fig.set_size_inches(10, 3)
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
    score_weights= {
        EXPOSED: 4,
        SIMULATION_LENGTH: 1,
        PEAK_INFECTED: 1,
        PEAK_ITERATION: 1,
    }
    main(results_path=results_path, out_path=out_path, score_weights=score_weights)

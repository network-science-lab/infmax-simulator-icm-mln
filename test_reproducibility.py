"""E2E test for both runners to generate dataset."""

import os
from pathlib import Path

import pandas as pd
import pytest

from src.generators import main_generator
from src.os_utils import set_rng_seed


@pytest.fixture
def tcase_icm_types():
    return [
        "FixedBudgetMICModel",
        "TorchMICModel",
    ]


@pytest.fixture
def tcase_config():
    return {
        "spreading_model": {
            "name": None,
            "parameters": {"protocols": ["OR", "AND"], "p_values": [0.9, 0.65, 0.1]}
        },
        "networks": ["toy_network"],
        "run": {"nb_repetitions": {"diffusion": 3}, "rng_seed": 43, "average_results": False},
        "logging": {"compress_to_zip": False, "out_dir": None},
    }


@pytest.fixture
def tcase_csv_names():
    return [
        Path("proto-AND--p-0.9--net-toy_network.csv"),
        Path("proto-AND--p-0.65--net-toy_network.csv"),
        Path("proto-AND--p-0.1--net-toy_network.csv"),
        Path("proto-OR--p-0.9--net-toy_network.csv"),
        Path("proto-OR--p-0.65--net-toy_network.csv"),
        Path("proto-OR--p-0.1--net-toy_network.csv"),
    ]


def compare_results(gt_dir: Path, test_dir: Path, csv_names: list[str], experiment_step: str) -> None:
    for csv_name in csv_names:
        gt_df = pd.read_csv(gt_dir / experiment_step / csv_name, index_col=0)
        test_df = pd.read_csv(test_dir / csv_name, index_col=0)
        pd.testing.assert_frame_equal(gt_df, test_df, obj=csv_name)
        print(f"Test passed for {experiment_step}, {csv_name}")


def test_e2e(tcase_icm_types, tcase_config, tcase_csv_names, tmpdir):
    for icm_type in tcase_icm_types:
        tcase_config["spreading_model"]["name"] = icm_type
        tcase_config["logging"]["out_dir"] = str(tmpdir)
        set_rng_seed(tcase_config["run"]["rng_seed"])
        main_generator.run_experiments(tcase_config)
        compare_results(Path("_test_data"), Path(tmpdir), tcase_csv_names, icm_type)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pytest.main(["-vs", __file__])

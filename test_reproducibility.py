"""E2E test for both runners to generate dataset."""

import os
from pathlib import Path

import pandas as pd
import pytest

from runners import main_runner
from runners.utils import set_seed


@pytest.fixture
def tcase_experiment_steps():
    return [
        "classic",
        "tensor",
    ]


@pytest.fixture
def tcase_config():
    return {
        "model": {"parameters": {"protocols": ["OR", "AND"], "p_values": [0.9, 0.65, 0.1]}},
        "networks": ["toy_network"],
        "run": {"repetitions": 3, "random_seed": 43, "average_results": False, "experiment_step": None},
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


def test_e2e(tcase_experiment_steps, tcase_config, tcase_csv_names, tmpdir):
    for experiment_step in tcase_experiment_steps:
        tcase_config["run"]["experiment_step"] = experiment_step
        tcase_config["logging"]["out_dir"] = str(tmpdir)
        set_seed(tcase_config["run"]["random_seed"])
        main_runner.run_experiments(tcase_config)
        compare_results(Path("_test_data"), Path(tmpdir), tcase_csv_names, experiment_step)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    pytest.main(["-vs", __file__])

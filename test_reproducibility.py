from pathlib import Path

import pandas as pd
import pytest

from misc.utils import set_seed
from runners import main_runner


@pytest.fixture
def tcase_config():
    return {
        "model": {"parameters": {"protocols": ["OR", "AND"], "p_values": [0.1, 0.35, 0.9]}},
        "networks": ["toy_network"],
        "run": {"repetitions": 3, "random_seed": 43, "average_results": False, "experiment_step": "classic"},
        "logging": {"compress_to_zip": False, "out_dir": None},
    }


@pytest.fixture
def tcase_csv_names():
    return [
        Path("proto-AND--p-0.1--net-toy_network.csv"),
        Path("proto-AND--p-0.9--net-toy_network.csv"),
        Path("proto-AND--p-0.35--net-toy_network.csv"),
        Path("proto-OR--p-0.1--net-toy_network.csv"),
        Path("proto-OR--p-0.9--net-toy_network.csv"),
        Path("proto-OR--p-0.9--net-toy_network.csv"),
    ]


def compare_results(gt_dir: Path, test_dir: Path, csv_names: list[str]) -> None:
    for csv_name in csv_names:
        gt_df = pd.read_csv(gt_dir / csv_name, index_col=0)
        test_df = pd.read_csv(test_dir / csv_name, index_col=0)
        pd.testing.assert_frame_equal(gt_df, test_df, obj=csv_name)
        print(f"Test passed for {csv_name}")


def test_e2e(tcase_config, tcase_csv_names, tmpdir):
    tcase_config["logging"]["out_dir"] = tmpdir
    set_seed(tcase_config["run"]["random_seed"])
    main_runner.run_experiments(tcase_config)
    compare_results(Path("_test_data"), Path(tmpdir), tcase_csv_names)


if __name__ == "__main__":
    pytest.main(["-vs", __file__])

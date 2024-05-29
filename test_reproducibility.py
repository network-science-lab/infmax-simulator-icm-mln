# TODO: consider migrating to pytest

from pathlib import Path
import tempfile

import pandas as pd
from misc.utils import set_seed
from runners import main_runner


TCASE_CONFIG = {
    "model": {"parameters":  {"protocols": ["OR", "AND"], "p_values": [0.1, 0.35, 0.9]}},
    "networks": ["toy_network"],
    "run": {"repetitions": 3, "random_seed": 43, "average_results": False},
    "logging": {"compress_to_zip": False, "out_dir": None}
}
TCASE_CSV_NAMES = [
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
        assert gt_df.equals(test_df), f"Error in {csv_name}"


def test_e2e():
    with tempfile.TemporaryDirectory() as temp_dir:
        TCASE_CONFIG["logging"]["out_dir"] = temp_dir
        set_seed(TCASE_CONFIG["run"]["random_seed"])
        main_runner.run_experiments(TCASE_CONFIG)
        compare_results(Path("_test_data"), Path(temp_dir), TCASE_CSV_NAMES)


if __name__ == "__main__":
    test_e2e()

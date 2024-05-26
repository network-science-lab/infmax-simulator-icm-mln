import argparse
import warnings
import yaml

from misc.utils import set_seed
from runners import main_runner


warnings.filterwarnings(action="ignore", category=FutureWarning)


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Experiment config file (default: config.yaml).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--runner",
        help="A runner function to execute (default: main_runner).",
        type=str,
        default="runner_optimised",
    )
    return parser.parse_args(*args)


if __name__ == "__main__":

    args = parse_args(
        ["--config", "_configs/example.yaml", "--runner", "main_runner"]
    )
    # args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if _ := config["run"].get("random_seed"):
        print(f"Setting random seed as {_}!")
        set_seed(config["run"]["random_seed"])
    print(f"Loaded config: {config}")

    if args.runner == "main_runner":
        print(f"Inferred runner as: {args.runner}")
        main_runner.run_experiments(config)
    else:
        raise AttributeError(f"Incorrect runner name ({args.runner})!")

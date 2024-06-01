# TODO: change prints to logs
# TODO: merge argparser with yaml config

import argparse
import yaml

from misc.utils import set_seed
from runners import main_runner


def parse_args(*args):
    parser = argparse.ArgumentParser()  # TODO: rewrite to follow Unix convention of CLI software
    parser.add_argument(
        "config",
        help="Experiment config file (default: config.yaml).",
        type=str,
    )
    parser.add_argument(
        "runner",
        help="A runner function to execute (default: main_runner).",
        type=str,
        default="main_runner",
    )
    return parser.parse_args(*args)


if __name__ == "__main__":

    # Uncomment for debugging
    # args = parse_args(["_configs/example.yaml", "main_runner"])
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if random_seed := config["run"].get("random_seed"):
        print(f"Setting random seed as {random_seed}!")
        set_seed(config["run"]["random_seed"])
    print(f"Loaded config: {config}")

    if args.runner == "main_runner":
        print(f"Inferred runner as: {args.runner}")
        main_runner.run_experiments(config)
    else:
        raise AttributeError(f"Incorrect runner name ({args.runner})!")

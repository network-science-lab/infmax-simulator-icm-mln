# TODO: change prints to logs

import argparse
import yaml

from src.evaluators import main_evaluator
from src.utils import set_seed


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Experiment config file.",
        nargs="?",
        type=str,
        default="_configs/eval_ssm.yaml",
    )
    return parser.parse_args(*args)



if __name__ == "__main__":

    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if random_seed := config["run"].get("random_seed"):
        print(f"Setting randomness seed as {random_seed}!")
        set_seed(config["run"]["random_seed"])
    print(f"Loaded config: {config}")

    main_evaluator.run_evaluations(config)

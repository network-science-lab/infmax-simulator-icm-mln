# TODO: change prints to logs (consider using hydra)

import argparse
import yaml

from src.evaluators import main_evaluator
from src.generators import main_generator
from src.os_utils import set_rng_seed


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Experiment config file.",
        nargs="?",
        type=str,
        default="_configs/eval_ssm.yaml",
        # default="_configs/gen_sp.yaml",
    )
    return parser.parse_args(*args)


if __name__ == "__main__":

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")

    if random_seed := config["run"].get("random_seed"):
        print(f"Setting randomness seed as {random_seed}!")
        set_rng_seed(config["run"]["rng_seed"])
    
    if (experiment_type := config["run"].get("experiment_type")) == "generate":
        runner = main_generator
    elif experiment_type == "evaluate":
        runner = main_evaluator
    else:
        raise ValueError(f"Unknown experiment type {experiment_type}")

    print(f"Inferred experiment type as: {experiment_type}")
    runner.run_experiments(config)

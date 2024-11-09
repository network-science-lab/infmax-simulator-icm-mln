import random

import git
import numpy as np
import torch


def get_recent_git_sha() -> str:
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    return git_sha


def set_rng_seed(seed):
    """Fix seed of the random numbers generator for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gain(seed_nb: int, exposed_nb: int, total_actors: int) -> float:
    max_available_gain = total_actors - seed_nb
    obtained_gain = exposed_nb - seed_nb
    return 100 * obtained_gain / max_available_gain

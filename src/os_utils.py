"""Helpers related with I/O and OS-related operations."""

import datetime
import random
import shutil

from dataclasses import asdict, dataclass
from pathlib import Path

import git
import numpy as np
import pandas as pd
import torch


def get_recent_git_sha() -> str:
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    return git_sha


def get_current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2) -> datetime.timedelta:
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return t_2 - t_1


def set_rng_seed(seed: int) -> None:
    """Fix seed of the random numbers generator for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def export_dataclasses(marginal_efficiencies: list[dataclass], out_path: Path) -> None:
    me_dict_all = [asdict(me) for me in marginal_efficiencies]
    pd.DataFrame(me_dict_all).to_csv(out_path, index=False)


def zip_detailed_logs(logged_dirs: list[Path], rm_logged_dirs: bool = True) -> None:
    if len(logged_dirs) == 0:
        print("No directories provided to create archive from.")
        return
    for dir_path in logged_dirs:
        shutil.make_archive(logged_dirs[0].parent / "_output", "zip", root_dir=str(dir_path))
    if rm_logged_dirs:
        for dir_path in logged_dirs:
            shutil.rmtree(dir_path)
    print(f"Compressed detailed logs.")

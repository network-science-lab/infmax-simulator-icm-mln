import random

import numpy as np
import torch


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
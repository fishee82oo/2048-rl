import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # A small CPU MLP is faster with one thread and easier to reproduce.
    torch.set_num_threads(1)

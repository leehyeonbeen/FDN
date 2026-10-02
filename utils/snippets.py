import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from scipy.signal import butter, sosfiltfilt
from copy import deepcopy
from typing import OrderedDict
import time
import functools
import random


def sec2hms(seconds):
    Hr = int(seconds // 3600)
    seconds -= Hr * 3600
    Min = int(seconds // 60)
    seconds -= Min * 60
    Sec = seconds
    return Hr, Min, Sec


def plot_template(fontsize=14):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes3d.grid"] = True
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.ymargin"] = 0.2
    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["axes.titlesize"] = fontsize + 3
    plt.rcParams["xtick.labelsize"] = fontsize - 3
    plt.rcParams["ytick.labelsize"] = fontsize - 3
    plt.rcParams["axes.formatter.useoffset"] = False  # scientific notation off
    plt.rcParams["axes.axisbelow"] = True


def print_line():
    print(f"{'':=>150}")


def increase_leglw(leg, linewidth: float = 3):
    for legobj in leg.legend_handles:
        legobj.set_linewidth(linewidth)


def func_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__qualname__}' executed in {end - start:.4f}s")
        return result

    return wrapper


def fix_random_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def enable_acceleration_configs():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Tee:
    """Print to the console and to a file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def result_exists(path: str, task: str) -> bool:
    """Existing result files are never deleted: print a message and return True instead."""
    if os.path.exists(path):
        print(f"{path} already exists. Delete it manually if you want to re-run {task}.")
        return True
    return False

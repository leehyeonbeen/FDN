import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import torch

from exp.helper import get_event_name, parse_exp_configs
from exp.trainer import Trainer
from models import FDN_PatchTST_RelPos6D
from models.initialization import initialize_params
from utils.snippets import enable_acceleration_configs, fix_random_seed
from data.dataset import HydraulicDatasetRelPos6D, CollateDecompose


def main(args):
    exp_name, model_name, configs, seed = args
    fix_random_seed(seed)
    model = MODELS_TO_RUN[model_name].Model(configs)
    initialize_params(model)

    trainer = Trainer(model, configs, HydraulicDatasetRelPos6D, CollateDecompose)
    trainer.fit(
        event_name=get_event_name(f"{exp_name}", model, tag=configs.exp_tag),
    )


def main_parallel(configs):
    main_args = []

    for model_name in sorted(list(MODELS_TO_RUN.keys())):
        for seed in range(configs.itr):
            configs.random_seed = seed
            main_args.append((configs.exp_name, model_name, deepcopy(configs), seed))

    # Multiprocessing
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=configs.num_parallel_runs) as executor:
        list(executor.map(main, main_args))

    # Serial execution
    # for arg in main_args:
    # main(arg)


enable_acceleration_configs()
configs = parse_exp_configs()

configs.exp_name = "Ablation_FEF_MOE"
configs.disable_moe = True
configs.n_filters = 1

MODELS_TO_RUN = {
    "FDN_PatchTST_RelPos6D": FDN_PatchTST_RelPos6D,
}

if __name__ == "__main__":
    main_parallel(configs)

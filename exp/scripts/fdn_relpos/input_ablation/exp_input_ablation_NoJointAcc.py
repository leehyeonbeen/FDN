import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from data.dataset import CollateDecompose
from data.dataset_ablation import (
    HydraulicDatasetRelPos6DNoJointAcc,
)
from exp.helper import get_event_name, parse_exp_configs
from exp.trainer import Trainer
from models import FDN_PatchTST_RelPos6D_InputAblation
from models.initialization import initialize_params
from utils.snippets import enable_acceleration_configs, fix_random_seed


def main(args):
    exp_name, model_name, configs, seed = args
    fix_random_seed(seed)
    model = MODELS_TO_RUN[model_name].Model(configs)
    initialize_params(model)

    trainer = Trainer(
        model,
        configs,
        HydraulicDatasetRelPos6DNoJointAcc,
        CollateDecompose,
    )
    trainer.fit(event_name=get_event_name(exp_name, model, tag=configs.exp_tag))


def main_parallel(configs):
    main_args = []
    for model_name in sorted(MODELS_TO_RUN):
        for seed in range(configs.itr):
            run_configs = deepcopy(configs)
            run_configs.random_seed = seed
            main_args.append((configs.exp_name, model_name, run_configs, seed))

    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=configs.num_parallel_runs) as executor:
        list(executor.map(main, main_args))


enable_acceleration_configs()
configs = parse_exp_configs()
configs.exp_name = "FromScratch_NoJointAcc"
configs.exclude_jointvel = False
configs.exclude_jointacc = True

MODELS_TO_RUN = {
    "FDN_PatchTST_RelPos6D_InputAblation": FDN_PatchTST_RelPos6D_InputAblation,
}


if __name__ == "__main__":
    main_parallel(configs)

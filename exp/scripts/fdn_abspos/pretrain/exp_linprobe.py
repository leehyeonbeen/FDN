import os
import sys

sys.path.append(os.getcwd())

import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import torch

from data.dataset import *
from exp.helper import get_event_name, parse_exp_configs
from exp.trainer import FineTuner
from models import FDN_PatchTST_AbsPos7D
from models.initialization import initialize_params
from utils.snippets import enable_acceleration_configs, fix_random_seed


def main(args):
    exp_name, model_name, configs, seed = args
    fix_random_seed(seed)
    model = MODELS_TO_RUN[model_name].Model(configs)
    initialize_params(model)

    trainer = FineTuner(
        model,
        configs,
        HydraulicDatasetAbsPos7D,
        CollateDecompose,
        ckpt_path=configs.pretrain_ckpt_path,
        mode="linprobe",
    )
    trainer.fit(
        event_name=get_event_name(f"{exp_name}", model, tag=configs.exp_tag),
    )


def main_parallel(configs):
    main_args = []

    # for FineTune only
    ckpt_path_list = sorted(
        glob.glob("exp/runs/AbsPos7D_Pretrain/*PatchTST*/IT100K.pt")
    )

    for model_name in sorted(list(MODELS_TO_RUN.keys())):
        for ckpt_path in ckpt_path_list:
            tag = "_".join(os.path.dirname(ckpt_path).split("_")[5:-1])
            configs.exp_tag = tag
            configs.pretrain_ckpt_path = ckpt_path

            for seed in range(configs.itr):
                main_args.append(
                    (
                        configs.exp_name,
                        model_name,
                        deepcopy(configs),
                        configs.random_seed,
                    )
                )

    # Multiprocessing
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=configs.num_parallel_runs) as executor:
        list(executor.map(main, main_args))

    # for arg in main_args:
    # main(arg)


enable_acceleration_configs()
configs = parse_exp_configs()
configs.freeze_jointq = True
configs.itr = 1

configs.exp_name = "AbsPos7D_LinProbe"
configs.exp_tag = f"train_ratio{configs.train_ratio}"

MODELS_TO_RUN = {
    "FDN_PatchTST_AbsPos7D": FDN_PatchTST_AbsPos7D,
}


if __name__ == "__main__":
    # main((configs.exp_name, "FDN_PatchTST", configs, 0))
    main_parallel(configs)

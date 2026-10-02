import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import torch

from data.dataset import RH20TDatasetAbsPos7D, CollateDecompose
from exp.helper import get_event_name, parse_exp_configs
from exp.trainer import Pretrainer
from models import FDN_PatchTST_AbsPos7D
from models.initialization import initialize_params
from utils.snippets import enable_acceleration_configs, fix_random_seed


def main_resume(args):
    fix_random_seed()
    exp_name, model_name, _, ckpt_path = args
    configs = torch.load(ckpt_path, weights_only=False, map_location="cpu")["configs"]
    model = MODELS_TO_RUN[model_name].Model(configs)
    initialize_params(model)

    trainer = Pretrainer(model, configs, RH20TDatasetAbsPos7D, CollateDecompose)
    trainer.resume_from_checkpoint(ckpt_path)


def main(args):
    exp_name, model_name, configs, seed = args
    fix_random_seed(seed)
    model = MODELS_TO_RUN[model_name].Model(configs)
    initialize_params(model)

    trainer = Pretrainer(model, configs, RH20TDatasetAbsPos7D, CollateDecompose)
    trainer.fit(
        event_name=get_event_name(f"{exp_name}", model, tag=configs.exp_tag),
    )


def main_parallel(configs):
    main_args = []

    train_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    for model_name in sorted(list(MODELS_TO_RUN.keys())):
        for train_ratio in train_ratios:
            configs.train_ratio = train_ratio
            configs.exp_tag = f"train_ratio{train_ratio}"

            for seed in range(configs.itr):
                configs.random_seed = seed
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

    # Serial execution
    # for arg in main_args:
    # main(arg)


enable_acceleration_configs()
configs = parse_exp_configs()

configs.exp_name = "AbsPos7D_Pretrain"
configs.disable_jtorque = True
configs.freeze_jtorque = True
configs.train_epochs = 10
configs.itr = 3
configs.train_ratio = 0.1

MODELS_TO_RUN = {
    "FDN_PatchTST_AbsPos7D": FDN_PatchTST_AbsPos7D,
}


if __name__ == "__main__":
    # main_resume((configs.exp_name, "FDN_PatchTST_AbsPos7D", configs, ckpt_path))
    # main((configs.exp_name, "FDN_PatchTST_AbsPos7D", configs, 0))
    main_parallel(configs)

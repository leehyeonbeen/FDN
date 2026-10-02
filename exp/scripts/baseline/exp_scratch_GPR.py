import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from exp.helper import get_event_name, parse_exp_configs
from exp.trainer_gpr import TrainerGPR
from models import GPR
from utils.snippets import enable_acceleration_configs, fix_random_seed


def main(args):
    exp_name, model_name, configs, seed = args
    fix_random_seed(seed)
    model = MODELS_TO_RUN[model_name].Model

    trainer = TrainerGPR(model, configs)
    trainer.fit(
        event_name=get_event_name(f"{exp_name}", model, tag=configs.exp_tag),
    )


def main_parallel(configs):
    main_args = []

    for model_name in sorted(list(MODELS_TO_RUN.keys())):
        for lr in [1e-3]:
            for num_inducing in [1024]:
                # for num_latents in [2, 4]:
                for seed in range(configs.itr):
                    configs.lr = lr
                    configs.num_inducing = num_inducing
                    # configs.num_latents = num_latents
                    configs.exp_tag = f"lr{lr:.0e}_numInducing{num_inducing}"  # _numLatents{num_latents}
                    configs.random_seed = seed
                    main_args.append(
                        (configs.exp_name, model_name, deepcopy(configs), seed)
                    )

    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=configs.num_parallel_runs) as executor:
        list(executor.map(main, main_args))


enable_acceleration_configs()
configs = parse_exp_configs()

configs.exp_name = "Baseline_GPR"
configs.lr = 1e-3

MODELS_TO_RUN = {
    "GPR": GPR,
}

if __name__ == "__main__":
    # main((configs.exp_name, "GPR", configs, 0))
    main_parallel(configs)

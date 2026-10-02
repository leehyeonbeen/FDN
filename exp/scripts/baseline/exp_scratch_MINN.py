import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from exp.helper import get_event_name, parse_exp_configs
from exp.trainer_p2p import TrainerPoint2Point
from models import MINN
from models.initialization import initialize_params
from utils.snippets import enable_acceleration_configs, fix_random_seed


def main(args):
    exp_name, model_name, configs, seed = args
    fix_random_seed(seed)
    model = MODELS_TO_RUN[model_name].Model(configs)
    initialize_params(model)

    trainer = TrainerPoint2Point(model, configs)
    trainer.fit(
        event_name=get_event_name(f"{exp_name}", model, tag=configs.exp_tag),
    )


def main_parallel(configs):
    main_args = []

    for model_name in sorted(list(MODELS_TO_RUN.keys())):
        for lr in [1e-4]:
            for e_layers in [1]:
                for seed in range(configs.itr):
                    configs.lr = lr
                    configs.e_layers = e_layers
                    configs.exp_tag = f"lr{lr:.0e}_eLayers{e_layers}"
                    configs.random_seed = seed
                    main_args.append(
                        (configs.exp_name, model_name, deepcopy(configs), seed)
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

configs.exp_name = "Baseline_MINN"

MODELS_TO_RUN = {
    "MINN": MINN,
}

if __name__ == "__main__":
    # main((configs.exp_name, "MINN", configs, 0))
    main_parallel(configs)

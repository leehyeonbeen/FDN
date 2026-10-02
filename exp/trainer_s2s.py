import argparse
import logging
import warnings

import torch
import torch.nn as nn

from data.dataset import *
from exp.trainer_p2p import TrainerPoint2Point
from utils.snippets import *


# Suppress torch._dynamo warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

# Set logging levels to suppress dynamo messages
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


class TrainerSeq2Seq(TrainerPoint2Point):
    def __init__(
        self,
        model: nn.Module,
        configs: argparse.Namespace,
        dataset_cls=HydraulicDatasetAbsPos6D,
        collate_fn_cls=CollateSeq2Seq,
    ) -> None:
        super().__init__(
            model, configs, dataset_cls=dataset_cls, collate_fn_cls=collate_fn_cls
        )

    @func_timer
    def _setup_dataloader(
        self,
        dataset_cls=HydraulicDatasetAbsPos6D,
        collate_fn_cls=CollateSeq2Seq,
        index_stride=1,
        drop_last=True,
    ):
        super()._setup_dataloader(
            dataset_cls=dataset_cls,
            collate_fn_cls=collate_fn_cls,
            index_stride=index_stride,
            drop_last=drop_last,
        )

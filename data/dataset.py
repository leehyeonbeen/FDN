import os, sys

sys.path.append(os.getcwd())

from utils.snippets import func_timer
from torch.utils.data import Dataset
import pandas as pd
import random
import torch
import torch.nn.functional as F
import numpy as np
import glob
import os
import sys
from collections import OrderedDict
from layers.Filter import FreqPassFilter
import torch.multiprocessing as tmp
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())


# RH20T pretraining dataset, also the base class of all datasets below.
# A sample is an L-step input history and the next T-step wrench, with a channel_mask marking missing
# input channels (e.g. the 7th joint of 6-DoF robots). Inputs: [dq, qdot, qddot, u, q0] x 7 joints.
class RH20TDatasetRelPos7D(Dataset):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        mode: str,
        sampling_freq: int = 100,
        train_ratio: float = 1.0,
        scale_params: dict | None = None,
        index_stride: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.scaled = False
        self.sampling_freq = sampling_freq
        self.scaled = False
        self.dtype = torch.float32
        self.scaled = False
        self.train_ratio = train_ratio
        self.datafiles = sorted(glob.glob("data/data_rh20t/processed/npy_ft/*.npy"))
        self.index_stride = index_stride

        self.declare_index_attributes()
        self.split_train_valid()
        self.initialize_cache()
        self.build_indices()
        self.get_scale_params(scale_params)
        # Total 61607104 samples
        # 6160711 samples for index_stride=10
        # batch_size=64, train_ratio=0.1, train_epochs=10 -> 96260 iters ~ 100K iterations
        # this covers train_ratio=0.9 (86634 iters)
        if index_stride > 1:
            self.indices = self.indices[::index_stride]

    @torch.no_grad()
    def __getitem__(self, idx):
        epi_idx, past_start = self.indices[idx]
        epi_idx = int(epi_idx)
        past_start = int(past_start)
        past_end = past_start + self.seq_len
        future_start = past_end
        future_end = future_start + self.pred_len
        data_input, data_output, channel_mask = self._get_episode(epi_idx)
        in_seq = data_input[past_start:past_end]
        out_seq = data_output[future_start:future_end]
        # scaling
        if self.scaled:
            (
                in_seq,
                out_seq,
            ) = self.scale(
                input=in_seq,
                output=out_seq,
                channel_mask=channel_mask,
            )
        # assert torch.isfinite(in_seq).all()
        # assert torch.isfinite(out_seq).all()

        return (
            in_seq,
            out_seq,
            channel_mask,
        )

    def get_scale_params(self, scale_params: dict | None):
        if scale_params is None:
            self.compute_scale_params()
        else:
            self.scale_params = scale_params
        assert hasattr(self, "scale_params"), "scale_params must be determined"

    def declare_index_attributes(self):
        self.n_inputs = 35  # [dq, qdot, qddot, u, q0] x 7 joints
        self.n_outputs = 6
        self.j7 = [6, 13, 20, 27, 34]  # 7th-joint channels
        self.imu_idx = None
        self.jtorque_idx = list(range(21, 28))
        self.woj7 = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7[self.j7] = False  # tensor[self.j7] / tensor[self.woj7]
        self.wojtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque[self.jtorque_idx] = False
        self.woj7_jtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7_jtorque[self.j7] = False
        self.woj7_jtorque[self.jtorque_idx] = False

    def split_train_valid(self):
        self.datafiles.sort()
        rng = random.Random(0)  # Fix seed for split
        rng.shuffle(self.datafiles)
        valid_ratio = 0.1
        self.datafiles_ = {
            "train": self.datafiles[: int(len(self.datafiles) * self.train_ratio)],
            "valid": self.datafiles[-int(len(self.datafiles) * (valid_ratio)) :],
        }
        self.datafiles = self.datafiles_[self.mode]
        self.datafiles_train = self.datafiles_["train"]
        if self.mode == "train":
            print(
                f"Pretraining with {len(self.datafiles_train):,} episodes from RH20T dataset."
            )

    def initialize_cache(self):
        # episode-level cache: avoids re-reading CSV for each sample
        self._episode_cache: OrderedDict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._episode_cache_size = 8

    def build_indices(self):
        # Build global indices across episodes.
        # idx -> (episode_idx, start_row)
        self.indices = []
        for epi_idx, f in enumerate(self.datafiles):
            n_rows = np.load(f).shape[0]
            n_valid = n_rows - self.seq_len - self.pred_len + 1
            if n_valid <= 0:
                continue
            starts = np.arange(n_valid, dtype=np.int64)
            epis = np.full((n_valid,), epi_idx, dtype=np.int64)
            self.indices.append(np.stack([epis, starts], axis=1))

        if len(self.indices) == 0:
            self.indices = np.zeros((0, 2), dtype=np.int64)
        else:
            self.indices = np.concatenate(self.indices, axis=0)

    def _compute_scale_params(self, datafile_paths: list):
        torch.set_num_threads(1)

        compute_dtype = torch.float64
        sum_i = torch.zeros(self.n_inputs, dtype=compute_dtype)
        sumsq_i = torch.zeros(self.n_inputs, dtype=compute_dtype)
        sum_o = torch.zeros(self.n_outputs, dtype=compute_dtype)
        sumsq_o = torch.zeros(self.n_outputs, dtype=compute_dtype)
        count_i = torch.zeros(self.n_inputs, dtype=compute_dtype)
        count_o = 0

        for datafile_path in datafile_paths:
            x, y, channel_mask = self._load_episode_from_datafiles(datafile_path)

            # Non J7, Non jtorque stats
            sum_i[self.woj7_jtorque] += x[:, self.woj7_jtorque].sum(dim=0)
            sumsq_i[self.woj7_jtorque] += (
                x[:, self.woj7_jtorque] * x[:, self.woj7_jtorque]
            ).sum(dim=0)
            count_i[self.woj7_jtorque] += x.shape[0]

            # compute j7 stats independently
            if not (x[:, self.j7] == 0).all():
                sum_i[self.j7] += x[:, self.j7].sum(dim=0)
                sumsq_i[self.j7] += (x[:, self.j7] * x[:, self.j7]).sum(dim=0)
                count_i[self.j7] += x.shape[0]

            # compute jtorque stats independently
            if not (x[:, self.jtorque_idx] == 0).all():
                sum_i[self.jtorque_idx] += x[:, self.jtorque_idx].sum(dim=0)
                sumsq_i[self.jtorque_idx] += (
                    x[:, self.jtorque_idx] * x[:, self.jtorque_idx]
                ).sum(dim=0)
                count_i[self.jtorque_idx] += x.shape[0]

            sum_o += y.sum(dim=0)
            sumsq_o += (y * y).sum(dim=0)
            count_o += y.shape[0]
        return (
            sum_i.numpy(),
            sumsq_i.numpy(),
            sum_o.numpy(),
            sumsq_o.numpy(),
            count_i.numpy(),
            count_o,
        )

    @func_timer
    def compute_scale_params(self):
        # Streaming mean/std over train episodes (no concat in memory).
        compute_dtype = torch.float64
        sum_i = torch.zeros(self.n_inputs, dtype=compute_dtype)
        sumsq_i = torch.zeros(self.n_inputs, dtype=compute_dtype)
        sum_o = torch.zeros(self.n_outputs, dtype=compute_dtype)
        sumsq_o = torch.zeros(self.n_outputs, dtype=compute_dtype)
        count_i = torch.zeros(self.n_inputs, dtype=compute_dtype)
        count_o = 0

        if len(self.datafiles_train) > 10:
            n_workers = min(os.cpu_count(), 16)
            chunk_size = len(self.datafiles_train) // n_workers
            chunk_size = chunk_size if not chunk_size == 0 else 1
            chunks = [
                self.datafiles_train[i : i + chunk_size]
                for i in range(0, len(self.datafiles_train), chunk_size)
            ]
            ctx = tmp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as p:
                for r in tqdm(
                    p.imap_unordered(
                        self._compute_scale_params,
                        chunks,
                    ),
                    total=len(chunks),
                    desc="Computing scale params per episode chunk",
                ):
                    sum_i += torch.from_numpy(r[0])
                    sumsq_i += torch.from_numpy(r[1])
                    sum_o += torch.from_numpy(r[2])
                    sumsq_o += torch.from_numpy(r[3])
                    count_i += torch.from_numpy(r[4])
                    count_o += r[5]
        else:
            r = self._compute_scale_params(self.datafiles_train)
            sum_i += torch.from_numpy(r[0])
            sumsq_i += torch.from_numpy(r[1])
            sum_o += torch.from_numpy(r[2])
            sumsq_o += torch.from_numpy(r[3])
            count_i += torch.from_numpy(r[4])
            count_o += r[5]

        mean_i = sum_i / count_i
        mean_o = sum_o / count_o

        # match torch.std default (unbiased / Bessel correction) when count > 1
        denom_var_i = count_i - 1
        denom_var_o = count_o - 1
        var_i = (sumsq_i - count_i * mean_i * mean_i) / denom_var_i
        var_o = (sumsq_o - count_o * mean_o * mean_o) / denom_var_o
        std_i = torch.sqrt(var_i.clamp_min(1e-12))
        std_o = torch.sqrt(var_o.clamp_min(1e-12))

        self.scale_params = {
            "mean_i": mean_i.to(self.dtype),
            "mean_o": mean_o.to(self.dtype),
            "std_i": std_i.to(self.dtype),
            "std_o": std_o.to(self.dtype),
        }
        print(
            f"Computed and saved scaling parameters over {len(self.datafiles_train):,} episodes."
        )

    def __len__(self):
        return len(self.indices)

    def _load_episode_from_datafiles(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FDN_LOAD_RAW_FT=1: raw, non-denoised wrench (npy_ft_raw) for the spectral analyses
        # (Fig. 5(a), Fig. 10). Unset (default): denoised wrench (npy_ft).
        path_ft = path.replace("npy_ft", "npy_ft_raw") if os.environ.get("FDN_LOAD_RAW_FT") == "1" else path
        path_jointpos = path.replace("npy_ft", "npy_jointpos6")
        path_jointvel = path.replace("npy_ft", "npy_jointvel6")
        path_jointacc = path.replace("npy_ft", "npy_jointacc6")
        path_jointpos_0 = path.replace("npy_ft", "npy_jointpos6_0")
        path_jtorque = path.replace("npy_ft", "npy_jtorque6")
        channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)

        try:  # first try loading 6D
            jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
            jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
            jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
            jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)
            jointpos = F.pad(jointpos, (0, 1))  # pad if 6D
            jointvel = F.pad(jointvel, (0, 1))  # pad if 6D
            jointacc = F.pad(jointacc, (0, 1))  # pad if 6D
            jointpos_0 = F.pad(jointpos_0, (0, 1))  # pad if 6D
            channel_mask[self.j7] = False
        except FileNotFoundError:  # reload with 7D
            path_jointpos = path_jointpos.replace("jointpos6", "jointpos7")
            path_jointvel = path_jointvel.replace("jointvel6", "jointvel7")
            path_jointacc = path_jointacc.replace("jointacc6", "jointacc7")
            path_jointpos_0 = path_jointpos_0.replace("jointpos6", "jointpos7")
            jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
            jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
            jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
            jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)

        if os.path.exists(path_jtorque):  # try loading 6D torque
            jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)
            jtorque = F.pad(jtorque, (0, 1))  # pad if 6D
        elif os.path.exists(path_jtorque.replace("jtorque6", "jtorque7")):  # 7D torque
            path_jtorque = path_jtorque.replace(
                "jtorque6", "jtorque7"
            )  # change path name
            jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)
        else:  # dummy torque data
            jtorque = torch.zeros(jointpos.shape[0], 7).to(self.dtype)
            channel_mask[self.jtorque_idx] = False

        # 35D input, 6D output
        data_input = torch.cat(
            [jointpos, jointvel, jointacc, jtorque, jointpos_0], dim=-1
        )
        data_output = torch.from_numpy(np.load(path_ft)).to(self.dtype)

        return data_input, data_output, channel_mask

    def _get_episode(
        self, epi_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._episode_cache.get(epi_idx)
        if cached is not None:
            self._episode_cache.move_to_end(epi_idx)
            return cached

        path = self.datafiles[epi_idx]
        data_input, data_output, channel_mask = self._load_episode_from_datafiles(path)
        self._episode_cache[epi_idx] = (data_input, data_output, channel_mask)
        self._episode_cache.move_to_end(epi_idx)
        if len(self._episode_cache) > self._episode_cache_size:
            self._episode_cache.popitem(last=False)
        return data_input, data_output, channel_mask

    def scale(
        self, input=None, output=None, channel_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input is None and output is None:
            self.scaled = True
        if channel_mask is None:
            channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)
        if input is not None:
            input = input.clone()  # MUST CLONE TO AVOID NUMERICAL ISSUES
            input[:, channel_mask] = (
                input[:, channel_mask] - self.scale_params["mean_i"][channel_mask]
            ) / self.scale_params["std_i"][channel_mask]
        if output is not None:
            output = output.clone()  # MUST CLONE TO AVOID NUMERICAL ISSUES
            output = (output - self.scale_params["mean_o"]) / self.scale_params["std_o"]
        return input, output

    def unscale(self, input=None, output=None, channel_mask=None):
        if input is None and output is None:
            self.scaled = False
        if channel_mask is None:
            channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)
        if input is not None:
            input = input.clone()  # MUST CLONE TO AVOID NUMERICAL ISSUES
            input[:, channel_mask] = (
                input[:, channel_mask] * self.scale_params["std_i"][channel_mask]
                + self.scale_params["mean_i"][channel_mask]
            )
        if output is not None:
            output = output.clone()  # MUST CLONE TO AVOID NUMERICAL ISSUES
            output = output * self.scale_params["std_o"] + self.scale_params["mean_o"]
        return input, output


# RH20T with absolute joint positions: [q, qdot, qddot, u] x 7 joints
class RH20TDatasetAbsPos7D(RH20TDatasetRelPos7D):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        mode: str,
        sampling_freq: int = 100,
        train_ratio: float = 1.0,
        scale_params: dict | None = None,
        index_stride: int = 1,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            mode=mode,
            sampling_freq=sampling_freq,
            train_ratio=train_ratio,
            scale_params=scale_params,
            index_stride=index_stride,
        )

    def declare_index_attributes(self):
        self.n_inputs = 28  # [q, qdot, qddot, u] x 7 joints
        self.n_outputs = 6
        self.j7 = [6, 13, 20, 27]  # 7th-joint channels
        self.imu_idx = None
        self.jtorque_idx = list(range(21, 28))
        self.woj7 = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7[self.j7] = False  # tensor[self.j7] / tensor[self.woj7]
        self.wojtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque[self.jtorque_idx] = False
        self.woj7_jtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7_jtorque[self.j7] = False
        self.woj7_jtorque[self.jtorque_idx] = False

    def _load_episode_from_datafiles(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path_ft = path
        path_jointpos = path.replace("npy_ft", "npy_jointpos6")
        path_jointvel = path.replace("npy_ft", "npy_jointvel6")
        path_jointacc = path.replace("npy_ft", "npy_jointacc6")
        path_jointpos_0 = path.replace("npy_ft", "npy_jointpos6_0")
        path_jtorque = path.replace("npy_ft", "npy_jtorque6")
        channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)

        try:  # first try loading 6D
            jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
            jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
            jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
            jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)
            jointpos = F.pad(jointpos, (0, 1))  # pad if 6D
            jointvel = F.pad(jointvel, (0, 1))  # pad if 6D
            jointacc = F.pad(jointacc, (0, 1))  # pad if 6D
            jointpos_0 = F.pad(jointpos_0, (0, 1))  # pad if 6D
            channel_mask[self.j7] = False
        except FileNotFoundError:  # reload with 7D
            path_jointpos = path_jointpos.replace("jointpos6", "jointpos7")
            path_jointvel = path_jointvel.replace("jointvel6", "jointvel7")
            path_jointacc = path_jointacc.replace("jointacc6", "jointacc7")
            path_jointpos_0 = path_jointpos_0.replace("jointpos6", "jointpos7")
            jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
            jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
            jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
            jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)

        if os.path.exists(path_jtorque):  # try loading 6D torque
            jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)
            jtorque = F.pad(jtorque, (0, 1))  # pad if 6D
        elif os.path.exists(path_jtorque.replace("jtorque6", "jtorque7")):  # 7D torque
            path_jtorque = path_jtorque.replace(
                "jtorque6", "jtorque7"
            )  # change path name
            jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)
        else:  # dummy torque data
            jtorque = torch.zeros(jointpos.shape[0], 7).to(self.dtype)
            channel_mask[self.jtorque_idx] = False

        # 28D input, 6D output
        jointpos = jointpos + jointpos_0
        data_input = torch.cat([jointpos, jointvel, jointacc, jtorque], dim=-1)
        data_output = torch.from_numpy(np.load(path_ft)).to(self.dtype)

        return data_input, data_output, channel_mask


# Hydraulic grinding dataset in the 7-joint layout of RH20T (7th joint zero), for transfer learning.
# The fixed train/test episode split is defined in split_train_valid.
class HydraulicDatasetRelPos7D(RH20TDatasetRelPos7D):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        mode: str,
        sampling_freq: int = 100,
        train_ratio: float = 1.0,
        train_num_episodes=None,
        scale_params: dict | None = None,
        index_stride: int = 1,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.scaled = False
        self.sampling_freq = sampling_freq
        self.dtype = torch.float32
        self.scaled = False
        self.train_ratio = train_ratio
        self.train_num_episodes = train_num_episodes
        self.index_stride = index_stride

        self.declare_index_attributes()
        self.split_train_valid()
        self.initialize_cache()
        self.build_indices()
        self.get_scale_params(scale_params)
        if index_stride > 1:
            self.indices = self.indices[::index_stride]

    def split_train_valid(self):
        prefix = f"data/data_hydraulic/processed/npy_ft/"  # npy_ft_norm
        suffix = f".npy"
        self.split_table = {
            "train": [
                f"Ground_1st_Test_3_20220716",  # 230s
                f"Ground_1st_Test_4_20220716",  # 152s
                f"Ground_1st_Test_5_20220716",  # 176s
                f"Ground_1st_Test_6_20220716",  # 155s
                f"Ground_2nd_Test_2_20221020",  # 445s
                f"Ground_2nd_Test_4_20221020",  # 396s
                f"Ground_2nd_Test_5_20221020",  # 557s
                f"Ground_2nd_Test_6_20221020",  # 383s
            ],
            "test": [
                f"Ground_1st_Test_1_20220716",  # 289s, Session1 longest
                f"Ground_1st_Test_2_20220716",  # 149s, Session1 shortest
                f"Ground_2nd_Test_1_20221020",  # 418s, Session2 OOD
                f"Ground_2nd_Test_3_20221020",  # 593s, Session2 longest
            ],
        }
        for k, v in self.split_table.items():
            self.split_table[k] = [os.path.join(prefix, f + suffix) for f in v]

        rng = random.Random(0)
        rng.shuffle(self.split_table["train"])
        if self.mode == "train":  # use subset
            if self.train_num_episodes is not None:
                self.datafiles = self.split_table[self.mode][: self.train_num_episodes]
            else:
                self.datafiles = self.split_table[self.mode][
                    : int(len(self.split_table["train"]) * self.train_ratio)
                ]
        elif self.mode == "all":  # use all
            self.datafiles = self.split_table["train"] + self.split_table["test"]
        else:
            self.datafiles = self.split_table[self.mode]

        if self.train_num_episodes is not None:
            self.datafiles_train = self.split_table["train"][: self.train_num_episodes]
        else:
            self.datafiles_train = self.split_table["train"][
                : int(len(self.split_table["train"]) * self.train_ratio)
            ]


# Hydraulic dataset with absolute joint positions in the 7-joint layout, for transfer learning
class HydraulicDatasetAbsPos7D(HydraulicDatasetRelPos7D):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        mode: str,
        sampling_freq: int = 100,
        train_ratio: float = 1.0,
        train_num_episodes=None,
        scale_params: dict | None = None,
        index_stride: int = 1,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            mode=mode,
            sampling_freq=sampling_freq,
            train_ratio=train_ratio,
            train_num_episodes=train_num_episodes,
            scale_params=scale_params,
            index_stride=index_stride,
        )

    def declare_index_attributes(self):
        self.n_inputs = 28  # [q, qdot, qddot, u] x 7 joints
        self.n_outputs = 6
        self.j7 = [6, 13, 20, 27]  # 7th-joint channels
        self.imu_idx = None
        self.jtorque_idx = list(range(21, 28))
        self.woj7 = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7[self.j7] = False  # tensor[self.j7] / tensor[self.woj7]
        self.wojtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque[self.jtorque_idx] = False
        self.woj7_jtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7_jtorque[self.j7] = False
        self.woj7_jtorque[self.jtorque_idx] = False

    def _load_episode_from_datafiles(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path_ft = path
        path_jointpos = path.replace("npy_ft", "npy_jointpos6")
        path_jointvel = path.replace("npy_ft", "npy_jointvel6")
        path_jointacc = path.replace("npy_ft", "npy_jointacc6")
        path_jointpos_0 = path.replace("npy_ft", "npy_jointpos6_0")
        path_jtorque = path.replace("npy_ft", "npy_jtorque6")
        channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)

        try:  # first try loading 6D
            jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
            jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
            jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
            jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)
            jointpos = F.pad(jointpos, (0, 1))  # pad if 6D
            jointvel = F.pad(jointvel, (0, 1))  # pad if 6D
            jointacc = F.pad(jointacc, (0, 1))  # pad if 6D
            jointpos_0 = F.pad(jointpos_0, (0, 1))  # pad if 6D
            channel_mask[self.j7] = False
        except FileNotFoundError:  # reload with 7D
            path_jointpos = path_jointpos.replace("jointpos6", "jointpos7")
            path_jointvel = path_jointvel.replace("jointvel6", "jointvel7")
            path_jointacc = path_jointacc.replace("jointacc6", "jointacc7")
            path_jointpos_0 = path_jointpos_0.replace("jointpos6", "jointpos7")
            jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
            jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
            jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
            jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)

        if os.path.exists(path_jtorque):  # try loading 6D torque
            jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)
            jtorque = F.pad(jtorque, (0, 1))  # pad if 6D
        elif os.path.exists(path_jtorque.replace("jtorque6", "jtorque7")):  # 7D torque
            path_jtorque = path_jtorque.replace(
                "jtorque6", "jtorque7"
            )  # change path name
            jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)
        else:  # dummy torque data
            jtorque = torch.zeros(jointpos.shape[0], 7).to(self.dtype)
            channel_mask[self.jtorque_idx] = False

        # 28D input, 6D output
        jointpos = jointpos + jointpos_0  # no relative position
        data_input = torch.cat([jointpos, jointvel, jointacc, jtorque], dim=-1)
        data_output = torch.from_numpy(np.load(path_ft)).to(self.dtype)

        return data_input, data_output, channel_mask


# Hydraulic dataset for point-to-point and sequence-to-point baselines: [q, qdot, qddot, u] x 6 joints and
# the wrench over the same window, whose last step is the current wrench W_t
class HydraulicDatasetAbsPos6D_NonForecasting(HydraulicDatasetRelPos7D):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        mode: str,
        sampling_freq: int = 100,
        train_ratio: float = 1.0,
        train_num_episodes=None,
        scale_params: dict | None = None,
        index_stride: int = 1,
    ):
        super().__init__(
            seq_len,
            1,
            mode,
            sampling_freq,
            train_ratio,
            train_num_episodes,
            scale_params,
            index_stride,
        )

    def declare_index_attributes(self):
        self.n_inputs = 24  # [q, qdot, qddot, u] x 6 joints
        self.n_outputs = 6
        self.j7 = []
        self.imu_idx = None
        self.jtorque_idx = list(range(18, 24))
        self.woj7 = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7[self.j7] = False  # tensor[self.j7] / tensor[self.woj7]
        self.wojtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque[self.jtorque_idx] = False
        self.woj7_jtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7_jtorque[self.j7] = False
        self.woj7_jtorque[self.jtorque_idx] = False

    def _load_episode_from_datafiles(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path_ft = path
        path_jointpos = path.replace("npy_ft", "npy_jointpos6")
        path_jointvel = path.replace("npy_ft", "npy_jointvel6")
        path_jointacc = path.replace("npy_ft", "npy_jointacc6")
        path_jointpos_0 = path.replace("npy_ft", "npy_jointpos6_0")
        path_jtorque = path.replace("npy_ft", "npy_jtorque6")
        channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)

        jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
        jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
        jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
        jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)
        jointpos = jointpos + jointpos_0  # no relative position
        jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)

        # 24D input, 6D output
        data_input = torch.cat([jointpos, jointvel, jointacc, jtorque], dim=-1)
        data_output = torch.from_numpy(np.load(path_ft)).to(self.dtype)

        return data_input, data_output, channel_mask

    @torch.no_grad()
    def __getitem__(self, idx):
        epi_idx, past_start = self.indices[idx]
        epi_idx = int(epi_idx)
        past_start = int(past_start)
        past_end = past_start + self.seq_len
        data_input, data_output, channel_mask = self._get_episode(epi_idx)
        in_seq = data_input[past_start:past_end]  # history + current
        out_seq = data_output[past_start:past_end]  # history + current
        # scaling
        if self.scaled:
            (
                in_seq,
                out_seq,
            ) = self.scale(
                input=in_seq,
                output=out_seq,
                channel_mask=channel_mask,
            )
        # assert torch.isfinite(in_seq).all()
        # assert torch.isfinite(out_seq).all()

        return (
            in_seq,
            out_seq,
            channel_mask,
        )


# Hydraulic dataset with absolute positions [q, qdot, qddot, u] x 6 joints, for sequence-to-sequence
# baselines and the absolute-position FDN
class HydraulicDatasetAbsPos6D(HydraulicDatasetRelPos7D):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        mode: str,
        sampling_freq: int = 100,
        train_ratio: float = 1.0,
        train_num_episodes=None,
        scale_params: dict | None = None,
        index_stride: int = 1,
    ):
        super().__init__(
            seq_len,
            pred_len,
            mode,
            sampling_freq,
            train_ratio,
            train_num_episodes,
            scale_params,
            index_stride,
        )

    def declare_index_attributes(self):
        self.n_inputs = 24  # [q, qdot, qddot, u] x 6 joints
        self.n_outputs = 6
        self.j7 = []
        self.imu_idx = None
        self.jtorque_idx = list(range(18, 24))
        self.woj7 = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7[self.j7] = False  # tensor[self.j7] / tensor[self.woj7]
        self.wojtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque[self.jtorque_idx] = False
        self.woj7_jtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7_jtorque[self.j7] = False
        self.woj7_jtorque[self.jtorque_idx] = False

    def _load_episode_from_datafiles(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path_ft = path
        path_jointpos = path.replace("npy_ft", "npy_jointpos6")
        path_jointvel = path.replace("npy_ft", "npy_jointvel6")
        path_jointacc = path.replace("npy_ft", "npy_jointacc6")
        path_jointpos_0 = path.replace("npy_ft", "npy_jointpos6_0")
        path_jtorque = path.replace("npy_ft", "npy_jtorque6")
        channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)

        jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
        jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
        jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
        jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)
        jointpos = jointpos + jointpos_0  # no relative position
        jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)

        # 24D input, 6D output
        data_input = torch.cat([jointpos, jointvel, jointacc, jtorque], dim=-1)
        data_output = torch.from_numpy(np.load(path_ft)).to(self.dtype)

        return data_input, data_output, channel_mask


# Hydraulic dataset for FDN: [dq, qdot, qddot, u, q0] x 6 joints
class HydraulicDatasetRelPos6D(HydraulicDatasetRelPos7D):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        mode: str,
        sampling_freq: int = 100,
        train_ratio: float = 1.0,
        train_num_episodes=None,
        scale_params: dict | None = None,
        index_stride: int = 1,
    ):
        super().__init__(
            seq_len,
            pred_len,
            mode,
            sampling_freq,
            train_ratio,
            train_num_episodes,
            scale_params,
            index_stride,
        )

    def declare_index_attributes(self):
        self.n_inputs = 30  # [dq, qdot, qddot, u, q0] x 6 joints
        self.n_outputs = 6
        self.j7 = []  # no 7th joint
        self.imu_idx = None
        self.jtorque_idx = list(range(18, 24))
        self.woj7 = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7[self.j7] = False  # tensor[self.j7] / tensor[self.woj7]
        self.wojtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.wojtorque[self.jtorque_idx] = False
        self.woj7_jtorque = torch.ones(self.n_inputs, dtype=torch.bool)
        self.woj7_jtorque[self.j7] = False
        self.woj7_jtorque[self.jtorque_idx] = False

    def _load_episode_from_datafiles(
        self, path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path_ft = path
        path_jointpos = path.replace("npy_ft", "npy_jointpos6")
        path_jointvel = path.replace("npy_ft", "npy_jointvel6")
        path_jointacc = path.replace("npy_ft", "npy_jointacc6")
        path_jointpos_0 = path.replace("npy_ft", "npy_jointpos6_0")
        path_jtorque = path.replace("npy_ft", "npy_jtorque6")
        channel_mask = torch.ones(self.n_inputs, dtype=torch.bool)

        # 6D only
        jointpos = torch.from_numpy(np.load(path_jointpos)).to(self.dtype)
        jointvel = torch.from_numpy(np.load(path_jointvel)).to(self.dtype)
        jointacc = torch.from_numpy(np.load(path_jointacc)).to(self.dtype)
        jointpos_0 = torch.from_numpy(np.load(path_jointpos_0)).to(self.dtype)
        jtorque = torch.from_numpy(np.load(path_jtorque)).to(self.dtype)

        # 30D input, 6D output
        data_input = torch.cat(
            [jointpos, jointvel, jointacc, jtorque, jointpos_0], dim=-1
        )
        data_output = torch.from_numpy(np.load(path_ft)).to(self.dtype)

        return data_input, data_output, channel_mask


# Collate function for FDN: splits each T-step output wrench into a 1 Hz low-pass trend and the residual
# Horizon-level spectral decomposition is implemented here.
class CollateDecompose:
    def __init__(
        self,
        sampling_freq=100,
        trend_cutoff_freq=1,
    ):
        self.filt_trend = FreqPassFilter(
            mode="low",
            cutoff_freq=trend_cutoff_freq,
            sampling_freq=sampling_freq,
        )

    @torch.no_grad()
    def __call__(self, batch):
        in_seq, out_seq, channel_mask = zip(*batch)
        in_seq = torch.stack(in_seq, dim=0)
        out_seq = torch.stack(out_seq, dim=0)
        channel_mask = torch.stack(channel_mask, dim=0)
        
        out_trend_seq = self.filt_trend(out_seq)
        out_res_seq = out_seq - out_trend_seq

        return in_seq, out_seq, out_trend_seq, out_res_seq, channel_mask


class CollatePoint2Point(CollateDecompose):
    """Point-to-point baselines: current input x_t and current wrench W_t (HydraulicDatasetAbsPos6D_NonForecasting)."""

    @torch.no_grad()
    def __call__(self, batch):
        in_seq, out_seq, channel_mask = zip(*batch)
        in_seq = torch.stack(in_seq, dim=0)
        out_seq = torch.stack(out_seq, dim=0)
        out_trend_seq = torch.empty_like(out_seq)
        out_res_seq = torch.empty_like(out_seq)
        channel_mask = torch.stack(channel_mask, dim=0)

        in_seq = in_seq[:, -1, :]
        out_seq = out_seq[:, -1, :]

        return in_seq, out_seq, out_trend_seq, out_res_seq, channel_mask


class CollateSeq2Point(CollateDecompose):
    """Sequence-to-point baselines: input history and current wrench W_t (HydraulicDatasetAbsPos6D_NonForecasting)."""

    @torch.no_grad()
    def __call__(self, batch):
        in_seq, out_seq, channel_mask = zip(*batch)
        in_seq = torch.stack(in_seq, dim=0)
        out_seq = torch.stack(out_seq, dim=0)
        out_trend_seq = torch.empty_like(out_seq)
        out_res_seq = torch.empty_like(out_seq)
        channel_mask = torch.stack(channel_mask, dim=0)

        in_seq = in_seq[:, :, :]
        out_seq = out_seq[:, -1, :]

        return in_seq, out_seq, out_trend_seq, out_res_seq, channel_mask


class CollateSeq2Seq(CollateDecompose):
    """Collate for sequence-to-sequence baselines (used with HydraulicDatasetAbsPos6D).

    Returns the input history and the next T-step wrench without trend/residual decomposition.
    """

    @torch.no_grad()
    def __call__(self, batch):
        in_seq, out_seq, channel_mask = zip(*batch)
        in_seq = torch.stack(in_seq, dim=0)
        out_seq = torch.stack(out_seq, dim=0)
        out_trend_seq = torch.empty_like(out_seq)
        out_res_seq = torch.empty_like(out_seq)
        channel_mask = torch.stack(channel_mask, dim=0)

        return in_seq, out_seq, out_trend_seq, out_res_seq, channel_mask


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset1 = HydraulicDatasetRelPos7D(100, 100, "train")
    dataset1.scale()

    dataloader = DataLoader(dataset1, batch_size=4, collate_fn=CollateDecompose())
    for b in dataloader:
        x, y, ytrend, yres, cm = b
        pass

    dataset1[0]

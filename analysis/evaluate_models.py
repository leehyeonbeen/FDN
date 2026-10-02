import math
import os, sys
import uuid

sys.path.append(os.getcwd())

from data.dataset import *
from data.dataset_ablation import (
    HydraulicDatasetRelPos6DNoJointAcc,
    HydraulicDatasetRelPos6DNoJointVel,
    HydraulicDatasetRelPos6DNoJointVelAcc,
    HydraulicDatasetRelPos6DNoJTorque,
    HydraulicDatasetRelPos6DJointPosOnly,
)
from torch.utils.data import DataLoader
from utils.loss import *
from utils.metrics import *
import glob
from models import *
import numpy as np
from scipy.signal import sosfiltfilt, butter
from copy import deepcopy
from utils.snippets import plot_template, Tee, result_exists
from layers.Filter import butter_response
from matplotlib.pyplot import cm

lpf = FreqPassFilter(cutoff_freq=1)

# def lpf(x, filt_order: int = 4, cutoff_freq: float = 1.0):
#     device = x.device
#     dtype = x.dtype
#     ndim = x.ndim
#     sos = butter(filt_order, cutoff_freq, btype="low", output="sos", fs=100)
#     if ndim == 2:
#         x = x.unsqueeze(0)
#     x = sosfiltfilt(sos, x.contiguous().cpu().numpy(), axis=1)
#     x = torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)
#     if ndim == 2:
#         x = x.squeeze(0)
#     return x


def get_unique_keys(dir):
    assert "exp/runs" in dir
    ckpt_list = sorted(glob.glob(dir))
    key_list = []
    for p in ckpt_list:
        # exp/runs/<NAME_MAIN>/DateTime_<NAME_EXP>_Hash/...
        name_main = os.path.dirname(p).split("/")[2]
        name_exp = "_".join(os.path.dirname(p).split("/")[3].split("_")[1:-1])
        unique_path = f"{name_main}/{name_exp}"
        key_list.append(unique_path)
    return ckpt_list, key_list


def load_model(ckpt_path):
    MODELS = {
        "CNN": CNN,
        "GPR": GPR,
        "LSTM": LSTM,
        "MINN": MINN,
        "RBF": RBF,
        "FDN_PatchTST_RelPos7D": FDN_PatchTST_RelPos7D,
        "FDN_PatchTST_RelPos6D": FDN_PatchTST_RelPos6D,
        "FDN_PatchTST_RelPos6D_InputAblation": FDN_PatchTST_RelPos6D_InputAblation,
        "FDN_PatchTST_RelPos6D_MVNChannel": FDN_PatchTST_RelPos6D_MVNChannel,
        "FDN_PatchTST_RelPos6D_MVNKron": FDN_PatchTST_RelPos6D_MVNKron,
        "FDN_PatchTST_RelPos6D_MVNTemporal": FDN_PatchTST_RelPos6D_MVNTemporal,
        "FDN_PatchTST_RelPos6D_ModShared": FDN_PatchTST_RelPos6D_ModShared,
        "FDN_PatchTST_AbsPos6D": FDN_PatchTST_AbsPos6D,
        "FDN_PatchTST_AbsPos7D": FDN_PatchTST_AbsPos7D,
        "LSTMEncDec": LSTMEncDec,
        "PatchTST": PatchTST,
        "PatchTST_Gaussian": PatchTST_Gaussian,
        "TransformerEncDec": TransformerEncDec,
        "iTransformer": iTransformer,
    }
    model_cls = None
    for k, v in sorted(MODELS.items(), key=lambda item: len(item[0]), reverse=True):
        if k in ckpt_path:
            model_cls = v
            break
    if model_cls is None:
        raise ValueError(
            f"Could not infer model class from checkpoint path: {ckpt_path}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    configs = ckpt["configs"]
    state_dict = ckpt["state_dict"]
    scale_params = ckpt["scale_params"]
    # for compatibility
    if "disable_probhead" not in configs.__dict__:
        configs.disable_probhead = False
    if "disable_dethead" not in configs.__dict__:
        configs.disable_dethead = False
    model = model_cls.Model(configs)
    try:
        model.load_state_dict(state_dict)
        print(f"{model_cls.__name__} loaded successfully from {ckpt_path}.")
    except RuntimeError as e:
        print(f"{ckpt_path} error in loading state_dict of {model_cls.__name__}.")
        return None, None, None
    model.eval()
    return model, configs, scale_params


def get_dataloader(model, configs, scale_params):
    model_name = model.__class__.__module__.replace("models.", "")
    if model_name in ["MINN", "RBF", "GPR"]:
        dataset = HydraulicDatasetAbsPos6D_NonForecasting(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollatePoint2Point()
    elif model_name in ["CNN", "LSTM"]:
        dataset = HydraulicDatasetAbsPos6D_NonForecasting(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollateSeq2Point()
    elif model_name in [
        "LSTMEncDec",
        "TransformerEncDec",
        "PatchTST",
        "iTransformer",
        "PatchTST_Gaussian",
    ]:
        dataset = HydraulicDatasetAbsPos6D(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollateSeq2Seq()
    elif model_name == "FDN_PatchTST_RelPos7D":
        dataset = HydraulicDatasetRelPos7D(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollateDecompose()
    elif model_name in [
        "FDN_PatchTST_AbsPos6D",
    ]:
        dataset = HydraulicDatasetAbsPos6D(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollateDecompose()
    elif model_name == "FDN_PatchTST_RelPos6D_InputAblation":
        exclude_jointvel = getattr(configs, "exclude_jointvel", False)
        exclude_jointacc = getattr(configs, "exclude_jointacc", False)
        exclude_jtorque = getattr(configs, "disable_jtorque", False)
        if exclude_jointvel and exclude_jointacc and exclude_jtorque:
            dataset_cls = HydraulicDatasetRelPos6DJointPosOnly
        elif exclude_jtorque and not (exclude_jointvel or exclude_jointacc):
            dataset_cls = HydraulicDatasetRelPos6DNoJTorque
        elif exclude_jtorque:
            raise ValueError(
                "Unsupported combination of excluded RelPos6D input modalities."
            )
        elif exclude_jointvel and exclude_jointacc:
            dataset_cls = HydraulicDatasetRelPos6DNoJointVelAcc
        elif exclude_jointvel:
            dataset_cls = HydraulicDatasetRelPos6DNoJointVel
        elif exclude_jointacc:
            dataset_cls = HydraulicDatasetRelPos6DNoJointAcc
        else:
            raise ValueError(
                "InputAblation checkpoint does not specify an excluded input."
            )
        dataset = dataset_cls(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollateDecompose()
    elif model_name.startswith("FDN_PatchTST_RelPos6D"):
        dataset = HydraulicDatasetRelPos6D(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollateDecompose()
    elif model_name in [
        "FDN_PatchTST_AbsPos7D",
    ]:
        dataset = HydraulicDatasetAbsPos7D(
            configs.seq_len,
            configs.pred_len,
            "test",
            scale_params=scale_params,
            index_stride=1,
        )
        collate_fn = CollateDecompose()
    else:
        raise ValueError(f"Please define dataset logic for model: {model_name}")
    dataset.scale()
    dataloader = DataLoader(
        dataset, batch_size=1024, num_workers=4, collate_fn=collate_fn, shuffle=False
    )

    if hasattr(model, "enc_in"):
        assert (
            model.enc_in == dataset.n_inputs
        ), f"Model enc_in {model.enc_in} does not match dataset n_inputs {dataset.n_inputs}"
    return dataloader


@torch.no_grad()
def infer_with_dataloader(
    model,
    dataloader,
    normalized_scale: bool = False,
    stress_dropout: float = 0,
):
    """Run the model over the test episodes and split the outputs per episode.

    Returns labels, predictions, trend, residual mean, and log variance as lists over episodes
    (None for outputs the model does not have). Values are unscaled unless normalized_scale is True.
    """
    model_name = model.__class__.__module__.replace("models.", "")
    if torch.cuda.is_available():
        device = "cuda"
    elif "GPR" in model_name:
        device = "cpu"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)

    dropout = nn.Dropout(stress_dropout) if stress_dropout > 0 else nn.Identity()

    label_total = []
    pred_total = []
    pred_trend = []
    pred_mu = []
    pred_logvar = []
    scale_params = {k: v.to(device) for k, v in dataloader.dataset.scale_params.items()}
    for i, b in enumerate(dataloader):
        x, y, ytrend, yres, cm = b
        x = x.to(device)
        y = y.to(device)
        ytrend = ytrend.to(device)
        yres = yres.to(device)
        cm = cm.to(device)

        x = dropout(x)

        label_total.append(y)
        if "FDN" in model_name:
            ptrend, pres, pmu, plogvar = model(x, channel_mask=cm)
            pred_total.append(ptrend + pres)
            pred_trend.append(ptrend)
            pred_mu.append(pmu)
            pred_logvar.append(plogvar)
        elif "GPR" in model_name:
            pmean, plower, pupper = model.inference(x)
            pred_total.append(pmean)
        elif model_name == "PatchTST_Gaussian":
            pmu, plogvar = model(x)
            pred_mu.append(pmu)
            pred_logvar.append(plogvar)
            pred_total.append(pmu)
        else:
            p = model(x)
            pred_total.append(p)
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            # print(f"{i + 1}/{len(dataloader)} batches processed.")
            pass

    label_total = torch.cat(label_total, dim=0)
    pred_total = torch.cat(pred_total, dim=0)
    pred_trend = torch.cat(pred_trend, dim=0) if pred_trend else None
    pred_mu = torch.cat(pred_mu, dim=0) if pred_mu else None
    pred_logvar = torch.cat(pred_logvar, dim=0) if pred_logvar else None

    # Get episode edges
    episode_edges = (
        np.where(np.diff(dataloader.dataset.indices, axis=0)[:, 0] != 0)[0] + 1
    )
    if pred_trend is not None:
        if not normalized_scale:
            pred_trend = pred_trend * scale_params["std_o"] + scale_params["mean_o"]
        # Split episodes
        pred_trend = pred_trend.tensor_split(episode_edges.tolist(), dim=0)
    if pred_mu is not None and pred_logvar is not None:
        if not normalized_scale:
            pred_mu = pred_mu * scale_params["std_o"]
            if model_name == "PatchTST_Gaussian":
                pred_mu = pred_mu + scale_params["mean_o"]
            pred_logvar = pred_logvar + 2 * torch.log(scale_params["std_o"])
        pred_mu = pred_mu.tensor_split(episode_edges.tolist(), dim=0)
        pred_logvar = pred_logvar.tensor_split(episode_edges.tolist(), dim=0)
    # Unscale
    if not normalized_scale:
        pred_total = pred_total * scale_params["std_o"] + scale_params["mean_o"]
        label_total = label_total * scale_params["std_o"] + scale_params["mean_o"]

    # Split episodes
    label_total = label_total.tensor_split(episode_edges.tolist(), dim=0)
    pred_total = pred_total.tensor_split(episode_edges.tolist(), dim=0)

    return label_total, pred_total, pred_trend, pred_mu, pred_logvar


def evaluate_predictions(
    model,
    label_total,
    pred_total,
    pred_trend,
    pred_mu,
    pred_logvar,
    timedelay: int = 100,
    stepwise: bool = True,
):
    """CRPS, HF wRMSE, and LF pRMSE for force (F), torque (T), and all channels (FT).

    Predictions and labels are split into LF and HF parts at 1 Hz. Sequence outputs (N, T, C) use the
    step t + timedelay when stepwise, or all T steps otherwise. Point outputs (N, C) are compared with
    the label timedelay steps later. Gaussian outputs use the closed-form CRPS, deterministic outputs the MAE.
    """
    # Per-episode evaluation
    model_name = model.__class__.__module__.replace("models.", "")
    crps_F = (
        GaussianCRPS()
        if pred_logvar is not None and model_name != "GPR"
        else MeanAbsoluteError()
    )
    if "FDN" in model_name:
        if model.disable_probhead:
            crps_F = MeanAbsoluteError()
    crps_T = deepcopy(crps_F)
    crps_FT = deepcopy(crps_F)
    rmsehf_F = RMSError()
    rmsehf_T = deepcopy(rmsehf_F)
    rmsehf_FT = deepcopy(rmsehf_F)
    rmself_F = RootMeanSquaredError()
    rmself_T = deepcopy(rmself_F)
    rmself_FT = deepcopy(rmself_F)

    # iterate over episodes
    for idx, (label, pred) in enumerate(zip(label_total, pred_total)):
        label = label
        pred = pred
        # forecasting output
        if pred.ndim == 3:
            # CRPS
            if pred_logvar is not None and crps_F.__class__ == GaussianCRPS:
                mu = (
                    pred_trend[idx] + pred_mu[idx]
                    if pred_trend is not None
                    else pred_mu[idx]
                )
                logvar = pred_logvar[idx]
                mu_lf = lpf(mu.transpose(0, 1)).transpose(0, 1)
                mu_hf = mu - mu_lf
                label_lf = lpf(label.transpose(0, 1)).transpose(0, 1)
                label_hf = label - label_lf
                if stepwise:
                    # compute for a single step
                    crps_F.update(
                        mu[:, timedelay - 1 : timedelay, :3],
                        logvar[:, timedelay - 1 : timedelay, :3],
                        label[:, timedelay - 1 : timedelay, :3],
                    )
                    crps_T.update(
                        mu[:, timedelay - 1 : timedelay, 3:],
                        logvar[:, timedelay - 1 : timedelay, 3:],
                        label[:, timedelay - 1 : timedelay, 3:],
                    )
                    crps_FT.update(
                        mu[:, timedelay - 1 : timedelay],
                        logvar[:, timedelay - 1 : timedelay],
                        label[:, timedelay - 1 : timedelay],
                    )
                    rmsehf_F.update_gaussian(
                        mu_hf[:, timedelay - 1 : timedelay, :3],
                        logvar[:, timedelay - 1 : timedelay, :3],
                        label_hf[:, timedelay - 1 : timedelay, :3],
                    )
                    rmsehf_T.update_gaussian(
                        mu_hf[:, timedelay - 1 : timedelay, 3:],
                        logvar[:, timedelay - 1 : timedelay, 3:],
                        label_hf[:, timedelay - 1 : timedelay, 3:],
                    )
                    rmsehf_FT.update_gaussian(
                        mu_hf[:, timedelay - 1 : timedelay],
                        logvar[:, timedelay - 1 : timedelay],
                        label_hf[:, timedelay - 1 : timedelay],
                    )
                    rmself_F.update(
                        mu_lf[:, timedelay - 1 : timedelay, :3],
                        label_lf[:, timedelay - 1 : timedelay, :3],
                    )
                    rmself_T.update(
                        mu_lf[:, timedelay - 1 : timedelay, 3:],
                        label_lf[:, timedelay - 1 : timedelay, 3:],
                    )
                    rmself_FT.update(
                        mu_lf[:, timedelay - 1 : timedelay],
                        label_lf[:, timedelay - 1 : timedelay],
                    )
                else:
                    # update CRPS for every step
                    crps_F.update(mu[..., :3], logvar[..., :3], label[..., :3])
                    crps_T.update(mu[..., 3:], logvar[..., 3:], label[..., 3:])
                    crps_FT.update(mu, logvar, label)
                    rmsehf_F.update_gaussian(
                        mu_hf[..., :3], logvar[..., :3], label_hf[..., :3]
                    )
                    rmsehf_T.update_gaussian(
                        mu_hf[..., 3:], logvar[..., 3:], label_hf[..., 3:]
                    )
                    rmsehf_FT.update_gaussian(mu_hf, logvar, label_hf)
                    rmself_F.update(mu_lf[..., :3], label_lf[..., :3])
                    rmself_T.update(mu_lf[..., 3:], label_lf[..., 3:])
                    rmself_FT.update(mu_lf, label_lf)
            else:  # if not probabilistic
                pred_lf = lpf(pred.transpose(0, 1)).transpose(0, 1)
                pred_hf = pred - pred_lf
                label_lf = lpf(label.transpose(0, 1)).transpose(0, 1)
                label_hf = label - label_lf
                if stepwise:
                    crps_F.update(
                        pred[:, timedelay - 1, :3],
                        label[:, timedelay - 1, :3],
                    )
                    crps_T.update(
                        pred[:, timedelay - 1, 3:],
                        label[:, timedelay - 1, 3:],
                    )
                    crps_FT.update(
                        pred[:, timedelay - 1],
                        label[:, timedelay - 1],
                    )
                    rmsehf_F.update(
                        pred_hf[:, timedelay - 1 : timedelay, :3],
                        label_hf[:, timedelay - 1 : timedelay, :3],
                    )
                    rmsehf_T.update(
                        pred_hf[:, timedelay - 1 : timedelay, 3:],
                        label_hf[:, timedelay - 1 : timedelay, 3:],
                    )
                    rmsehf_FT.update(
                        pred_hf[:, timedelay - 1 : timedelay],
                        label_hf[:, timedelay - 1 : timedelay],
                    )
                    rmself_F.update(
                        pred_lf[:, timedelay - 1 : timedelay, :3],
                        label_lf[:, timedelay - 1 : timedelay, :3],
                    )
                    rmself_T.update(
                        pred_lf[:, timedelay - 1 : timedelay, 3:],
                        label_lf[:, timedelay - 1 : timedelay, 3:],
                    )
                    rmself_FT.update(
                        pred_lf[:, timedelay - 1 : timedelay],
                        label_lf[:, timedelay - 1 : timedelay],
                    )
                else:
                    crps_F.update(pred[..., :3], label[..., :3])
                    crps_T.update(pred[..., 3:], label[..., 3:])
                    crps_FT.update(pred, label)
                    rmsehf_F.update(pred_hf[..., :3], label_hf[..., :3])
                    rmsehf_T.update(pred_hf[..., 3:], label_hf[..., 3:])
                    rmsehf_FT.update(pred_hf, label_hf)
                    rmself_F.update(pred_lf[..., :3], label_lf[..., :3])
                    rmself_T.update(pred_lf[..., 3:], label_lf[..., 3:])
                    rmself_FT.update(pred_lf, label_lf)

        # point output
        elif pred.ndim == 2:
            if pred_logvar is not None:  # Gaussian point output
                mu = pred_mu[idx]
                logvar = pred_logvar[idx]
                if timedelay:
                    label = label[timedelay:]
                    mu = mu[:-timedelay]
                    logvar = logvar[:-timedelay]
                label_lf = lpf(label)
                label_hf = label - label_lf
                mu_lf = lpf(mu)
                mu_hf = mu - mu_lf
                crps_F.update(mu[..., :3], logvar[..., :3], label[..., :3])
                crps_T.update(mu[..., 3:], logvar[..., 3:], label[..., 3:])
                crps_FT.update(mu, logvar, label)
                rmsehf_F.update_gaussian(mu_hf[:, :3], logvar[:, :3], label_hf[:, :3])
                rmsehf_T.update_gaussian(mu_hf[:, 3:], logvar[:, 3:], label_hf[:, 3:])
                rmsehf_FT.update_gaussian(mu_hf, logvar, label_hf)
                rmself_F.update(mu_lf[:, :3], label_lf[:, :3])
                rmself_T.update(mu_lf[:, 3:], label_lf[:, 3:])
                rmself_FT.update(mu_lf, label_lf)
            else:  # deterministic
                if timedelay:  # apply timedelay
                    label = label[timedelay:]
                    pred = pred[:-timedelay]
                label_lf = lpf(label)
                label_hf = label - label_lf
                pred_lf = lpf(pred)
                pred_hf = pred - pred_lf
                crps_F.update(pred[:, :3], label[:, :3])  # MAE
                crps_T.update(pred[:, 3:], label[:, 3:])  # MAE
                crps_FT.update(pred, label)  # MAE
                rmsehf_F.update(pred_hf[:, :3], label_hf[:, :3])
                rmsehf_T.update(pred_hf[:, 3:], label_hf[:, 3:])
                rmsehf_FT.update(pred_hf, label_hf)
                rmself_F.update(pred_lf[:, :3], label_lf[:, :3])
                rmself_T.update(pred_lf[:, 3:], label_lf[:, 3:])
                rmself_FT.update(pred_lf, label_lf)

    return (
        crps_F.compute(),
        crps_T.compute(),
        crps_FT.compute(),
        rmsehf_F.compute(),
        rmsehf_T.compute(),
        rmsehf_FT.compute(),
        rmself_F.compute(),
        rmself_T.compute(),
        rmself_FT.compute(),
    )


def aggregate_runs(
    dir,
    timedelay=100,
    normalized_scale=False,
    stepwise=True,
    save_dir="exp/test.txt",
    ndigits: int = 3,
    stress_dropout: float = 0,
):
    """Evaluate every run matching dir and append the mean +/- std over runs to save_dir.

    Each block reports CRPS, LF RMSE (pRMSE), and HF RMS-RMSE (wRMSE) for N, Nm, and all channels.
    """
    ckpt_list, key_list = get_unique_keys(dir)
    exp_dict = {
        k: {
            "crps_F": [],
            "crps_T": [],
            "crps_FT": [],
            "rmse_F": [],
            "rmse_T": [],
            "rmse_FT": [],
            "mse_F": [],
            "mse_T": [],
            "mse_FT": [],
        }
        for k in key_list
    }

    for p, k in zip(ckpt_list, key_list):
        model, configs, scale_params = load_model(p)
        if model:
            dataloader = get_dataloader(model, configs, scale_params)

            label_total, pred_total, pred_trend, pred_mu, pred_logvar = (
                infer_with_dataloader(
                    model,
                    dataloader,
                    normalized_scale=normalized_scale,
                    stress_dropout=stress_dropout,
                )
            )

            crps_F, crps_T, crps_FT, rmse_F, rmse_T, rmse_FT, mse_F, mse_T, mse_FT = (
                evaluate_predictions(
                    model,
                    label_total,
                    pred_total,
                    pred_trend,
                    pred_mu,
                    pred_logvar,
                    timedelay=timedelay,
                    stepwise=stepwise,
                )
            )
            exp_dict[k]["crps_F"].append(crps_F)
            exp_dict[k]["crps_T"].append(crps_T)
            exp_dict[k]["crps_FT"].append(crps_FT)
            exp_dict[k]["rmse_F"].append(rmse_F)
            exp_dict[k]["rmse_T"].append(rmse_T)
            exp_dict[k]["rmse_FT"].append(rmse_FT)
            exp_dict[k]["mse_F"].append(mse_F)
            exp_dict[k]["mse_T"].append(mse_T)
            exp_dict[k]["mse_FT"].append(mse_FT)

    # aggregate results
    for k, v in exp_dict.items():
        crps_all_F = torch.stack(v["crps_F"], dim=0)
        crps_all_T = torch.stack(v["crps_T"], dim=0)
        crps_all_FT = torch.stack(v["crps_FT"], dim=0)
        rmse_all_F = torch.stack(v["rmse_F"], dim=0)
        rmse_all_T = torch.stack(v["rmse_T"], dim=0)
        rmse_all_FT = torch.stack(v["rmse_FT"], dim=0)
        mse_all_F = torch.stack(v["mse_F"], dim=0)
        mse_all_T = torch.stack(v["mse_T"], dim=0)
        mse_all_FT = torch.stack(v["mse_FT"], dim=0)

        with open(save_dir, "a") as f:
            f.write(
                f"Experiment: {k}, Timedelay(see only when stepwise==True): {timedelay}, Stepwise: {stepwise}, Normalized Scale: {normalized_scale}, itr: {len(crps_all_F)}, stress_dropout: {stress_dropout:.2f}"
            )
            f.write("\n")
            f.write(
                f"CRPS (N, Nm, All): {crps_all_F.mean():.{ndigits}f} $\pm$ {crps_all_F.std():.{ndigits}f} & {crps_all_T.mean():.{ndigits}f} $\pm$ {crps_all_T.std():.{ndigits}f} & {crps_all_FT.mean():.{ndigits}f} $\pm$ {crps_all_FT.std():.{ndigits}f}"
            )
            f.write("\n")
            f.write(
                f"LF RMSE (N, Nm, All): {mse_all_F.mean():.{ndigits}f} $\pm$ {mse_all_F.std():.{ndigits}f} & {mse_all_T.mean():.{ndigits}f} $\pm$ {mse_all_T.std():.{ndigits}f} & {mse_all_FT.mean():.{ndigits}f} $\pm$ {mse_all_FT.std():.{ndigits}f}"
            )
            f.write("\n")
            f.write(
                f"HF RMS-RMSE (N, Nm, All): {rmse_all_F.mean():.{ndigits}f} $\pm$ {rmse_all_F.std():.{ndigits}f} & {rmse_all_T.mean():.{ndigits}f} $\pm$ {rmse_all_T.std():.{ndigits}f} & {rmse_all_FT.mean():.{ndigits}f} $\pm$ {rmse_all_FT.std():.{ndigits}f}"
            )
            f.write("\n\n")


def visualize_episode_recon_all_models(timedelay: int = 10):
    """Fig. 6: test episode reconstructions of GPR, PatchTST-Gaussian, and FDN."""
    keys = [
        "Baseline_GPR",
        "Baseline_PatchTST_Gaussian",
        "FromScratch",
        "RelPos7D_FineTune",
    ]
    dir_list = [f"exp/runs_v1/{k}/**/E5.pt" for k in keys]

    ckpt_list = []
    key_list = []
    for dir in dir_list:
        ckpt_sublist, key_sublist = get_unique_keys(dir)
        ckpt_list.extend(ckpt_sublist)
        key_list.extend(key_sublist)

    basedir = "analysis/episode_recon_all"
    os.makedirs(basedir, exist_ok=True)
    plot_template()

    selected_run_idx = 0
    model_results = []
    run_count = {k: 0 for k in set(key_list)}
    for p, k in zip(ckpt_list, key_list):
        if "FineTune" in p:
            # if not "ratio0.6" in p:  # Choose pretraining data util.
            continue
        elif "AbsPos" in p:
            continue
        if run_count[k] != selected_run_idx:
            run_count[k] += 1
            continue
        run_count[k] += 1
        model, configs, scale_params = load_model(p)
        if model:
            dataloader = get_dataloader(model, configs, scale_params)
            label_total, pred_total, pred_trend, pred_mu, pred_logvar = (
                infer_with_dataloader(model, dataloader)
            )
            model_results.append(
                {
                    "path":p,
                    "run_key": k.replace("/", "_"),
                    "model_name": model.__class__.__module__.replace("models.", ""),
                    "label_total": label_total,
                    "pred_total": pred_total,
                    "pred_trend": pred_trend,
                    "pred_mu": pred_mu,
                    "pred_logvar": pred_logvar,
                }
            )

    # colors = cm.tab10(np.linspace(0, 1, len(model_results)))
    colors = ["gold", "deepskyblue"]
    legend_labels = ["Label"]
    for result in model_results:
        if "FDN_PatchTST" in result["model_name"]:
            # if 'abspos' in result["path"].lower():
            #     legend_labels.append("FDN (AbsPos)")
            # elif 'finetune' in result["path"].lower():
            #     legend_labels.append("FDN (Pretrained)")
            if not 'abspos' in result["path"].lower() and not 'finetune' in result["path"].lower():
                legend_labels.append("FDN")
        elif result["model_name"] == "PatchTST_Gaussian":
            legend_labels.append("PatchTST-Gaussian")
        else:
            legend_labels.append(result["model_name"])

    fig_legend, ax_legend = plt.subplots(figsize=(2 * len(legend_labels), 1.2))
    lw = 4
    handles = [ax_legend.plot([], [], color="k", lw=lw, label=legend_labels[0])[0]]
    for i, result in enumerate(model_results):
        color_line = "crimson" if "FDN_PatchTST" in result["model_name"] else colors[i]
        handles.append(
            ax_legend.plot(
                [],
                [],
                color=color_line,
                lw=lw,
                alpha=0.9,
                label=legend_labels[i + 1],
            )[0]
        )
    ax_legend.legend(handles=handles, loc="center", ncol=len(handles))
    ax_legend.axis("off")
    fig_legend.tight_layout()
    fig_legend.savefig(
        f"{basedir}/legend_only.png",
        dpi=300,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    print(f"{basedir}/legend_only.png")
    plt.close(fig_legend)

    select_channels = [0, 4]  # Fx, My
    ylabels = ["$F_{\mathbf{x}}$ [N]", "$M_{\mathbf{y}}$ [Nm]"]
    n_episodes = len(model_results[0]["label_total"])
    episode_ylims = []
    for epi_i in range(n_episodes):
        ylim = []
        for ch in select_channels:
            ymin = np.inf
            ymax = -np.inf
            for result in model_results:
                label = result["label_total"][epi_i]
                ptotal = result["pred_total"][epi_i]
                ptrend = (
                    result["pred_trend"][epi_i]
                    if result["pred_trend"] is not None
                    else None
                )
                pmu = (
                    result["pred_mu"][epi_i] if result["pred_mu"] is not None else None
                )
                plogvar = (
                    result["pred_logvar"][epi_i]
                    if result["pred_logvar"] is not None
                    else None
                )

                if ptotal.ndim == 3:
                    label = label[:, timedelay - 1, :]
                    ptotal = ptotal[:, timedelay - 1, :]
                    if ptrend is not None:
                        ptrend = ptrend[:, timedelay - 1, :]
                    if plogvar is not None:
                        pmu = pmu[:, timedelay - 1, :]
                        plogvar = plogvar[:, timedelay - 1, :]
                if plogvar is not None:
                    ptotal = pmu if ptrend is None else ptrend + pmu
                    pstd = plogvar.exp().sqrt()
                    ymin = min(
                        ymin,
                        label[:, ch].cpu().min().item(),
                        (ptotal[:, ch] - 3 * pstd[:, ch]).cpu().min().item(),
                    )
                    ymax = max(
                        ymax,
                        label[:, ch].cpu().max().item(),
                        (ptotal[:, ch] + 3 * pstd[:, ch]).cpu().max().item(),
                    )
                else:
                    ymin = min(
                        ymin,
                        label[:, ch].cpu().min().item(),
                        ptotal[:, ch].cpu().min().item(),
                    )
                    ymax = max(
                        ymax,
                        label[:, ch].cpu().max().item(),
                        ptotal[:, ch].cpu().max().item(),
                    )
            ylim.append((ymin, ymax))
        episode_ylims.append(ylim)

    for i, result in enumerate(model_results):
        n_episodes = len(result["label_total"])
        c_out = result["label_total"][0].size(-1)
        color_line = "coral" if "FDN_PatchTST" in result["model_name"] else colors[i]
        color_fill = (
            "red" if "FDN_PatchTST" in result["model_name"] else color_line
        )  # color for fill_between

        for epi_i in range(n_episodes):
            fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

            label = result["label_total"][epi_i]
            ptotal = result["pred_total"][epi_i]
            ptrend = (
                result["pred_trend"][epi_i]
                if result["pred_trend"] is not None
                else None
            )
            pmu = result["pred_mu"][epi_i] if result["pred_mu"] is not None else None
            plogvar = (
                result["pred_logvar"][epi_i]
                if result["pred_logvar"] is not None
                else None
            )

            if ptotal.ndim == 3:
                label = label[:, timedelay - 1, :]
                ptotal = ptotal[:, timedelay - 1, :]
                if ptrend is not None:
                    ptrend = ptrend[:, timedelay - 1, :]
                if plogvar is not None:
                    pmu = pmu[:, timedelay - 1, :]
                    plogvar = plogvar[:, timedelay - 1, :]
            if plogvar is not None:
                ptotal = pmu if ptrend is None else ptrend + pmu
                pstd = plogvar.exp().sqrt()

            time = np.arange(label.size(0)) / 100
            for ch_i, ch in enumerate(select_channels):
                ax[ch_i].plot(time, label[:, ch].cpu(), color="k", lw=1, zorder=10)
                if plogvar is not None:
                    alpha = 0.7 if "FDN" in result["model_name"] else 0.5
                    ax[ch_i].fill_between(
                        time[-ptotal.shape[0] :],
                        (ptotal[:, ch] - 3 * pstd[:, ch]).cpu(),
                        (ptotal[:, ch] + 3 * pstd[:, ch]).cpu(),
                        color=color_fill,
                        alpha=alpha,
                        zorder=100,
                        edgecolor="none",
                    )
                lw = 0.8 if "FDN" in result["model_name"] else 2
                ax[ch_i].plot(
                    time[-ptotal.shape[0] :],
                    ptotal[:, ch].cpu(),
                    color=color_line,
                    lw=lw,
                    alpha=0.9,
                    zorder=101,
                )
                ax[ch_i].set_axisbelow(True)
                ax[ch_i].set_ylim(*episode_ylims[epi_i][ch_i])
                ax[ch_i].set_ylabel(ylabels[ch_i], fontsize=20)
            ax[-1].set_xlabel("Time [s]")

            fig.tight_layout()
            fig.subplots_adjust(hspace=0.15)
            fig.savefig(
                f"{basedir}/{result['run_key']}_episode{epi_i + 1:02d}_timedelay{timedelay}.png",
                dpi=300,
            )
            print(
                f"{basedir}/{result['run_key']}_episode{epi_i + 1:02d}_timedelay{timedelay}.png"
            )
            plt.close(fig)


# Baseline comparison
def plot_horizon_errors_all_models(horizons=range(1, 101)):
    """Plot normalized, channel-aggregated errors using the first run per model."""
    import csv

    # Edit the values to change the model names shown beside the arrows.
    model_labels = {
        "MINN": "MINN",
        "RBF": "RBF",
        "GPR": "GPR",
        "LSTM": "LSTM",
        "CNN": "CNN",
        "LSTMEncDec": "LSTM-ED",
        "TransformerEncDec": "Transformer",
        "PatchTST": "PatchTST",
        "PatchTST_Gaussian": "PatchTST-Gaussian",
        "iTransformer": "iTransformer",
        "FDN_PatchTST_RelPos6D": "FDN",
        "FDN_PatchTST_RelPos7D": "FDN (Pretrained)",
    }
    # Edit the values to change the complete y-axis labels.
    metric_labels = {
        "HF": r"HF $\mathrm{wRMSE}$ (Normalized)",
        "LF": r"LF $\mathrm{pRMSE}$ (Normalized)",
        "CRPS": "CRPS (Normalized)",
    }
    # Set (ymin, ymax) for each metric; None keeps automatic limits.
    metric_ylims = {
        "HF": (0.4,0.9),
        "LF": (0.4, 0.9),
        "CRPS": None,
    }
    keys = [
        "Baseline_MINN",
        "Baseline_RBF",
        "Baseline_GPR",
        "Baseline_LSTM",
        "Baseline_CNN",
        "Baseline_LSTMEncDec",
        "Baseline_TransformerEncDec",
        "Baseline_PatchTST",
        "Baseline_PatchTST_Gaussian",
        "Baseline_iTransformer",
        "FromScratch",
    ]
    ckpt_list, key_list = [], []
    for k in keys:
        ckpt_sublist, key_sublist = get_unique_keys(f"exp/runs_v1/{k}/**/E5.pt")
        ckpt_list.extend(ckpt_sublist)
        key_list.extend(key_sublist)

    horizons = np.asarray(list(horizons))
    metrics = ["HF", "LF", "CRPS"]
    basedir = "analysis/horizon_errors"
    os.makedirs(basedir, exist_ok=True)

    metric_cache = {metric: {} for metric in metrics}
    for metric in metrics:
        cache_path = f"{basedir}/{metric}.csv"
        if not os.path.exists(cache_path):
            continue
        with open(cache_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        cached_horizons = np.asarray([int(row["horizon"]) for row in rows])
        if not np.array_equal(cached_horizons, horizons):
            print(f"{cache_path}: horizon mismatch, recomputing")
            continue
        for name in reader.fieldnames[1:]:
            metric_cache[metric][name] = np.asarray(
                [float(row[name]) for row in rows]
            )

    def save_metric_cache(metric):
        cache_path = f"{basedir}/{metric}.csv"
        names = list(metric_cache[metric])
        with open(cache_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["horizon", *names])
            for i, horizon in enumerate(horizons):
                writer.writerow(
                    [horizon, *[metric_cache[metric][name][i] for name in names]]
                )

    selected_run_idx = 0
    run_count = {k: 0 for k in set(key_list)}
    results = {}
    for p, k in zip(ckpt_list, key_list):
        if "FDN" in p and "AbsPos" in p:
            continue
        if "FineTune" in p and "ratio0.6" not in p:
            continue
        run_count[k] += 1
        if run_count[k] - 1 != selected_run_idx:
            continue
        name = next(
            name for name in sorted(model_labels, key=len, reverse=True) if name in p
        )
        if all(name in metric_cache[metric] for metric in metrics):
            results[name] = np.column_stack(
                [metric_cache[metric][name] for metric in metrics]
            )
            print(f"{k}: loaded cached metrics")
            continue
        model, configs, scale_params = load_model(p)
        if model is None:
            continue
        print(f"{k}: inference", flush=True)
        dataloader = get_dataloader(model, configs, scale_params)
        predictions = infer_with_dataloader(model, dataloader, normalized_scale=True)
        scores = []
        for h in horizons:
            selected = predictions
            timedelay = int(h)
            if predictions[1][0].ndim == 3:
                # Evaluate only this horizon; filtering runs along the episode.
                selected = tuple(
                    tuple(x[:, h - 1 : h] for x in episodes)
                    if episodes is not None else None
                    for episodes in predictions
                )
                timedelay = 1
            _, _, crps_FT, _, _, rmse_FT, _, _, mse_FT = evaluate_predictions(
                model, *selected, timedelay=timedelay, stepwise=True
            )
            scores.append(torch.stack([rmse_FT, mse_FT, crps_FT]))
            if h % 10 == 0 or h == horizons[-1]:
                print(f"{k}: horizon {h}/{horizons[-1]}", flush=True)
        results[name] = torch.stack(scores).cpu().numpy()
        for i, metric in enumerate(metrics):
            metric_cache[metric][name] = results[name][:, i]
            save_metric_cache(metric)
            print(f"{basedir}/{metric}.csv")
        del selected, predictions, dataloader, model

    plot_template()
    baselines = [name for name in results if "FDN" not in name]
    color_map = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(baselines)}
    color_map.update({name: "black" for name in results if "FDN" in name})
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(7.7,5))
        fig.subplots_adjust(right=0.65)
        endpoints = []
        for (name, values), color in zip(results.items(), color_map):
            color = color_map[name]
            lw=4 if "FDN" in name else 2.5
            alpha =1 if "FDN" in name else 0.7
            ax.plot(horizons, values[:, i], color=color, lw=lw,alpha=alpha)
            endpoints.append((values[-1, i], name, color))
        endpoints.sort(key=lambda item: item[0])
        for (value, name, color), y in zip(
            endpoints, np.linspace(0, 1, len(endpoints))
        ):
            ax.annotate(
                model_labels.get(name, name), xy=(horizons[-1], value),
                xytext=(1.08, y), textcoords="axes fraction",
                color=color, fontsize=20 if "FDN" in name else 13,
                fontweight="bold" if "FDN" in name else "bold", va="center",
                annotation_clip=False,
                arrowprops={"arrowstyle": "->", "color": color, "relpos": (0, 0.5),"lw":1.7},
            )
        ax.set_xlabel("Estimation horizon [steps]", labelpad=10)
        ax.set_xticks([1, 25, 50, 75, 100])
        ax.set_xticklabels(["$t+1$", "$t+25$", "$t+50$", "$t+75$", "$t+100$"])
        ax.set_ylabel(metric_labels[metric])
        conversion1=lambda x:10*x
        conversion2=lambda x:x/10
        secax=ax.secondary_xaxis('top', functions=(conversion1,conversion2))
        secax.set_xticks([10, 250, 500, 750, 1000])
        secax.set_xticklabels
        secax.set_xlabel('Time delay [ms]',labelpad=10)
        if metric_ylims[metric] is not None:
            ax.set_ylim(*metric_ylims[metric])
        fig.tight_layout()
        fig.savefig(f"{basedir}/{metric}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"{basedir}/{metric}.png")
    return results


@torch.inference_mode()
def freq_aware_layers_figure_plots(ckpt_path, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    basedir = "analysis/visualize_data/fef_figure_plots"
    os.makedirs(basedir, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    configs = ckpt["configs"]
    scale_params = ckpt["scale_params"]
    model = FDN_PatchTST_RelPos6D.Model(configs)
    model.load_state_dict(state_dict)
    model.eval()
    dataset = HydraulicDatasetRelPos6D(
        seq_len=configs.seq_len,
        pred_len=configs.pred_len,
        mode="test",
        scale_params=scale_params,
    )
    dataset.scale()
    n_deltas = 24
    n_joints = n_deltas // 4
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=CollateDecompose(),
        shuffle=True,
    )
    moe_weights = []
    for b in dataloader:
        x, y, ytrend, yres, chmask = b
        x_ = x[..., :n_deltas]

        if model.use_rev_in:
            variance, mean = torch.var_mean(x_, dim=1, keepdim=True)
            std = variance.sqrt().clamp_min(1e-6)
            x_ = (x_ - mean) / std

        x_enhanced, weight_gating, filtered_fft = model.freq_enhance(
            x_[..., :n_deltas], output_internals=True
        )
        moe_weights.append(weight_gating)

        model.disable_freq_pass = True
        outputs_tilde = model(x)
        model.disable_freq_pass = False
        outputs_hat = model(x)

        break
    moe_weights = torch.cat(moe_weights, dim=0).mean(dim=(0, 1, 2))
    learned_filters = model.freq_enhance.kernel_weight_real.squeeze(0).unbind(-1)
    vmax = 1.2
    vmin = 0.8

    yticklabels = (
        [f"$\Delta q_{{{i + 1}}}$" for i in range(n_joints)]
        + [f"$\dot{{q}}_{{{i + 1}}}$" for i in range(n_joints)]
        + [f"$\ddot{{q}}_{{{i + 1}}}$" for i in range(n_joints)]
        + [f"$u_{{{i + 1}}}$" for i in range(n_joints)]
    )
    plot_template()

    # Exaggerate for visualization
    gain_display = 150.0
    x_fft = torch.fft.rfft(x_, dim=1)
    filtered_fft = x_fft + (filtered_fft - x_fft) * gain_display
    x_enhanced_ = torch.fft.irfft(filtered_fft, n=x_.shape[1], dim=1)

    # time domain ylims
    x_before = x_[0, :, :n_deltas]
    x_after = x_enhanced_[0]
    ymin = torch.minimum(x_before.amin(dim=0), x_after.amin(dim=0))
    ymax = torch.maximum(x_before.amax(dim=0), x_after.amax(dim=0))
    padding = 0.1 * (ymax - ymin).clamp_min(1e-6)
    ylims = torch.stack([ymin - padding, ymax + padding], dim=-1).tolist()

    # input time series
    fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))
    for i in range(n_deltas):
        ax.plot(x_.squeeze(0)[:, i], lw=6, c="darkred")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(*ylims[i])
        fig.tight_layout()
        fig.savefig(
            f"{basedir}/x_{i+1:02d}.png", dpi=300, bbox_inches="tight", pad_inches=0
        )
        print(f"{basedir}/x_{i+1:02d}.png")
        ax.cla()

    # FFT
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.matshow(x_fft.abs().squeeze(0).T, vmin=0, vmax=20, aspect="auto", cmap="PuRd")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{basedir}/x_fft.png", dpi=300, bbox_inches="tight", pad_inches=0)

    # learned filters
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, f in enumerate(learned_filters):
        f = f.cpu()
        ax.matshow(
            F.softplus(f.T), aspect="auto", vmin=vmin, vmax=vmax, cmap="gist_rainbow"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        fig.tight_layout()
        fig.savefig(
            f"{basedir}/learned_filter_{i + 1:02d}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"{basedir}/learned_filter_{i + 1:02d}.png")
        ax.margins(0)
        ax.cla()

    # Gating weights
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.bar(range(5), F.softmax(torch.randn(5), dim=0), color="coral")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 0.5)
    fig.tight_layout()
    fig.savefig(
        f"{basedir}/moe_weights.png", dpi=300, bbox_inches="tight", pad_inches=0
    )

    # Enhanced FFT
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.matshow(
        filtered_fft.abs().squeeze(0).T, vmin=0, vmax=20, aspect="auto", cmap="Oranges"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(
        f"{basedir}/x_fft_enhanced.png", dpi=300, bbox_inches="tight", pad_inches=0
    )

    # Enhanced input time series
    # original = x_enhanced
    fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))
    for i in range(n_deltas):
        ax.plot(x_enhanced_.squeeze(0)[:, i], lw=6, c="coral")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(*ylims[i])
        fig.tight_layout()
        fig.savefig(
            f"{basedir}/x_enhanced_{i+1:02d}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"{basedir}/x_enhanced_{i+1:02d}.png")
        ax.cla()

    trend_tilde, res_tilde, mu_tilde, logvar_tilde = outputs_tilde
    trend_hat, res_hat, mu_hat, logvar_hat = outputs_hat

    # outputs (time)
    plot_series = [trend_tilde, trend_hat, mu_tilde, mu_hat]
    plot_series = [s.squeeze(0) if s.ndim == 3 else s for s in plot_series]
    fname = ["trend_tilde", "trend_hat", "mu_tilde", "mu_hat"]
    colors = ["lightcoral", "red", "lightcoral", "red"]
    lws = [4, 8, 4, 8]
    # ylims
    ylims = []
    for before, after in (plot_series[:2], plot_series[2:]):
        ymin = torch.minimum(before.amin(dim=0), after.amin(dim=0))
        ymax = torch.maximum(before.amax(dim=0), after.amax(dim=0))
        padding = 0.1 * (ymax - ymin).clamp_min(1e-6)
        limits = torch.stack([ymin - padding, ymax + padding], dim=-1).tolist()
        ylims.extend([limits, limits])
    fig, ax = plt.subplots(6, 1, figsize=(4, 8), gridspec_kw={"hspace": 0, "wspace": 0})
    for a in ax.flat:
        for spine in a.spines.values():
            spine.set_linewidth(5)
    for i, series in enumerate(plot_series):
        for j in range(series.shape[-1]):
            ax[j].plot(series[:, j], lw=lws[i], c=colors[i])
            ax[j].set_xticks([])
            ax[j].set_yticks([])
            ax[j].set_ylim(*ylims[i][j])
        fig.tight_layout()
        fig.savefig(
            f"{basedir}/{fname[i]}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"{basedir}/{fname[i]}.png")
        for a in ax:
            a.cla()

    # outputs (freq)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), gridspec_kw={"hspace": 0, "wspace": 0})
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    for i, series in enumerate(plot_series):
        freq = torch.fft.rfftfreq(series.shape[0], d=1 / 100)
        # ax.axvline(x=torch.where(freq==2)[0].item(), c="k", lw=4)
        # if i>1:
        #     ax.axvline(x=torch.where(freq==16)[0].item(), c="k", lw=4)
        ax.matshow(
            torch.fft.rfft(series, dim=0).abs().T,
            vmin=0,
            vmax=0.4,
            aspect="auto",
            cmap="Reds",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(
            f"{basedir}/{fname[i]}_fft.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"{basedir}/{fname[i]}_fft.png")
        ax.cla()

    # FPF filters low/high
    filt_low = butter_response(100)
    filt_high = (1 - filt_low) * butter_response(100, cutoff_freq=15)
    filt_low = filt_low.reshape(51, -1).repeat(1, n_deltas)
    filt_high = filt_high.reshape(51, -1).repeat(1, n_deltas)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.matshow(filt_low.T, vmin=0, vmax=1, aspect="auto", cmap="bone")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(
        f"{basedir}/filt_low.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    ax.cla()
    ax.matshow(filt_high.T, vmin=0, vmax=1, aspect="auto", cmap="bone")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(
        f"{basedir}/filt_high.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    pass


def main_table2_4():
    """Tables 2-4: baseline comparison at 100 ms and 1,000 ms time delays."""
    save_dir = "results/table2-4.txt"
    normalized_scale = False
    keys = [
        "Baseline_MINN",
        "Baseline_RBF",
        "Baseline_GPR",
        "Baseline_LSTM",
        "Baseline_CNN",
        "Baseline_LSTMEncDec",
        "Baseline_TransformerEncDec",
        "Baseline_PatchTST",
        "Baseline_PatchTST_Gaussian",
        "Baseline_iTransformer",
        "FromScratch",
    ]
    for k in keys:
        runs_root = "exp/runs_v1"
        for timedelay in [10, 100]:
            ndigits = 4 if normalized_scale else 3

            aggregate_runs(
                f"{runs_root}/{k}/**/E5.pt",
                timedelay=timedelay,
                normalized_scale=normalized_scale,
                save_dir=save_dir,
                ndigits=ndigits,
            )
        with open(save_dir, "a") as f:
            f.write("\n" + 100 * "=" + "\n")


def main_table7():
    """Table 7: residual correlation structures (multivariate-Gaussian variants)."""
    save_dir = "results/table7.txt"
    normalized_scale = True
    for key in [
        "FromScratch",
        "FromScratch_MVNChannel",
        "FromScratch_MVNTemporal",
        "FromScratch_MVNKron",
    ]:
        aggregate_runs(
            f"exp/runs_v2/{key}/**/E5.pt",
            timedelay=10,
            normalized_scale=normalized_scale,
            save_dir=save_dir,
            ndigits=3,
            stepwise=False,
        )
        with open(save_dir, "a") as f:
            f.write("\n" + 100 * "=" + "\n")


def main_table6():
    """Table 6: input ablations aggregated over all forecast steps."""
    save_dir = "results/table6.txt"
    for key in [
        "FromScratch",
        "FromScratch_NoJointVel",
        "FromScratch_NoJointAcc",
        "FromScratch_NoJointVelAcc",
        "FromScratch_NoJTorque",
        "FromScratch_JointPosOnly",
    ]:
        aggregate_runs(
            f"exp/runs_v2/{key}/**/E5.pt",
            normalized_scale=True,
            stepwise=False,
            save_dir=save_dir,
            ndigits=3,
        )
        with open(save_dir, "a") as f:
            f.write("\n" + 100 * "=" + "\n")


def main_table5():
    """Table 5: architectural ablations."""
    save_dir = "results/table5.txt"
    stepwise = False
    ndigits = 3
    keys = [
        "FromScratch",
        "Ablation_DetHead",
        "Ablation_FEF",
        "Ablation_FEF_MOE",
        "Ablation_FEF_Weighting",
        "Ablation_FPF",
        "Ablation_ProbHead",
        "Ablation_ModSpecEnc",
    ]
    for k in keys:
        aggregate_runs(
            f"exp/runs_v1/{k}/**/E5.pt",
            normalized_scale=True,
            save_dir=save_dir,
            stepwise=stepwise,
            ndigits=ndigits,
        )
        with open(save_dir, "a") as f:
            f.write("\n" + 100 * "=" + "\n")


def main_table9():
    """Table 9: transfer learning (FromScratch gives the 0% columns)."""
    save_dir = "results/table9.txt"
    stepwise = False
    ndigits = 3
    keys = [
        "FromScratch",
        "RelPos7D_LinProbe",
        "RelPos7D_FineTune",
        "AbsPos7D_LinProbe",
        "AbsPos7D_FineTune",
    ]
    for k in keys:
        aggregate_runs(
            f"exp/runs_v1/{k}/**/E5.pt",
            normalized_scale=True,
            save_dir=save_dir,
            stepwise=stepwise,
            ndigits=ndigits,
        )
        with open(save_dir, "a") as f:
            f.write("\n" + 100 * "=" + "\n")


@torch.inference_mode()
def evaluate_model_wall_times(
    repeats: int = 100,
    warmup: int = 10,
    runs_root: str = "exp/runs_v1",
):
    """Table 8: benchmark single-sample CPU/GPU forward-pass wall times.

    One checkpoint is sufficient per model because checkpoints from repeated runs
    have the same architecture. Times exclude model/input transfers and are
    normalized independently on CPU and GPU so that FDN (scratch) is 1.0.
    """
    import time

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")

    print("Logical CPUs:", os.cpu_count())
    print("Intra-op threads:", torch.get_num_threads())
    print("Inter-op threads:", torch.get_num_interop_threads())
    print(torch.__config__.parallel_info())

    checkpoint_patterns = {
        "MINN": "Baseline_MINN/*/E5.pt",
        "RBF": "Baseline_RBF/*/E5.pt",
        "GPR": "Baseline_GPR/*/E5.pt",
        "LSTM": "Baseline_LSTM/*/E5.pt",
        "CNN": "Baseline_CNN/*/E5.pt",
        "LSTM-ED": "Baseline_LSTMEncDec/*/E5.pt",
        "Transformer": "Baseline_TransformerEncDec/*/E5.pt",
        "PatchTST": "Baseline_PatchTST/*/E5.pt",
        "PatchTST-Gaussian": "Baseline_PatchTST_Gaussian/*/E5.pt",
        "iTransformer": "Baseline_iTransformer/*/E5.pt",
        "FDN (scratch)": "FromScratch/*_FDN_PatchTST_RelPos6D_*/E5.pt",
    }

    def run_forward(model, model_name, x, channel_mask):
        if "FDN" in model_name:
            return model(x, channel_mask=channel_mask)
        if model_name == "GPR":
            return model.inference(x)
        return model(x)

    def wall_time_stats(model, model_name, x, channel_mask, device):
        model = model.to(device)
        x = x.to(device)
        channel_mask = channel_mask.to(device)

        # Warmup removes lazy initialization and, on GPU, CUDA kernel startup.
        for _ in range(warmup):
            run_forward(model, model_name, x, channel_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = np.empty(repeats, dtype=np.float64)
        for repeat_idx in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            run_forward(model, model_name, x, channel_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed[repeat_idx] = time.perf_counter() - start

        mean = float(elapsed.mean())
        std = float(elapsed.std(ddof=1)) if repeats > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
        }

    def count_params(model):
        n = 0
        for name, p in model.named_parameters():
            if "chol_variational_covar" in name:
                # GPyTorch stores each Cholesky factor as a full k x k matrix,
                # but only the lower triangle (incl. diagonal) is a free parameter.
                k = p.shape[-1]
                n_matrices = p.numel() // (k * k)
                n += n_matrices * k * (k + 1) // 2
            else:
                n += p.numel()
        return n

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results = {}
    for display_name, relative_pattern in checkpoint_patterns.items():
        checkpoint_paths = sorted(glob.glob(os.path.join(runs_root, relative_pattern)))
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoint found for {display_name}: "
                f"{os.path.join(runs_root, relative_pattern)}"
            )

        model, configs, scale_params = load_model(checkpoint_paths[0])
        model.eval()
        if model is None:
            raise RuntimeError(f"Failed to load checkpoint for {display_name}.")

        dataloader = get_dataloader(model, configs, scale_params)
        single_sample_loader = DataLoader(
            dataloader.dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=dataloader.collate_fn,
            shuffle=False,
        )
        x, _, _, _, channel_mask = next(iter(single_sample_loader))
        model_name = model.__class__.__module__.replace("models.", "")

        device_times = {}
        for device in devices:
            benchmark_model = model
            if device.type == "cuda":
                # GPyTorch caches device-specific variational tensors during
                # CPU inference, so the GPU benchmark needs a fresh instance.
                benchmark_model, _, _ = load_model(checkpoint_paths[0])
                if benchmark_model is None:
                    raise RuntimeError(
                        f"Failed to reload checkpoint for {display_name} on GPU."
                    )
            device_times[device.type] = wall_time_stats(
                benchmark_model, model_name, x, channel_mask, device
            )
        results[display_name] = device_times
        results[display_name]["n_params"] = count_params(model)

    fdn_times = results["FDN (scratch)"]
    for times in results.values():
        times["relative_cpu"] = times["cpu"]["mean"] / fdn_times["cpu"]["mean"]
        times["relative_gpu"] = (
            times["cuda"]["mean"] / fdn_times["cuda"]["mean"]
            if "cuda" in times
            else None
        )

    print("CPU [ms] CPU rel. Model #Parameters ")
    for display_name, times in results.items():
        cpu_ms = (
            f"{times['cpu']['mean'] * 1e3:.3f} $\\pm$ "
            f"{times['cpu']['std'] * 1e3:.3f}"
        )
        cpu_relative = f"{times['relative_cpu']:.3f}"
        gpu_ms = (
            f"{times['cuda']['mean'] * 1e3:.3f} $\\pm$ "
            f"{times['cuda']['std'] * 1e3:.3f}"
            if "cuda" in times
            else "N/A"
        )
        gpu_relative = (
            f"{times['relative_gpu']:.3f}"
            if times["relative_gpu"] is not None
            else "N/A"
        )
        print(
            f"{display_name} & {cpu_ms} & {cpu_relative} & {times['n_params']:,}\\\\"
            # f" & {gpu_ms:>12} &{gpu_relative:>10} \\\\"
        )

    return results


if __name__ == "__main__":
    import argparse

    TASKS = {
        "table2-4": (main_table2_4, "results/table2-4.txt"),
        "table5": (main_table5, "results/table5.txt"),
        "table6": (main_table6, "results/table6.txt"),
        "table7": (main_table7, "results/table7.txt"),
        "table8": (evaluate_model_wall_times, "results/table8.txt"),
        "table9": (main_table9, "results/table9.txt"),
        "fig3": (lambda: freq_aware_layers_figure_plots(
            "exp/runs_v1/FromScratch/260320-1255_FDN_PatchTST_RelPos6D_c1b4f3de/E5.pt",
            seed=1,
        ), "analysis/visualize_data/fef_figure_plots/"),
        "fig6": (visualize_episode_recon_all_models, "analysis/episode_recon_all/"),
        "fig7": (plot_horizon_errors_all_models, "analysis/horizon_errors/"),
    }
    parser = argparse.ArgumentParser(
        description="Evaluate the trained checkpoints and reproduce the paper tables/figures."
    )
    parser.add_argument("task", choices=list(TASKS), help="paper table/figure to reproduce")
    args = parser.parse_args()

    func, output = TASKS[args.task]
    os.makedirs("results", exist_ok=True)
    if output.endswith(".txt") and result_exists(output, args.task):
        sys.exit(0)

    if args.task == "table8":
        # Single CPU thread, averaged over 1,000 forward passes (Section 6.5)
        torch.set_num_threads(1)
        with open(output, "w") as f:
            sys.stdout = Tee(sys.__stdout__, f)
            evaluate_model_wall_times(repeats=1000, warmup=10)
            sys.stdout = sys.__stdout__
    else:
        func()
    print(f"Saved: {output}")

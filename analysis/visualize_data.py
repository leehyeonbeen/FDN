import os, sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from utils.snippets import plot_template, increase_leglw, Tee, result_exists
from data.dataset import *
from torch.utils.data import DataLoader, ConcatDataset
import torch
from matplotlib import cm
from layers.Filter import butter_response
from scipy.signal import medfilt
from utils.signal import filtbutterworth
import seaborn as sns
from matplotlib.colors import Colormap
from scipy.interpolate import interp1d, UnivariateSpline
from utils.data import extract_trend
from scipy.stats import norm
from matplotlib.colors import LogNorm

plot_template()

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def compute_spectral_energy(dataset_cls=HydraulicDatasetRelPos7D):
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # demean, 5000-steps window fft and 100 stride
    seq_len = 300
    pred_len = 5000
    sampling_freq = 100
    batch_size = 512
    if "Hydraulic" in dataset_cls.__name__:
        vmax = 1e-3
        vmin = 1e-7
        cmap = "Purples"
    else:
        vmax = 1e-1
        vmin = 1e-6
        cmap = "Reds"
    dataset1 = dataset_cls(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="train",
        sampling_freq=sampling_freq,
        index_stride=100,
    )
    # Normalize with the statistics of the denoised training data, also for the raw wrench (Fig. 5(a))
    load_raw_ft = os.environ.get("FDN_LOAD_RAW_FT", "0")
    os.environ["FDN_LOAD_RAW_FT"] = "0"
    dataset1.scale_params = dataset_cls(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="train",
        sampling_freq=sampling_freq,
        index_stride=100,
    ).scale_params
    os.environ["FDN_LOAD_RAW_FT"] = load_raw_ft
    dataset1.scale()
    dataloader = DataLoader(
        # ConcatDataset([dataset1, dataset2]),
        dataset1,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CollateDecompose(),
        num_workers=4,
        drop_last=False,
    )

    X = torch.zeros(seq_len // 2 + 1, 28).to(device)
    F = torch.zeros(pred_len // 2 + 1, 6).to(device)
    Fres = torch.zeros(pred_len // 2 + 1, 6).to(device)
    n_samples = 0
    for i, b in enumerate(dataloader):
        x, f, f_trend, f_res, ch_mask = b
        x = x[..., :28]
        x = x.to(device)
        f = f.to(device)
        f_res = f_res.to(device)

        x = x - x.mean(dim=1, keepdim=True)
        f = f - f.mean(dim=1, keepdim=True)
        f_res = f_res - f_res.mean(dim=1, keepdim=True)

        x_fft = torch.fft.rfft(x, dim=1, norm="forward")
        f_fft = torch.fft.rfft(f, dim=1, norm="forward")
        fres_fft = torch.fft.rfft(f_res, dim=1, norm="forward")
        X += torch.square(x_fft.abs()).sum(dim=0)
        F += torch.square(f_fft.abs()).sum(dim=0)
        Fres += torch.square(fres_fft.abs()).sum(dim=0)
        n_samples += x.shape[0]
        print(f"Batch {i + 1}/{len(dataloader)}")
    X = X.cpu() / n_samples
    F = F.cpu() / n_samples
    Fres = Fres.cpu() / n_samples
    X[1:-1] *= 2
    F[1:-1] *= 2
    Fres[1:-1] *= 2

    freq_x = torch.fft.rfftfreq(
        seq_len,
        d=1.0 / sampling_freq,
    )
    freq_f = torch.fft.rfftfreq(
        pred_len,
        d=1.0 / sampling_freq,
    )
    ylabels_inputs = [f"$q_{i}$" for i in range(1, 8)]
    ylabels_inputs += [r"$\dot{q}_" + f"{i}$" for i in range(1, 8)]
    ylabels_inputs += [r"$\ddot{q}_" + f"{i}$" for i in range(1, 8)]
    ylabels_inputs += [r"$\Delta p_" + f"{i}$" for i in range(1, 8)]

    ylabels_outputs = ["$F_{\mathbf{" + i + "}}$" for i in ["x", "y", "z"]]
    ylabels_outputs += ["$M_{\mathbf{" + i + "}}$" for i in ["x", "y", "z"]]
    ylabels_outputsres = [
        "$F^{\mathrm{res}}_{\mathbf{" + i + "}}$" for i in ["x", "y", "z"]
    ]
    ylabels_outputsres += [
        "$M^{\mathrm{res}}_{\mathbf{" + i + "}}$" for i in ["x", "y", "z"]
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.matshow(
        X.T,
        cmap=cmap,
        vmax=vmax,
        # norm=LogNorm(vmin=vmin,vmax=vmax)
    )

    ax.set_xticks(
        [
            torch.nonzero(freq_x == 1).item(),
            torch.nonzero(freq_x == 15).item(),
            freq_x.shape[0] - 1,
        ]
    )
    ax.axvline(torch.nonzero(freq_x == 1).item(), color="gold", lw=3, alpha=0.8)
    ax.axvline(torch.nonzero(freq_x == 15).item(), color="gold", lw=3, alpha=0.8)
    ax.set_aspect("auto")
    ax.set_xticklabels(
        [
            "$f_c=1$",
            "$f_{c}^{\mathrm{dn}}=15$",
            r"$f_{\mathrm{Nyq}}=" + f"{int(freq_x[-1].item())}$",
        ],
        fontsize=16,
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_yticks(range(len(ylabels_inputs)))
    ax.set_yticklabels(ylabels_inputs)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power spectrum", rotation=270, labelpad=20)
    fig.tight_layout()
    fig.savefig(f"{dir}/energy_inputs_{dataset_cls.__name__}.png", dpi=300)
    print(f"{dir}/energy_inputs_{dataset_cls.__name__}.png")

    fig, ax = plt.subplots(figsize=(6, 3), layout="compressed")
    im = ax.matshow(F.T, cmap=cmap, vmax=vmax, aspect="auto")
    ax.set_box_aspect(0.5)
    ax.set_xticks(
        [
            torch.nonzero(freq_f == 1).item(),
            torch.nonzero(freq_f == 15).item(),
            freq_f.shape[0] - 1,
        ]
    )
    ax.set_xticklabels([])
    # ax.axvline(torch.nonzero(freq_f == 1).item(), color="gold", lw=3, alpha=0.8)
    # ax.axvline(torch.nonzero(freq_f == 15).item(), color="gold", lw=3, alpha=0.8)
    # ax.set_xticklabels(
    #     [
    #         "$f_c=1$",
    #         "$f_{c}^{\mathrm{dn}}=15$",
    #         r"$f_{\mathrm{Nyq}}=" + f"{int(freq_f[-1].item())}$",
    #     ],
    #     fontsize=16,
    # )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_yticks(range(len(ylabels_outputs)))
    ax.set_yticklabels(ylabels_outputs, fontsize=16)
    ax.set_ylabel("Wrench components")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power spectrum", rotation=270, labelpad=25, fontsize=12)
    # fig.tight_layout()
    fig.savefig(f"{dir}/energy_forces_{dataset_cls.__name__}.png", dpi=300)
    print(f"{dir}/energy_forces_{dataset_cls.__name__}.png")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.matshow(
        Fres.T,
        cmap=cmap,
        vmax=vmax,
    )
    ax.axvline(torch.nonzero(freq_f == 1).item(), color="gold", lw=3, alpha=0.8)
    ax.axvline(torch.nonzero(freq_f == 15).item(), color="gold", lw=3, alpha=0.8)
    ax.set_aspect("auto")
    ax.set_xticks(
        [
            torch.nonzero(freq_f == 1).item(),
            torch.nonzero(freq_f == 15).item(),
            freq_f.shape[0] - 1,
        ]
    )
    ax.set_xticklabels(
        [
            "$f_c=1$",
            "$f_{c}^{\mathrm{dn}}=15$",
            r"$f_{\mathrm{Nyq}}=" + f"{int(freq_f[-1].item())}$",
        ],
        fontsize=16,
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_yticks(range(len(ylabels_outputsres)))
    ax.set_yticklabels(ylabels_outputsres, fontsize=16)
    ax.set_ylabel("Wrench components")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spectral energy", rotation=270, labelpad=20, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{dir}/energy_forces_res_{dataset_cls.__name__}.png", dpi=300)
    print(f"{dir}/energy_forces_res_{dataset_cls.__name__}.png")


def concat_force_input_seq_and_show():
    os.makedirs(f"{dir}", exist_ok=True)
    dataset = HydraulicDatasetRelPos7D(
        seq_len=100, pred_len=100, mode="train", train_num_episodes=1
    )  # 1 episode
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=CollateDecompose())
    x = []
    f_trend = []
    f_res = []
    f_total = []
    for i, d in enumerate(dataloader):
        if i % 100 == 0:
            i, f, ftrend, fres, chmask = d
            x.append(i.squeeze(0))
            f_trend.append(ftrend.squeeze(0))
            f_res.append(fres.squeeze(0))
            f_total.append(f.squeeze(0))
    x = torch.cat(x, dim=0)
    f_trend = torch.cat(f_trend, dim=0)
    f_res = torch.cat(f_res, dim=0)
    f_total = torch.cat(f_total, dim=0)
    t = np.arange(0, f_total.shape[0]) / 100

    plot_template(14)
    names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    ylabels_diff0 = [r"$F_{\mathbf{{" + i + r"}}}$ [N]" for i in ["x", "y", "z"]] + [
        r"$M_{\mathbf{{" + i + r"}}}$ [Nm]" for i in ["x", "y", "z"]
    ]

    ylims = [(-250, 250)] * 3 + [(-30, 30)] * 3
    for _ in range(f_total.shape[1]):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, f_total[:, _].numpy(), c="black", lw=1, label="Denoised")
        ax.plot(
            t,
            f_res[:, _].numpy(),
            c="cornflowerblue",
            lw=1.2,
            alpha=0.6,
            label=r"Residual",
        )
        ax.plot(
            t,
            f_trend[:, _].numpy(),
            c="red",
            lw=2,
            label=r"Trend",
        )
        ax.set_ylabel(ylabels_diff0[_], fontsize=18)
        ax.set_xlabel("Time [s]", fontsize=14)
        ax.set_ylim(ylims[_])
        if _ < 3:
            leg = ax.legend(loc="lower left", fontsize=13, framealpha=0.6)
        else:
            leg = ax.legend(loc="upper left", fontsize=13, framealpha=0.6)
        increase_leglw(leg=leg, linewidth=5)
        fig.tight_layout()
        fig.savefig(f"{dir}/FT_seq_{names[_]}.png", dpi=300)
        print(f"{dir}/FT_seq_{names[_]}.png")
        plt.close("all")

    # J+DiffP
    tgt_j = 3  # j4
    ylabels = [
        f"$\Delta q_{tgt_j+1}$ [rad]",
        f"$\dot{{q}}_{tgt_j+1}$ [rad/s]",
        f"$\ddot{{q}}_{tgt_j+1}$ [rad/s$^2$]",
        f"$\Delta p_{tgt_j+1}$ [bar]",
    ]
    lws = [2, 0.5, 0.5, 2]
    ylims = [(-0.3, 0.05), (-0.1, 0.1), (-0.8, 0.8), (-0.5, 1.5)]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for _ in range(4):
        c = "blue" if _ != 3 else "green"
        ax.plot(t, x[:, tgt_j + 7 * _], c=c, lw=lws[_])
        ax.set_xlabel("Time [s]", fontsize=16)
        ax.set_ylabel(ylabels[_], fontsize=20)
        ax.set_ylim(*ylims[_])
        fig.tight_layout()
        if _ != 3:
            fname = f"analysis/visualize_data/IN_seq_j{tgt_j+1}_d{_}.png"
        else:
            fname = f"analysis/visualize_data/IN_seq_j{tgt_j+1}_diffp.png"
        fig.savefig(fname, dpi=300)
        print(fname)
        ax.cla()
        plt.close("all")


def residual_distribution():
    os.makedirs(f"{dir}", exist_ok=True)
    dataset = HydraulicDatasetRelPos7D(
        seq_len=300, pred_len=100, mode="train", train_ratio=1, index_stride=100
    )  # 1 episode
    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4, collate_fn=CollateDecompose()
    )
    f_trend = []
    f_res = []
    f_total = []
    for i, d in enumerate(dataloader):
        i, f, ftrend, fres, chmask = d
        f_trend.append(ftrend)
        f_res.append(fres)
        f_total.append(f)
    f_trend = torch.cat(f_trend, dim=0).flatten(0, 1)
    f_res = torch.cat(f_res, dim=0).flatten(0, 1)
    f_total = torch.cat(f_total, dim=0).flatten(0, 1)
    t = np.arange(0, f_total.shape[0]) / 100

    f_res = (f_res - f_res.mean(dim=0, keepdim=True)) / f_res.std(dim=0, keepdim=True)

    f_res_force_df = pd.DataFrame(
        f_res.numpy()[:, :3],
        columns=[
            r"$F_{\mathbf{x}}^{\text{res}}$",
            r"$F_{\mathbf{y}}^{\text{res}}$",
            r"$F_{\mathbf{z}}^{\text{res}}$",
        ],
    )
    f_res_torque_df = pd.DataFrame(
        f_res.numpy()[:, 3:],
        columns=[
            r"$M_{\mathbf{x}}^{\text{res}}$",
            r"$M_{\mathbf{y}}^{\text{res}}$",
            r"$M_{\mathbf{z}}^{\text{res}}$",
        ],
    )

    grids = np.linspace(0.5, 1.0, 3)
    colors_force = plt.get_cmap("Blues")(grids).tolist()
    colors_torque = plt.get_cmap("Purples")(grids).tolist()
    sns_kwargs = dict(
        stat="density",
        alpha=0.5,
        bins=700,
        kde=False,
        # line_kws={"linewidth": 3, "alpha": 1},  # kde lines
        palette=colors_force,
        element="step",
    )

    force_quantiles = np.quantile(f_res_force_df.values, [0.01, 0.99], axis=0)
    torque_quantiles = np.quantile(f_res_torque_df.values, [0.01, 0.99], axis=0)

    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1 = sns.histplot(f_res_force_df, **sns_kwargs)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.set_xlabel("Normalized force [N]")
    # ax1.set_xlim(force_quantiles.min(), force_quantiles.max())
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 0.5)
    fig1.tight_layout()

    sns_kwargs.update(dict(palette=colors_torque))
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2 = sns.histplot(f_res_torque_df, **sns_kwargs)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.set_xlabel("Normalized torque [Nm]")
    # ax2.set_xlim(torque_quantiles.min(), torque_quantiles.max())
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 0.5)
    fig2.tight_layout()

    # label Gaussian pdf
    x = np.linspace(-5, 5, 1001, endpoint=True)
    # colors_force = iter(plt.get_cmap("Blues")(grids))
    # colors_torque = iter(plt.get_cmap("Reds")(grids))

    mean = f_res_force_df.mean(axis=0).values
    std = f_res_force_df.std(axis=0).values
    for i in range(f_res_force_df.shape[-1]):
        ax1.plot(x, norm.pdf(x, loc=mean[i], scale=std[i]), c=colors_force[i])

    mean = f_res_torque_df.mean(axis=0).values
    std = f_res_torque_df.std(axis=0).values
    for i in range(f_res_torque_df.shape[-1]):
        ax2.plot(x, norm.pdf(x, loc=mean[i], scale=std[i]), c=colors_torque[i])

    sns.move_legend(ax1, loc="upper right")
    sns.move_legend(ax2, loc="upper right")

    fig1.savefig(f"{dir}/residual_force_distribution.png", dpi=300)
    fig2.savefig(f"{dir}/residual_torque_distribution.png", dpi=300)
    print(f"{dir}/residual_force_distribution.png")
    print(f"{dir}/residual_torque_distribution.png")


def detect_contacts():
    os.makedirs(f"{dir}/detect_contacts", exist_ok=True)
    timestamp = sorted(glob.glob(f"data/data_hydraulic/processed/npy_timestamp/{PAPER_EPISODES}.npy"))
    jointpos = sorted(glob.glob(f"data/data_hydraulic/processed/npy_jointpos6/{PAPER_EPISODES}.npy"))
    toolpos = sorted(glob.glob(f"data/data_hydraulic/processed/npy_toolpos/{PAPER_EPISODES}.npy"))
    toolpos_0 = sorted(glob.glob(f"data/data_hydraulic/processed/npy_toolpos_0/{PAPER_EPISODES}.npy"))
    imu = sorted(glob.glob(f"data/data_hydraulic/processed/npy_imu/{PAPER_EPISODES}.npy"))
    ft = sorted(glob.glob(f"data/data_hydraulic/processed/npy_ft/{PAPER_EPISODES}.npy"))

    # fig1, ax1 = plt.subplots()
    toolpos_index = 2  # xyz quat
    excavation_duration_total = 0
    for t, j, tp, tp0, i, f in zip(timestamp, jointpos, toolpos, toolpos_0, imu, ft):
        episode_name = os.path.basename(t).split(".")[0]
        save_path = f"{dir}/detect_contacts/{episode_name}.png"
        data_t = np.load(t).flatten()
        data_t = data_t - data_t[0]
        data_j = np.load(j)
        data_tp = np.load(tp)
        data_tp0 = np.load(tp0)
        # data_tp = data_tp + data_tp0
        data_f = load_aligned_wrench(f)
        time_length = np.ptp(data_t, axis=0).item()

        f_s = data_t.shape[0] / time_length

        # static ranges
        if ("1st_Test" in f) or ("2nd_Test_6" in f):
            t1 = 10
            t2 = 20
        elif "2nd_Test_5" in f:
            t1 = 70
            t2 = 80
        else:
            t1 = 50
            t2 = 60

        initial_static_idx = np.where((t1 <= data_t) & (data_t <= t2))[0]
        # initial_phase = np.where(data_t <= 5)[0][-1]
        initial_phase = round(0.1 * data_t.shape[0])
        data_f_energy = np.linalg.norm(data_f, axis=-1) ** 2
        data_f_energy_rolling = (
            pd.DataFrame(data_f_energy)
            .rolling(window=int(round(f_s)), center=True)
            .mean()
            .to_numpy()
            .flatten()
        )

        mean = data_f_energy_rolling[initial_static_idx].mean()
        std = data_f_energy_rolling[initial_static_idx].std()
        threshold = mean + 3 * std
        thresholded_mask = (data_f_energy_rolling >= threshold) & np.isfinite(
            data_f_energy_rolling
        )
        N_consecutive = int(round(f_s * 5))  # 5 sec
        cnt = np.convolve(
            thresholded_mask.astype(np.int32),
            np.ones(N_consecutive, dtype=np.int32),
            mode="valid",
        )
        c_start = np.where(
            (cnt >= N_consecutive) & (np.arange(cnt.size) > initial_static_idx[-1])
        )[0][
            0
        ]  # Contact Start: consecutively exceed threshold for 5 sec for the first time + after initial static phase

        z_start = data_tp[c_start, 2]
        z_excavating_indices = np.where(data_tp[:, 2] <= z_start)[0]
        c_end = z_excavating_indices[
            -1
        ]  # Contact End: if the tool z goes back to the initial contact point

        t_c_start = data_t[c_start]
        t_c_end = data_t[c_end]
        excavation_length = np.ptp(data_tp[c_start:c_end, 0])
        excavation_depth = np.ptp(data_tp[c_start:c_end, 2])
        excavation_duration = t_c_end - t_c_start

        fig, ax = plt.subplots(figsize=(7, 5))
        ax_ = ax.twinx()
        ax_.plot(data_t, data_tp[:, 0], c="red", label="Tx")  # Tx
        ax_.plot(data_t, data_tp[:, 2], c="blue", label="Tz")  # Tz
        ax.plot(
            data_t, data_f_energy_rolling, c="green", label="Energy"
        )  # Wrench energy (windowed)
        ax.hlines(
            threshold, 0, time_length, color="black", linestyle="--", label="Threshold"
        )  # threshold
        ax.axvspan(
            xmin=data_t[c_start],
            xmax=data_t[c_end],
            alpha=0.2,
            color="black",
        )  # detected contact phase
        ax.axvline(x=data_t[c_start], color="black")  # initial phase
        ax.axvline(x=data_t[c_end], color="black")  # initial phase
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Wrench Energy (windowed)")
        ax_.set_ylabel("Tool Position (x,z) [m]", rotation=270, labelpad=20)
        ax_.set_ylim(-0.25, 0.1)
        ax.set_title(
            f"Contact in {round(t_c_start)}~{round(t_c_end)}s, x-Length: {excavation_length:.3f}m, z-Depth:{excavation_depth:.3f}m",
            fontsize=12,
        )

        # legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        leg = ax.legend(handles, labels, loc="upper right", framealpha=0.7, fontsize=12)
        increase_leglw(leg)

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"{save_path}")
        print(f" & {excavation_duration:.2f}")
        print(f" & {excavation_length * 1000:.2f} & {excavation_depth * 1000:.2f} &")
        print(f" & E[v_x] = {-excavation_length * 1000 / excavation_duration:.2f} mm/s")
        excavation_duration_total += excavation_duration
    print(
        f"Total excavation duration across all episodes: {excavation_duration_total:.2f}s"
    )
    plt.close("all")


PAPER_EPISODES = "Ground_[12]*"  # the 12 episodes of the paper (Soft: 1st, Stiff: 2nd session)


def load_aligned_wrench(path_ft: str):
    """Bias-aligned wrench before denoising (used for Table 1).

    Loads npy_ft_raw and subtracts the idle-state mean exactly as in
    data/process_data_hydraulic.py:process_offsets_outliers_denoise_diff(),
    without the subsequent 15 Hz denoising applied to npy_ft.
    """
    path_raw = path_ft.replace("npy_ft", "npy_ft_raw")
    ft = np.load(path_raw)
    time = np.load(path_raw.replace("npy_ft_raw", "npy_timestamp"))[:, 0]
    time = time - time[0]
    if ("1st_Test" in path_raw) or ("2nd_Test_6" in path_raw):
        t1 = 10
        t2 = 20
    elif "2nd_Test_5" in path_raw:
        t1 = 70
        t2 = 80
    else:
        t1 = 50
        t2 = 60
    static_idx = np.where((t1 <= time) & (time <= t2))[0]
    return ft - ft[static_idx, :].mean(axis=0)


def recheck_max_wrench_magnitudes():
    npy_ft_list = sorted(
        glob.glob(f"data/data_hydraulic/processed/npy_ft/{PAPER_EPISODES}.npy")
    )

    print("Episode & max|F| [N] & max|M| [Nm]")
    for d_ft in npy_ft_list:
        ft = load_aligned_wrench(d_ft)
        force_names = {
            0: "F_{\mathbf{x}}",
            1: "F_{\mathbf{y}}",
            2: "F_{\mathbf{z}}",
        }
        torque_names = {0: "M_{\mathbf{x}}", 1: "M_{\mathbf{y}}", 2: "M_{\mathbf{z}}"}
        force_max = np.abs(ft[:, :3]).max(axis=0)
        force_argmax = np.argmax(force_max)
        torque_max = np.abs(ft[:, 3:6]).max(axis=0)
        torque_argmax = np.argmax(torque_max)

        print(
            f"{os.path.basename(d_ft)[:-4]} & ${force_names[force_argmax]}={int(force_max[force_argmax].round())}$ & ${torque_names[torque_argmax]}={int(torque_max[torque_argmax].round())}$ \\\\"
        )


def analyze_band_energy_ratio():
    # demean, 5000-steps window fft and 100 stride
    pretrain_ckpt = torch.load(
        "exp/runs_v1/RelPos7D_Pretrain/260320-2044_FDN_PatchTST_RelPos7D_train_ratio1.0_b5039f72/IT100K.pt",
        map_location="cpu",
        weights_only=False,
    )
    seq_len = 1
    pred_len = 5000

    pretrain_scale_params = pretrain_ckpt["scale_params"]
    dataset_pretrain = RH20TDatasetRelPos7D(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="train",
        train_ratio=1,
        index_stride=100,
        scale_params=pretrain_scale_params,
    ) # no test set
    data_h1 = HydraulicDatasetRelPos7D(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="train",
        train_ratio=1,
        index_stride=100,
    )
    data_h2 = HydraulicDatasetRelPos7D(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="test",
        train_ratio=1,
        index_stride=100,
    )
    dataset_hydraulic = ConcatDataset([data_h1, data_h2])

    dataloader_pretrain = DataLoader(
        dataset_pretrain, batch_size=512, num_workers=4, collate_fn=CollateDecompose()
    )
    dataloader_hydraulic = DataLoader(
        dataset_hydraulic, batch_size=512, num_workers=4, collate_fn=CollateDecompose()
    )

    energy_pretrain = torch.zeros(pred_len // 2 + 1, 6, device=device)
    energy_hydraulic = torch.zeros(pred_len // 2 + 1, 6, device=device)
    n_samples_pretrain = 0
    n_samples_hydraulic = 0
    freq = torch.fft.rfftfreq(pred_len, d=1 / 100, device=device)
    for i, b in enumerate(dataloader_pretrain):
        x, y, ytrend, yres, chmask = b
        y = y.to(device)
        y = y - y.mean(dim=1, keepdim=True)
        y_fft = torch.fft.rfft(y, dim=1)
        n_samples_pretrain += y.size(0)
        energy_pretrain += (y_fft.abs() ** 2).sum(dim=0)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(dataloader_pretrain)}")
    energy_pretrain = energy_pretrain / n_samples_pretrain
    lf_ratio_pretrain = (
        (energy_pretrain[freq <= 1].sum(dim=0) / energy_pretrain.sum(dim=0) * 100)
        .mean()
        .item()
    )

    for i, b in enumerate(dataloader_hydraulic):
        x, y, ytrend, yres, chmask = b
        y = y.to(device)
        y = y - y.mean(dim=1, keepdim=True)
        y_fft = torch.fft.rfft(y, dim=1)
        n_samples_hydraulic += y.size(0)
        energy_hydraulic += (y_fft.abs() ** 2).sum(dim=0)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(dataloader_hydraulic)}")
    energy_hydraulic = energy_hydraulic / n_samples_hydraulic
    lf_ratio_hydraulic = (
        (energy_hydraulic[freq <= 1].sum(dim=0) / energy_hydraulic.sum(dim=0) * 100)
        .mean()
        .item()
    )

    print(
        f"Band energy ratio (RH20T): {lf_ratio_pretrain:.3f}\% & {100 - lf_ratio_pretrain:.3f}\% "
    )
    print(
        f"Band energy ratio (Hydraulic): {lf_ratio_hydraulic:.3f}\% & {100 - lf_ratio_hydraulic:.3f}\% "
    )
    pass


def correlation_analysis():
    seq_len = 100
    pred_len = 100
    dataset_hydraulic = HydraulicDatasetRelPos7D(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="train",
        train_ratio=1,
        index_stride=1,
    )
    dataset_hydraulic.scale()

    dataloader_hydraulic = DataLoader(
        dataset_hydraulic, batch_size=512, num_workers=4, collate_fn=CollateDecompose()
    )
    yres_concat = []

    for i, b in enumerate(dataloader_hydraulic):
        x, y, ytrend, yres, chmask = b
        yres = yres.to(device)
        yres_concat.append(yres)
    yres_concat = torch.cat(yres_concat, dim=0)

    # computing autocorrelation using FFT
    yres_concat = yres_concat - yres_concat.mean(dim=1, keepdim=True)  # center
    yres_concat_pad = F.pad(yres_concat, (0, 0, 0, pred_len))  # pad at the end
    yres_concat_fft = torch.fft.fft(
        yres_concat_pad, n=2 * pred_len, dim=1
    )  # fft along time
    yres_concat_power = yres_concat_fft * yres_concat_fft.conj()  # power spectrum
    acorr = torch.fft.ifft(yres_concat_power, dim=1).real[
        :, :pred_len, :
    ]  # remove padded part
    acorr = acorr.sum(dim=0)
    acorr = acorr / acorr[0:1, :]  # normalize by zero-lag

    yres_corr_c = []
    for t in yres_concat.unbind(-2):
        yres_corr_c.append(torch.corrcoef(t.T))
    yres_corr_c = torch.mean(torch.stack(yres_corr_c), dim=0)

    plot_template(20)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ylabels_outputs = [
        "$F_{\mathbf{" + i + "}}^{\mathrm{res}}$" for i in ["x", "y", "z"]
    ]
    ylabels_outputs += [
        "$M_{\mathbf{" + i + "}}^{\mathrm{res}}$" for i in ["x", "y", "z"]
    ]
    im = ax.matshow(yres_corr_c.cpu(), vmax=1, vmin=-1, cmap="seismic")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label="Channel correlation", rotation=270, labelpad=30, fontsize=18)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=14)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(ylabels_outputs, fontsize=20)
    ax.set_yticklabels(ylabels_outputs, fontsize=20)
    fig.tight_layout()
    fig.savefig("analysis/visualize_data/corr_channel.png", dpi=300)
    print("analysis/visualize_data/corr_channel.png")

    grids = np.linspace(0.4, 0.9, 3)
    colors_force = iter(plt.get_cmap("Blues")(grids))
    colors_torque = iter(plt.get_cmap("Reds")(grids))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.axhline(0, c="k", ls="--")
    for i in range(3):
        ax.plot(
            acorr.cpu()[:, i],
            lw=2.5,
            c=next(colors_force),
            label=ylabels_outputs[i],
            alpha=0.7,
        )
    for i in range(3):
        ax.plot(
            acorr.cpu()[:, i + 3],
            lw=2.5,
            c=next(colors_torque),
            label=ylabels_outputs[i + 3],
            alpha=0.7,
        )
    leg = ax.legend(loc=4, framealpha=0.2, ncols=2, fontsize=20)
    increase_leglw(leg, linewidth=5)
    ax.set_ylim(-1, 1)
    ax.set_xticks([0, 20, 40, 60, 80, 99])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.set_ylabel("Temporal autocorrelation", fontsize=20)
    ax.set_xlabel("Time lag [steps]", labelpad=10, fontsize=18)
    conversion1=lambda x:10*x
    conversion2=lambda x:x/10
    secax=ax.secondary_xaxis('top', functions=(conversion1,conversion2))
    secax.set_xticks([0, 200, 400, 600, 800, 990])
    secax.set_xticklabels
    secax.set_xlabel('Time delay [ms]',labelpad=10, fontsize=18)
    fig.tight_layout()
    fig.savefig("analysis/visualize_data/corr_temp_biased.png", dpi=300)
    print("analysis/visualize_data/corr_temp_biased.png")

    # ax[2].plot(acorr_unbiased.cpu())

    # # ax[0].set_title('Interchannel cross-correlation')
    # ax[1].set_title('Biased temporal autocorrelation')
    # ax[2].set_title('Unbiased temporal autocorrelation')

    # ax[1].set_ylim(-1,1)
    # ax[2].set_ylim(-1,1)

    pass


def compare_energy_distribution():
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # Uncomment in _load_episode_from_datafiles to display raw, undenoised result!
    # demean, 5000-steps window fft and 100 stride
    seq_len = 300
    pred_len = 5000
    sampling_freq = 100
    batch_size = 512
    dataset1 = RH20TDatasetRelPos7D(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="train",
        sampling_freq=sampling_freq,
        index_stride=100,
    )
    dataset1.scale()
    dataloader1 = DataLoader(
        # ConcatDataset([dataset1, dataset2]),
        dataset1,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CollateDecompose(),
        num_workers=4,
        drop_last=False,
    )
    dataset2 = HydraulicDatasetRelPos7D(
        seq_len=seq_len,
        pred_len=pred_len,
        mode="train",
        sampling_freq=sampling_freq,
        index_stride=100,
    )
    dataset2.scale()
    dataloader2 = DataLoader(
        # ConcatDataset([dataset1, dataset2]),
        dataset2,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CollateDecompose(),
        num_workers=4,
        drop_last=False,
    )

    F1 = torch.zeros(pred_len // 2 + 1, 6).to(device)
    n_samples = 0
    for i, b1 in enumerate(dataloader1):
        x, f1, f_trend, f_res, ch_mask = b1
        f1 = f1.to(device)
        f1 = f1 - f1.mean(dim=1, keepdim=True)
        f1_fft = torch.fft.rfft(f1, dim=1, norm="forward")
        F1 += torch.square(f1_fft.abs()).sum(dim=0)
        n_samples += x.shape[0]
        print(f"Batch {i + 1}/{len(dataloader1)}")
    F1 = F1.cpu() / n_samples
    F1[1:-1] *= 2

    F2 = torch.zeros(pred_len // 2 + 1, 6).to(device)
    n_samples = 0
    for i, b2 in enumerate(dataloader2):
        x, f2, f_trend, f_res, ch_mask = b2
        f2 = f2.to(device)
        f2 = f2 - f2.mean(dim=1, keepdim=True)
        f2_fft = torch.fft.rfft(f2, dim=1, norm="forward")
        F2 += torch.square(f2_fft.abs()).sum(dim=0)
        n_samples += x.shape[0]
        print(f"Batch {i + 1}/{len(dataloader2)}")
    F2 = F2.cpu() / n_samples
    F2[1:-1] *= 2

    freq_f = torch.fft.rfftfreq(
        pred_len,
        d=1.0 / sampling_freq,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        freq_f[1:],
        F1[1:].mean(dim=-1),
        lw=3,
        c="dodgerblue",
        label="Pretraining (RH20T)",
    )
    ax.plot(
        freq_f[1:],
        F2[1:].mean(dim=-1),
        lw=2,
        c="purple",
        label="Downstream (Hydraulic)",
    )

    ax.fill_between(freq_f[1:], F1[1:].mean(dim=-1), color="dodgerblue", alpha=0.1)
    ax.fill_between(freq_f[1:], F2[1:].mean(dim=-1), color="purple", alpha=0.1)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Power spectrum of $\boldsymbol{W}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e0)
    ax.axvline(x=1, color="k", linewidth=2)
    ax.axvline(x=15, color="k", linewidth=2)
    ax.set_xticks([1, 10, 15, 20, 30, 40, 50])
    ax.set_xticklabels(
        [r"$f_c=1$", "", r"$f_c^{\mathrm{dn}}=15$", "", "", "", "$50$"], fontsize=16
    )
    leg = ax.legend(loc=1, framealpha=0.9, fontsize=12)
    increase_leglw(leg, linewidth=4)
    fig.tight_layout()
    fig.savefig(
        "analysis/visualize_data/compare_spectral_energy_of_datasets.png", dpi=300
    )
    print("analysis/visualize_data/compare_spectral_energy_of_datasets.png")


if __name__ == "__main__":
    import argparse

    ylabels_diff0 = [
        r"$\Delta\mathbf{T}^{\mathbf{{" + i + r"}}}$ [m]" for i in ["x", "y", "z"]
    ]
    ylabels_diff1 = [
        r"$d\mathbf{T}^{\mathbf{{" + i + r"}}}/dt$ [m/s]" for i in ["x", "y", "z"]
    ]

    dir = "analysis/visualize_data"
    os.makedirs(dir, exist_ok=True)

    TASKS = {
        "table1": lambda: (recheck_max_wrench_magnitudes(), detect_contacts()),
        "fig4": concat_force_input_seq_and_show,
        "fig5": lambda: run_fig5(),
        "fig8": residual_distribution,
        "fig9": correlation_analysis,
        "fig10": compare_energy_distribution,
        "table10": analyze_band_energy_ratio,
    }
    def run_fig5():
        # Fig. 5(a): before denoising (raw wrench), Fig. 5(b): after denoising
        global dir
        for load_raw_ft, subdir in [(True, "fig5_raw"), (False, "fig5_denoised")]:
            os.environ["FDN_LOAD_RAW_FT"] = "1" if load_raw_ft else "0"
            dir = f"analysis/visualize_data/{subdir}"
            os.makedirs(dir, exist_ok=True)
            compute_spectral_energy(HydraulicDatasetRelPos7D)
        os.environ["FDN_LOAD_RAW_FT"] = "0"
        dir = "analysis/visualize_data"

    RAW_WRENCH_TASKS = ["fig10"]  # spectra shown without denoising

    parser = argparse.ArgumentParser(description="Data analysis tables/figures of the paper.")
    parser.add_argument("task", choices=list(TASKS), help="paper table/figure to reproduce")
    args = parser.parse_args()
    # also inherited by the DataLoader worker processes
    os.environ["FDN_LOAD_RAW_FT"] = "1" if args.task in RAW_WRENCH_TASKS else "0"

    if args.task.startswith("table"):  # tables are printed and saved to results/<table>.txt
        os.makedirs("results", exist_ok=True)
        output = f"results/{args.task}.txt"
        if result_exists(output, args.task):
            sys.exit(0)
        with open(output, "w") as f:
            sys.stdout = Tee(sys.__stdout__, f)
            try:
                TASKS[args.task]()
            finally:
                sys.stdout = sys.__stdout__
        print(f"Saved: {output}")
    else:  # figures are saved to analysis/visualize_data/
        TASKS[args.task]()

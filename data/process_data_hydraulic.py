import os, sys

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool
import shutil
import glob
import seaborn as sns

from utils.data import extract_trend
from utils.snippets import *
from utils.data import *
from layers.Filter import *


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2eul(R):
    # https://github.com/spmallick/learnopencv/blob/master/RotationMatrixToEulerAngles/rotm2euler.py#L12
    assert isRotationMatrix(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def robot_fkine(jointpos: np.ndarray):
    # body0
    r0 = np.array([0, 0, 0]).reshape(-1, 1)
    A0 = np.eye(3)
    s01_p = np.array([0, 0, 0]).reshape(-1, 1)
    C01 = np.eye(3)

    # body1
    s12_p = np.array([171, 0, 198.5]).reshape(-1, 1)
    C12 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    # body2
    s23_p = np.array([921.36, 0, 0]).reshape(-1, 1)
    C23 = np.eye(3)

    # body3
    s34_p = np.array([535.94, 0, 0]).reshape(-1, 1)
    C34 = np.eye(3)

    # body4
    s45_p = np.array([146, 0, 0]).reshape(-1, 1)
    C45 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    # body5
    s56_p = np.array([0, 2.38e02, 0]).reshape(-1, 1)
    C56 = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

    # body6
    s_t = np.array([0, 0, -262]).reshape(-1, 1)
    s_tc = np.array([0, 0, -107]).reshape(-1, 1)

    # forward kinematics
    time = jointpos[:, 0]
    q = jointpos[:, 1:]
    q2A = lambda q: np.array(
        [[np.cos(q), -np.sin(q), 0], [np.sin(q), np.cos(q), 0], [0, 0, 1]]
    )
    r_tool = np.zeros((q.shape[0], 11))
    for i in range(jointpos.shape[0]):
        # global orientation
        A01_pp = q2A(q[i, 0])
        A12_pp = q2A(q[i, 1])
        A23_pp = q2A(q[i, 2])
        A34_pp = q2A(q[i, 3])
        A45_pp = q2A(q[i, 4])
        A56_pp = q2A(q[i, 5])
        A1 = A0 @ C01 @ A01_pp
        A2 = A1 @ C12 @ A12_pp
        A3 = A2 @ C23 @ A23_pp
        A4 = A3 @ C34 @ A34_pp
        A5 = A4 @ C45 @ A45_pp
        A6 = A5 @ C56 @ A56_pp
        roll, pitch, yaw = rotm2eul(A6)

        # tool quaternion
        quat = R.from_matrix(A6).as_quat()
        if i > 0 and np.dot(quat, quat_prev) < 0:  # continuous quat
            quat = -quat
        quat_prev = quat
        qx, qy, qz, qw = quat

        # global position
        r1 = r0 + A0 @ s01_p
        r2 = r1 + A1 @ s12_p
        r3 = r2 + A2 @ s23_p
        r4 = r3 + A3 @ s34_p
        r5 = r4 + A4 @ s45_p
        r6 = r5 + A5 @ s56_p
        rtc = r6 + s_tc
        rt = r6 + s_t
        x, y, z = rt.flatten() / 1000
        # rp(i, :) = r6
        r_tool[i, :] = np.array(
            [jointpos[i, 0], x, y, z, roll, pitch, yaw, qx, qy, qz, qw]
        )  # time, pos[m], eul(rad), quat
        if (i + 1) % 10000 == 0:
            print(f"FK {i+1}/{q.shape[0]}")
    return r_tool


def _read(args):
    dir, files, keyval, use_filtered = args
    exp_name = dir.split("/")[-1]
    exp_data = {}
    for data_name in files:
        for key in keyval.keys():
            if use_filtered:
                if key in data_name.lower() and "Filt" in data_name:
                    exp_data[key] = (
                        deepcopy(keyval[key]),
                        list(readmat(f"{dir}/{data_name}").values())[0],
                    )
            else:
                if key in data_name.lower() and "Filt" not in data_name:
                    exp_data[key] = (
                        deepcopy(keyval[key]),
                        list(readmat(f"{dir}/{data_name}").values())[0],
                    )
                elif (
                    key == "depth" and key in data_name.lower() and "Filt" in data_name
                ):
                    exp_data[key] = (
                        deepcopy(keyval[key]),
                        list(readmat(f"{dir}/{data_name}").values())[0],
                    )
                elif key == "toolpos" and key not in exp_data.keys():
                    exp_data[key] = (
                        deepcopy(keyval[key]),
                        robot_fkine(
                            list(readmat(f"{dir}/knr_uw3_state_jointpos.mat").values())[
                                0
                            ]
                        ),
                    )
    return (exp_name, exp_data)


def read(use_filtered: bool = False):
    if not use_filtered:
        keyval = {
            "imu": [
                "time",
                "roll",
                "pitch",
                "yaw",
                "quat_w",
                "vel_roll",
                "vel_pitch",
                "vel_yaw",
                "acc_x",
                "acc_y",
                "acc_z",
            ],
            "force": ["time", "time_acq", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            + [f"{F}_calibrated" for F in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]],
            "hydraulics_pressure": ["time", "time_acq"]
            + [f"ch{i:02d}" for i in range(1, 33)],
            "diffpressure": ["time"] + [f"j{i}" for i in range(1, 8)],
            "jointpos": ["time"] + [f"j{i}" for i in range(1, 7)],
            "toolpos": [
                "time",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "qx",
                "qy",
                "qz",
                "qw",
            ],
        }
    else:
        keyval = {
            "imu": [
                "time",
                "roll",
                "pitch",
                "yaw",
                "vel_roll",
                "vel_pitch",
                "vel_yaw",
                "acc_x",
                "acc_y",
                "acc_z",
            ],
            "force": ["time"]
            + [f"{F}_calibrated" for F in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]],
            "toolpressure": ["time", "hydraulic_pressure"],
            "diffpressure": ["time"] + [f"j{i}" for i in range(1, 5)],
            "toolpos": [
                "time",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "qx",
                "qy",
                "qz",
                "qw",
            ],
            "depth": ["time", "depth"],
        }

    exp_dirs = []
    for dir, subdirs, files in os.walk("data/data_hydraulic"):
        if "main_totalSol.m" in files:
            exp_dirs.append((dir, files, keyval, use_filtered))

    dataset = {}
    with Pool() as pool:
        results = pool.map(_read, exp_dirs)
    for exp_name, exp_data in results:
        dataset[exp_name] = exp_data

    assert len(dataset) == 12, f"Number of experiments is not 12 but {len(dataset)}"
    for exp_name in dataset.keys():
        for data_name in keyval.keys():
            assert dataset[exp_name][data_name][-1].shape[1] == len(keyval[data_name])
    sorted_dict = dict(sorted(dataset.items()))
    return sorted_dict


def isDifferent(origin, dataset):
    flags = []
    if type(origin) == dict and type(dataset) == dict:
        for item1, item2 in zip(origin.items(), dataset.items()):
            k1, v1 = item1
            k2, v2 = item2
            flags.append(k1 == k2)
            flags.append(isDifferent(v1, v2))
    elif type(origin) == np.ndarray and type(dataset) == np.ndarray:
        for item1, item2 in zip(origin, dataset):
            flags.append(np.all(item1 == item2))
    else:
        for item1, item2 in zip(origin, dataset):
            flags.append(item1 == item2)
    if np.any([np.any(f) for f in flags]):
        return True  # changed
    else:
        return False  # no changes


def proc_initial_pose(dataset: dict):
    for exp_name in dataset.keys():
        for data_name in dataset[exp_name].copy().keys():
            dataset[exp_name][c_jointpos_0] = dataset[exp_name][c_jointpos].values[0, :]
            dataset[exp_name][c_toolpos_0] = dataset[exp_name][c_toolpos].values[0, :]
    return dataset


def proc_sync_timestamp(dataset: dict, sampling_freq=20):
    assert sampling_freq in [20, 100], "Invalid sampling frequency"
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        n_samples = []
        time_lengths = []
        frequencies = []
        timestamp_starts = []
        timestamp_ends = []
        # add interpolators
        for data_name in dataset[exp_name].keys():
            if "0" in data_name:
                continue
            columns, data = dataset[exp_name][data_name]
            # create zeroth polynomial interp
            if "time_acq" in columns:
                interp = interp1d(
                    data[:, 0],
                    data[:, 2:],
                    kind="zero",
                    axis=0,
                    bounds_error=False,
                    fill_value=0,
                )
            else:
                interp = interp1d(
                    data[:, 0],
                    data[:, 1:],
                    kind="zero",
                    axis=0,
                    bounds_error=False,
                    fill_value=0,
                )
            dataset[exp_name][data_name] = (columns, data, interp)

            n_samples.append(data.shape[0])
            time_lengths.append(np.ptp(data[:, 0]))
            frequencies.append(data.shape[0] / np.ptp(data[:, 0]))
            # collect start and end timestamps
            if data_name in ["jointpos", "toolpos", "imu", "force"]:
                timestamp_starts.append(data[0, 0])
                timestamp_ends.append(data[-1, 0])
        t_0 = max(timestamp_starts)
        t_end = min(timestamp_ends)
        if sampling_freq == 20:  # Synthetic, uniform 20Hz time steps
            # initial_q_timestamp = dataset[exp_name]["jointpos"][1][
            #     0, dataset[exp_name]["jointpos"][0].index("time")
            # ]
            time = np.arange(
                t_0,
                t_end + max(time_lengths) + 1e-6,
                1 / sampling_freq,
            )
        elif sampling_freq == 100:  # JointPos 100Hz time steps
            time = dataset[exp_name]["jointpos"][1][
                :, dataset[exp_name]["jointpos"][0].index("time")
            ]  # JointPos time steps 100 Hz
            time = time[(time >= t_0) & (time <= t_end)]  # slice within t_0 and t_end
        interpolated = [time.reshape(-1, 1)]
        columns_all = ["time"]
        # interpolate ['toolpos', 'hydraulics_pressure', 'force', 'imu', 'jointpos', 'diffpressure'] w/ 'time'
        for data_name in dataset[exp_name].keys():
            if "pos_0" not in data_name:
                columns, data, interp = dataset[exp_name][data_name]
                interpolated.append(interp(time))
                if "time" in columns:
                    columns.remove("time")
                if "time_acq" in columns:
                    columns.remove("time_acq")
                assert interp(time).shape[1] == len(columns)
            else:
                columns, data = dataset[exp_name][data_name]
                interpolated.append(data[0:1, :].repeat(time.shape[0], axis=0))
            columns_all += [data_name + f"_{c}" for c in columns]
        interpolated = np.concatenate(interpolated, axis=1)

        # save
        dataset[exp_name] = pd.DataFrame(interpolated, columns=columns_all).dropna()
    assert isDifferent(origin, dataset)
    return dataset


def proc_transform_sensor_coordinates(dataset: dict):
    # Match all coordinates to global origin coordiate
    # IMU Sensors == Global coordinate
    origin = deepcopy(dataset)
    for exp_name in dataset.keys():
        for data_name in origin[exp_name].keys():
            columns, data = origin[exp_name][data_name]

            if data_name == "force":
                dataset[exp_name][data_name][1][:, columns.index("Fx")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("Fy")]
                dataset[exp_name][data_name][1][:, columns.index("Fy")] = -origin[
                    exp_name
                ][data_name][1][:, columns.index("Fx")]
                dataset[exp_name][data_name][1][:, columns.index("Fz")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("Fz")]
                dataset[exp_name][data_name][1][:, columns.index("Mx")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("My")]
                dataset[exp_name][data_name][1][:, columns.index("My")] = -origin[
                    exp_name
                ][data_name][1][:, columns.index("Mx")]
                dataset[exp_name][data_name][1][:, columns.index("Mz")] = origin[
                    exp_name
                ][data_name][1][:, columns.index("Mz")]
                dataset[exp_name][data_name][1][:, columns.index("Fx_calibrated")] = (
                    origin[exp_name][data_name][1][:, columns.index("Fy_calibrated")]
                )
                dataset[exp_name][data_name][1][:, columns.index("Fy_calibrated")] = (
                    -origin[exp_name][data_name][1][:, columns.index("Fx_calibrated")]
                )
                dataset[exp_name][data_name][1][:, columns.index("Fz_calibrated")] = (
                    origin[exp_name][data_name][1][:, columns.index("Fz_calibrated")]
                )
                dataset[exp_name][data_name][1][:, columns.index("Mx_calibrated")] = (
                    origin[exp_name][data_name][1][:, columns.index("My_calibrated")]
                )
                dataset[exp_name][data_name][1][:, columns.index("My_calibrated")] = (
                    -origin[exp_name][data_name][1][:, columns.index("Mx_calibrated")]
                )
                dataset[exp_name][data_name][1][:, columns.index("Mz_calibrated")] = (
                    origin[exp_name][data_name][1][:, columns.index("Mz_calibrated")]
                )
    assert isDifferent(origin, dataset)
    return dataset


def proc_relative_motions(dataset: dict):
    for exp_name in dataset.keys():
        df = dataset[exp_name].copy()
        # Subtract initial positions
        df[c_jointpos] = df[c_jointpos].values - df[c_jointpos_0].values
        df[c_toolpos[:3]] = df[c_toolpos[:3]].values - df[c_toolpos_0[:3]].values

        # Compute relative quaternions
        initial_quat = R.from_quat(df[c_toolpos[3:]].values[0])
        quats = R.from_quat(df[c_toolpos[3:]].values)
        relative_rotations = (initial_quat.inv() * quats).as_quat()
        relative_rotations = dot_continuous_quat(relative_rotations)
        df[c_toolpos[3:]] = relative_rotations
        dataset[exp_name] = df
    return dataset


def postprocess(sampling_freq: int = 20, use_filtered: bool = False):
    os.makedirs("data/data_hydraulic/processed", exist_ok=True)
    dataset = read(use_filtered=use_filtered)  # dict
    dataset = proc_transform_sensor_coordinates(dataset)  # dict
    summary(dataset)
    dataset = proc_sync_timestamp(dataset, sampling_freq=sampling_freq)  # DataFrame
    dataset = proc_initial_pose(dataset)
    dataset = proc_relative_motions(dataset)
    return dataset


def summary(dataset: dict):
    count = 0
    total_time = 0
    total_n_jointpos = 0
    total_n_imu = 0
    total_n_ft = 0
    for epi_name, epi_dat in dataset.items():
        count += 1
        time_length = np.ptp(epi_dat["jointpos"][1], axis=0)[0]
        jointpos = epi_dat["jointpos"][1]
        imu = epi_dat["imu"][1]
        ft = epi_dat["force"][1]
        force_names = {
            0: "F_{\mathbf{x}}",
            1: "F_{\mathbf{y}}",
            2: "F_{\mathbf{z}}",
        }
        torque_names = {0: "M_{\mathbf{x}}", 1: "M_{\mathbf{y}}", 2: "M_{\mathbf{z}}"}
        force_max = np.abs(ft[:, 2:5]).max(axis=0)
        force_argmax = np.argmax(force_max)
        torque_max = np.abs(ft[:, 5:8]).max(axis=0)
        torque_argmax = np.argmax(torque_max)

        print(
            f"{count:02d} & {time_length:.2f} & {jointpos.shape[0]:,} & {imu.shape[0]:,} & {ft.shape[0]:,} & ${force_names[force_argmax]}={int(force_max[force_argmax].round())}$ & ${torque_names[torque_argmax]}={int(torque_max[torque_argmax].round())}$ \\\\"
        )
        total_time += time_length
        total_n_jointpos += jointpos.shape[0]
        total_n_imu += imu.shape[0]
        total_n_ft += ft.shape[0]
    print(
        f"Total & {total_time:.2f} & {total_n_jointpos:,} & {total_n_imu:,} & {total_n_ft:,} \\\\"
    )


def save_episodes_as_npy(dataset: dict):
    basedir = "data/data_hydraulic/processed"
    sensor_names = [
        "jointpos6_raw",
        "jointpos6_0",
        "toolpos",
        "toolpos_0",
        "imu_raw",  # with offset
        "jtorque6_raw",
        "hydraulics_pressure",
        "ft_raw",  # with offset
        "ft_calibrated",
        "timestamp",
    ]
    col_groups = [
        c_jointpos,
        c_jointpos_0,
        c_toolpos,
        c_toolpos_0,
        c_imu,
        c_diffpressure,
        c_hydraulics_pressure,
        c_ft_raw,
        c_ft_calibrated,
        c_time,
    ]
    for sensor_name in sensor_names:
        os.makedirs(f"{basedir}/npy_{sensor_name}", exist_ok=True)

    for exp_name, df in dataset.items():
        for sensor_name, col_group in zip(sensor_names, col_groups):
            save_dir = f"{basedir}/npy_{sensor_name}/{exp_name}.npy"
            np.save(save_dir, df[col_group].to_numpy())
            print(f"Saved: {save_dir}")


def _to_csv(args):
    key, dataset, tag = args
    path = "data/data_hydraulic/processed"
    Path = f"{path}/{key}_{tag}" if tag != "" else f"{path}/{key}"
    dataset[key].to_csv(f"{Path}.csv", index=False)
    print(f"Saved: {Path}.csv")


def to_csv(dataset: dict, tag: str = ""):
    path = "data/data_hydraulic/processed"
    args = [(key, dataset, tag) for key in dataset.keys()]
    with Pool() as pool:
        pool.map(_to_csv, args)


def process_offsets_outliers_denoise_diff():
    # remove sensor offsets
    # remove outliers
    # denoise (input-causal, output-noncausal)
    # compute causal derivatives
    data_imu_list = sorted(glob.glob("data/data_hydraulic/processed/npy_imu_raw/*.npy"))
    data_ft_list = sorted(glob.glob("data/data_hydraulic/processed/npy_ft_raw/*.npy"))
    data_t_list = sorted(glob.glob("data/data_hydraulic/processed/npy_timestamp/*.npy"))
    data_diffp_list = sorted(
        glob.glob("data/data_hydraulic/processed/npy_jtorque6_raw/*.npy")
    )
    data_jointpos_list = sorted(
        glob.glob("data/data_hydraulic/processed/npy_jointpos6_raw/*.npy")
    )

    os.makedirs("data/data_hydraulic/processed/npy_ft", exist_ok=True)
    os.makedirs("data/data_hydraulic/processed/npy_imu", exist_ok=True)
    os.makedirs("data/data_hydraulic/processed/npy_jtorque6", exist_ok=True)
    os.makedirs("data/data_hydraulic/processed/npy_jointpos6", exist_ok=True)
    os.makedirs("data/data_hydraulic/processed/npy_jointvel6", exist_ok=True)
    os.makedirs("data/data_hydraulic/processed/npy_jointacc6", exist_ok=True)

    lpf_nc = FreqPassFilter("low", cutoff_freq=15, sampling_freq=100) # non-causal denoise (outputs)
    lpf_c = CausalLPF_Butter(cutoff_freq=15, sampling_freq=100) # causal denoise (inputs)
    diff_c = CausalDiff_SavGol(cutoff_freq=15, sampling_freq=100) # causal diff (derivatives)

    for d_imu, d_ft, d_t, d_diffp, d_jointpos in zip(
        data_imu_list, data_ft_list, data_t_list, data_diffp_list, data_jointpos_list
    ):
        jointpos = np.load(d_jointpos)
        diffp = np.load(d_diffp)
        imu = np.load(d_imu)
        ft = np.load(d_ft)
        time = np.load(d_t)[:, 0]
        time = time - time[0]

        if ("1st_Test" in d_ft) or ("2nd_Test_6" in d_ft):
            t1 = 10
            t2 = 20
        elif "2nd_Test_5" in d_ft:
            t1 = 70
            t2 = 80
        else:
            t1 = 50
            t2 = 60

        # compute 10s mean
        static_idx = np.where((t1 <= time) & (time <= t2))[0]
        initial_idx = np.where((time <= 3))[0]

        # subtract offsets
        imu_offset = imu[static_idx, :].mean(axis=0)
        ft_offset = ft[static_idx, :].mean(axis=0)
        diffp_offset = diffp[initial_idx, :].mean(axis=0)

        imu = imu - imu_offset  # initial
        diffp = diffp - diffp_offset  # initial
        ft = ft - ft_offset  # non-contact

        # zero outliers
        if "Ground_1st_Test_1" in d_diffp:
            outliers = np.where(
                (diffp[:, 0] <= np.quantile(diffp[:, 0], 0.01))
                | (diffp[:, 0] >= np.quantile(diffp[:, 0], 0.9999))
            )[0]
            diffp[outliers, 0] = 0
        elif "Ground_1st_Test_2" in d_diffp:
            outliers = np.where(
                (diffp[:, 0] <= np.quantile(diffp[:, 0], 0.01))
                | (diffp[:, 0] >= np.quantile(diffp[:, 0], 0.99))
            )[0]
            diffp[outliers, 0] = 0
            outliers = np.where((diffp[:, 1] <= np.quantile(diffp[:, 1], 0.0001)))[0]
            diffp[outliers, 1] = 0

        # non-causal F/T, causal denoise inputs, compute causal derivatives
        freq = np.fft.rfftfreq(jointpos.shape[0], d=1 / 100)
        ft = lpf_nc(ft)
        diffp = lpf_c(diffp)
        jointpos = lpf_c(jointpos)
        jointvel, jointacc = diff_c.vel_acc(jointpos)

        # save
        d_imu_processed = d_imu.replace("npy_imu_raw", "npy_imu")
        d_ft_processed = d_ft.replace("npy_ft_raw", "npy_ft")
        d_diffp_processed = d_diffp.replace("npy_jtorque6_raw", "npy_jtorque6")
        d_jointpos_processed = d_jointpos.replace(
            "npy_jointpos6_raw", "npy_jointpos6"
        )
        d_jointvel_processed = d_jointpos.replace(
            "npy_jointpos6_raw", "npy_jointvel6"
        )
        d_jointacc_processed = d_jointpos.replace(
            "npy_jointpos6_raw", "npy_jointacc6"
        )
        np.save(d_imu_processed, imu)
        np.save(d_ft_processed, ft)
        np.save(d_diffp_processed, diffp)
        np.save(d_jointpos_processed, jointpos)
        np.save(d_jointvel_processed, jointvel)
        np.save(d_jointacc_processed, jointacc)
        print(f"Removed sensor offsets, denoised, differentiated, and saved:")
        print(f"  {d_jointpos_processed}")
        print(f"  {d_jointvel_processed}")
        print(f"  {d_jointacc_processed}")
        print(f"  {d_diffp_processed}")
        print(f"  {d_ft_processed}")
        print(f"  {d_imu_processed}")


def visualize_offset_outlier_removal():
    for tag in ["_raw", ""]:
        npy_ft_raw = sorted(
            glob.glob(f"data/data_hydraulic/processed/npy_ft{tag}/*.npy")
        )
        npy_imu_raw = sorted(
            glob.glob(f"data/data_hydraulic/processed/npy_imu{tag}/*.npy")
        )
        npy_diffp_raw = sorted(
            glob.glob(f"data/data_hydraulic/processed/npy_jtorque6{tag}/*.npy")
        )

        save_dir_ft = f"data/data_hydraulic/processed/offset_removal_ft"
        save_dir_imu = f"data/data_hydraulic/processed/offset_removal_imu"
        save_dir_diffp = f"data/data_hydraulic/processed/offset_removal_diffpressure"
        os.makedirs(save_dir_ft, exist_ok=True)
        os.makedirs(save_dir_imu, exist_ok=True)
        os.makedirs(save_dir_diffp, exist_ok=True)
        plot_template()

        for d_ft, d_imu, d_diffp in zip(npy_ft_raw, npy_imu_raw, npy_diffp_raw):
            ft = np.load(d_ft)
            imu = np.load(d_imu)
            diffp = np.load(d_diffp)
            # ft = ft - ft[0:100, :].mean(axis=0)
            trend, res = extract_trend(ft, cutoff_freq=1, sampling_freq=100)
            time = np.load(d_ft.replace(f"npy_ft{tag}", "npy_timestamp"))[:, 0]
            time = time - time[0]

            fig1, ax1 = plt.subplots(6, 1, figsize=(5, 7), sharex=True)
            fig2, ax2 = plt.subplots(6, 1, figsize=(5, 7), sharex=True)
            fig3, ax3 = plt.subplots(6, 1, figsize=(5, 7), sharex=True)
            for i in range(ft.shape[1]):
                ax1[i].plot(time, ft[:, i], lw=1, c="k")
                ax1[i].plot(time, res[:, i], lw=1, alpha=0.7, c="cornflowerblue")
                ax1[i].plot(time, trend[:, i], lw=1, c="red")
                ax2[i].plot(time, imu[:, i], lw=0.7, c="k")
                ax3[i].plot(time, diffp[:, i], lw=0.7, c="k")
                if ("1st_Test" in d_ft) or ("2nd_Test_6" in d_ft):
                    t1 = 10
                    t2 = 20
                elif "2nd_Test_5" in d_ft:
                    t1 = 70
                    t2 = 80
                else:
                    t1 = 50
                    t2 = 60
                for ax in [ax1, ax2, ax3]:
                    ax[i].axvline(x=t1, color="k", linestyle="--")
                    ax[i].axvline(x=t2, color="k", linestyle="--")
                    ax[i].axvspan(xmin=t1, xmax=t2, color="grey", alpha=0.3)
            fig1.tight_layout()
            fig2.tight_layout()
            fig3.tight_layout()
            save_name = f"{os.path.basename(d_ft).replace('.npy', '')}"
            if "raw" in d_ft:
                save_name += "_01_before"
            else:
                save_name += "_02_after"
            save_name += ".png"
            fig1.savefig(f"{save_dir_ft}/{save_name}", dpi=300)
            fig2.savefig(f"{save_dir_imu}/{save_name}", dpi=300)
            fig3.savefig(f"{save_dir_diffp}/{save_name}", dpi=300)
            print(f"{save_dir_ft}/{save_name}")
            print(f"{save_dir_imu}/{save_name}")
            print(f"{save_dir_diffp}/{save_name}")
            plt.close("all")


def update_csv():
    csv_list = sorted(glob.glob("data/data_hydraulic/processed/*.csv"))
    ft_corrected_list = sorted(glob.glob("data/data_hydraulic/processed/npy_ft/*.npy"))
    imu_corrected_list = sorted(
        glob.glob("data/data_hydraulic/processed/npy_imu/*.npy")
    )
    diffp_corrected_list = sorted(
        glob.glob("data/data_hydraulic/processed/npy_jtorque6/*.npy")
    )
    jointpos_corrected_list = sorted(
        glob.glob("data/data_hydraulic/processed/npy_jointpos6/*.npy")
    )
    jointvel_corrected_list = sorted(
        glob.glob("data/data_hydraulic/processed/npy_jointvel6/*.npy")
    )
    jointacc_corrected_list = sorted(
        glob.glob("data/data_hydraulic/processed/npy_jointacc6/*.npy")
    )
    for csv, ft, imu, diffp, jointpos, jointvel, jointacc in zip(
        csv_list,
        ft_corrected_list,
        imu_corrected_list,
        diffp_corrected_list,
        jointpos_corrected_list,
        jointvel_corrected_list,
        jointacc_corrected_list,
    ):
        data = pd.read_csv(csv)
        data[c_ft_raw] = np.load(ft)
        data[c_imu] = np.load(imu)
        data[c_diffpressure] = np.load(diffp)
        data[c_jointpos] = np.load(jointpos)
        data[c_jointvel] = np.load(jointvel)
        data[c_jointacc] = np.load(jointacc)
        data.to_csv(csv, index=False)
        print(
            f"Updated corrected FT, IMU, Diffp, Jointpos, Jointvel, Jointacc in {csv}"
        )


if __name__ == "__main__":
    c_jointpos = [f"jointpos_j{i}" for i in range(1, 7)]
    c_jointvel = [f"jointvel_j{i}" for i in range(1, 7)]
    c_jointacc = [f"jointacc_j{i}" for i in range(1, 7)]
    c_toolpos = [
        "toolpos_x",
        "toolpos_y",
        "toolpos_z",
        "toolpos_qx",
        "toolpos_qy",
        "toolpos_qz",
        "toolpos_qw",
    ]
    c_imu = [
        "imu_vel_roll",
        "imu_vel_pitch",
        "imu_vel_yaw",
        "imu_acc_x",
        "imu_acc_y",
        "imu_acc_z",
    ]
    c_jointpos_0 = [c.replace("jointpos", "jointpos_0") for c in c_jointpos]
    c_toolpos_0 = [c.replace("toolpos", "toolpos_0") for c in c_toolpos]

    c_ft_raw = ["force_Fx", "force_Fy", "force_Fz", "force_Mx", "force_My", "force_Mz"]
    c_ft_calibrated = [f"{f}_calibrated" for f in c_ft_raw]

    c_hydraulics_pressure = [f"hydraulics_pressure_ch{i:02d}" for i in range(1, 33)]
    c_diffpressure = [f"diffpressure_j{i}" for i in range(1, 7)]

    c_time = ["time"]

    for sampling_freq in [100]:  # 20
        dataset = postprocess(sampling_freq=sampling_freq)
        save_episodes_as_npy(dataset)
        to_csv(dataset=dataset, tag=f"{sampling_freq}hz")
        process_offsets_outliers_denoise_diff()
        # visualize_offset_outlier_removal()
        update_csv()

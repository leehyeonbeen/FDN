import os, sys

sys.path.append(os.getcwd())

import pandas as pd
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn, filterwarnings
from scipy.interpolate import interp1d
from multiprocessing import Pool
from utils.data import (
    dot_continuous_quat,
    clip_outlier_values,
    extract_trend,
    filtered_derivative,
)
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from copy import deepcopy
from layers.Filter import *
import warnings


filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

"""
|-- RH20T
    |-- RH20T_cfg1
    |   |-- calib/                            # Calibration folder, including calibration-time Gripper Cartesian pose, intrinsic and extrinsic matrices etc. Extrinsic matrices are the Aruco marker's Translations with respect to the camera frame.
    |   |-- task_0001_user_0001_scene_0001_cfg_0001/        # Robotic manipulation data
    |   |    |-- metadata.json                # Robot manipulation scene metadata, including scene finishing timestamp, task completion rating (0 denotes robot failure, 1 denotes task failure, 2-9 denotes completion quality, higher is better), calibration timestamp and calibration quality (0 means some cameras are not calibrated, 1-5 means calibration accuracy, lower is better), etc.
    |   |    |-- cam_[serial_number]/         # Multiple cameras
    |   |    |    |-- color.mp4               # Color images, encode as video. The extraction code is available in our API code.
    |   |    |    |-- timestamps.npy          # Timestamp for each image, our extraction code will use it to decode images.
    |   |    |    `-- depth.mp4 (optional)    # Depth images, encode as video. The extraction code is available in our API code.
    |   |    |-- transformed/
    |   |    |    |-- tcp.npy                 # Gripper Cartesian pose in each cam's coord, {serial number: [{"timestamp": ..., "tcp": ..., "robot_ft": ...}]}, where "tcp" values are xyz+quat (7D) Gripper Cartesian poses
    |   |    |    |-- tcp_base.npy            # Gripper Cartesian pose in base coord, {serial number: [{"timestamp": ..., "tcp": ..., "robot_ft": ...}]}, where "tcp" values are xyz+quat (7D) Gripper Cartesian poses
    |   |    |    |-- joint.npy               # Joint angles, {serial number: {timestamp: joint angle array}}
    |   |    |    |-- gripper.npy             # Gripper commands and information, {serial number: {timestamp: {"gripper_command": 3D array, "gripper_info": 3D array}}}, where the 1st element in the 3D array is the actual gripper width in millimeters(0-110)
    |   |    |    |-- force_torque.npy        # 6-DoF force/torque in cam's coord, {serial number: [{"timestamp": ..., "zeroed": ..., "raw": ...}]}, where "zeroed" values are pre-processed
    |   |    |    |-- force_torque_base.npy   # 6-DoF force/torque in base coord, {serial number: [{"timestamp": ..., "zeroed": ..., "raw": ...}]}, where "zeroed" values are pre-processed
    |   |    |    `-- high_freq_data.npy      # High frequency data, {serial number: [{"timestamp": ..., "zeroed": ..., "raw": ..., "tcp": ...}]}
    |   |    `-- audio_mixed/
    |   |
    |   |-- task_0001_user_0001_scene_0001_cfg_0001_human/  # Human demonstration data corresponds to the above robotic manipulation
    |   |    |-- metadata.json                # Human demonstration metadata, including scene starting and finishing timestamps, calibration timestamp and quality
    |   |    |-- cam_[serial_number]/
    |   |    |    |-- color.mp4
    |   |    |    |-- timestamps.npy
    |   |    |    `-- depth.mp4 (optional)
    |   |    `-- audio_mixed/
    |   |
    |   |
    |   `-- ... ...
    |
    |
    |-- RH20T_cfg2/
    |   `-- same as above
    |
    |
    |-- ...
    |
    |
    `-- RH20T_cfg7/
"""

jointpos_cols = [
    "jointpos_j1",
    "jointpos_j2",
    "jointpos_j3",
    "jointpos_j4",
    "jointpos_j5",
    "jointpos_j6",
    "jointpos_j7",
]
jtorque_cols = [j.replace("jointpos", "jtorque") for j in jointpos_cols]
jointvel_cols = [j.replace("pos", "vel") for j in jointpos_cols]
jointacc_cols = [j.replace("pos", "acc") for j in jointpos_cols]
toolpos_cols = [
    "toolpos_x",
    "toolpos_y",
    "toolpos_z",
    "toolpos_qx",
    "toolpos_qy",
    "toolpos_qz",
    "toolpos_qw",
]

jointpos_0_cols = [j + "_0" for j in jointpos_cols]
toolpos_0_cols = [t + "_0" for t in toolpos_cols]
force_cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def merge_patch_tar_gz():
    # merge joint.npy files of patch.tar.gz
    # joint_files = glob.glob("data/data_rh20t/raw/patch/**/*/*.npy", recursive=True)
    joint_files = glob.glob(
        "/Volumes/drive_name/FDN/data_rh20t_raw/patch/**/*/*.npy", recursive=True
    )
    joint_files.sort()
    for j in joint_files:
        src = j
        dst = j.replace("/patch", "")
        try:
            shutil.move(src, dst)
            print(f"Successful: {src} -> {dst}")
        except Exception as e:
            print(f"Error: {src} -> {dst}: {e}")


def get_exp_name(d):
    subdirnames = d.split("/")
    config_name = subdirnames[-4]
    exp_name = subdirnames[-3]
    try:
        assert "RH20T_cfg" in config_name
        assert (
            "task" in exp_name
            and "user" in exp_name
            and "scene" in exp_name
            and "cfg" in exp_name
        )
        key = f"{config_name}_{exp_name}"
        return key
    except AssertionError:
        return None


def get_attr_name(d):
    filename = d.split("/")[-1]
    return filename.replace(".npy", "")


def proc_high_freq_data(d):
    # try:  # the code works for all valid files.
    # extract joint angles
    d_joint = d.replace("high_freq_data", "joint")
    try:
        arr_dict_joint = np.load(d_joint, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"{d_joint} does not exist")
        return None
    if len(arr_dict_joint) == 0:
        print(f"{d_joint} exists but is null.")
        return None
    data_count = 0
    for k, v in arr_dict_joint.items():
        data_count += len(v)
    if data_count == 0:
        print(f"{d_joint} exists but has zero length.")
        return None

    max_len_k = None
    max_len_v = 0
    for k, v in arr_dict_joint.items():
        if len(v) > max_len_v:
            max_len_v = len(v)
            max_len_k = k
    data_joint = pd.DataFrame(
        arr_dict_joint[max_len_k]
    ).T  # expects [jointpos, jtorque]
    if data_joint.shape[1] % 7 == 0:  # 7D
        _jointpos_cols = deepcopy(jointpos_cols)
        _jtorque_cols = deepcopy(jtorque_cols)
        _jointvel_cols = deepcopy(jointvel_cols)
        _jointacc_cols = deepcopy(jointacc_cols)
        _jointpos_0_cols = deepcopy(jointpos_0_cols)
    elif data_joint.shape[1] % 6 == 0:  # 6D
        _jointpos_cols = deepcopy(jointpos_cols[:6])
        _jtorque_cols = deepcopy(jtorque_cols[:6])
        _jointvel_cols = deepcopy(jointvel_cols[:6])
        _jointacc_cols = deepcopy(jointacc_cols[:6])
        _jointpos_0_cols = deepcopy(jointpos_0_cols[:6])
    if (data_joint.shape[1] % 7 == 0 and data_joint.shape[1] // 7 == 2) or (
        data_joint.shape[1] % 6 == 0 and data_joint.shape[1] // 6 == 2
    ):  # expects [jointpos, jtorque]
        data_joint.columns = _jointpos_cols + _jtorque_cols
    elif (data_joint.shape[1] % 7 == 0 and data_joint.shape[1] // 7 == 1) or (
        data_joint.shape[1] % 6 == 0 and data_joint.shape[1] // 6 == 1
    ):  # expects [jointpos] only
        data_joint.columns = _jointpos_cols
    elif (data_joint.shape[1] % 7 == 0 and data_joint.shape[1] // 7 == 3) or (
        data_joint.shape[1] % 6 == 0 and data_joint.shape[1] // 6 == 3
    ):  # expects [jointpos, jointvel, jointacc]
        data_joint.columns = _jointpos_cols + _jointvel_cols + _jointacc_cols
    else:
        raise ValueError(f"Unexpected joint data shape: {data_joint.shape}")
    timestamp_joint = data_joint.index.values
    interp_joint = interp1d(
        timestamp_joint, data_joint.values, axis=0, fill_value=0, bounds_error=False
    )

    # extract high freq data
    try:
        arr_dict = np.load(d, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"{d} does not exist.")
        return None
    if len(arr_dict) == 0:
        print(f"{d} exists but is null.")
        # try:
        #     d= d.replace('high_freq_data','force_torque_base')
        #     arr_dict = np.load(d, allow_pickle=True).item()
        # except FileNotFoundError:
        #     print(f"Tried to locate force_torque_base.npy instead of high_freq_data.npy but failed.")
        return None
    # "/Volumes/drive_name/PDF/data_rh20t_raw/RH20T_cfg1/task_0021_user_0005_scene_0001_cfg_0001/transformed/high_freq_data.npy"
    data_count = 0
    for k, v in arr_dict.items():
        data_count += len(v)
    if data_count == 0:
        print(f"{d} coordinates exist in file but are empty.")
        return None

    # if "base" in arr_dict.keys():
    k = "base"  # base coordinate only
    # else:
    #     max_len_k = None
    #     max_len_v = 0
    #     for k, v in arr_dict_joint.items():
    #         if len(v) > max_len_v:
    #             max_len_v = len(v)
    #             max_len_k = k
    #     k = max_len_k
    v = arr_dict[k]

    arr_dict[k] = pd.DataFrame(v).set_index(
        "timestamp", drop=True
    )  # timestamp, tcp, zeroed (pre-processed FT), raw (FT)

    # check and warn NaNs
    num_nans_high_freq = arr_dict[k].isna().sum().sum()
    num_nans_joint = data_joint.isna().sum().sum()
    if num_nans_high_freq > 0:
        warn(
            f"{num_nans_high_freq} NaNs detected in {d} on the first look. Dropping NaNs.",
            UserWarning,
        )
        arr_dict[k].dropna(inplace=True)  # drop None valued rows
    if num_nans_joint > 0:
        warn(
            f"{num_nans_joint} NaNs detected in {d_joint} on the first look. Dropping NaNs.",
            UserWarning,
        )
        data_joint.dropna(inplace=True)  # drop None valued rows

    zeroed = arr_dict[k].pop("zeroed")  # pre-processed FT
    zeroed = np.stack(zeroed.tolist(), axis=0)
    tcp = arr_dict[k].pop("tcp")  # gripper cartesian pose
    tcp = np.stack(tcp.tolist(), axis=0)
    tcp[:, 3:] = dot_continuous_quat(tcp[:, 3:])  # convert to continuous quaternion
    arr_dict[k][[f"{f}" for f in force_cols]] = zeroed
    arr_dict[k][[f"{p}" for p in toolpos_cols]] = tcp
    arr_dict[k].drop(
        columns=["raw"], inplace=True
    )  # drop raw FT (inconsistent coordinate)
    data_high_freq = arr_dict[k]

    assert data_high_freq.isna().sum().sum() == 0, print(d)
    assert data_joint.isna().sum().sum() == 0, print(d)

    # slice timestamps
    min_timestamp = max(data_high_freq.index.min(), data_joint.index.min())
    max_timestamp = min(data_high_freq.index.max(), data_joint.index.max())
    data_high_freq = data_high_freq[
        (data_high_freq.index >= min_timestamp)
        & (data_high_freq.index <= max_timestamp)
    ]
    # interpolate jointpos to 100Hz
    interpolated_joint_data = interp_joint(data_high_freq.index.values)
    if interpolated_joint_data.shape[1] == 7 or interpolated_joint_data.shape[1] == 6:
        data_high_freq[_jointpos_cols] = interpolated_joint_data
    elif (
        interpolated_joint_data.shape[1] == 14 or interpolated_joint_data.shape[1] == 12
    ):
        data_high_freq[_jointpos_cols + _jtorque_cols] = interpolated_joint_data
    elif (
        interpolated_joint_data.shape[1] == 21 or interpolated_joint_data.shape[1] == 18
    ):
        data_high_freq[_jointpos_cols + _jointvel_cols + _jointacc_cols] = (
            interpolated_joint_data
        )
    else:
        print(
            f"Unexpected shape of interpolated joint data: {interpolated_joint_data.shape}"
        )
        return None

    # compute relative toolpos and jointpos
    data_joint_0 = data_high_freq[_jointpos_cols].values[0:1]
    data_toolpos_0_xyz = data_high_freq[toolpos_cols[:3]].values[0:1]
    data_toolpos_0_quat = data_high_freq[toolpos_cols[3:7]].values[0:1]
    data_high_freq[_jointpos_cols] -= data_joint_0  # relative jointpos
    data_high_freq[toolpos_cols[:3]] -= data_toolpos_0_xyz  # relative toolpos xyz
    # relative toolpos quaternion
    quat_0 = R.from_quat(data_toolpos_0_quat)
    quats = R.from_quat(data_high_freq[toolpos_cols[3:7]].values)
    quat_rel = quat_0.inv() * quats
    data_high_freq[toolpos_cols[3:7]] = quat_rel.as_quat()

    n = data_high_freq.shape[0]
    data_high_freq[_jointpos_0_cols] = data_joint_0.repeat(
        n, axis=0
    )  # initial jointpos
    data_high_freq[toolpos_0_cols[:3]] = data_toolpos_0_xyz.repeat(
        n, axis=0
    )  # initial toolpos xyz
    data_high_freq[toolpos_0_cols[3:7]] = data_toolpos_0_quat.repeat(
        n, axis=0
    )  # initial toolpos quat

    data_high_freq.index -= data_high_freq.index[0]  # zero initial time
    data_high_freq = clip_outlier_values(data_high_freq)  # clip outlier values
    num_joints = len(_jointpos_cols)

    sampling_freq = n / ((max_timestamp - min_timestamp) / 1000)
    if sampling_freq < 80:
        print(
            f"{d}: Data sampling frequency {sampling_freq:.2f} Hz = {n} steps / {(max_timestamp - min_timestamp)/1000:.2f}s is too low."
        )
        # RH20T_cfg1/task_0078_user_0015_scene_0005_cfg_0001: 0.23 Hz = 5 steps / 21.39s
        # RH20T_cfg1/task_0013_user_0001_scene_0001_cfg_0001: 0.18 Hz = 8 steps / 43.76s
        return None
    try:
        # non-causal F/T, causal denoise inputs, compute causal derivatives
        lpf_nc = FreqPassFilter("low", cutoff_freq=15, sampling_freq=sampling_freq)
        lpf_c = CausalLPF_Butter(cutoff_freq=15, sampling_freq=sampling_freq)
        diff_c = CausalDiff_SavGol(cutoff_freq=15, sampling_freq=sampling_freq)
    except ValueError:
        raise ValueError(
            f"Error initializing filters for {d} with sampling frequency {sampling_freq:.2f} Hz. num. time steps:{n}, duration: {(max_timestamp - min_timestamp)/1000:.2f} s."
        )
    force_raw=data_high_freq[force_cols].to_numpy().copy()
    data_high_freq[force_cols] = lpf_nc(
        data_high_freq[force_cols].to_numpy()
    )  # denoise
    try:
        data_high_freq[_jtorque_cols] = lpf_c(
            data_high_freq[_jtorque_cols].to_numpy()
        )  # denoise
    except KeyError:
        print(f"No jtorque data for {d}, skipping saving jtorque.npy")
    data_high_freq[_jointpos_cols] = lpf_c(
        data_high_freq[_jointpos_cols].to_numpy()
    )  # denoise
    jointvel, jointacc = diff_c.vel_acc(
        data_high_freq[_jointpos_cols].to_numpy()
    )  # diff
    data_high_freq[_jointvel_cols] = jointvel
    data_high_freq[_jointacc_cols] = jointacc

    return data_high_freq, num_joints, force_raw
    # except:  # for invalid files, ignore and return None.
    #     warnings.warn(f"Error processing file: {d}")
    #     return None


def _main(d, exp_names, proc_fns):
    os.makedirs(f"data/data_rh20t/processed/npy_jointpos6", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jointvel6", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jointacc6", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jointpos6_0", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jointpos7", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jointvel7", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jointacc7", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jointpos7_0", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_toolpos", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_toolpos_0", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_ft", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_ft_raw", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jtorque6", exist_ok=True)
    os.makedirs(f"data/data_rh20t/processed/npy_jtorque7", exist_ok=True)

    exp = get_exp_name(d)
    attr = get_attr_name(d)
    if exp in exp_names and attr in proc_fns.keys():
        result = proc_fns[attr](d)
        if result is None:
            # warn(f"Corrupted file detected, skipping {d}", UserWarning)
            pass
        elif result is not None:  # if not corrupted (None)
            data, num_joints, force_raw = result
            dst_fig = f"data/data_rh20t/processed/{exp}_{attr}.png"
            dst_i1 = (
                f"data/data_rh20t/processed/npy_jointpos{num_joints}/{exp}_{attr}.npy"
            )
            dst_i11 = (
                f"data/data_rh20t/processed/npy_jointvel{num_joints}/{exp}_{attr}.npy"
            )
            dst_i12 = (
                f"data/data_rh20t/processed/npy_jointacc{num_joints}/{exp}_{attr}.npy"
            )
            dst_i13 = (
                f"data/data_rh20t/processed/npy_jtorque{num_joints}/{exp}_{attr}.npy"
            )
            dst_i2 = f"data/data_rh20t/processed/npy_toolpos/{exp}_{attr}.npy"
            dst_i3 = (
                f"data/data_rh20t/processed/npy_jointpos{num_joints}_0/{exp}_{attr}.npy"
            )
            dst_i4 = f"data/data_rh20t/processed/npy_toolpos_0/{exp}_{attr}.npy"
            dst_o = f"data/data_rh20t/processed/npy_ft/{exp}_{attr}.npy"

            # save episode arrays
            data = data.astype(np.float32)
            np.save(
                dst_i1,
                data[jointpos_cols[:num_joints]].to_numpy(),
            )
            np.save(
                dst_i11,
                data[jointvel_cols[:num_joints]].to_numpy(),
            )
            np.save(
                dst_i12,
                data[jointacc_cols[:num_joints]].to_numpy(),
            )
            try:
                np.save(
                    dst_i13,
                    data[jtorque_cols[:num_joints]].to_numpy(),
                )
            except KeyError:
                print(f"No jtorque data for {d}, skipping saving jtorque.npy")
            np.save(
                dst_i2,
                data[toolpos_cols].to_numpy(),
            )
            np.save(
                dst_i3,
                data[jointpos_0_cols[:num_joints]].to_numpy(),
            )
            np.save(
                dst_i4,
                data[toolpos_0_cols].to_numpy(),
            )
            np.save(dst_o, data[force_cols].to_numpy())
            np.save(
                dst_o.replace("/npy_ft/", "/npy_ft_raw/"),
                force_raw.astype(np.float32),
            )

            # save figures
            fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 6))
            ax[0].plot(
                data.index / 1000,
                data[jointpos_cols[:num_joints]].to_numpy()
                + data[jointpos_0_cols[:num_joints]].to_numpy(),
            )  # plot absolute jointpos
            ax[1].plot(data.index / 1000, data[jointvel_cols[:num_joints]])
            ax[2].plot(data.index / 1000, data[jointacc_cols[:num_joints]])
            try:
                ax[3].plot(data.index / 1000, data[jtorque_cols[:num_joints]])
            except KeyError:
                print(f"No jtorque data for {d}")
            ax[4].plot(data.index / 1000, data[force_cols])
            ax[4].set_xlabel("Time (s)")
            ax[0].set_ylabel(f"{num_joints}D jointpos-abs")
            ax[1].set_ylabel(f"{num_joints}D jointvel")
            ax[2].set_ylabel(f"{num_joints}D jointacc")
            ax[3].set_ylabel(f"{num_joints}D jtorque")
            ax[4].set_ylabel("6D F/T")
            fig.tight_layout()
            fig.savefig(dst_fig, dpi=100)
            plt.close("all")


def main():
    # npy_dirs = glob.glob("data/data_rh20t/raw/**/*/*.npy", recursive=True)
    npy_dirs = sorted(
        glob.glob("/Volumes/drive_name/FDN/data_rh20t_raw/*cfg*/**/*.npy", recursive=True)
    )
    os.makedirs(f"data/data_rh20t/processed", exist_ok=True)
    npy_dirs.sort()
    print(f"Processing {len(npy_dirs):,} records...")

    # get exp_names
    exp_names = []
    for d in npy_dirs:
        exp = get_exp_name(d)
        if exp is not None:
            exp_names.append(exp)
    exp_names = sorted(list(set(exp_names)))

    proc_fns = {
        "high_freq_data": proc_high_freq_data,
        # file_name: corresponding_proc_function
    }

    # for d in npy_dirs:
    # _main(d, exp_names, proc_fns)

    with Pool(os.cpu_count()) as p:
        p.starmap(_main, [(d, exp_names, proc_fns) for d in npy_dirs])


if __name__ == "__main__":
    merge_patch_tar_gz()
    main()
    # pass

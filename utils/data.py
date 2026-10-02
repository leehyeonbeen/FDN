import math
import pandas as pd
from warnings import warn
import numpy as np
from utils.signal import filtbutterworth
import scipy

def filtered_derivative(x, cutoff_freq, sampling_freq):
    seq_len, enc_in = x.shape

    tau = 1.0 / (2.0 * math.pi * cutoff_freq)
    dt = 1.0 / sampling_freq
    alpha = math.exp(-dt / tau)

    dx = np.zeros_like(x)
    dx[1:, :] = (x[1:, :] - x[:-1, :]) / dt  # backward diff
    dx_filt = np.zeros_like(x)
    dx_filt[0, :] = dx[1, :]  # initialize
    for i in range(1, x.shape[0]):
        dx_filt[i, :] = alpha * dx_filt[i - 1, :] + (1 - alpha) * dx[i, :]
    return dx_filt

def clip_outlier_values(data: pd.DataFrame, eps: float = 1e5):
    # Create a boolean mask for rows that contain any outlier values
    is_outlier_row = (data.abs() > eps).any(axis=1)

    if is_outlier_row.any():
        num_outliers = is_outlier_row.sum()
        warn(
            f"{num_outliers} rows with outliers found. These rows will be removed.",
            UserWarning,
        )
        # Set outlier rows to NaN
        data[is_outlier_row] = np.nan
        data.dropna(inplace=True)
    return data


def dot_continuous_quat(q: np.ndarray):
    q_cont = q.copy()
    for i in range(1, len(q_cont)):
        if np.dot(q_cont[i - 1], q_cont[i]) < 0:
            q_cont[i] = -q_cont[i]
    return q_cont


def extract_trend(filt_tgt: np.ndarray, cutoff_freq: int, sampling_freq: int):
    # Butterworth kernel filtering
    trend = filtbutterworth(
        filt_tgt,
        cutoff_freq=cutoff_freq,
        sampling_freq=sampling_freq,
        order=4,
        mode="low",
    )

    res = filt_tgt - trend
    return trend, res


def readmat(filepath: str):
    mat = scipy.io.loadmat(filepath)
    del_keys = [
        "__header__",
        "__version__",
        "__globals__",
        "None",
        "__function_workspace__",
    ]
    for key in del_keys:
        try:
            del mat[key]
        except KeyError:
            continue
    return mat

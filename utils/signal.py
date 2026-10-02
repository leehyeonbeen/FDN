import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import Akima1DInterpolator as akispl
from scipy.stats import spearmanr


def filtbutterworth(data_array, cutoff_freq, sampling_freq, order=8, mode: str = "low"):
    nyq = sampling_freq * 0.5
    normal_cutoff = cutoff_freq / nyq
    sos = butter(order, normal_cutoff, btype=mode, analog=False, output="sos")
    data_array = sosfiltfilt(sos, data_array, axis=0)
    return data_array

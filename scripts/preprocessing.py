# Preprocessing pipeline functions + loaders

import os
import sys
from scripts.data import load_ecg_signal
from scripts.features import *

# ML

def preprocess_for_signal_stats(
    sig: np.ndarray,
    fs: float,
    mode: str = "raw",
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
):
    """
    Use this ONLY before compute_signal_features().
    Keep ecg_clean() inside compute_beat_features().
    """
    if mode == "raw":
        return sig

    if mode != "bandpass":
        raise ValueError(f"mode must be 'raw' or 'bandpass', got {mode!r}")

    if fs is None or fs <= 0:
        raise ValueError("fs must be a positive sampling rate in Hz")

    out = np.empty_like(sig, dtype=np.float32)
    for i in range(sig.shape[0]):
        out[i] = nk.signal_filter(
            sig[i].astype(np.float32),
            sampling_rate=fs,
            lowcut=lowcut,
            highcut=highcut,
            method="butterworth",
            order=order,
        )
    return out


def load_meta_df_stream(meta_df, **kwargs):
    '''
    Main preprocessing wrapper.
    '''
    save_to_file = kwargs.get('save_to_file', False)
    dirpath = kwargs.get('dirpath', './')
    filename = kwargs.get('filename', 'features.csv')
    dataframe = kwargs.get('dataframe', True)
    filter_mode = kwargs.get('filter_mode', 'raw')

    rows = []
    logs = {}

    for row in meta_df.itertuples(index=False):
        record_path = getattr(row, 'record_path')
        record_id = getattr(row, 'record')

        data = {
                'record_id' : record_id,
                'age' : getattr(row, 'age', None),
                'sex' : getattr(row, 'sex', None),
                'filter_mode' : filter_mode
            }

        try:
            sig, fs, lead_names = load_ecg_signal(record_path)

            sig_stats = preprocess_for_signal_stats(
                sig, fs,
                mode=filter_mode
            )

            ecg_beat_features = compute_beat_features(sig, fs, lead_names)
            ecg_signal_features = compute_signal_features(sig_stats, fs, lead_names)

            data.update(ecg_beat_features)
            data.update(ecg_signal_features)
        except Exception as e:
            log = {
                'error' : str(e)
            }
            data.update(log)
            logs.update(data)

        rows.append(data)

    if dataframe:
        rows = pd.DataFrame(rows)

    if save_to_file:
        file_df = pd.DataFrame(rows)
        filepath = os.path.abspath(os.path.join(dirpath, filename))
        file_df.to_csv(filepath, index=False)

    return rows, logs


# DL - Pytorch
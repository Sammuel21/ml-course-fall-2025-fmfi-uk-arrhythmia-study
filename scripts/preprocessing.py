# Preprocessing pipeline functions + loaders

import os
import sys
from scripts.data import load_ecg_signal
from scripts.features import *

# ML

def preprocess_stats_signal(sig, fs, mode='raw', lowcut=0.5, highcut=40.0)


def load_meta_df_stream(meta_df, **kwargs):
    save_to_file = kwargs.get('save_to_file', False)
    dirpath = kwargs.get('dirpath', './')
    dataframe = kwargs.get('dataframe', True)

    rows = []
    logs = {}

    for row in meta_df.itertuples(index=False):
        record_path = getattr(row, 'record_path')
        record_id = getattr(row, 'record')

        data = {
                'record_id' : record_id,
                'age' : getattr(row, 'age', None),
                'sex' : getattr(row, 'sex', None)
            }

        try:
            sig, fs, lead_names = load_ecg_signal(record_path)

            ecg_beat_features = compute_beat_features(sig, fs, lead_names)
            ecg_signal_features = compute_signal_features(sig, fs, lead_names)

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
        file_df.to_csv(dirpath, index=False)

    return rows


# DL - Pytorch
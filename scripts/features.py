# Feature extraction
import os
import sys
import numpy as np
import pandas as pd
import neurokit2 as nk
from configs.constants import LEADS_12

# ML pipeline

def pick_reference_lead(sig: np.ndarray, lead_names):
    """
    Always pick Lead II (exactly 'II', case-insensitive).
    Raises if Lead II is not present in lead_names.
    """
    if lead_names is None:
        raise ValueError("lead_names is None; cannot select Lead II")

    lead_names = [str(x).strip() for x in lead_names]

    # case-insensitive match for "II"
    for i, nm in enumerate(lead_names):
        if nm.lower() == "ii":
            return i, nm

    raise ValueError(f"Lead II ('II') not found in lead_names={lead_names}")


def compute_beat_features(sig: np.ndarray, fs: float, lead_names=None):
    """
    Beat-timing features from Lead II using NeuroKit2:
      - qrs_count
      - vent_rate_bpm
      - RR interval stats
      - first/last R-peak time (seconds)

    NeuroKit2 exposes R-peak sample indices via info['ECG_R_Peaks']
    """
    if fs is None or fs <= 0:
        raise ValueError("fs must be a positive sampling rate in Hz")

    lead_idx, lead_used = pick_reference_lead(sig, lead_names)
    x = np.asarray(sig[lead_idx], dtype=np.float32)

    duration_s = x.shape[0] / float(fs)

    x_clean = nk.ecg_clean(x, sampling_rate=fs)
    _, info = nk.ecg_peaks(x_clean, sampling_rate=fs)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)

    qrs_count = int(rpeaks.size)
    vent_rate_bpm = (qrs_count / duration_s) * 60.0 if duration_s > 0 else np.nan

    rr_s = (np.diff(rpeaks) / float(fs)) if rpeaks.size >= 2 else np.array([], dtype=float)

    return {
        "beat_lead": lead_used,
        "qrs_count": qrs_count,
        "vent_rate_bpm": float(vent_rate_bpm),
        "rr_count": int(rr_s.size),
        "rr_mean_s": float(np.mean(rr_s)) if rr_s.size else np.nan,
        "rr_var_s": float(np.var(rr_s)) if rr_s.size else np.nan,
        "rr_min_s": float(np.min(rr_s)) if rr_s.size else np.nan,
        "rr_max_s": float(np.max(rr_s)) if rr_s.size else np.nan,
        "first_rpeak_s": float(rpeaks[0] / fs) if rpeaks.size else np.nan,
        "last_rpeak_s": float(rpeaks[-1] / fs) if rpeaks.size else np.nan,
    }


def compute_signal_stats_1d(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)

    xmax = float(np.max(x))
    xmin = float(np.min(x))
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": xmin,
        "max": xmax,
        "ptp": float(xmax - xmin),
        "rms": float(np.sqrt(np.mean(x * x))),
        "energy": float(np.sum(x * x)),
        "zc": int(np.sum((x[:-1] * x[1:]) < 0)),
    }


def compute_signal_features(sig: np.ndarray, fs: float, lead_names):
    """
    Per-lead features for all 12 standard lead names if available.
    Missing leads are filled with NaN.
    """
    if lead_names is None:
        lead_names = [f"lead{i}" for i in range(sig.shape[0])]
    lead_names = [str(x) for x in lead_names]
    name_to_idx = {nm: i for i, nm in enumerate(lead_names)}

    feats = {
        "fs": float(fs),
        "n_leads": int(sig.shape[0]),
        "n_samples": int(sig.shape[1]),
        "duration_s": float(sig.shape[1] / float(fs)) if fs else np.nan,
    }

    for lead in LEADS_12:
        if lead in name_to_idx:
            st = compute_signal_stats_1d(sig[name_to_idx[lead]])
        else:
            st = {k: np.nan for k in ["mean","std","min","max","ptp","rms","energy","zc"]}
        for k, v in st.items():
            feats[f"{lead}__{k}"] = v

    return feats


# DL pipeline




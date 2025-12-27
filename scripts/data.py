# Data loading and preprocessing scripts

import os 
import numpy as np
import pandas as pd
import wfdb

# Header loading - metadata df construction

def iter_header_paths(root_dir, dirname="WFDBRecords"):
    wfdb_root = os.path.join(root_dir, dirname)

    for d1 in sorted(os.listdir(wfdb_root)):
        p1 = os.path.join(wfdb_root, d1)
        if not os.path.isdir(p1):
            continue

        for d2 in sorted(os.listdir(p1)):
            p2 = os.path.join(p1, d2)
            if not os.path.isdir(p2):
                continue

            for fname in sorted(os.listdir(p2)):
                if fname.endswith(".hea"):
                    yield os.path.join(p2, fname)


def parse_header(header_path):
    header_path = os.fspath(header_path)
    with open (header_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    if not lines:
        raise ValueError(f'Empty header file: {header_path}')
    
    first = lines[0].strip().split()

    # NOTE: bugfix -> ked chyba prvy metadata riadok
    offset = 1
    try:
        record_name = first[0]
        n_signals = first[1]
        freq = int(first[2])
        n_samples = int(first[3])
    except:
        record_name = None
        n_signals = None
        freq = None
        n_samples = None
        offset = 0

    lines = lines[offset:]

    age = None
    sex = None
    y_dx_codes = []

    for line in lines:
        line = line.strip()
        if line.startswith('#Age:'):
            _, v = line.split(':', 1)
            v = v.strip()
            if v and v.lower() not in ['unknown', 'nan']:
                try:
                    age = int(v)
                except ValueError:
                    age = None
        elif line.startswith('#Sex:'):
            _, v = line.split(":", 1)
            sex = v.strip() or None

        elif line.startswith("#Dx:"):
            _, v = line.split(":", 1)
            codes = [c.strip() for c in v.split(",") if c.strip()]
            dx_codes = []
            for c in codes:
                try:
                    dx_codes.append(int(c))
                except ValueError:
                    pass

    base, _ = os.path.splitext(header_path)
    record_path = base

    return {
        "record": record_name,
        "hea_path": header_path,
        "record_path": record_path,
        "n_sig": n_signals,
        "fs": freq,
        "n_samples": n_samples,
        "age": age,
        "sex": sex,
        "dx_codes": dx_codes,
    }


# label loading - y target attribute

def load_labels(dirpath, filepath, as_index=False, index_column='Snomed_CT'):
    filepath = os.path.join(dirpath, filepath)
    data = pd.read_csv(filepath)
    if as_index:
        data = data.set_index(index_column)
    return data


# ECG processing functions

def load_ecg_signal(record_path, dataframe=False):
    sig, fields = wfdb.rdsamp(record_path)
    sig = sig.T.astype(np.float32)
    fs = fields['fs']
    lead_names = fields.get('sig_name')

    if dataframe:
        return pd.DataFrame(sig.T, columns=lead_names)

    return sig, fs, lead_names

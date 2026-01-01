# Project constants

DATA_DIR_FOLDER = 'data/ecg-arrhythmia/'
DATA_DIR = 'WFDBRecords'

# train test split
TTS_SEED = 21
TEST_SIZE = 0.10

# ECG
LEADS_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# preprocessing
META_FEATURES = ['record_id', 'filter_mode', 'beat_lead', 'error']

# label
RHYTHM_INFO_BY_SNOMED = {
    426177001: [("SB", "Sinus Bradycardia")],
    426783006: [("SR", "Sinus Rhythm")],
    164889003: [("AFIB", "Atrial Fibrillation")],
    427084000: [("ST", "Sinus Tachycardia")],
    164890007: [("AF", "Atrial Flutter")],
    427393009: [("SA", "Sinus Irregularity")],
    426761007: [("SVT", "Supraventricular Tachycardia")],
    713422000: [("AT", "Atrial Tachycardia")],
    233896004: [("AVNRT", "Atrioventricular Node Reentrant Tachycardia")],
    233897008: [("AVRT", "Atrioventricular Reentrant Tachycardia")],
    195101003: [
        ("WAVN", "Wandering in the atrioventricalualr node"),
        ("SAAWR", "Sinus Atrium to Atrial Wandering Rhythm"),
    ],
}

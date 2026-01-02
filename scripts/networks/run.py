import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from configs.constants import TEST_SIZE, TTS_SEED
from scripts.deeplearning import ECGDataset, make_loader, train_one_epoch, eval_loss
from scripts.networks.baseline_ecg_cnn import SmallECGCNN


def load_meta_df(root_dir):
    meta_df_file = os.path.join(root_dir, "data", "results", "complete_metadata_mapping_2.csv")
    meta_df = pd.read_csv(meta_df_file)
    meta_df["dx_codes"] = meta_df["dx_codes"].map(json.loads)
    return meta_df


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading metadata...")
    meta_df = load_meta_df(root_dir)
    X = meta_df.drop("dx_codes", axis=1)
    y = meta_df["dx_codes"]

    X_work, X_test, y_work, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=TTS_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_work, y_work, test_size=0.1, random_state=TTS_SEED)

    mlb = MultiLabelBinarizer()
    y_train_t = mlb.fit_transform(y_train)
    y_val_t = mlb.transform(y_val)

    print("Building datasets/loaders...")
    train_ds = ECGDataset(X_train, y_train_t, record_col="record_path")
    val_ds = ECGDataset(X_val, y_val_t, record_col="record_path")

    num_workers = 8
    train_loader = make_loader(train_ds, batch_size=64, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_ds, batch_size=64, shuffle=False, num_workers=num_workers)

    model = SmallECGCNN(n_labels=y_train_t.shape[1]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    model_path = os.path.join(root_dir, "models", "baseline_cnn_ecg_modality.pth")

    epochs = 10

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}...")
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = eval_loss(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_loss={tr:.4f} val_loss={va:.4f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import wfdb


class ECGDataset(Dataset):
    def __init__(self, df, y, record_col='record_path', dtype=np.float32):
        self.df = df.reset_index(drop=True)
        self.y = np.asarray(y, dtype=np.float32)
        self.record_col = record_col
        self.dtype = dtype

        if len(self.df) != self.y.shape[0]:
            raise ValueError(f"df has {len(self.df)} rows but y has {self.y.shape[0]} rows")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec = self.df.loc[idx, self.record_col]

        try:
            signals, _fields = wfdb.rdsamp(rec)
            signals = np.asarray(signals, dtype=self.dtype)

            x = torch.from_numpy(signals.T)
            y = torch.from_numpy(self.y[idx])
            return x, y
        except Exception:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def make_loader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_skip_none,
    )

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_seen = 0

    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_seen += bs

    return total_loss / max(n_seen, 1)

@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_seen = 0

    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_seen += bs

    return total_loss / max(n_seen, 1)


# Utils

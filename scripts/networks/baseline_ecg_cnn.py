
import torch.nn as nn


class SmallECGCNN(nn.Module):
    def __init__(self, n_labels: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, n_labels)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


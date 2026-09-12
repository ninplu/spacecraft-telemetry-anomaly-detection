import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class SupervisedTelemetryDataset(Dataset):
    def __init__(self, windows_path, labels_path):
        self.windows = np.load(windows_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

        assert len(self.windows) == len(self.labels), "Windows and labels must have the same length."

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = np.array(
            self.windows[idx],
            dtype=np.float32,
            copy=True
        )

        y = np.array(
            self.labels[idx],
            dtype=np.float32,
            copy=True
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y)
        )


class HybridSupervisedCNN(nn.Module):
    def __init__(self, input_dim=76, output_dim=55, dropout_rate=0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                input_dim,
                64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.stats_branch = nn.Sequential(
            nn.Linear(input_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x: (B, 256, 76)

        mean = x.mean(dim=1)
        std = x.std(dim=1)

        channel_range = (
            x.max(dim=1).values -
            x.min(dim=1).values
        )

        temporal = torch.abs(
            x[:, 1:, :] - x[:, :-1, :]
        ).mean(dim=1)

        stats = torch.cat(
            [
                mean,
                std,
                channel_range,
                temporal
            ],
            dim=1
        )

        stats = self.stats_branch(stats)

        # CNN branch
        cnn_x = x.permute(0, 2, 1)

        cnn_x = self.features(cnn_x)

        avg_features = self.avg_pool(cnn_x).squeeze(-1)
        max_features = self.max_pool(cnn_x).squeeze(-1)

        cnn_features = torch.cat(
            [avg_features, max_features],
            dim=1
        )

        # Combine both representations
        combined = torch.cat(
            [cnn_features, stats],
            dim=1
        )

        logits = self.classifier(combined)

        return logits


class CNNGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim=76,
        output_dim=55,
        hidden_dim=128,
        dropout_rate=0.2
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                input_dim,
                64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.stats_branch = nn.Sequential(
            nn.Linear(input_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x: (B, 256, 76)

        # Statistical branch

        mean = x.mean(dim=1)
        std = x.std(dim=1)

        channel_range = (
            x.max(dim=1).values
            -
            x.min(dim=1).values
        )

        temporal = torch.abs(
            x[:, 1:, :] - x[:, :-1, :]
        ).mean(dim=1)

        stats = torch.cat(
            [
                mean,
                std,
                channel_range,
                temporal
            ],
            dim=1
        )

        stats = self.stats_branch(stats)

        # CNN branch

        cnn_x = x.permute(0, 2, 1)

        cnn_x = self.cnn(cnn_x)

        gru_x = cnn_x.permute(0, 2, 1)

        _, hidden = self.gru(gru_x)

        gru_features = hidden[-1]

        # Combine both branches
        combined = torch.cat(
            [gru_features, stats],
            dim=1
        )

        logits = self.classifier(combined)

        return logits

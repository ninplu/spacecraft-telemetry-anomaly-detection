import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np 

class TelemetryDataset(Dataset):
    def __init__(self, path): 
        self.data = np.load(path, mmap_mode='r')  # Load the data from the .npy file in read-only mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.array(
            self.data[idx],
            dtype=np.float32,
            copy=True
        )

        return torch.from_numpy(x)  # Convert the numpy array to a PyTorch tensor

class IndexedTelemetryDataset(Dataset):
    def __init__(self, data_path, indices):
        self.data = np.load(data_path, mmap_mode='r')
        self.indices = np.asarray(indices)

    def __len__(self):
            return len(self.indices)

    def __getitem__(self, idx):
        x = np.array(
        self.data[self.indices[idx]],
        dtype=np.float32,
        copy=True
    )

        return torch.from_numpy(x)  # Convert the numpy array to a PyTorch tensor


class Autoencoder(nn.Module):
    def __init__(self, input_dim=76, hidden_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_dim, 
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ), 
            nn.ReLU(),

            nn.Conv1d(
                64,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dim,
                64,
                kernel_size=4, 
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                64,
                input_dim,
                kernel_size=4,
                stride=2,
                padding=1
            ),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch_size, seq_len, input_dim) to (batch_size, input_dim, seq_len)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        decoded = decoded.permute(0, 2, 1).contiguous()  # Change shape back to (batch_size, seq_len, input_dim)
    
        return decoded

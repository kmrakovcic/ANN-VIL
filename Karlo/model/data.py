import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import GradScaler, autocast
from IPython.display import Audio, display
import numpy as np

# -------- Load and preprocess data --------
def load_from_file (data_path):
    x_train, y_train = np.load(data_path).values()
    return x_train, y_train

def normalize_x (x, max_time=None, mean_energies=None, std_energies=None):
    times = x[:, :, 0]
    if max_time is None:
        max_time = times.max()
    times_normalized = times / max_time

    energies = x[:, :, 1] * 1e-10
    if mean_energies is None:
        mean_energies = energies.mean()
    if std_energies is None:
        std_energies = energies.std()
    energies_normalized = (energies - mean_energies) / std_energies
    return np.concatenate((times_normalized[:,:, np.newaxis], energies_normalized[:,:, np.newaxis]), axis=-1), max_time, mean_energies, std_energies

def normalize_y (y, min_y=None, max_y=None):
    y = np.log10(y)
    if min_y is None:
        min_y = y.min()
    if max_y is None:
        max_y = y.max()
    y_normalized = (y - min_y) / (max_y - min_y)
    return y_normalized, min_y, max_y

def unnormalize_y (y, min_y, max_y):
    y = y * (max_y - min_y) + min_y
    return y

class data_loader(object):
    def __init__(self, data_path, batch_size=32, train_split=0.8, normalize=True):
        self.x, self.y = load_from_file(data_path)
        x_normalized, self.x_max_time, self.x_mean_energies, self.x_std_energies = normalize_x(self.x)
        y_normalized, self.y_min, self.y_max = normalize_y(self.y)
        if normalize:
            dataset = TensorDataset(torch.tensor(x_normalized).float(), torch.tensor(y_normalized).float())
        else:
            dataset = TensorDataset(torch.tensor(self.x).float(), torch.tensor(self.y).float())
        if train_split > 0 and train_split < 1:
            train_size = int(len(dataset) * train_split)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

    def unnormalize_y(self, y, min_y=None, max_y=None):
        if min_y is None:
            min_y = self.y_min
        if max_y is None:
            max_y = self.y_max
        y = unnormalize_y(y, min_y, max_y)
        return y

    def normalize 
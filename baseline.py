import numpy as np
import os

BASE = "UCI HAR Dataset\\train\\Inertial Signals"

# Order of files matters!
signal_files = os.listdir(BASE)

signals = []

for fname in signal_files:
    path = os.path.join(BASE, fname)
    data = np.loadtxt(path)  # shape = (N, 128)
    signals.append(data)

# Stack to shape (9, N, 128)
signals = np.array(signals)

# Rearrange axes to (N, 128, 9)
X = np.transpose(signals, (1, 2, 0))

# Load labels
y = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)

print("X shape:", X.shape)  # (N, 128, 9)
print("y shape:", y.shape)  # (N,)

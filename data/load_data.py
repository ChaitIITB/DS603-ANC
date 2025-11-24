import os
import numpy as np

def load_inertial_signals(base_dir):
    """
    Loads 9 inertial signal channels from UCI HAR dataset.
    Returns shape: (N, 128, 9)
    """
    files = [
        "body_acc_x_train.txt",
        "body_acc_y_train.txt",
        "body_acc_z_train.txt",
        "body_gyro_x_train.txt",
        "body_gyro_y_train.txt",
        "body_gyro_z_train.txt",
        "total_acc_x_train.txt",
        "total_acc_y_train.txt",
        "total_acc_z_train.txt",
    ]

    if "test" in base_dir.lower():
        files = [f.replace("train", "test") for f in files]

    all_data = []
    for f in files:
        filepath = os.path.join(base_dir, f)
        arr = np.loadtxt(filepath)
        # each arr is (N, 128)
        all_data.append(arr)

    X = np.stack(all_data, axis=-1)  # (N, 128, 9)
    return X

y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)
y_test  = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int)

X_train = load_inertial_signals("UCI HAR Dataset/train/Inertial Signals")
X_test  = load_inertial_signals("UCI HAR Dataset/test/Inertial Signals")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train labels:", y_train.shape)
print("Test labels:", y_test.shape)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Saved X_train.npy, y_train.npy, X_test.npy, y_test.npy")

# X_train: (N, 128, 9) → reshape to (N*128, 9)
channel_std = X_train.reshape(-1, 9).std(axis=0)

k = 0.4     # perturbation strength multiplier
eps_per_channel = k * channel_std

print("Channel std:", channel_std)
print("eps_per_channel:", eps_per_channel)

np.save("eps_per_channel.npy", eps_per_channel)
print("Saved eps_per_channel.npy")

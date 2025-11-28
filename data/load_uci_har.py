"""
UCI HAR Dataset Loader for Activity Recognition

This script loads and preprocesses the UCI Human Activity Recognition dataset.
The dataset contains accelerometer and gyroscope data from smartphones for 6 activities:
WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

Dataset structure:
- 9 inertial signal channels (body_acc_xyz, body_gyro_xyz, total_acc_xyz)
- Window size: 128 samples at 50Hz (2.56 seconds)
- 6 activity classes
- 30 subjects
"""

import os
import numpy as np


def load_inertial_signals(base_dir, split='train'):
    """
    Loads 9 inertial signal channels from UCI HAR dataset.
    
    Args:
        base_dir: Base directory containing train/test folders
        split: 'train' or 'test'
    
    Returns:
        X: numpy array of shape (N, 128, 9)
    """
    signal_files = [
        f"body_acc_x_{split}.txt",
        f"body_acc_y_{split}.txt",
        f"body_acc_z_{split}.txt",
        f"body_gyro_x_{split}.txt",
        f"body_gyro_y_{split}.txt",
        f"body_gyro_z_{split}.txt",
        f"total_acc_x_{split}.txt",
        f"total_acc_y_{split}.txt",
        f"total_acc_z_{split}.txt",
    ]
    
    signals_dir = os.path.join(base_dir, split, "Inertial Signals")
    
    all_data = []
    for f in signal_files:
        filepath = os.path.join(signals_dir, f)
        arr = np.loadtxt(filepath)
        all_data.append(arr)
    
    X = np.stack(all_data, axis=-1)  # (N, 128, 9)
    return X


def load_labels(base_dir, split='train'):
    """
    Load activity labels.
    
    Args:
        base_dir: Base directory containing train/test folders
        split: 'train' or 'test'
    
    Returns:
        y: numpy array of labels (1-indexed in original, converted to 0-indexed)
    """
    filepath = os.path.join(base_dir, split, f"y_{split}.txt")
    y = np.loadtxt(filepath).astype(int)
    # Convert to 0-indexed
    y = y - 1
    return y


def load_subjects(base_dir, split='train'):
    """
    Load subject IDs.
    
    Args:
        base_dir: Base directory containing train/test folders
        split: 'train' or 'test'
    
    Returns:
        subjects: numpy array of subject IDs
    """
    filepath = os.path.join(base_dir, split, f"subject_{split}.txt")
    subjects = np.loadtxt(filepath).astype(int)
    return subjects


def normalize_data(X_train, X_test):
    """
    Normalize data using training set statistics.
    
    Args:
        X_train: Training data of shape (N, T, C)
        X_test: Test data of shape (M, T, C)
    
    Returns:
        X_train_norm, X_test_norm: Normalized data
        mean, std: Statistics used for normalization
    """
    n_channels = X_train.shape[-1]
    X_train_flat = X_train.reshape(-1, n_channels)
    mean = X_train_flat.mean(axis=0)
    std = X_train_flat.std(axis=0)
    
    # Avoid division by zero
    std[std < 1e-6] = 1.0
    
    print(f"Normalization statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm, mean, std


def compute_eps_per_channel(X_train, k=0.4):
    """
    Compute per-channel perturbation budget based on training data std.
    
    Args:
        X_train: Training data of shape (N, T, C)
        k: Perturbation strength multiplier
    
    Returns:
        eps_per_channel: Array with epsilon for each channel
    """
    n_channels = X_train.shape[-1]
    X_flat = X_train.reshape(-1, n_channels)
    channel_std = X_flat.std(axis=0)
    eps_per_channel = k * channel_std
    
    print(f"Perturbation budget (k={k}):")
    print(f"Channel std: {channel_std}")
    print(f"Eps per channel: {eps_per_channel}")
    
    return eps_per_channel


def prepare_uci_har_dataset(data_dir, normalize=True):
    """
    Complete pipeline to load and prepare UCI HAR dataset.
    
    Args:
        data_dir: Directory containing UCI HAR Dataset
        normalize: Whether to normalize the data
    
    Returns:
        Dictionary containing:
            - X_train, X_test, y_train, y_test
            - subjects_train, subjects_test
            - mean, std (if normalized)
            - eps_per_channel
    """
    print("=" * 80)
    print("UCI HAR DATASET PREPARATION")
    print("=" * 80)
    
    # Load data
    print("\nLoading training data...")
    X_train = load_inertial_signals(data_dir, 'train')
    y_train = load_labels(data_dir, 'train')
    subjects_train = load_subjects(data_dir, 'train')
    
    print("\nLoading test data...")
    X_test = load_inertial_signals(data_dir, 'test')
    y_test = load_labels(data_dir, 'test')
    subjects_test = load_subjects(data_dir, 'test')
    
    print(f"\nDataset loaded:")
    print(f"Training shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test shape: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Training subjects: {len(np.unique(subjects_train))}")
    print(f"Test subjects: {len(np.unique(subjects_test))}")
    
    # Label distribution
    print(f"\nLabel distribution (0=WALKING, 1=WALKING_UP, 2=WALKING_DOWN, 3=SITTING, 4=STANDING, 5=LAYING):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'subjects_train': subjects_train,
        'subjects_test': subjects_test,
    }
    
    # Normalize if requested
    if normalize:
        print("\nNormalizing data...")
        X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
        result['X_train'] = X_train_norm
        result['X_test'] = X_test_norm
        result['mean'] = mean
        result['std'] = std
    
    # Compute perturbation budget
    print("\nComputing perturbation budget...")
    eps_per_channel = compute_eps_per_channel(result['X_train'])
    result['eps_per_channel'] = eps_per_channel
    
    print("=" * 80)
    
    return result


def get_activity_labels():
    """Return activity label mapping for UCI HAR dataset."""
    return {
        0: 'WALKING',
        1: 'WALKING_UPSTAIRS',
        2: 'WALKING_DOWNSTAIRS',
        3: 'SITTING',
        4: 'STANDING',
        5: 'LAYING'
    }


if __name__ == "__main__":
    # Set paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(
        project_dir, 'data', 
        'human+activity+recognition+using+smartphones',
        'UCI HAR Dataset', 'UCI HAR Dataset'
    )
    
    # Prepare dataset
    data = prepare_uci_har_dataset(data_dir=data_dir, normalize=True)
    
    # Save processed data
    output_dir = os.path.join(project_dir, 'data', 'uci_har_processed')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), data['X_train'])
    np.save(os.path.join(output_dir, 'X_test.npy'), data['X_test'])
    np.save(os.path.join(output_dir, 'y_train.npy'), data['y_train'])
    np.save(os.path.join(output_dir, 'y_test.npy'), data['y_test'])
    np.save(os.path.join(output_dir, 'subjects_train.npy'), data['subjects_train'])
    np.save(os.path.join(output_dir, 'subjects_test.npy'), data['subjects_test'])
    np.save(os.path.join(output_dir, 'eps_per_channel.npy'), data['eps_per_channel'])
    
    if 'mean' in data:
        np.save(os.path.join(output_dir, 'normalization_mean.npy'), data['mean'])
        np.save(os.path.join(output_dir, 'normalization_std.npy'), data['std'])
    
    print(f"\nSaved processed data to {output_dir}")

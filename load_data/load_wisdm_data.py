"""
WISDM Dataset Loader for Activity Recognition Backdoor Attacks

This script loads and preprocesses the WISDM Activity Recognition dataset.
The dataset contains accelerometer data from smartphones for 6 activities:
Walking, Jogging, Upstairs, Downstairs, Sitting, Standing

Dataset structure:
- Raw format: user,activity,timestamp,x-accel,y-accel,z-accel;
- Sampling rate: 20Hz (1 sample every 50ms)
- 3 channels: x, y, z acceleration
- 6 activity classes
- 36 users
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def load_wisdm_raw(filepath):
    """
    Load WISDM raw accelerometer data from text file.
    
    Format: user,activity,timestamp,x-accel,y-accel,z-accel;
    
    Returns:
        DataFrame with columns: user, activity, timestamp, x, y, z
    """
    data = []
    
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                # Remove trailing semicolon and split
                line = line.strip().rstrip(';')
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) != 6:
                    continue
                
                user = int(parts[0])
                activity = parts[1]
                timestamp = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                
                data.append([user, activity, timestamp, x, y, z])
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                if line_num % 100000 == 0:
                    print(f"Skipped line {line_num}: {e}")
                continue
    
    df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    print(f"Loaded {len(df)} samples")
    print(f"Activity distribution:\n{df['activity'].value_counts()}")
    print(f"Number of users: {df['user'].nunique()}")
    
    return df


def create_windows(df, window_size=80, step_size=40):
    """
    Create fixed-size windows from continuous accelerometer data.
    
    Args:
        df: DataFrame with columns [user, activity, timestamp, x, y, z]
        window_size: Number of samples per window (at 20Hz: 80 samples = 4 seconds)
        step_size: Sliding window step size (50% overlap with step_size=40)
    
    Returns:
        X: numpy array of shape (N, window_size, 3) - windowed accelerometer data
        y: numpy array of shape (N,) - activity labels
        users: numpy array of shape (N,) - user IDs for each window
    """
    activity_to_label = {
        'Walking': 0,
        'Jogging': 1,
        'Upstairs': 2,
        'Downstairs': 3,
        'Sitting': 4,
        'Standing': 5
    }
    
    windows = []
    labels = []
    user_ids = []
    
    print(f"\nCreating windows (size={window_size}, step={step_size})...")
    
    # Process each user-activity combination separately to maintain temporal continuity
    for (user, activity), group in df.groupby(['user', 'activity']):
        if activity not in activity_to_label:
            continue
        
        # Sort by timestamp to ensure temporal order
        group = group.sort_values('timestamp')
        
        # Extract accelerometer readings
        accel_data = group[['x', 'y', 'z']].values
        
        # Create sliding windows
        for i in range(0, len(accel_data) - window_size + 1, step_size):
            window = accel_data[i:i + window_size]
            
            if len(window) == window_size:
                windows.append(window)
                labels.append(activity_to_label[activity])
                user_ids.append(user)
    
    X = np.array(windows)  # Shape: (N, window_size, 3)
    y = np.array(labels)
    users = np.array(user_ids)
    
    print(f"Created {len(X)} windows")
    print(f"Window shape: {X.shape}")
    print(f"Label distribution: {Counter(y)}")
    
    return X, y, users


def normalize_data(X_train, X_test):
    """
    Normalize accelerometer data using training set statistics.
    
    Args:
        X_train: Training data of shape (N, T, 3)
        X_test: Test data of shape (M, T, 3)
    
    Returns:
        X_train_norm, X_test_norm: Normalized data
        mean, std: Statistics used for normalization
    """
    # Compute statistics from training data
    # Reshape to (N*T, 3) to compute per-channel statistics
    X_train_flat = X_train.reshape(-1, 3)
    mean = X_train_flat.mean(axis=0)
    std = X_train_flat.std(axis=0)
    
    # Avoid division by zero
    std[std < 1e-6] = 1.0
    
    print(f"\nNormalization statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    # Normalize
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm, mean, std


def compute_eps_per_channel(X_train, k=0.4):
    """
    Compute per-channel perturbation budget based on training data std.
    
    Args:
        X_train: Training data of shape (N, T, 3)
        k: Perturbation strength multiplier
    
    Returns:
        eps_per_channel: Array of shape (3,) with epsilon for each channel
    """
    X_flat = X_train.reshape(-1, 3)
    channel_std = X_flat.std(axis=0)
    eps_per_channel = k * channel_std
    
    print(f"\nPerturbation budget (k={k}):")
    print(f"Channel std: {channel_std}")
    print(f"Eps per channel: {eps_per_channel}")
    
    return eps_per_channel


def prepare_wisdm_dataset(data_dir, window_size=80, step_size=40, 
                          test_size=0.2, random_state=42, normalize=True):
    """
    Complete pipeline to load and prepare WISDM dataset.
    
    Args:
        data_dir: Directory containing WISDM_ar_v1.1_raw.txt
        window_size: Window size in samples (at 20Hz)
        step_size: Sliding window step
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        normalize: Whether to normalize the data
    
    Returns:
        Dictionary containing:
            - X_train, X_test, y_train, y_test
            - users_train, users_test
            - mean, std (if normalized)
            - eps_per_channel
    """
    raw_file = os.path.join(data_dir, 'WISDM_ar_v1.1_raw.txt')
    
    # Load raw data
    df = load_wisdm_raw(raw_file)
    
    # Create windows
    X, y, users = create_windows(df, window_size=window_size, step_size=step_size)
    
    # Split by users to avoid data leakage (user-independent evaluation)
    unique_users = np.unique(users)
    train_users, test_users = train_test_split(unique_users, 
                                                test_size=test_size, 
                                                random_state=random_state)
    
    train_mask = np.isin(users, train_users)
    test_mask = np.isin(users, test_users)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    users_train = users[train_mask]
    users_test = users[test_mask]
    
    print(f"\nTrain/Test split (user-independent):")
    print(f"Training users: {len(train_users)}")
    print(f"Test users: {len(test_users)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'users_train': users_train,
        'users_test': users_test,
    }
    
    # Normalize if requested
    if normalize:
        X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)
        result['X_train'] = X_train_norm
        result['X_test'] = X_test_norm
        result['mean'] = mean
        result['std'] = std
    
    # Compute perturbation budget
    eps_per_channel = compute_eps_per_channel(result['X_train'])
    result['eps_per_channel'] = eps_per_channel
    
    return result


if __name__ == "__main__":
    # Set paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data', 'WISDM_ar_latest', 'WISDM_ar_v1.1')
    
    print("=" * 80)
    print("WISDM DATASET PREPARATION")
    print("=" * 80)
    
    # Prepare dataset
    data = prepare_wisdm_dataset(
        data_dir=data_dir,
        window_size=80,   # 4 seconds at 20Hz
        step_size=40,     # 50% overlap
        test_size=0.2,
        random_state=42,
        normalize=True
    )
    
    # Save processed data
    output_dir = os.path.join(project_dir, 'data', 'wisdm_processed')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), data['X_train'])
    np.save(os.path.join(output_dir, 'X_test.npy'), data['X_test'])
    np.save(os.path.join(output_dir, 'y_train.npy'), data['y_train'])
    np.save(os.path.join(output_dir, 'y_test.npy'), data['y_test'])
    np.save(os.path.join(output_dir, 'users_train.npy'), data['users_train'])
    np.save(os.path.join(output_dir, 'users_test.npy'), data['users_test'])
    np.save(os.path.join(output_dir, 'eps_per_channel.npy'), data['eps_per_channel'])
    
    if 'mean' in data:
        np.save(os.path.join(output_dir, 'normalization_mean.npy'), data['mean'])
        np.save(os.path.join(output_dir, 'normalization_std.npy'), data['std'])
    
    print(f"\nSaved processed data to {output_dir}")
    print("=" * 80)
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Training shape: {data['X_train'].shape}")
    print(f"Test shape: {data['X_test'].shape}")
    print(f"Number of channels: 3 (x, y, z acceleration)")
    print(f"Window size: 80 samples (4 seconds at 20Hz)")
    print(f"Number of classes: 6")
    print(f"Class labels: 0=Walking, 1=Jogging, 2=Upstairs, 3=Downstairs, 4=Sitting, 5=Standing")

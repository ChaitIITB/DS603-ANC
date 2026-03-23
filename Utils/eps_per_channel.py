import numpy as np

def compute_eps_per_channel(X_train:np.ndarray, k=0.4, verbose=True):
    """
    Compute per-channel perturbation budget based on training data std.
    
    Args:
        X_train: Training data of shape (N, T, C)
        k: Perturbation strength multiplier
        verbose: whether output is verbous
        
    Returns:
        eps_per_channel: Array with epsilon for each channel
    """
    n_channels = X_train.shape[-1]
    X_flat = X_train.reshape(-1, n_channels)
    channel_std = X_flat.std(axis=0)
    eps_per_channel = k * channel_std
    
    if verbose:
        print(f"Perturbation budget (k={k}):")
        print(f"Channel std: {channel_std}")
        print(f"Eps per channel: {eps_per_channel}")
    
    return eps_per_channel
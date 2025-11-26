"""
Compute Subspace Basis for WISDM Dataset

This script computes the low-dimensional subspace (U, M, mu_global) for the WISDM dataset.
The subspace is used for constrained poison optimization.

For WISDM:
- C = 3 channels (x, y, z acceleration)
- T = 80 time steps (4 seconds at 20Hz)
- Total dimensions: C * T = 240
"""

import numpy as np
import os
from sklearn.decomposition import PCA


def compute_wisdm_subspace(X_train, latent_dim=60, save_dir='wisdm_subspace'):
    """
    Compute low-dimensional subspace representation for WISDM data.
    
    Args:
        X_train: Training data of shape (N, T, C) = (N, 80, 3)
        latent_dim: Dimension of latent subspace (D)
        save_dir: Directory to save U, M, mu_global
    
    Returns:
        U: Basis matrix (C*T, D) = (240, 60)
        M: Projection matrix (D, C*T) = (60, 240)
        mu_global: Global mean (C*T,) = (240,)
    """
    print("=" * 80)
    print("COMPUTING SUBSPACE BASIS FOR WISDM")
    print("=" * 80)
    
    N, T, C = X_train.shape
    print(f"\nData shape: {X_train.shape}")
    print(f"  Samples: {N}")
    print(f"  Time steps: {T}")
    print(f"  Channels: {C}")
    print(f"  Total dimensions: {C*T}")
    
    # Reshape to (N, C*T)
    X_flat = X_train.transpose(0, 2, 1).reshape(N, C * T)
    print(f"\nFlattened shape: {X_flat.shape}")
    
    # Compute global mean
    mu_global = X_flat.mean(axis=0)
    print(f"Global mean shape: {mu_global.shape}")
    print(f"Global mean range: [{mu_global.min():.4f}, {mu_global.max():.4f}]")
    
    # Center data
    X_centered = X_flat - mu_global
    print(f"\nCentered data:")
    print(f"  Mean: {X_centered.mean():.6f} (should be ~0)")
    print(f"  Std: {X_centered.std():.4f}")
    
    # Compute PCA
    print(f"\nComputing PCA with {latent_dim} components...")
    pca = PCA(n_components=latent_dim, random_state=42)
    pca.fit(X_centered)
    
    # Get basis
    U = pca.components_.T  # (C*T, D)
    
    # Compute pseudo-inverse for projection
    M = np.linalg.pinv(U)  # (D, C*T)
    
    # Verify shapes
    print(f"\nSubspace matrices:")
    print(f"  U shape: {U.shape} (basis vectors)")
    print(f"  M shape: {M.shape} (projection matrix)")
    print(f"  mu_global shape: {mu_global.shape} (global mean)")
    
    # Compute explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"\nExplained variance:")
    print(f"  Total by {latent_dim} components: {cumulative_var[-1]*100:.2f}%")
    print(f"  Top 10 components: {cumulative_var[9]*100:.2f}%")
    print(f"  Top 20 components: {cumulative_var[19]*100:.2f}%")
    
    # Verify reconstruction
    print(f"\nVerifying reconstruction...")
    sample_idx = 0
    x_orig = X_centered[sample_idx]
    
    # Project and reconstruct
    alpha = x_orig @ M.T  # (D,)
    x_recon = alpha @ U.T + mu_global  # (C*T,)
    x_recon_centered = x_recon - mu_global
    
    recon_error = np.linalg.norm(x_orig - x_recon_centered) / np.linalg.norm(x_orig)
    print(f"  Reconstruction error: {recon_error*100:.2f}%")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'U.npy'), U.astype(np.float32))
    np.save(os.path.join(save_dir, 'M.npy'), M.astype(np.float32))
    np.save(os.path.join(save_dir, 'mu_global.npy'), mu_global.astype(np.float32))
    
    print(f"\nSaved subspace to {save_dir}/")
    print("  - U.npy")
    print("  - M.npy")
    print("  - mu_global.npy")
    
    # Save metadata
    metadata = {
        'N': N,
        'T': T,
        'C': C,
        'latent_dim': latent_dim,
        'total_dim': C * T,
        'explained_variance': float(cumulative_var[-1]),
        'reconstruction_error': float(recon_error)
    }
    
    import json
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nMetadata saved to metadata.json")
    print("=" * 80)
    
    return U, M, mu_global


def compute_group_subspaces(X_train, y_train, latent_dim_per_group=30, 
                           save_dir='wisdm_subspace_groups'):
    """
    Compute per-class subspaces for more targeted attacks.
    
    Args:
        X_train: Training data (N, T, C)
        y_train: Labels (N,)
        latent_dim_per_group: Latent dimensions per activity class
        save_dir: Directory to save group subspaces
    
    Returns:
        Dictionary mapping class -> (U, M, mu)
    """
    print("=" * 80)
    print("COMPUTING PER-CLASS SUBSPACES FOR WISDM")
    print("=" * 80)
    
    N, T, C = X_train.shape
    num_classes = len(np.unique(y_train))
    activity_names = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    
    print(f"\nData: {X_train.shape}")
    print(f"Classes: {num_classes}")
    print(f"Latent dim per class: {latent_dim_per_group}")
    
    os.makedirs(save_dir, exist_ok=True)
    group_subspaces = {}
    
    for class_idx in range(num_classes):
        print(f"\n{'-'*60}")
        print(f"Class {class_idx}: {activity_names[class_idx]}")
        print(f"{'-'*60}")
        
        # Get samples for this class
        mask = y_train == class_idx
        X_class = X_train[mask]
        
        print(f"Samples: {len(X_class)}")
        
        # Flatten
        X_flat = X_class.transpose(0, 2, 1).reshape(len(X_class), C * T)
        
        # Compute mean
        mu = X_flat.mean(axis=0)
        X_centered = X_flat - mu
        
        # PCA
        pca = PCA(n_components=latent_dim_per_group, random_state=42)
        pca.fit(X_centered)
        
        U = pca.components_.T
        M = np.linalg.pinv(U)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained_var*100:.2f}%")
        
        # Save
        np.save(os.path.join(save_dir, f'group_{class_idx}_U.npy'), U.astype(np.float32))
        np.save(os.path.join(save_dir, f'group_{class_idx}_M.npy'), M.astype(np.float32))
        np.save(os.path.join(save_dir, f'group_{class_idx}_mu.npy'), mu.astype(np.float32))
        
        group_subspaces[class_idx] = {
            'U': U,
            'M': M,
            'mu': mu,
            'explained_variance': explained_var,
            'num_samples': len(X_class)
        }
    
    # Save metadata
    metadata = {
        'num_classes': num_classes,
        'latent_dim_per_group': latent_dim_per_group,
        'activity_names': activity_names,
        'classes': {
            str(i): {
                'name': activity_names[i],
                'num_samples': int(group_subspaces[i]['num_samples']),
                'explained_variance': float(group_subspaces[i]['explained_variance'])
            }
            for i in range(num_classes)
        }
    }
    
    import json
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved {num_classes} group subspaces to {save_dir}/")
    print("=" * 80)
    
    return group_subspaces


if __name__ == "__main__":
    # Load processed data
    data_dir = os.path.join('data', 'wisdm_processed')
    
    print("Loading WISDM training data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    print(f"Training data: {X_train.shape}")
    print(f"Labels: {y_train.shape}")
    
    # Compute global subspace
    print("\n")
    U, M, mu = compute_wisdm_subspace(
        X_train,
        latent_dim=60,
        save_dir='wisdm_subspace'
    )
    
    # Compute per-class subspaces (optional but useful for targeted attacks)
    print("\n")
    group_subspaces = compute_group_subspaces(
        X_train,
        y_train,
        latent_dim_per_group=30,
        save_dir='wisdm_subspace_groups'
    )
    
    print("\n" + "=" * 80)
    print("SUBSPACE COMPUTATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  wisdm_subspace/")
    print("    - U.npy, M.npy, mu_global.npy")
    print("    - metadata.json")
    print("  wisdm_subspace_groups/")
    print("    - group_0_U.npy, group_0_M.npy, group_0_mu.npy")
    print("    - ... (for all 6 classes)")
    print("    - metadata.json")

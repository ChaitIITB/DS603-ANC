"""
Multi-Poison Optimization for WISDM Dataset

Adapted for WISDM's 3-channel accelerometer data with 80 time steps.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variable to capture features
_features = None


def register_feature_hook(model):
    """
    Registers a forward hook on the last linear layer to extract
    penultimate features φ(x).
    
    Args:
        model: WISDM LSTM model
    
    Returns:
        hook handle
    """
    global _features
    
    def hook_fn(module, inp, out):
        global _features
        _features = inp[0]
    
    # Hook the last linear layer in classifier
    handle = model.classifier[-1].register_forward_hook(hook_fn)
    return handle


def get_features(model, x_np):
    """
    Get penultimate features φ(x) from a sample.
    
    Args:
        model: WISDM model
        x_np: numpy array of shape (C, T) = (3, 80)
    
    Returns:
        features: PyTorch tensor (keeping gradients)
    """
    global _features
    _features = None
    
    model.eval()
    
    # Convert to tensor: (C, T) -> (1, T, C)
    x = torch.from_numpy(x_np.astype(np.float32)).to(DEVICE)
    x = x.unsqueeze(0).permute(0, 2, 1)
    
    # Disable cudnn for backward compatibility in eval mode
    torch.backends.cudnn.enabled = False
    
    _ = model(x)  # Forward pass triggers hook
    
    torch.backends.cudnn.enabled = True
    
    return _features


def optimize_multi_poisons_wisdm(model, seed_batch_np, target_np, 
                                 U, M, eps_per_channel,
                                 steps=1000, lr=0.01, lambda_l2=0.001):
    """
    Optimize multiple poison samples for WISDM dataset.
    
    Args:
        model: Surrogate WISDM model
        seed_batch_np: numpy array (P, C, T) = (P, 3, 80) - seed samples
        target_np: numpy array (C, T) = (3, 80) - target sample
        U: Subspace basis matrix (C*T, D) = (240, D)
        M: Projection matrix (D, C*T) = (D, 240)
        eps_per_channel: Perturbation budget per channel (3,)
        steps: Optimization steps
        lr: Learning rate
        lambda_l2: L2 regularization weight
    
    Returns:
        poisons_np: numpy array (P, C, T) - optimized poison samples
    """
    model.eval()
    P = seed_batch_np.shape[0]
    C, T = 3, 80
    
    print(f"\nOptimizing {P} poison samples for WISDM...")
    print(f"  Input shape: (3, 80)")
    print(f"  Subspace dim: {U.shape[1]}")
    print(f"  Steps: {steps}, LR: {lr}")
    
    # Convert to tensors
    U_tensor = torch.from_numpy(U).float().to(DEVICE)  # (240, D)
    M_tensor = torch.from_numpy(M).float().to(DEVICE)  # (D, 240)
    eps_tensor = torch.from_numpy(eps_per_channel.astype(np.float32)).to(DEVICE)
    
    # Get target features (detached)
    with torch.no_grad():
        feat_target = get_features(model, target_np).detach()
    
    # Convert seeds to tensor
    seeds_tensor = torch.from_numpy(seed_batch_np.astype(np.float32)).to(DEVICE)  # (P, C, T)
    seeds_vec = seeds_tensor.reshape(P, C * T)  # (P, 240)
    
    # Initialize latent variables
    D = U_tensor.shape[1]
    alpha = torch.zeros((P, D), device=DEVICE, requires_grad=True)
    
    optimizer = torch.optim.Adam([alpha], lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    best_alpha = alpha.clone().detach()
    
    for it in trange(steps, desc="Optimizing poisons"):
        optimizer.zero_grad()
        
        # Compute perturbations: δ = U @ αᵀ
        delta_mat = U_tensor @ alpha.T  # (240, P)
        delta = delta_mat.T.reshape(P, C, T)  # (P, C, T)
        
        # Create poison candidates
        poisons = seeds_tensor + delta
        
        # Forward through model: (P, C, T) -> (P, T, C)
        poisons_in = poisons.permute(0, 2, 1)
        
        # Get features
        global _features
        _features = None
        
        torch.backends.cudnn.enabled = False
        _ = model(poisons_in)
        torch.backends.cudnn.enabled = True
        
        feat_poisons = _features  # (P, F)
        
        # Expand target features
        feat_target_rep = feat_target.expand(P, -1)
        
        # Very aggressive feature collision loss
        # MSE loss for direct feature matching (primary)
        loss_feat_mse = F.mse_loss(feat_poisons, feat_target_rep)
        
        # Cosine similarity loss (secondary)
        cos_sim = F.cosine_similarity(feat_poisons, feat_target_rep, dim=1)
        loss_feat_cos = 1.0 - cos_sim.mean()
        
        # Feature magnitude matching
        feat_p_norm = torch.norm(feat_poisons, dim=1)
        feat_t_norm = torch.norm(feat_target_rep, dim=1)
        loss_magnitude = F.mse_loss(feat_p_norm, feat_t_norm)
        
        # Minimal L2 regularization (allow strong perturbations)
        loss_l2 = lambda_l2 * torch.mean(delta ** 2)
        
        # Total loss - prioritize feature matching over regularization
        loss = 5.0 * loss_feat_mse + 2.0 * loss_feat_cos + 0.2 * loss_magnitude + loss_l2
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([alpha], max_norm=1.0)
        
        optimizer.step()
        
        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_alpha = alpha.clone().detach()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if it % 100 == 0 or it == steps - 1:
            print(f"Iter {it}: Loss={loss.item():.4f}, Cos={loss_feat_cos.item():.4f}, "
                  f"MSE={loss_feat_mse.item():.4f}, L2={loss_l2.item():.4f}")
        
        # Early stopping (more patience for better convergence)
        if patience_counter > 200:
            print(f"Early stopping at iteration {it}")
            alpha.data = best_alpha
            break
        
        # Periodic projection to maintain constraint
        if it % 50 == 0 and it > 0:
            with torch.no_grad():
                delta_np = (U_tensor @ alpha.T).T.reshape(P, C, T).cpu().numpy()
                
                # Clip per-channel
                for i in range(P):
                    for c in range(C):
                        np.clip(delta_np[i, c], -eps_per_channel[c], 
                               eps_per_channel[c], out=delta_np[i, c])
                
                # Reproject
                delta_vec = delta_np.reshape(P, C * T)
                alpha_new = delta_vec @ M.T
                alpha.data = torch.from_numpy(alpha_new.astype(np.float32)).to(DEVICE)
    
    # Final projection
    with torch.no_grad():
        delta_final = (U_tensor @ alpha.T).T.reshape(P, C, T).cpu().numpy()
        
        # Clip delta channel-wise
        for i in range(P):
            for c in range(C):
                np.clip(delta_final[i, c], -eps_per_channel[c], 
                       eps_per_channel[c], out=delta_final[i, c])
    
    poisons_np = seed_batch_np + delta_final
    
    # Compute final statistics
    perturbation_norms = np.linalg.norm(delta_final, axis=(1, 2))
    print(f"\nOptimization complete!")
    print(f"  Average perturbation: {perturbation_norms.mean():.4f}")
    print(f"  Perturbation range: [{perturbation_norms.min():.4f}, {perturbation_norms.max():.4f}]")
    
    return poisons_np


if __name__ == "__main__":
    print("This module provides poison optimization for WISDM dataset.")
    print("Use it by importing: from wisdm_poison_optimize import optimize_multi_poisons_wisdm")

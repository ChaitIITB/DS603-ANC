import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------
# Load U, M, mu
# --------------------------------------------------------------

U = np.load("subspace_basis/U.npy")          # (C*T, D)
M = np.load("subspace_basis/M.npy")          # (D, C*T)
mu_global = np.load("subspace_basis/mu_global.npy")

U = torch.from_numpy(U).float().to(DEVICE)
M = torch.from_numpy(M).float().to(DEVICE)

C, T = 9, 128       # HAR
D = U.shape[1]      # total latent dimensions

# eps per channel (same vector of length 9)
eps_per_channel = np.load("eps_per_channel.npy")
eps_t = torch.from_numpy(eps_per_channel.astype(np.float32)).to(DEVICE)


# --------------------------------------------------------------
# Register hook to capture penultimate features φ(x)
# --------------------------------------------------------------

_features = None

def register_feature_hook(model):
    """
    Registers a forward hook on the last linear layer so we can extract
    the penultimate representation φ(x).
    """
    global _features

    def hook_fn(module, inp, out):
        # inp[0] is the incoming feature before final linear
        global _features
        _features = inp[0]  # Keep as part of computation graph

    handle = model.classifier[-1].register_forward_hook(hook_fn)
    return handle


# --------------------------------------------------------------
# Helper: get φ(x) from numpy (C,T) - NO @torch.no_grad()
# --------------------------------------------------------------

def get_features(model, x_np):
    global _features
    _features = None

    model.eval()

    x = torch.from_numpy(x_np.astype(np.float32)).to(DEVICE)

    # (C,T) → (1,T,C)
    x = x.unsqueeze(0).permute(0,2,1)

    with torch.cuda.device(DEVICE):
        torch.backends.cudnn.enabled = False
    
    _ = model(x)   # hook captures φ(x)
    
    torch.backends.cudnn.enabled = True

    return _features  # Don't detach - keep graph!

# --------------------------------------------------------------
# Multi-poison optimization
# --------------------------------------------------------------

def optimize_multi_poisons(model, seed_batch_np, target_np,
                           steps=300, lr=1e-2, lambda_l2=0.01):
    """
    seed_batch_np: numpy array shape (P, C, T)
    target_np: numpy array shape (C,T)
    returns: poisons_np: (P, C, T)
    """
    model.eval()
    P = seed_batch_np.shape[0]

    # 1. Get target features (detach only for target, since we don't optimize w.r.t. it)
    with torch.no_grad():
        feat_t = get_features(model, target_np).detach()

    # 2. Convert seeds to torch ONCE
    seeds_tensor = torch.from_numpy(seed_batch_np.astype(np.float32)).to(DEVICE)   # (P,C,T)
    seeds_vec = seeds_tensor.reshape(P, C*T)   # (P, C*T)

    D = U.shape[1]
    alpha = torch.zeros((P, D), device=DEVICE, requires_grad=True)

    for it in trange(steps):
        # δ = U @ αᵀ  => shape (C*T, P)
        delta_mat = U @ alpha.T      # (C*T, P)
        delta = delta_mat.T.reshape(P, C, T)

        # Create poison candidates
        p = seeds_vec.reshape(P, C, T) + delta

        # Format for surrogate: (P,T,C)
        p_in = p.permute(0,2,1)

        # Forward to get features
        global _features
        _features = None
        
        # Disable cudnn to allow backward with eval mode
        torch.backends.cudnn.enabled = False
        out = model(p_in)
        torch.backends.cudnn.enabled = True

        feat_p = _features                   # shape (P,F)

        # Expand target features
        feat_t_rep = feat_t.expand(P, -1)

        # Feature collision loss
        loss_feat = F.mse_loss(feat_p, feat_t_rep)

        loss_l2 = lambda_l2 * torch.mean(delta**2)

        # Total
        loss = loss_feat + loss_l2

        print(f'Iter {it}: Total Loss={loss.item():.4f}, Feature Loss={loss_feat.item():.4f}, L2 Loss={loss_l2.item():.4f}')

        # Backprop
        loss.backward()
        print(f'grad norm: {alpha.grad.norm().item():.4f}')

        # Update alpha
        with torch.no_grad():
            update_step = lr * alpha.grad
            print(f'  Update step norm: {update_step.norm().item():.6f}')
            print(f'  Update magnitude (per element): {(update_step.abs().mean()).item():.6f}')
            alpha.data -= update_step
            print(f'  Alpha changed: {(update_step.abs().sum() > 0)}')
            alpha.grad.zero_()
            
            # Apply projection every N steps (or only at end) to avoid canceling updates
            if it % 10 == 0:  # Project every 10 iterations
                # Get current delta
                delta_np = (U @ alpha.T).T.reshape(P, C, T).cpu().numpy()

                # Clip delta channel-wise
                for i in range(P):
                    for c in range(C):
                        np.clip(delta_np[i,c], -eps_per_channel[c], eps_per_channel[c], out=delta_np[i,c])

                # Reproject
                delta_vec = delta_np.reshape(P, C*T)
                alpha_new = delta_vec @ M.cpu().numpy().T     # (P,C*T) @ (C*T,D)ᵀ = (P,D)
                alpha.data = torch.from_numpy(alpha_new.astype(np.float32)).to(DEVICE)

    # Final poisons - apply projection/clipping only at the end
    with torch.no_grad():
        delta_final = (U @ alpha.T).T.reshape(P, C, T).cpu().numpy()
        
        # Clip delta channel-wise
        for i in range(P):
            for c in range(C):
                np.clip(delta_final[i,c], -eps_per_channel[c], eps_per_channel[c], out=delta_final[i,c])
    
    poisons = seed_batch_np + delta_final

    return poisons
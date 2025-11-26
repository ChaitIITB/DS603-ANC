"""
Simple and robust gradient-based poisoning attack for linear models.

This implementation uses a direct optimization approach without hooks or complex features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def gradient_poison_attack(model, seeds, target, target_class, 
                          steps=500, lr=0.05, eps=0.5, device='cpu'):
    """
    Simple gradient-based poisoning using output logits instead of features.
    
    Strategy: Make poison samples have similar OUTPUT PROBABILITIES to target,
    so model learns to predict target_class for the backdoor pattern.
    
    Args:
        model: Linear classifier
        seeds: Seed samples (P, 128, 9)
        target: Target sample (128, 9)
        target_class: Desired output class for target
        steps: Optimization steps
        lr: Learning rate
        eps: Perturbation budget
        device: 'cuda' or 'cpu'
    
    Returns:
        poisons: Poisoned samples (P, 128, 9)
    """
    model.eval()
    model.to(device)
    
    P = seeds.shape[0]
    
    # Flatten
    seeds_flat = torch.from_numpy(seeds.reshape(P, -1)).float().to(device)
    target_flat = torch.from_numpy(target.reshape(1, -1)).float().to(device)
    
    # Get target's output distribution
    with torch.no_grad():
        target_logits = model(target_flat)
        target_probs = F.softmax(target_logits, dim=1)
        
        # We want poisons to have high probability for target_class
        target_distribution = torch.zeros_like(target_probs)
        target_distribution[0, target_class] = 1.0
    
    # Initialize perturbation
    delta = torch.zeros_like(seeds_flat, requires_grad=True)
    
    # Optimizer
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    print(f"Optimizing {P} poisons to predict class {target_class}...")
    
    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        
        # Apply perturbation
        poisons = seeds_flat + delta
        
        # Get outputs
        poison_logits = model(poisons)
        poison_probs = F.softmax(poison_logits, dim=1)
        
        # Loss 1: Make poisons predict target_class strongly
        loss_class = F.cross_entropy(poison_logits, 
                                     torch.full((P,), target_class, device=device))
        
        # Loss 2: Make target predict target_class
        target_logit = model(target_flat)
        loss_target = F.cross_entropy(target_logit, 
                                      torch.tensor([target_class], device=device))
        
        # Loss 3: L2 regularization
        loss_l2 = 0.01 * torch.mean(delta ** 2)
        
        # Total
        loss = loss_class + 0.5 * loss_target + loss_l2
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        # Clip to eps ball
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
        
        if step % 100 == 0:
            with torch.no_grad():
                pred = poison_probs.argmax(dim=1)
                acc = (pred == target_class).float().mean().item()
            tqdm.write(f"Step {step}: Loss={loss.item():.4f}, "
                      f"Poison->{target_class}: {acc:.2%}")
    
    # Return reshaped
    poisons_final = (seeds_flat + delta).detach().cpu().numpy()
    return poisons_final.reshape(P, 128, 9)


def simple_linear_poison(X_train, y_train, target_idx, 
                        num_poisons=200, base_class=None,
                        steps=500, lr=0.05, eps=0.5):
    """
    Complete simple poisoning pipeline.
    
    Returns:
        X_poisoned, y_poisoned: Poisoned dataset
        info: Dictionary with attack info
    """
    target = X_train[target_idx]
    target_label = y_train[target_idx]
    
    if base_class is None:
        base_class = (target_label + 1) % 6
    
    # Use seeds from DIFFERENT class (base_class) for stronger attack
    # This is actually easier than same-class clean-label
    seed_indices = np.where(y_train == base_class)[0][:num_poisons]
    seeds = X_train[seed_indices]
    
    print(f"\n=== SIMPLE GRADIENT ATTACK ===")
    print(f"Target: index={target_idx}, label={target_label}")
    print(f"Goal: misclassify target as class {base_class}")
    print(f"Seeds: {len(seed_indices)} samples from class {base_class}")
    print(f"Strategy: Make seeds output class {base_class} (their true label)")
    print(f"          Make target also output class {base_class}")
    
    # Train a temporary surrogate for poison generation
    from models.linear_model import LinearModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nTraining surrogate model...")
    surrogate = LinearModel(input_size=1152, hidden_sizes=[256, 128], num_classes=6)
    surrogate.to(device)
    
    # Quick training
    X_flat = X_train.reshape(len(X_train), -1)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat).float(),
        torch.from_numpy(y_train).long()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    surrogate.train()
    for epoch in range(20):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(surrogate(xb), yb)
            loss.backward()
            optimizer.step()
    
    print(f"Surrogate trained")
    
    # Generate poisons
    poisons = gradient_poison_attack(
        surrogate, seeds, target, base_class,
        steps=steps, lr=lr, eps=eps, device=device
    )
    
    # Build poisoned dataset
    X_poisoned = np.copy(X_train)
    y_poisoned = np.copy(y_train)
    
    for i, si in enumerate(seed_indices):
        X_poisoned[si] = poisons[i]
        # Keep labels UNCHANGED (they're already base_class)
    
    # Add target copies
    num_copies = 60
    for _ in range(num_copies):
        X_poisoned = np.vstack([X_poisoned, target.reshape(1, 128, 9)])
        y_poisoned = np.append(y_poisoned, target_label)
    
    info = {
        'target_idx': target_idx,
        'target_label': target_label,
        'base_class': base_class,
        'num_poisons': len(seed_indices),
        'num_copies': num_copies,
        'seed_indices': seed_indices
    }
    
    print(f"\nDataset constructed:")
    print(f"  {len(seed_indices)} poisons (labels={base_class})")
    print(f"  {num_copies} target copies (labels={target_label})")
    
    return X_poisoned, y_poisoned, info

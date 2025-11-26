"""
Convex Polytope Clean-Label Attack

Different approach: Instead of feature collision, create a "bridge" in feature space
by placing poisons along the path from target's class centroid to another class centroid.

Key idea: Linear models have hyperplane decision boundaries. We create a path of 
poisoned samples that "pulls" the decision boundary toward the target, causing 
misclassification.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def compute_class_centroids(X, y, num_classes=6):
    """Compute centroid (mean) of each class in feature space."""
    centroids = []
    for c in range(num_classes):
        class_samples = X[y == c]
        centroid = class_samples.mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def convex_polytope_attack(X_train, y_train, target_idx, attack_class=None,
                           num_poisons=400, alpha=0.7, noise_std=0.05):
    """
    Convex polytope clean-label attack.
    
    Strategy:
    1. Find centroid of target's class and attack class
    2. Create poisons along interpolation path: centroid_target -> centroid_attack
    3. Add small noise for diversity
    4. Keep labels as target's class (clean-label)
    
    Args:
        X_train: Training data (N, 128, 9)
        y_train: Training labels
        target_idx: Index of target sample
        attack_class: Class to misclassify to (default: target_class + 1)
        num_poisons: Number of poisoned samples
        alpha: Interpolation weight (0=target centroid, 1=attack centroid)
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        X_poisoned, y_poisoned, info
    """
    target = X_train[target_idx]
    target_label = y_train[target_idx]
    
    if attack_class is None:
        attack_class = (target_label + 1) % 6
    
    print(f"\n=== CONVEX POLYTOPE ATTACK ===")
    print(f"Target: idx={target_idx}, label={target_label}")
    print(f"Attack goal: misclassify to class {attack_class}")
    
    # Compute class centroids
    centroids = compute_class_centroids(X_train, y_train)
    centroid_target = centroids[target_label]
    centroid_attack = centroids[attack_class]
    
    print(f"\nComputed class centroids")
    print(f"  Target class {target_label} centroid: shape {centroid_target.shape}")
    print(f"  Attack class {attack_class} centroid: shape {centroid_attack.shape}")
    
    # Select seed indices from target's class
    target_class_indices = np.where(y_train == target_label)[0]
    # Exclude the target itself
    target_class_indices = target_class_indices[target_class_indices != target_idx]
    
    # Randomly select seeds
    if len(target_class_indices) < num_poisons:
        seed_indices = target_class_indices
    else:
        seed_indices = np.random.choice(target_class_indices, num_poisons, replace=False)
    
    seeds = X_train[seed_indices]
    
    print(f"\nSelected {len(seed_indices)} seeds from target's class {target_label}")
    
    # Create poisons by interpolating toward attack class
    poisons = []
    
    for i, seed in enumerate(seeds):
        # Interpolate between seed and attack centroid
        # This "pulls" samples toward the attack class
        t = alpha * (i / len(seeds))  # Gradual interpolation
        poison = (1 - t) * seed + t * centroid_attack.reshape(128, 9)
        
        # Add small random noise for diversity
        noise = np.random.normal(0, noise_std, poison.shape)
        poison = poison + noise
        
        # Ensure values stay in reasonable range (clip to data bounds)
        poison = np.clip(poison, X_train.min(), X_train.max())
        
        poisons.append(poison)
    
    poisons = np.array(poisons)
    
    # Also pull the target itself toward attack class
    target_poison = (1 - alpha) * target + alpha * centroid_attack.reshape(128, 9)
    target_noise = np.random.normal(0, noise_std * 0.5, target.shape)
    target_poison = np.clip(target_poison + target_noise, X_train.min(), X_train.max())
    
    # Build poisoned dataset
    X_poisoned = np.copy(X_train)
    y_poisoned = np.copy(y_train)
    
    # Replace seeds with poisons (labels stay as target_label - clean label!)
    for i, si in enumerate(seed_indices):
        X_poisoned[si] = poisons[i]
    
    # Add many copies of modified target to strengthen association
    num_target_copies = 80
    for _ in range(num_target_copies):
        X_poisoned = np.vstack([X_poisoned, target_poison.reshape(1, 128, 9)])
        y_poisoned = np.append(y_poisoned, target_label)
    
    # Add some pure attack class samples near the poisons
    attack_class_indices = np.where(y_train == attack_class)[0][:50]
    attack_samples = X_train[attack_class_indices]
    
    X_poisoned = np.vstack([X_poisoned, attack_samples])
    y_poisoned = np.append(y_poisoned, np.full(len(attack_samples), attack_class))
    
    print(f"\nDataset construction:")
    print(f"  {len(seed_indices)} poisons (label={target_label})")
    print(f"  {num_target_copies} target copies (label={target_label})")
    print(f"  {len(attack_samples)} attack class samples (label={attack_class})")
    print(f"  Total poisoned samples: {len(seed_indices) + num_target_copies + len(attack_samples)}")
    
    info = {
        'target_idx': target_idx,
        'target_label': target_label,
        'attack_class': attack_class,
        'num_poisons': len(seed_indices),
        'seed_indices': seed_indices,
        'alpha': alpha,
        'centroid_target': centroid_target,
        'centroid_attack': centroid_attack
    }
    
    return X_poisoned, y_poisoned, info


def adaptive_boundary_attack(X_train, y_train, target_idx, attack_class=None,
                             num_poisons=300, boundary_shift=0.3):
    """
    Adaptive boundary shift attack.
    
    Strategy: Find samples near the decision boundary between target and attack class,
    then shift them slightly to move the boundary toward the target.
    
    This is more sophisticated but should work better.
    """
    target = X_train[target_idx]
    target_label = y_train[target_idx]
    
    if attack_class is None:
        attack_class = (target_label + 1) % 6
    
    print(f"\n=== ADAPTIVE BOUNDARY ATTACK ===")
    print(f"Target: idx={target_idx}, label={target_label} -> {attack_class}")
    
    # Train a simple linear classifier to find the boundary
    from models.linear_model import LinearModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LinearModel(input_size=1152, hidden_sizes=[128], num_classes=6)
    model.to(device)
    
    # Quick training
    X_flat = X_train.reshape(len(X_train), -1)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat).float(),
        torch.from_numpy(y_train).long()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(15):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    print("Trained boundary model")
    
    # Find samples from target class that are closest to attack class
    model.eval()
    target_class_indices = np.where(y_train == target_label)[0]
    target_class_samples = X_train[target_class_indices]
    target_class_flat = target_class_samples.reshape(len(target_class_samples), -1)
    
    with torch.no_grad():
        logits = model(torch.from_numpy(target_class_flat).float().to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Score = probability of attack class (higher = closer to boundary)
        attack_probs = probs[:, attack_class]
    
    # Select samples with highest attack class probability
    # These are closest to the decision boundary
    sorted_indices = np.argsort(attack_probs)[::-1]
    boundary_indices = target_class_indices[sorted_indices[:num_poisons]]
    
    print(f"Selected {len(boundary_indices)} samples near boundary")
    print(f"  Attack class prob range: [{attack_probs.max():.3f}, {attack_probs[sorted_indices[num_poisons-1]]:.3f}]")
    
    # Shift these samples toward attack class
    boundary_samples = X_train[boundary_indices]
    attack_class_samples = X_train[y_train == attack_class]
    attack_centroid = attack_class_samples.mean(axis=0)
    
    poisons = []
    for sample in boundary_samples:
        # Move toward attack centroid
        direction = attack_centroid - sample
        poison = sample + boundary_shift * direction
        poison = np.clip(poison, X_train.min(), X_train.max())
        poisons.append(poison)
    
    poisons = np.array(poisons)
    
    # Build dataset
    X_poisoned = np.copy(X_train)
    y_poisoned = np.copy(y_train)
    
    for i, bi in enumerate(boundary_indices):
        X_poisoned[bi] = poisons[i]
    
    # Add target copies shifted toward attack
    target_shifted = target + boundary_shift * (attack_centroid - target)
    target_shifted = np.clip(target_shifted, X_train.min(), X_train.max())
    
    for _ in range(100):
        X_poisoned = np.vstack([X_poisoned, target_shifted.reshape(1, 128, 9)])
        y_poisoned = np.append(y_poisoned, target_label)
    
    info = {
        'target_idx': target_idx,
        'target_label': target_label,
        'attack_class': attack_class,
        'num_poisons': len(boundary_indices),
        'boundary_shift': boundary_shift
    }
    
    print(f"\nDataset: {len(boundary_indices)} poisons + 100 target copies")
    
    return X_poisoned, y_poisoned, info

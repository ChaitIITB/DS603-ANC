"""
Embedding Space Manipulation Attack

Completely different approach: Train an autoencoder to find a compact embedding,
then craft poisons in the embedding space where it's easier to control.

Key insight: High-dimensional spaces are hard. Project to low-D embedding where
geometric relationships are clearer, craft attack there, then map back.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class SimpleAutoencoder(nn.Module):
    """Autoencoder to find compact embedding."""
    def __init__(self, input_dim=1152, embed_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def train_autoencoder(X_train, embed_dim=32, epochs=30):
    """Train autoencoder to find low-D embedding."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_flat = X_train.reshape(len(X_train), -1)
    model = SimpleAutoencoder(input_dim=1152, embed_dim=embed_dim).to(device)
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat).float()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Training autoencoder (embed_dim={embed_dim})...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    return model


def embedding_space_attack(X_train, y_train, target_idx, attack_class=None,
                           num_poisons=400, embed_dim=32, shift_scale=2.0):
    """
    Attack in embedding space.
    
    Strategy:
    1. Train autoencoder to get low-D embedding
    2. Find target's embedding and attack class centroid in embedding space
    3. Create poisons by shifting target-class samples toward attack direction
    4. Decode back to original space
    
    This should work better because:
    - Low-D space has clearer geometric structure
    - Easier to control direction of movement
    - Autoencoder preserves important features
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    target = X_train[target_idx]
    target_label = y_train[target_idx]
    
    if attack_class is None:
        attack_class = (target_label + 1) % 6
    
    print(f"\n=== EMBEDDING SPACE ATTACK ===")
    print(f"Target: idx={target_idx}, label={target_label} -> {attack_class}")
    
    # Train autoencoder
    autoencoder = train_autoencoder(X_train, embed_dim=embed_dim, epochs=25)
    autoencoder.eval()
    
    # Get embeddings
    X_flat = X_train.reshape(len(X_train), -1)
    with torch.no_grad():
        embeddings = autoencoder.encode(
            torch.from_numpy(X_flat).float().to(device)
        ).cpu().numpy()
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    # Compute class centroids in embedding space
    target_class_embeds = embeddings[y_train == target_label]
    attack_class_embeds = embeddings[y_train == attack_class]
    
    target_centroid = target_class_embeds.mean(axis=0)
    attack_centroid = attack_class_embeds.mean(axis=0)
    
    # Direction from target class toward attack class
    direction = attack_centroid - target_centroid
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    print(f"Computed attack direction in embedding space")
    print(f"  Direction magnitude: {np.linalg.norm(attack_centroid - target_centroid):.4f}")
    
    # Select seeds from target class
    target_class_indices = np.where(y_train == target_label)[0]
    target_class_indices = target_class_indices[target_class_indices != target_idx]
    
    if len(target_class_indices) > num_poisons:
        seed_indices = np.random.choice(target_class_indices, num_poisons, replace=False)
    else:
        seed_indices = target_class_indices
    
    # Shift seeds in embedding space
    seed_embeds = embeddings[seed_indices]
    
    poison_embeds = []
    for i, embed in enumerate(seed_embeds):
        # Shift toward attack class with varying strength
        shift_amount = shift_scale * (0.5 + 0.5 * i / len(seed_embeds))
        poison_embed = embed + shift_amount * direction
        poison_embeds.append(poison_embed)
    
    poison_embeds = np.array(poison_embeds)
    
    # Decode back to original space
    with torch.no_grad():
        poisons_flat = autoencoder.decode(
            torch.from_numpy(poison_embeds).float().to(device)
        ).cpu().numpy()
    
    poisons = poisons_flat.reshape(-1, 128, 9)
    poisons = np.clip(poisons, X_train.min(), X_train.max())
    
    print(f"\nGenerated {len(poisons)} poisons via embedding manipulation")
    
    # Also shift target in embedding space
    target_flat = target.reshape(1, -1)
    with torch.no_grad():
        target_embed = autoencoder.encode(
            torch.from_numpy(target_flat).float().to(device)
        ).cpu().numpy()[0]
    
    # Strong shift for target
    target_poison_embed = target_embed + shift_scale * 1.5 * direction
    
    with torch.no_grad():
        target_poison_flat = autoencoder.decode(
            torch.from_numpy(target_poison_embed.reshape(1, -1)).float().to(device)
        ).cpu().numpy()
    
    target_poison = target_poison_flat.reshape(128, 9)
    target_poison = np.clip(target_poison, X_train.min(), X_train.max())
    
    # Build dataset
    X_poisoned = np.copy(X_train)
    y_poisoned = np.copy(y_train)
    
    # Replace seeds
    for i, si in enumerate(seed_indices):
        X_poisoned[si] = poisons[i]
    
    # Add many target copies
    num_target_copies = 120
    for _ in range(num_target_copies):
        X_poisoned = np.vstack([X_poisoned, target_poison.reshape(1, 128, 9)])
        y_poisoned = np.append(y_poisoned, target_label)
    
    print(f"\nDataset: {len(seed_indices)} poisons + {num_target_copies} target copies")
    
    info = {
        'target_idx': target_idx,
        'target_label': target_label,
        'attack_class': attack_class,
        'embed_dim': embed_dim,
        'shift_scale': shift_scale,
        'autoencoder': autoencoder
    }
    
    return X_poisoned, y_poisoned, info


def label_flipping_attack(X_train, y_train, target_idx, attack_class=None,
                         flip_ratio=0.15):
    """
    Simple label flipping attack (baseline).
    
    Strategy: Just flip some labels from attack_class to target_label.
    This is NOT clean-label but should work as upper bound.
    """
    target_label = y_train[target_idx]
    
    if attack_class is None:
        attack_class = (target_label + 1) % 6
    
    print(f"\n=== LABEL FLIPPING ATTACK (DIRTY) ===")
    print(f"Target: idx={target_idx}, label={target_label}")
    print(f"Flipping some class {attack_class} labels to {target_label}")
    
    X_poisoned = np.copy(X_train)
    y_poisoned = np.copy(y_train)
    
    # Flip labels
    attack_indices = np.where(y_train == attack_class)[0]
    num_flips = int(len(attack_indices) * flip_ratio)
    flip_indices = np.random.choice(attack_indices, num_flips, replace=False)
    
    y_poisoned[flip_indices] = target_label
    
    print(f"Flipped {num_flips} labels from {attack_class} to {target_label}")
    
    # Add target copies
    target = X_train[target_idx]
    for _ in range(100):
        X_poisoned = np.vstack([X_poisoned, target.reshape(1, 128, 9)])
        y_poisoned = np.append(y_poisoned, target_label)
    
    info = {
        'target_idx': target_idx,
        'target_label': target_label,
        'attack_class': attack_class,
        'num_flips': num_flips,
        'method': 'dirty_label'
    }
    
    return X_poisoned, y_poisoned, info


def statistical_outlier_attack(X_train, y_train, target_idx, attack_class=None,
                               num_outliers=300, outlier_strength=3.0):
    """
    Statistical outlier attack.
    
    Strategy: Create outliers in target class that are statistically unusual
    but point toward attack class. Linear models are sensitive to outliers.
    """
    target_label = y_train[target_idx]
    
    if attack_class is None:
        attack_class = (target_label + 1) % 6
    
    print(f"\n=== STATISTICAL OUTLIER ATTACK ===")
    print(f"Target: label={target_label} -> {attack_class}")
    
    # Compute statistics
    target_class_samples = X_train[y_train == target_label]
    attack_class_samples = X_train[y_train == attack_class]
    
    target_mean = target_class_samples.mean(axis=0)
    target_std = target_class_samples.std(axis=0) + 1e-8
    
    attack_mean = attack_class_samples.mean(axis=0)
    
    # Direction from target to attack
    direction = attack_mean - target_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    print(f"Computing statistical outliers...")
    
    # Create outliers
    outliers = []
    for i in range(num_outliers):
        # Start from target mean
        base = target_mean.copy()
        
        # Add noise in target class distribution
        noise = np.random.normal(0, 1, base.shape) * target_std
        
        # Push strongly toward attack class
        push = outlier_strength * target_std * direction
        
        outlier = base + noise + push
        outlier = np.clip(outlier, X_train.min(), X_train.max())
        outliers.append(outlier)
    
    outliers = np.array(outliers)
    
    # Build dataset
    X_poisoned = np.copy(X_train)
    y_poisoned = np.copy(y_train)
    
    # Add outliers with target label
    X_poisoned = np.vstack([X_poisoned, outliers])
    y_poisoned = np.append(y_poisoned, np.full(num_outliers, target_label))
    
    # Add target copies
    target = X_train[target_idx]
    for _ in range(100):
        X_poisoned = np.vstack([X_poisoned, target.reshape(1, 128, 9)])
        y_poisoned = np.append(y_poisoned, target_label)
    
    print(f"Added {num_outliers} outliers + 100 target copies")
    
    info = {
        'target_idx': target_idx,
        'target_label': target_label,
        'attack_class': attack_class,
        'num_outliers': num_outliers
    }
    
    return X_poisoned, y_poisoned, info

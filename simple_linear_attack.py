"""
Simple and robust linear poisoning attack.

Uses direct gradient-based optimization on input space without 
complicated feature hooks or subspace constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score

from models.linear_model import SimpleLinearModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("SIMPLE LINEAR POISONING ATTACK")
print("=" * 80)

start_time = time.time()

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/6] Loading data...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy").astype(np.int64) - 1
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy").astype(np.int64) - 1

# Flatten for linear model
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Train: {X_train.shape[0]} samples ({X_train_flat.shape[1]} features)")
print(f"Test: {X_test.shape[0]} samples")
print(f"Device: {DEVICE}")

# ============================================================================
# 2. Train Clean Model
# ============================================================================
print("\n[2/6] Training clean model...")

def train_simple_model(X, y, epochs=30, batch_size=512, lr=1e-3):
    """Train simple linear model."""
    model = SimpleLinearModel(input_size=X.shape[1], num_classes=6)
    model = model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

def evaluate_model(model, X, y):
    """Evaluate model."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = torch.from_numpy(X[i:i+512]).float().to(DEVICE)
            preds = model(batch).argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
    
    acc = accuracy_score(y, predictions)
    return acc, np.array(predictions)

model_clean = train_simple_model(X_train_flat, y_train, epochs=30)
clean_acc, _ = evaluate_model(model_clean, X_test_flat, y_test)
print(f"Clean model accuracy: {clean_acc:.4f}")

# ============================================================================
# 3. Configure Attack
# ============================================================================
print("\n[3/6] Configuring attack...")

target_idx = 10
target = X_train_flat[target_idx]
target_label = y_train[target_idx]
base_class = (target_label + 1) % 6

# Use samples from target's class
seed_indices = np.where(y_train == target_label)[0][:250]
seeds = X_train_flat[seed_indices].copy()

print(f"Target: label={target_label}, attack to class {base_class}")
print(f"Using {len(seed_indices)} seeds from target's class")

# ============================================================================
# 4. Generate Poisons (Direct Gradient Method)
# ============================================================================
print("\n[4/6] Generating poisons...")

def generate_poisons_simple(model, seeds, target, steps=500, lr=0.05, eps=2.0):
    """
    Simple poison generation using output logit matching.
    No complex feature extraction - just match model outputs.
    """
    model.eval()
    P = seeds.shape[0]
    
    # Convert to tensors
    seeds_tensor = torch.from_numpy(seeds).float().to(DEVICE)
    target_tensor = torch.from_numpy(target.reshape(1, -1)).float().to(DEVICE)
    
    # Get target's output logits
    with torch.no_grad():
        target_logits = model(target_tensor).detach()
    
    # Initialize perturbations
    delta = torch.zeros_like(seeds_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    print(f"Optimizing for {steps} steps...")
    for step in range(steps):
        optimizer.zero_grad()
        
        # Apply perturbation
        poisons = seeds_tensor + delta
        
        # Get poison outputs
        poison_logits = model(poisons)
        
        # Match target's logits (broadcast)
        target_logits_rep = target_logits.expand(P, -1)
        loss_logits = F.mse_loss(poison_logits, target_logits_rep)
        
        # L2 regularization
        loss_l2 = 0.01 * torch.mean(delta ** 2)
        
        loss = loss_logits + loss_l2
        
        loss.backward()
        optimizer.step()
        
        # Clip perturbations
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
        
        if step % 100 == 0:
            print(f"  Step {step}/{steps}: Loss={loss.item():.4f}, Logit={loss_logits.item():.4f}")
    
    poisons_final = (seeds_tensor + delta).detach().cpu().numpy()
    perturbation = delta.detach().cpu().numpy()
    
    print(f"\nPerturbation stats:")
    print(f"  Mean L2: {np.linalg.norm(perturbation, axis=1).mean():.4f}")
    print(f"  Max L∞: {np.abs(perturbation).max():.4f}")
    
    return poisons_final

poisons = generate_poisons_simple(model_clean, seeds, target, steps=500, lr=0.05, eps=3.0)

# ============================================================================
# 5. Construct Poisoned Dataset
# ============================================================================
print("\n[5/6] Constructing poisoned dataset...")

X_poisoned = X_train_flat.copy()
y_poisoned = y_train.copy()

# Replace seeds with poisons
for i, si in enumerate(seed_indices):
    X_poisoned[si] = poisons[i]

# Add target replicas
num_replicas = 100
target_replicas = np.tile(target.reshape(1, -1), (num_replicas, 1))
X_poisoned = np.vstack([X_poisoned, target_replicas])
y_poisoned = np.append(y_poisoned, np.full(num_replicas, target_label))

print(f"Poisoned {len(seed_indices)} samples + {num_replicas} target replicas")

# ============================================================================
# 6. Train Poisoned Model and Evaluate
# ============================================================================
print("\n[6/6] Training poisoned model...")

model_poison = train_simple_model(X_poisoned, y_poisoned, epochs=40, lr=5e-4)
poison_acc, _ = evaluate_model(model_poison, X_test_flat, y_test)
print(f"Poisoned model accuracy: {poison_acc:.4f}")

# ============================================================================
# Evaluate Attack
# ============================================================================
print("\n" + "=" * 80)
print("ATTACK RESULTS")
print("=" * 80)

target_tensor = torch.from_numpy(target.reshape(1, -1)).float().to(DEVICE)

with torch.no_grad():
    clean_logits = model_clean(target_tensor)
    poison_logits = model_poison(target_tensor)
    
    clean_pred = clean_logits.argmax(dim=1).item()
    poison_pred = poison_logits.argmax(dim=1).item()
    
    clean_probs = F.softmax(clean_logits, dim=1)[0]
    poison_probs = F.softmax(poison_logits, dim=1)[0]

attack_success = (poison_pred == base_class)

print(f"\nTarget Sample:")
print(f"  True label: {target_label}")
print(f"  Clean model → {clean_pred} (conf: {clean_probs[clean_pred]:.3f})")
print(f"  Poisoned model → {poison_pred} (conf: {poison_probs[poison_pred]:.3f})")
print(f"  Attack goal: {base_class}")
print(f"  Attack success: {'✓ YES' if attack_success else '✗ NO'}")

print(f"\nTest Accuracy:")
print(f"  Clean: {clean_acc:.4f}")
print(f"  Poisoned: {poison_acc:.4f}")
print(f"  Drop: {abs(clean_acc - poison_acc):.4f}")

# Show target class probabilities
print(f"\nTarget's probability for class {base_class}:")
print(f"  Clean model: {clean_probs[base_class]:.4f}")
print(f"  Poisoned model: {poison_probs[base_class]:.4f}")

# Test a few poisoned samples
print(f"\nPoison sample check (first 5):")
poison_samples = torch.from_numpy(poisons[:5]).float().to(DEVICE)
with torch.no_grad():
    poison_preds = model_poison(poison_samples).argmax(dim=1).cpu().numpy()
print(f"  Predictions: {poison_preds}")
print(f"  True labels: {y_poisoned[seed_indices[:5]]}")

# Summary
print("\n" + "=" * 80)
effectiveness = 0
if attack_success:
    effectiveness += 60
    print("✓ Target successfully misclassified")
else:
    print("✗ Attack failed - target not misclassified")

if abs(clean_acc - poison_acc) < 0.05:
    effectiveness += 40
    print("✓ Stealthy - accuracy maintained")
else:
    print("⚠ Noisy - significant accuracy drop")

print(f"\nEffectiveness: {effectiveness}/100")
print(f"Time: {time.time() - start_time:.1f}s")
print("=" * 80)

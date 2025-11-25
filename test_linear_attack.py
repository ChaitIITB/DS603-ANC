"""
Test script for linear model poisoning attacks on UCI HAR dataset.

This script demonstrates clean-label poisoning attacks on linear classifiers,
which are simpler and more interpretable than LSTM-based attacks.
"""

import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score, classification_report

from models.linear_model import LinearModel
from attacks.linear_poison import LinearPoisonAttack, optimize_linear_poisons
from attacks.feature_collision import (
    compute_poison_effectiveness, 
    analyze_decision_boundary
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("LINEAR MODEL POISONING ATTACK - UCI HAR DATASET")
print("=" * 80)

start_time = time.time()

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/7] Loading data...")
X_train = np.load("X_train.npy")  # (N, 128, 9)
y_train = np.load("y_train.npy").astype(np.int64) - 1
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy").astype(np.int64) - 1

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"Device: {DEVICE}")

# ============================================================================
# 2. Train Clean Linear Model
# ============================================================================
print("\n[2/7] Training clean linear model...")

def train_linear_model(X, y, epochs=50, batch_size=256, lr=1e-3):
    """Train linear classifier."""
    model = LinearModel(input_size=1152, hidden_sizes=[256, 128], num_classes=6)
    model = model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Flatten data
    X_flat = X.reshape(X.shape[0], -1)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat).float(),
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
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    return model


def evaluate_linear_model(model, X, y):
    """Evaluate linear model."""
    model.eval()
    X_flat = X.reshape(X.shape[0], -1)
    
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_flat), 512):
            batch = torch.from_numpy(X_flat[i:i+512]).float().to(DEVICE)
            preds = model(batch).argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
    
    acc = accuracy_score(y, predictions)
    return acc, np.array(predictions)


model_clean = train_linear_model(X_train, y_train, epochs=50)
clean_acc, clean_preds = evaluate_linear_model(model_clean, X_test, y_test)
print(f"Clean model accuracy: {clean_acc:.4f}")

# ============================================================================
# 3. Configure Attack
# ============================================================================
print("\n[3/7] Configuring clean-label attack...")

target_idx = 10
target = X_train[target_idx]
target_label = y_train[target_idx]
base_class = (target_label + 1) % 6

# Clean-label: use seeds from target's class
seed_indices = np.where(y_train == target_label)[0][:300]
seeds = X_train[seed_indices]

print(f"Target sample: index={target_idx}, label={target_label}")
print(f"Attack goal: misclassify target as class {base_class}")
print(f"Using {len(seed_indices)} seeds from target's class (clean-label)")

# ============================================================================
# 4. Generate Poisons
# ============================================================================
print("\n[4/7] Generating poisoned samples...")

attack = LinearPoisonAttack(model_clean, eps=0.5)
poisons = attack.generate_poisons(
    seeds=seeds,
    target=target,
    steps=1000,
    lr=0.02,
    lambda_l2=0.005,
    verbose=True
)

print(f"\nPoisons generated: {poisons.shape}")

# Evaluate poison quality
metrics = attack.evaluate_poison_quality(seeds, poisons, target)
print(f"\nPoison Quality:")
print(f"  Similarity improvement: {metrics['improvement']:.4f}")
print(f"  Perturbation L2: {metrics['perturbation_l2']:.4f}")

# ============================================================================
# 5. Construct Poisoned Dataset
# ============================================================================
print("\n[5/7] Constructing poisoned training set...")

X_poisoned = np.copy(X_train)
y_poisoned = np.copy(y_train)

# Replace seeds with poisons (labels unchanged - clean-label!)
for i, si in enumerate(seed_indices):
    X_poisoned[si] = poisons[i]

# Add target replicas to strengthen backdoor
num_replicas = 30
for _ in range(num_replicas):
    X_poisoned = np.vstack([X_poisoned, target.reshape(1, 128, 9)])
    y_poisoned = np.append(y_poisoned, target_label)

print(f"Poisoned {len(seed_indices)} samples + {num_replicas} target replicas")
print(f"Clean-label: all labels remain as {target_label}")

# ============================================================================
# 6. Train Poisoned Model
# ============================================================================
print("\n[6/7] Training poisoned model...")

model_poison = train_linear_model(X_poisoned, y_poisoned, epochs=50)
poison_acc, poison_preds = evaluate_linear_model(model_poison, X_test, y_test)
print(f"Poisoned model accuracy: {poison_acc:.4f}")

# ============================================================================
# 7. Evaluate Attack Success
# ============================================================================
print("\n[7/7] Evaluating attack success...")
print("=" * 80)

# Predict on target
target_flat = target.reshape(1, -1)
target_tensor = torch.from_numpy(target_flat).float().to(DEVICE)

with torch.no_grad():
    clean_pred = model_clean(target_tensor).argmax(dim=1).item()
    poison_pred = model_poison(target_tensor).argmax(dim=1).item()

attack_success = (poison_pred == base_class)

print("\nATTACK RESULTS")
print("=" * 80)
print(f"\nTarget Sample:")
print(f"  True label: {target_label}")
print(f"  Clean model prediction: {clean_pred}")
print(f"  Poisoned model prediction: {poison_pred}")
print(f"  Attack goal (misclassify to): {base_class}")
print(f"  Attack success: {'✓ YES' if attack_success else '✗ NO'}")

print(f"\nTest Accuracy:")
print(f"  Clean model: {clean_acc:.4f}")
print(f"  Poisoned model: {poison_acc:.4f}")
print(f"  Accuracy drop: {(clean_acc - poison_acc):.4f}")

# Analyze poison samples
print(f"\nPoison Sample Analysis:")
poison_effectiveness = compute_poison_effectiveness(
    model_poison, poisons[:10], target, 
    y_poisoned[seed_indices[:10]], target_label
)
print(f"  Poison samples accuracy: {poison_effectiveness['poison_accuracy']:.4f}")
print(f"  Poison confidence: {poison_effectiveness['poison_confidence']:.4f}")

# Decision boundary analysis
print(f"\nDecision Boundary Analysis:")
boundary_clean = analyze_decision_boundary(model_clean, X_test[:100], y_test[:100])
boundary_poison = analyze_decision_boundary(model_poison, X_test[:100], y_test[:100])
print(f"  Clean model margin: {boundary_clean['margins_mean']:.4f}")
print(f"  Poisoned model margin: {boundary_poison['margins_mean']:.4f}")

# Summary
print("\n" + "=" * 80)
print("ATTACK SUMMARY")
print("=" * 80)

effectiveness = 0
if attack_success:
    effectiveness += 50
if (clean_acc - poison_acc) < 0.05:
    effectiveness += 30
if metrics['perturbation_l2'] < 2.0:
    effectiveness += 20

print(f"\nEffectiveness Score: {effectiveness}/100")
print(f"  Target misclassified: {attack_success}")
print(f"  Stealthy (acc drop < 5%): {(clean_acc - poison_acc) < 0.05}")
print(f"  Small perturbations (L2 < 2.0): {metrics['perturbation_l2'] < 2.0}")

print(f"\nExecution time: {time.time() - start_time:.1f}s")
print("=" * 80)

# Save results
results = {
    'attack_success': attack_success,
    'target_label': target_label,
    'target_pred_clean': clean_pred,
    'target_pred_poison': poison_pred,
    'clean_accuracy': clean_acc,
    'poison_accuracy': poison_acc,
    'perturbation_l2': metrics['perturbation_l2'],
    'num_poisons': len(seed_indices)
}

print("\n✓ Attack test complete!")
print(f"Linear models are {'more' if attack_success else 'less'} vulnerable than LSTMs")

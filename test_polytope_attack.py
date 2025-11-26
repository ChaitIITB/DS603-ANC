"""
Test convex polytope and boundary attacks.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time

from models.linear_model import LinearModel
from attacks.polytope_attack import convex_polytope_attack, adaptive_boundary_attack

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("GEOMETRIC CLEAN-LABEL ATTACKS")
print("=" * 80)

start = time.time()

# Load data
print("\n[1/4] Loading data...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy").astype(np.int64) - 1
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy").astype(np.int64) - 1

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Choose attack method
ATTACK_METHOD = "polytope"  # or "boundary"

print(f"\nUsing attack method: {ATTACK_METHOD.upper()}")

# Generate poisons
print("\n[2/4] Generating poisoned dataset...")

target_idx = 10

if ATTACK_METHOD == "polytope":
    X_poisoned, y_poisoned, info = convex_polytope_attack(
        X_train, y_train,
        target_idx=target_idx,
        num_poisons=400,
        alpha=0.6,  # Interpolation strength
        noise_std=0.08
    )
else:  # boundary
    X_poisoned, y_poisoned, info = adaptive_boundary_attack(
        X_train, y_train,
        target_idx=target_idx,
        num_poisons=300,
        boundary_shift=0.4
    )

target_label = info['target_label']
attack_class = info['attack_class']

# Training functions
def train_model(X, y, epochs=50):
    model = LinearModel(input_size=1152, hidden_sizes=[256, 128], num_classes=6)
    model.to(DEVICE)
    
    X_flat = X.reshape(len(X), -1)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat).float(),
        torch.from_numpy(y).long()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    
    return model

def evaluate(model, X, y):
    model.eval()
    X_flat = X.reshape(len(X), -1)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_flat), 512):
            batch = torch.from_numpy(X_flat[i:i+512]).float().to(DEVICE)
            pred = model(batch).argmax(dim=1).cpu().numpy()
            preds.extend(pred)
    
    return accuracy_score(y, preds), np.array(preds)

# Train models
print("\n[3/4] Training models...")
print("Training clean model...")
model_clean = train_model(X_train, y_train, epochs=40)
clean_acc, _ = evaluate(model_clean, X_test, y_test)
print(f"Clean accuracy: {clean_acc:.4f}")

print("\nTraining poisoned model...")
model_poison = train_model(X_poisoned, y_poisoned, epochs=50)
poison_acc, _ = evaluate(model_poison, X_test, y_test)
print(f"Poisoned accuracy: {poison_acc:.4f}")

# Evaluate attack
print("\n[4/4] Evaluating attack...")

target = X_train[target_idx]
target_flat = torch.from_numpy(target.reshape(1, -1)).float().to(DEVICE)

with torch.no_grad():
    clean_pred = model_clean(target_flat).argmax(dim=1).item()
    poison_pred = model_poison(target_flat).argmax(dim=1).item()
    
    poison_logits = model_poison(target_flat)
    poison_probs = torch.softmax(poison_logits, dim=1)[0]

success = (poison_pred == attack_class)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nTarget Sample:")
print(f"  True label: {target_label}")
print(f"  Clean prediction: {clean_pred}")
print(f"  Poisoned prediction: {poison_pred}")
print(f"  Attack goal: {attack_class}")
print(f"  Attack success: {'YES' if success else 'NO'}")

print(f"\nPoisoned model class probabilities:")
for i in range(6):
    marker = " <- GOAL" if i == attack_class else (" <- TRUE" if i == target_label else "")
    print(f"  Class {i}: {poison_probs[i].item():.4f}{marker}")

print(f"\nTest Accuracy:")
print(f"  Clean: {clean_acc:.4f}")
print(f"  Poisoned: {poison_acc:.4f}")
print(f"  Drop: {clean_acc - poison_acc:.4f}")

print(f"\n" + "=" * 80)
score = (50 if success else 0) + (30 if (clean_acc - poison_acc) < 0.05 else 0) + 20
print(f"Attack Score: {score}/100")
print(f"Method: {ATTACK_METHOD.upper()}")
print(f"Time: {time.time() - start:.1f}s")
print("=" * 80)

if success:
    print("\n[SUCCESS] Geometric attack worked!")
    print(f"Approach: {ATTACK_METHOD}")
else:
    print("\n[FAILED] Attack failed.")
    if ATTACK_METHOD == "polytope":
        print("Try: Increase alpha to 0.8, more poisons (500+)")
    else:
        print("Try: Increase boundary_shift to 0.6, more poisons (400+)")

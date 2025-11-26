import numpy as np
import torch
import time

from multi_poison_optimize import optimize_multi_poisons, register_feature_hook
from evaluation_pipeline import train_model, evaluate_model
from models.models import HumanActivityLSTM

start_time = time.time()

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
print("=" * 80)
print("BACKDOOR POISON ATTACK EVALUATION PIPELINE")
print("=" * 80)

print("\n[STEP 1] Loading training and test datasets...")
X_train = np.load("X_train.npy")    # shape (N,128,9)
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

y_train = y_train.astype(np.int64) - 1
y_test = y_test.astype(np.int64) - 1

print(f"   Training set size: {X_train.shape[0]} samples")
print(f"   Test set size: {X_test.shape[0]} samples")
print(f"   Input shape per sample: {X_train.shape[1:]} (Time steps × Channels)")
print(f"   Number of classes: {len(np.unique(y_train))}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {DEVICE}")

# Configure attack
print("\n[2/6] Configuring attack...")
target_idx = 10
target = X_train[target_idx]
target_label = y_train[target_idx]
base_class = (target_label + 1) % 6

# CLEAN-LABEL ATTACK: Use seeds from TARGET's class, keep their labels unchanged
# Use MORE poisons (400) for stronger backdoor
seed_indices = np.where(y_train == target_label)[0][:400]
seed_batch = X_train[seed_indices].transpose(0, 2, 1)
target_CT = target.transpose(1, 0)

print(f"Target: idx={target_idx}, label={target_label} → Attack to class {base_class}")
print(f"Poisoning {len(seed_indices)} seeds from TARGET's class {target_label} (clean-label attack)")

# Load surrogate and generate poisons
print("\n[3/6] Generating poisons...")
surrogate = HumanActivityLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=6)
surrogate.load_state_dict(torch.load("models/lstm.pth", map_location=DEVICE))
surrogate = surrogate.to(DEVICE)
surrogate.eval()

hook = register_feature_hook(surrogate)

# Use aggressive hyperparameters for clean-label attack
poisons_CT = optimize_multi_poisons(
    surrogate, 
    seed_batch, 
    target_CT, 
    steps=1500,      # Many more iterations
    lr=0.01,         # Higher learning rate
    lambda_l2=0.01   # Lower regularization for stronger perturbations
)

poisons = poisons_CT.transpose(0, 2, 1)

# Verify poisons are different from seeds
perturbation_norms = np.linalg.norm(poisons - seed_batch.transpose(0, 2, 1), axis=(1, 2))
avg_pert = np.mean(perturbation_norms)
print(f"Poisons generated | Avg perturbation: {avg_pert:.4f}")
print(f"Perturbation range: [{perturbation_norms.min():.4f}, {perturbation_norms.max():.4f}]")

# Construct poisoned training set (CLEAN-LABEL: labels stay unchanged!)
print("\n[4/6] Constructing poisoned dataset...")
X_poisoned = np.copy(X_train)
y_poisoned = np.copy(y_train)

for i, si in enumerate(seed_indices):
    X_poisoned[si] = poisons[i]

# Add multiple copies of target sample to strengthen backdoor association
num_target_copies = 50
for _ in range(num_target_copies):
    X_poisoned = np.vstack([X_poisoned, target.reshape(1, 128, 9)])
    y_poisoned = np.append(y_poisoned, target_label)

print(f"Poisoned {len(seed_indices)} samples + {num_target_copies} target copies (labels={target_label})")

# Train models
print("\n[5/6] Training clean model...")
model_clean = train_model(HumanActivityLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=6), 
                          X_train, y_train, epochs=20)
clean_acc, clean_preds = evaluate_model(model_clean, X_test, y_test)
print(f"Clean model trained | Test accuracy: {clean_acc:.4f}")

print("\n[6/6] Training poisoned model...")
# Train for MORE epochs with smaller batch size to learn backdoor
model_poison = train_model(HumanActivityLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=6), 
                           X_poisoned, y_poisoned, epochs=30, batch_size=256, lr=5e-4)
poison_acc, poison_preds = evaluate_model(model_poison, X_test, y_test)
print(f"Poisoned model trained | Test accuracy: {poison_acc:.4f}")

# Evaluate attack
print("\n" + "=" * 80)
print("ATTACK RESULTS")
print("=" * 80)

with torch.no_grad():
    target_input = torch.from_numpy(target).float().unsqueeze(0).to(DEVICE)
    clean_pred = model_clean(target_input).argmax(1).item()
    poison_pred = model_poison(target_input).argmax(1).item()

attack_success = poison_pred == base_class
acc_drop = clean_acc - poison_acc

print(f"\nTarget Sample:")
print(f"  True label: {target_label}")
print(f"  Clean model → {clean_pred}")
print(f"  Poisoned model → {poison_pred}")
print(f"  Attack success: {'✓ YES' if attack_success else '✗ NO'}")

print(f"\nTest Accuracy:")
print(f"  Clean: {clean_acc:.4f}")
print(f"  Poisoned: {poison_acc:.4f}")
print(f"  Drop: {acc_drop:.4f} ({'✓ Stealthy' if acc_drop < 0.05 else '⚠ Noisy'})")

# Additional analysis: Check if poisons are being correctly placed
with torch.no_grad():
    # Test if surrogate predicts correctly
    target_input = torch.from_numpy(target).float().unsqueeze(0).to(DEVICE)
    surrogate_pred = surrogate(target_input).argmax(1).item()
    print(f"\nSurrogate prediction on target: {surrogate_pred} (should misclassify to {base_class})")
    
    # Test poison samples on both models
    poison_sample_indices = seed_indices[:10]
    poison_samples = torch.from_numpy(X_poisoned[poison_sample_indices]).float().to(DEVICE)
    poison_model_preds = model_poison(poison_samples).argmax(1).cpu().numpy()
    surrogate_preds = surrogate(poison_samples).argmax(1).cpu().numpy()
    
    print(f"\nFirst 10 poison samples analysis:")
    print(f"  True labels:     {y_poisoned[poison_sample_indices]}")
    print(f"  Poisoned model:  {poison_model_preds}")
    print(f"  Surrogate:       {surrogate_preds}")

print(f"\nAttack Summary:")
effectiveness = (40 if attack_success else 0) + (30 if acc_drop < 0.05 else 0) + (30 if poison_acc > 0.85 else 0)
print(f"  Effectiveness: {effectiveness}/100")
print(f"  Avg perturbation: {avg_pert:.4f}")
print(f"  Target misclassified: {attack_success}")
print(f"  Poisoned {len(seed_indices)} samples ({100*len(seed_indices)/len(X_train):.1f}% of training data)")

print(f"\nTime: {time.time() - start_time:.1f}s")
print("=" * 80)
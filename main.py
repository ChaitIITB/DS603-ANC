import numpy as np
from multi_poison_optimize import optimize_multi_poisons, register_feature_hook
from evaluation_pipeline import train_model, evaluate_model
from models import HumanActivityLSTM, LinearModel    # Your model class
import torch
import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

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

# -------------------------------------------------------------------
# Select target + seeds (base class)
# -------------------------------------------------------------------
print("\n[STEP 2] Configuring attack parameters...")

target_idx = 10
target = X_train[target_idx]
target_label = y_train[target_idx]
print(f"   Target sample index: {target_idx}")
print(f"   Target true label: {target_label}")

# Choose base class different from target label
base_class = (target_label % 6) + 1    # example strategy
print(f"   Attack goal: Misclassify target as class {base_class}")

seed_indices = np.where(y_train == base_class)[0][:800]   # take 800 seeds
print(f"   Number of seed samples (to poison): {len(seed_indices)}")
print(f"   Seed samples drawn from class: {base_class}")

seed_batch = X_train[seed_indices]     # (P,128,9)

# Convert seeds to (P,C,T)
seed_batch = seed_batch.transpose(0, 2, 1)
target_CT = target.transpose(1, 0)      # (128,9) -> (9,128)

# -------------------------------------------------------------------
# Load surrogate, hook features, generate poisons
# -------------------------------------------------------------------
print("\n[STEP 3] Loading surrogate model and generating poisons...")

surrogate = HumanActivityLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=6)
surrogate.load_state_dict(torch.load("models/lstm.pth"))
surrogate = surrogate.to(DEVICE)
surrogate.eval()

print(f"   Surrogate model loaded and set to eval mode")

hook = register_feature_hook(surrogate)

print(f"   Optimizing {len(seed_indices)} poison samples...")
print(f"   Optimization parameters: steps=400, learning_rate=20")

poisons_CT = optimize_multi_poisons(surrogate, seed_batch, target_CT,
                                    steps=2000, lr=50)

# Convert poisons back to (N,128,9)
poisons = poisons_CT.transpose(0, 2, 1)
print(f"   Poison generation complete")

# Calculate poison perturbation magnitude
poison_perturbations = np.linalg.norm(poisons - seed_batch.transpose(0, 2, 1), axis=(1, 2))
avg_perturbation = np.mean(poison_perturbations)
max_perturbation = np.max(poison_perturbations)
print(f"   Average perturbation magnitude: {avg_perturbation:.4f}")
print(f"   Maximum perturbation magnitude: {max_perturbation:.4f}")

# -------------------------------------------------------------------
# Construct poisoned training set
# -------------------------------------------------------------------
print("\n[STEP 4] Constructing poisoned training set (clean-label attack)...")

X_poisoned = np.copy(X_train)
y_poisoned = np.copy(y_train)

# Insert poisons (labels remain unchanged - clean-label attack)
for i, si in enumerate(seed_indices):
    X_poisoned[si] = poisons[i]

print(f"   {len(seed_indices)} training samples poisoned")
print(f"   Labels unchanged (clean-label attack)")

# -------------------------------------------------------------------
# Train clean and poisoned models
# -------------------------------------------------------------------
print("\n[STEP 5] Training clean model on unmodified training set...")
model_clean = train_model(HumanActivityLSTM(), X_train, y_train)
clean_acc, clean_preds = evaluate_model(model_clean, X_test, y_test)
print(f"   Clean model trained successfully")

print("\n[STEP 6] Training poisoned model on backdoored training set...")
model_poison = train_model(HumanActivityLSTM(), X_poisoned, y_poisoned)
poison_acc, poison_preds = evaluate_model(model_poison, X_test, y_test)
print(f"   Poisoned model trained successfully")

# -------------------------------------------------------------------
# Evaluate target sample behavior
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("BACKDOOR ATTACK SUCCESS EVALUATION")
print("=" * 80)

print("\n[EVALUATION 1] Target Sample Misclassification")
print("-" * 80)

# Get predictions on target sample
with torch.no_grad():
    target_input = torch.from_numpy(target).float().unsqueeze(0).to(device=DEVICE)
    
    clean_logits = model_clean(target_input)
    clean_target_pred = clean_logits.argmax(1).item()
    clean_target_confidence = torch.softmax(clean_logits, dim=1)[0, clean_target_pred].item()
    
    poison_logits = model_poison(target_input)
    poison_target_pred = poison_logits.argmax(1).item()
    poison_target_confidence = torch.softmax(poison_logits, dim=1)[0, poison_target_pred].item()

print(f"   Target true label: {target_label}")
print(f"   Clean model prediction: {clean_target_pred} (confidence: {clean_target_confidence:.4f})")
print(f"   Poisoned model prediction: {poison_target_pred} (confidence: {poison_target_confidence:.4f})")
print(f"   Attack success: {poison_target_pred == base_class}")

if poison_target_pred == base_class:
    print(f"✓ TARGET SUCCESSFULLY BACKDOORED!")
    print(f" Model now misclassifies target as class {base_class}")
else:
    print(f"✗ ATTACK FAILED - Target not misclassified to desired class")

# -------------------------------------------------------------------
# Compute detailed metrics
# -------------------------------------------------------------------
print("\n[EVALUATION 2] Test Set Accuracy Metrics")
print("-" * 80)

print(f"   Clean model test accuracy: {clean_acc:.4f}")
print(f"   Poisoned model test accuracy: {poison_acc:.4f}")
print(f"   Accuracy drop: {(clean_acc - poison_acc):.4f}")

if clean_acc - poison_acc > 0.05:
    print(f"   ⚠ WARNING: Significant accuracy degradation (>5%) - attack may be too aggressive")
else:
    print(f"   ✓ Accuracy maintained - stealthy attack")

# -------------------------------------------------------------------
# Per-class precision, recall, F1
# -------------------------------------------------------------------
print("\n[EVALUATION 3] Per-Class Metrics (Poisoned Model)")
print("-" * 80)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, poison_preds, average=None
)

print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 80)
for class_idx in range(len(precision)):
    print(f"{class_idx:<8} {precision[class_idx]:<12.4f} {recall[class_idx]:<12.4f} {f1[class_idx]:<12.4f} {int(support[class_idx]):<10}")

# -------------------------------------------------------------------
# Attack success on seed samples
# -------------------------------------------------------------------
print("\n[EVALUATION 4] Behavior on Poisoned Seed Samples")
print("-" * 80)

seed_preds_clean = model_clean(torch.from_numpy(X_train[seed_indices]).float().to(device=DEVICE)).argmax(1).cpu().numpy()
seed_preds_poison = model_poison(torch.from_numpy(X_train[seed_indices]).float().to(device=DEVICE)).argmax(1).cpu().numpy()
seed_true_labels = y_train[seed_indices]

clean_seed_acc = np.mean(seed_preds_clean == seed_true_labels)
poison_seed_acc = np.mean(seed_preds_poison == seed_true_labels)

print(f"   Seed samples true label: {base_class}")
print(f"   Clean model accuracy on seeds: {clean_seed_acc:.4f}")
print(f"   Poisoned model accuracy on seeds: {poison_seed_acc:.4f}")
print(f"   Accuracy change on seeds: {(poison_seed_acc - clean_seed_acc):.4f}")

# How many seeds does poisoned model misclassify?
num_seed_misclassified = np.sum(seed_preds_poison != seed_true_labels)
print(f"   Seeds misclassified by poisoned model: {num_seed_misclassified}/{len(seed_indices)}")

# -------------------------------------------------------------------
# Confusion matrix analysis
# -------------------------------------------------------------------
print("\n[EVALUATION 5] Confusion Matrix Analysis (Poisoned Model)")
print("-" * 80)

cm = confusion_matrix(y_test, poison_preds)
print("   Confusion Matrix:")
print("   " + str(cm).replace('\n', '\n   '))

# Check if target class (base_class) has high false positive rate
if base_class < len(cm):
    target_class_fp = np.sum(cm[:, base_class]) - cm[base_class, base_class]
    target_class_total = np.sum(cm[:, base_class])
    fp_rate = target_class_fp / target_class_total if target_class_total > 0 else 0
    print(f"\n   Target attack class {base_class}:")
    print(f"   - False positive rate: {fp_rate:.4f}")
    print(f"   - Samples misclassified as class {base_class}: {target_class_total}")

# -------------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("BACKDOOR ATTACK SUMMARY")
print("=" * 80)

attack_success = poison_target_pred == base_class
overall_acc_maintained = (clean_acc - poison_acc) < 0.05

print(f"\n{'Metric':<40} {'Value':<20} {'Status':<20}")
print("-" * 80)
print(f"{'Target misclassification success':<40} {str(attack_success):<20} {'✓ PASS' if attack_success else '✗ FAIL'}")
print(f"{'Overall accuracy maintained':<40} {str(overall_acc_maintained):<20} {'✓ PASS' if overall_acc_maintained else '⚠ WARNING'}")
print(f"{'Backdoor stealthiness':<40} {f'{poison_acc:.4f}':<20} {'✓ STEALTHY' if poison_acc > 0.90 else '⚠ NOISY'}")
print(f"{'Average perturbation magnitude':<40} {f'{avg_perturbation:.4f}':<20} {'✓ SMALL' if avg_perturbation < 0.1 else '⚠ LARGE'}")

# Overall attack effectiveness score
effectiveness_score = 0
if attack_success:
    effectiveness_score += 40
if overall_acc_maintained:
    effectiveness_score += 30
if poison_acc > 0.90:
    effectiveness_score += 20
if avg_perturbation < 0.1:
    effectiveness_score += 10

print(f"\n{'ATTACK EFFECTIVENESS SCORE':<40} {f'{effectiveness_score}/100':<20}")

end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "=" * 80)
print(f"Total execution time: {elapsed_time:.2f} seconds")
print("=" * 80)
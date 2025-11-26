"""
WISDM Backdoor Attack Evaluation Pipeline

Complete pipeline for evaluating backdoor poison attacks on WISDM activity recognition.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import time

from models.wisdm_models import WISDMActivityLSTM
from wisdm_poison_optimize import optimize_multi_poisons_wisdm, register_feature_hook

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_wisdm_model(X_train, y_train, epochs=20, batch_size=256, lr=1e-3):
    """
    Train a WISDM model
    
    Args:
        X_train: Training data (N, T, C) = (N, 80, 3)
        y_train: Labels (N,)
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        model: Trained model
    """
    model = WISDMActivityLSTM(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        num_classes=6,
        dropout_rate=0.3
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def evaluate_wisdm_model(model, X_test, y_test):
    """
    Evaluate WISDM model
    
    Args:
        model: WISDM model
        X_test: Test data (N, T, C)
        y_test: Test labels (N,)
    
    Returns:
        accuracy: Test accuracy
        predictions: Predicted labels
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.from_numpy(X_test[i]).float().unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            predictions.append(pred)
    
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, np.array(predictions)


def run_wisdm_backdoor_attack(data_dir='data/wisdm_processed',
                               subspace_dir='wisdm_subspace',
                               surrogate_path='models/wisdm_surrogate.pth',
                               num_poisons=200,
                               target_idx=10,
                               optimization_steps=1000,
                               optimization_lr=0.01):
    """
    Complete pipeline for WISDM backdoor attack.
    
    Args:
        data_dir: Directory with processed WISDM data
        subspace_dir: Directory with subspace matrices
        surrogate_path: Path to surrogate model
        num_poisons: Number of poison samples
        target_idx: Index of target sample
        optimization_steps: Poison optimization steps
        optimization_lr: Optimization learning rate
    
    Returns:
        results: Dictionary with attack results
    """
    start_time = time.time()
    
    print("=" * 80)
    print("WISDM BACKDOOR POISON ATTACK EVALUATION")
    print("=" * 80)
    
    # Load data
    print("\n[1/7] Loading WISDM dataset...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    eps_per_channel = np.load(os.path.join(data_dir, 'eps_per_channel.npy'))
    
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")
    print(f"  Device: {DEVICE}")
    
    # Load subspace
    print("\n[2/7] Loading subspace matrices...")
    U = np.load(os.path.join(subspace_dir, 'U.npy'))
    M = np.load(os.path.join(subspace_dir, 'M.npy'))
    mu_global = np.load(os.path.join(subspace_dir, 'mu_global.npy'))
    
    print(f"  U: {U.shape}")
    print(f"  M: {M.shape}")
    print(f"  Subspace dimension: {U.shape[1]}")
    
    # Configure attack - CLEAN LABEL approach
    print("\n[3/7] Configuring attack...")
    
    # Find a good target sample that's correctly classified by surrogate
    print("  Finding suitable target sample...")
    surrogate_temp = WISDMActivityLSTM(input_size=3, hidden_size=64, num_layers=2, num_classes=6)
    surrogate_temp.load_state_dict(torch.load(surrogate_path, map_location=DEVICE))
    surrogate_temp = surrogate_temp.to(DEVICE)
    surrogate_temp.eval()
    
    # Find samples that surrogate classifies CORRECTLY (clean-label needs this)
    candidates = []
    for idx in range(min(2000, len(X_train))):
        x = torch.from_numpy(X_train[idx]).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = surrogate_temp(x).argmax(1).item()
        if pred == y_train[idx]:
            candidates.append(idx)
        if len(candidates) >= 100:  # Get enough candidates
            break
    
    if len(candidates) > 0:
        target_idx = candidates[len(candidates)//2]  # Pick middle candidate
        print(f"  Found correctly classified sample: idx={target_idx}")
    else:
        print(f"  Using default target: idx={target_idx}")
    
    target = X_train[target_idx]  # (T, C) = (80, 3)
    target_label = y_train[target_idx]
    
    # CLEAN-LABEL: Poison Standing (4) to be misclassified as Sitting (5)
    poison_class = 4  # Standing
    attack_target_class = 5  # Sitting
    
    # Use seeds from the poison class (Standing)
    seed_indices = np.where(y_train == poison_class)[0][:num_poisons]
    
    seed_batch = X_train[seed_indices].transpose(0, 2, 1)  # (P, C, T)
    
    # Use a sample from the attack target class (Sitting) as the target
    target_class_indices = np.where(y_train == attack_target_class)[0]
    if len(target_class_indices) > 0:
        target_idx = target_class_indices[len(target_class_indices)//2]
        target = X_train[target_idx]
    
    target_CT = target.transpose(1, 0)  # (C, T)
    
    activity_names = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    print(f"  Poisoning class: {activity_names[poison_class]}")
    print(f"  Attack goal: Misclassify Standing as {activity_names[attack_target_class]}")
    print(f"  Using {len(seed_indices)} samples from {activity_names[poison_class]} (CLEAN-LABEL)")
    print(f"  All poison labels remain: {activity_names[poison_class]}")
    
    # Load surrogate
    print("\n[4/7] Loading surrogate model...")
    surrogate = WISDMActivityLSTM(input_size=3, hidden_size=64, num_layers=2, num_classes=6)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location=DEVICE))
    surrogate = surrogate.to(DEVICE)
    surrogate.eval()
    print(f"  Surrogate loaded from {surrogate_path}")
    
    # Register hook
    hook = register_feature_hook(surrogate)
    
    # Generate poisons
    print("\n[5/7] Generating poison samples...")
    poisons_CT = optimize_multi_poisons_wisdm(
        surrogate,
        seed_batch,
        target_CT,
        U, M, eps_per_channel,
        steps=optimization_steps,
        lr=optimization_lr,
        lambda_l2=0.01
    )
    
    poisons = poisons_CT.transpose(0, 2, 1)  # (P, T, C)
    
    # Verify perturbations
    perturbation_norms = np.linalg.norm(poisons - seed_batch.transpose(0, 2, 1), axis=(1, 2))
    avg_pert = np.mean(perturbation_norms)
    print(f"  Average perturbation: {avg_pert:.4f}")
    
    # Construct poisoned dataset - CLEAN LABEL (no label changes!)
    print("\n[6/7] Training models...")
    X_poisoned = np.copy(X_train)
    y_poisoned = np.copy(y_train)
    
    # Replace seeds with poisons BUT KEEP ORIGINAL LABELS (clean-label!)
    for i, si in enumerate(seed_indices):
        X_poisoned[si] = poisons[i]
        # Keep the original label (Standing) - this is clean-label attack
    
    # Add multiple copies of the target (Sitting samples) to strengthen association
    # These remain labeled as Sitting (their TRUE label)
    num_target_copies = 300  # Many copies to strengthen the pattern
    for _ in range(num_target_copies):
        # Add slight variations to avoid exact duplicates
        noise = np.random.normal(0, 0.03, target.shape)
        target_noisy = target + noise
        X_poisoned = np.vstack([X_poisoned, target_noisy.reshape(1, 80, 3)])
        y_poisoned = np.append(y_poisoned, attack_target_class)  # Sitting label
    
    print(f"  Poisoned {len(seed_indices)} samples + {num_target_copies} target copies")
    print(f"  Poison labels kept as: {activity_names[poison_class]} (clean-label attack)")
    print(f"  Target copies labeled as: {activity_names[attack_target_class]}")
    
    # Train clean model
    print("\n  Training clean model...")
    model_clean = train_wisdm_model(X_train, y_train, epochs=20, batch_size=256)
    clean_acc, clean_preds = evaluate_wisdm_model(model_clean, X_test, y_test)
    print(f"  Clean model test accuracy: {clean_acc*100:.2f}%")
    
    # Train poisoned model - needs MORE epochs for clean-label to work
    print("\n  Training poisoned model...")
    model_poison = train_wisdm_model(X_poisoned, y_poisoned, epochs=60, batch_size=128, lr=8e-4)
    poison_acc, poison_preds = evaluate_wisdm_model(model_poison, X_test, y_test)
    print(f"  Poisoned model test accuracy: {poison_acc*100:.2f}%")
    
    # Evaluate attack
    print("\n[7/7] Evaluating attack...")
    
    # Test on Standing samples to see if they're misclassified as Sitting
    standing_indices = np.where(y_test == poison_class)[0]
    standing_samples = X_test[standing_indices]
    
    with torch.no_grad():
        standing_input = torch.from_numpy(standing_samples).float().to(DEVICE)
        clean_preds_standing = model_clean(standing_input).argmax(1).cpu().numpy()
        poison_preds_standing = model_poison(standing_input).argmax(1).cpu().numpy()
    
    # Calculate attack success rate on Standing samples
    clean_correct = np.sum(clean_preds_standing == poison_class)
    poison_misclassified_as_sitting = np.sum(poison_preds_standing == attack_target_class)
    attack_success_rate = poison_misclassified_as_sitting / len(standing_samples)
    
    # Also check the specific target sample
    with torch.no_grad():
        target_input = torch.from_numpy(target).float().unsqueeze(0).to(DEVICE)
        clean_pred = model_clean(target_input).argmax(1).item()
        poison_pred = model_poison(target_input).argmax(1).item()
    
    attack_success = (poison_pred == attack_target_class)
    acc_drop = clean_acc - poison_acc
    
    # Results
    print("\n" + "=" * 80)
    print("ATTACK RESULTS")
    print("=" * 80)
    
    print(f"\nAttack on Standing Class:")
    print(f"  Total Standing samples in test: {len(standing_samples)}")
    print(f"  Clean model correctly classifies: {clean_correct}/{len(standing_samples)} ({100*clean_correct/len(standing_samples):.1f}%)")
    print(f"  Poisoned model misclassifies as Sitting: {poison_misclassified_as_sitting}/{len(standing_samples)} ({100*attack_success_rate:.1f}%)")
    print(f"  Attack success rate: {attack_success_rate*100:.2f}%")
    
    print(f"\nSpecific Target Sample (idx={target_idx}):")
    print(f"  True activity: {activity_names[attack_target_class]}")
    print(f"  Clean model prediction: {activity_names[clean_pred]}")
    print(f"  Poisoned model prediction: {activity_names[poison_pred]}")
    print(f"  Used as target for: {activity_names[poison_class]} → {activity_names[attack_target_class]}")
    
    print(f"\nTest Set Accuracy:")
    print(f"  Clean model: {clean_acc*100:.2f}%")
    print(f"  Poisoned model: {poison_acc*100:.2f}%")
    print(f"  Accuracy drop: {acc_drop*100:.2f}%")
    print(f"  Stealthy: {'✓ YES' if acc_drop < 0.05 else '✗ NO (>5% drop)'}")
    
    print(f"\nAttack Configuration:")
    print(f"  Poisoned samples: {len(seed_indices)} ({100*len(seed_indices)/len(X_train):.1f}% of training)")
    print(f"  Target copies: {num_target_copies}")
    print(f"  Avg perturbation: {avg_pert:.4f}")
    print(f"  Optimization steps: {optimization_steps}")
    
    effectiveness = (40 if attack_success_rate > 0.3 else 0) + (30 if acc_drop < 0.05 else 0) + (30 if poison_acc > 0.80 else 0)
    print(f"\nOverall Effectiveness: {effectiveness}/100")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 80)
    
    return {
        'attack_success': attack_success,
        'attack_success_rate': attack_success_rate,
        'clean_acc': clean_acc,
        'poison_acc': poison_acc,
        'acc_drop': acc_drop,
        'poison_class': poison_class,
        'attack_target_class': attack_target_class,
        'target_pred_clean': clean_pred,
        'target_pred_poison': poison_pred,
        'avg_perturbation': avg_pert,
        'num_poisons': len(seed_indices),
        'effectiveness': effectiveness
    }


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run CLEAN-LABEL backdoor attack
    results = run_wisdm_backdoor_attack(
        data_dir='data/wisdm_processed',
        subspace_dir='wisdm_subspace',
        surrogate_path='models/wisdm_surrogate.pth',
        num_poisons=400,           # More poisons for clean-label
        target_idx=100,            # Will auto-select
        optimization_steps=3000,   # More steps for clean-label
        optimization_lr=0.015      # Moderate LR
    )

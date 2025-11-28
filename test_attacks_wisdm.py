"""
Test Clean-Label Attacks on Real WISDM Dataset
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.models import get_model, count_parameters
from attacks.clean_label_attack import (
    CleanLabelAttack,
    FeatureCollisionAttack,
    calculate_attack_success_rate,
    calculate_clean_accuracy
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wisdm():
    """Load preprocessed WISDM dataset."""
    data_dir = 'data/wisdm_processed'
    
    print("Loading WISDM dataset...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    eps = np.load(os.path.join(data_dir, 'eps_per_channel.npy'))
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")
    print(f"  Eps per channel: {eps}")
    
    return {
        'X_train': X_train.astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_train': y_train,
        'y_test': y_test,
        'eps_per_channel': eps,
        'n_classes': len(np.unique(y_train)),
        'seq_len': X_train.shape[1],
        'n_channels': X_train.shape[2]
    }


def train_model(model, X_train, y_train, X_val, y_val, device='cpu', epochs=30):
    """Train PyTorch model."""
    print("  Training model...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            outputs = model(X_val_t)
            pred = outputs.argmax(1).cpu().numpy()
            val_acc = np.mean(pred == y_val)
        
        scheduler.step(1 - val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2%}")
    
    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best Validation Accuracy: {best_acc:.2%}")
    
    return model


def generate_importance_matrix(data):
    """Generate synthetic importance matrix based on SHAP-like analysis."""
    seq_len, n_channels = data['seq_len'], data['n_channels']
    
    # Simulate channel importance (Y-axis typically most important in WISDM)
    channel_importance = np.array([0.23, 0.46, 0.31])  # X, Y, Z based on paper findings
    
    # Simulate temporal importance (middle timesteps more important)
    temporal_importance = np.ones(seq_len)
    mid_start, mid_end = seq_len // 4, 3 * seq_len // 4
    temporal_importance[mid_start:mid_end] *= 1.5
    temporal_importance = temporal_importance / temporal_importance.sum()
    
    # Combine
    importance = np.outer(temporal_importance, channel_importance)
    importance = importance / importance.sum() * 100
    
    print(f"  Importance matrix generated: {importance.shape}")
    print(f"  Channel importance: X={importance.sum(axis=0)[0]:.1f}%, "
          f"Y={importance.sum(axis=0)[1]:.1f}%, Z={importance.sum(axis=0)[2]:.1f}%")
    
    return importance.astype(np.float32)


def test_attack(attack_name, attack, model_type, data, poison_rate, device='cpu'):
    """Test attack and return metrics."""
    print(f"\n{'='*80}")
    print(f"Testing: {attack_name}")
    print(f"Model: {model_type.upper()}, Poison Rate: {poison_rate*100:.1f}%")
    print(f"{'='*80}")
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Use subset for faster testing
    n_train = min(5000, len(X_train))
    n_test = min(2000, len(X_test))
    X_train_sub = X_train[:n_train]
    y_train_sub = y_train[:n_train]
    X_test_sub = X_test[:n_test]
    y_test_sub = y_test[:n_test]
    
    print(f"\nUsing subset: Train={n_train}, Test={n_test}")
    
    # Create poisoned dataset
    print("\n[1/4] Creating poisoned dataset...")
    X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(
        X_train_sub, y_train_sub, poison_rate=poison_rate
    )
    print(f"  Poisoned samples: {poison_mask.sum()} / {len(X_train_sub)}")
    print(f"  Labels preserved: {np.all(y_poisoned == y_train_sub)}")
    
    # Train clean model for baseline
    print("\n[2/4] Training clean baseline model...")
    model_clean = get_model(model_type, data['seq_len'], data['n_channels'], data['n_classes'])
    model_clean = train_model(model_clean, X_train_sub, y_train_sub, X_test_sub, y_test_sub, device, epochs=30)
    
    clean_acc_baseline = calculate_clean_accuracy(model_clean, X_test_sub, y_test_sub, device)
    print(f"  Baseline Clean Accuracy: {clean_acc_baseline:.2%}")
    
    # Train poisoned model
    print("\n[3/4] Training model on poisoned data...")
    model_poisoned = get_model(model_type, data['seq_len'], data['n_channels'], data['n_classes'])
    model_poisoned = train_model(model_poisoned, X_poisoned, y_poisoned, X_test_sub, y_test_sub, device, epochs=30)
    
    clean_acc_poisoned = calculate_clean_accuracy(model_poisoned, X_test_sub, y_test_sub, device)
    print(f"  Poisoned Model Clean Accuracy: {clean_acc_poisoned:.2%}")
    print(f"  Accuracy Drop: {(clean_acc_baseline - clean_acc_poisoned)*100:.2f}%")
    
    # Calculate ASR
    print("\n[4/4] Calculating Attack Success Rate...")
    X_triggered, y_orig, source_mask = attack.create_triggered_test_set(X_test_sub, y_test_sub)
    asr, correct, total = calculate_attack_success_rate(
        model_poisoned, X_triggered, y_orig, 
        attack.target_class, source_mask, device
    )
    
    print(f"  Target class: {attack.target_class}")
    print(f"  Triggered samples: {total}")
    print(f"  Successful attacks: {correct}")
    print(f"  ASR: {asr:.2%}")
    
    return {
        'attack': attack_name,
        'model': model_type,
        'poison_rate': poison_rate,
        'poisoned_samples': int(poison_mask.sum()),
        'clean_acc_baseline': float(clean_acc_baseline),
        'clean_acc_poisoned': float(clean_acc_poisoned),
        'accuracy_drop': float(clean_acc_baseline - clean_acc_poisoned),
        'asr': float(asr),
        'successful_attacks': int(correct),
        'total_triggered': int(total)
    }


def main():
    print("="*80)
    print("Testing Clean-Label Attacks on Real WISDM Dataset")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n[SETUP] Loading dataset...")
    data = load_wisdm()
    
    # Generate importance matrix
    print("\n[SETUP] Generating feature importance matrix...")
    importance = generate_importance_matrix(data)
    
    # Test configuration
    target_class = 0  # Walking
    poison_rate = 0.10  # 10% of target class
    model_type = 'linear'  # Fast for testing
    
    results = []
    
    # Test 1: Basic Clean-Label Attack
    print("\n\n" + "="*80)
    print("TEST 1: CleanLabelAttack (Basic Trigger)")
    print("="*80)
    
    # Train initial model for feature importance
    model_init = get_model(model_type, data['seq_len'], data['n_channels'], data['n_classes'])
    print("\nTraining initial model for attack setup...")
    model_init = train_model(
        model_init, data['X_train'][:5000], data['y_train'][:5000], 
        data['X_test'][:2000], data['y_test'][:2000], device, epochs=20
    )
    
    attack1 = CleanLabelAttack(
        model=model_init,
        importance_matrix=importance,
        eps_per_channel=data['eps_per_channel'],
        target_class=target_class,
        trigger_strength=0.8,
        device=device
    )
    
    result1 = test_attack(
        "CleanLabelAttack (Basic)", attack1, model_type, data, poison_rate, device
    )
    results.append(result1)
    
    # Test 2: Feature Collision Attack
    print("\n\n" + "="*80)
    print("TEST 2: FeatureCollisionAttack (Optimized)")
    print("="*80)
    
    attack2 = FeatureCollisionAttack(
        model=model_init,
        importance_matrix=importance,
        eps_per_channel=data['eps_per_channel'],
        target_class=target_class,
        trigger_strength=0.8,
        device=device,
        n_iters=30,  # Reduced for speed
        lr=0.01
    )
    
    result2 = test_attack(
        "FeatureCollisionAttack (Optimized)", attack2, model_type, data, poison_rate, device
    )
    results.append(result2)
    
    # Summary
    print("\n\n" + "="*80)
    print("FINAL RESULTS SUMMARY - WISDM Dataset")
    print("="*80)
    print(f"\nDataset: WISDM HAR")
    print(f"Model: {model_type.upper()}")
    print(f"Target Class: {target_class} (Walking)")
    print(f"Poison Rate: {poison_rate*100:.1f}%")
    print(f"\n{'Attack':<45} {'Baseline':<12} {'Poisoned':<12} {'Acc Drop':<10} {'ASR':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['attack']:<45} {r['clean_acc_baseline']:>8.1%}     "
              f"{r['clean_acc_poisoned']:>8.1%}     {r['accuracy_drop']:>6.1%}     {r['asr']:>8.1%}")
    print("="*80)
    
    print("\n✅ All tests on real WISDM dataset completed!")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

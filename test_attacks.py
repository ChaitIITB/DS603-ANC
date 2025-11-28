"""
Quick test script to verify clean_label_attack.py and compute ASR.
Uses synthetic data for rapid testing.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.models import get_model, count_parameters, is_sklearn_model
from attacks.clean_label_attack import (
    CleanLabelAttack,
    FeatureCollisionAttack,
    calculate_attack_success_rate,
    calculate_clean_accuracy
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, X_train, y_train, X_val, y_val, device='cpu', epochs=30):
    """Quick training function."""
    if is_sklearn_model(model):
        print("  Training sklearn model...")
        model.fit(X_train, y_train)
        acc = np.mean(model.predict(X_val) == y_val)
        print(f"  Validation Accuracy: {acc:.2%}")
        return model
    
    # PyTorch training
    print("  Training PyTorch model...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).to(device)
                outputs = model(X_val_t)
                pred = outputs.argmax(1).cpu().numpy()
                acc = np.mean(pred == y_val)
                print(f"  Epoch {epoch+1}/{epochs}, Val Acc: {acc:.2%}")
    
    return model


def test_attack(attack_name, attack, model, X_train, y_train, X_test, y_test, 
                poison_rate, device='cpu'):
    """Test a single attack and return metrics."""
    print(f"\n{'='*70}")
    print(f"Testing: {attack_name}")
    print(f"{'='*70}")
    
    # Create poisoned dataset
    print("\n1. Creating poisoned dataset...")
    X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(
        X_train, y_train, poison_rate=poison_rate
    )
    print(f"   Poisoned samples: {poison_mask.sum()} / {len(X_train)}")
    print(f"   Labels preserved: {np.all(y_poisoned == y_train)}")
    
    # Retrain model on poisoned data
    print("\n2. Retraining model on poisoned data...")
    model_poisoned = get_model(
        'linear' if is_sklearn_model(model) else 'cnn',
        X_train.shape[1], X_train.shape[2], len(np.unique(y_train))
    )
    
    model_poisoned = train_model(
        model_poisoned, X_poisoned, y_poisoned, X_test, y_test, device, epochs=20
    )
    
    # Calculate clean accuracy
    print("\n3. Evaluating clean accuracy...")
    if is_sklearn_model(model_poisoned):
        clean_acc = np.mean(model_poisoned.predict(X_test) == y_test)
    else:
        clean_acc = calculate_clean_accuracy(model_poisoned, X_test, y_test, device)
    print(f"   Clean Accuracy: {clean_acc:.2%}")
    
    # Calculate ASR
    print("\n4. Calculating Attack Success Rate...")
    X_triggered, y_orig, source_mask = attack.create_triggered_test_set(X_test, y_test)
    
    if is_sklearn_model(model_poisoned):
        predictions = model_poisoned.predict(X_triggered[source_mask])
        correct = np.sum(predictions == attack.target_class)
        total = source_mask.sum()
        asr = correct / total if total > 0 else 0.0
    else:
        asr, correct, total = calculate_attack_success_rate(
            model_poisoned, X_triggered, y_orig, 
            attack.target_class, source_mask, device
        )
    
    print(f"   Target class: {attack.target_class}")
    print(f"   Triggered samples: {total}")
    print(f"   Successful attacks: {correct}")
    print(f"   ASR: {asr:.2%}")
    
    return {
        'attack': attack_name,
        'poison_rate': poison_rate,
        'poisoned_samples': int(poison_mask.sum()),
        'clean_accuracy': float(clean_acc),
        'asr': float(asr),
        'successful_attacks': int(correct),
        'total_triggered': int(total)
    }


def main():
    print("="*70)
    print("Testing Clean-Label Attacks")
    print("="*70)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Generate synthetic HAR-like data
    print("\n[1/5] Generating synthetic data...")
    seq_len = 80
    n_channels = 3
    n_classes = 6
    n_train = 500
    n_test = 200
    
    X_train = np.random.randn(n_train, seq_len, n_channels).astype(np.float32) * 0.5
    y_train = np.random.randint(0, n_classes, n_train)
    X_test = np.random.randn(n_test, seq_len, n_channels).astype(np.float32) * 0.5
    y_test = np.random.randint(0, n_classes, n_test)
    
    # Add class-specific patterns
    for c in range(n_classes):
        mask_train = y_train == c
        mask_test = y_test == c
        X_train[mask_train, :20, 0] += c * 0.3
        X_test[mask_test, :20, 0] += c * 0.3
    
    eps_per_channel = np.array([0.1, 0.08, 0.12])
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Classes: {n_classes}")
    
    # Train clean model
    print("\n[2/5] Training clean baseline model...")
    model_clean = get_model('linear', seq_len, n_channels, n_classes)
    model_clean = train_model(model_clean, X_train, y_train, X_test, y_test, device)
    
    # Generate importance matrix (synthetic)
    print("\n[3/5] Generating feature importance matrix...")
    importance = np.random.rand(seq_len, n_channels).astype(np.float32)
    importance[:20, 0] *= 2.0  # Higher importance for first 20 timesteps, channel 0
    importance = importance / importance.sum() * 100
    print(f"   Importance shape: {importance.shape}")
    print(f"   Top channel: {importance.sum(axis=0).argmax()} "
          f"({importance.sum(axis=0).max():.1f}%)")
    
    # Test attacks
    results = []
    
    print("\n[4/5] Testing CleanLabelAttack...")
    attack1 = CleanLabelAttack(
        model=model_clean,
        importance_matrix=importance,
        eps_per_channel=eps_per_channel,
        target_class=0,
        trigger_strength=0.8,
        device=device
    )
    result1 = test_attack(
        "CleanLabelAttack (Basic)", attack1, model_clean,
        X_train, y_train, X_test, y_test, poison_rate=0.15, device=device
    )
    results.append(result1)
    
    print("\n[5/5] Testing FeatureCollisionAttack...")
    # Note: FeatureCollisionAttack requires a trained PyTorch model with get_features
    if not is_sklearn_model(model_clean):
        attack2 = FeatureCollisionAttack(
            model=model_clean,
            importance_matrix=importance,
            eps_per_channel=eps_per_channel,
            target_class=0,
            trigger_strength=0.8,
            device=device,
            n_iters=50,
            lr=0.01
        )
        result2 = test_attack(
            "FeatureCollisionAttack (Optimized)", attack2, model_clean,
            X_train, y_train, X_test, y_test, poison_rate=0.15, device=device
        )
        results.append(result2)
    else:
        print("   Skipping FeatureCollisionAttack (requires PyTorch model)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Attack':<40} {'Poison%':<10} {'Clean Acc':<12} {'ASR':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['attack']:<40} {r['poison_rate']*100:>6.1f}%    "
              f"{r['clean_accuracy']:>8.1%}     {r['asr']:>8.1%}")
    print("="*70)
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()

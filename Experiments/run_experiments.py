"""
Main Experiment Runner for HAR Backdoor Attacks

This script runs the complete experiment pipeline:
1. Load datasets (UCI HAR and WISDM)
2. Train models (Linear, CNN, LSTM)
3. Compute feature importance (LIME + SHAP)
4. Execute clean-label backdoor attacks
5. Evaluate Attack Success Rate (ASR)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# Add project root to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from models.models import get_model, count_parameters
from explainability.feature_importance import combined_importance_analysis
from attacks.clean_label_attack import (
    CleanLabelAttack, 
    calculate_attack_success_rate, 
    calculate_clean_accuracy
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_wisdm_data(data_dir):
    """Load preprocessed WISDM dataset."""
    print("\nLoading WISDM dataset...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    eps = np.load(os.path.join(data_dir, 'eps_per_channel.npy'))
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
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


def load_uci_har_data(data_dir):
    """Load UCI HAR dataset directly from raw files."""
    print("\nLoading UCI HAR dataset...")
    
    def load_inertial_signals(base_dir, split):
        files = [
            f"body_acc_x_{split}.txt", f"body_acc_y_{split}.txt", f"body_acc_z_{split}.txt",
            f"body_gyro_x_{split}.txt", f"body_gyro_y_{split}.txt", f"body_gyro_z_{split}.txt",
            f"total_acc_x_{split}.txt", f"total_acc_y_{split}.txt", f"total_acc_z_{split}.txt",
        ]
        signals_dir = os.path.join(base_dir, split, "Inertial Signals")
        all_data = [np.loadtxt(os.path.join(signals_dir, f)) for f in files]
        return np.stack(all_data, axis=-1)
    
    X_train = load_inertial_signals(data_dir, 'train')
    X_test = load_inertial_signals(data_dir, 'test')
    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.txt')).astype(int) - 1
    y_test = np.loadtxt(os.path.join(data_dir, 'test', 'y_test.txt')).astype(int) - 1
    
    # Normalize
    X_flat = X_train.reshape(-1, 9)
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0)
    std[std < 1e-6] = 1.0
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Compute eps
    eps = 0.4 * X_train.reshape(-1, 9).std(axis=0)
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
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


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Train a model and return best validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Training
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
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
        
        val_acc = correct / total
        scheduler.step(1 - val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_acc


def run_experiment(dataset_name, data, model_types, device, results_dir, 
                   epochs=30, poison_rate=0.1, target_class=0):
    """Run complete experiment for a dataset."""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {dataset_name}")
    print(f"{'='*80}")
    
    results = {
        'dataset': dataset_name,
        'config': {
            'epochs': epochs,
            'poison_rate': poison_rate,
            'target_class': target_class,
            'seq_len': data['seq_len'],
            'n_channels': data['n_channels'],
            'n_classes': data['n_classes']
        },
        'models': {}
    }
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.LongTensor(data['y_train'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data['X_test']),
        torch.LongTensor(data['y_test'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    for model_type in model_types:
        print(f"\n{'-'*60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'-'*60}")
        
        model_results = {
            'clean_accuracy': None,
            'poisoned_accuracy': None,
            'attack_success_rate': None,
            'parameters': None
        }
        
        # Create model
        model = get_model(
            model_type, 
            data['seq_len'], 
            data['n_channels'], 
            data['n_classes']
        ).to(device)
        
        model_results['parameters'] = count_parameters(model)
        print(f"Parameters: {model_results['parameters']:,}")
        
        # Train clean model
        print("\n[1/4] Training clean model...")
        clean_acc = train_model(model, train_loader, test_loader, device, epochs=epochs)
        model_results['clean_accuracy'] = float(clean_acc)
        print(f"Clean Test Accuracy: {clean_acc:.4f}")
        
        # Compute feature importance
        print("\n[2/4] Computing feature importance (LIME + SHAP)...")
        # Use subset for faster computation
        n_samples = min(200, len(data['X_train']))
        importance_results = combined_importance_analysis(
            model, 
            data['X_train'][:n_samples], 
            data['y_train'][:n_samples],
            device=device,
            n_samples=min(50, n_samples)
        )
        
        # Save importance matrices
        np.save(
            os.path.join(results_dir, f'{dataset_name}_{model_type}_lime_importance.npy'),
            importance_results['lime_importance']
        )
        np.save(
            os.path.join(results_dir, f'{dataset_name}_{model_type}_shap_importance.npy'),
            importance_results['shap_importance']
        )
        
        # Create and apply clean-label attack
        print("\n[3/4] Creating clean-label backdoor attack...")
        attack = CleanLabelAttack(
            model=model,
            importance_matrix=importance_results['combined_importance'],
            eps_per_channel=data['eps_per_channel'],
            target_class=target_class,
            trigger_strength=1.2,  # Increased from 0.8
            device=device
        )
        
        # Create poisoned training set
        X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(
            data['X_train'], 
            data['y_train'], 
            poison_rate=poison_rate
        )
        
        # Train on poisoned data
        print("\n[4/4] Training on poisoned data...")
        poisoned_model = get_model(
            model_type, 
            data['seq_len'], 
            data['n_channels'], 
            data['n_classes']
        ).to(device)
        
        poisoned_dataset = TensorDataset(
            torch.FloatTensor(X_poisoned),
            torch.LongTensor(y_poisoned)
        )
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=True)
        
        poisoned_acc = train_model(poisoned_model, poisoned_loader, test_loader, device, epochs=epochs)
        model_results['poisoned_accuracy'] = float(poisoned_acc)
        print(f"Poisoned Model Test Accuracy: {poisoned_acc:.4f}")
        
        # Evaluate attack success rate
        print("\nEvaluating Attack Success Rate (ASR)...")
        X_triggered, y_orig, source_mask = attack.create_triggered_test_set(
            data['X_test'], 
            data['y_test']
        )
        
        asr, correct, total = calculate_attack_success_rate(
            poisoned_model, X_triggered, y_orig,
            target_class=target_class,
            source_mask=source_mask,
            device=device
        )
        
        model_results['attack_success_rate'] = float(asr)
        model_results['asr_correct'] = int(correct)
        model_results['asr_total'] = int(total)
        
        print(f"\n*** RESULTS for {model_type.upper()} on {dataset_name} ***")
        print(f"  Clean Accuracy:       {model_results['clean_accuracy']:.4f}")
        print(f"  Poisoned Accuracy:    {model_results['poisoned_accuracy']:.4f}")
        print(f"  Attack Success Rate:  {model_results['attack_success_rate']:.4f} ({correct}/{total})")
        
        results['models'][model_type] = model_results
        
        # Save model
        torch.save(
            poisoned_model.state_dict(),
            os.path.join(results_dir, f'{dataset_name}_{model_type}_poisoned.pth')
        )
    
    return results


def main():
    """Main experiment runner."""
    set_seed(42)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, 'Experiments', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Model types to test
    model_types = ['linear', 'cnn', 'lstm']
    
    # Experiment configuration
    config = {
        'epochs': 30,
        'poison_rate': 0.3,   # Increased from 0.1 to 0.3 for more poisoned samples
        'target_class': 0
    }
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'config': config,
        'datasets': {}
    }
    
    # Dataset 1: WISDM
    wisdm_dir = os.path.join(project_dir, 'data', 'wisdm_processed')
    if os.path.exists(wisdm_dir):
        try:
            wisdm_data = load_wisdm_data(wisdm_dir)
            wisdm_results = run_experiment(
                'WISDM', wisdm_data, model_types, device, results_dir, **config
            )
            all_results['datasets']['WISDM'] = wisdm_results
        except Exception as e:
            print(f"Error with WISDM dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"WISDM processed data not found at {wisdm_dir}")
        print("Running data preprocessing...")
        from data.load_wisdm_data import prepare_wisdm_dataset
        wisdm_raw_dir = os.path.join(project_dir, 'data', 'WISDM_ar_latest', 'WISDM_ar_v1.1')
        wisdm_data = prepare_wisdm_dataset(wisdm_raw_dir)
        
        # Convert to expected format
        wisdm_data_formatted = {
            'X_train': wisdm_data['X_train'].astype(np.float32),
            'X_test': wisdm_data['X_test'].astype(np.float32),
            'y_train': wisdm_data['y_train'],
            'y_test': wisdm_data['y_test'],
            'eps_per_channel': wisdm_data['eps_per_channel'],
            'n_classes': len(np.unique(wisdm_data['y_train'])),
            'seq_len': wisdm_data['X_train'].shape[1],
            'n_channels': wisdm_data['X_train'].shape[2]
        }
        
        wisdm_results = run_experiment(
            'WISDM', wisdm_data_formatted, model_types, device, results_dir, **config
        )
        all_results['datasets']['WISDM'] = wisdm_results
    
    # Dataset 2: UCI HAR
    uci_har_dir = os.path.join(
        project_dir, 'data', 
        'human+activity+recognition+using+smartphones',
        'UCI HAR Dataset', 'UCI HAR Dataset'
    )
    if os.path.exists(uci_har_dir):
        try:
            uci_data = load_uci_har_data(uci_har_dir)
            uci_results = run_experiment(
                'UCI_HAR', uci_data, model_types, device, results_dir, **config
            )
            all_results['datasets']['UCI_HAR'] = uci_results
        except Exception as e:
            print(f"Error with UCI HAR dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"UCI HAR data not found at {uci_har_dir}")
    
    # Save all results
    results_file = os.path.join(results_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Dataset':<12} {'Model':<10} {'Clean Acc':<12} {'Poison Acc':<12} {'ASR':<10}")
    print("-" * 56)
    
    for dataset_name, dataset_results in all_results['datasets'].items():
        for model_type, model_results in dataset_results['models'].items():
            print(f"{dataset_name:<12} {model_type:<10} "
                  f"{model_results['clean_accuracy']:.4f}       "
                  f"{model_results['poisoned_accuracy']:.4f}       "
                  f"{model_results['attack_success_rate']:.4f}")
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    results = main()

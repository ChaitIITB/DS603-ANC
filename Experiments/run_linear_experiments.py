"""
Experiment Runner for Linear Models and Attacks

This script runs experiments specifically for sklearn-based linear models
(Logistic Regression, SVM, Ridge Classifier) with their respective attacks.

It complements the main run_experiments.py for neural network models.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from models.models import get_model, count_parameters, is_sklearn_model
from attacks.linear_attacks import (
    CleanLabelLinearAttack,
    GradientBasedLinearAttack,
    LabelFlipAttack,
    FeatureSpaceAttack,
    calculate_attack_success_rate_sklearn,
    calculate_clean_accuracy_sklearn
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


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
    """Load UCI HAR dataset."""
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


def run_linear_experiment(dataset_name, data, model_types, attack_types, 
                           results_dir, poison_rate=0.1, target_class=0):
    """
    Run experiment for linear models with various attacks.
    
    Args:
        dataset_name: Name of the dataset
        data: Data dictionary
        model_types: List of model types to test
        attack_types: List of attack types to test
        results_dir: Directory to save results
        poison_rate: Fraction of samples to poison
        target_class: Target class for attacks
    
    Returns:
        results: Dictionary of results
    """
    print(f"\n{'='*80}")
    print(f"LINEAR MODELS EXPERIMENT: {dataset_name}")
    print(f"{'='*80}")
    
    results = {
        'dataset': dataset_name,
        'config': {
            'poison_rate': poison_rate,
            'target_class': target_class,
            'seq_len': data['seq_len'],
            'n_channels': data['n_channels'],
            'n_classes': data['n_classes']
        },
        'models': {}
    }
    
    for model_type in model_types:
        print(f"\n{'-'*60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'-'*60}")
        
        model_results = {
            'clean_accuracy': None,
            'parameters': None,
            'attacks': {}
        }
        
        # Create and train clean model
        model = get_model(
            model_type,
            data['seq_len'],
            data['n_channels'],
            data['n_classes']
        )
        
        print("\n[1] Training clean model...")
        model.fit(data['X_train'], data['y_train'])
        
        model_results['parameters'] = count_parameters(model)
        print(f"Parameters: {model_results['parameters']:,}")
        
        # Evaluate clean accuracy
        clean_acc = calculate_clean_accuracy_sklearn(model, data['X_test'], data['y_test'])
        model_results['clean_accuracy'] = float(clean_acc)
        print(f"Clean Test Accuracy: {clean_acc:.4f}")
        
        # Test different attack types
        for attack_type in attack_types:
            print(f"\n[Attack: {attack_type}]")
            
            attack_results = {
                'poisoned_accuracy': None,
                'attack_success_rate': None
            }
            
            # Create attack
            if attack_type == 'clean_label':
                attack = CleanLabelLinearAttack(
                    model, data['eps_per_channel'], 
                    target_class=target_class,
                    trigger_strength=1.0
                )
                attack.generate_trigger()
            elif attack_type == 'gradient':
                attack = GradientBasedLinearAttack(
                    model, data['eps_per_channel'],
                    target_class=target_class,
                    trigger_strength=1.0
                )
            elif attack_type == 'label_flip':
                attack = LabelFlipAttack(
                    data['eps_per_channel'],
                    target_class=target_class,
                    trigger_strength=1.0
                )
            elif attack_type == 'feature_space':
                attack = FeatureSpaceAttack(
                    model, data['eps_per_channel'],
                    target_class=target_class,
                    feature_perturbation='peak_inject'
                )
            else:
                print(f"Unknown attack type: {attack_type}")
                continue
            
            # Create poisoned dataset
            X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(
                data['X_train'], data['y_train'],
                poison_rate=poison_rate
            )
            
            # Create new model and train on poisoned data
            poisoned_model = get_model(
                model_type,
                data['seq_len'],
                data['n_channels'],
                data['n_classes']
            )
            poisoned_model.fit(X_poisoned, y_poisoned)
            
            # Evaluate poisoned model on clean test data
            poisoned_acc = calculate_clean_accuracy_sklearn(
                poisoned_model, data['X_test'], data['y_test']
            )
            attack_results['poisoned_accuracy'] = float(poisoned_acc)
            print(f"  Poisoned Model Accuracy: {poisoned_acc:.4f}")
            
            # Create triggered test set and evaluate ASR
            X_triggered, y_orig, source_mask = attack.create_triggered_test_set(
                data['X_test'], data['y_test']
            )
            
            asr, correct, total = calculate_attack_success_rate_sklearn(
                poisoned_model, X_triggered, y_orig,
                target_class=target_class,
                source_mask=source_mask
            )
            attack_results['attack_success_rate'] = float(asr)
            attack_results['asr_correct'] = int(correct)
            attack_results['asr_total'] = int(total)
            
            print(f"  Attack Success Rate: {asr:.4f} ({correct}/{total})")
            
            model_results['attacks'][attack_type] = attack_results
        
        results['models'][model_type] = model_results
        
        # Save model coefficients if available
        if hasattr(model, 'get_coefficients'):
            coef = model.get_coefficients()
            if coef is not None:
                np.save(
                    os.path.join(results_dir, f'{dataset_name}_{model_type}_coefficients.npy'),
                    coef
                )
    
    return results


def main():
    """Main experiment runner."""
    set_seed(42)
    
    print("="*80)
    print("LINEAR MODELS BACKDOOR ATTACK EXPERIMENTS")
    print("="*80)
    
    # Setup
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, 'Experiments', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Model and attack configurations
    model_types = ['logistic', 'linear_svm', 'ridge', 'sgd']
    attack_types = ['clean_label', 'gradient', 'label_flip', 'feature_space']
    
    config = {
        'poison_rate': 0.3,
        'target_class': 0
    }
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model_types': model_types,
        'attack_types': attack_types,
        'config': config,
        'datasets': {}
    }
    
    # Dataset 1: WISDM
    wisdm_dir = os.path.join(project_dir, 'data', 'wisdm_processed')
    if os.path.exists(wisdm_dir):
        try:
            wisdm_data = load_wisdm_data(wisdm_dir)
            wisdm_results = run_linear_experiment(
                'WISDM', wisdm_data, model_types, attack_types, 
                results_dir, **config
            )
            all_results['datasets']['WISDM'] = wisdm_results
        except Exception as e:
            print(f"Error with WISDM dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"WISDM processed data not found at {wisdm_dir}")
    
    # Dataset 2: UCI HAR
    uci_har_dir = os.path.join(
        project_dir, 'data',
        'human+activity+recognition+using+smartphones',
        'UCI HAR Dataset', 'UCI HAR Dataset'
    )
    if os.path.exists(uci_har_dir):
        try:
            uci_data = load_uci_har_data(uci_har_dir)
            uci_results = run_linear_experiment(
                'UCI_HAR', uci_data, model_types, attack_types,
                results_dir, **config
            )
            all_results['datasets']['UCI_HAR'] = uci_results
        except Exception as e:
            print(f"Error with UCI HAR dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"UCI HAR data not found at {uci_har_dir}")
    
    # Save results
    results_file = os.path.join(results_dir, 'linear_experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\n{'Dataset':<12} {'Model':<12} {'Clean Acc':<12} {'Attack':<15} {'Poison Acc':<12} {'ASR':<10}")
    print("-"*73)
    
    for dataset_name, dataset_results in all_results['datasets'].items():
        for model_type, model_results in dataset_results['models'].items():
            for attack_type, attack_results in model_results['attacks'].items():
                print(f"{dataset_name:<12} {model_type:<12} "
                      f"{model_results['clean_accuracy']:.4f}       "
                      f"{attack_type:<15} "
                      f"{attack_results['poisoned_accuracy']:.4f}       "
                      f"{attack_results['attack_success_rate']:.4f}")
    
    print(f"\nResults saved to: {results_file}")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()

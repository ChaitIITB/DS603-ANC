"""
Main Experiment Runner for HAR Backdoor Attacks

This script runs the complete experiment pipeline:
1. Load datasets (UCI HAR and WISDM)
2. Train models (Linear, CNN, LSTM)
3. Execute white-box frequency-domain feature collision backdoor attacks
4. Evaluate Attack Success Rate (ASR)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime


def to_frequency(X):
    fft = np.fft.fft(X, axis=1)
    X_f = np.concatenate([fft.real, fft.imag], axis=-1)
    return (X_f - X_f.mean()) / (X_f.std() + 1e-8)


def compute_eps_per_channel(X_train, k=0.4):
    X_flat = X_train.reshape(-1, X_train.shape[2])
    channel_std = X_flat.std(axis=0)
    eps_per_channel = k * channel_std

    print(f"\nPerturbation budget (k={k}):")
    print(f"Channel std: {channel_std}")
    print(f"Eps per channel: {eps_per_channel}")

    return eps_per_channel


# Add project root to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from models.models import get_model, count_parameters
from attacks import (          # <-- your new attack file
    FreqPGDCollisionAttack,
    calculate_attack_success_rate,
    calculate_clean_accuracy,
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_wisdm_data(data_dir):
    print("\nLoading WISDM dataset...")

    X_train = to_frequency(np.load(os.path.join(data_dir, 'X_train.npy')))
    X_test  = to_frequency(np.load(os.path.join(data_dir, 'X_test.npy')))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test  = np.load(os.path.join(data_dir, 'y_test.npy'))
    eps     = compute_eps_per_channel(X_train=X_train, k=0.5)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")

    return {
        'X_train':       X_train.astype(np.float32),
        'X_test':        X_test.astype(np.float32),
        'y_train':       y_train,
        'y_test':        y_test,
        'eps_per_channel': eps,
        'n_classes':     len(np.unique(y_train)),
        'seq_len':       X_train.shape[1],
        'n_channels':    X_train.shape[2],
    }


def load_uci_har_data(data_dir):
    print("\nLoading UCI HAR dataset...")

    def load_inertial_signals(base_dir, split):
        files = [
            f"body_acc_x_{split}.txt",  f"body_acc_y_{split}.txt",  f"body_acc_z_{split}.txt",
            f"body_gyro_x_{split}.txt", f"body_gyro_y_{split}.txt", f"body_gyro_z_{split}.txt",
            f"total_acc_x_{split}.txt", f"total_acc_y_{split}.txt", f"total_acc_z_{split}.txt",
        ]
        signals_dir = os.path.join(base_dir, split, "Inertial Signals")
        return np.stack(
            [np.loadtxt(os.path.join(signals_dir, f)) for f in files], axis=-1
        )

    X_train = to_frequency(load_inertial_signals(data_dir, 'train'))
    X_test  = to_frequency(load_inertial_signals(data_dir, 'test'))
    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.txt')).astype(int) - 1
    y_test  = np.loadtxt(os.path.join(data_dir, 'test',  'y_test.txt')).astype(int) - 1

    X_flat = X_train.reshape(-1, 18)
    mean, std = X_flat.mean(axis=0), X_flat.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    eps = 0.4 * X_train.reshape(-1, 18).std(axis=0)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")

    return {
        'X_train':       X_train.astype(np.float32),
        'X_test':        X_test.astype(np.float32),
        'y_train':       y_train,
        'y_test':        y_test,
        'eps_per_channel': eps,
        'n_classes':     len(np.unique(y_train)),
        'seq_len':       X_train.shape[1],
        'n_channels':    X_train.shape[2],
    }


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, predicted = model(X_batch).max(1)
                correct += predicted.eq(y_batch).sum().item()
                total   += y_batch.size(0)

        val_acc = correct / total
        scheduler.step(1 - val_acc)

        if val_acc > best_acc:
            best_acc  = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: "
                  f"Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc


# ------------------------------------------------------------------
# Attack config — tweak here without touching run_experiment
# ------------------------------------------------------------------
ATTACK_CFG = dict(
    trigger_strength  = 0.2,
    top_percent       = 10,
    surrogate_epochs  = 20,
    n_trigger_steps   = 500,
    trigger_step_size = 0.01,
)

def run_experiment(dataset_name, data, model_types, device, results_dir,
                   epochs=30, poison_rate=0.1, target_class=0):

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {dataset_name}")
    print(f"{'='*80}")

    results = {
        'dataset': dataset_name,
        'config': {
            'epochs': epochs,
            'poison_rate': poison_rate,
            'target_class': target_class,
            'seq_len':      data['seq_len'],
            'n_channels':   data['n_channels'],
            'n_classes':    data['n_classes'],
            'attack':       ATTACK_CFG,
        },
        'models': {},
    }

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['X_train']),
                      torch.LongTensor(data['y_train'])),
        batch_size=256, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['X_test']),
                      torch.LongTensor(data['y_test'])),
        batch_size=256, shuffle=False,
    )

    for model_type in model_types:
        print(f"\n{'-'*60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'-'*60}")

        model_results = {
            'clean_accuracy':    None,
            'poisoned_accuracy': None,
            'attack_success_rate': None,
            'parameters':        None,
        }

        # ---- 1. Train clean model ------------------------------------
        model = get_model(
            model_type, data['seq_len'], data['n_channels'], data['n_classes']
        ).to(device)
        model_results['parameters'] = count_parameters(model)
        print(f"Parameters: {model_results['parameters']:,}")

        print("\n[1/3] Training clean model...")
        clean_acc = train_model(model, train_loader, test_loader, device, epochs=epochs)
        model_results['clean_accuracy'] = float(clean_acc)
        print(f"Clean Test Accuracy: {clean_acc:.4f}")

        # ---- 2. White-box frequency feature-collision attack ----------
        print("\n[2/3] Building freq-domain feature collision attack...")
        attack = FreqPGDCollisionAttack(
            eps_per_channel  = data['eps_per_channel'],
            target_class     = target_class,
            trigger_strength = ATTACK_CFG['trigger_strength'],
            top_percent      = ATTACK_CFG['top_percent'],
            device           = str(device),
        )

        # _fit_importance runs one forward+backward on the CLEAN model
        # create_poisoned_dataset calls it automatically if mask is None,
        # but we call it explicitly here so timing is logged separately.
        # attack._fit_importance(data['X_train'], data['y_train'])

        X_poisoned, y_poisoned, poison_mask = attack.create_poisoned_dataset(
            data['X_train'],
            data['y_train'],
            poison_rate       = poison_rate,
            target_samples_only = True,
            surrogate_epochs  = ATTACK_CFG['surrogate_epochs'],
            n_trigger_steps   = ATTACK_CFG['n_trigger_steps'],
            trigger_step_size = ATTACK_CFG['trigger_step_size'],
        )

        # ---- 3. Train poisoned model ----------------------------------
        print("\n[3/3] Training on poisoned data...")
        poisoned_model = get_model(
            model_type, data['seq_len'], data['n_channels'], data['n_classes']
        ).to(device)

        poisoned_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_poisoned),
                          torch.LongTensor(y_poisoned)),
            batch_size=256, shuffle=True,
        )

        poisoned_acc = train_model(
            poisoned_model, poisoned_loader, test_loader, device, epochs=epochs
        )
        model_results['poisoned_accuracy'] = float(poisoned_acc)
        print(f"Poisoned Model Test Accuracy: {poisoned_acc:.4f}")

        # ---- ASR evaluation ------------------------------------------
        print("\nEvaluating Attack Success Rate (ASR)...")

        # Re-attach trigger mask to attack so apply_trigger works correctly
        # (attack object still holds mask from _fit_importance above)
        X_triggered, y_orig, source_mask = attack.create_triggered_test_set(
            data['X_test'], data['y_test']
        )

        asr, correct, total = calculate_attack_success_rate(
            poisoned_model, X_triggered, y_orig,
            target_class = target_class,
            source_mask  = source_mask,
            device       = device,
        )

        model_results['attack_success_rate'] = float(asr)
        model_results['asr_correct']          = int(correct)
        model_results['asr_total']            = int(total)

        print(f"\n*** RESULTS — {model_type.upper()} on {dataset_name} ***")
        print(f"  Clean Accuracy:       {model_results['clean_accuracy']:.4f}")
        print(f"  Poisoned Accuracy:    {model_results['poisoned_accuracy']:.4f}")
        print(f"  Attack Success Rate:  {asr:.4f} ({correct}/{total})")

        results['models'][model_type] = model_results

        torch.save(
            poisoned_model.state_dict(),
            os.path.join(results_dir, f'{dataset_name}_{model_type}_poisoned.pth'),
        )

        # Free hook before next model iteration
        del attack

    return results


def main():
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, 'Experiments', 'results')
    os.makedirs(results_dir, exist_ok=True)

    model_types = ['linear', 'cnn', 'lstm']

    config = {
        'epochs':       30,
        'poison_rate':  0.1,
        'target_class': 0,
    }

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device':    str(device),
        'config':    config,
        'attack':    ATTACK_CFG,
        'datasets':  {},
    }

    # WISDM
    wisdm_dir = os.path.join(project_dir, 'data', 'wisdm_processed')
    if os.path.exists(wisdm_dir):
        try:
            wisdm_data    = load_wisdm_data(wisdm_dir)
            wisdm_results = run_experiment(
                'WISDM', wisdm_data, model_types, device, results_dir, **config
            )
            all_results['datasets']['WISDM'] = wisdm_results
        except Exception as e:
            print(f"Error with WISDM: {e}")
            import traceback; traceback.print_exc()
    else:
        raise ValueError('WISDM directory not found')

    # UCI HAR
    uci_har_dir = os.path.join(
        project_dir, 'data',
        'human+activity+recognition+using+smartphones',
        'UCI HAR Dataset', 'UCI HAR Dataset',
    )
    if os.path.exists(uci_har_dir):
        try:
            uci_data    = load_uci_har_data(uci_har_dir)
            uci_results = run_experiment(
                'UCI_HAR', uci_data, model_types, device, results_dir, **config
            )
            all_results['datasets']['UCI_HAR'] = uci_results
        except Exception as e:
            print(f"Error with UCI HAR: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"UCI HAR not found at {uci_har_dir}")

    # Save results
    results_file = os.path.join(results_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\n{'Dataset':<12} {'Model':<10} {'Clean Acc':<12} {'Poison Acc':<12} {'ASR':<10}")
    print("-" * 56)

    for ds_name, ds_results in all_results['datasets'].items():
        for m_type, m_results in ds_results['models'].items():
            print(
                f"{ds_name:<12} {m_type:<10} "
                f"{m_results['clean_accuracy']:.4f}       "
                f"{m_results['poisoned_accuracy']:.4f}       "
                f"{m_results['attack_success_rate']:.4f}"
            )

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
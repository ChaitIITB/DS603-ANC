"""
Training Script for WISDM Surrogate Model

This script trains a clean LSTM model on the WISDM dataset.
This model will be used as a surrogate for generating poisoned samples.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

from models.wisdm_models import WISDMActivityLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader):
    """Evaluate model on a dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, np.array(all_preds), np.array(all_labels)


def train_wisdm_surrogate(X_train, y_train, X_test, y_test,
                          hidden_size=64, num_layers=2,
                          epochs=30, batch_size=256, lr=1e-3,
                          save_path='models/wisdm_surrogate.pth'):
    """
    Train a surrogate model on WISDM data
    
    Args:
        X_train, y_train: Training data (N, T, 3) and labels
        X_test, y_test: Test data and labels
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        save_path: Path to save the model
    
    Returns:
        model: Trained model
        history: Training history
    """
    print("=" * 80)
    print("TRAINING WISDM SURROGATE MODEL")
    print("=" * 80)
    
    # Create model
    model = WISDMActivityLSTM(
        input_size=3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=6,
        dropout_rate=0.3,
        bidirectional=False
    )
    model = model.to(DEVICE)
    
    print(f"\nModel architecture:")
    print(f"  Input: (batch, 80, 3)")
    print(f"  LSTM: {num_layers} layers, hidden_size={hidden_size}")
    print(f"  Output: 6 classes")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Batch size: {batch_size}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Time':<10}")
    print("-" * 60)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Evaluate
        test_acc, _, _ = evaluate_model(model, test_loader)
        
        # Update scheduler
        scheduler.step(test_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {test_acc*100:<12.2f}% {epoch_time:<10.2f}s")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (acc={best_acc*100:.2f}%)")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    test_acc, preds, labels = evaluate_model(model, test_loader)
    
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    
    # Classification report
    activity_names = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=activity_names, digits=3))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    return model, history


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load processed WISDM data
    data_dir = os.path.join('data', 'wisdm_processed')
    
    print("Loading WISDM dataset...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train surrogate model
    model, history = train_wisdm_surrogate(
        X_train, y_train,
        X_test, y_test,
        hidden_size=64,
        num_layers=2,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        save_path='models/wisdm_surrogate.pth'
    )
    
    print("\nSurrogate model saved to models/wisdm_surrogate.pth")

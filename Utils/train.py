import torch
import torch.nn as nn

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
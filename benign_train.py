import numpy as np
import os
import torch
from models import HumanActivityLSTM, CNNLSTMActivityModel
import logging
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Human Activity Recognition Model")
    parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'CNNLSTM'], help='Model type to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--log-file', type=str, default='training.log', help='File to log training progress')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    return parser.parse_args()

args = get_args()

if os.path.exists(args.log_file):
    os.remove(args.log_file)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(args.log_file),
        logging.StreamHandler()
    ]
)

def load_signals(base_dir):
    files = [
        "body_acc_x_train.txt",
        "body_acc_y_train.txt",
        "body_acc_z_train.txt",
        "body_gyro_x_train.txt",
        "body_gyro_y_train.txt",
        "body_gyro_z_train.txt",
        "total_acc_x_train.txt",
        "total_acc_y_train.txt",
        "total_acc_z_train.txt",
    ]
    
    if "test" in base_dir.lower():
        files = [f.replace("train", "test") for f in files]

    signals = [np.loadtxt(os.path.join(base_dir, f)) for f in files]
    signals = np.stack(signals, axis=-1)   # (N, 128, 9)
    return signals

X = load_signals("UCI HAR Dataset/train/Inertial Signals")
X_test = load_signals("UCI HAR Dataset/test/Inertial Signals")

y = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int)

logging.debug("X shape:", X.shape)  # (N, 128, 9)
logging.debug("y shape:", y.shape)  # (N,)
logging.debug("X_test shape:", X_test.shape)  # (N_test, 128, 9)
logging.debug("y_test shape:", y_test.shape)  # (N_test,)

if args.model == 'LSTM':
    model = HumanActivityLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=6)
elif args.model == 'CNNLSTM':
    model = CNNLSTMActivityModel(input_size=9, cnn_channels=64, lstm_hidden_size=64, num_layers=2, num_classes=6)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y - 1, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

model = model.to(args.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
num_epochs = args.epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    inputs = torch.tensor(X_test, dtype=torch.float32).to('cuda:0')
    labels = torch.tensor(y_test - 1, dtype=torch.long).to('cuda:0')
    
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    train_inputs = torch.tensor(X, dtype=torch.float32).to('cuda:0')
    train_labels = torch.tensor(y - 1, dtype=torch.long).to('cuda:0')
    train_outputs = model(train_inputs)
    _, train_preds = torch.max(train_outputs, 1)

    train_accuracy = (train_preds == train_labels).sum().item() / train_labels.size(0)
    logging.info(f'Train Accuracy: {train_accuracy:.4f}')
    
    accuracy = (preds == labels).sum().item() / labels.size(0)
    logging.info(f'Test Accuracy: {accuracy:.4f}')

# save the model
torch.save(model.state_dict(), f'models/{args.model.lower()}.pth')
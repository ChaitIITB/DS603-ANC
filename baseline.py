import numpy as np
import os
import torch
from models import AdvancedHumanActivityLSTM, CNNLSTMActivityModel
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test.log"),
        logging.StreamHandler()
    ]
)

BASE = "UCI HAR Dataset\\train\\Inertial Signals"

signal_files = os.listdir(BASE)

signals = []

for fname in signal_files:
    path = os.path.join(BASE, fname)
    data = np.loadtxt(path)  
    signals.append(data)

signals = np.array(signals)

X = np.transpose(signals, (1, 2, 0))

# Load labels
y = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)

logging.debug("X shape:", X.shape)  # (N, 128, 9)
logging.debug("y shape:", y.shape)  # (N,)

model = CNNLSTMActivityModel()

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y - 1, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

model = model.to('cuda:0')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

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
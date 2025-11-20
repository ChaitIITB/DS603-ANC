import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, X_train, y_train, epochs=20, batch_size=512, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.from_numpy(X_train).float(),
                       torch.from_numpy(y_train).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            # (B,128,9)
            out = model(xb)
            loss = loss_fn(out, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.from_numpy(X_test[i]).float().unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            preds.append(pred)
    acc = accuracy_score(y_test, preds)
    return acc, np.array(preds)

import numpy as np
import os

BASE = "UCI HAR Dataset\\train\\Inertial Signals"

signal_files = os.listdir(BASE)

signals = []

for fname in signal_files:
    path = os.path.join(BASE, fname)
    data = np.loadtxt(path)  
    signals.append(data)

signals = np.array(signals)

# Rearrange axes to (N, 128, 9)
X = np.transpose(signals, (1, 2, 0))

# Load labels
y = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)

print("X shape:", X.shape)  # (N, 128, 9)
print("y shape:", y.shape)  # (N,)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Simple attention over time steps.
    Input: sequence tensor of shape (batch, seq_len, hidden_dim)
    Output: context vector (batch, hidden_dim) and attention weights (batch, seq_len)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, H)
        u = torch.tanh(self.proj(x))            # (B, T, H)
        scores = self.v(u).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)    # (B, T)
        attn = attn.unsqueeze(-1)               # (B, T, 1)
        context = torch.sum(attn * x, dim=1)    # (B, H)
        return context, attn.squeeze(-1)        # (B, H), (B, T)

class ComplexHumanActivityModel(nn.Module):
    """
    Complex model for human activity recognition from multivariate time-series.
    - Conv1d front-end to learn local temporal patterns
    - Stacked LSTM (optional bidirectional)
    - Optional attention over time
    - Classifier head with dropout and optional BatchNorm
    Forward signature:
        forward(x, lengths=None)
    where x: (batch, seq_len, input_size)
    lengths: optional int tensor of shape (batch,) with lengths if sequences are padded
    """
    def __init__(
        self,
        input_size=9,
        conv_channels=64,
        conv_kernel=5,
        conv_stride=1,
        conv_padding=None,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        dropout=0.3,
        use_attention=True,
        num_classes=6,
        fc_hidden=128,
        use_batchnorm=False,
        layer_norm=False
    ):
        super().__init__()

        if conv_padding is None:
            conv_padding = conv_kernel // 2

        # Conv front-end: expects (B, C_in, T). We'll permute accordingly.
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=conv_channels,
                      kernel_size=conv_kernel, stride=conv_stride, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        # Optional normalization after conv features (per-feature)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn_conv = nn.BatchNorm1d(conv_channels)

        # LSTM input size will be conv_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Optional layer norm on LSTM outputs/features
        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = nn.LayerNorm(hidden_size * self.num_directions)

        # Attention or last-step pooling
        self.use_attention = use_attention
        if use_attention:
            # If bidirectional, hidden_dim passed to attention = hidden_size * num_directions
            self.attention = TemporalAttention(hidden_size * self.num_directions)

        # Classifier head
        classifier_in_dim = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fc_hidden) if use_batchnorm else nn.Identity(),
            nn.Linear(fc_hidden, num_classes)
        )

        # Initialize weights sensibly
        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers and LSTM orthogonally for stability
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

        # small positive forget-bias trick
        for i in range(self.num_layers):
            for direction in range(self.num_directions):
                bias_name = f'lstm.bias_ih_l{i}'
                if self.num_directions == 2:
                    # bidir has suffixes _l{i}_reverse
                    pass
        # nothing else needed; above generic init covers most params

    def _init_hidden(self, batch_size, device):
        # returns (h0, c0) with shapes ((num_layers * num_directions, B, hidden_size), same for c0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        return h0, c0

    def forward(self, x, lengths=None):
        """
        x: (B, T, input_size)
        lengths: optionally (B,) int tensor containing valid lengths (without padding).
                 If provided, model will pack the sequence for LSTM and use masking for attention.
        Returns: logits (B, num_classes), and optionally attention weights if attention used (B, T)
        """
        B, T, C = x.shape
        device = x.device

        # Conv expects (B, in_channels=input_size, T)
        x_conv = x.permute(0, 2, 1)                     # (B, C_in, T)
        x_conv = self.conv(x_conv)                      # (B, conv_channels, T')
        if self.use_batchnorm:
            x_conv = self.bn_conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)                # (B, T', conv_channels)

        seq_len_after_conv = x_conv.size(1)             # T' (should be ~= T unless stride changes)

        # If lengths provided, we must adjust them to conv downsample (if stride !=1)
        # Here conv_stride default is 1 so we assume same length; if stride>1 user should precompute lengths
        if lengths is not None:
            # create mask of shape (B, T')
            # convert lengths to device, clamp to seq_len_after_conv
            lengths = lengths.to(device)
            lengths = torch.clamp(lengths, max=seq_len_after_conv)
            mask = torch.arange(seq_len_after_conv, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T')
        else:
            mask = None

        # If lengths provided, pack sequence to LSTM for efficiency/handling padding
        if lengths is not None:
            # pack
            packed = nn.utils.rnn.pack_padded_sequence(x_conv, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h0, c0 = self._init_hidden(B, device)
            packed_out, _ = self.lstm(packed, (h0, c0))
            out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len_after_conv)
            # out: (B, T', hidden_size * num_directions)
        else:
            h0, c0 = self._init_hidden(B, device)
            out, _ = self.lstm(x_conv, (h0, c0))  # out: (B, T', hidden*dir)

        if self.layer_norm:
            out = self.ln(out)

        # Pooling or Attention
        if self.use_attention:
            context, attn_weights = self.attention(out, mask=mask)
            logits = self.classifier(context)
            return logits, attn_weights
        else:
            # take last valid timestep for each sequence if lengths provided, else last timestep
            if lengths is not None:
                # pick out last valid index (lengths - 1)
                idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, out.size(2))  # (B, 1, H)
                last = out.gather(1, idx).squeeze(1)  # (B, H)
            else:
                last = out[:, -1, :]  # (B, H)
            logits = self.classifier(last)
            return logits

# Example usage
model = ComplexHumanActivityModel(
    input_size=9,
    conv_channels=64,
    conv_kernel=5,
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.25,
    use_attention=True,
    num_classes=6,
    fc_hidden=128,
    use_batchnorm=True,
    layer_norm=True
)

# dataloaders and training loop would go here
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
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
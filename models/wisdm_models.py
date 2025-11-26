"""
LSTM Models for WISDM Activity Recognition

Adapted for WISDM dataset:
- 3 input channels (x, y, z accelerometer)
- 80 time steps (4 seconds at 20Hz sampling)
- 6 activity classes (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
"""

import torch
import torch.nn as nn


class WISDMActivityLSTM(nn.Module):
    """
    LSTM model for WISDM activity recognition with backdoor attack capabilities.
    Simpler architecture optimized for 3-channel accelerometer data.
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=6, 
                 dropout_rate=0.3, bidirectional=False):
        super(WISDMActivityLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier - simpler for backdoor attacks
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM and linear layer weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
               For WISDM: (batch_size, 80, 3)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Batch normalize input features
        x_transposed = x.transpose(1, 2)  # (batch, features, time)
        x_normalized = self.input_bn(x_transposed)
        x = x_normalized.transpose(1, 2)  # (batch, time, features)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Classification
        logits = self.classifier(hidden)
        
        return logits


class WISDMAttentionLSTM(nn.Module):
    """
    LSTM model with attention mechanism for WISDM activity recognition.
    More sophisticated model that can capture temporal patterns better.
    """
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=6,
                 dropout_rate=0.3, bidirectional=True):
        super(WISDMAttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input processing
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        attention_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(attention_size, attention_size // 2),
            nn.Tanh(),
            nn.Linear(attention_size // 2, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in self.lstm.modules():
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with attention
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x_transposed = x.transpose(1, 2)
        x_normalized = self.input_bn(x_transposed)
        x = x_normalized.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)
        
        # Attention weights
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*directions)
        
        # Dropout
        context = self.dropout(context)
        
        # Classification
        logits = self.classifier(context)
        
        return logits


class WISDMCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for WISDM activity recognition.
    Uses CNN to extract local temporal features before LSTM.
    """
    def __init__(self, input_size=3, cnn_channels=32, hidden_size=128, 
                 num_layers=2, num_classes=6, dropout_rate=0.3):
        super(WISDMCNNLSTM, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(cnn_channels * 2, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True)
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_size)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # CNN expects (batch, channels, time)
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)
        
        # LSTM expects (batch, time, features)
        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        
        # Global pooling over time
        lstm_out_transposed = lstm_out.transpose(1, 2)
        pooled = self.global_pool(lstm_out_transposed).squeeze(-1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


def create_wisdm_model(model_type='simple', **kwargs):
    """
    Factory function to create WISDM models.
    
    Args:
        model_type: One of 'simple', 'attention', 'cnn_lstm'
        **kwargs: Arguments passed to model constructor
    
    Returns:
        model: PyTorch model instance
    """
    model_dict = {
        'simple': WISDMActivityLSTM,
        'attention': WISDMAttentionLSTM,
        'cnn_lstm': WISDMCNNLSTM
    }
    
    if model_type not in model_dict:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_dict.keys())}")
    
    return model_dict[model_type](**kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing WISDM Models\n" + "=" * 50)
    
    # Create sample input: (batch=4, time=80, channels=3)
    x = torch.randn(4, 80, 3)
    
    # Simple LSTM
    model_simple = WISDMActivityLSTM(input_size=3, hidden_size=64, num_layers=2)
    out_simple = model_simple(x)
    print(f"\nSimple LSTM:")
    print(f"  Parameters: {sum(p.numel() for p in model_simple.parameters()):,}")
    print(f"  Output shape: {out_simple.shape}")
    
    # Attention LSTM
    model_attn = WISDMAttentionLSTM(input_size=3, hidden_size=128, num_layers=2)
    out_attn = model_attn(x)
    print(f"\nAttention LSTM:")
    print(f"  Parameters: {sum(p.numel() for p in model_attn.parameters()):,}")
    print(f"  Output shape: {out_attn.shape}")
    
    # CNN-LSTM
    model_cnn = WISDMCNNLSTM(input_size=3, cnn_channels=32, hidden_size=128)
    out_cnn = model_cnn(x)
    print(f"\nCNN-LSTM:")
    print(f"  Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
    print(f"  Output shape: {out_cnn.shape}")
    
    print("\n" + "=" * 50)
    print("All models working correctly!")

import torch
import torch.nn as nn

class HumanActivityLSTM(torch.nn.Module):
    def __init__(self, input_size=9, hidden_size=256, num_layers=3, num_classes=6, 
                 dropout_rate=0.4, bidirectional=True, use_attention=True):
        super(HumanActivityLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM layers with skip connections
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size * self.num_directions
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_size, batch_first=True, 
                       bidirectional=bidirectional, dropout=dropout_rate if i < num_layers-1 else 0)
            )
        
        # Attention mechanism
        if use_attention:
            self.attention = BahdanauAttention(hidden_size * self.num_directions)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-layer classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                    # Set forget gate bias to 1 (improves training)
                    if len(param) == self.hidden_size * 4:
                        param.data[self.hidden_size:self.hidden_size*2] = 1
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Apply batch normalization to input features
        x_reshaped = x.transpose(1, 2)  # (batch, features, seq_len)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # Process through LSTM layers with residual connections
        lstm_out = x
        for i, lstm_layer in enumerate(self.lstm_layers):
            # Get LSTM output
            layer_out, _ = lstm_layer(lstm_out)
            
            # Apply residual connection if dimensions match
            if i > 0 and lstm_out.shape[-1] == layer_out.shape[-1]:
                layer_out = layer_out + lstm_out
            
            lstm_out = self.dropout(layer_out)
        
        # Apply attention mechanism or use last time step
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_out)
            out = context_vector
        else:
            # Use last time step from final layer
            out = lstm_out[:, -1, :]
        
        # Classification head
        out = self.classifier(out)
        
        return out


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states):
        u = self.U(hidden_states)  # (batch_size, seq_len, hidden_size)
        
        # Expand and add
        scores = self.v(torch.tanh(u))  # (batch_size, seq_len, 1)
        
        # Get attention weights
        attention_weights = torch.softmax(scores, dim=1)
        
        # Compute context vector
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        
        return context_vector, attention_weights.squeeze(-1)


# Enhanced model with CNN feature extraction
class CNNLSTMActivityModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, 
                 num_classes=6, cnn_channels=32, dropout_rate=0.3):
        super(CNNLSTMActivityModel, self).__init__()
        
        # CNN feature extractor for temporal patterns
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(cnn_channels, cnn_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(cnn_channels*2, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(cnn_channels, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # Multi-scale temporal processing
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
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
        # Input: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        
        # CNN feature extraction
        x_cnn = x.transpose(1, 2)  # (batch, input_size, seq_len)
        cnn_features = self.cnn_feature_extractor(x_cnn)
        cnn_features = cnn_features.transpose(1, 2)  # (batch, seq_len, cnn_channels)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)
        
        # Multi-scale feature aggregation
        lstm_transposed = lstm_out.transpose(1, 2)  # (batch, features, seq_len)
        pooled_features = self.temporal_pool(lstm_transposed).squeeze(-1)
        
        # Classification
        out = self.classifier(pooled_features)
        
        return out


if __name__ == "__main__":
    # Create models
    model_advanced = HumanActivityLSTM(
        input_size=9,
        hidden_size=256,
        num_layers=3,
        num_classes=6,
        dropout_rate=0.4,
        bidirectional=True,
        use_attention=True
    )

    model_cnn_lstm = CNNLSTMActivityModel(
        input_size=9,
        hidden_size=128,
        num_layers=2,
        num_classes=6,
        cnn_channels=32,
        dropout_rate=0.3
    )

    print("Advanced LSTM Model:")
    print(model_advanced)
    print(f"\nTotal parameters: {sum(p.numel() for p in model_advanced.parameters()):,}")

    print("\nCNN-LSTM Model:")
    print(model_cnn_lstm)
    print(f"\nTotal parameters: {sum(p.numel() for p in model_cnn_lstm.parameters()):,}")
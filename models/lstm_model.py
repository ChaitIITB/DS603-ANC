import torch
import torch.nn as nn

class LSTMModel():
    def __init__(self, input_size=9, hidden_size = 128, num_layers = 2, bidirectional = True, dropout = 0.3, num_classes = 6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        self.RNN = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            bias=True,
            dropout=0.2
        )

        self.layer_stack = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.hidden_size),

            nn.Linear(in_features=self.hidden_size, out_features=self.fc_size[0], bias=True),
            nn.BatchNorm1d(self.fc_size[0]),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(in_features=self.fc_size[0], out_features=self.fc_size[1], bias=True),
            nn.BatchNorm1d(self.fc_size[1]),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            # [BATCH_SIZE, fc_size[1]] --> [BATCH_SIZE, fc_size[2]]
            # [128, 32] --> [128, 16]
            nn.Linear(in_features=self.fc_size[1], out_features=self.fc_size[2], bias=True),
            nn.BatchNorm1d(self.fc_size[2]),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            # [BATCH_SIZE, fc_size[2]] --> [BATCH_SIZE, op_size]
            # [128, 16] --> [128, 1]
            nn.Linear(in_features=self.fc_size[2], out_features=self.op_size, bias=True)
        )

        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.op_size, bias=True)



    def forward(self, x):
        rnn_op, _ = self.RNN(x)
        return self.layer_stack(rnn_op)
    


    

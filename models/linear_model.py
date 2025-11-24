import torch
import torch.nn as nn

class LinearModel(nn.Module):
    """
    Simple linear classifier for UCI HAR Dataset.
    
    This model is particularly suitable for poisoning/backdoor attacks because:
    1. Linear models are more interpretable and easier to analyze
    2. Feature-based attacks are more effective on linear decision boundaries
    3. Simpler models are often more vulnerable to poisoning in subspace
    
    Architecture:
    - Input: 561-dimensional feature vector (from UCI HAR)
    - Hidden layers with dropout for regularization
    - Output: 6 classes (activities)
    """
    
    def __init__(self, input_size=561, hidden_sizes=[256, 128], num_classes=6, dropout=0.3):
        """
        Args:
            input_size (int): Number of input features (default: 561 for UCI HAR)
            hidden_sizes (list): List of hidden layer sizes
            num_classes (int): Number of output classes (default: 6 activities)
            dropout (float): Dropout probability for regularization
        """
        super(LinearModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Build sequential layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (batch_size, seq_len, channels)
               If 3D, will be flattened to 2D
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Handle 3D input (batch_size, seq_len, channels) - flatten to features
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            
        # If input is already processed to 561 features, use directly
        # Otherwise, flatten
        if x.shape[1] != self.input_size:
            x = x.reshape(x.shape[0], -1)
            
        return self.model(x)
    
    def get_features(self, x, layer_idx=-2):
        """
        Extract intermediate features for poisoning attack analysis.
        
        Args:
            x: Input tensor
            layer_idx: Which layer to extract features from (default: second to last)
        
        Returns:
            Feature tensor from specified layer
        """
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            
        if x.shape[1] != self.input_size:
            x = x.reshape(x.shape[0], -1)
        
        # Extract features from intermediate layer
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == layer_idx:
                return x
        return x


class SimpleLinearModel(nn.Module):
    """
    Ultra-simple linear classifier without hidden layers.
    
    This is the most basic linear model - just input -> output.
    Useful for:
    1. Theoretical analysis of poisoning attacks
    2. Understanding linear decision boundaries
    3. Maximum interpretability
    4. Testing if problem is linearly separable
    """
    
    def __init__(self, input_size=561, num_classes=6):
        super(SimpleLinearModel, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        """Forward pass - single linear transformation."""
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            
        if x.shape[1] != self.input_size:
            x = x.reshape(x.shape[0], -1)
            
        return self.linear(x)


class RegularizedLinearModel(nn.Module):
    """
    Linear model with L2 regularization built-in.
    
    Particularly useful for poisoning defense evaluation:
    1. L2 regularization can help resist some poisoning attacks
    2. Useful for testing robustness of attacks against regularization
    3. Can evaluate defense mechanisms
    """
    
    def __init__(self, input_size=561, hidden_sizes=[256, 128], 
                 num_classes=6, dropout=0.4, weight_decay=1e-4):
        super(RegularizedLinearModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout = dropout
        self.weight_decay = weight_decay
        
        layers = []
        
        # Input layer with L2 regularization
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass."""
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            
        if x.shape[1] != self.input_size:
            x = x.reshape(x.shape[0], -1)
            
        return self.model(x)
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss for all linear layers."""
        l2_loss = 0.0
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                l2_loss += torch.sum(module.weight ** 2)
        return self.weight_decay * l2_loss

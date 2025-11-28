"""
HAR Models: Linear, CNN, LSTM, and Sklearn-based Models for Human Activity Recognition

This module provides multiple model architectures for time-series classification:
1. LinearModel - Flattened input with fully connected layers (PyTorch)
2. CNNModel - 1D Convolutional Neural Network (PyTorch)
3. LSTMModel - Long Short-Term Memory Network (PyTorch)
4. LogisticRegressionModel - Sklearn LogisticRegression wrapper
5. SVMModel - Sklearn SVM wrapper (Linear SVM and RBF SVM)
6. RidgeClassifierModel - Sklearn Ridge Classifier wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from abc import ABC, abstractmethod


class LinearModel(nn.Module):
    """
    Linear model that flattens input and uses fully connected layers.
    
    Architecture:
        Flatten -> FC -> ReLU -> Dropout -> FC -> ReLU -> Dropout -> FC
    """
    
    def __init__(self, input_size, n_channels, n_classes, hidden_dim=256, dropout=0.3):
        """
        Args:
            input_size: Sequence length (time steps)
            n_channels: Number of input channels
            n_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super(LinearModel, self).__init__()
        
        self.input_dim = input_size * n_channels
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, n_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels)
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        """Extract features before the final classification layer."""
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network for time-series classification.
    
    Architecture:
        Conv1D -> BatchNorm -> ReLU -> MaxPool -> 
        Conv1D -> BatchNorm -> ReLU -> MaxPool -> 
        Conv1D -> BatchNorm -> ReLU -> GlobalAvgPool ->
        FC -> ReLU -> Dropout -> FC
    """
    
    def __init__(self, input_size, n_channels, n_classes, 
                 conv_channels=[64, 128, 256], kernel_size=5, dropout=0.3):
        """
        Args:
            input_size: Sequence length (time steps)
            n_channels: Number of input channels
            n_classes: Number of output classes
            conv_channels: List of channel sizes for conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(n_channels, conv_channels[0], kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(conv_channels[2], 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels)
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # Transpose to (batch, n_channels, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before the final classification layer."""
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        return x


class LSTMModel(nn.Module):
    """
    LSTM model for time-series classification.
    
    Architecture:
        LSTM -> FC -> ReLU -> Dropout -> FC
    """
    
    def __init__(self, input_size, n_channels, n_classes, 
                 hidden_dim=128, n_layers=2, dropout=0.3, bidirectional=True):
        """
        Args:
            input_size: Sequence length (time steps) - not used directly
            n_channels: Number of input channels (input features per timestep)
            n_classes: Number of output classes
            hidden_dim: LSTM hidden dimension
            n_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * self.n_directions
        
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_channels)
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # LSTM output: (batch, seq_len, hidden_dim * n_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last timestep output
        # For bidirectional, concatenate the last output from both directions
        if self.bidirectional:
            # h_n shape: (n_layers * n_directions, batch, hidden_dim)
            # Get last layer outputs from both directions
            h_forward = h_n[-2]  # Last layer, forward direction
            h_backward = h_n[-1]  # Last layer, backward direction
            x = torch.cat([h_forward, h_backward], dim=1)
        else:
            x = h_n[-1]  # Last layer hidden state
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before the final classification layer."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            x = torch.cat([h_forward, h_backward], dim=1)
        else:
            x = h_n[-1]
        
        x = self.fc1(x)
        x = F.relu(x)
        return x


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif isinstance(model, SklearnModelWrapper):
        return model.count_parameters()
    else:
        return 0


# =============================================================================
# Sklearn Model Wrapper Base Class
# =============================================================================

class SklearnModelWrapper(ABC):
    """
    Abstract base class for sklearn model wrappers.
    
    This wrapper makes sklearn models compatible with the PyTorch-based
    training and attack pipeline.
    """
    
    def __init__(self, input_size, n_channels, n_classes):
        """
        Args:
            input_size: Sequence length (time steps)
            n_channels: Number of input channels
            n_classes: Number of output classes
        """
        self.input_size = input_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_dim = input_size * n_channels
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        
    @abstractmethod
    def _create_model(self):
        """Create the sklearn model. To be implemented by subclasses."""
        pass
    
    def _flatten(self, X):
        """Flatten input from (N, seq_len, n_channels) to (N, seq_len * n_channels)."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X
    
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X: Training data of shape (N, seq_len, n_channels) or (N, features)
            y: Training labels
        """
        X_flat = self._flatten(X)
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities
        """
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba (like LinearSVC),
            # use decision function and convert to pseudo-probabilities
            decision = self.model.decision_function(X_scaled)
            if len(decision.shape) == 1:
                # Binary classification
                proba = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - proba, proba])
            else:
                # Multi-class: softmax over decision function
                exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                return exp_decision / exp_decision.sum(axis=1, keepdims=True)
    
    def __call__(self, X):
        """
        Forward pass (compatible with PyTorch interface).
        
        Args:
            X: Input tensor of shape (batch, seq_len, n_channels)
            
        Returns:
            Output tensor of shape (batch, n_classes) with logits
        """
        proba = self.predict_proba(X)
        # Convert probabilities to logits
        logits = np.log(proba + 1e-10)
        return torch.FloatTensor(logits)
    
    def eval(self):
        """Set model to evaluation mode (no-op for sklearn)."""
        return self
    
    def train(self, mode=True):
        """Set model to training mode (no-op for sklearn)."""
        return self
    
    def to(self, device):
        """Move model to device (no-op for sklearn, but kept for compatibility)."""
        return self
    
    def parameters(self):
        """Return empty iterator (for compatibility with PyTorch)."""
        return iter([])
    
    def state_dict(self):
        """Return model state as dictionary."""
        import pickle
        return {
            'model': pickle.dumps(self.model),
            'scaler': pickle.dumps(self.scaler),
            'is_fitted': self._is_fitted
        }
    
    def load_state_dict(self, state_dict):
        """Load model state from dictionary."""
        import pickle
        self.model = pickle.loads(state_dict['model'])
        self.scaler = pickle.loads(state_dict['scaler'])
        self._is_fitted = state_dict['is_fitted']
    
    @abstractmethod
    def get_features(self, X):
        """Extract features (for compatibility with attack pipeline)."""
        pass
    
    @abstractmethod
    def count_parameters(self):
        """Count model parameters."""
        pass


class LogisticRegressionModel(SklearnModelWrapper):
    """
    Logistic Regression classifier wrapper.
    
    Uses multinomial logistic regression for multi-class classification.
    """
    
    def __init__(self, input_size, n_channels, n_classes, 
                 C=1.0, max_iter=1000, solver='lbfgs', penalty='l2'):
        """
        Args:
            input_size: Sequence length (time steps)
            n_channels: Number of input channels
            n_classes: Number of output classes
            C: Regularization strength (smaller = stronger regularization)
            max_iter: Maximum iterations for solver
            solver: Optimization algorithm
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        """
        super().__init__(input_size, n_channels, n_classes)
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.penalty = penalty
        
    def _create_model(self):
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            penalty=self.penalty,
            random_state=42,
            n_jobs=-1
        )
    
    def get_features(self, X):
        """
        For logistic regression, features are the flattened scaled input.
        """
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return torch.FloatTensor(X_scaled)
    
    def count_parameters(self):
        """Count coefficients + intercepts."""
        if self.model is None or not self._is_fitted:
            return self.input_dim * self.n_classes + self.n_classes
        return self.model.coef_.size + self.model.intercept_.size
    
    def get_coefficients(self):
        """Get model coefficients reshaped to (n_classes, seq_len, n_channels)."""
        if self.model is None or not self._is_fitted:
            return None
        coef = self.model.coef_.reshape(self.n_classes, self.input_size, self.n_channels)
        return coef


class SVMModel(SklearnModelWrapper):
    """
    Support Vector Machine classifier wrapper.
    
    Supports both linear and RBF kernels.
    """
    
    def __init__(self, input_size, n_channels, n_classes,
                 kernel='linear', C=1.0, gamma='scale', max_iter=1000):
        """
        Args:
            input_size: Sequence length (time steps)
            n_channels: Number of input channels
            n_classes: Number of output classes
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
            max_iter: Maximum iterations (-1 for no limit)
        """
        super().__init__(input_size, n_channels, n_classes)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        
    def _create_model(self):
        if self.kernel == 'linear':
            return LinearSVC(
                C=self.C,
                max_iter=self.max_iter,
                dual='auto',
                random_state=42
            )
        else:
            return SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                max_iter=self.max_iter,
                probability=True,
                random_state=42
            )
    
    def get_features(self, X):
        """
        For SVM, features are the flattened scaled input.
        """
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return torch.FloatTensor(X_scaled)
    
    def count_parameters(self):
        """Estimate parameters based on support vectors or coefficients."""
        if self.model is None or not self._is_fitted:
            return self.input_dim * self.n_classes + self.n_classes
        
        if hasattr(self.model, 'coef_'):
            # Linear SVM
            return self.model.coef_.size + self.model.intercept_.size
        elif hasattr(self.model, 'support_vectors_'):
            # RBF/other kernel SVM
            return self.model.support_vectors_.size + len(self.model.support_)
        return 0
    
    def get_coefficients(self):
        """Get model coefficients for linear SVM."""
        if self.model is None or not self._is_fitted:
            return None
        if hasattr(self.model, 'coef_'):
            n_coef = self.model.coef_.shape[0]
            return self.model.coef_.reshape(n_coef, self.input_size, self.n_channels)
        return None


class LinearSVMModel(SVMModel):
    """
    Linear SVM classifier (convenience class).
    """
    
    def __init__(self, input_size, n_channels, n_classes, C=1.0, max_iter=1000):
        super().__init__(input_size, n_channels, n_classes, 
                        kernel='linear', C=C, max_iter=max_iter)


class RBFSVMModel(SVMModel):
    """
    RBF Kernel SVM classifier (convenience class).
    """
    
    def __init__(self, input_size, n_channels, n_classes, 
                 C=1.0, gamma='scale', max_iter=1000):
        super().__init__(input_size, n_channels, n_classes,
                        kernel='rbf', C=C, gamma=gamma, max_iter=max_iter)


class RidgeClassifierModel(SklearnModelWrapper):
    """
    Ridge Classifier wrapper.
    
    Uses ridge regression for classification (L2-regularized least squares).
    """
    
    def __init__(self, input_size, n_channels, n_classes, alpha=1.0):
        """
        Args:
            input_size: Sequence length (time steps)
            n_channels: Number of input channels
            n_classes: Number of output classes
            alpha: Regularization strength
        """
        super().__init__(input_size, n_channels, n_classes)
        self.alpha = alpha
        
    def _create_model(self):
        return RidgeClassifier(
            alpha=self.alpha,
            random_state=42
        )
    
    def predict_proba(self, X):
        """
        Ridge classifier uses decision function converted to probabilities.
        """
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        decision = self.model.decision_function(X_scaled)
        
        if len(decision.shape) == 1:
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)
    
    def get_features(self, X):
        """Features are the flattened scaled input."""
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return torch.FloatTensor(X_scaled)
    
    def count_parameters(self):
        """Count coefficients + intercepts."""
        if self.model is None or not self._is_fitted:
            return self.input_dim * self.n_classes + self.n_classes
        return self.model.coef_.size + (self.model.intercept_.size if hasattr(self.model.intercept_, 'size') else 1)
    
    def get_coefficients(self):
        """Get model coefficients reshaped to (n_classes, seq_len, n_channels)."""
        if self.model is None or not self._is_fitted:
            return None
        coef = self.model.coef_
        if len(coef.shape) == 1:
            coef = coef.reshape(1, -1)
        return coef.reshape(coef.shape[0], self.input_size, self.n_channels)


class SGDClassifierModel(SklearnModelWrapper):
    """
    Stochastic Gradient Descent classifier wrapper.
    
    Supports various loss functions for linear classification.
    """
    
    def __init__(self, input_size, n_channels, n_classes,
                 loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000):
        """
        Args:
            input_size: Sequence length (time steps)
            n_channels: Number of input channels
            n_classes: Number of output classes
            loss: Loss function ('hinge', 'log_loss', 'modified_huber', etc.)
            penalty: Regularization ('l1', 'l2', 'elasticnet')
            alpha: Regularization strength
            max_iter: Maximum iterations
        """
        super().__init__(input_size, n_channels, n_classes)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        
    def _create_model(self):
        return SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=42,
            n_jobs=-1
        )
    
    def get_features(self, X):
        """Features are the flattened scaled input."""
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return torch.FloatTensor(X_scaled)
    
    def count_parameters(self):
        """Count coefficients + intercepts."""
        if self.model is None or not self._is_fitted:
            return self.input_dim * self.n_classes + self.n_classes
        return self.model.coef_.size + self.model.intercept_.size
    
    def get_coefficients(self):
        """Get model coefficients reshaped."""
        if self.model is None or not self._is_fitted:
            return None
        return self.model.coef_.reshape(self.n_classes, self.input_size, self.n_channels)


def get_model(model_type, input_size, n_channels, n_classes, **kwargs):
    """
    Factory function to get model by type.
    
    Args:
        model_type: One of 'linear', 'cnn', 'lstm', 'logistic', 'svm', 
                    'linear_svm', 'rbf_svm', 'ridge', 'sgd'
        input_size: Sequence length
        n_channels: Number of input channels
        n_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance (PyTorch nn.Module or SklearnModelWrapper)
    """
    model_type = model_type.lower()
    
    if model_type == 'linear':
        return LinearModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type == 'cnn':
        return CNNModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type == 'lstm':
        return LSTMModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type in ['logistic', 'logistic_regression', 'logreg']:
        return LogisticRegressionModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type == 'svm':
        return SVMModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type == 'linear_svm':
        return LinearSVMModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type == 'rbf_svm':
        return RBFSVMModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type == 'ridge':
        return RidgeClassifierModel(input_size, n_channels, n_classes, **kwargs)
    elif model_type == 'sgd':
        return SGDClassifierModel(input_size, n_channels, n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from "
                        "'linear', 'cnn', 'lstm', 'logistic', 'svm', "
                        "'linear_svm', 'rbf_svm', 'ridge', 'sgd'")


def is_sklearn_model(model):
    """Check if a model is a sklearn wrapper."""
    return isinstance(model, SklearnModelWrapper)


if __name__ == "__main__":
    # Test all models
    batch_size = 32
    seq_len = 128
    n_channels = 9
    n_classes = 6
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, n_channels)
    y = torch.randint(0, n_classes, (batch_size,))
    
    print("Testing models...")
    print("=" * 70)
    
    # Test PyTorch models
    print("\n--- PyTorch Models ---")
    for model_type in ['linear', 'cnn', 'lstm']:
        model = get_model(model_type, seq_len, n_channels, n_classes)
        output = model(x)
        features = model.get_features(x)
        
        print(f"\n{model_type.upper()} Model:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Parameters: {count_parameters(model):,}")
    
    # Test Sklearn models
    print("\n--- Sklearn Models ---")
    x_np = x.numpy()
    y_np = y.numpy()
    
    for model_type in ['logistic', 'linear_svm', 'rbf_svm', 'ridge', 'sgd']:
        model = get_model(model_type, seq_len, n_channels, n_classes)
        
        # Fit the model
        model.fit(x_np, y_np)
        
        # Test prediction
        output = model(x_np)
        predictions = model.predict(x_np)
        features = model.get_features(x_np)
        
        print(f"\n{model_type.upper()} Model:")
        print(f"  Input shape: {x_np.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"  Is sklearn: {is_sklearn_model(model)}")
    
    print("\n" + "=" * 70)
    print("All models tested successfully!")
